"""
Retriever.

The historical evaluation uses plain similarity search with k=8.
The public session retriever also enforces upload ownership and prioritizes
mentions of the requested ticker/company within a wider candidate set.
"""

from functools import lru_cache
from datetime import datetime, timezone
import re
import time

from src import config
from src.observability import (
    get_langfuse_client_if_enabled,
    safe_document_reference,
    safe_observation_update,
    text_fingerprint,
)
from src.vector_store import get_session_vector_store, get_vector_store


RETRIEVER_K = config.RETRIEVER_K


@lru_cache(maxsize=4)
def get_retriever(k=None):
    """
    Create (once per k) the retriever backed by PgVector similarity search.
    """
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs={
            "k": k or config.RETRIEVER_K,
        },
    )

    return retriever


def _retriever_k(retriever):
    """Read the effective top-k without changing the retriever configuration."""
    direct_k = getattr(retriever, "k", None)
    if direct_k is not None:
        return direct_k

    search_kwargs = getattr(retriever, "search_kwargs", None)
    if isinstance(search_kwargs, dict) and search_kwargs.get("k") is not None:
        return search_kwargs["k"]

    return config.RETRIEVER_K


def invoke_retriever_observed(retriever, query, observation_name="rag-retrieval"):
    """Invoke a retriever and emit privacy-safe Langfuse retrieval telemetry.

    This wrapper deliberately does not send the raw query or document contents.
    It records only a query fingerprint, retrieval configuration, returned count,
    and source/page/chunk references. If Langfuse is disabled or unavailable,
    retrieval behaves exactly as before.
    """
    query = str(query or "").strip()
    langfuse = get_langfuse_client_if_enabled()

    if langfuse is None:
        return retriever.invoke(query)

    documents = None
    retrieval_error = None
    started = time.perf_counter()

    try:
        with langfuse.start_as_current_observation(
            as_type="retriever",
            name=observation_name,
            input={
                "query_fingerprint": text_fingerprint(query),
                "query_length": len(query),
            },
            metadata={
                "retriever_class": type(retriever).__name__,
                "search_type": config.RETRIEVER_SEARCH_TYPE,
                "top_k": _retriever_k(retriever),
            },
        ) as observation:
            try:
                documents = retriever.invoke(query)
            except Exception as exception:
                retrieval_error = exception
                safe_observation_update(
                    observation,
                    level="ERROR",
                    status_message=type(exception).__name__,
                    metadata={
                        "retriever_class": type(retriever).__name__,
                        "search_type": config.RETRIEVER_SEARCH_TYPE,
                        "top_k": _retriever_k(retriever),
                        "latency_seconds": round(time.perf_counter() - started, 4),
                    },
                )
                raise

            references = [
                safe_document_reference(document, rank=index)
                for index, document in enumerate(documents, start=1)
            ]
            latency_seconds = round(time.perf_counter() - started, 4)

            safe_observation_update(
                observation,
                output={
                    "retrieved_count": len(documents),
                    "documents": references,
                },
                metadata={
                    "retriever_class": type(retriever).__name__,
                    "search_type": config.RETRIEVER_SEARCH_TYPE,
                    "top_k": _retriever_k(retriever),
                    "retrieved_count": len(documents),
                    "latency_seconds": latency_seconds,
                },
                level="DEFAULT" if documents else "WARNING",
                status_message=None if documents else "empty_retrieval",
            )

        return documents

    except Exception:
        if retrieval_error is not None:
            # Preserve the original retrieval failure instead of hiding it behind
            # an observability error.
            raise retrieval_error

        if documents is not None:
            # Retrieval already succeeded; a tracing/export issue must not make
            # the application repeat the PgVector query or fail the request.
            return documents

        # Langfuse failed before retrieval started. Run the original path once.
        return retriever.invoke(query)


def retrieve_documents(query, k=None):
    """
    Retrieve the documents relevant to a user query.
    """
    retriever = get_retriever(k)

    # Historical/offline evaluation intentionally keeps the original untraced
    # path so benchmark runs do not generate production observability traffic.
    documents = retriever.invoke(query)

    return documents


class SessionRetriever:
    """Search the curated table and only this browser session's uploads.

    Filtering happens in PostgreSQL before top-k ranking. No unfiltered
    search of the upload table is exposed by this class.
    """

    def __init__(self, owner_id, k=None):
        from src.session_documents import validate_owner_id

        self.owner_id = validate_owner_id(owner_id)
        self.k = k or config.RETRIEVER_K

    def invoke(self, query):
        from src.entity_resolver import company_name_tokens, detect_explicit_tickers

        candidates = detect_explicit_tickers(query)
        ticker = candidates[0] if len(candidates) == 1 else None
        name = company_name_tokens(query) if ticker is None else ()
        focus = bool(ticker or len(name) >= 2)
        candidate_k = max(self.k * 6, self.k) if focus else self.k

        primary = get_vector_store().similarity_search_with_score(
            query, k=candidate_k, filter={"category": "primary"}
        )
        personal = get_session_vector_store().similarity_search_with_score(
            query,
            k=candidate_k,
            filter={
                "$and": [
                    {"owner_id": self.owner_id},
                    {"expires_at": {"$gt": datetime.now(timezone.utc)}},
                ]
            },
        )
        if ticker:
            pattern = re.compile(rf"\b{re.escape(ticker)}\b", re.I)
            relevant = lambda document: bool(pattern.search(document.page_content))
        elif len(name) >= 2:
            patterns = [re.compile(rf"\b{re.escape(word)}\b", re.I) for word in name]
            relevant = lambda document: all(
                pattern.search(document.page_content) for pattern in patterns
            )
        else:
            relevant = lambda document: True

        ranked = sorted(
            primary + personal,
            key=lambda pair: (not relevant(pair[0]), pair[1]),
        )
        return [document for document, _ in ranked[: self.k]]


def get_session_retriever(owner_id, k=None):
    """Never cache a retriever across visitor sessions."""
    return SessionRetriever(owner_id, k=k)
