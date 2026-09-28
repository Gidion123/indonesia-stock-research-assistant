"""
Retriever.

The historical evaluation uses plain similarity search with k=8.
The public session retriever also enforces upload ownership and prioritizes
mentions of the requested ticker/company within a wider candidate set.
"""

from functools import lru_cache
from datetime import datetime, timezone
import re

from src import config
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


def retrieve_documents(query, k=None):
    """
    Retrieve the documents relevant to a user query.
    """
    retriever = get_retriever(k)

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
