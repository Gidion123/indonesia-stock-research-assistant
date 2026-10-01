"""
RAG chain: retrieve -> build context -> generate a cited answer.

Two changes from the first version, both about being able to trust the
numbers later:

1. Components are cached. `ask_question()` used to rebuild the retriever,
   the vector store and the embedding model on every single call.
2. Failures are reported as failures. The old code silently re-invoked the
   LLM when the answer came back empty and returned a plain string either
   way, so an infrastructure failure was indistinguishable from a wrong
   answer. Now the result carries a `status`, and callers can exclude
   failed rows from their metrics instead of scoring them as mistakes.
"""

import time
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

from src import config
from src.observability import (
    get_langfuse_client_if_enabled,
    invoke_llm_observed,
    safe_error_type,
    safe_observation_update,
    text_fingerprint,
)
from src.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_DENGAN_RIWAYAT
from src.retriever import get_retriever, invoke_retriever_observed


LLM_PROVIDER = config.LLM_PROVIDER
GROQ_MODEL = config.GROQ_MODEL
DEEPSEEK_MODEL = config.DEEPSEEK_MODEL


# Error fragments that mean "try again later", not "wrong answer".
TRANSIENT_ERROR_HINTS = (
    "429",
    "rate limit",
    "rate_limit",
    "timeout",
    "timed out",
    "overload",
    "503",
    "502",
    "connection",
)


@lru_cache(maxsize=1)
def get_llm():
    """
    Create (once) the LLM used by the RAG application.
    The provider can be switched between DeepSeek and Groq via LLM_PROVIDER.
    """
    # DeepSeek is the default for this project because Groq's free tier
    # kept hitting rate limits during development.
    if config.LLM_PROVIDER == "deepseek":
        from langchain_openai import ChatOpenAI

        import os

        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY tidak ditemukan di file .env."
            )

        return ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=api_key,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )

    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.GROQ_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_retries=0,
        )

    raise ValueError(
        f"LLM_PROVIDER tidak dikenal: {config.LLM_PROVIDER!r}. "
        "Gunakan 'deepseek' atau 'groq'."
    )


@lru_cache(maxsize=2)
def get_prompt(dengan_riwayat=False):
    """
    Create (once per shape) the grounding prompt template.

    Two templates rather than one with an empty slot: a question with no
    history has to use exactly the same prompt as before memory existed,
    so evaluation results stay comparable.
    """
    if dengan_riwayat:
        return ChatPromptTemplate.from_template(SYSTEM_PROMPT_DENGAN_RIWAYAT)

    return ChatPromptTemplate.from_template(SYSTEM_PROMPT)


def format_documents(documents):
    """
    Turn retrieved documents into a context string carrying the source
    and page metadata that the citation format depends on.
    """
    formatted_documents = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")
        content = document.page_content.strip()

        formatted_documents.append(
            f"[{source} hal.{page}]\n{content}"
        )

    return "\n\n".join(formatted_documents)


def create_rag_chain(retriever=None, llm=None, dengan_riwayat=False):
    """
    Return the RAG chain components: retriever, prompt, llm.

    Both can be injected. Without that, the RAG path could only be tested
    against a live PostgreSQL and a live API key, which meant it was never
    tested at all - on the path that gets used the most.
    """
    return (
        retriever if retriever is not None else get_retriever(),
        get_prompt(dengan_riwayat),
        llm if llm is not None else get_llm(),
    )


def _classify_error(error):
    """
    Separate a temporary infrastructure failure from a permanent one.
    """
    message = str(error).lower()

    if any(hint in message for hint in TRANSIENT_ERROR_HINTS):
        return "transient_error"

    return "error"


def _rag_result_level(status):
    if status == "ok":
        return "DEFAULT"
    if status in {"empty_retrieval", "transient_error"}:
        return "WARNING"
    return "ERROR"


def _rag_result_summary(result):
    """Privacy-safe trace output; the generated answer/context are not exported."""
    return {
        "status": result.get("status"),
        "has_answer": bool(result.get("answer")),
        "retrieved_count": len(result.get("documents") or []),
        "chunk_count": len(result.get("chunk_ids") or []),
        "latency_seconds": result.get("latency_seconds"),
        "error_type": safe_error_type(result.get("error")),
    }


def _run_rag_pipeline(
    question,
    retriever,
    llm,
    riwayat,
    kueri,
):
    """Run the original RAG behavior; observability must not change its result."""
    started = time.perf_counter()

    retriever, prompt, llm = create_rag_chain(
        retriever,
        llm,
        dengan_riwayat=bool(riwayat),
    )

    # The retrieval child observation records only safe metadata and document
    # references. Raw PDF/upload contents are never sent to Langfuse here.
    documents = invoke_retriever_observed(
        retriever,
        kueri,
        observation_name="rag-retrieval",
    )

    chunk_ids = [
        document.metadata.get("chunk_id")
        for document in documents
    ]

    if not documents:
        return {
            "question": question,
            "answer": config.REFUSAL_MESSAGE,
            "documents": [],
            "context": "",
            "chunk_ids": [],
            "status": "empty_retrieval",
            "error": None,
            "latency_seconds": round(time.perf_counter() - started, 4),
        }

    context = format_documents(documents)

    slot = {"context": context, "question": question}

    if riwayat:
        slot["riwayat"] = riwayat

    messages = prompt.format_messages(**slot)

    try:
        response = invoke_llm_observed(
            llm,
            messages,
            purpose="rag_answer",
            question=question,
            provider=config.LLM_PROVIDER,
            safe_metadata={
                "context_length": len(context),
                "retrieved_count": len(documents),
                "has_history": bool(riwayat),
            },
        )
        answer = str(response.content or "").strip()

        status = "ok" if answer else "error"
        error = None if answer else "LLM mengembalikan jawaban kosong."

    except Exception as exception:
        answer = None
        status = _classify_error(exception)
        error = f"{type(exception).__name__}: {exception}"

    return {
        "question": question,
        "answer": answer,
        "documents": documents,
        "context": context,
        "chunk_ids": chunk_ids,
        "status": status,
        "error": error,
        "latency_seconds": round(time.perf_counter() - started, 4),
    }


def ask_question(
    question,
    k=None,
    retriever=None,
    llm=None,
    riwayat=None,
    kueri_retrieval=None,
):
    """
    Retrieve relevant context and generate a grounded answer.

    `riwayat` and `kueri_retrieval` are there for follow-up turns:

        kueri_retrieval  the sentence used to SEARCH for documents.
                         "Kalau prospeknya bagaimana?" contains nothing
                         searchable, so the caller may fill it out with a
                         ticker from memory. What is sent to the model is
                         still the original question.

        riwayat          the last few turns, only so the model can work
                         out what is being referred to. Facts must still
                         come from CONTEXT.

    Both default to None, and in that case this path is identical to what
    it was before memory existed, prompt included.

    Returns a dict with:
        question          str
        answer            str | None
        documents         list[Document]
        context           str
        chunk_ids         list[str]
        status            "ok" | "empty_retrieval" | "transient_error" | "error"
        error             str | None
        latency_seconds   float
    """
    question = str(question or "").strip()

    if not question:
        raise ValueError("Pertanyaan tidak boleh kosong.")

    riwayat = str(riwayat or "").strip()
    kueri = str(kueri_retrieval or "").strip() or question

    langfuse = get_langfuse_client_if_enabled()

    if langfuse is None:
        return _run_rag_pipeline(
            question=question,
            retriever=retriever,
            llm=llm,
            riwayat=riwayat,
            kueri=kueri,
        )

    result = None
    pipeline_error = None

    try:
        with langfuse.start_as_current_observation(
            as_type="chain",
            name="rag-answer",
            input={
                "question_fingerprint": text_fingerprint(question),
                "question_length": len(question),
            },
            metadata={
                "llm_provider": config.LLM_PROVIDER,
                "retriever_search_type": config.RETRIEVER_SEARCH_TYPE,
                "has_history": bool(riwayat),
                "retrieval_query_rewritten": kueri != question,
            },
        ) as observation:
            try:
                result = _run_rag_pipeline(
                    question=question,
                    retriever=retriever,
                    llm=llm,
                    riwayat=riwayat,
                    kueri=kueri,
                )
            except Exception as exception:
                pipeline_error = exception
                safe_observation_update(
                    observation,
                    level="ERROR",
                    status_message=type(exception).__name__,
                    output={
                        "status": "unhandled_error",
                        "error_type": type(exception).__name__,
                    },
                )
                raise

            safe_observation_update(
                observation,
                output=_rag_result_summary(result),
                level=_rag_result_level(result.get("status")),
                status_message=result.get("status"),
                metadata={
                    "llm_provider": config.LLM_PROVIDER,
                    "retriever_search_type": config.RETRIEVER_SEARCH_TYPE,
                    "has_history": bool(riwayat),
                    "retrieval_query_rewritten": kueri != question,
                    "requested_k_argument": k,
                },
            )

        return result

    except Exception:
        if pipeline_error is not None:
            # Preserve the original application exception exactly as before.
            raise pipeline_error

        if result is not None:
            # The RAG request already succeeded. A Langfuse/export failure must
            # not turn a valid user answer into an application error.
            return result

        # Langfuse failed before the application pipeline started. Execute the
        # original RAG path once without requiring observability to be healthy.
        return _run_rag_pipeline(
            question=question,
            retriever=retriever,
            llm=llm,
            riwayat=riwayat,
            kueri=kueri,
        )
