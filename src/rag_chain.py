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
from src.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_DENGAN_RIWAYAT
from src.retriever import get_retriever


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

    started = time.perf_counter()

    retriever, prompt, llm = create_rag_chain(
        retriever,
        llm,
        dengan_riwayat=bool(riwayat),
    )

    documents = retriever.invoke(kueri)

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
        response = llm.invoke(messages)
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
