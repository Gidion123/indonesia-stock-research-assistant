"""
Orchestrator. One job: routing, not doing.

    QUESTION
        |
    router.classify_router_intent()
        |
        +-- OUT_OF_SCOPE -> stop, nothing else is called
        +-- RAG ----------> entity_resolver.resolve_ticker()
        |                       (izinkan_semantik=False)
        |                       |
        |                       +-- resolved -> identity for memory
        |                       +-- otherwise -> no identity
        |                       |
        |                   rag_chain.ask_question()
        |
        +-- LIVE_* -------> entity_resolver.resolve_ticker()
                                |
                                +-- ambiguous  -> ask for clarification
                                +-- not_found  -> say we do not know
                                +-- resolved   -> live_price / live_compare

This file deliberately holds no price logic, no calculations, no
retrieval and no symbol building. All of that lives in its own module;
what stays here is the order of steps and the shape of the result.

Nothing above touches Yahoo before entity resolution is done, and the
RAG and OUT_OF_SCOPE paths never touch it at all - the resolver itself
makes no network calls. That falls out of the order above rather than
being held in place by a flag, and `tests/test_network_budget.py`
proves it by counting requests.
"""

import time

from src import config, live_compare, live_price, session_context
from src.entity_resolver import (
    STATUS_AMBIGUOUS,
    STATUS_RESOLVED,
    resolve_ticker,
)
from src.prompts import (
    PESAN_DI_LUAR_CAKUPAN,
    PESAN_TICKER_TIDAK_DIKENAL,
    pesan_ambigu,
)
from src.router import (
    INTENT_LIVE_COMPARE,
    INTENT_LIVE_PRICE,
    INTENT_OUT_OF_SCOPE,
    INTENT_RAG,
    classify_router_intent,
)


# ============================================================
# RESULT SHAPE
# ============================================================
#
# Every path returns the same shape. Keys that do not apply are still
# present with empty values, so the caller (UI, evaluation, tests) does
# not need to know which path ran.

KUNCI_HASIL = (
    "question",
    "intent",
    "status",
    "answer",
    "router",
    "resolution",
    "market_data",
    "documents",
    "sources_used",
    "profit_loss",
    "error",
    "latency_seconds",
    "trace",
)


def _hasil(
    question,
    intent,
    status,
    answer,
    mulai,
    router=None,
    resolution=None,
    market_data=None,
    documents=None,
    sources_used=None,
    profit_loss=None,
    error=None,
):
    router = router or {"method": "none", "reason": "", "llm_calls": 0}
    resolusi_llm = (resolution or {}).get("llm_calls", 0)

    return {
        "question": question,
        "intent": intent,
        "status": status,
        "answer": answer,
        "router": router,
        "resolution": resolution,
        "market_data": market_data,
        "documents": documents or [],
        "sources_used": sources_used or [],
        "profit_loss": profit_loss,
        "error": error,
        "latency_seconds": round(time.perf_counter() - mulai, 4),
        # Cost trace. It exists so efficiency claims can be measured
        # rather than believed - used by scripts/evaluate_router.py and
        # the tests.
        "trace": {
            "router_llm_calls": router.get("llm_calls", 0),
            "resolution_llm_calls": resolusi_llm,
            "answer_llm_calls": 1 if status == "ok" and intent != INTENT_LIVE_PRICE else 0,
            "market_data_requested": market_data is not None,
            "market_data_from_cache": bool(
                (market_data or {}).get("from_cache")
            ),
        },
    }


# ============================================================
# RAG PATH
# ============================================================

def _jalur_rag(
    question,
    mulai,
    router,
    llm=None,
    retriever=None,
    k=None,
    riwayat=None,
    kueri_retrieval=None,
    resolution=None,
):
    from src.rag_chain import ask_question

    hasil = ask_question(
        question,
        k=k,
        retriever=retriever,
        llm=llm,
        riwayat=riwayat,
        kueri_retrieval=kueri_retrieval,
    )

    return _hasil(
        question,
        INTENT_RAG,
        hasil["status"],
        hasil["answer"],
        mulai,
        router=router,
        # The identity is returned too, so conversation memory knows
        # which stock was just discussed - including when the question
        # is pure research and touches no market data.
        resolution=resolution,
        documents=hasil["documents"],
        sources_used=["knowledge_base"],
        error=hasil["error"],
    )


# ============================================================
# ENTRY POINT
# ============================================================

def jawab(
    question,
    session_context_data=None,
    llm=None,
    retriever=None,
    k=None,
    izinkan_llm_router=True,
    gunakan_llm_untuk_harga=False,
    riwayat=None,
    kueri_retrieval=None,
    **kompat,
):
    """
    Answer one user question.

    `session_context_data` holds identity only ({ticker, company,
    market}); a price is never read from it.

    `riwayat` and `kueri_retrieval` come from `src/memory.py` and are
    both optional. Without them this path behaves exactly as it did
    before memory existed, which is what keeps the evaluation scripts
    measuring the same thing.
    """
    mulai = time.perf_counter()

    # The old parameter name is still accepted so this refactor does not
    # break existing callers.
    if session_context_data is None:
        session_context_data = kompat.get("session_context")

    question = str(question or "").strip()

    if not question:
        return _hasil(
            question,
            None,
            "empty_question",
            "Silakan tulis pertanyaan Anda.",
            mulai,
        )

    session_context_data = session_context.validasi(session_context_data)

    # -------- ROUTER --------
    router = classify_router_intent(
        question,
        llm=llm,
        izinkan_llm=izinkan_llm_router,
    )
    intent = router["intent"]

    if intent == INTENT_OUT_OF_SCOPE:
        return _hasil(
            question,
            intent,
            "out_of_scope",
            PESAN_DI_LUAR_CAKUPAN,
            mulai,
            router=router,
        )

    if intent == INTENT_RAG:
        # Identity for the research path: explicit and session levels
        # only.
        #
        # The semantic level is switched off here on purpose. It calls
        # PgVector and the LLM, and paying that on EVERY research
        # question just to populate memory is not worth it. The first
        # two levels are free and deterministic - and Yahoo stays
        # untouched, because the resolver never touches it anyway.
        resolusi_riset = resolve_ticker(
            question,
            session_context=session_context_data,
            izinkan_semantik=False,
        )

        if resolusi_riset["status"] != STATUS_RESOLVED:
            resolusi_riset = None

        result = _jalur_rag(
            question,
            mulai,
            router,
            llm=llm,
            retriever=retriever,
            k=k,
            riwayat=riwayat,
            kueri_retrieval=kueri_retrieval,
            resolution=resolusi_riset,
        )

        if resolusi_riset is None and result.get("status") == "ok":
            from src.entity_resolver import resolve_research_alias

            result["resolution"] = resolve_research_alias(
                question, result.get("documents") or []
            )

        return result

    # -------- ENTITY RESOLUTION (live paths; RAG resolved above) ------
    resolution = resolve_ticker(
        question,
        session_context=session_context_data,
        retriever=retriever,
        llm=llm,
        k=k,
    )

    if resolution["status"] == STATUS_AMBIGUOUS:
        return _hasil(
            question,
            intent,
            "ambiguous",
            pesan_ambigu(resolution["candidates"]),
            mulai,
            router=router,
            resolution=resolution,
        )

    if resolution["status"] != STATUS_RESOLVED:
        # This is where "harga emas dunia hari ini" stops. Without this
        # stopping point, that question used to get a full RAG answer,
        # complete with document citations, for a number that is in no
        # document at all.
        return _hasil(
            question,
            intent,
            "unresolved_entity",
            PESAN_TICKER_TIDAK_DIKENAL,
            mulai,
            router=router,
            resolution=resolution,
        )

    # -------- DISPATCH --------
    if intent == INTENT_LIVE_PRICE:
        bagian = live_price.tangani(
            question,
            resolution,
            llm=llm,
            gunakan_llm=gunakan_llm_untuk_harga,
        )
    else:
        bagian = live_compare.tangani(
            question,
            resolution,
            llm=llm,
            retriever=retriever,
            k=k,
        )

    return _hasil(
        question,
        intent,
        bagian["status"],
        bagian["answer"],
        mulai,
        router=router,
        resolution=resolution,
        market_data=bagian["market_data"],
        documents=bagian["documents"],
        sources_used=bagian["sources_used"],
        profit_loss=bagian["profit_loss"],
        error=bagian["error"],
    )


def perbarui_konteks_sesi(konteks_lama, hasil):
    """
    Conversation context for the next turn. Identity only.
    """
    return session_context.perbarui(konteks_lama, hasil.get("resolution"))


__all__ = [
    "KUNCI_HASIL",
    "jawab",
    "perbarui_konteks_sesi",
    "PESAN_DI_LUAR_CAKUPAN",
]
