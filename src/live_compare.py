"""
The LIVE_COMPARE path: market price compared against the research
documents.

The riskiest path in this system, because two sources with different
natures feed one answer:

    market_data    : today's price, present in no document
    knowledge base : targets and analysis, with page numbers

Three rules here:

1.  Research numbers get citations, market numbers do not. A citation
    on a market price is a fake citation - that page does not contain
    today's price.
2.  Profit/loss is computed by Python, not the LLM. The LLM only
    explains it.
3.  If the price fetch fails, the LLM is not called at all. A model
    asked to "compare" without numbers will invent them.
"""

from src.market_data import ambil_harga, format_market_data
from src.market_symbols import build_yahoo_symbol
from src.profit_loss import (
    ekstrak_harga_entry,
    format_profit_loss,
    hitung_profit_loss,
    ringkas_profit_loss,
)
from src.prompts import (
    PESAN_DI_LUAR_KNOWLEDGE_BASE,
    PESAN_TICKER_TIDAK_DIKENAL,
    PROMPT_LIVE_COMPARE,
    pesan_harga_gagal,
)


def _label(resolution):
    return resolution.get("company") or resolution.get("ticker")


def _format_context(documents):
    bagian = []

    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "dokumen")
        page = metadata.get("page")

        sitasi = f"[{source} hal.{page}]" if page is not None else f"[{source}]"
        isi = getattr(doc, "page_content", "") or ""

        bagian.append(f"{sitasi}\n{isi}")

    return "\n\n".join(bagian)


def _ambil_dokumen(question, retriever, k=None):
    from src import config

    try:
        if retriever is None:
            from src.retriever import get_retriever

            retriever = get_retriever(k or config.RETRIEVER_K)

        return retriever.invoke(question), None

    except Exception as exception:
        return [], f"{type(exception).__name__}: {exception}"


def tangani(question, resolution, llm=None, retriever=None, k=None):
    """
    Return the result fragment for the LIVE_COMPARE path.
    """
    symbol = build_yahoo_symbol(
        resolution.get("ticker"),
        resolution.get("market"),
    )

    if not symbol:
        return {
            "answer": PESAN_TICKER_TIDAK_DIKENAL,
            "status": "unresolved_entity",
            "market_data": None,
            "documents": [],
            "sources_used": [],
            "profit_loss": None,
            "error": None,
        }

    data = ambil_harga(symbol)

    # Without a price there is nothing to compare. Stop here, before the
    # LLM can be asked to compare something against a number that does
    # not exist.
    if data["status"] != "ok":
        return {
            "answer": pesan_harga_gagal(symbol, data["status"]),
            "status": data["status"],
            "market_data": data,
            "documents": [],
            "sources_used": ["market_data"],
            "profit_loss": None,
            "error": data["error"],
        }

    # Profit/loss: Python computes it, before the LLM sees anything.
    harga_entry = ekstrak_harga_entry(question)
    profit_loss = hitung_profit_loss(harga_entry, data["price"])

    documents, galat_retrieval = _ambil_dokumen(question, retriever, k)

    # A stock the documents do not cover. "There is no research on it"
    # is different from "I do not know that stock" - the second would be
    # untrue, because the price was just fetched successfully.
    if not documents:
        bagian = [format_market_data(data, _label(resolution))]

        if profit_loss:
            bagian.append(ringkas_profit_loss(profit_loss, _label(resolution)))

        bagian.append(
            PESAN_DI_LUAR_KNOWLEDGE_BASE.format(
                ticker=resolution.get("ticker") or symbol
            )
        )

        return {
            "answer": "\n\n".join(b for b in bagian if b),
            "status": "no_research_coverage",
            "market_data": data,
            "documents": [],
            "sources_used": ["market_data"],
            "profit_loss": profit_loss,
            "error": galat_retrieval,
        }

    try:
        if llm is None:
            from src.rag_chain import get_llm

            llm = get_llm()

        response = llm.invoke(
            PROMPT_LIVE_COMPARE.format(
                market_data=format_market_data(data, _label(resolution)),
                profit_loss=format_profit_loss(profit_loss)
                or "PERHITUNGAN POSISI: tidak diminta.",
                context=_format_context(documents),
                question=question,
            )
        )
        jawaban = str(response.content or "").strip()
        status = "ok" if jawaban else "error"
        error = None if jawaban else "LLM mengembalikan jawaban kosong."

    except Exception as exception:
        from src.rag_chain import _classify_error

        jawaban = None
        status = _classify_error(exception)
        error = f"{type(exception).__name__}: {exception}"

    return {
        "answer": jawaban,
        "status": status,
        "market_data": data,
        "documents": documents,
        "sources_used": ["market_data", "knowledge_base"],
        "profit_loss": profit_loss,
        "error": error,
    }


__all__ = ["tangani"]
