"""
The LIVE_COMPARE paths: explicit market quotes compared deterministically,
or one stock's market price compared against research documents.

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

from src.market_data import ambil_harga, format_market_data, ringkas_untuk_pengguna
from src.market_symbols import build_yahoo_symbol
from src.observability import invoke_llm_observed
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

        context = _format_context(documents)
        prompt = PROMPT_LIVE_COMPARE.format(
            market_data=format_market_data(data, _label(resolution)),
            profit_loss=format_profit_loss(profit_loss)
            or "PERHITUNGAN POSISI: tidak diminta.",
            context=context,
            question=question,
        )
        response = invoke_llm_observed(
            llm,
            prompt,
            purpose="live_compare",
            question=question,
            safe_metadata={
                "context_length": len(context),
                "retrieved_count": len(documents),
                "has_profit_loss": bool(profit_loss),
                "market_data_available": True,
            },
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


def tangani_multi(question, resolutions):
    """Compare explicit stocks using provider-labelled quotes, without an LLM.

    Per-symbol results stay in market_data.items; no arbitrary single stock
    becomes the conversation identity. Missing prices cannot enter arithmetic.
    """
    quotes, paragraphs = [], []
    for resolution in resolutions:
        symbol = build_yahoo_symbol(resolution["ticker"], resolution["market"])
        if symbol is None:
            data = {"symbol": resolution["ticker"], "status": "invalid_symbol",
                    "price": None}
        else:
            data = ambil_harga(symbol)
        quotes.append(data)
        if data["status"] != "ok":
            paragraphs.append(
                f"{data['symbol']}: " + pesan_harga_gagal(data["symbol"], data["status"])
            )
            continue
        source = {
            "eodhd_eod": "EODHD (EOD)",
            "fast_info": "Yahoo Finance",
            "history": "Yahoo Finance (history)",
        }.get(data.get("source"), "tidak tersedia")
        paragraphs.append(
            ringkas_untuk_pengguna(data, _label(resolution))
            + f"\nSumber: {source}. Waktu data: {data.get('as_of') or 'tidak tersedia'}."
        )

    available = [data for data in quotes if data["status"] == "ok"]
    status = "ok" if len(available) == len(quotes) else "partial"
    if not available:
        status = "market_data_unavailable"
    if len(available) == len(quotes) and len(available) >= 2:
        currencies = {data.get("currency") for data in available}
        if len(currencies) == 1 and None not in currencies and "" not in currencies:
            low = min(available, key=lambda data: data["price"])
            high = max(available, key=lambda data: data["price"])
            difference = high["price"] - low["price"]
            if difference == 0:
                paragraphs.append("Harga nominal per saham sama pada data yang tersedia.")
            else:
                paragraphs.append(
                    f"Harga nominal per saham {high['symbol']} lebih tinggi daripada "
                    f"{low['symbol']} dengan selisih {high['currency']} {difference:,.2f}."
                )
            paragraphs.append(
                "Ini perbandingan harga nominal per saham, bukan penilaian valuasi. "
                "Harga memakai waktu/sumber masing-masing di atas, bukan snapshot serentak."
            )
        else:
            paragraphs.append("Selisih nominal tidak dihitung karena mata uang berbeda atau tidak diketahui.")
    else:
        paragraphs.append("Perbandingan belum lengkap karena sebagian data harga tidak tersedia.")

    return {
        "answer": "\n\n".join(paragraphs),
        "status": status,
        "market_data": {"items": quotes, "from_cache": bool(quotes) and all(
            data.get("from_cache", False) for data in quotes
        )},
        "documents": [], "sources_used": ["market_data"],
        "profit_loss": None, "error": None,
    }


__all__ = ["tangani", "tangani_multi"]
