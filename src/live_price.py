"""
The LIVE_PRICE path: current market price, no research documents.

This path is kept as cheap as possible on purpose. For "harga BBRI
sekarang" there is nothing to synthesise: one number comes from
market_data and Python assembles the sentence. Calling the LLM just to
wrap a single number adds latency, cost, and one more chance for the
number to change on the way through.
"""

from src.market_data import (
    ambil_harga,
    format_market_data,
    ringkas_untuk_pengguna,
)
from src.market_symbols import build_yahoo_symbol
from src.prompts import (
    PESAN_TICKER_TIDAK_DIKENAL,
    pesan_harga_gagal,
)


def _label(resolution):
    return resolution.get("company") or resolution.get("ticker")


def tangani(question, resolution, llm=None, gunakan_llm=False):
    """
    Return the result fragment for the LIVE_PRICE path.

    The caller (assistant) fills in `question`, `intent`, `router`,
    `resolution` and `latency_seconds`.
    """
    symbol = build_yahoo_symbol(
        resolution.get("ticker"),
        resolution.get("market"),
    )

    # An unknown exchange means the price symbol cannot be built
    # correctly. Guessing here returns another company's price with no
    # error message at all, so this path stops instead.
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

    if data["status"] != "ok":
        return {
            "answer": pesan_harga_gagal(symbol, data["status"]),
            "status": data["status"],
            "market_data": data,
            "documents": [],
            "sources_used": ["market_data"],
            "profit_loss": None,
            # The technical detail is kept for logs; the user-facing
            # message above is already free of jargon.
            "error": data["error"],
        }

    if not gunakan_llm:
        return {
            "answer": ringkas_untuk_pengguna(data, _label(resolution)),
            "status": "ok",
            "market_data": data,
            "documents": [],
            "sources_used": ["market_data"],
            "profit_loss": None,
            "error": None,
        }

    from src.prompts import PROMPT_LIVE_PRICE

    try:
        if llm is None:
            from src.rag_chain import get_llm

            llm = get_llm()

        response = llm.invoke(
            PROMPT_LIVE_PRICE.format(
                market_data=format_market_data(data, _label(resolution)),
                question=question,
            )
        )
        jawaban = str(response.content or "").strip()

    except Exception:
        jawaban = ""

    # An LLM failure must not throw away a price we already have: fall
    # back to the sentence Python assembles.
    if not jawaban:
        jawaban = ringkas_untuk_pengguna(data, _label(resolution))

    return {
        "answer": jawaban,
        "status": "ok",
        "market_data": data,
        "documents": [],
        "sources_used": ["market_data"],
        "profit_loss": None,
        "error": None,
    }


__all__ = ["tangani"]
