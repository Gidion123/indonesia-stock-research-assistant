"""
Conversation memory: what is being talked about, not what it costs.

Conversation memory and the market-data cache are often taken for the
same thing. They are not, and mixing them is the easiest way to serve a
stale price as the current one:

    conversation memory : "saham itu" -> BBRI          (identity)
    market data cache   : BBRI.JK -> 4210, 60 seconds  (numbers, TTL)

So this module may only hold identity. Price fields are rejected
explicitly by `validasi`, which raises `KontekSesiTidakValid` when it
finds one - rather than being merely "left unset" - so the mistake is
caught at write time instead of on screen. `buat_konteks` itself only
takes ticker, company and market.

Memory is also not kept as module-global state. Each Streamlit session
holds its own dict via `st.session_state`, so two users never see each
other's context.
"""

FIELD_IDENTITAS = ("ticker", "company", "market")

# Fields that must NEVER enter conversation memory, for any reason.
FIELD_TERLARANG = (
    "price",
    "current_price",
    "previous_close",
    "change",
    "change_percent",
    "as_of",
    "yahoo_symbol",
    "market_data",
    "currency",
)


class KontekSesiTidakValid(ValueError):
    """Raised when market data tries to enter conversation memory."""


def buat_konteks(ticker=None, company=None, market=None):
    """
    A new context holding identity only.
    """
    if not ticker:
        return None

    return {
        "ticker": str(ticker).upper().strip(),
        "company": company or None,
        "market": market or None,
    }


def dari_resolution(resolution):
    """
    Take the identity out of an entity resolution result.

    Returns None when the resolution produced no ticker - a question
    that points at no stock must not overwrite the context already in
    play.
    """
    if not resolution:
        return None

    return buat_konteks(
        ticker=resolution.get("ticker"),
        company=resolution.get("company"),
        market=resolution.get("market"),
    )


def validasi(konteks):
    """
    Make sure no market data has slipped in.
    """
    if konteks is None:
        return None

    if not isinstance(konteks, dict):
        raise KontekSesiTidakValid(
            f"Konteks sesi harus dict, bukan {type(konteks).__name__}."
        )

    terlarang = [k for k in FIELD_TERLARANG if k in konteks]

    if terlarang:
        raise KontekSesiTidakValid(
            "Data pasar tidak boleh disimpan di memori percakapan: "
            + ", ".join(terlarang)
            + ". Ambil ulang lewat market_data."
        )

    return konteks


def perbarui(konteks_lama, resolution):
    """
    The context for the next turn.

    The old context is kept when this turn produced no new stock, so a
    reference like "saham itu" stays alive across questions that name no
    issuer at all.
    """
    baru = dari_resolution(resolution)

    if baru is not None:
        return validasi(baru)

    return validasi(konteks_lama)


__all__ = [
    "FIELD_IDENTITAS",
    "FIELD_TERLARANG",
    "KontekSesiTidakValid",
    "buat_konteks",
    "dari_resolution",
    "validasi",
    "perbarui",
]
