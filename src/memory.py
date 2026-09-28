"""
Conversation memory: connects the next question to the previous one.

Two layers, kept separate because they have different lifetimes:

    identity    "saham itu" -> BBRI
                held by `src/session_context.py`, only ticker/company/
                market, never a price.

    history     the last few turns as short text, so the model
                understands references like "prospeknya" or "perusahaan
                tersebut".

This module stores nothing itself. It only turns one memory dict into
the next one; the caller is what keeps it (`st.session_state` under
Streamlit). Without that, two users would share the same memory.

Rules that hold:

* Prices NEVER enter memory. `session_context.validasi()` rejects them,
  and the history only holds text already shown to the user - it is not
  a source of truth for numbers. The next turn still refetches the
  price from market_data and the documents from PgVector.
* Memory never overrides a ticker written explicitly in the question.
  It only fills in what is missing.
* With no memory, everything behaves exactly as it did before - which
  is what keeps the evaluation scripts comparable, since there is no
  memory there.
"""

from src import session_context


# The WHOLE conversation is remembered for as long as the session runs.
# No turn is dropped: the only things that clear memory are the "new
# conversation" button and a page reload, both of which start a new
# session.
#
# What is capped is how much history GOES INTO the prompt, and the cap
# is counted in characters, not turns. The difference is real: a long
# conversation keeps its full memory on the app side, while what is
# sent to the model stays a sensible size.
BATAS_KARAKTER_RIWAYAT = 4000

# Answers are truncated: all the model needs is to remember what was
# discussed, not to repeat the whole thing.
BATAS_KARAKTER_JAWABAN = 260

def memori_baru():
    """
    Empty memory for a new conversation.
    """
    return {"konteks": None, "giliran": [], "saham_dibahas": []}


def _pastikan(memori):
    if not memori:
        return memori_baru()

    return {
        "konteks": session_context.validasi(memori.get("konteks")),
        "giliran": list(memori.get("giliran") or []),
        "saham_dibahas": list(memori.get("saham_dibahas") or []),
    }


def konteks_identitas(memori):
    """
    The identity part, in the form `assistant.jawab()` accepts.
    """
    return _pastikan(memori)["konteks"]


def merujuk_giliran_sebelumnya(pertanyaan):
    """
    Does this question hang on the previous turn?
    """
    from src.entity_resolver import merujuk_percakapan_sebelumnya

    return merujuk_percakapan_sebelumnya(pertanyaan)


def _ringkas(teks, batas):
    teks = " ".join(str(teks or "").split())

    if len(teks) <= batas:
        return teks

    return teks[:batas].rstrip() + "…"


def riwayat_teks(memori, batas=BATAS_KARAKTER_RIWAYAT):
    """
    The history as a text block for the prompt.

    Every turn is kept in memory; what gets picked here is whatever fits
    into `batas` characters, counted from the NEWEST backwards. A long
    conversation still feels connected because the latest turns are the
    ones that determine what the current question means.

    The list of stocks already discussed goes on the first line. That
    part survives the whole session even once the earliest turns no
    longer fit - so "the stock we talked about first" does not simply
    vanish.

    Returns an empty string when there are no turns yet; the caller uses
    that to pick the prompt without history, leaving the old behaviour
    unchanged.
    """
    memori = _pastikan(memori)
    giliran = memori["giliran"]

    if not giliran:
        return ""

    baris = []
    terpakai = 0

    for item in reversed(giliran):
        peran = "Pengguna" if item["peran"] == "user" else "Asisten"
        teks = f"{peran}: {item['isi']}"

        if terpakai + len(teks) > batas and baris:
            break

        baris.append(teks)
        terpakai += len(teks) + 1

    baris.reverse()

    saham = memori["saham_dibahas"]

    if saham:
        baris.insert(0, f"(Saham yang sudah dibahas: {', '.join(saham)})")

    return "\n".join(baris)


def saham_dibahas(memori):
    """
    Every stock discussed in this session, in order.
    """
    return list(_pastikan(memori)["saham_dibahas"])


def kueri_pencarian(pertanyaan, memori):
    """
    The question used to SEARCH for documents.

    The original question still goes to the model as written; only the
    search query is filled out. "Kalau prospeknya bagaimana?" contains
    no searchable word at all, so without this padding retrieval returns
    random chunks.

    The ticker is only APPENDED, never substituted. If the question
    already names an issuer itself, memory stays out of it.
    """
    pertanyaan = str(pertanyaan or "").strip()
    memori = _pastikan(memori)

    konteks = memori["konteks"]

    if not konteks or not konteks.get("ticker"):
        return pertanyaan

    if not merujuk_giliran_sebelumnya(pertanyaan):
        return pertanyaan

    # Already names an issuer itself: leave it alone.
    # `resolve_explicit` returns None when no stock code is written
    # directly in the question.
    from src.entity_resolver import resolve_explicit

    eksplisit = resolve_explicit(pertanyaan)

    if eksplisit and eksplisit.get("ticker"):
        return pertanyaan

    ticker = konteks["ticker"]
    perusahaan = konteks.get("company")

    tambahan = ticker if not perusahaan else f"{ticker} {perusahaan}"

    return f"{pertanyaan} {tambahan}".strip()


def perbarui(memori, pertanyaan, hasil):
    """
    Memory for the next turn.

    `hasil` is the output of `assistant.jawab()`. The identity comes
    from its resolution part - and if this turn produced no new stock,
    the old identity is kept, so references stay alive across questions
    that name no issuer at all.
    """
    memori = _pastikan(memori)

    konteks = session_context.perbarui(
        memori["konteks"],
        (hasil or {}).get("resolution"),
    )

    giliran = memori["giliran"]

    giliran.append(
        {
            "peran": "user",
            "isi": _ringkas(pertanyaan, BATAS_KARAKTER_JAWABAN),
        }
    )

    jawaban = (hasil or {}).get("answer")

    if jawaban:
        giliran.append(
            {
                "peran": "assistant",
                "isi": _ringkas(jawaban, BATAS_KARAKTER_JAWABAN),
            }
        )

    daftar = memori["saham_dibahas"]
    ticker_baru = (konteks or {}).get("ticker")

    if ticker_baru and ticker_baru not in daftar:
        daftar.append(ticker_baru)

    return {
        "konteks": session_context.validasi(konteks),
        # Not truncated. The session is the limit.
        "giliran": giliran,
        "saham_dibahas": daftar,
    }


def lupakan():
    """
    Start a new conversation. The name is blunt on purpose: the "new
    conversation" button has to actually forget, not pile up.
    """
    return memori_baru()


__all__ = [
    "BATAS_KARAKTER_RIWAYAT",
    "BATAS_KARAKTER_JAWABAN",
    "memori_baru",
    "konteks_identitas",
    "merujuk_giliran_sebelumnya",
    "riwayat_teks",
    "saham_dibahas",
    "kueri_pencarian",
    "perbarui",
    "lupakan",
]
