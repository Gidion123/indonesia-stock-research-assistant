"""
Router: decides whether a question needs the research documents, the
market price, or both.

Four intents:

    RAG           research documents only
    LIVE_PRICE    market price only
    LIVE_COMPARE  market price compared against the documents
    OUT_OF_SCOPE  not this system's job

Patterns run first because they are free and easy to audit. Words like
"sekarang" or "hari ini" mean the question needs a price; "target",
"TP" or "menurut riset" mean it needs a document. A question with both
is LIVE_COMPARE. The LLM is only called when the patterns do not give
a clear answer.

When the router is unsure it picks the option that fetches MORE data.
An unused price costs one network call; a stale answer misleads the
user. And when the choice is between OUT_OF_SCOPE and RAG it picks
RAG: a refusal here happens before any document has been looked at, so
the router must not be the reason a legitimate question gets turned
away.
"""

import re

from src import config


INTENT_RAG = "RAG"
INTENT_LIVE_PRICE = "LIVE_PRICE"
INTENT_LIVE_COMPARE = "LIVE_COMPARE"
INTENT_OUT_OF_SCOPE = "OUT_OF_SCOPE"

SEMUA_INTENT = (
    INTENT_RAG,
    INTENT_LIVE_PRICE,
    INTENT_LIVE_COMPARE,
    INTENT_OUT_OF_SCOPE,
)


# ============================================================
# 1. DETERMINISTIC PATTERNS
# ============================================================

# Signals that the user wants current market conditions.
POLA_WAKTU_SEKARANG = re.compile(
    r"\b("
    r"sekarang|saat\s+ini|hari\s+ini|terkini|terbaru|"
    r"live|real\s*time|realtime|update|terupdate|"
    r"barusan|detik\s+ini|current|today|latest|now"
    r")\b",
    flags=re.IGNORECASE,
)

# Signals that the user is referring to the research documents.
POLA_RUJUKAN_RISET = re.compile(
    r"\b("
    r"target|tp\s*\d?|take\s*profit|"
    r"riset|analisa|analisis|laporan|dokumen|rekomendasi|"
    r"prospek|fundamental|valuasi|katalis|risiko|"
    r"stop\s*loss|sl\b|entry|support|resistance|"
    r"menurut|berdasarkan|disebutkan|dibahas"
    r")\b",
    flags=re.IGNORECASE,
)

# Signals an explicit comparison between now and the research.
POLA_PERBANDINGAN = re.compile(
    r"\b("
    r"sudah\s+(?:mencapai|sampai|tembus|nyampe)|"
    r"belum\s+(?:mencapai|sampai|tembus)|"
    r"tercapai|mencapai\s+target|"
    r"dibanding(?:kan)?|"
    r"masih\s+(?:di\s+)?(?:bawah|atas)|"
    r"selisih|gap|upside|downside|"
    r"versus|vs\b"
    r")\b",
    flags=re.IGNORECASE,
)

# Signals a profit/loss question based on the market price.
#
# This covers cases like:
#   "BBRI sekarang saya untung atau rugi?"
#   "Dengan harga BBRI sekarang profit saya berapa persen?"
#
# On its own the word "rugi" is deliberately not enough to make a
# question LIVE_COMPARE, because "risiko kerugian menurut riset" is
# still a RAG question.
POLA_PROFIT_LOSS = re.compile(
    r"\b("
    r"untung|rugi|profit|loss|return|"
    r"keuntungan|kerugian|cuan|minus|"
    r"persentase\s+(?:untung|rugi|profit|loss|return)"
    r")\b",
    flags=re.IGNORECASE,
)

# Signals a pure price question.
POLA_HARGA = re.compile(
    r"\b("
    r"harga|price|berapa(?:an)?|quote|kuotasi|"
    r"nilai\s+saham|posisi\s+harga|"
    # "mana yang lebih murah sekarang" compares two stocks at today's
    # price - that is a price need, not a document need. Deliberately
    # NOT in POLA_PERBANDINGAN: that pattern is for comparing now
    # against the research, and "mana yang lebih bagus menurut riset"
    # must still land on RAG.
    r"murah|mahal"
    r")\b",
    flags=re.IGNORECASE,
)

# Topics that are clearly outside the scope. Kept short on purpose:
# this list only catches the obvious cases, everything else goes to the
# LLM or falls through to RAG. Refusing too aggressively in the router
# costs far more than letting one odd question reach retrieval.
POLA_JELAS_DI_LUAR = re.compile(
    r"\b("
    r"cuaca|resep|masak|film|sepak\s*bola|jadwal\s+kereta|"
    r"presiden|ibu\s*kota|terjemah|translate|"
    r"tinggi\s+badan|umur\s+kamu"
    r")\b",
    flags=re.IGNORECASE,
)


def _cocok(pola, teks):
    return bool(pola.search(teks))


def classify_by_pattern(question):
    """
    Classify without the LLM.

    Returns (intent, reason), or (None, reason) when the patterns are
    not decisive - that is the signal to call the LLM.
    """
    teks = str(question or "").strip()

    if not teks:
        return INTENT_OUT_OF_SCOPE, "Pertanyaan kosong."

    if _cocok(POLA_JELAS_DI_LUAR, teks):
        return INTENT_OUT_OF_SCOPE, "Topiknya jelas di luar cakupan saham."

    sekarang = _cocok(POLA_WAKTU_SEKARANG, teks)
    riset = _cocok(POLA_RUJUKAN_RISET, teks)
    banding = _cocok(POLA_PERBANDINGAN, teks)
    harga = _cocok(POLA_HARGA, teks)
    profit_loss = _cocok(POLA_PROFIT_LOSS, teks)

    # A profit/loss calculation needs the current market price plus the
    # user's own entry price.
    #
    # "risiko kerugian menurut riset" stays RAG because it carries no
    # marker of current market conditions or price.
    if profit_loss and (sekarang or harga or "entry" in teks.lower()):
        return (
            INTENT_LIVE_COMPARE,
            "Menanyakan keuntungan/kerugian berdasarkan harga pasar.",
        )

    # An explicit comparison always needs both sources, with or without
    # the word "sekarang": "BBRI sudah mencapai target belum" has no
    # time word in it, but cannot be answered without a current price.
    if banding and (riset or harga or sekarang):
        return (
            INTENT_LIVE_COMPARE,
            "Ada kata pembanding dan rujukan harga/riset.",
        )

    if sekarang and riset:
        return (
            INTENT_LIVE_COMPARE,
            "Menyebut kondisi sekarang sekaligus isi riset.",
        )

    if sekarang and harga:
        return INTENT_LIVE_PRICE, "Menanyakan harga untuk kondisi sekarang."

    if riset:
        return INTENT_RAG, "Merujuk isi dokumen riset."

    return None, "Pola tidak tegas."


# ============================================================
# 2. LLM CLASSIFICATION
# ============================================================

def classify_by_llm(question, llm=None):
    """
    Fallback when the patterns are not decisive. One call, small JSON.
    """
    from src.prompts import PROMPT_ROUTER
    from src.utils import extract_json_object

    try:
        if llm is None:
            from src.rag_chain import get_llm

            llm = get_llm()

        response = llm.invoke(PROMPT_ROUTER.format(question=question))
        payload = extract_json_object(str(response.content or ""))

    except Exception as exception:
        # An LLM failure is not a reason to turn the user away. Fall
        # back to RAG, which is how this system behaved before the
        # router existed.
        return (
            INTENT_RAG,
            f"LLM router gagal ({type(exception).__name__}), "
            "jatuh ke RAG.",
            0,
        )

    intent = str(payload.get("intent") or "").upper().strip()

    if intent not in SEMUA_INTENT:
        return (
            INTENT_RAG,
            f"LLM mengembalikan intent tak dikenal ({intent!r}), "
            "jatuh ke RAG.",
            1,
        )

    return intent, str(payload.get("reason") or "")[:200], 1


# ============================================================
# 3. ENTRY POINT
# ============================================================

def classify_router_intent(question, llm=None, izinkan_llm=True):
    """
    Work out the intent of a question.

    Returns a dict:
        intent      one of SEMUA_INTENT
        method      "pattern" | "llm" | "fallback"
        reason      short explanation
        llm_calls   0 or 1
    """
    question = str(question or "").strip()

    intent, alasan = classify_by_pattern(question)

    if intent is not None:
        return {
            "intent": intent,
            "method": "pattern",
            "reason": alasan,
            "llm_calls": 0,
        }

    if not izinkan_llm:
        return {
            "intent": INTENT_RAG,
            "method": "fallback",
            "reason": "Pola tidak tegas dan LLM dimatikan; default RAG.",
            "llm_calls": 0,
        }

    intent, alasan, panggilan = classify_by_llm(question, llm=llm)

    return {
        "intent": intent,
        "method": "llm" if panggilan else "fallback",
        "reason": alasan,
        "llm_calls": panggilan,
    }


def butuh_harga(intent):
    return intent in {INTENT_LIVE_PRICE, INTENT_LIVE_COMPARE}


def butuh_dokumen(intent):
    return intent in {INTENT_RAG, INTENT_LIVE_COMPARE}


__all__ = [
    "INTENT_RAG",
    "INTENT_LIVE_PRICE",
    "INTENT_LIVE_COMPARE",
    "INTENT_OUT_OF_SCOPE",
    "SEMUA_INTENT",
    "classify_by_pattern",
    "classify_by_llm",
    "classify_router_intent",
    "butuh_harga",
    "butuh_dokumen",
]