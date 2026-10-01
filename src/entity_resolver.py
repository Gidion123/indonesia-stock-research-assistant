"""
Entity resolution at query time, replacing the pre-processing ticker
registry.

The old version scanned 63 PDF pages at startup, pulled out 91
candidates, and called the LLM to validate all of them - to be ready
for questions that might never be asked. Confirming that AKRA is an
equity is worth nothing until a user actually types AKRA.

Entities are now resolved only when they are needed, in three levels,
cheapest first:

    1. EXPLICIT   regex finds a code, its shape decides the exchange
                  0 LLM calls, 0 PgVector queries, 0 Yahoo requests
    2. SESSION    the question refers back to the conversation
                  0 LLM calls, 0 PgVector queries
    3. SEMANTIC   a company name with no code
                  1 PgVector query + 1 LLM call

Level 1 is safe because of what filters it. A regex cannot tell a
ticker from an ordinary capitalised word: "YANG", "RISK" and "MACD" all
pass `[A-Z]{4}`. The filter is the common-word list below, and the
final check is `market_data` returning `not_found` when a price is
actually requested - not the resolver.

The resolver does NOT call Yahoo Finance to validate a ticker. The old
version did, and the result was that every capitalised word in a user's
question became a network request, including on questions that needed
no price at all. The exchange guess here comes purely from the SHAPE of
the code; a wrong guess only ends in `not_found` from market_data,
which is cheap.

Level 3 stays honest through a grounding check. The LLM knows plenty of
tickers from general knowledge, and that is the danger: an answer that
cannot be traced back to a document breaks the RAG property. So after
the LLM answers, Python checks that the code or the company name really
does appear in the retrieved context.
"""

import re

from src import config
from src.observability import invoke_llm_observed

# The resolver must NOT touch market_data or yfinance. What is imported
# below is purely symbol-shape constants - used to strip a suffix the
# user typed ("BBRI.JK" -> "BBRI") and to validate the exchange code
# the LLM returns. Neither can trigger a network call.
from src.market_symbols import (
    BURSA_DEFAULT,
    BURSA_DIKENAL,
    bersihkan_sufiks,
)


# ============================================================
# 1. CANDIDATES FROM THE QUESTION
# ============================================================

# Four-letter IDX tickers; MP and other short codes are caught by the
# second pattern when written with an explicit marker.
POLA_KANDIDAT = [
    r"\b([A-Z]{2,5})\.JK\b",       # BBRI.JK
    r"\$([A-Z]{1,5})\b",           # $BBRI
    r"\b([A-Z]{4})\b",             # BBRI - but also YANG, SAYA, RISK
]

# Candidate filter. It holds words that show up often in Indonesian
# questions plus market terms that are not issuers - it is not a list
# of issuers, so a newly listed company needs no entry anywhere.
KATA_UMUM = {
    # Indonesian
    "YANG", "SAYA", "ANDA", "KAMU", "ATAU", "DARI", "PADA", "UNTUK",
    "AKAN", "SUDAH", "BELUM", "MASIH", "BISA", "HARI", "SAAT", "INI",
    "ITU", "APA", "BERAPA", "MANA", "KAPAN", "KENAPA", "BAGAIMANA",
    "HARGA", "SAHAM", "PASAR", "BURSA", "NAIK", "TURUN", "BAIK",
    "LEBIH", "PALING", "SAJA", "JUGA", "TIDAK", "TOLONG", "COBA",
    # common English
    "BASE", "CASE", "EXIT", "ZONE", "RISK", "PLAN", "HOLD", "LOSS",
    "FLOW", "FAIR", "FULL", "BEAR", "BULL", "DEEP", "VERY", "WITH",
    "FROM", "THAT", "THIS", "WHAT", "WHEN", "PRICE", "STOCK",
    # market terms that are not issuers
    "IHSG", "MSCI", "FTSE", "MACD", "BVPS", "UMKM", "BUMN", "APBN",
    "OPEC", "YTD", "TTM", "NPL", "NIM", "ROE", "PBV",
    "DER", "CAR", "ASP", "SRBI", "DHE",
}

# CASA is deliberately NOT in KATA_UMUM even though in the research
# documents it means Current Account Savings Account: CASA is also a
# real IDX ticker. Filtering it here would mean a user genuinely asking
# about that stock is never served. What keeps the two apart is the
# router: "rasio CASA menurut riset" is routed to RAG, so it never
# reaches the live-price path. The resolver still sees it - assistant
# calls resolve_ticker(..., izinkan_semantik=False) on the RAG path too
# - but a wrong identity there only costs a stale entry in conversation
# memory, not a network call.

# Words that get past KATA_UMUM but show up often in questions. Only
# used by the lowercase scan. The set also holds 3, 5 and 6 letter
# entries that the `[a-z]{4}` scan can never produce; they are harmless
# and kept so the list reads as one vocabulary.
KATA_UMUM_HURUF_KECIL = {
    "BELI", "JUAL", "LAGI", "KALO", "KALAU", "GIMANA", "MAU", "DONG",
    "NIH", "SIH", "BUAT", "PUNYA", "BAGUS", "JELEK", "MAHAL", "MURAH",
    "TREN", "DATA", "INFO", "CARI", "LIAT", "LIHAT", "TAHU", "MASUK",
    "BANK", "EMAS", "USD", "IDR", "RUPIAH", "DOLAR", "TARGET",
    "LABA", "RUGI",
}

# Candidate cap for the lowercase scan. This module makes no network
# calls, so extra candidates cost nothing but an AMBIGUOUS result; past
# three the question is almost certainly not about one specific stock,
# and stopping early keeps the clarification message short.
MAKS_KANDIDAT_HURUF_KECIL = 3

# Questions that refer back to something already mentioned.
POLA_RUJUKAN_SESI = re.compile(
    r"\b("
    r"tersebut|itu|tadi|barusan|sebelumnya|"
    r"saham\s+nya|sahamnya|emiten\s+nya|emitennya|"
    r"dia"
    r")\b|\b\w{4,}nya\b",
    flags=re.IGNORECASE,
)

# A short question made entirely of conversational and financial words has
# no independent subject. It can safely refer to the active stock. This is
# intentionally a vocabulary of QUESTION WORDS, not a registry of tickers:
# new issuers such as POWR work without updating this list.
KATA_LANJUTAN_TANPA_SUBJEK = {
    "apa", "apakah", "berapa", "bagaimana", "gimana", "kalau", "lalu",
    "terus", "dengan", "untuk", "yang", "di", "ke", "pada", "saat",
    "sekarang", "ini", "hari", "sudah", "belum", "mencapai", "tembus",
    "menurut", "berdasarkan", "riset", "laporan", "dokumen", "saham",
    "emiten", "perusahaan", "target", "harga", "tp", "valuasi", "prospek",
    "risiko", "fundamental", "dividen", "laba", "pendapatan", "kinerja",
    "potensi", "entry", "support", "resistance", "stop", "loss",
    "katalis", "rekomendasi", "bagus", "menarik", "murah", "mahal",
    "naik", "turun", "profit", "rugi", "untung", "posisi", "analisis",
    "hasil", "bisnis", "kondisi", "terkini", "dari",
}
KATA_TOPIK_LANJUTAN = {
    "target", "harga", "valuasi", "prospek", "risiko", "fundamental",
    "dividen", "laba", "pendapatan", "kinerja", "entry", "support",
    "resistance", "katalis", "rekomendasi", "profit", "rugi", "untung",
    "mencapai", "bisnis", "kondisi",
}

# Query-time company clues. These are generic question/report words, not
# ticker names; aliases themselves are read from retrieved documents.
KATA_BUKAN_NAMA_PERUSAHAAN = {
    "Apa", "Apakah", "Bagaimana", "Berapa", "Menurut", "Berdasarkan",
    "Riset", "Saham", "Dokumen", "Laporan", "Analisis", "Prospek",
    "Target", "Harga", "Valuasi", "Data", "Center", "Tbk", "Persero",
    "Pada", "Dari", "Kalau", "Lalu",
}


def company_name_tokens(question):
    """Return distinctive Title Case words in a named company question."""
    return tuple(
        word for word in re.findall(r"\b[A-Z][a-z]{2,}\b", str(question or ""))
        if word not in KATA_BUKAN_NAMA_PERUSAHAAN
    )


def resolve_research_alias(question, documents):
    """Infer a company ticker only from an unambiguous retrieved name/code pair.

    For example, a question about "Cikarang Listrindo" may find a chunk
    with "PT Cikarang Listrindo Tbk (POWR)". No fixed issuer registry, LLM,
    or market API is involved. If documents suggest multiple codes, leave
    identity unset rather than guessing.
    """
    name = company_name_tokens(question)
    if len(name) < 2:
        return None

    phrase = re.compile(
        r"\b" + r"\s+".join(map(re.escape, name)) +
        r"\b(?:\s+(?:Tbk|Ltd|Inc|Persero)|\s+\(Persero\)|[.,]){0,4}"
        r"\s*\(([A-Z]{4})\)",
        re.I,
    )
    candidates = set()
    for document in documents:
        for match in phrase.finditer(document.page_content):
            ticker = match.group(1).upper()
            if ticker not in KATA_UMUM:
                candidates.add(ticker)

    if len(candidates) != 1:
        return None

    ticker = candidates.pop()
    return _hasil(
        STATUS_RESOLVED,
        ticker=ticker,
        company=" ".join(name),
        market=BURSA_DEFAULT,
        level="document_alias",
        confidence="high",
        reason=f"Nama perusahaan dan kode {ticker} muncul bersama di dokumen riset.",
    )


def detect_explicit_tickers(question):
    """
    Stock codes written directly in the question.

    What comes back are candidates, not confirmed tickers. Filtering
    happens inline here, against KATA_UMUM and, for the lowercase scan,
    KATA_UMUM_HURUF_KECIL. The final check is `market_data` returning
    `not_found` when a price is actually requested.
    """
    teks = str(question or "")

    ditemukan = []

    for pola in POLA_KANDIDAT:
        for match in re.finditer(pola, teks):
            kode = match.group(1).upper()

            if kode in KATA_UMUM:
                continue

            if kode not in ditemukan:
                ditemukan.append(kode)

    if ditemukan:
        return ditemukan

    # Lowercase scan. People type "harga bbri sekarang" far more often
    # than "harga BBRI sekarang", and an uppercase-only regex would push
    # all of those questions onto the expensive semantic path - or fail
    # outright if the issuer is not in the knowledge base.
    #
    # Only runs when the uppercase scan found nothing, and its results
    # get the same exchange handling as any other candidate.
    for match in re.finditer(r"\b([a-z]{4})\b", teks):
        kode = match.group(1).upper()

        if kode in KATA_UMUM or kode in KATA_UMUM_HURUF_KECIL:
            continue

        if kode not in ditemukan:
            ditemukan.append(kode)

        if len(ditemukan) >= MAKS_KANDIDAT_HURUF_KECIL:
            break

    return ditemukan


def normalize_ticker(value):
    """
    BBRI    -> BBRI
    BBRI.JK -> BBRI
    bbri    -> BBRI
    """
    if not value:
        return None

    ticker = bersihkan_sufiks(value)

    if not re.fullmatch(r"[A-Z]{1,5}", ticker):
        return None

    return ticker


def merujuk_percakapan_sebelumnya(question):
    text = str(question or "").strip()
    if POLA_RUJUKAN_SESI.search(text):
        return True

    words = re.findall(r"[a-z]+", text.lower())
    return (
        1 < len(words) <= 12
        and all(word in KATA_LANJUTAN_TANPA_SUBJEK for word in words)
        and any(word in KATA_TOPIK_LANJUTAN for word in words)
    )


# ============================================================
# 2. RESULT SHAPE
# ============================================================

STATUS_RESOLVED = "resolved"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_NOT_FOUND = "not_found"
STATUS_UNVERIFIED = "unverified"
STATUS_ERROR = "error"

LEVEL_EKSPLISIT = "explicit"
LEVEL_SESI = "session"
LEVEL_SEMANTIK = "semantic"


def _hasil(
    status,
    ticker=None,
    company=None,
    market=None,
    level=None,
    candidates=None,
    confidence=None,
    evidence=None,
    llm_calls=0,
    reason="",
):
    return {
        "ticker": ticker,
        "company": company,
        "market": market,
        "level": level,
        "status": status,
        "candidates": candidates or [],
        "confidence": confidence,
        "evidence": evidence or [],
        "llm_calls": llm_calls,
        "reason": reason,
    }


# ============================================================
# 3. LEVEL 1 - EXPLICIT TICKER
# ============================================================

# The shape of an IDX stock code: four capital letters. This is the
# Indonesia Stock Exchange convention, not a list of issuers - no
# company name is hardcoded, so a newly listed company is served
# automatically.
POLA_BENTUK_IDX = re.compile(r"^[A-Z]{4}$")


def _tebak_bursa(kode):
    """
    Pick the exchange from the SHAPE of the code, with no network call.

    This assistant serves Indonesian equity research, so any four-letter
    code is treated as IDX. Only codes that are not exactly four letters
    (MP, for example) reach the NYSE branch.

    That means a four-letter US ticker is misread: AAPL matches
    POLA_BENTUK_IDX and becomes AAPL.JK. The guess is allowed to be
    wrong because it is cheap to be wrong - `market_data` reports
    `not_found` when the symbol does not exist. The dangerous option is
    the opposite one, calling the exchange from inside the resolver just
    to guess, because that turns every capitalised word into a network
    request.
    """
    if POLA_BENTUK_IDX.match(kode):
        return BURSA_DEFAULT

    return "NYSE"


def resolve_explicit(question):
    """
    Level 1. Fully deterministic: no LLM, no PgVector, no Yahoo Finance.

    `detect_explicit_tickers` does the candidate filtering against the
    common-word lists. Whatever survives becomes the ticker directly;
    whether it is real is settled later by market_data, when a price is
    requested.
    """
    kandidat = detect_explicit_tickers(question)

    if not kandidat:
        return None

    if len(kandidat) > 1:
        return _hasil(
            STATUS_AMBIGUOUS,
            level=LEVEL_EKSPLISIT,
            candidates=list(kandidat),
            confidence="low",
            reason=(
                "Pertanyaan menyebut lebih dari satu kode saham: "
                + ", ".join(kandidat)
            ),
        )

    kode = kandidat[0]
    market = _tebak_bursa(kode)

    return _hasil(
        STATUS_RESOLVED,
        ticker=kode,
        company=None,
        market=market,
        level=LEVEL_EKSPLISIT,
        confidence="high",
        reason=f"Kode {kode} tertulis eksplisit di pertanyaan.",
    )


def resolve_explicit_many(question):
    """Resolve each explicit code; caller must establish comparison intent.

    This does not reinterpret semantic alternatives or change the single-
    entity resolver's ambiguity contract. No session fallback or I/O occurs.
    """
    tickers = detect_explicit_tickers(question)
    if len(tickers) < 2:
        return []
    return [resolve_explicit(f"${ticker}") for ticker in tickers]


# ============================================================
# 4. LEVEL 2 - SESSION CONTEXT
# ============================================================

def resolve_from_session(question, session_context):
    """
    Level 2. Resolves a REFERENCE; it does not store FACTS.

    That line is deliberate: conversation memory may say that "bank
    tersebut" means BBRI, but the price and the analysis are still
    fetched fresh from yfinance and PgVector. Memory is never the source
    of truth for a number.
    """
    if not session_context:
        return None

    if not merujuk_percakapan_sebelumnya(question):
        return None

    ticker = normalize_ticker(session_context.get("ticker"))

    if not ticker:
        return None

    # Session context stores identity only: ticker, company, market. The
    # Yahoo symbol is deliberately NOT kept there - that belongs to
    # market_data, and keeping it in conversation memory blurs the line
    # between "what we are talking about" and "market data".
    market = session_context.get("market") or BURSA_DEFAULT

    if market not in BURSA_DIKENAL:
        market = BURSA_DEFAULT

    return _hasil(
        STATUS_RESOLVED,
        ticker=ticker,
        company=session_context.get("company"),
        market=market,
        level=LEVEL_SESI,
        confidence="medium",
        reason=(
            f"Pertanyaan merujuk percakapan sebelumnya; saham terakhir "
            f"yang dibahas adalah {ticker}."
        ),
    )


# ============================================================
# 5. LEVEL 3 - SEMANTIC RESOLUTION
# ============================================================

def _teks_context(documents):
    return "\n\n".join(
        f"[{d.metadata.get('source', '?')} hal.{d.metadata.get('page', '?')}]"
        f"\n{d.page_content}"
        for d in documents
    )


def _didukung_context(ticker, company, context):
    """
    Grounding check. Guards against the LLM answering from general
    knowledge instead of from the documents.

    Either one appearing is enough: documents do not always write the
    code and the name side by side.
    """
    context_atas = context.upper()

    if ticker and re.search(rf"\b{re.escape(ticker)}\b", context_atas):
        return True, f"kode {ticker} muncul di context"

    if company:
        # Match the distinguishing words of the company name, not the
        # whole string: documents write "PT Bank Rakyat Indonesia
        # (Persero) Tbk" with varying punctuation.
        kata_penting = [
            kata
            for kata in re.findall(r"[A-Za-z]{4,}", company)
            if kata.upper() not in {"PERSERO", "INDONESIA", "CORP",
                                    "CORPORATION", "LIMITED", "GROUP"}
        ]

        if kata_penting and all(
            kata.upper() in context_atas for kata in kata_penting
        ):
            return True, f"nama '{company}' muncul di context"

    return False, "kode maupun nama perusahaan tidak ada di context"


def resolve_semantic(
    question,
    retriever=None,
    llm=None,
    k=None,
):
    """
    Level 3. One PgVector query plus one LLM call.

    No Yahoo Finance call happens here. The LLM proposes an entity from
    the context, Python verifies that the proposal really is in the
    context, and the identity is handed back to the caller.
    """
    from src.prompts import PROMPT_ENTITY_RESOLUTION
    from src.utils import extract_json_object

    try:
        if retriever is None:
            from src.retriever import get_retriever

            retriever = get_retriever(k or config.RETRIEVER_K)

        documents = retriever.invoke(question)

    except Exception as exception:
        return _hasil(
            STATUS_ERROR,
            level=LEVEL_SEMANTIK,
            reason=f"Retrieval gagal: {type(exception).__name__}: {exception}",
        )

    if not documents:
        return _hasil(
            STATUS_NOT_FOUND,
            level=LEVEL_SEMANTIK,
            reason="Retriever tidak mengembalikan dokumen apa pun.",
        )

    context = _teks_context(documents)

    try:
        if llm is None:
            from src.rag_chain import get_llm

            llm = get_llm()

        prompt = PROMPT_ENTITY_RESOLUTION.format(
            context=context,
            question=question,
        )

        response = invoke_llm_observed(
            llm,
            prompt,
            purpose="entity_resolution",
            question=question,
            provider=config.LLM_PROVIDER,
            safe_metadata={
                "context_length": len(context),
                "retrieved_count": len(documents),
            },
        )
        payload = extract_json_object(str(response.content or ""))

    except Exception as exception:
        return _hasil(
            STATUS_ERROR,
            level=LEVEL_SEMANTIK,
            llm_calls=1,
            reason=f"LLM gagal: {type(exception).__name__}: {exception}",
        )

    ticker = normalize_ticker(payload.get("ticker"))
    company = str(payload.get("company") or "").strip() or None
    market = str(payload.get("market") or "UNKNOWN").upper().strip()
    confidence = str(payload.get("confidence") or "low").lower().strip()

    alternatif = [
        kode
        for kode in (
            normalize_ticker(x) for x in (payload.get("alternatives") or [])
        )
        if kode
    ]

    bukti = [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "chunk_id": d.metadata.get("chunk_id"),
        }
        for d in documents[:3]
    ]

    # The LLM says it is ambiguous itself.
    if not ticker and alternatif:
        return _hasil(
            STATUS_AMBIGUOUS,
            level=LEVEL_SEMANTIK,
            candidates=alternatif,
            confidence="low",
            evidence=bukti,
            llm_calls=1,
            reason=str(payload.get("reason") or "")[:200]
            or "LLM menilai pertanyaannya menunjuk lebih dari satu emiten.",
        )

    if not ticker:
        return _hasil(
            STATUS_NOT_FOUND,
            level=LEVEL_SEMANTIK,
            evidence=bukti,
            llm_calls=1,
            reason=str(payload.get("reason") or "")[:200]
            or "LLM tidak menemukan entitas saham di pertanyaan ini.",
        )

    # Grounding: the code or name must really be in the context.
    didukung, alasan_grounding = _didukung_context(ticker, company, context)

    if not didukung:
        return _hasil(
            STATUS_NOT_FOUND,
            level=LEVEL_SEMANTIK,
            evidence=bukti,
            llm_calls=1,
            candidates=[ticker],
            reason=(
                f"LLM mengusulkan {ticker}, tetapi {alasan_grounding}. "
                "Usulan ditolak karena tidak bersumber dari dokumen."
            ),
        )

    # The exchange has to be known. If the LLM is unsure (market
    # "UNKNOWN") the entity does not count as resolved: without an
    # exchange the price symbol cannot be built correctly, and a wrong
    # symbol returns another company's price with no error at all.
    if market not in BURSA_DIKENAL:
        return _hasil(
            STATUS_NOT_FOUND,
            ticker=ticker,
            company=company,
            market=market,
            level=LEVEL_SEMANTIK,
            evidence=bukti,
            llm_calls=1,
            confidence=confidence,
            reason=(
                f"{ticker} dikenali, tetapi bursanya tidak dapat "
                f"dipastikan (market={market})."
            ),
        )

    return _hasil(
        STATUS_RESOLVED,
        ticker=ticker,
        company=company,
        market=market,
        level=LEVEL_SEMANTIK,
        confidence=confidence,
        evidence=bukti,
        llm_calls=1,
        reason=str(payload.get("reason") or "")[:200]
        or f"{ticker} diambil dari context hasil retrieval.",
    )


# ============================================================
# 6. ENTRY POINT
# ============================================================

def resolve_ticker(
    question,
    session_context=None,
    retriever=None,
    llm=None,
    k=None,
    izinkan_semantik=True,
):
    """
    Work out which stock this question is about.

    The levels run cheapest first. The semantic level only runs when the
    two before it came up empty, so a question like "harga BBRI
    sekarang" never touches PgVector or the LLM.

    Possible statuses:
        resolved    ticker ready to use
        ambiguous   several options -> the chatbot must ask back
        not_found   no stock entity could be pinned down
        error       retrieval or LLM failed
    """
    question = str(question or "").strip()

    if not question:
        return _hasil(STATUS_NOT_FOUND, reason="Pertanyaan kosong.")

    hasil = resolve_explicit(question)

    if hasil is not None:
        return hasil

    hasil = resolve_from_session(question, session_context)

    if hasil is not None:
        return hasil

    if not izinkan_semantik:
        return _hasil(
            STATUS_NOT_FOUND,
            reason="Tidak ada kode saham eksplisit, resolusi semantik dimatikan.",
        )

    return resolve_semantic(
        question,
        retriever=retriever,
        llm=llm,
        k=k,
    )


__all__ = [
    "STATUS_RESOLVED",
    "STATUS_AMBIGUOUS",
    "STATUS_NOT_FOUND",
    "STATUS_UNVERIFIED",
    "STATUS_ERROR",
    "LEVEL_EKSPLISIT",
    "LEVEL_SESI",
    "LEVEL_SEMANTIK",
    "detect_explicit_tickers",
    "normalize_ticker",
    "merujuk_percakapan_sebelumnya",
    "resolve_explicit",
    "resolve_explicit_many",
    "resolve_from_session",
    "resolve_semantic",
    "resolve_ticker",
]
