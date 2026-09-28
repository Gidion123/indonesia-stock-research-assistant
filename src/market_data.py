"""
Market data: the only place in this project that touches yfinance.

Three rules are enforced here, each one because of a failure this
project actually hit:

1.  A symbol that cannot be pinned down is NOT requested.
    yfinance does not raise on a wrong symbol - it quietly returns
    another security's price. `LYC` with no suffix is not Lynas; it is a
    different stock on a US exchange. A wrong price is far more
    dangerous than no price, so an unknown exchange yields None rather
    than a bare ticker.

2.  Failures are reported as failures.
    Like `ask_question()`, the result carries a `status` so the caller
    can tell "no price available" from "the price is zero".

3.  No real-time claims.
    yfinance data is delayed and not tick-by-tick. Every result carries
    `as_of` and `disclaimer`, and the prompts have to show them.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from src import config
from src.market_symbols import (
    BURSA_DEFAULT,
    BURSA_KE_SUFIKS_YAHOO,
    build_yahoo_symbol,
)


# ============================================================
# 1. EXCHANGE -> YAHOO SYMBOL MAPPING
# ============================================================
#
# This is Yahoo Finance's mechanical convention, not domain knowledge:
# no company name and no ticker is hardcoded here, only the mapping
# from exchange code to symbol suffix.

# The symbol conventions live in `market_symbols` so `entity_resolver`
# can use them without importing this module - and therefore with no
# way to trigger a network request.
BURSA_DIDUKUNG = tuple(sorted(BURSA_KE_SUFIKS_YAHOO))

DISCLAIMER_HARGA = (
    "Harga dari Yahoo Finance dan dapat tertunda beberapa menit; "
    "bukan kuotasi real-time bursa."
)

DISCLAIMER_EODHD = (
    "Fallback EODHD: harga penutupan akhir hari (EOD), bukan kuotasi "
    "intraday atau real-time."
)


# ============================================================
# 2. PRICE CACHE WITH TTL
# ============================================================
#
# lru_cache is not used here: a price cached forever is wrong within
# minutes. A short TTL is enough to stop one chat session hitting
# yfinance over and over for the same ticker.

CACHE_TTL_SECONDS = 60

# Data EOD tidak berubah setiap menit. Cache lebih lama juga menghemat
# kuota free EODHD.
EODHD_CACHE_TTL_SECONDS = 6 * 60 * 60

_CACHE_HARGA: dict = {}

# Counters for price lookups. `requests` goes up once per uncached
# `ambil_harga`, not once per HTTP call: a single lookup can hit two
# Yahoo endpoints (`fast_info`, then the `history()` fallback). They
# exist so the claim "RAG does not call Yahoo" can be backed by a
# number instead of believed.
_STATISTIK = {"requests": 0, "cache_hits": 0}


def statistik_market_data():
    """Copy of the counters: uncached price lookups, cache hits."""
    return dict(_STATISTIK)


def reset_statistik_market_data():
    _STATISTIK["requests"] = 0
    _STATISTIK["cache_hits"] = 0


def bersihkan_cache_harga():
    """Used by tests, and when the user asks for an explicit refresh."""
    _CACHE_HARGA.clear()


def _ambil_dari_cache(symbol):
    entry = _CACHE_HARGA.get(symbol)

    if not entry:
        return None

    disimpan_pada, payload = entry

    ttl = (
        EODHD_CACHE_TTL_SECONDS
        if payload.get("source") == "eodhd_eod"
        else CACHE_TTL_SECONDS
    )

    if time.time() - disimpan_pada > ttl:
        _CACHE_HARGA.pop(symbol, None)
        return None

    return payload


# ============================================================
# 3. PRICE FETCHING
# ============================================================

TRANSIENT_HINTS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "429",
    "503",
    "502",
    "rate limit",
    # Network blocked by a proxy or firewall. Worth separating from
    # "unknown symbol": if the network is down, concluding that a ticker
    # is invalid is simply the wrong conclusion.
    "403",
    "failed to perform",
    "curl:",
    "tunnel",
    "ssl",
    "proxy",
    "name resolution",
    "unreachable",
)


def _klasifikasi_error(exception):
    message = str(exception).lower()

    if any(hint in message for hint in TRANSIENT_HINTS):
        return "transient_error"

    return "error"


# Yahoo refusing because it has been asked too often. Different from
# other transient failures: trying the second endpoint now only adds
# load to a quota that is already spent, and will be refused too.
POLA_RATE_LIMIT = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
)


def _kena_rate_limit(exception):
    if exception is None:
        return False

    jejak = f"{type(exception).__name__} {exception}".lower()

    return any(pola in jejak for pola in POLA_RATE_LIMIT)


def _galat_paling_menentukan(*kandidat):
    """
    Given several failures, pick the one that explains the most.

    Network failures win over the rest. The reason is asymmetric: an
    empty `fast_info` raises a plain KeyError - a failure that looks
    "local" - when the real cause may be a dead network on the next hop.
    If that KeyError were the one used, the status would become
    `not_found`, turning "Yahoo is unreachable" into "this stock does
    not exist", a wrong conclusion that gets passed on to the user.
    """
    galat = [item for item in kandidat if item is not None]

    for item in galat:
        if _klasifikasi_error(item) == "transient_error":
            return item

    return galat[0] if galat else None


def _hasil_kosong(symbol, status, error=None):
    return {
        "symbol": symbol,
        "price": None,
        "previous_close": None,
        "change": None,
        "change_percent": None,
        "currency": None,
        "exchange": None,
        "short_name": None,
        "as_of": None,
        "source": None,
        "status": status,
        "error": error,
        "disclaimer": DISCLAIMER_HARGA,
    }


def _senyapkan_yfinance():
    """
    yfinance prints its own warnings ("possibly delisted", cookie
    failures) through logging. In a chat app those land in the middle of
    the conversation, and in tests they bury the real failure. The
    status this module returns already carries the same information in
    structured form.
    """
    import logging

    for nama in ("yfinance", "peewee", "urllib3"):
        logging.getLogger(nama).setLevel(logging.CRITICAL)


def _ambil_fast_info(ticker_obj):
    """
    yfinance moves fields around between versions. `fast_info` is much
    cheaper than `.info` (it does not pull the whole company profile),
    so it is tried first.

    Returns (data, first_error) - the FIRST error collected while
    reading the fields, `galat[0]`, not the last one. Returning an error
    at all is what matters: without it a dead network cannot be told
    apart from a symbol that genuinely does not exist, and concluding
    "this ticker is invalid" because a proxy blocked Yahoo is a wrong
    conclusion that spreads to the resolver.
    """
    galat: list = []

    try:
        fast = ticker_obj.fast_info
    except Exception as exception:
        return {}, exception

    def ambil(*nama):
        for key in nama:
            value = None

            # yfinance throws all sorts of exceptions from inside
            # (network errors included) when a field is accessed lazily.
            # Whatever comes out is treated as "this field is missing",
            # so status classification happens in one place only.
            try:
                value = fast[key]
            except Exception as exception:
                galat.append(exception)

                try:
                    value = getattr(fast, key, None)
                except Exception as exception_lagi:
                    galat.append(exception_lagi)
                    value = None

            if value is not None:
                return value

        return None

    data = {
        "price": ambil("last_price", "lastPrice"),
        "previous_close": ambil("previous_close", "previousClose"),
        "currency": ambil("currency"),
        "exchange": ambil("exchange"),
    }

    return data, (galat[0] if galat else None)


def _ambil_dari_history(ticker_obj):
    """
    Fallback for when `fast_info` gives no price.

    `fast_info` is the cheapest path but also the most fragile: it moves
    between yfinance versions and often comes back EMPTY without raising
    anything, which makes it indistinguishable from a symbol that is not
    listed. `history()` uses the chart endpoint, which is far more
    stable, and gives the previous close from the second-to-last row at
    the same time.

    This is not a competing second source: it only runs when the first
    path produced no number, so a question that already succeeded costs
    no extra request.

    Returns (data, error). Any failure is returned as an error rather
    than swallowed, keeping status classification in one place.
    """
    try:
        frame = ticker_obj.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
        )
    except Exception as exception:
        return {}, exception

    try:
        if frame is None or len(frame) == 0:
            return {}, None

        penutupan = [
            float(nilai)
            for nilai in frame["Close"].tolist()
            # NaN != NaN. Holiday rows can be empty.
            if nilai == nilai
        ]

        if not penutupan:
            return {}, None

        return (
            {
                "price": penutupan[-1],
                "previous_close": (
                    penutupan[-2] if len(penutupan) >= 2 else None
                ),
                # history() does not carry this metadata; leave it empty
                # rather than guess.
                "currency": None,
                "exchange": None,
            },
            None,
        )

    except Exception as exception:
        return {}, exception


# ============================================================
# 3A. EODHD FALLBACK
# ============================================================
#
# Yahoo remains the primary market-data provider. EODHD is deliberately
# not queried when Yahoo succeeds. It is only a safety net for temporary
# Yahoo failures such as rate limiting or network problems.
#
# The free EODHD path used here is End-of-Day data. It must therefore
# never be presented as an intraday or real-time quote.


def _ambil_eodhd_eod(symbol):
    """
    Fetch the latest available EOD close from EODHD.

    This fallback is deliberately limited to IDX symbols (`.JK`) for now.
    The project is focused on Indonesian equities, and silently translating
    symbols from other exchanges without an explicit provider mapping would
    recreate the same class of bug that `market_symbols.py` prevents.

    Returns a normal market-data payload when successful, otherwise None.
    None means "fallback unavailable", not "the symbol definitely does not
    exist".
    """
    if config.MARKET_DATA_FALLBACK != "eodhd":
        return None

    if not config.EODHD_API_KEY:
        return None

    symbol = str(symbol or "").upper().strip()

    if not symbol.endswith(".JK"):
        return None

    # Ask for enough calendar days to reliably include at least two
    # trading sessions around weekends and ordinary exchange holidays.
    hari_ini = datetime.now(timezone.utc).date()
    tanggal_mulai = hari_ini - timedelta(days=14)

    query = urlencode(
        {
            "api_token": config.EODHD_API_KEY,
            "fmt": "json",
            "period": "d",
            "order": "d",
            "from": tanggal_mulai.isoformat(),
            "to": hari_ini.isoformat(),
        }
    )

    url = (
        f"{config.EODHD_BASE_URL}/eod/"
        f"{quote(symbol, safe='')}?{query}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "StockResearchAssistant/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(
            request,
            timeout=config.MARKET_DATA_TIMEOUT_SECONDS,
        ) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )

    except Exception:
        # Do not include the URL or raw request in the returned error.
        # The EODHD API key lives in the query string and must never leak
        # into the UI, application logs, or evaluation output.
        return None

    if not isinstance(payload, list) or not payload:
        return None

    baris = [
        item
        for item in payload
        if isinstance(item, dict)
        and item.get("close") is not None
        and item.get("date")
    ]

    if not baris:
        return None

    # Defensive sorting: do not depend on the provider preserving the
    # requested descending order forever.
    baris.sort(
        key=lambda item: str(item.get("date") or ""),
        reverse=True,
    )

    terbaru = baris[0]
    sebelumnya = baris[1] if len(baris) >= 2 else None

    try:
        price = float(terbaru["close"])

        previous_close = (
            float(sebelumnya["close"])
            if sebelumnya is not None
            and sebelumnya.get("close") is not None
            else None
        )

    except (TypeError, ValueError, KeyError):
        return None

    if price <= 0:
        return None

    if previous_close is not None and previous_close > 0:
        change = price - previous_close
        change_percent = change / previous_close * 100.0
    else:
        change = None
        change_percent = None

    return {
        "symbol": symbol,
        "price": price,
        "previous_close": previous_close,
        "change": (
            round(change, 4)
            if change is not None
            else None
        ),
        "change_percent": (
            round(change_percent, 4)
            if change_percent is not None
            else None
        ),
        "currency": "IDR",
        "exchange": "XIDX",
        "short_name": None,
        # For EOD data the trading date is more meaningful than the time
        # at which our server happened to perform the HTTP request.
        "as_of": terbaru.get("date"),
        "source": "eodhd_eod",
        "status": "ok",
        "error": None,
        "disclaimer": DISCLAIMER_EODHD,
    }


def _fallback_atau_transient(symbol, yahoo_error):
    """
    Try the configured EODHD fallback after a transient Yahoo failure.

    A failed fallback must never hide the original reason Yahoo failed.
    If EODHD cannot provide a valid number, keep `price=None` and preserve
    the Yahoo transient error instead of guessing or fabricating a value.
    """
    fallback = _ambil_eodhd_eod(symbol)

    if fallback is not None:
        _CACHE_HARGA[symbol] = (
            time.time(),
            fallback,
        )

        return dict(
            fallback,
            from_cache=False,
        )

    detail = (
        f"{type(yahoo_error).__name__}: {yahoo_error}"
        if yahoo_error is not None
        else "Yahoo Finance sementara tidak tersedia."
    )

    return _hasil_kosong(
        symbol,
        "transient_error",
        detail,
    )


def ambil_harga(symbol, gunakan_cache=True):
    """
    Fetch the last price for one Yahoo symbol.

    Returns a dict with a `status`:
        ok               price available
        invalid_symbol   symbol empty or could not be built
        not_found        yfinance does not know this symbol
        transient_error  network or rate limit - safe to retry
        error            some other failure
        unavailable      yfinance is not installed

    A price is NEVER invented: if no number comes back, `price` is None
    and the status is something other than "ok".
    """
    symbol = str(symbol or "").upper().strip()

    if not symbol:
        return _hasil_kosong(
            symbol,
            "invalid_symbol",
            "Simbol kosong.",
        )

    if gunakan_cache:
        cached = _ambil_dari_cache(symbol)

        if cached is not None:
            _STATISTIK["cache_hits"] += 1

            return dict(
                cached,
                from_cache=True,
            )

    try:
        import yfinance
    except ImportError:
        return _hasil_kosong(
            symbol,
            "unavailable",
            "Paket yfinance belum terpasang.",
        )

    _senyapkan_yfinance()
    _STATISTIK["requests"] += 1

    try:
        ticker_obj = yfinance.Ticker(symbol)

        data, galat = _ambil_fast_info(ticker_obj)
        sumber = "fast_info"
        price = data.get("price")

        if price is None and _kena_rate_limit(galat):
            # The quota is spent. The second endpoint will be refused
            # too, and trying it only extends the refusal window.

            # Do not retry Yahoo here. Instead, try the independent
            # EODHD End-of-Day fallback. It is explicitly labelled EOD
            # so the caller never mistakes it for an intraday quote.
            return _fallback_atau_transient(
                symbol,
                galat,
            )

        if price is None:
            # Second path. See `_ambil_dari_history`: fast_info comes
            # back empty far more often than a symbol genuinely does not
            # exist.
            data_history, galat_history = _ambil_dari_history(
                ticker_obj
            )

            if data_history.get("price") is not None:
                # Keep fast_info's metadata if it managed to fill in;
                # history does not carry it.
                data = {
                    **data_history,
                    "currency": data.get("currency"),
                    "exchange": data.get("exchange"),
                }

                sumber = "history"
                price = data["price"]
                galat = None

            else:
                galat = _galat_paling_menentukan(
                    galat,
                    galat_history,
                )

        if price is None:

            # No price AND a network error: what failed is the lookup,
            # not the symbol. The status has to be transient so the
            # resolver does not conclude the ticker is invalid.
            if galat is not None:
                status = _klasifikasi_error(galat)

                if status == "transient_error":
                    # Yahoo failed for an operational reason rather than
                    # because the symbol is certainly invalid. This is the
                    # only class of failure where the secondary provider
                    # is allowed to take over.
                    return _fallback_atau_transient(
                        symbol,
                        galat,
                    )

            return _hasil_kosong(
                symbol,
                "not_found",
                f"yfinance tidak mengembalikan harga untuk {symbol}.",
            )

        previous_close = data.get("previous_close")

        if previous_close:
            change = float(price) - float(previous_close)
            change_percent = (
                change / float(previous_close) * 100.0
            )
        else:
            change = None
            change_percent = None

        hasil = {
            "symbol": symbol,
            "price": float(price),
            "previous_close": (
                float(previous_close)
                if previous_close
                else None
            ),
            "change": (
                round(change, 4)
                if change is not None
                else None
            ),
            "change_percent": (
                round(change_percent, 4)
                if change_percent is not None
                else None
            ),
            "currency": data.get("currency"),
            "exchange": data.get("exchange"),
            "short_name": None,
            "as_of": datetime.now(
                timezone.utc
            ).isoformat(timespec="seconds"),
            # Which path produced this number. Recorded so "why is the
            # price slightly different" has an answer, not a guess.
            "source": sumber,
            "status": "ok",
            "error": None,
            "disclaimer": DISCLAIMER_HARGA,
        }

        _CACHE_HARGA[symbol] = (
            time.time(),
            hasil,
        )

        return dict(
            hasil,
            from_cache=False,
        )

    except Exception as exception:
        status = _klasifikasi_error(exception)

        if status == "transient_error":
            # Catch operational failures that escaped the normal
            # fast_info/history paths and give the EOD fallback the same
            # opportunity to recover the request.
            return _fallback_atau_transient(
                symbol,
                exception,
            )

        return _hasil_kosong(
            symbol,
            status,
            f"{type(exception).__name__}: {exception}",
        )


# A symbol that is certain to exist as long as Yahoo is reachable: the
# IHSG index. Used as a canary, not as data.
SIMBOL_CANARY = "^JKSE"

CANARY_TTL_SECONDS = 120

_STATUS_CANARY: dict = {}


def jaringan_sehat():
    """
    Is Yahoo actually reachable right now?

    This is needed because yfinance SWALLOWS network failures without
    raising: when a proxy blocks Yahoo, `fast_info` just returns empty
    values - exactly like a symbol that is not listed. Without the
    canary a dead network reads as "every ticker is invalid", and the
    resolver would reject correct tickers.

    The result is cached briefly because the health check is asked
    repeatedly within a session, and a canary lookup is itself a Yahoo
    request. The TTL keeps that to one request every two minutes.
    """
    sekarang = time.time()
    entry = _STATUS_CANARY.get("nilai")

    if entry and sekarang - entry[0] <= CANARY_TTL_SECONDS:
        return entry[1]

    hasil = ambil_harga(
        SIMBOL_CANARY,
        gunakan_cache=False,
    )

    sehat = hasil["status"] == "ok"

    _STATUS_CANARY["nilai"] = (
        sekarang,
        sehat,
    )

    return sehat


def bersihkan_cache_canary():
    _STATUS_CANARY.clear()


# `periksa_simbol`/`symbol_exists` were deliberately REMOVED.
#
# `entity_resolver` used to call them to confirm a code really was a
# ticker before passing it on. The effect was that every capitalised
# word in a user's question became a Yahoo Finance request - including
# on RAG questions that needed no price at all.
#
# The resolver now decides identity deterministically, and correctness
# proves itself here: a wrong symbol yields status "not_found" on a
# price request that was going to happen anyway.


# ============================================================
# 4. FORMATTING FOR THE PROMPT
# ============================================================

def format_market_data(data, nama_perusahaan=None):
    """
    Turn an `ambil_harga()` result into a text block for the prompt.

    The disclaimer and the timestamp always come along: the model must
    not present these numbers as a live exchange quote.
    """
    if not data or data.get("status") != "ok":
        return "DATA PASAR: tidak tersedia."

    mata_uang = data.get("currency") or ""
    harga = data["price"]

    # EODHD free fallback represents a completed daily close, not an
    # intraday quote. Keep that distinction visible inside the LLM prompt.
    menggunakan_eod = (
        data.get("source") == "eodhd_eod"
    )

    if menggunakan_eod:
        judul_data = "DATA PASAR (EOD fallback):"
        label_harga = "penutupan EOD"
    else:
        judul_data = "DATA PASAR (live):"
        label_harga = "harga terakhir"

    baris = [
        judul_data,
        f"  simbol         : {data['symbol']}",
    ]

    if nama_perusahaan:
        baris.append(
            f"  perusahaan     : {nama_perusahaan}"
        )

    baris.append(
        f"  {label_harga:<15}: "
        f"{mata_uang} {harga:,.2f}".rstrip()
    )

    if data.get("previous_close") is not None:
        baris.append(
            f"  penutupan lalu : {mata_uang} "
            f"{data['previous_close']:,.2f}".rstrip()
        )

    if data.get("change_percent") is not None:
        arah = "+" if data["change"] >= 0 else ""

        baris.append(
            f"  perubahan      : {arah}{data['change']:,.2f} "
            f"({arah}{data['change_percent']:.2f}%)"
        )

    if menggunakan_eod:
        baris.append(
            f"  tanggal EOD     : {data['as_of']}"
        )
    else:
        baris.append(
            f"  diambil pada   : {data['as_of']}"
        )

    baris.append(
        f"  catatan        : {data['disclaimer']}"
    )

    return "\n".join(baris)


def ringkas_untuk_pengguna(data, nama_perusahaan=None):
    """
    One sentence to show the user directly without an LLM, used by the
    fast LIVE_PRICE path.
    """
    if not data or data.get("status") != "ok":
        return None

    mata_uang = data.get("currency") or ""
    label = nama_perusahaan or data["symbol"]

    # Do not print the symbol twice when the company name is unknown and
    # the label is the symbol itself.
    judul = (
        label
        if label == data["symbol"]
        else f"{label} ({data['symbol']})"
    )

    menggunakan_eod = (
        data.get("source") == "eodhd_eod"
    )

    if menggunakan_eod:
        kalimat = (
            f"{judul} memiliki harga penutupan EOD terakhir "
            f"{mata_uang} {data['price']:,.2f}"
        )

        if data.get("as_of"):
            kalimat += f" pada {data['as_of']}"

    else:
        kalimat = (
            f"{judul} terakhir di "
            f"{mata_uang} {data['price']:,.2f}"
        )

    if data.get("change_percent") is not None:
        arah = "+" if data["change"] >= 0 else ""

        kalimat += (
            f" ({arah}{data['change_percent']:.2f}%)"
        )

    return (
        f"{kalimat}. "
        f"{data['disclaimer']}"
    )


__all__ = [
    "BURSA_KE_SUFIKS_YAHOO",
    "BURSA_DIDUKUNG",
    "BURSA_DEFAULT",
    "DISCLAIMER_HARGA",
    "DISCLAIMER_EODHD",
    "build_yahoo_symbol",
    "ambil_harga",
    "statistik_market_data",
    "reset_statistik_market_data",
    "format_market_data",
    "ringkas_untuk_pengguna",
    "bersihkan_cache_harga",
    "bersihkan_cache_canary",
    "jaringan_sehat",
]