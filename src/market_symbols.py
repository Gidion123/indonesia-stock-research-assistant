"""
Exchange symbol conventions. Pure data: no I/O, no network, no
dependency on any other module.

This module exists so `entity_resolver` can recognise exchange codes
without importing `market_data`. That split is deliberate: the resolver
decides a stock's IDENTITY, market_data is what talks to Yahoo Finance.
As long as the resolver imports market_data there is a standing
temptation to call the exchange just to guess an identity, which turns
every capitalised word in a question into a network request.

What is in here is Yahoo Finance's mechanical convention (exchange ->
symbol suffix), not domain knowledge: no company name and no ticker is
hardcoded anywhere in this file.
"""

BURSA_KE_SUFIKS_YAHOO = {
    "IDX": ".JK",      # Indonesia Stock Exchange
    "NYSE": "",        # United States
    "NASDAQ": "",
    "US": "",
    "ASX": ".AX",      # Australia
    "LSE": ".L",       # United Kingdom
    "SGX": ".SI",      # Singapore
    "HKEX": ".HK",     # Hong Kong
    "TSE": ".T",       # Japan
    "KRX": ".KS",      # South Korea
    "SET": ".BK",      # Thailand
    "BURSA": ".KL",    # Malaysia
}

BURSA_DIKENAL = frozenset(BURSA_KE_SUFIKS_YAHOO)

BURSA_DIDUKUNG = tuple(sorted(BURSA_KE_SUFIKS_YAHOO))

# This assistant serves Indonesian equity research.
BURSA_DEFAULT = "IDX"


def build_yahoo_symbol(ticker, market=BURSA_DEFAULT):
    """
    Build a Yahoo symbol from a ticker plus an exchange code.

    Returns None when the exchange is not known. None here means "do not
    ask for a price", not "use the ticker as-is" - a bare ticker on
    Yahoo will match a different US security and quietly return the
    wrong price.

        build_yahoo_symbol("BBRI", "IDX")  -> "BBRI.JK"
        build_yahoo_symbol("MP", "NYSE")   -> "MP"
        build_yahoo_symbol("LYC", "ASX")   -> "LYC.AX"
        build_yahoo_symbol("LYC", "OTHER") -> None
    """
    ticker = str(ticker or "").upper().strip()

    if not ticker:
        return None

    market = str(market or "").upper().strip()

    suffix = BURSA_KE_SUFIKS_YAHOO.get(market)

    if suffix is None:
        return None

    return f"{ticker}{suffix}"


def bersihkan_sufiks(ticker):
    """
    "BBRI.JK" -> "BBRI",  "LYC.AX" -> "LYC",  "bbri" -> "BBRI"
    """
    kode = str(ticker or "").upper().strip()

    for sufiks in BURSA_KE_SUFIKS_YAHOO.values():
        if sufiks and kode.endswith(sufiks):
            return kode[: -len(sufiks)]

    return kode


__all__ = [
    "BURSA_KE_SUFIKS_YAHOO",
    "BURSA_DIKENAL",
    "BURSA_DIDUKUNG",
    "BURSA_DEFAULT",
    "build_yahoo_symbol",
    "bersihkan_sufiks",
]
