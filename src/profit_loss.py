"""
Profit/loss calculation. Deterministic, no LLM.

Language models get arithmetic wrong confidently, and a wrong financial
number does not look wrong. So the split is strict:

    Python : pulls out the entry price, computes the difference and the
             percentage
    LLM    : explains a result that is already calculated

The LLM never receives the sum to do; it receives the answer.
"""

import re


# ============================================================
# 1. ENTRY PRICE PARSER
# ============================================================
#
# The old version required the number to sit right after the verb
# ("beli di 4.000"), so it failed on the most common form of all -
# "beli BBRI di 4.000", with the stock code in between. The pattern
# below allows a few words in the gap but still requires a position
# keyword, so other numbers in the sentence do not get picked up.

KATA_POSISI = r"(?:beli|belinya|dibeli|entry|masuk|modal|average|avg|nyangkut)"

# At most three words between the position keyword and the number.
# Enough for "beli BBRI di" or "masuk saham BBRI di", not enough to
# cross into another clause.
#
# The lookahead matters: without it "Rp4.000" is eaten as one of those
# "words" (it starts with the letter R) and the number disappears -
# exactly the bug that made "Saya beli BBRI di Rp4.000" unreadable.
JEDA = r"(?:\s+(?!rp\.?\s*\d)(?!idr\s*\d)[A-Za-z][\w.]*){0,3}"

PENGANTAR_HARGA = r"(?:\s*(?:di|pada|dengan|seharga|sebesar|harga|:|@))?"

ANGKA = r"([0-9][0-9.,]*)"

POLA_ENTRY = re.compile(
    rf"{KATA_POSISI}{JEDA}{PENGANTAR_HARGA}\s*(?:rp\.?\s*|idr\s*)?{ANGKA}",
    flags=re.IGNORECASE,
)

# Words marking a number as NOT a purchase price. "Target BBRI 5.000"
# and "stop loss di 3.800" contain numbers, but those numbers belong to
# the research, not to the user's position.
POLA_BUKAN_ENTRY = re.compile(
    r"\b(target|tp\s*\d?|take\s*profit|stop\s*loss|sl\b|resistance|"
    r"support|proyeksi|prediksi)\b",
    flags=re.IGNORECASE,
)


def parse_angka_indonesia(teks):
    """
    "4.000"    -> 4000.0     (dot = thousands separator)
    "4,000"    -> 4000.0
    "4.210,50" -> 4210.50
    "4210.50"  -> 4210.50    (dot = decimal point)
    "9.75"     -> 975.0      (integer part too short; dot = thousands)

    A lone dot is only read as a decimal point when it is the only dot,
    one or two digits follow it, AND the integer part in front of it is
    longer than one digit. Indonesian prices are written "4.000", so a
    single digit before the dot is far more likely to be thousands than
    a decimal. That is why "9.75" comes back as 975.0, not 9.75.
    """
    angka = str(teks or "").strip()

    if not angka:
        return None

    if "." in angka and "," in angka:
        angka = angka.replace(".", "").replace(",", ".")

    elif "," in angka:
        bagian = angka.split(",")

        if len(bagian) == 2 and len(bagian[1]) in (1, 2):
            angka = angka.replace(",", ".")
        else:
            angka = angka.replace(",", "")

    elif "." in angka:
        bagian = angka.split(".")

        # "4210.50" -> decimal; "4.000" / "1.234.567" / "9.75" ->
        # thousands.
        if len(bagian) == 2 and len(bagian[1]) in (1, 2) and len(bagian[0]) > 1:
            pass
        else:
            angka = angka.replace(".", "")

    try:
        nilai = float(angka)
    except ValueError:
        return None

    return nilai if nilai > 0 else None


def ekstrak_harga_entry(question):
    """
    The purchase price the user mentioned, or None.

    Returns None for sentences where the number clearly is not a
    purchase price ("Target BBRI 5.000") - better to calculate nothing
    than to calculate a profit from somebody else's target number.
    """
    teks = str(question or "")

    match = POLA_ENTRY.search(teks)

    if not match:
        return None

    # If a "target"/"stop loss" marker sits between the position keyword
    # and the number, the number belongs to that marker.
    antara = teks[match.start():match.start(1)]

    if POLA_BUKAN_ENTRY.search(antara):
        return None

    return parse_angka_indonesia(match.group(1))


# ============================================================
# 2. CALCULATION
# ============================================================

POSISI_PROFIT = "profit"
POSISI_LOSS = "loss"
POSISI_BREAKEVEN = "breakeven"


def hitung_profit_loss(harga_entry, harga_sekarang):
    """
        difference     = harga_sekarang - harga_entry
        return_percent = difference / harga_entry * 100

        difference > 0 -> profit
        difference < 0 -> loss
        difference = 0 -> breakeven

    Returns None when either price is missing. None means "not
    calculated", and the caller has to treat it that way - not as zero.
    """
    if harga_entry is None or harga_sekarang is None:
        return None

    try:
        entry = float(harga_entry)
        sekarang = float(harga_sekarang)
    except (TypeError, ValueError):
        return None

    if entry <= 0:
        return None

    selisih = sekarang - entry
    persen = selisih / entry * 100.0

    if selisih > 0:
        posisi = POSISI_PROFIT
    elif selisih < 0:
        posisi = POSISI_LOSS
    else:
        posisi = POSISI_BREAKEVEN

    return {
        "entry_price": entry,
        "current_price": sekarang,
        "difference": selisih,
        "return_percent": persen,
        "position": posisi,
    }


# ============================================================
# 3. PRESENTATION
# ============================================================

LABEL_POSISI = {
    POSISI_PROFIT: "untung",
    POSISI_LOSS: "rugi",
    POSISI_BREAKEVEN: "impas",
}


def format_profit_loss(profit_loss, mata_uang="Rp"):
    """
    Display-ready block of numbers. Also used as part of the prompt, so
    the LLM explains these numbers instead of recomputing them.
    """
    if not profit_loss:
        return ""

    entry = profit_loss["entry_price"]
    sekarang = profit_loss["current_price"]
    selisih = profit_loss["difference"]
    persen = profit_loss["return_percent"]
    posisi = LABEL_POSISI.get(profit_loss["position"], profit_loss["position"])

    tanda = "+" if selisih > 0 else ""

    return (
        "PERHITUNGAN POSISI (dihitung Python, bukan model):\n"
        f"  harga beli    : {mata_uang}{entry:,.2f}\n"
        f"  harga sekarang: {mata_uang}{sekarang:,.2f}\n"
        f"  selisih       : {tanda}{mata_uang}{selisih:,.2f}\n"
        f"  return        : {tanda}{persen:.2f}%\n"
        f"  posisi        : {posisi}"
    )


def ringkas_profit_loss(profit_loss, label=None, mata_uang="Rp"):
    """One sentence to display without the LLM."""
    if not profit_loss:
        return None

    posisi = profit_loss["position"]
    selisih = profit_loss["difference"]
    persen = profit_loss["return_percent"]
    nama = label or "Posisi Anda"

    if posisi == POSISI_BREAKEVEN:
        return (
            f"{nama} impas: harga sekarang sama dengan harga beli "
            f"{mata_uang}{profit_loss['entry_price']:,.0f}."
        )

    kata = LABEL_POSISI[posisi]

    return (
        f"{nama} {kata} {mata_uang}{abs(selisih):,.0f} per lembar "
        f"({persen:+.2f}%): beli di {mata_uang}"
        f"{profit_loss['entry_price']:,.0f}, sekarang {mata_uang}"
        f"{profit_loss['current_price']:,.0f}."
    )


__all__ = [
    "POSISI_PROFIT",
    "POSISI_LOSS",
    "POSISI_BREAKEVEN",
    "parse_angka_indonesia",
    "ekstrak_harga_entry",
    "hitung_profit_loss",
    "format_profit_loss",
    "ringkas_profit_loss",
]
