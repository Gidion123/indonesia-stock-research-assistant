"""
Why the price is not coming through.

    python -m scripts.diagnose_market_data
    python -m scripts.diagnose_market_data --simbol BBRI.JK

The UI deliberately keeps technical errors away from the user - the real
status sits in the `error` field of the result, not on screen. This
script opens that field, then tries each yfinance endpoint one at a time
so that "price not available" can be narrowed down to one of:

    network / proxy       cannot reach Yahoo at all
    rate limit            Yahoo is refusing, too many requests
    yfinance version      the endpoint being used has changed
    symbol                the symbol really is not listed

This script only reads. Nothing is changed.
"""

import argparse


def _garis(judul):
    print()
    print("=" * 68)
    print(judul)
    print("=" * 68)


def versi():
    _garis("VERSI")

    try:
        import yfinance

        print(f"  yfinance          : {yfinance.__version__}")
    except Exception as exception:
        print(f"  yfinance          : GAGAL diimpor -> {exception!r}")
        return

    try:
        import curl_cffi

        print(f"  curl_cffi         : {getattr(curl_cffi, '__version__', 'ada')}")
    except Exception:
        print("  curl_cffi         : tidak terpasang")

    try:
        import requests

        print(f"  requests          : {requests.__version__}")
    except Exception:
        pass

    from pathlib import Path

    pin = Path("requirements.txt")

    if pin.exists():
        for baris in pin.read_text(encoding="utf-8").splitlines():
            if baris.lower().startswith("yfinance"):
                print(f"  dipatok di requirements.txt: {baris.strip()}")


def lewat_modul(simbol):
    """Exactly the path the application uses."""
    from src.market_data import ambil_harga, statistik_market_data

    _garis(f"LEWAT src/market_data.py  ({simbol})")

    hasil = ambil_harga(simbol, gunakan_cache=False)

    print(f"  status            : {hasil['status']}")
    print(f"  sumber            : {hasil.get('source')}")
    print(f"  price             : {hasil['price']}")
    print(f"  previous_close    : {hasil['previous_close']}")
    print(f"  currency          : {hasil['currency']}")
    print(f"  error             : {hasil['error']}")
    print(f"  statistik         : {statistik_market_data()}")

    return hasil


def endpoint_satu_per_satu(simbol):
    """
    Try each endpoint directly, so a failure can be pinned to one of them.
    """
    _garis(f"ENDPOINT yfinance SATU PER SATU  ({simbol})")

    try:
        import yfinance
    except Exception as exception:
        print(f"  yfinance tidak bisa diimpor: {exception!r}")
        return

    import logging

    for nama in ("yfinance", "peewee", "urllib3"):
        logging.getLogger(nama).setLevel(logging.CRITICAL)

    ticker = yfinance.Ticker(simbol)

    # --- fast_info ---
    try:
        fast = ticker.fast_info
        kunci = list(fast.keys()) if hasattr(fast, "keys") else "(bukan dict)"
        print(f"  fast_info         : OK, kunci={kunci}")

        for nama in ("last_price", "previous_close", "currency"):
            try:
                print(f"      {nama:<16}= {fast[nama]}")
            except Exception as exception:
                print(f"      {nama:<16}! {type(exception).__name__}: {exception}")

    except Exception as exception:
        print(f"  fast_info         : GAGAL {type(exception).__name__}: {exception}")

    # --- history ---
    try:
        frame = ticker.history(period="5d", interval="1d", auto_adjust=False)
        print(f"  history(5d)       : OK, {len(frame)} baris")

        if len(frame):
            print(f"      Close terakhir = {frame['Close'].tolist()[-2:]}")

    except Exception as exception:
        print(f"  history(5d)       : GAGAL {type(exception).__name__}: {exception}")

    # --- download ---
    try:
        frame = yfinance.download(
            simbol,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        print(f"  download(5d)      : OK, {len(frame)} baris")

    except Exception as exception:
        print(f"  download(5d)      : GAGAL {type(exception).__name__}: {exception}")


def kesimpulan(hasil):
    _garis("BACA HASILNYA BEGINI")

    status = hasil["status"] if hasil else "?"
    pesan = str((hasil or {}).get("error") or "").lower()

    if status == "ok":
        print("  Harga berhasil diambil. Kalau UI masih bilang tidak")
        print("  tersedia, masalahnya bukan di sini — kirimkan lagi")
        print("  pertanyaan persis yang Anda ketik di Streamlit.")
        return

    if "403" in pesan or "tunnel" in pesan or "proxy" in pesan:
        print("  Yahoo tidak bisa dihubungi dari mesin ini (proxy/firewall).")
        print("  Bukan masalah kode.")

    elif "429" in pesan or "rate" in pesan or "too many" in pesan:
        print("  Yahoo membatasi permintaan (rate limit). yfinance versi")
        print("  lama paling sering kena ini; versi baru memakai curl_cffi")
        print("  yang jauh lebih jarang ditolak.")

    elif status == "not_found":
        print("  Kedua endpoint jalan tetapi simbolnya tidak mengembalikan")
        print("  harga. Periksa simbolnya, atau bursa sedang tutup dan")
        print("  tidak ada data sama sekali untuk 5 hari terakhir.")

    else:
        print("  Kegagalan lain. Lihat baris `error` di atas dan bandingkan")
        print("  dengan bagian ENDPOINT: endpoint mana yang masih jalan")
        print("  menentukan apakah ini soal versi yfinance atau jaringan.")

    print()
    print("  Kirimkan seluruh keluaran skrip ini apa adanya.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simbol", default="BBRI.JK")
    args = parser.parse_args()

    versi()
    hasil = lewat_modul(args.simbol)
    endpoint_satu_per_satu(args.simbol)
    kesimpulan(hasil)


if __name__ == "__main__":
    main()
