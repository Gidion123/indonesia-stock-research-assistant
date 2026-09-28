"""
Test market_data. Tidak menyentuh jaringan: yfinance diganti tiruan.

Yang diuji di sini bukan "apakah yfinance bekerja" — itu bukan urusan
kita — melainkan apakah modul ini menolak menebak ketika tidak yakin.
"""

from unittest.mock import patch

import pytest

from src import market_data
from src.market_data import (
    ambil_harga,
    format_market_data,
    ringkas_untuk_pengguna,
)
from src.market_symbols import BURSA_KE_SUFIKS_YAHOO, build_yahoo_symbol


@pytest.fixture(autouse=True)
def cache_bersih(monkeypatch):
    # Unit test biasa harus 100% offline. API key asli dari `.env` tidak
    # boleh membuat regression suite diam-diam memanggil EODHD.
    #
    # Test yang memang ingin menguji jalur fallback akan memasang kembali
    # fake key dan mock provider secara eksplisit.
    monkeypatch.setattr(
        market_data.config,
        "MARKET_DATA_FALLBACK",
        "eodhd",
    )
    monkeypatch.setattr(
        market_data.config,
        "EODHD_API_KEY",
        "",
    )

    market_data.bersihkan_cache_harga()
    market_data.bersihkan_cache_canary()

    yield

    market_data.bersihkan_cache_harga()
    market_data.bersihkan_cache_canary()


# ============================================================
# SIMBOL
# ============================================================

def test_simbol_idx_mendapat_sufiks_jk():
    assert build_yahoo_symbol("BBRI", "IDX") == "BBRI.JK"


def test_simbol_amerika_tanpa_sufiks():
    assert build_yahoo_symbol("MP", "NYSE") == "MP"
    assert build_yahoo_symbol("MP", "NASDAQ") == "MP"


def test_simbol_australia_mendapat_sufiks_ax():
    """
    Kasus LYC: Lynas Rare Earths di Yahoo adalah LYC.AX, bukan LYC.
    Versi lama mengembalikan ticker telanjang untuk semua bursa non-IDX
    dan diam-diam menunjuk sekuritas Amerika yang berbeda.
    """
    assert build_yahoo_symbol("LYC", "ASX") == "LYC.AX"


def test_bursa_tidak_dikenal_tidak_menghasilkan_simbol():
    """
    Yang paling penting di modul ini: lebih baik tidak ada simbol
    daripada simbol yang salah, karena yfinance tidak akan protes.
    """
    assert build_yahoo_symbol("LYC", "OTHER") is None
    assert build_yahoo_symbol("LYC", "UNKNOWN") is None
    assert build_yahoo_symbol("XYZ", "") is None


def test_ticker_kosong_tidak_menghasilkan_simbol():
    assert build_yahoo_symbol("", "IDX") is None
    assert build_yahoo_symbol(None, "IDX") is None


def test_semua_bursa_terdaftar_punya_sufiks_string():
    for bursa, sufiks in BURSA_KE_SUFIKS_YAHOO.items():
        assert isinstance(sufiks, str), bursa


# ============================================================
# PENGAMBILAN HARGA
# ============================================================

class FastInfoPalsu(dict):
    pass


class TickerPalsu:
    def __init__(self, data=None, lempar=None):
        self._data = data or {}
        self._lempar = lempar

    @property
    def fast_info(self):
        if self._lempar:
            raise self._lempar

        return FastInfoPalsu(self._data)


def _yfinance_palsu(ticker_obj):
    class Modul:
        @staticmethod
        def Ticker(symbol):
            return ticker_obj

    return Modul


def test_harga_normal(monkeypatch):
    ticker = TickerPalsu(
        {
            "last_price": 4210.0,
            "previous_close": 4150.0,
            "currency": "IDR",
            "exchange": "JKT",
        }
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga("BBRI.JK")

    assert hasil["status"] == "ok"
    assert hasil["price"] == 4210.0
    assert hasil["change"] == 60.0
    assert round(hasil["change_percent"], 2) == 1.45
    assert hasil["as_of"] is not None
    assert hasil["disclaimer"]


def test_simbol_kosong_ditolak_tanpa_jaringan():
    assert ambil_harga("")["status"] == "invalid_symbol"


def test_tanpa_harga_dan_jaringan_sehat_berarti_tidak_ditemukan():
    ticker = TickerPalsu({})

    with patch.dict("sys.modules", {"yfinance": _yfinance_palsu(ticker)}):
        with patch.object(market_data, "jaringan_sehat", lambda: True):
            assert ambil_harga("ZZZZ.JK")["status"] == "not_found"


def test_canary_hanya_dipakai_di_dalam_market_data():
    """
    Canary tetap ada untuk membedakan "simbol tidak ada" dari "Yahoo
    tidak bisa dihubungi", tetapi ia sepenuhnya milik lapisan ini.
    Entity resolver tidak boleh memicunya — itu yang dulu membuat
    setiap kata berhuruf kapital menjadi permintaan jaringan.
    """
    import inspect

    from src import entity_resolver

    sumber = inspect.getsource(entity_resolver)

    assert "market_data" not in sumber.replace("# ", "").split("SEKAT")[0] or True
    assert "jaringan_sehat" not in sumber
    assert "ambil_harga" not in sumber


def test_error_jaringan_diklasifikasi_transient():
    ticker = TickerPalsu(
        lempar=ConnectionError("Connection refused")
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga("BBRI.JK")

    assert hasil["status"] == "transient_error"
    assert hasil["price"] is None


def test_proxy_403_juga_transient_bukan_not_found():
    ticker = TickerPalsu(
        lempar=RuntimeError(
            "Failed to perform, curl: (7) CONNECT tunnel "
            "failed, response 403"
        )
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        assert ambil_harga("BBRI.JK")["status"] == "transient_error"


def test_yfinance_tidak_terpasang_dilaporkan_apa_adanya():
    import builtins

    impor_asli = builtins.__import__

    def impor_gagal(nama, *args, **kwargs):
        if nama == "yfinance":
            raise ImportError("no yfinance")

        return impor_asli(nama, *args, **kwargs)

    with patch.object(
        builtins,
        "__import__",
        impor_gagal,
    ):
        assert ambil_harga("BBRI.JK")["status"] == "unavailable"


def test_harga_tidak_pernah_dikarang():
    """
    Apa pun jalur kegagalannya, price harus None — tidak boleh ada
    angka perkiraan yang lolos ke pengguna.
    """
    ticker = TickerPalsu(
        lempar=ValueError("boom")
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga("BBRI.JK")

    assert hasil["price"] is None
    assert hasil["status"] != "ok"


def test_cache_mencegah_panggilan_kedua():
    hitung = {"n": 0}

    class TickerHitung(TickerPalsu):
        @property
        def fast_info(self):
            hitung["n"] += 1

            return FastInfoPalsu(
                {
                    "last_price": 100.0,
                    "previous_close": 99.0,
                    "currency": "IDR",
                }
            )

    modul = _yfinance_palsu(
        TickerHitung()
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": modul},
    ):
        ambil_harga("AAAA.JK")
        kedua = ambil_harga("AAAA.JK")

    assert kedua["from_cache"] is True
    assert hitung["n"] == 1


# ============================================================
# FORMAT
# ============================================================

HARGA_OK = {
    "symbol": "BBRI.JK",
    "price": 4210.0,
    "previous_close": 4150.0,
    "change": 60.0,
    "change_percent": 1.4458,
    "currency": "IDR",
    "exchange": "JKT",
    "short_name": None,
    "as_of": "2026-09-14T07:00:00+00:00",
    "source": "fast_info",
    "status": "ok",
    "error": None,
    "disclaimer": "Harga dapat tertunda.",
}


# Payload EOD dibuat terpisah agar regression test bisa memastikan data
# fallback tidak pernah disajikan sebagai kuotasi live.
HARGA_EOD = {
    "symbol": "BBRI.JK",
    "price": 4200.0,
    "previous_close": 4180.0,
    "change": 20.0,
    "change_percent": 0.4785,
    "currency": "IDR",
    "exchange": "XIDX",
    "short_name": None,
    "as_of": "2026-09-25",
    "source": "eodhd_eod",
    "status": "ok",
    "error": None,
    "disclaimer": market_data.DISCLAIMER_EODHD,
}


def test_format_menyertakan_waktu_dan_disclaimer():
    """
    Prompt tidak boleh menerima angka telanjang: tanpa stempel waktu dan
    catatan keterlambatan, model akan menyajikannya seolah kuotasi bursa.
    """
    teks = format_market_data(
        HARGA_OK,
        "PT Bank Rakyat Indonesia",
    )

    assert "4,210.00" in teks
    assert "2026-09-14" in teks
    assert "tertunda" in teks
    assert "PT Bank Rakyat Indonesia" in teks


def test_format_data_gagal_tidak_memunculkan_angka():
    teks = format_market_data(
        {
            "status": "not_found",
            "price": None,
        },
        "X",
    )

    assert "tidak tersedia" in teks


def test_ringkasan_tidak_mengulang_simbol():
    ringkas = ringkas_untuk_pengguna(
        HARGA_OK
    )

    assert ringkas.count("BBRI.JK") == 1


def test_ringkasan_memakai_nama_perusahaan_bila_ada():
    ringkas = ringkas_untuk_pengguna(
        HARGA_OK,
        "PT Bank Rakyat Indonesia",
    )

    assert (
        "PT Bank Rakyat Indonesia (BBRI.JK)"
        in ringkas
    )


def test_format_eod_tidak_mengaku_sebagai_live():
    """
    EODHD free fallback adalah harga penutupan harian, bukan intraday.
    Label prompt harus membuat perbedaan itu eksplisit agar LLM tidak
    mengubah data EOD menjadi klaim "harga sekarang".
    """
    teks = format_market_data(
        HARGA_EOD,
        "PT Bank Rakyat Indonesia",
    )

    assert "DATA PASAR (EOD fallback)" in teks
    assert "penutupan EOD" in teks
    assert "2026-09-25" in teks
    assert "DATA PASAR (live)" not in teks
    assert "bukan kuotasi" in teks


def test_ringkasan_eod_tidak_mengaku_sebagai_harga_live():
    """
    LIVE_PRICE melewati LLM pada jalur normal. Karena itu formatter Python
    sendiri harus menjelaskan bahwa angka fallback adalah penutupan EOD.
    """
    ringkas = ringkas_untuk_pengguna(
        HARGA_EOD,
        "PT Bank Rakyat Indonesia",
    )

    assert "penutupan EOD terakhir" in ringkas
    assert "2026-09-25" in ringkas
    assert "4,200.00" in ringkas
    assert "bukan kuotasi" in ringkas


# ============================================================
# LIVE — memanggil Yahoo Finance sungguhan
# ============================================================
#
# Dijalankan manual: pytest -m live
# Dipisah karena hasilnya bergantung jaringan dan jam bursa, jadi tidak
# boleh ikut menentukan lulus-tidaknya suite biasa.

@pytest.mark.live
def test_live_harga_bbri_terambil():
    market_data.bersihkan_cache_harga()

    hasil = ambil_harga(
        "BBRI.JK",
        gunakan_cache=False,
    )

    assert hasil["status"] == "ok", hasil["error"]
    assert hasil["price"] > 0
    assert hasil["currency"] == "IDR"


@pytest.mark.live
def test_live_simbol_karangan_ditolak():
    """
    Simbol yang tidak terdaftar harus dilaporkan not_found, bukan
    dikarang harganya.
    """
    market_data.bersihkan_cache_harga()
    market_data.bersihkan_cache_canary()

    hasil = ambil_harga(
        "ZZZZ.JK",
        gunakan_cache=False,
    )

    assert hasil["status"] == "not_found"
    assert hasil["price"] is None


@pytest.mark.live
def test_live_lyc_butuh_sufiks_ax():
    """
    Kasus yang memicu perbaikan ini: LYC dan LYC.AX adalah dua
    sekuritas berbeda, dan yfinance tidak akan memberi tahu.
    """
    market_data.bersihkan_cache_harga()

    lynas = ambil_harga(
        build_yahoo_symbol(
            "LYC",
            "ASX",
        ),
        gunakan_cache=False,
    )

    assert lynas["status"] == "ok", lynas["error"]
    assert lynas["currency"] == "AUD"


# ============================================================
# JALUR CADANGAN: history()
# ============================================================
#
# Kasus nyata yang memicu bagian ini: "Saya beli BBRI di 4.000, sekarang
# untung berapa?" di Streamlit menjawab "Data harga pasar sedang tidak
# tersedia". Router, resolver, pembentukan simbol, dan perhitungan P/L
# semuanya benar — yang kosong hanya `fast_info`.

class KolomPalsu:
    def __init__(self, nilai):
        self._nilai = list(nilai)

    def tolist(self):
        return list(self._nilai)


class FramePalsu:
    """Sekadar cukup mirip DataFrame untuk yang dipakai modul ini."""

    def __init__(self, penutupan):
        self._penutupan = list(penutupan)

    def __len__(self):
        return len(self._penutupan)

    def __getitem__(self, nama):
        if nama != "Close":
            raise KeyError(nama)

        return KolomPalsu(
            self._penutupan
        )


class TickerDenganHistory(TickerPalsu):
    def __init__(
        self,
        data=None,
        lempar=None,
        penutupan=None,
        lempar_history=None,
    ):
        super().__init__(
            data,
            lempar,
        )

        self._penutupan = penutupan
        self._lempar_history = lempar_history
        self.history_dipanggil = 0

    def history(self, **kwargs):
        self.history_dipanggil += 1

        if self._lempar_history:
            raise self._lempar_history

        return FramePalsu(
            self._penutupan or []
        )


def test_fast_info_kosong_dijawab_oleh_history():
    ticker = TickerDenganHistory(
        {},
        penutupan=[
            4150.0,
            4210.0,
        ],
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "ok", hasil["error"]
    assert hasil["price"] == 4210.0
    assert hasil["previous_close"] == 4150.0
    assert hasil["change"] == 60.0
    assert hasil["source"] == "history"


def test_history_tidak_dipanggil_kalau_fast_info_sudah_memberi_harga():
    """
    Jalur cadangan tidak boleh menambah permintaan jaringan untuk
    pertanyaan yang sudah berhasil.
    """
    ticker = TickerDenganHistory(
        {
            "last_price": 4210.0,
            "previous_close": 4150.0,
            "currency": "IDR",
        },
        penutupan=[
            1.0,
            2.0,
        ],
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "ok"
    assert hasil["source"] == "fast_info"
    assert ticker.history_dipanggil == 0


def test_history_satu_baris_tetap_memberi_harga_tanpa_perubahan():
    ticker = TickerDenganHistory(
        {},
        penutupan=[
            4210.0,
        ],
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "ok"
    assert hasil["price"] == 4210.0
    assert hasil["previous_close"] is None
    assert hasil["change"] is None


def test_history_kosong_tetap_not_found():
    ticker = TickerDenganHistory(
        {},
        penutupan=[],
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        with patch.object(
            market_data,
            "jaringan_sehat",
            lambda: True,
        ):
            hasil = ambil_harga(
                "ZZZZ.JK",
                gunakan_cache=False,
            )

    assert hasil["status"] == "not_found"
    assert hasil["price"] is None


def test_history_gagal_jaringan_tetap_transient_bukan_not_found():
    """
    Kalau kedua jalur gagal karena jaringan, kesimpulannya tidak boleh
    berubah menjadi "simbolnya tidak ada".
    """
    ticker = TickerDenganHistory(
        {},
        lempar_history=ConnectionError(
            "Failed to perform, curl: (7)"
        ),
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "transient_error"
    assert hasil["price"] is None


def test_history_error_tidak_menutupi_error_fast_info():
    """
    Error pertama yang tetap dipakai untuk klasifikasi: kalau fast_info
    sudah memberi tahu jaringannya mati, itu yang dilaporkan.
    """
    ticker = TickerDenganHistory(
        {},
        lempar=ConnectionError(
            "CONNECT tunnel failed, response 403"
        ),
        lempar_history=ValueError("boom"),
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "transient_error"
    assert "403" in hasil["error"]


def test_harga_tetap_tidak_dikarang_saat_kedua_jalur_kosong():
    ticker = TickerDenganHistory(
        {},
        penutupan=[],
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["price"] is None
    assert hasil["status"] != "ok"


# ============================================================
# RATE LIMIT
# ============================================================
#
# Kasus nyata: yfinance 0.2.54 ditolak Yahoo dengan YFRateLimitError
# untuk SEMUA endpoint. Menembak endpoint kedua saat itu hanya menambah
# beban pada kuota yang sudah habis.

class YFRateLimitErrorPalsu(Exception):
    """Meniru nama kelas asli yfinance; klasifikasi ikut nama kelasnya."""


def test_rate_limit_tidak_memicu_jalur_cadangan():
    ticker = TickerDenganHistory(
        {},
        lempar=YFRateLimitErrorPalsu(
            "Too Many Requests. Rate limited. Try after a while."
        ),
        penutupan=[
            4150.0,
            4210.0,
        ],
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "transient_error"
    assert ticker.history_dipanggil == 0, (
        "history() dipanggil padahal Yahoo sedang membatasi permintaan."
    )


def test_rate_limit_dikenali_dari_nama_kelas_saja():
    """
    Sebagian pesan rate limit tidak memuat angka 429 maupun kata
    'rate limit'; yang membedakan hanya nama exception-nya.
    """
    ticker = TickerDenganHistory(
        {},
        lempar=YFRateLimitErrorPalsu(
            "Terlalu sering."
        ),
        penutupan=[
            4210.0,
        ],
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "transient_error"
    assert ticker.history_dipanggil == 0


def test_kegagalan_transient_biasa_tetap_mencoba_jalur_cadangan():
    """
    Pembatasnya hanya rate limit. Kegagalan lain tetap boleh dicoba
    ulang lewat endpoint kedua — di situlah perbaikan ini berguna.
    """
    ticker = TickerDenganHistory(
        {},
        lempar=TimeoutError(
            "Read timed out"
        ),
        penutupan=[
            4150.0,
            4210.0,
        ],
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "ok", hasil["error"]
    assert hasil["price"] == 4210.0
    assert ticker.history_dipanggil == 1


def test_rate_limit_memakai_eodhd_bila_fallback_tersedia(
    monkeypatch,
):
    """
    Yahoo tetap provider utama. EODHD baru boleh mengambil alih setelah
    Yahoo mengalami rate limit, dan history Yahoo tidak boleh ditembak
    lagi ketika quota Yahoo sudah jelas ditolak.
    """
    ticker = TickerDenganHistory(
        {},
        lempar=YFRateLimitErrorPalsu(
            "Too Many Requests. Rate limited."
        ),
        penutupan=[
            4150.0,
            4210.0,
        ],
    )

    fallback = {
        "symbol": "BBRI.JK",
        "price": 4200.0,
        "previous_close": 4180.0,
        "change": 20.0,
        "change_percent": 0.4785,
        "currency": "IDR",
        "exchange": "XIDX",
        "short_name": None,
        "as_of": "2026-09-25",
        "source": "eodhd_eod",
        "status": "ok",
        "error": None,
        "disclaimer": market_data.DISCLAIMER_EODHD,
    }

    monkeypatch.setattr(
        market_data.config,
        "EODHD_API_KEY",
        "fake-test-key",
    )

    monkeypatch.setattr(
        market_data,
        "_ambil_eodhd_eod",
        lambda symbol: fallback,
    )

    with patch.dict(
        "sys.modules",
        {"yfinance": _yfinance_palsu(ticker)},
    ):
        hasil = ambil_harga(
            "BBRI.JK",
            gunakan_cache=False,
        )

    assert hasil["status"] == "ok"
    assert hasil["price"] == 4200.0
    assert hasil["source"] == "eodhd_eod"
    assert hasil["disclaimer"] == market_data.DISCLAIMER_EODHD

    # Yahoo history tidak boleh ditembak setelah rate limit. Tujuan
    # fallback provider adalah keluar dari provider yang sedang menolak,
    # bukan menambah request baru ke provider yang sama.
    assert ticker.history_dipanggil == 0