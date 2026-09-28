"""
Test perhitungan untung/rugi.

Fungsi murni, tanpa jaringan dan tanpa LLM — dan itulah intinya: angka
finansial tidak boleh bergantung pada model yang bisa salah hitung
dengan percaya diri.
"""

import pytest

from src.profit_loss import (
    POSISI_BREAKEVEN,
    POSISI_LOSS,
    POSISI_PROFIT,
    ekstrak_harga_entry,
    format_profit_loss,
    hitung_profit_loss,
    parse_angka_indonesia,
    ringkas_profit_loss,
)


# ============================================================
# PARSER ANGKA
# ============================================================

@pytest.mark.parametrize(
    "teks,harapan",
    [
        ("4.000", 4000.0),
        ("4,000", 4000.0),
        ("4.210,50", 4210.50),
        ("4210.50", 4210.50),
        ("1.234.567", 1234567.0),
        ("210", 210.0),
        ("13,75", 13.75),
    ],
)
def test_parse_angka_indonesia(teks, harapan):
    assert parse_angka_indonesia(teks) == harapan


def test_angka_nol_dan_kosong_ditolak():
    assert parse_angka_indonesia("0") is None
    assert parse_angka_indonesia("") is None
    assert parse_angka_indonesia("bukan angka") is None


# ============================================================
# PARSER HARGA ENTRY
# ============================================================

@pytest.mark.parametrize(
    "pertanyaan",
    [
        "Saya beli BBRI di 4.000",
        "Saya beli BBRI di Rp4.000",
        "Saya beli BBRI pada harga Rp4.000",
        "entry BBRI di 4.000",
        "Saya masuk BBRI di 4.000",
        "Modal saya di BBRI 4.000",
        "Kalau saya beli BBRI di 4.000, sekarang untung berapa?",
        "nyangkut di BBRI 4.000",
        "average saya BBRI 4.000",
    ],
)
def test_variasi_kalimat_harga_beli(pertanyaan):
    """
    Bug yang ditemukan saat audit: pola lama menuntut angka berada
    tepat setelah kata kerja, sehingga bentuk paling umum —
    "beli BBRI di 4.000", dengan kode saham di tengah — selalu gagal.
    """
    assert ekstrak_harga_entry(pertanyaan) == 4000.0, pertanyaan


@pytest.mark.parametrize(
    "pertanyaan",
    [
        "Target BBRI 5.000, sekarang bagaimana?",
        "Berapa stop loss BBRI menurut riset?",
        "Harga BBRI sekarang berapa?",
        "BBRI sudah mencapai target belum?",
        "Apa katalis utama BUMI?",
    ],
)
def test_angka_yang_bukan_harga_beli_diabaikan(pertanyaan):
    """
    Angka target milik riset, bukan posisi pengguna. Menghitung untung
    dari angka target orang lain menghasilkan angka yang terlihat
    masuk akal dan sepenuhnya salah.
    """
    assert ekstrak_harga_entry(pertanyaan) is None, pertanyaan


def test_desimal_pada_harga_beli():
    assert ekstrak_harga_entry("saya beli di 4.210,50") == 4210.50


# ============================================================
# PERHITUNGAN
# ============================================================

def test_profit():
    hasil = hitung_profit_loss(4000, 4210)

    assert hasil["difference"] == 210.0
    assert round(hasil["return_percent"], 2) == 5.25
    assert hasil["position"] == POSISI_PROFIT


def test_loss():
    hasil = hitung_profit_loss(4000, 3800)

    assert hasil["difference"] == -200.0
    assert round(hasil["return_percent"], 2) == -5.0
    assert hasil["position"] == POSISI_LOSS


def test_breakeven():
    hasil = hitung_profit_loss(4000, 4000)

    assert hasil["difference"] == 0.0
    assert hasil["return_percent"] == 0.0
    assert hasil["position"] == POSISI_BREAKEVEN


def test_tanpa_salah_satu_harga_tidak_menghitung():
    assert hitung_profit_loss(None, 4210) is None
    assert hitung_profit_loss(4000, None) is None
    assert hitung_profit_loss(None, None) is None


def test_entry_nol_atau_negatif_ditolak():
    """Pembagian dengan nol tidak boleh menjadi persentase tak hingga."""
    assert hitung_profit_loss(0, 4210) is None
    assert hitung_profit_loss(-100, 4210) is None


def test_input_bukan_angka_tidak_melempar_exception():
    assert hitung_profit_loss("bukan angka", 4210) is None


# ============================================================
# PENYAJIAN
# ============================================================

def test_format_menyatakan_siapa_yang_menghitung():
    """
    Blok ini masuk ke prompt. Menyebut bahwa angkanya sudah dihitung
    adalah bagian dari instruksi supaya model tidak menghitung ulang.
    """
    teks = format_profit_loss(hitung_profit_loss(4000, 4210))

    assert "dihitung Python" in teks
    assert "5.25%" in teks
    assert "4,000.00" in teks
    assert "4,210.00" in teks


def test_format_kosong_saat_tidak_ada_perhitungan():
    assert format_profit_loss(None) == ""


def test_ringkasan_untuk_pengguna():
    ringkas = ringkas_profit_loss(
        hitung_profit_loss(4000, 4210), "BBRI"
    )

    assert "BBRI" in ringkas
    assert "untung" in ringkas
    assert "+5.25%" in ringkas


def test_ringkasan_rugi_memakai_kata_rugi():
    ringkas = ringkas_profit_loss(hitung_profit_loss(4000, 3800), "BBRI")

    assert "rugi" in ringkas
    assert "-5.00%" in ringkas


def test_ringkasan_impas():
    ringkas = ringkas_profit_loss(hitung_profit_loss(4000, 4000), "BBRI")

    assert "impas" in ringkas
