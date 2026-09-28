"""
Anggaran jaringan: membuktikan Yahoo Finance tidak dipanggil berlebihan.

Berbeda dari test lain, yang diukur di sini bukan kebenaran jawaban
melainkan BIAYA-nya. Klaim "arsitektur baru lebih efisien" tidak ada
artinya tanpa angka, dan angka itu gampang meleset diam-diam ketika
seseorang menambahkan satu pemanggilan validasi di tempat yang salah.

Penghitungnya dipasang di titik paling dalam — tepat sebelum yfinance
disentuh — sehingga cache, jalur cepat, maupun jalur semantik semuanya
terukur oleh ukuran yang sama.
"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src import assistant, market_data


DOKUMEN = [
    Document(
        page_content="Target TP 1 BBRI berada di Rp3.400.",
        metadata={"source": "ihsg.pdf", "page": 12, "chunk_id": "a1"},
    )
]


class RetrieverPalsu:
    def __init__(self, documents=None):
        self._documents = DOKUMEN if documents is None else documents

    def invoke(self, query):
        return self._documents


class LLMPalsu:
    def __init__(self, isi="Penjelasan."):
        self.isi = isi
        self.dipanggil = 0

    def invoke(self, prompt):
        self.dipanggil += 1

        class R:
            content = self.isi

        return R()


class TickerPalsu:
    def __init__(self, data):
        self._data = data

    @property
    def fast_info(self):
        return dict(self._data)


def yfinance_palsu(pencatat, data=None):
    """
    Modul yfinance tiruan yang mencatat setiap simbol yang diminta.
    """
    data = data or {
        "last_price": 4210.0,
        "previous_close": 4150.0,
        "currency": "IDR",
        "exchange": "JKT",
    }

    class Modul:
        @staticmethod
        def Ticker(symbol):
            pencatat.append(symbol)
            return TickerPalsu(data)

    return Modul


@pytest.fixture
def hitung_request():
    """
    Jalankan sesuatu, lalu periksa berapa simbol yang benar-benar
    diminta ke yfinance.
    """
    pencatat = []

    market_data.bersihkan_cache_harga()
    market_data.bersihkan_cache_canary()
    market_data.reset_statistik_market_data()

    with patch.dict("sys.modules", {"yfinance": yfinance_palsu(pencatat)}):
        yield pencatat

    market_data.bersihkan_cache_harga()
    market_data.bersihkan_cache_canary()


# ============================================================
# NOL PERMINTAAN
# ============================================================

@pytest.mark.parametrize(
    "question",
    [
        "Bagaimana prospek BBRI?",
        "Berapa target harga BBRI menurut riset?",
        "Apa katalis utama saham BUMI?",
        "Bagaimana prospek IHSG pada 2026?",
        "Apa risiko terbesar di sektor perbankan menurut laporan?",
        "Berapa rasio CASA bank itu menurut riset?",
        "Apa YANG dimaksud dengan RISK di laporan itu?",
    ],
)
def test_pertanyaan_rag_nol_request_yahoo(hitung_request, question):
    """
    Inilah perbaikan terbesar refactor ini.

    Sebelumnya entity resolver memvalidasi setiap kandidat ke Yahoo,
    jadi pertanyaan riset murni pun memicu permintaan jaringan untuk
    setiap kata berhuruf kapital di dalamnya.
    """
    with patch.object(assistant, "_jalur_rag") as rag:
        rag.return_value = {"answer": "x", "status": "ok"}
        assistant.jawab(question, izinkan_llm_router=False)

    assert hitung_request == [], question


@pytest.mark.parametrize(
    "question",
    [
        "Cuaca di Medan hari ini gimana?",
        "Siapa presiden Indonesia sekarang?",
        "Resep nasi goreng dong",
    ],
)
def test_out_of_scope_nol_request_yahoo(hitung_request, question):
    assistant.jawab(question)

    assert hitung_request == []


def test_ambiguitas_nol_request_yahoo(hitung_request):
    """Yahoo tidak boleh disentuh sebelum entitas pasti."""
    hasil = assistant.jawab("BBRI dan BMRI mana yang lebih murah sekarang?")

    assert hasil["status"] == "ambiguous"
    assert hitung_request == []


def test_entitas_tak_didukung_nol_request_yahoo(hitung_request):
    llm = LLMPalsu(
        '{"company": "", "ticker": "", "market": "UNKNOWN", '
        '"confidence": "low", "alternatives": []}'
    )

    hasil = assistant.jawab(
        "Berapa harga emas dunia hari ini?",
        llm=llm,
        retriever=RetrieverPalsu(),
    )

    assert hasil["status"] == "unresolved_entity"
    assert hitung_request == []


def test_pertanyaan_kosong_nol_request_yahoo(hitung_request):
    assistant.jawab("")

    assert hitung_request == []


def test_import_modul_tidak_memicu_request(hitung_request):
    """
    Import tidak boleh punya efek samping jaringan: Streamlit mengimpor
    ulang pada setiap rerun.
    """
    import importlib

    for nama in (
        "src.entity_resolver",
        "src.router",
        "src.assistant",
        "src.market_symbols",
        "src.session_context",
        "src.profit_loss",
    ):
        importlib.reload(importlib.import_module(nama))

    assert hitung_request == []


# ============================================================
# TEPAT SATU PERMINTAAN
# ============================================================

def test_live_price_tepat_satu_request(hitung_request):
    assistant.jawab("Harga BBRI sekarang?")

    assert hitung_request == ["BBRI.JK"]


def test_live_compare_tepat_satu_request(hitung_request):
    assistant.jawab(
        "BBRI sudah mencapai target belum?",
        llm=LLMPalsu(),
        retriever=RetrieverPalsu(),
    )

    assert hitung_request == ["BBRI.JK"]


def test_konteks_sesi_tetap_mengambil_harga_baru(hitung_request):
    """
    Memori percakapan menyimpan identitas, bukan harga — jadi rujukan
    "saham itu" tetap memicu pengambilan harga.
    """
    assistant.jawab(
        "Kalau saham itu sekarang berapa?",
        session_context_data={
            "ticker": "BBRI",
            "company": None,
            "market": "IDX",
        },
    )

    assert hitung_request == ["BBRI.JK"]


def test_pertanyaan_berulang_memakai_cache(hitung_request):
    """
    Dua pertanyaan sama berturut-turut hanya boleh menembak jaringan
    sekali, selama TTL cache belum lewat.
    """
    assistant.jawab("Harga BBRI sekarang?")
    assistant.jawab("Harga BBRI sekarang?")

    assert hitung_request == ["BBRI.JK"]

    statistik = market_data.statistik_market_data()

    assert statistik["cache_hits"] >= 1


# ============================================================
# JEJAK BIAYA
# ============================================================

def test_statistik_market_data_terhitung(hitung_request):
    market_data.reset_statistik_market_data()

    assistant.jawab("Harga BBRI sekarang?")

    assert market_data.statistik_market_data()["requests"] == 1


def test_jalur_cepat_nol_llm(hitung_request):
    llm = LLMPalsu()

    hasil = assistant.jawab("Harga BBRI sekarang?", llm=llm)

    assert llm.dipanggil == 0
    assert hasil["trace"]["router_llm_calls"] == 0
    assert hasil["trace"]["resolution_llm_calls"] == 0
