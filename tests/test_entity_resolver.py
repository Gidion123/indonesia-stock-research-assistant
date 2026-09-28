"""
Test entity resolver.

Tanpa jaringan, tanpa database, tanpa LLM sungguhan. Yang paling
penting diuji di sini: resolver TIDAK BOLEH menyentuh Yahoo Finance
sama sekali, pada jalur mana pun.
"""

import ast
import inspect

from langchain_core.documents import Document

from src import entity_resolver
from src.entity_resolver import (
    LEVEL_EKSPLISIT,
    LEVEL_SEMANTIK,
    LEVEL_SESI,
    STATUS_AMBIGUOUS,
    STATUS_NOT_FOUND,
    STATUS_RESOLVED,
    detect_explicit_tickers,
    merujuk_percakapan_sebelumnya,
    normalize_ticker,
    resolve_semantic,
    resolve_research_alias,
    resolve_ticker,
)


class RetrieverPalsu:
    def __init__(self, documents):
        self._documents = documents
        self.dipanggil = 0

    def invoke(self, query):
        self.dipanggil += 1
        return self._documents


class LLMPalsu:
    def __init__(self, isi):
        self.isi = isi
        self.dipanggil = 0

    def invoke(self, prompt):
        self.dipanggil += 1

        class R:
            content = self.isi

        return R()


DOKUMEN_BBRI = [
    Document(
        page_content=(
            "PT Bank Rakyat Indonesia (Persero) Tbk (BBRI) menjadi "
            "pilihan utama dengan target TP 1 di Rp3.400."
        ),
        metadata={"source": "ihsg.pdf", "page": 12, "chunk_id": "a1"},
    )
]


# ============================================================
# BATAS ARSITEKTUR — TIDAK ADA YAHOO DI SINI
# ============================================================

def test_resolver_tidak_menyentuh_yahoo_finance():
    """
    Ini kriteria arsitektur, bukan detail implementasi.

    Versi sebelumnya memvalidasi setiap kandidat ke Yahoo Finance.
    Akibatnya "Apa YANG dimaksud RISK di laporan itu?" — pertanyaan RAG
    murni — tetap memicu permintaan jaringan untuk setiap kata kapital.
    """
    # Diperiksa lewat AST, bukan pencarian teks: komentar yang
    # MENJELASKAN kenapa Yahoo tidak dipanggil tidak boleh membuat test
    # ini gagal, dan sebaliknya nama yang tersembunyi di dalam kode
    # tidak boleh lolos hanya karena tidak tertulis apa adanya.
    pohon = ast.parse(inspect.getsource(entity_resolver))

    identifier = set()

    for node in ast.walk(pohon):
        if isinstance(node, ast.Name):
            identifier.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifier.add(node.attr)
        elif isinstance(node, ast.alias):
            identifier.add(node.name.split(".")[-1])
            if node.asname:
                identifier.add(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            identifier.update(node.module.split("."))

    for terlarang in (
        "yfinance",
        "market_data",
        "ambil_harga",
        "periksa_simbol",
        "jaringan_sehat",
        "symbol_exists",
    ):
        assert terlarang not in identifier, terlarang


def test_resolver_tidak_mengimpor_market_data():
    """
    Dependency harus satu arah: resolver -> identitas, market_data ->
    Yahoo. Bukan resolver -> market_data -> Yahoo.
    """
    pohon = ast.parse(inspect.getsource(entity_resolver))

    diimpor = set()

    for node in ast.walk(pohon):
        if isinstance(node, ast.ImportFrom) and node.module:
            diimpor.add(node.module)
        elif isinstance(node, ast.Import):
            diimpor.update(a.name for a in node.names)

    assert "src.market_data" not in diimpor
    assert "yfinance" not in diimpor


def test_resolver_tidak_mengembalikan_yahoo_symbol():
    """
    Pembentukan simbol adalah tanggung jawab market_symbols/market_data.
    Resolver hanya menyerahkan ticker + market.
    """
    hasil = resolve_ticker("Harga BBRI sekarang?", izinkan_semantik=False)

    assert "yahoo_symbol" not in hasil
    assert hasil["ticker"] == "BBRI"
    assert hasil["market"] == "IDX"


# ============================================================
# DETEKSI KANDIDAT
# ============================================================

def test_kode_huruf_besar_terdeteksi():
    assert detect_explicit_tickers("Harga BBRI sekarang?") == ["BBRI"]


def test_kode_huruf_kecil_terdeteksi():
    """
    Orang mengetik apa adanya di chat. Kalau hanya huruf besar yang
    dikenali, hampir semua pertanyaan nyata meleset ke jalur semantik
    yang jauh lebih mahal.
    """
    assert detect_explicit_tickers("harga bbri sekarang") == ["BBRI"]


def test_sufiks_jk_dan_dollar_dikenali():
    assert detect_explicit_tickers("cek BBRI.JK dong") == ["BBRI"]
    assert detect_explicit_tickers("cek $MP") == ["MP"]


def test_kata_umum_tidak_menjadi_kandidat():
    assert detect_explicit_tickers(
        "Apa YANG dimaksud RISK dan BASE di laporan itu?"
    ) == []


def test_pertanyaan_tanpa_saham_tidak_menghasilkan_kandidat():
    assert detect_explicit_tickers("mau beli saham lagi nih gimana") == []
    assert detect_explicit_tickers("harga emas dunia hari ini") == []


def test_normalize_ticker():
    assert normalize_ticker("bbri") == "BBRI"
    assert normalize_ticker("BBRI.JK") == "BBRI"
    assert normalize_ticker("LYC.AX") == "LYC"
    assert normalize_ticker("") is None
    assert normalize_ticker("TERLALUPANJANG") is None


def test_deteksi_rujukan_percakapan():
    assert merujuk_percakapan_sebelumnya("Harga bank tersebut sekarang?")
    assert merujuk_percakapan_sebelumnya("saham itu gimana?")
    assert not merujuk_percakapan_sebelumnya("Berapa harga BBRI?")


# ============================================================
# TINGKAT 1 — EKSPLISIT (DETERMINISTIK)
# ============================================================

def test_ticker_eksplisit_selesai_tanpa_llm_dan_tanpa_retrieval():
    """
    Inti desain: kode eksplisit tidak boleh menyentuh PgVector, LLM,
    maupun Yahoo.
    """
    llm = LLMPalsu("{}")
    retriever = RetrieverPalsu(DOKUMEN_BBRI)

    hasil = resolve_ticker(
        "Harga BBRI sekarang?", llm=llm, retriever=retriever
    )

    assert hasil["status"] == STATUS_RESOLVED
    assert hasil["level"] == LEVEL_EKSPLISIT
    assert hasil["ticker"] == "BBRI"
    assert hasil["market"] == "IDX"
    assert hasil["llm_calls"] == 0
    assert llm.dipanggil == 0
    assert retriever.dipanggil == 0


def test_bentuk_empat_huruf_dianggap_idx():
    """
    Konvensi bentuk, bukan daftar emiten: emiten IDX baru otomatis
    terlayani tanpa perlu didaftarkan di mana pun.
    """
    for kode in ("BBRI", "BMRI", "ANTM", "BREN"):
        hasil = resolve_ticker(f"Harga {kode} sekarang?", izinkan_semantik=False)

        assert hasil["market"] == "IDX", kode
        assert hasil["ticker"] == kode


def test_kode_non_idx_diperlakukan_sebagai_saham_amerika():
    hasil = resolve_ticker("Harga $MP sekarang?", izinkan_semantik=False)

    assert hasil["status"] == STATUS_RESOLVED
    assert hasil["ticker"] == "MP"
    assert hasil["market"] == "NYSE"


def test_casa_tidak_disaring_daftar_hitam():
    """
    CASA berarti rasio dana murah DI DOKUMEN, tetapi juga kode emiten
    IDX yang sah. Router yang memisahkan keduanya: "rasio CASA menurut
    riset" adalah RAG dan tidak pernah sampai ke sini.
    """
    hasil = resolve_ticker("berapa harga casa hari ini", izinkan_semantik=False)

    assert hasil["status"] == STATUS_RESOLVED
    assert hasil["ticker"] == "CASA"


def test_explicit_comparison_resolves_each_ticker_independently():
    hasil = entity_resolver.resolve_explicit_many(
        "Bandingkan harga BBRI dan BMRI sekarang."
    )
    assert [item["ticker"] for item in hasil] == ["BBRI", "BMRI"]
    assert all(item["status"] == STATUS_RESOLVED for item in hasil)
    assert all(item["market"] == "IDX" for item in hasil)
    assert all(item["llm_calls"] == 0 for item in hasil)


def test_multiple_codes_without_comparison_keep_single_entity_ambiguity():
    hasil = resolve_ticker("Harga BBRI atau BMRI sekarang?", izinkan_semantik=False)
    assert hasil["status"] == STATUS_AMBIGUOUS
    assert set(hasil["candidates"]) == {"BBRI", "BMRI"}
    assert hasil["ticker"] is None


def test_explicit_many_deduplicates_and_handles_marked_short_ticker():
    result = entity_resolver.resolve_explicit_many("Bandingkan $MP vs BBRI dan BBRI")
    assert [item["ticker"] for item in result] == ["MP", "BBRI"]
    assert [item["market"] for item in result] == ["NYSE", "IDX"]
    assert entity_resolver.resolve_explicit_many("Harga BBRI?") == []


# ============================================================
# TINGKAT 2 — SESI
# ============================================================

KONTEKS_BBRI = {
    "ticker": "BBRI",
    "company": "PT Bank Rakyat Indonesia (Persero) Tbk",
    "market": "IDX",
}


def test_rujukan_percakapan_diselesaikan_dari_konteks_sesi():
    llm = LLMPalsu("{}")

    hasil = resolve_ticker(
        "Kalau saham itu sekarang berapa?",
        session_context=KONTEKS_BBRI,
        llm=llm,
        izinkan_semantik=False,
    )

    assert hasil["status"] == STATUS_RESOLVED
    assert hasil["level"] == LEVEL_SESI
    assert hasil["ticker"] == "BBRI"
    assert hasil["llm_calls"] == 0
    assert llm.dipanggil == 0


def test_lanjutan_powr_dengan_akhiran_nya_dan_subjek_tersembunyi():
    context = {"ticker": "POWR", "company": None, "market": "IDX"}
    for question in (
        "Berapa harganya sekarang?",
        "Apa target harga?",
        "Bagaimana laba?",
        "Bagaimana valuasi?",
        "Apakah sekarang sudah mencapai target?",
    ):
        assert merujuk_percakapan_sebelumnya(question)
        result = resolve_ticker(
            question, session_context=context, izinkan_semantik=False
        )
        assert result["ticker"] == "POWR"
        assert result["level"] == LEVEL_SESI

    # LABA is also an actual issuer code when written explicitly.
    assert resolve_ticker(
        "Prospek LABA?", session_context=context, izinkan_semantik=False
    )["ticker"] == "LABA"


def test_pertanyaan_subjek_baru_tidak_diseret_ke_powr():
    context = {"ticker": "POWR", "company": None, "market": "IDX"}
    for question in (
        "Bagaimana prospek data center Indonesia?",
        "Bagaimana prospek PT Cikarang Listrindo Tbk?",
        "Berapa harga emas dunia hari ini?",
        "Apa faktor utama yang memengaruhi IHSG pada 2026?",
    ):
        assert not merujuk_percakapan_sebelumnya(question)
        result = resolve_ticker(
            question, session_context=context, izinkan_semantik=False
        )
        assert result["ticker"] is None


def test_nama_perusahaan_di_riset_menjadi_identitas_tanpa_registry():
    documents = [Document(
        page_content=(
            "Telkom (TLKM) membangun data center. PT Cikarang Listrindo Tbk "
            "(POWR) memasok listrik. Indosat (ISAT) juga dibahas."
        ),
        metadata={"source": "riset.pdf", "page": 33},
    )]
    result = resolve_research_alias(
        "Bagaimana prospek PT Cikarang Listrindo Tbk menurut riset?",
        documents,
    )
    assert result["ticker"] == "POWR"
    assert result["market"] == "IDX"
    assert result["llm_calls"] == 0


def test_nama_perusahaan_tidak_ditebak_jika_bukti_ambigu():
    documents = [
        Document(
            page_content=f"PT Cikarang Listrindo Tbk ({ticker}) tercatat.",
            metadata={"source": "riset.pdf", "page": page},
        )
        for page, ticker in ((1, "POWR"), (2, "ABCD"))
    ]
    assert resolve_research_alias("Prospek Cikarang Listrindo?", documents) is None
    assert resolve_research_alias("Prospek data center?", documents) is None


def test_ticker_eksplisit_mengalahkan_konteks_sesi():
    """
    Session context tidak boleh menimpa apa yang pengguna sebut sendiri.
    """
    hasil = resolve_ticker(
        "Bagaimana prospek BMRI?",
        session_context=KONTEKS_BBRI,
        izinkan_semantik=False,
    )

    assert hasil["ticker"] == "BMRI"
    assert hasil["level"] == LEVEL_EKSPLISIT


def test_konteks_sesi_tidak_dipakai_tanpa_kata_rujukan():
    """
    "Harga emas hari ini" tidak boleh berubah menjadi BBRI hanya karena
    BBRI dibahas sebelumnya.
    """
    hasil = resolve_ticker(
        "Berapa harga emas dunia hari ini?",
        session_context=KONTEKS_BBRI,
        izinkan_semantik=False,
    )

    assert hasil["status"] != STATUS_RESOLVED


# ============================================================
# TINGKAT 3 — SEMANTIK
# ============================================================

def test_nama_perusahaan_diselesaikan_dari_context():
    llm = LLMPalsu(
        '{"company": "PT Bank Rakyat Indonesia (Persero) Tbk", '
        '"ticker": "BBRI", "market": "IDX", "confidence": "high", '
        '"alternatives": [], "reason": "disebut di context"}'
    )

    hasil = resolve_semantic(
        "Berapa harga Bank Rakyat Indonesia sekarang?",
        retriever=RetrieverPalsu(DOKUMEN_BBRI),
        llm=llm,
    )

    assert hasil["status"] == STATUS_RESOLVED
    assert hasil["level"] == LEVEL_SEMANTIK
    assert hasil["ticker"] == "BBRI"
    assert hasil["market"] == "IDX"
    assert hasil["llm_calls"] == 1


def test_ticker_yang_tidak_ada_di_context_ditolak():
    """
    Pengaman grounding. LLM hafal ribuan ticker dari pengetahuan
    umumnya; jawaban yang tidak bisa ditelusuri ke dokumen merusak
    sifat RAG-nya.
    """
    llm = LLMPalsu(
        '{"company": "PT Unilever Indonesia Tbk", "ticker": "UNVR", '
        '"market": "IDX", "confidence": "high", "alternatives": []}'
    )

    hasil = resolve_semantic(
        "Harga Unilever sekarang?",
        retriever=RetrieverPalsu(DOKUMEN_BBRI),
        llm=llm,
    )

    assert hasil["status"] == STATUS_NOT_FOUND
    assert "context" in hasil["reason"]


def test_llm_boleh_menyatakan_ambigu():
    llm = LLMPalsu(
        '{"company": "", "ticker": "", "market": "UNKNOWN", '
        '"confidence": "low", "alternatives": ["BREN", "BRPT"], '
        '"reason": "grup Barito punya beberapa emiten"}'
    )

    hasil = resolve_semantic(
        "Harga saham Barito sekarang?",
        retriever=RetrieverPalsu(DOKUMEN_BBRI),
        llm=llm,
    )

    assert hasil["status"] == STATUS_AMBIGUOUS
    assert hasil["candidates"] == ["BREN", "BRPT"]


def test_bursa_tidak_diketahui_bukan_resolved():
    """
    Tanpa bursa, simbol harga tidak bisa dibentuk dengan benar — dan
    simbol yang salah menghasilkan harga perusahaan lain tanpa error.
    """
    llm = LLMPalsu(
        '{"company": "BBRI", "ticker": "BBRI", "market": "UNKNOWN", '
        '"confidence": "medium", "alternatives": []}'
    )

    hasil = resolve_semantic(
        "Harga BBRI di bursa mana ya?",
        retriever=RetrieverPalsu(DOKUMEN_BBRI),
        llm=llm,
    )

    assert hasil["status"] == STATUS_NOT_FOUND


def test_retriever_kosong_tidak_memanggil_llm():
    llm = LLMPalsu("{}")

    hasil = resolve_semantic(
        "Harga sesuatu?", retriever=RetrieverPalsu([]), llm=llm
    )

    assert hasil["status"] == STATUS_NOT_FOUND
    assert llm.dipanggil == 0


def test_json_rusak_tidak_melempar_exception():
    llm = LLMPalsu("maaf saya tidak bisa menjawab")

    hasil = resolve_semantic(
        "Harga apa ya?", retriever=RetrieverPalsu(DOKUMEN_BBRI), llm=llm
    )

    assert hasil["status"] == STATUS_NOT_FOUND


# ============================================================
# DUA KASUS DARI DATASET EVALUASI
# ============================================================
#
# `live_04` dan `ambigu_02` hanya bisa diselesaikan lewat jalur
# semantik, jadi di evaluator mode pola keduanya dilewati. Kemampuannya
# tetap harus terbukti — dan di sinilah pembuktiannya, dengan retriever
# dan LLM tiruan sehingga tidak butuh PgVector maupun API key.

DOKUMEN_BARITO = [
    Document(
        page_content=(
            "Grup Barito diwakili beberapa emiten: PT Barito Renewables "
            "Tbk (BREN), PT Barito Pacific Tbk (BRPT), dan PT Chandra "
            "Asri Pacific Tbk (TPIA)."
        ),
        metadata={"source": "barito.pdf", "page": 4, "chunk_id": "b1"},
    )
]


def test_live_04_nama_perusahaan_resolve_ke_bbri():
    """
    "Berapa harga saham Bank Rakyat Indonesia saat ini?"

    Tidak ada kode di pertanyaan, jadi jalur eksplisit dan sesi tidak
    menghasilkan apa-apa. Jalur semantik yang harus menyelesaikannya.
    """
    llm = LLMPalsu(
        '{"company": "PT Bank Rakyat Indonesia (Persero) Tbk", '
        '"ticker": "BBRI", "market": "IDX", "confidence": "high", '
        '"alternatives": [], "reason": "disebut di context"}'
    )

    hasil = resolve_ticker(
        "Berapa harga saham Bank Rakyat Indonesia saat ini?",
        retriever=RetrieverPalsu(DOKUMEN_BBRI),
        llm=llm,
    )

    assert hasil["status"] == STATUS_RESOLVED
    assert hasil["level"] == LEVEL_SEMANTIK
    assert hasil["ticker"] == "BBRI"
    assert hasil["market"] == "IDX"


def test_ambigu_02_barito_minta_klarifikasi():
    """
    "Harga saham Barito sekarang berapa?"

    Grup Barito punya beberapa emiten. Menebak salah satu jauh lebih
    buruk daripada bertanya: pengguna akan menerima harga perusahaan
    yang tidak ia maksud, tanpa tanda apa pun bahwa itu salah.
    """
    llm = LLMPalsu(
        '{"company": "", "ticker": "", "market": "UNKNOWN", '
        '"confidence": "low", "alternatives": ["BREN", "BRPT", "TPIA"], '
        '"reason": "grup Barito punya beberapa emiten"}'
    )

    hasil = resolve_ticker(
        "Harga saham Barito sekarang berapa?",
        retriever=RetrieverPalsu(DOKUMEN_BARITO),
        llm=llm,
    )

    assert hasil["status"] == STATUS_AMBIGUOUS
    assert hasil["ticker"] is None
    assert set(hasil["candidates"]) == {"BREN", "BRPT", "TPIA"}


def test_jalur_semantik_hanya_dipakai_saat_dibutuhkan():
    """
    Pertanyaan dengan kode eksplisit tidak boleh menyentuh jalur
    semantik sama sekali — itu jaminan biaya, bukan sekadar kerapian.
    """
    llm = LLMPalsu("{}")
    retriever = RetrieverPalsu(DOKUMEN_BBRI)

    resolve_ticker("Harga BBRI sekarang?", retriever=retriever, llm=llm)

    assert llm.dipanggil == 0
    assert retriever.dipanggil == 0
