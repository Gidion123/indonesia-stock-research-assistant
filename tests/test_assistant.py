"""
Test orkestrasi end-to-end, semuanya dengan tiruan.

Yang dijaga di sini adalah janji-janji yang menyeberangi modul:
jalur cepat benar-benar cepat, dua sumber tidak tertukar, untung/rugi
dihitung Python, memori sesi tidak pernah menjadi sumber angka, dan
sistem tidak menjawab ketika tidak tahu.
"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src import assistant, live_compare, live_price, session_context
from src.router import (
    INTENT_LIVE_COMPARE,
    INTENT_LIVE_PRICE,
    INTENT_OUT_OF_SCOPE,
)


HARGA_BBRI = {
    "symbol": "BBRI.JK",
    "price": 4210.0,
    "previous_close": 4150.0,
    "change": 60.0,
    "change_percent": 1.4458,
    "currency": "IDR",
    "exchange": "JKT",
    "short_name": None,
    "as_of": "2026-09-14T07:00:00+00:00",
    "status": "ok",
    "error": None,
    "disclaimer": "Harga dapat tertunda.",
}

HARGA_GAGAL = dict(
    HARGA_BBRI,
    price=None,
    status="transient_error",
    error="YFRateLimitError: Too Many Requests",
)

DOKUMEN = [
    Document(
        page_content="Target TP 1 BBRI berada di Rp3.400.",
        metadata={"source": "ihsg.pdf", "page": 12, "chunk_id": "a1"},
    )
]


class RetrieverPalsu:
    def __init__(self, documents=None):
        self._documents = DOKUMEN if documents is None else documents
        self.dipanggil = 0

    def invoke(self, query):
        self.dipanggil += 1
        return self._documents


class LLMPalsu:
    def __init__(self, isi="Jawaban."):
        self.isi = isi
        self.prompt_terakhir = None
        self.dipanggil = 0

    def invoke(self, prompt):
        self.dipanggil += 1
        self.prompt_terakhir = prompt

        class R:
            content = self.isi

        return R()


class HargaPalsu:
    """Pengganti market_data.ambil_harga yang menghitung panggilannya."""

    def __init__(self, data=None):
        self.data = data or HARGA_BBRI
        self.panggilan = []

    def __call__(self, symbol, **kwargs):
        self.panggilan.append(symbol)
        return self.data

    @property
    def jumlah(self):
        return len(self.panggilan)


def jawab_dengan_harga(question, harga, **kwargs):
    """
    Kedua jalur live mengimpor `ambil_harga` sendiri, jadi keduanya
    ditambal bersamaan.
    """
    with patch.object(live_price, "ambil_harga", harga), patch.object(
        live_compare, "ambil_harga", harga
    ):
        return assistant.jawab(question, **kwargs)


# ============================================================
# ANGGARAN JARINGAN — YAHOO TIDAK BOLEH DIPANGGIL BERLEBIHAN
# ============================================================

@pytest.mark.parametrize(
    "question",
    [
        "Bagaimana prospek BBRI?",
        "Berapa target harga BBRI menurut riset?",
        "Apa katalis utama saham BUMI?",
        "Apa risiko terbesar di sektor perbankan menurut laporan?",
        "Berapa rasio CASA bank itu menurut riset?",
    ],
)
def test_rag_tidak_pernah_memanggil_yahoo(question):
    """
    Kriteria arsitektur paling penting: pertanyaan riset tidak
    membutuhkan harga, jadi tidak boleh ada satu pun permintaan pasar.
    """
    harga = HargaPalsu()

    with patch.object(assistant, "_jalur_rag") as rag:
        rag.return_value = {"answer": "x", "status": "ok"}
        jawab_dengan_harga(question, harga, izinkan_llm_router=False)

    assert harga.jumlah == 0, question


def test_out_of_scope_tidak_memanggil_yahoo_maupun_llm():
    harga = HargaPalsu()
    llm = LLMPalsu()

    hasil = jawab_dengan_harga(
        "Cuaca di Medan hari ini gimana?", harga, llm=llm
    )

    assert hasil["intent"] == INTENT_OUT_OF_SCOPE
    assert harga.jumlah == 0
    assert llm.dipanggil == 0
    assert hasil["market_data"] is None


def test_ambiguitas_tidak_memanggil_yahoo():
    """
    Yahoo tidak boleh dipanggil sebelum entity resolution selesai —
    dua ticker berarti belum selesai.
    """
    harga = HargaPalsu()

    hasil = jawab_dengan_harga(
        "BBRI dan BMRI mana yang lebih murah sekarang?", harga
    )

    assert hasil["status"] == "ambiguous"
    assert harga.jumlah == 0


def test_entitas_tidak_dikenal_tidak_memanggil_yahoo():
    """
    "Harga emas dunia hari ini" — router benar menandainya butuh harga,
    resolver yang menolak. Yahoo tidak pernah disentuh.
    """
    harga = HargaPalsu()
    llm = LLMPalsu(
        '{"company": "", "ticker": "", "market": "UNKNOWN", '
        '"confidence": "low", "alternatives": []}'
    )

    hasil = jawab_dengan_harga(
        "Berapa harga emas dunia hari ini?",
        harga,
        llm=llm,
        retriever=RetrieverPalsu(),
    )

    assert hasil["status"] == "unresolved_entity"
    assert harga.jumlah == 0
    assert hasil["market_data"] is None


def test_pertanyaan_kosong_tidak_memanggil_apa_pun():
    harga = HargaPalsu()

    hasil = jawab_dengan_harga("", harga)

    assert hasil["status"] == "empty_question"
    assert harga.jumlah == 0


def test_live_price_memanggil_yahoo_tepat_sekali():
    harga = HargaPalsu()

    hasil = jawab_dengan_harga("Harga BBRI sekarang?", harga)

    assert hasil["intent"] == INTENT_LIVE_PRICE
    assert harga.panggilan == ["BBRI.JK"]


def test_live_compare_memanggil_yahoo_setelah_entity_resolved():
    harga = HargaPalsu()

    hasil = jawab_dengan_harga(
        "BBRI sudah mencapai target belum?",
        harga,
        llm=LLMPalsu("Penjelasan."),
        retriever=RetrieverPalsu(),
    )

    assert hasil["intent"] == INTENT_LIVE_COMPARE
    assert harga.panggilan == ["BBRI.JK"]


def test_konteks_sesi_memicu_pengambilan_harga_baru():
    """
    Harga TIDAK boleh diambil dari memori percakapan; setiap permintaan
    kondisi terkini harus menembak market_data lagi.
    """
    harga = HargaPalsu()
    konteks = {"ticker": "BBRI", "company": None, "market": "IDX"}

    hasil = jawab_dengan_harga(
        "Kalau saham itu sekarang berapa?",
        harga,
        session_context_data=konteks,
    )

    assert hasil["resolution"]["level"] == "session"
    assert harga.panggilan == ["BBRI.JK"]


# ============================================================
# ANGGARAN LLM
# ============================================================

def test_harga_eksplisit_tidak_memanggil_llm_sama_sekali():
    llm = LLMPalsu()
    retriever = RetrieverPalsu()
    harga = HargaPalsu()

    hasil = jawab_dengan_harga(
        "Harga BBRI sekarang?", harga, llm=llm, retriever=retriever
    )

    assert hasil["status"] == "ok"
    assert "4,210" in hasil["answer"]
    assert llm.dipanggil == 0
    assert retriever.dipanggil == 0
    assert hasil["trace"]["router_llm_calls"] == 0
    assert hasil["trace"]["resolution_llm_calls"] == 0


def test_jalur_cepat_menyertakan_disclaimer():
    hasil = jawab_dengan_harga("Harga BBRI sekarang?", HargaPalsu())

    assert "tertunda" in hasil["answer"]


# ============================================================
# KEGAGALAN HARGA
# ============================================================

def test_harga_gagal_tidak_memunculkan_angka():
    hasil = jawab_dengan_harga(
        "Harga BBRI sekarang?", HargaPalsu(HARGA_GAGAL)
    )

    assert hasil["status"] != "ok"
    assert "4,210" not in (hasil["answer"] or "")
    assert "4210" not in (hasil["answer"] or "")


def test_error_mentah_tidak_bocor_ke_pengguna():
    """
    Pengguna tidak boleh melihat "YFRateLimitError". Detailnya tetap
    tersedia di field `error` untuk log.
    """
    hasil = jawab_dengan_harga(
        "Harga BBRI sekarang?", HargaPalsu(HARGA_GAGAL)
    )

    assert "YFRateLimitError" not in hasil["answer"]
    assert "Exception" not in hasil["answer"]
    assert "YFRateLimitError" in (hasil["error"] or "")


def test_compare_tidak_memanggil_llm_saat_harga_gagal():
    """
    Model yang diminta membandingkan tanpa angka akan mengarang angka.
    """
    llm = LLMPalsu("seharusnya tidak dipanggil")

    hasil = jawab_dengan_harga(
        "BBRI sudah mencapai target belum?",
        HargaPalsu(HARGA_GAGAL),
        llm=llm,
        retriever=RetrieverPalsu(),
    )

    assert llm.dipanggil == 0
    assert hasil["status"] != "ok"


# ============================================================
# DUA SUMBER TIDAK TERTUKAR
# ============================================================

def test_live_compare_mengirim_kedua_sumber_ke_llm():
    llm = LLMPalsu("Harga 4.210, target TP1 Rp3.400 [ihsg.pdf hal.12].")

    hasil = jawab_dengan_harga(
        "BBRI sudah mencapai target belum?",
        HargaPalsu(),
        llm=llm,
        retriever=RetrieverPalsu(),
    )

    assert set(hasil["sources_used"]) == {"market_data", "knowledge_base"}

    prompt = llm.prompt_terakhir
    assert "DATA PASAR" in prompt
    assert "CONTEXT" in prompt
    assert "4,210" in prompt
    assert "Rp3.400" in prompt


def test_prompt_compare_melarang_sitasi_untuk_harga_pasar():
    from src.prompts import PROMPT_LIVE_COMPARE

    assert "TIDAK BOLEH diberi citation" in PROMPT_LIVE_COMPARE


def test_saham_tanpa_riset_dijawab_jujur():
    """
    "Tidak ada risetnya" berbeda dari "saya tidak tahu saham itu".
    """
    hasil = jawab_dengan_harga(
        "BBRI sudah mencapai target belum?",
        HargaPalsu(),
        retriever=RetrieverPalsu([]),
    )

    assert hasil["status"] == "no_research_coverage"
    assert "tidak membahas" in hasil["answer"]


# ============================================================
# PROFIT / LOSS DETERMINISTIK
# ============================================================

def test_profit_loss_dihitung_secara_deterministik():
    hasil = jawab_dengan_harga(
        "Kalau saya beli BBRI di 4.000, sekarang untung berapa?",
        HargaPalsu(),
        llm=LLMPalsu("Penjelasan hasil investasi."),
        retriever=RetrieverPalsu(),
    )

    pl = hasil["profit_loss"]

    assert pl is not None
    assert pl["entry_price"] == 4000.0
    assert pl["current_price"] == 4210.0
    assert pl["difference"] == 210.0
    assert round(pl["return_percent"], 2) == 5.25
    assert pl["position"] == "profit"


def test_profit_loss_rugi_dihitung_dengan_benar():
    hasil = jawab_dengan_harga(
        "Saya beli BBRI di 4.000, sekarang rugi berapa?",
        HargaPalsu(dict(HARGA_BBRI, price=3800.0)),
        llm=LLMPalsu("Penjelasan."),
        retriever=RetrieverPalsu(),
    )

    pl = hasil["profit_loss"]

    assert pl["difference"] == -200.0
    assert round(pl["return_percent"], 2) == -5.0
    assert pl["position"] == "loss"


def test_profit_loss_impas_dihitung_dengan_benar():
    hasil = jawab_dengan_harga(
        "Saya beli BBRI di 4.000, sekarang untung berapa?",
        HargaPalsu(dict(HARGA_BBRI, price=4000.0)),
        llm=LLMPalsu("Penjelasan."),
        retriever=RetrieverPalsu(),
    )

    pl = hasil["profit_loss"]

    assert pl["difference"] == 0.0
    assert pl["return_percent"] == 0.0
    assert pl["position"] == "breakeven"


def test_tanpa_harga_entry_tidak_mengarang_profit_loss():
    hasil = jawab_dengan_harga(
        "BBRI sudah mencapai target belum?",
        HargaPalsu(),
        llm=LLMPalsu("Penjelasan."),
        retriever=RetrieverPalsu(),
    )

    assert hasil["profit_loss"] is None


def test_harga_entry_tidak_diambil_dari_angka_target():
    """
    "Target BBRI 5.000" memuat angka, tetapi angka itu milik riset —
    bukan harga beli pengguna.
    """
    hasil = jawab_dengan_harga(
        "Target BBRI 5.000, sekarang bagaimana?",
        HargaPalsu(),
        llm=LLMPalsu("Penjelasan."),
        retriever=RetrieverPalsu(),
    )

    assert hasil["profit_loss"] is None


def test_llm_menerima_hasil_hitungan_bukan_soal_hitungan():
    llm = LLMPalsu("Penjelasan.")

    jawab_dengan_harga(
        "Saya beli BBRI di 4.000, sekarang untung berapa?",
        HargaPalsu(),
        llm=llm,
        retriever=RetrieverPalsu(),
    )

    prompt = llm.prompt_terakhir

    assert "PERHITUNGAN POSISI" in prompt
    assert "5.25%" in prompt
    assert "Jangan menghitung ulang" in prompt


# ============================================================
# KLARIFIKASI
# ============================================================

def test_ambigu_bertanya_balik_dengan_kandidat_terisi():
    """
    Bug sebelumnya: pesan dikirim mentah sehingga "{kandidat}" muncul
    di layar pengguna.
    """
    hasil = jawab_dengan_harga(
        "BBRI dan BMRI mana yang lebih murah sekarang?", HargaPalsu()
    )

    assert hasil["status"] == "ambiguous"
    assert "{kandidat}" not in hasil["answer"]
    assert "BBRI" in hasil["answer"]
    assert "BMRI" in hasil["answer"]


# ============================================================
# MEMORI SESI
# ============================================================

def test_konteks_sesi_hanya_menyimpan_identitas_bukan_harga():
    hasil = jawab_dengan_harga("Harga BBRI sekarang?", HargaPalsu())

    konteks = assistant.perbarui_konteks_sesi(None, hasil)

    assert konteks["ticker"] == "BBRI"
    assert set(konteks) == {"ticker", "company", "market"}


def test_rag_nama_perusahaan_powr_mengisi_memori_tanpa_llm_tambahan():
    from src import memory

    documents = [Document(
        page_content="PT Cikarang Listrindo Tbk (POWR) memasok daya data center.",
        metadata={"source": "riset.pdf", "page": 33},
    )]
    llm = LLMPalsu("POWR memasok daya [riset.pdf hal.33].")
    first = assistant.jawab(
        "Bagaimana prospek PT Cikarang Listrindo Tbk menurut riset?",
        retriever=RetrieverPalsu(documents),
        llm=llm,
        izinkan_llm_router=False,
    )
    state = memory.perbarui(memory.memori_baru(), first["question"], first)

    assert first["resolution"]["ticker"] == "POWR"
    assert memory.konteks_identitas(state)["ticker"] == "POWR"
    assert "POWR Cikarang Listrindo" in memory.kueri_pencarian("Apa target harga?", state)
    assert llm.dipanggil == 1


def test_konteks_dengan_data_pasar_ditolak():
    """
    Dijaga saat ditulis, bukan saat tampil: memori percakapan yang
    memuat harga adalah cara termudah menyajikan angka basi sebagai
    angka sekarang.
    """
    with pytest.raises(session_context.KontekSesiTidakValid):
        assistant.jawab(
            "Harga BBRI sekarang?",
            session_context_data={"ticker": "BBRI", "price": 4210.0},
        )


def test_konteks_lama_bertahan_saat_putaran_tanpa_saham():
    konteks = {"ticker": "BBRI", "company": None, "market": "IDX"}

    assert (
        assistant.perbarui_konteks_sesi(konteks, {"resolution": None})
        == konteks
    )


# ============================================================
# BENTUK HASIL
# ============================================================

def test_setiap_jawaban_membawa_kunci_yang_sama():
    """
    Satu bentuk hasil untuk semua jalur, supaya UI dan evaluasi tidak
    perlu tahu jalur mana yang dipakai.
    """
    kasus = [
        jawab_dengan_harga("Harga BBRI sekarang?", HargaPalsu()),
        jawab_dengan_harga("Cuaca hari ini?", HargaPalsu()),
        jawab_dengan_harga(
            "BBRI dan BMRI mana yang lebih murah sekarang?", HargaPalsu()
        ),
    ]

    for hasil in kasus:
        for kunci in assistant.KUNCI_HASIL:
            assert kunci in hasil, (kunci, hasil["question"])


def test_jejak_biaya_tersedia():
    hasil = jawab_dengan_harga("Harga BBRI sekarang?", HargaPalsu())

    for kunci in (
        "router_llm_calls",
        "resolution_llm_calls",
        "market_data_requested",
        "market_data_from_cache",
    ):
        assert kunci in hasil["trace"]


# ============================================================
# DI LUAR CAKUPAN — JALUR END-TO-END
# ============================================================

def test_penolakan_sistem_dikenali_detektor_refusal():
    """
    Metrik `out_of_scope_refusal_rate` mengukur `is_refusal()`, bukan
    perilaku sistem. Ketika kalimat penolakan diperbaiki tanpa
    memperbarui detektornya, angka OOS jatuh dari 80% ke 20% padahal
    sistem menolak dengan benar di kelima pertanyaan.

    Sekarang polanya diturunkan dari konstanta yang sama, jadi keduanya
    tidak bisa lagi berbeda — test ini yang menjaganya.
    """
    from src import config
    from src.answer_evaluation import is_refusal
    from src.prompts import (
        PESAN_DI_LUAR_CAKUPAN,
        PESAN_TICKER_TIDAK_DIKENAL,
    )

    for pesan in (
        config.REFUSAL_MESSAGE,
        PESAN_DI_LUAR_CAKUPAN,
        PESAN_TICKER_TIDAK_DIKENAL,
    ):
        assert is_refusal(pesan), pesan[:60]


def test_jawaban_asli_tidak_dianggap_penolakan():
    """Detektor yang terlalu longgar sama merusaknya."""
    from src.answer_evaluation import is_refusal

    for jawaban in (
        "TP1 BBRI adalah Rp3.400 [ihsg.pdf hal.12].",
        "BBRI terakhir di IDR 4.210,00 (+1,45%).",
        "Katalis utama BUMI adalah diversifikasi emas [a.pdf hal.9].",
    ):
        assert not is_refusal(jawaban), jawaban


def test_harga_komoditas_ditolak_tanpa_menyentuh_yahoo():
    """
    "Berapa harga emas dunia hari ini?" — router benar menandainya
    butuh harga, resolver yang menolak. Tidak ada nama komoditas yang
    di-hardcode di mana pun: yang menghentikannya adalah pemeriksaan
    grounding, karena LLM tidak menemukan emiten di context.
    """
    from src.answer_evaluation import is_refusal

    harga = HargaPalsu()
    llm = LLMPalsu(
        '{"company": "", "ticker": "", "market": "UNKNOWN", '
        '"confidence": "low", "alternatives": []}'
    )

    hasil = jawab_dengan_harga(
        "Berapa harga emas dunia hari ini?",
        harga,
        llm=llm,
        retriever=RetrieverPalsu(),
    )

    assert hasil["status"] == "unresolved_entity"
    assert harga.jumlah == 0
    assert is_refusal(hasil["answer"])


# ============================================================
# ISOLASI MEMORI ANTAR GILIRAN
# ============================================================

def test_giliran_kedua_merujuk_saham_giliran_pertama():
    """
    Turn 1: "Bagaimana prospek BBRI menurut riset?"
    Turn 2: "Kalau saham itu sekarang berapa?"  -> BBRI
    """
    harga = HargaPalsu()

    # Giliran 1 memakai jalur RAG; retriever & llm disuntikkan supaya
    # test ini tidak butuh PgVector maupun API key.
    giliran1 = jawab_dengan_harga(
        "Bagaimana prospek BBRI menurut riset?",
        harga,
        llm=LLMPalsu("BBRI diproyeksikan menuju Rp3.400 [ihsg.pdf hal.12]."),
        retriever=RetrieverPalsu(),
    )

    konteks = assistant.perbarui_konteks_sesi(None, giliran1)

    assert konteks is not None
    assert konteks["ticker"] == "BBRI"

    harga2 = HargaPalsu()

    giliran2 = jawab_dengan_harga(
        "Kalau saham itu sekarang berapa?",
        harga2,
        session_context_data=konteks,
    )

    assert giliran2["intent"] == INTENT_LIVE_PRICE
    assert giliran2["resolution"]["level"] == "session"
    assert giliran2["resolution"]["ticker"] == "BBRI"
    assert harga2.panggilan == ["BBRI.JK"]


def test_giliran_kedua_di_luar_cakupan_tidak_mewarisi_saham():
    """
    Turn 1: "Bagaimana prospek BBRI menurut riset?"
    Turn 2: "Berapa harga emas dunia hari ini?"

    Konteks sesi berisi BBRI, tetapi pertanyaan kedua tidak merujuk
    apa pun. Mewarisi BBRI di sini akan menyajikan harga bank sebagai
    jawaban atas pertanyaan tentang emas — salah, dan tanpa satu pun
    tanda bahwa itu salah.
    """
    konteks = {
        "ticker": "BBRI",
        "company": "PT Bank Rakyat Indonesia (Persero) Tbk",
        "market": "IDX",
    }

    harga = HargaPalsu()
    llm = LLMPalsu(
        '{"company": "", "ticker": "", "market": "UNKNOWN", '
        '"confidence": "low", "alternatives": []}'
    )

    hasil = jawab_dengan_harga(
        "Berapa harga emas dunia hari ini?",
        harga,
        session_context_data=konteks,
        llm=llm,
        retriever=RetrieverPalsu(),
    )

    assert hasil["status"] == "unresolved_entity"
    assert hasil["resolution"]["ticker"] is None
    assert hasil["resolution"]["level"] != "session"
    assert harga.jumlah == 0, "Yahoo tidak boleh ditembak untuk BBRI"
    assert "4,210" not in (hasil["answer"] or "")
