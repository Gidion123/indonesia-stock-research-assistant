"""
Memori percakapan.

Dua hal yang dijaga di sini, dan keduanya pernah menjadi masalah nyata
di project ini:

1.  Percakapan harus nyambung. "Kalau prospeknya bagaimana?" setelah
    membahas BBRI harus tetap mencari dokumen BBRI — sebelum ada memori,
    pertanyaan seperti itu dikirim apa adanya ke retrieval dan pulang
    membawa chunk acak.

2.  Memori tidak boleh menjadi sumber angka. Ia menyelesaikan RUJUKAN,
    bukan menyimpan FAKTA. Harga selalu diambil ulang.
"""

import pytest

from src import memory, session_context


# ============================================================
# BENTUK DASAR
# ============================================================

def test_memori_baru_kosong():
    m = memory.memori_baru()

    assert m["konteks"] is None
    assert m["giliran"] == []
    assert memory.riwayat_teks(m) == ""


def test_memori_none_diperlakukan_sebagai_kosong():
    assert memory.konteks_identitas(None) is None
    assert memory.riwayat_teks(None) == ""
    assert memory.kueri_pencarian("apa itu BBRI?", None) == "apa itu BBRI?"


def _hasil(ticker=None, jawaban="Jawaban.", company=None, market="IDX"):
    resolution = (
        None
        if ticker is None
        else {
            "ticker": ticker,
            "company": company,
            "market": market,
            "status": "resolved",
        }
    )

    return {"answer": jawaban, "resolution": resolution}


# ============================================================
# IDENTITAS
# ============================================================

def test_identitas_tersimpan_dari_giliran_pertama():
    m = memory.perbarui(
        memory.memori_baru(),
        "Harga BBRI sekarang berapa?",
        _hasil("BBRI"),
    )

    assert memory.konteks_identitas(m)["ticker"] == "BBRI"


def test_identitas_bertahan_saat_giliran_berikutnya_tanpa_saham():
    m = memory.perbarui(memory.memori_baru(), "Harga BBRI?", _hasil("BBRI"))
    m = memory.perbarui(m, "Kalau prospeknya bagaimana?", _hasil(None))

    assert memory.konteks_identitas(m)["ticker"] == "BBRI"


def test_identitas_berganti_saat_saham_lain_dibahas():
    m = memory.perbarui(memory.memori_baru(), "Harga BBRI?", _hasil("BBRI"))
    m = memory.perbarui(m, "Kalau BUMI?", _hasil("BUMI"))

    assert memory.konteks_identitas(m)["ticker"] == "BUMI"


def test_harga_tidak_boleh_masuk_memori():
    """
    Batas yang paling mudah dilanggar dan paling mahal akibatnya:
    harga basi yang disajikan sebagai harga sekarang.
    """
    rusak = {"konteks": {"ticker": "BBRI", "price": 4210}, "giliran": []}

    with pytest.raises(session_context.KontekSesiTidakValid):
        memory.konteks_identitas(rusak)


# ============================================================
# RIWAYAT
# ============================================================

def test_riwayat_memuat_pertanyaan_dan_jawaban():
    m = memory.perbarui(
        memory.memori_baru(),
        "Harga BBRI sekarang berapa?",
        _hasil("BBRI", jawaban="BBRI terakhir di Rp4.210."),
    )

    teks = memory.riwayat_teks(m)

    assert "Pengguna: Harga BBRI sekarang berapa?" in teks
    assert "Asisten: BBRI terakhir di Rp4.210." in teks


def test_seluruh_percakapan_diingat_selama_sesi():
    """
    Memori menyimpan SEMUA giliran sesi ini. Yang menghapusnya hanya
    "Percakapan baru" atau memuat ulang halaman.
    """
    m = memory.memori_baru()

    for nomor in range(40):
        m = memory.perbarui(m, f"Pertanyaan {nomor}", _hasil("BBRI"))

    # 40 pertanyaan + 40 jawaban, tidak ada yang dibuang.
    assert len(m["giliran"]) == 80


def test_percakapan_pendek_masuk_prompt_seluruhnya():
    """
    Batasnya per karakter, bukan per giliran. Percakapan yang belum
    besar tetap ikut utuh — termasuk pertanyaan paling awal.
    """
    m = memory.memori_baru()

    for nomor in range(40):
        m = memory.perbarui(m, f"Pertanyaan {nomor}", _hasil("BBRI"))

    teks = memory.riwayat_teks(m)

    assert "Pertanyaan 0" in teks
    assert "Pertanyaan 39" in teks


def test_riwayat_panjang_dipangkas_dari_yang_paling_lama():
    """
    Ketika percakapan benar-benar besar, yang dikirim ke model dipangkas
    — dan yang dipertahankan adalah giliran TERBARU, karena itu yang
    menentukan maksud pertanyaan sekarang.
    """
    m = memory.memori_baru()

    for nomor in range(60):
        m = memory.perbarui(
            m,
            f"Pertanyaan {nomor} " + "x" * 200,
            _hasil("BBRI", jawaban=f"Jawaban {nomor} " + "y" * 200),
        )

    teks = memory.riwayat_teks(m)

    assert len(teks) <= memory.BATAS_KARAKTER_RIWAYAT + 200
    assert "Pertanyaan 59" in teks
    assert "Pertanyaan 0 " not in teks

    # Dipangkas hanya untuk prompt; ingatannya sendiri tetap utuh.
    assert len(m["giliran"]) == 120


def test_batas_riwayat_bisa_diatur_pemanggil():
    m = memory.memori_baru()

    for nomor in range(10):
        m = memory.perbarui(m, f"Pertanyaan {nomor}", _hasil("BBRI"))

    sempit = memory.riwayat_teks(m, batas=80)

    assert len(sempit) <= 280
    assert "Pertanyaan 9" in sempit


def test_saham_yang_pernah_dibahas_bertahan_sepanjang_sesi():
    """
    Giliran paling awal boleh tidak muat lagi di prompt, tetapi saham
    yang pernah dibahas tidak boleh hilang dari ingatan.
    """
    m = memory.perbarui(memory.memori_baru(), "Prospek BBRI?", _hasil("BBRI"))
    m = memory.perbarui(m, "Kalau BUMI?", _hasil("BUMI"))

    for nomor in range(40):
        m = memory.perbarui(m, f"Lanjutan {nomor}", _hasil(None))

    assert memory.saham_dibahas(m) == ["BBRI", "BUMI"]
    assert "BBRI" in memory.riwayat_teks(m)
    assert "BUMI" in memory.riwayat_teks(m)


def test_jawaban_panjang_dipotong():
    panjang = "A" * 2000

    m = memory.perbarui(
        memory.memori_baru(), "Tanya", _hasil("BBRI", jawaban=panjang)
    )

    isi = [g for g in m["giliran"] if g["peran"] == "assistant"][0]["isi"]

    assert len(isi) <= memory.BATAS_KARAKTER_JAWABAN + 1


def test_giliran_tanpa_jawaban_tidak_menyisipkan_baris_kosong():
    m = memory.perbarui(
        memory.memori_baru(), "Tanya", {"answer": None, "resolution": None}
    )

    assert [g["peran"] for g in m["giliran"]] == ["user"]


# ============================================================
# KUERI PENCARIAN
# ============================================================

def test_pertanyaan_lanjutan_dilengkapi_ticker():
    m = memory.perbarui(memory.memori_baru(), "Harga BBRI?", _hasil("BBRI"))

    kueri = memory.kueri_pencarian("Kalau prospeknya bagaimana?", m)

    assert "BBRI" in kueri
    assert kueri.startswith("Kalau prospeknya bagaimana?")


def test_powr_lanjutan_tanpa_akhiran_nya_memakai_saham_aktif():
    m = memory.perbarui(memory.memori_baru(), "Prospek POWR?", _hasil("POWR"))

    assert memory.kueri_pencarian("Apa target harga?", m) == "Apa target harga? POWR"
    assert memory.kueri_pencarian("Bagaimana valuasi?", m) == "Bagaimana valuasi? POWR"
    assert memory.kueri_pencarian("Bagaimana laba?", m) == "Bagaimana laba? POWR"
    assert memory.kueri_pencarian("Apakah sekarang sudah mencapai target?", m).endswith("POWR")


@pytest.mark.parametrize(
    "pertanyaan",
    [
        "Bagaimana prospeknya?",
        "Apa risikonya?",
        "Saham tersebut bagaimana?",
        "Kalau targetnya berapa?",
        "Yang tadi itu bagaimana?",
    ],
)
def test_berbagai_bentuk_rujukan_dikenali(pertanyaan):
    assert memory.merujuk_giliran_sebelumnya(pertanyaan)


@pytest.mark.parametrize(
    "pertanyaan",
    [
        "Apa faktor utama yang memengaruhi IHSG pada 2026?",
        "Bagaimana prospek BUMI menurut riset?",
        "Siapa presiden Indonesia tahun 2035?",
    ],
)
def test_pertanyaan_berdiri_sendiri_tidak_dianggap_rujukan(pertanyaan):
    assert not memory.merujuk_giliran_sebelumnya(pertanyaan)


def test_pertanyaan_berdiri_sendiri_tidak_dilengkapi():
    """
    Pertanyaan yang punya subjek sendiri tidak boleh diseret ke topik
    lama — di situlah memori berubah dari membantu menjadi merusak.
    """
    m = memory.perbarui(memory.memori_baru(), "Harga BBRI?", _hasil("BBRI"))

    pertanyaan = "Apa faktor utama yang memengaruhi IHSG pada 2026?"

    assert memory.kueri_pencarian(pertanyaan, m) == pertanyaan


def test_pergantian_topik_tidak_mewarisi_powr():
    m = memory.perbarui(memory.memori_baru(), "Prospek POWR?", _hasil("POWR"))

    for question in (
        "Bagaimana prospek PT Cikarang Listrindo Tbk menurut riset?",
        "Bagaimana prospek data center Indonesia?",
        "Apa faktor utama yang memengaruhi IHSG pada 2026?",
        "Saya punya saham lain, bagaimana?",
    ):
        assert memory.kueri_pencarian(question, m) == question


def test_ticker_eksplisit_tidak_ditimpa_memori():
    m = memory.perbarui(memory.memori_baru(), "Harga BBRI?", _hasil("BBRI"))

    pertanyaan = "Bagaimana prospeknya BUMI?"
    kueri = memory.kueri_pencarian(pertanyaan, m)

    assert kueri == pertanyaan
    assert "BBRI" not in kueri


def test_tanpa_identitas_kueri_tidak_berubah():
    m = memory.memori_baru()
    pertanyaan = "Bagaimana prospeknya?"

    assert memory.kueri_pencarian(pertanyaan, m) == pertanyaan


def test_nama_perusahaan_ikut_kalau_diketahui():
    m = memory.perbarui(
        memory.memori_baru(),
        "Harga BBRI?",
        _hasil("BBRI", company="Bank Rakyat Indonesia"),
    )

    kueri = memory.kueri_pencarian("Bagaimana prospeknya?", m)

    assert "BBRI" in kueri
    assert "Bank Rakyat Indonesia" in kueri


# ============================================================
# LUPAKAN
# ============================================================

def test_percakapan_baru_benar_benar_melupakan():
    m = memory.perbarui(memory.memori_baru(), "Harga BBRI?", _hasil("BBRI"))
    m = memory.lupakan()

    assert memory.konteks_identitas(m) is None
    assert memory.riwayat_teks(m) == ""
    assert memory.kueri_pencarian("Bagaimana prospeknya?", m) == (
        "Bagaimana prospeknya?"
    )


# ============================================================
# RANTAI LENGKAP: app -> assistant -> rag_chain
# ============================================================
#
# Bagian paling rawan bukan modul memorinya, melainkan sambungannya.
# Retriever dan LLM diganti stub, jadi test ini berjalan tanpa
# PostgreSQL maupun API key.

class RetrieverStub:
    def __init__(self):
        self.kueri = []

    def invoke(self, kueri):
        self.kueri.append(kueri)

        from langchain_core.documents import Document

        return [
            Document(
                page_content="BBRI target TP1 Rp3.400.",
                metadata={"source": "riset.pdf", "page": 9,
                          "chunk_id": "abc123"},
            )
        ]


class LLMStub:
    def __init__(self):
        self.pesan = []

    def invoke(self, messages):
        self.pesan.append(messages)

        class Balasan:
            content = "Jawaban [riset.pdf hal.9]."

        return Balasan()


def _jalankan(pertanyaan, memori):
    from src.assistant import jawab

    retriever, llm = RetrieverStub(), LLMStub()

    hasil = jawab(
        pertanyaan,
        session_context_data=memory.konteks_identitas(memori),
        riwayat=memory.riwayat_teks(memori),
        kueri_retrieval=memory.kueri_pencarian(pertanyaan, memori),
        retriever=retriever,
        llm=llm,
        izinkan_llm_router=False,
    )

    return hasil, retriever, llm


def test_pertanyaan_lanjutan_mencari_dokumen_saham_yang_sedang_dibahas():
    memori = memory.perbarui(
        memory.memori_baru(),
        "Bagaimana prospek BBRI menurut riset?",
        _hasil("BBRI", jawaban="Prospek BBRI positif [riset.pdf hal.9]."),
    )

    _, retriever, _ = _jalankan("Kalau targetnya berapa?", memori)

    assert "BBRI" in retriever.kueri[0], (
        f"Kueri pencarian kehilangan konteks: {retriever.kueri[0]!r}"
    )


def test_pertanyaan_asli_tetap_utuh_ke_model():
    """
    Yang dilengkapi hanya kueri pencarian. Pertanyaan yang dibaca model
    harus tetap kalimat pengguna apa adanya.
    """
    memori = memory.perbarui(
        memory.memori_baru(), "Prospek BBRI?", _hasil("BBRI")
    )

    _, _, llm = _jalankan("Kalau targetnya berapa?", memori)

    teks = llm.pesan[0][0].content

    assert "PERTANYAAN:\nKalau targetnya berapa?" in teks


def test_riwayat_sampai_ke_prompt():
    memori = memory.perbarui(
        memory.memori_baru(),
        "Prospek BBRI?",
        _hasil("BBRI", jawaban="Prospek BBRI positif."),
    )

    _, _, llm = _jalankan("Kalau targetnya berapa?", memori)

    teks = llm.pesan[0][0].content

    assert "PERCAKAPAN SEBELUMNYA:" in teks
    assert "Prospek BBRI?" in teks


def test_tanpa_memori_prompt_sama_persis_seperti_sebelumnya():
    """
    Inilah yang membuat angka evaluasi tetap sebanding: tanpa memori,
    tidak ada blok riwayat sama sekali di prompt.
    """
    _, retriever, llm = _jalankan(
        "Bagaimana prospek BBRI menurut riset?", memory.memori_baru()
    )

    teks = llm.pesan[0][0].content

    assert "PERCAKAPAN SEBELUMNYA" not in teks
    assert retriever.kueri[0] == "Bagaimana prospek BBRI menurut riset?"


def test_identitas_ikut_terbawa_dari_pertanyaan_riset():
    """
    Sebelumnya jalur RAG tidak pernah memanggil resolver, sehingga
    "prospek BBRI" tidak membuat BBRI diingat — dan pertanyaan lanjutan
    kehilangan konteks.
    """
    hasil, _, _ = _jalankan(
        "Bagaimana prospek BBRI menurut riset?", memory.memori_baru()
    )

    memori = memory.perbarui(
        memory.memori_baru(), "Bagaimana prospek BBRI menurut riset?", hasil
    )

    assert memory.konteks_identitas(memori)["ticker"] == "BBRI"


def test_jalur_riset_tidak_menambah_panggilan_llm_untuk_memori():
    """
    Resolusi identitas di jalur RAG memakai tingkat eksplisit dan sesi
    saja. Kalau semantik ikut menyala, setiap pertanyaan riset membayar
    satu panggilan LLM tambahan hanya demi mengisi memori.
    """
    hasil, _, _ = _jalankan(
        "Bagaimana prospek BBRI menurut riset?", memory.memori_baru()
    )

    assert hasil["trace"]["resolution_llm_calls"] == 0
