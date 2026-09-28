"""
Test router, dan akurasinya diukur terhadap dataset berlabel.

Argumen inti FONDASI.md berlaku di sini juga: metrik yang tidak bisa
gagal bukan metrik. Router menambah tiga mode gagal baru — salah intent,
salah entity, dan menjawab LIVE padahal seharusnya RAG — jadi ketiganya
harus punya angka, bukan kesan.
"""

import json

import pytest

from src import config
from src.router import (
    INTENT_LIVE_COMPARE,
    INTENT_LIVE_PRICE,
    INTENT_OUT_OF_SCOPE,
    INTENT_RAG,
    butuh_dokumen,
    butuh_harga,
    classify_by_pattern,
    classify_router_intent,
)


ROUTER_EVAL_PATH = config.EVAL_DATASET_DIR / "router_eval.json"


def muat_dataset():
    with ROUTER_EVAL_PATH.open(encoding="utf-8") as file:
        return json.load(file)["items"]


# ============================================================
# POLA DETERMINISTIK
# ============================================================

@pytest.mark.parametrize(
    "question,expected",
    [
        ("Berapa target harga BBRI menurut riset?", INTENT_RAG),
        ("Harga BBRI sekarang berapa?", INTENT_LIVE_PRICE),
        ("BBRI sudah mencapai target belum?", INTENT_LIVE_COMPARE),
        ("Harga sekarang masih di bawah TP1?", INTENT_LIVE_COMPARE),
        ("Cuaca hari ini gimana?", INTENT_OUT_OF_SCOPE),
    ],
)
def test_pola_mengenali_kasus_jelas(question, expected):
    intent, _ = classify_by_pattern(question)

    assert intent == expected


def test_pola_menyerah_pada_kasus_tidak_tegas():
    """
    Pola yang memaksakan jawaban untuk semua input akan salah diam-diam.
    Lebih baik mengembalikan None dan menyerahkannya ke LLM.
    """
    intent, _ = classify_by_pattern("BBRI gimana ya")

    assert intent is None


def test_pertanyaan_kosong_di_luar_cakupan():
    intent, _ = classify_by_pattern("")

    assert intent == INTENT_OUT_OF_SCOPE


def test_perbandingan_tanpa_kata_waktu_tetap_live_compare():
    """
    "sudah mencapai target belum" tidak memuat kata "sekarang", tetapi
    mustahil dijawab tanpa harga terkini.
    """
    intent, _ = classify_by_pattern("BUMI sudah tembus target?")

    assert intent == INTENT_LIVE_COMPARE


# ============================================================
# KEGAGALAN LLM
# ============================================================

class LLMGagal:
    def invoke(self, prompt):
        raise RuntimeError("API down")


class LLMNgawur:
    def invoke(self, prompt):
        class R:
            content = '{"intent": "ENTAH_APA"}'

        return R()


def test_llm_gagal_jatuh_ke_rag_bukan_menolak():
    """
    Router yang rusak tidak boleh menjadi alasan pengguna ditolak. RAG
    adalah perilaku sistem ini sebelum router ada, jadi itu default yang
    aman.
    """
    hasil = classify_router_intent("BBRI gimana ya", llm=LLMGagal())

    assert hasil["intent"] == INTENT_RAG


def test_intent_tak_dikenal_jatuh_ke_rag():
    hasil = classify_router_intent("BBRI gimana ya", llm=LLMNgawur())

    assert hasil["intent"] == INTENT_RAG


def test_pola_tidak_memanggil_llm():
    hitung = {"n": 0}

    class LLMHitung:
        def invoke(self, prompt):
            hitung["n"] += 1
            raise AssertionError("LLM tidak boleh dipanggil")

    hasil = classify_router_intent(
        "Harga BBRI sekarang berapa?",
        llm=LLMHitung(),
    )

    assert hasil["llm_calls"] == 0
    assert hitung["n"] == 0


# ============================================================
# KEBUTUHAN SUMBER
# ============================================================

def test_peta_kebutuhan_sumber():
    assert butuh_harga(INTENT_LIVE_PRICE)
    assert butuh_harga(INTENT_LIVE_COMPARE)
    assert not butuh_harga(INTENT_RAG)

    assert butuh_dokumen(INTENT_RAG)
    assert butuh_dokumen(INTENT_LIVE_COMPARE)
    assert not butuh_dokumen(INTENT_LIVE_PRICE)


# ============================================================
# AKURASI TERHADAP DATASET
# ============================================================

def test_dataset_router_ada_dan_berlabel_lengkap():
    items = muat_dataset()

    assert len(items) >= 20

    for item in items:
        assert item["expected_intent"] in {
            INTENT_RAG,
            INTENT_LIVE_PRICE,
            INTENT_LIVE_COMPARE,
            INTENT_OUT_OF_SCOPE,
        }, item["id"]


def test_akurasi_router_hanya_dengan_pola():
    """
    Angka ini yang harus dipantau saat pola diubah. Ambangnya sengaja
    tidak 100%: sebagian baris dataset memang dirancang tidak tegas dan
    seharusnya diserahkan ke LLM.
    """
    items = muat_dataset()

    benar = 0
    diserahkan_ke_llm = 0
    salah = []

    for item in items:
        intent, _ = classify_by_pattern(item["question"])

        if intent is None:
            diserahkan_ke_llm += 1
            continue

        if intent == item["expected_intent"]:
            benar += 1
        else:
            salah.append((item["id"], item["expected_intent"], intent))

    diputuskan = len(items) - diserahkan_ke_llm
    akurasi = benar / diputuskan if diputuskan else 0.0

    print(
        f"\nrouter (pola saja): {benar}/{diputuskan} benar "
        f"({akurasi:.1%}), {diserahkan_ke_llm} diserahkan ke LLM"
    )

    for id_, diharapkan, didapat in salah:
        print(f"  SALAH {id_}: harap {diharapkan}, dapat {didapat}")

    assert akurasi >= 0.85, salah


def test_tidak_ada_pertanyaan_sah_ditolak_router():
    """
    False refusal di router lebih berbahaya daripada di lapisan lain,
    karena terjadi sebelum dokumen apa pun dilihat: tidak ada bukti yang
    pernah diperiksa sebelum pengguna ditolak.
    """
    items = muat_dataset()

    salah_tolak = []

    for item in items:
        if item["expected_intent"] == INTENT_OUT_OF_SCOPE:
            continue

        intent, _ = classify_by_pattern(item["question"])

        if intent == INTENT_OUT_OF_SCOPE:
            salah_tolak.append(item["id"])

    assert salah_tolak == [], salah_tolak
