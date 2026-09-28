"""
Penjaga satu-sumber-kebenaran untuk dataset evaluasi.

Dulu ada dua berkas: `retrieval_eval.json` memasok pertanyaan,
`answer_eval.json` memasok expected_keywords. Gold chunk diturunkan dari
keyword di berkas kedua, tetapi retrieval dijalankan memakai pertanyaan
dari berkas pertama. Ketika dua pertanyaan di berkas kedua dipertajam,
tidak ada satu pun test yang gagal — yang terjadi hanyalah dua skrip
melaporkan angka berbeda untuk sistem yang sama (0,80 vs 0,90), dan gold
untuk dua soal diturunkan bagi pertanyaan yang tidak pernah ditanyakan.

Test di berkas ini menjaga agar bentuk kegagalan itu tidak bisa kembali.
"""

import json

import pytest

from src import config
from src.evaluation import (
    load_evaluation_dataset,
    load_expected_keywords,
)


# ============================================================
# SATU SUMBER
# ============================================================

def test_semua_jalur_membaca_berkas_yang_sama():
    """
    Pertanyaan untuk retrieval dan keyword untuk gold harus berasal dari
    berkas yang sama. Kalau tidak, keduanya bisa menyimpang diam-diam.
    """
    assert config.RETRIEVAL_EVAL_PATH == config.EVAL_QUESTIONS_PATH
    assert config.ANSWER_EVAL_PATH == config.EVAL_QUESTIONS_PATH


@pytest.mark.parametrize(
    "nama_lama",
    ["retrieval_eval.json", "answer_eval.json"],
)
def test_dataset_lama_sudah_tidak_ada(nama_lama):
    """
    Berkas lama tidak boleh hidup kembali: selama ia ada, seseorang bisa
    mengeditnya dan mengira itu berpengaruh.
    """
    assert not (config.EVAL_DATASET_DIR / nama_lama).exists(), (
        f"{nama_lama} muncul lagi. Dataset evaluasi hanya boleh satu: "
        f"{config.EVAL_QUESTIONS_PATH.name}."
    )


# ============================================================
# BENTUK ITEM
# ============================================================

def test_id_unik():
    ids = [item["id"] for item in load_evaluation_dataset()]

    assert len(ids) == len(set(ids))


def test_setiap_item_punya_pertanyaan_tidak_kosong():
    for item in load_evaluation_dataset():
        assert item["question"].strip(), item["id"]


def test_pertanyaan_gold_dan_pertanyaan_retrieval_identik():
    """
    Inti dari perbaikan ini: satu id, satu pertanyaan. Dibaca ulang dari
    disk supaya yang diperiksa adalah berkasnya, bukan hasil cache.
    """
    with config.EVAL_QUESTIONS_PATH.open(encoding="utf-8") as file:
        mentah = json.load(file)

    dari_loader = {
        item["id"]: item["question"]
        for item in load_evaluation_dataset()
    }

    for item in mentah:
        assert dari_loader[item["id"]] == item["question"]


def test_gold_keywords_hanya_mempersempit_bukan_mengganti():
    """
    `gold_keywords` boleh memilih sebagian dari `expected_keywords` supaya
    gold jatuh pada paragraf, bukan pada potongan overlap. Ia tidak boleh
    memperkenalkan istilah baru — kalau boleh, gold dan jawaban yang
    dinilai bisa membicarakan dua hal yang berbeda.
    """
    for item in load_evaluation_dataset():
        gold_keywords = item.get("gold_keywords")

        if gold_keywords is None:
            continue

        assert gold_keywords, item["id"]

        for keyword in gold_keywords:
            assert keyword in item["expected_keywords"], (
                f"{item['id']}: gold_keyword {keyword!r} tidak ada di "
                f"expected_keywords."
            )


def test_in_scope_punya_sumber_dan_keyword():
    keywords = load_expected_keywords()

    for item in load_evaluation_dataset():
        if item["type"] != "in_scope":
            continue

        assert item["expected_sources"], item["id"]
        assert item["expected_keywords"], item["id"]
        assert keywords[item["id"]], item["id"]


def test_out_of_scope_tidak_punya_sumber_maupun_keyword():
    for item in load_evaluation_dataset():
        if item["type"] == "out_of_scope":
            assert item["expected_sources"] == []
            assert item["expected_keywords"] == []


# ============================================================
# GOLD YANG TERSIMPAN
# ============================================================

def test_setiap_pertanyaan_in_scope_punya_gold_chunk():
    from src.gold_chunks import load_gold_chunks

    gold = load_gold_chunks()

    for item in load_evaluation_dataset():
        if item["type"] != "in_scope":
            continue

        entry = gold.get(item["id"])

        assert entry and entry["gold_chunk_ids"], (
            f"{item['id']} tidak punya gold chunk. Jalankan "
            f"`python -m scripts.build_gold_chunks`."
        )


def test_gold_tersimpan_memakai_keyword_yang_berlaku_sekarang():
    """
    chunk_id bersifat content-addressed, jadi gold basi tidak terlihat
    basi — ia hanya menghasilkan HitRate yang rendah. Yang diperiksa di
    sini murah: keyword yang tercatat di gold harus sama dengan keyword
    yang dipakai loader hari ini.
    """
    from src.gold_chunks import load_gold_chunks

    gold = load_gold_chunks()
    keywords = load_expected_keywords()

    for question_id, entry in gold.items():
        assert entry["keywords"] == list(keywords[question_id]), (
            f"{question_id}: gold diturunkan dari {entry['keywords']}, "
            f"sedangkan dataset sekarang memakai {keywords[question_id]}. "
            f"Jalankan `python -m scripts.build_gold_chunks`."
        )
