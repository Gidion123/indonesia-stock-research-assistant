"""
Unit tests for the answer-evaluation scoring functions.

These run without a database and without an LLM, which is the point:
the scoring logic is exactly the part that must be correct before any
number it produces can be trusted.
"""

import json

import pytest
from langchain_core.documents import Document

from src.answer_evaluation import (
    citation_accuracy,
    contains_citation,
    contains_expected_keyword,
    extract_numbers,
    is_refusal,
    keyword_matches,
    matched_keywords,
    summarise_answers,
)


@pytest.mark.parametrize("end_to_end", [False, True])
def test_saved_answer_report_identifies_dataset_and_scoring_keywords(
    monkeypatch, tmp_path, end_to_end,
):
    from scripts import evaluate_answers as script
    from src import gold_chunks
    from src.evaluation import dataset_fingerprint

    dataset = [{"id": "sample", "question": "Apa risikonya?",
                "type": "in_scope", "expected_keywords": ["pengeboran"]}]
    gold = {"sample": {"gold_chunk_ids": ["abc"]}}
    monkeypatch.setattr(script, "load_answer_evaluation_dataset", lambda _: dataset)
    monkeypatch.setattr(gold_chunks, "load_gold_chunks", lambda: gold)
    monkeypatch.setattr(script, "_pasang_penghitung_huggingface", lambda: [])
    monkeypatch.setattr(script, "evaluate_answers", lambda *a, **kw: {
        "answer_rate": 0.0, "citation_rate": 0.0, "false_refusal_rate": 0.0,
        "out_of_scope_refusal_rate": 0.0, "details": [],
    })
    output = tmp_path / "answers.json"
    monkeypatch.setattr(script, "RESULT_PATH", str(output))
    monkeypatch.setattr(script, "RESULT_PATH_END_TO_END", str(output))

    script.run_evaluation(end_to_end=end_to_end)
    first = json.loads(output.read_text())
    assert first["dataset_fingerprint"] == dataset_fingerprint(dataset, gold)
    assert first["mode"] == ("end_to_end" if end_to_end else "rag_only")
    assert first["retrieval_mode"] == "similarity"

    # Retrieval labels can stay identical while answer scoring changes.
    # The extra digest must expose that difference in either output mode.
    dataset[0]["expected_keywords"].append("PLN")
    script.run_evaluation(end_to_end=end_to_end)
    second = json.loads(output.read_text())
    assert second["dataset_fingerprint"] == first["dataset_fingerprint"]
    assert second["question_dataset_sha256"] != first["question_dataset_sha256"]


# ============================================================
# NUMBERS
# ============================================================

def test_extract_numbers_handles_indonesian_format():
    assert extract_numbers("Rp3.400") == [3400.0]
    assert extract_numbers("13,75%") == [13.75]
    assert extract_numbers("9.134,70") == [9134.70]
    assert extract_numbers("51.50") == [51.50]


def test_numeric_keyword_tolerates_formatting():
    assert keyword_matches("Target TP 1 berada di Rp 3.400,00", "3.400")
    assert keyword_matches("Target TP 1 berada di 3400", "3.400")


def test_numeric_keyword_rejects_substring_coincidence():
    """
    The old substring matcher scored '1,0' as found inside '31,08'.
    """
    assert not keyword_matches("Rasio tercatat 31,08 dan 1,05", "1,0")
    assert keyword_matches("Beban bunga menyentuh rasio 1,0x", "1,0")


def test_text_keyword_still_matches_as_substring():
    assert keyword_matches("Sektor perbankan diuntungkan", "perbankan")
    assert not keyword_matches("Sektor energi diuntungkan", "perbankan")


# ============================================================
# KEYWORD AGGREGATION
# ============================================================

def test_all_keywords_are_required():
    """
    The old implementation used any(): one keyword out of three was
    enough to count the answer as correct.
    """
    answer = "ANTM berperan sebagai aset defensif."

    assert matched_keywords(answer, ["ANTM", "defensif", "hedge"]) == [
        "ANTM",
        "defensif",
    ]
    assert not contains_expected_keyword(
        answer, ["ANTM", "defensif", "hedge"]
    )
    assert contains_expected_keyword(answer, ["ANTM", "defensif"])


def test_empty_keywords_do_not_count_as_correct():
    assert not contains_expected_keyword("apa pun", [])


# ============================================================
# CITATION
# ============================================================

DOCUMENTS = [
    Document(
        page_content="x",
        metadata={"source": "Analisis IHSG 2026 & Saham Unggulan.pdf",
                  "page": 12},
    ),
    Document(
        page_content="y",
        metadata={"source": "Analisis Mendalam Saham Barito Group.pdf",
                  "page": 3},
    ),
]


def test_contains_citation_detects_format():
    assert contains_citation("TP1 Rp3.400 [Analisis IHSG 2026 & "
                             "Saham Unggulan.pdf hal.12].")
    assert not contains_citation("TP1 Rp3.400 (hal. 12).")


def test_citation_accuracy_accepts_valid_pair():
    answer = (
        "TP1 Rp3.400 [Analisis IHSG 2026 & Saham Unggulan.pdf hal.12]."
    )

    assert citation_accuracy(answer, DOCUMENTS) == 1.0


def test_citation_accuracy_rejects_invented_page():
    """
    A citation whose page was never in the context is the failure mode
    the presence-only check could not see.
    """
    answer = (
        "TP1 Rp3.400 [Analisis IHSG 2026 & Saham Unggulan.pdf hal.99]."
    )

    assert citation_accuracy(answer, DOCUMENTS) == 0.0


def test_citation_accuracy_is_partial_when_mixed():
    answer = (
        "A [Analisis IHSG 2026 & Saham Unggulan.pdf hal.12] dan "
        "B [Analisis Mendalam Saham Barito Group.pdf hal.99]."
    )

    assert citation_accuracy(answer, DOCUMENTS) == 0.5


def test_citation_accuracy_is_none_without_citation():
    assert citation_accuracy("tanpa sitasi", DOCUMENTS) is None


# ============================================================
# REFUSAL
# ============================================================

def test_refusal_detection_tolerates_paraphrase():
    assert is_refusal("Maaf, informasi itu tidak ada di dokumen saya.")
    assert is_refusal("Informasi tersebut tidak tersedia dalam konteks.")
    assert is_refusal("Saya tidak dapat menjawab pertanyaan tersebut.")
    assert not is_refusal("TP1 BBRI adalah Rp3.400 [a.pdf hal.12].")


# ============================================================
# AGGREGATION
# ============================================================

def _row(**kwargs):
    base = {
        "id": "x",
        "question": "q",
        "type": "in_scope",
        "status": "ok",
        "error": None,
        "latency_seconds": 1.0,
        "answer": "a",
        "retrieved_chunk_ids": [],
        "expected_keywords": [],
        "matched_keywords": [],
        "keyword_match": True,
        "citation": True,
        "citation_accuracy": 1.0,
        "refusal": False,
    }
    base.update(kwargs)
    return base


def test_failed_rows_are_excluded_not_scored_as_wrong():
    """
    An API failure must never be counted as a wrong answer. That single
    confusion is what produced the misleading baseline in the earlier
    version of this project.
    """
    results = [
        _row(id="a"),
        _row(
            id="b",
            status="transient_error",
            answer=None,
            keyword_match=None,
            citation=None,
            citation_accuracy=None,
            refusal=None,
        ),
    ]

    summary = summarise_answers(results)

    assert summary["failed_rows"] == 1
    assert summary["in_scope_questions"] == 1
    assert summary["answer_rate"] == 1.0


def test_run_is_invalid_when_too_many_rows_fail():
    results = [_row(id=str(index)) for index in range(9)]
    results.append(
        _row(
            id="bad",
            status="error",
            keyword_match=None,
            citation=None,
            citation_accuracy=None,
            refusal=None,
        )
    )

    summary = summarise_answers(results)

    assert summary["failed_ratio"] == 0.1
    assert summary["run_is_valid"] is False


def test_out_of_scope_refusal_is_counted_separately():
    results = [
        _row(id="a", type="in_scope", refusal=False, keyword_match=True),
        _row(id="b", type="out_of_scope", refusal=True, keyword_match=False),
    ]

    summary = summarise_answers(results)

    assert summary["out_of_scope_refusal_rate"] == 1.0
    assert summary["false_refusal_rate"] == 0.0
