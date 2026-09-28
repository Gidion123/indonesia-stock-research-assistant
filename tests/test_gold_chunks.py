"""
Tests for gold chunk derivation and the chunk-level retrieval metrics.

No database and no LLM: the retriever is replaced by a list of chunk ids,
which is all the metric functions actually need.
"""

import json

from langchain_core.documents import Document

from src.evaluation import chunk_metrics, source_metrics
from src.gold_chunks import (
    GOLD_MODE_NONE,
    GOLD_MODE_RELAXED,
    GOLD_MODE_STRICT,
    audit_gold_chunks,
    build_gold_chunks,
    derive_gold_for_item,
    load_gold_chunks,
    save_gold_chunks,
)
from src.preprocessing import split_documents


def _chunks():
    documents = [
        Document(
            page_content="Target TP 1 BBRI berada di Rp3.400 dan TP 2 Rp3.800.",
            metadata={"source": "ihsg.pdf", "page": 12},
        ),
        Document(
            page_content="IHSG mengalami koreksi pada awal 2026.",
            metadata={"source": "ihsg.pdf", "page": 1},
        ),
        Document(
            page_content="BREN menunjukkan margin EBITDA yang tinggi.",
            metadata={"source": "barito.pdf", "page": 5},
        ),
    ]

    return split_documents(documents)


# ============================================================
# DERIVATION
# ============================================================

def test_strict_mode_requires_all_keywords_in_one_chunk():
    chunks = _chunks()

    entry = derive_gold_for_item(
        chunks,
        {"id": "q1", "expected_sources": ["ihsg.pdf"]},
        ["TP 1", "Rp3.400"],
    )

    assert entry["gold_mode"] == GOLD_MODE_STRICT
    assert len(entry["gold_chunk_ids"]) == 1


def test_relaxed_mode_when_keywords_are_spread_out():
    chunks = _chunks()

    entry = derive_gold_for_item(
        chunks,
        {"id": "q2", "expected_sources": ["ihsg.pdf"]},
        ["Rp3.400", "koreksi"],
    )

    assert entry["gold_mode"] == GOLD_MODE_RELAXED
    assert len(entry["gold_chunk_ids"]) == 2


def test_no_gold_when_keyword_is_absent():
    chunks = _chunks()

    entry = derive_gold_for_item(
        chunks,
        {"id": "q3", "expected_sources": ["ihsg.pdf"]},
        ["kata-yang-tidak-ada"],
    )

    assert entry["gold_mode"] == GOLD_MODE_NONE
    assert entry["gold_chunk_ids"] == []


def test_gold_is_restricted_to_expected_sources():
    chunks = _chunks()

    entry = derive_gold_for_item(
        chunks,
        {"id": "q4", "expected_sources": ["ihsg.pdf"]},
        ["BREN"],
    )

    assert entry["gold_chunk_ids"] == []


def test_out_of_scope_items_get_no_gold():
    chunks = _chunks()

    gold = build_gold_chunks(
        [{"id": "oos_01", "type": "out_of_scope", "expected_sources": []}],
        {"oos_01": []},
        chunks=chunks,
    )

    assert gold == {}


# ============================================================
# AUDIT
# ============================================================

def test_audit_flags_saturated_items():
    gold = {
        "q1": {
            "gold_chunk_ids": [str(index) for index in range(40)],
            "gold_mode": GOLD_MODE_STRICT,
            "keywords": ["IHSG"],
        },
        "q2": {
            "gold_chunk_ids": ["a"],
            "gold_mode": GOLD_MODE_STRICT,
            "keywords": ["Rp3.400"],
        },
    }

    audit = audit_gold_chunks(gold)

    assert audit["total_items"] == 2
    assert audit["max_gold_chunks"] == 40
    assert audit["saturated_items"][0][0] == "q1"


def test_gold_annotation_round_trips(tmp_path):
    gold = {
        "q1": {
            "gold_chunk_ids": ["abc123"],
            "gold_mode": GOLD_MODE_STRICT,
            "keywords": ["TP 1"],
        }
    }

    path = tmp_path / "gold_chunks.json"
    save_gold_chunks(gold, file_path=path)

    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    assert load_gold_chunks(file_path=path) == gold


def test_load_gold_chunks_returns_empty_when_missing(tmp_path):
    assert load_gold_chunks(file_path=tmp_path / "nope.json") == {}


# ============================================================
# METRICS
# ============================================================

def test_chunk_metrics_rewards_early_hit():
    metrics = chunk_metrics(
        retrieved_chunk_ids=["a", "b", "c", "d"],
        gold_chunk_ids=["a"],
        ks=(1, 3),
    )

    assert metrics["hit@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["precision@1"] == 1.0


def test_chunk_metrics_penalises_late_hit():
    metrics = chunk_metrics(
        retrieved_chunk_ids=["x", "y", "a"],
        gold_chunk_ids=["a"],
        ks=(1, 3),
    )

    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert round(metrics["mrr"], 4) == round(1 / 3, 4)


def test_chunk_metrics_can_fail_completely():
    """
    The property the source-level metric lacked: it must be possible
    to score zero.
    """
    metrics = chunk_metrics(
        retrieved_chunk_ids=["x", "y", "z"],
        gold_chunk_ids=["a", "b"],
        ks=(1, 3),
    )

    assert metrics["hit@3"] == 0.0
    assert metrics["recall@3"] == 0.0
    assert metrics["mrr"] == 0.0


def test_chunk_metrics_partial_recall():
    metrics = chunk_metrics(
        retrieved_chunk_ids=["a", "x"],
        gold_chunk_ids=["a", "b"],
        ks=(2,),
    )

    assert metrics["recall@2"] == 0.5
    assert metrics["hit@2"] == 1.0


def test_chunk_metrics_returns_none_without_gold():
    assert chunk_metrics(["a"], [], ks=(1,)) is None


def test_source_metrics_still_works_for_comparison():
    metrics = source_metrics(
        ["a.pdf", "b.pdf"],
        ["b.pdf"],
    )

    assert metrics["hit"] == 1.0
    assert metrics["reciprocal_rank"] == 0.5
