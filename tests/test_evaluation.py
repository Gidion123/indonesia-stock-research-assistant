import pytest

from src import config
from src.evaluation import (
    load_evaluation_dataset,
    load_expected_keywords,
)


# ============================================================
# DATASET SHAPE
# ============================================================

def test_evaluation_dataset_can_be_loaded():
    assert len(load_evaluation_dataset()) == 33


def test_dataset_split_between_in_scope_and_out_of_scope():
    dataset = load_evaluation_dataset()

    in_scope = [i for i in dataset if i["type"] == "in_scope"]
    out_of_scope = [i for i in dataset if i["type"] == "out_of_scope"]

    assert len(in_scope) == 28
    assert len(out_of_scope) == 5


def test_in_scope_questions_have_expected_sources():
    for item in load_evaluation_dataset():
        if item["type"] == "in_scope":
            assert len(item["expected_sources"]) > 0


def test_out_of_scope_questions_have_no_expected_sources():
    for item in load_evaluation_dataset():
        if item["type"] == "out_of_scope":
            assert item["expected_sources"] == []


def test_both_datasets_share_the_same_ids():
    """
    Gold chunks are derived by joining the two datasets on id, so the ids
    must line up exactly.
    """
    retrieval_ids = [item["id"] for item in load_evaluation_dataset()]
    keyword_ids = list(load_expected_keywords().keys())

    assert retrieval_ids == keyword_ids


def test_in_scope_questions_have_keywords():
    keywords = load_expected_keywords()

    for item in load_evaluation_dataset():
        if item["type"] == "in_scope":
            assert keywords[item["id"]], (
                f"{item['id']} tidak punya expected_keywords, sehingga "
                f"gold chunk tidak bisa diturunkan."
            )


# ============================================================
# END TO END (butuh database)
# ============================================================

@pytest.mark.db
@pytest.mark.model
def test_evaluate_retrieval_reports_both_layers():
    from src.evaluation import evaluate_retrieval

    results = evaluate_retrieval(load_evaluation_dataset()[:3])

    assert "chunk_level" in results
    assert "legacy_source_level" in results
    assert "gold_audit" in results
    assert results["config"]["retriever_k"] == config.RETRIEVER_K
