"""
Retrieval evaluation.

Reports two layers side by side. The chunk level - Recall@k, HitRate@k,
MRR and Precision@k measured against gold chunks - is the one that can
actually fail, so it is the one that says whether retrieval works.

The source level is the original metric. It is kept only so the new
numbers can be lined up against earlier runs. With four documents and
k=8 it scores about 1.0 by construction, so it is reported under
`legacy_source_level` and should not be quoted as a result.
"""

import hashlib
import json

from src import config
from src.gold_chunks import (
    audit_gold_chunks,
    build_gold_chunks,
    load_gold_chunks,
    save_gold_chunks,
)
from src.retriever import retrieve_documents


def load_evaluation_dataset(file_path=None):
    """
    Load the evaluation dataset.
    """
    path = config.EVAL_QUESTIONS_PATH if file_path is None else file_path

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_expected_keywords(file_path=None):
    """
    Keywords for deriving gold chunks, keyed by question id.

    `expected_keywords` answers "what must appear in the generated
    answer". Finding the passage that holds the answer is a different
    job, and for some questions the two disagree. equity_03 expects both
    "stripping ratio" and "Rp172"; four chunks discuss the stripping
    ratio, but only one of them also prints that number, so requiring
    both would shrink gold from those four chunks down to that single
    one.

    `gold_keywords` is the override for those questions. equity_03 is
    the only item that sets it, to ["stripping ratio"] alone. When it is
    absent, which is the normal case, the two jobs agree and
    `expected_keywords` is used.
    """
    path = config.EVAL_QUESTIONS_PATH if file_path is None else file_path

    with open(path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return {
        item["id"]: (
            item.get("gold_keywords")
            or item.get("expected_keywords", [])
        )
        for item in dataset
    }


# ============================================================
# METRICS
# ============================================================

def chunk_metrics(retrieved_chunk_ids, gold_chunk_ids, ks):
    """
    Recall@k, HitRate@k, MRR and Precision@k for one question.
    """
    gold = set(gold_chunk_ids)

    if not gold:
        return None

    metrics = {}

    for k in ks:
        top_k = retrieved_chunk_ids[:k]
        hits = [chunk_id for chunk_id in top_k if chunk_id in gold]

        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
        metrics[f"recall@{k}"] = len(set(hits)) / len(gold)
        metrics[f"precision@{k}"] = len(hits) / k if k else 0.0

    reciprocal_rank = 0.0

    for position, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold:
            reciprocal_rank = 1.0 / position
            break

    metrics["mrr"] = reciprocal_rank

    return metrics


def source_metrics(retrieved_sources, expected_sources):
    """
    The original source-level metric, kept for comparison only.
    """
    expected = set(expected_sources)

    positions = [
        index
        for index, source in enumerate(retrieved_sources, start=1)
        if source in expected
    ]

    return {
        "hit": 1.0 if positions else 0.0,
        "recall": 1.0 if positions else 0.0,
        "reciprocal_rank": 1.0 / positions[0] if positions else 0.0,
    }


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def dataset_fingerprint(dataset, gold):
    """
    Identify WHAT was measured, not the result.

    Two HitRate numbers may only be compared when the questions and the
    gold chunks behind them are the same. This fingerprint makes that
    condition machine-checkable, so a fix to the dataset can never be
    read as a fix to the system.
    """
    bahan = []

    for item in sorted(dataset, key=lambda i: i["id"]):
        entry = gold.get(item["id"], {})
        bahan.append(
            "|".join(
                [
                    item["id"],
                    item.get("question", ""),
                    item.get("type", ""),
                    ",".join(sorted(entry.get("gold_chunk_ids", []))),
                ]
            )
        )

    teks = "\n".join(bahan).encode("utf-8")

    return hashlib.sha1(teks).hexdigest()[:12]


# ============================================================
# RUNNER
# ============================================================

def evaluate_retrieval(
    dataset,
    k=None,
    gold=None,
    rebuild_gold=False,
    ks=None,
):
    """
    Evaluate retrieval at chunk level (primary) and source level (legacy).
    """
    k = k or config.RETRIEVER_K
    ks = tuple(ks or config.EVALUATION_K_VALUES)

    if gold is None:
        gold = load_gold_chunks()

    if rebuild_gold or not gold:
        gold = build_gold_chunks(dataset, load_expected_keywords())
        save_gold_chunks(gold)

    gold_audit = audit_gold_chunks(gold)

    results = []

    for item in dataset:
        question = item["question"]

        documents = retrieve_documents(question, k=k)

        retrieved_chunk_ids = [
            document.metadata.get("chunk_id")
            for document in documents
        ]

        retrieved_sources = [
            document.metadata.get("source", "")
            for document in documents
        ]

        entry = gold.get(item["id"], {})

        row = {
            "id": item["id"],
            "question": question,
            "type": item["type"],
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_sources": retrieved_sources,
            "expected_sources": list(item.get("expected_sources") or []),
            "gold_chunk_ids": entry.get("gold_chunk_ids", []),
            "gold_mode": entry.get("gold_mode", "none"),
        }

        row["source_level"] = source_metrics(
            retrieved_sources,
            row["expected_sources"],
        )

        row["chunk_level"] = chunk_metrics(
            retrieved_chunk_ids,
            row["gold_chunk_ids"],
            ks,
        )

        results.append(row)

    in_scope = [row for row in results if row["type"] == "in_scope"]

    scored = [row for row in in_scope if row["chunk_level"] is not None]

    chunk_summary = {}

    for cut_off in ks:
        chunk_summary[f"hit_rate@{cut_off}"] = _mean(
            [row["chunk_level"][f"hit@{cut_off}"] for row in scored]
        )
        chunk_summary[f"recall@{cut_off}"] = _mean(
            [row["chunk_level"][f"recall@{cut_off}"] for row in scored]
        )
        chunk_summary[f"precision@{cut_off}"] = _mean(
            [row["chunk_level"][f"precision@{cut_off}"] for row in scored]
        )

    chunk_summary["mrr"] = _mean(
        [row["chunk_level"]["mrr"] for row in scored]
    )

    legacy = {
        "hit_rate_at_k": _mean(
            [row["source_level"]["hit"] for row in in_scope]
        ),
        "recall_at_k": _mean(
            [row["source_level"]["recall"] for row in in_scope]
        ),
        "mrr": _mean(
            [row["source_level"]["reciprocal_rank"] for row in in_scope]
        ),
        "note": (
            "Metrik level sumber. Dengan 4 dokumen dan k=8 metrik ini "
            "hampir tidak bisa gagal; disimpan hanya untuk pembanding "
            "historis, bukan untuk dilaporkan sebagai hasil."
        ),
    }

    return {
        "config": config.describe(),
        "k": k,
        "ks": list(ks),
        "dataset_fingerprint": dataset_fingerprint(dataset, gold),
        "total_questions": len(results),
        "in_scope_questions": len(in_scope),
        "scored_questions": len(scored),
        "total_gold_chunks": sum(
            len(row["gold_chunk_ids"]) for row in in_scope
        ),
        "gold_audit": gold_audit,
        "chunk_level": chunk_summary,
        "legacy_source_level": legacy,
        "details": results,
    }
