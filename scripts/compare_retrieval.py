"""Measure an offline neighboring-chunk candidate; never change app retrieval.

    python -m scripts.compare_retrieval

Same questions/gold, same final k=8. Keep four similarity seeds, then add
each seed's preceding chunk on the same page (next chunk at page start),
deduplicate, and fill unused slots from the original ranking. No keywords
or gold labels are used to select documents. The candidate is an experiment,
not a production retriever. Also report the existing session retriever with
an empty fresh owner, since the CLI similarity baseline differs from the UI.
"""

import json
from unittest.mock import patch

from src import config
from src.evaluation import evaluate_retrieval, load_evaluation_dataset
from src.preprocessing import load_and_split_documents
from src.retriever import get_session_retriever, retrieve_documents
from src.session_documents import new_owner_id


def neighbor_candidate(documents, chunks, k=8):
    pages = {}
    for chunk in chunks:
        key = (chunk.metadata["source"], chunk.metadata["page"])
        pages.setdefault(key, []).append(chunk)
    selected = list(documents[: k // 2])
    seen = {d.metadata["chunk_id"] for d in selected}
    for seed in list(selected):
        page = pages[(seed.metadata["source"], seed.metadata["page"])]
        index = next(i for i, d in enumerate(page)
                     if d.metadata["chunk_id"] == seed.metadata["chunk_id"])
        neighbor_index = index - 1 if index else 1
        if neighbor_index >= len(page):
            continue
        neighbor = page[neighbor_index]
        if neighbor.metadata["chunk_id"] not in seen:
            selected.append(neighbor)
            seen.add(neighbor.metadata["chunk_id"])
    for document in documents:
        if document.metadata["chunk_id"] not in seen:
            selected.append(document)
            seen.add(document.metadata["chunk_id"])
    return selected[:k]


def main():
    dataset = load_evaluation_dataset()
    chunks = load_and_split_documents()
    # Cache baseline retrieval once; candidate selection uses no extra API.
    rankings = {q["question"]: retrieve_documents(q["question"], k=8) for q in dataset}
    with patch("src.evaluation.retrieve_documents", lambda q, k: rankings[q][:k]):
        baseline = evaluate_retrieval(dataset)
    with patch("src.evaluation.retrieve_documents",
               lambda q, k: neighbor_candidate(rankings[q], chunks, k)):
        candidate = evaluate_retrieval(dataset)
    session = get_session_retriever(new_owner_id())
    with patch("src.evaluation.retrieve_documents", lambda q, k: session.invoke(q)[:k]):
        app = evaluate_retrieval(dataset)
    assert baseline["dataset_fingerprint"] == candidate["dataset_fingerprint"] == app["dataset_fingerprint"]
    before = {r["id"]: r for r in baseline["details"]}
    gained, lost = [], []
    for row in candidate["details"]:
        if row["chunk_level"] is None:
            continue
        change = row["chunk_level"]["hit@8"] - before[row["id"]]["chunk_level"]["hit@8"]
        if change > 0:
            gained.append(row["id"])
        elif change < 0:
            lost.append(row["id"])
    output = {
        "candidate_description": __doc__,
        "dataset_fingerprint": baseline["dataset_fingerprint"],
        "gained_hit_at_8": gained, "lost_hit_at_8": lost,
        "baseline": baseline, "candidate": candidate, "existing_session": app,
    }
    path = config.EVAL_RESULTS_DIR / "retrieval_experiment.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for label, result in (("similarity", baseline), ("neighbor candidate", candidate), ("existing session", app)):
        print(label, result["chunk_level"])
    print("Gained:", gained, "Lost:", lost)
    print("Saved:", path)


if __name__ == "__main__":
    main()
