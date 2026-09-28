"""
Answer quality evaluation.

    python -m scripts.evaluate_answers                # RAG only (baseline)
    python -m scripts.evaluate_answers --end-to-end   # via router + resolver

Two modes, on purpose:

RAG only (default)
    Calls `ask_question()` directly. This is the historical baseline for
    the project. Compare runs only when questions, gold annotations,
    and answer-scoring keywords are unchanged.

End-to-end
    Calls `assistant.jawab()` with its default similarity retriever,
    without the UI's session retriever or conversation memory. The
    difference shows on out-of-scope questions: "Berapa harga emas dunia
    hari ini?" is forced to be answered from the documents in RAG-only
    mode, while end-to-end the resolver stops it before a single
    document is read.
"""

import argparse
import hashlib
import json
from pathlib import Path

from src import config
from src.answer_evaluation import (
    evaluate_answers,
    load_answer_evaluation_dataset,
)


# One source of questions for every evaluator. See the comment in
# src/config.py: two separate dataset files once drifted apart and made
# two scripts report different numbers for the same system.
DATASET_PATH = config.EVAL_QUESTIONS_PATH

# One file per mode. Both used to write to the same file, so running
# --end-to-end overwrote the RAG-only baseline, and the number it was
# meant to be compared against survived only on screen.
RESULT_PATH = "evaluation/results/answer_results.json"
RESULT_PATH_END_TO_END = "evaluation/results/answer_results_end_to_end.json"


def _ask_end_to_end(question):
    """
    Adapt `assistant.jawab()` to the shape evaluate_answers expects.

    This is not the same shape as `ask_question()`. It carries the
    subset the evaluator actually reads - answer, status, documents,
    chunk_ids, error, latency_seconds - and leaves out `question` and
    `context`, which no metric looks at. It then adds two keys
    `ask_question()` does not return at all, `intent` and `trace`.
    """
    from src.assistant import jawab

    hasil = jawab(question)

    documents = hasil.get("documents") or []

    return {
        "answer": hasil.get("answer"),
        "status": "ok" if hasil.get("status") in {
            "ok",
            "out_of_scope",
            "unresolved_entity",
            "ambiguous",
            "no_research_coverage",
        } else hasil.get("status", "ok"),
        "documents": documents,
        "chunk_ids": [
            d.metadata.get("chunk_id")
            for d in documents
            if getattr(d, "metadata", None)
        ],
        "error": hasil.get("error"),
        "latency_seconds": hasil.get("latency_seconds"),
        # Passed through so the report can answer "why was this one not
        # answered" and "what did this run cost". Without them a routing
        # mistake only shows up as a lower Answer Rate.
        "intent": hasil.get("intent"),
        "trace": hasil.get("trace"),
    }


def run_evaluation(end_to_end=False):
    """
    Run answer evaluation and save the results.
    """

    print(
        "Memuat answer evaluation dataset..."
    )

    dataset = load_answer_evaluation_dataset(
        DATASET_PATH
    )

    print(
        f"Jumlah pertanyaan: {len(dataset)}"
    )

    print(
        "Menjalankan answer evaluation "
        + ("(end-to-end: router + resolver)..." if end_to_end
           else "(RAG saja — baseline)...")
    )

    # The network counter goes on BEFORE the model is built, so that "no
    # requests to Hugging Face" can be proven instead of claimed. See
    # scripts/diagnose_embeddings.py for the detailed version.
    permintaan_hf = _pasang_penghitung_huggingface()

    from src.market_data import (
        reset_statistik_market_data,
        statistik_market_data,
    )

    reset_statistik_market_data()

    results = evaluate_answers(
        dataset,
        ask=_ask_end_to_end if end_to_end else None,
    )

    results["mode"] = "end_to_end" if end_to_end else "rag_only"
    # Answer reports must identify the same questions/gold as retrieval.
    # Also hash expected keywords: answer scoring can change without a
    # change to the retrieval gold IDs.
    from src.evaluation import dataset_fingerprint
    from src.gold_chunks import load_gold_chunks

    results["dataset_fingerprint"] = dataset_fingerprint(dataset, load_gold_chunks())
    results["question_dataset_sha256"] = hashlib.sha256(
        json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    results["retrieval_mode"] = "similarity"

    from src import embeddings as _emb

    results["embedding_source"] = _emb.sumber_model()
    results["huggingface_requests"] = len(permintaan_hf)
    results["yahoo_requests"] = statistik_market_data()

    result_path = Path(
        RESULT_PATH_END_TO_END if end_to_end else RESULT_PATH
    )

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=== RAG Answer Evaluation ==="
    )

    print(
        f"Answer Rate             : "
        f"{results['answer_rate']:.4f}"
    )

    print(
        f"Citation Rate           : "
        f"{results['citation_rate']:.4f}"
    )

    print(
        f"False Refusal Rate      : "
        f"{results['false_refusal_rate']:.4f}"
    )

    print(
        f"Out-of-Scope Refusal    : "
        f"{results['out_of_scope_refusal_rate']:.4f}"
    )

    _cetak_biaya(results)
    _cetak_baris_bermasalah(results)

    print()
    print(f"Hasil disimpan ke: {result_path}")


def _pasang_penghitung_huggingface():
    """
    Record every HTTP request to huggingface.co during the run.

    With a warm model cache the count has to be 0. If it is not, the
    model was pulled over the network again and the latency of the first
    question does not reflect the system.
    """
    import requests.sessions

    tercatat = []
    asli = requests.sessions.Session.request

    def terbungkus(self, method, url, *args, **kwargs):
        if "huggingface.co" in str(url) or "hf.co" in str(url):
            tercatat.append((method, str(url)))

        return asli(self, method, url, *args, **kwargs)

    requests.sessions.Session.request = terbungkus

    return tercatat


def _cetak_biaya(results):
    print()
    print("--- biaya & jaringan ---")
    print(f"Sumber model embedding  : {results.get('embedding_source')}")
    print(f"Permintaan Hugging Face : {results.get('huggingface_requests')}")

    yahoo = results.get("yahoo_requests") or {}
    print(
        f"Permintaan Yahoo        : {yahoo.get('requests', 0)} "
        f"(cache hit: {yahoo.get('cache_hits', 0)})"
    )

    llm = results.get("llm_calls")

    if llm:
        print(
            f"Panggilan LLM           : {llm['total']} "
            f"(router {llm['router']}, resolver {llm['resolution']}, "
            f"jawaban {llm['answer']})"
        )

    if results.get("intents"):
        print(f"Intent                  : {results['intents']}")


def _cetak_baris_bermasalah(results):
    """
    Aggregate numbers do not say which question broke. This does -
    including the intent the router picked, because the commonest reason
    an in-scope question goes unanswered is that it never reached the
    RAG path at all.
    """
    bermasalah = [
        row
        for row in results["details"]
        if row["type"] == "in_scope"
        and (row.get("keyword_match") is False or row.get("refusal"))
    ]

    if not bermasalah:
        return

    print()
    print(f"--- soal in-scope bermasalah: {len(bermasalah)} ---")

    for row in bermasalah:
        penanda = []

        if row.get("refusal"):
            penanda.append("DITOLAK")

        if row.get("keyword_match") is False:
            penanda.append("keyword tidak lengkap")

        if not row.get("citation"):
            penanda.append("tanpa sitasi")

        print(
            f"  {row['id']:<14} intent={str(row.get('intent')):<13} "
            f"chunk={len(row.get('retrieved_chunk_ids') or []):<3} "
            f"{', '.join(penanda)}"
        )
        print(
            f"      cocok {row.get('matched_keywords')} "
            f"dari {row.get('expected_keywords')}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help=(
            "nilai lewat router + entity resolver, dengan retriever similarity "
            "default tanpa sesi UI, bukan RAG saja"
        ),
    )
    args = parser.parse_args()

    run_evaluation(end_to_end=args.end_to_end)


if __name__ == "__main__":
    main()
