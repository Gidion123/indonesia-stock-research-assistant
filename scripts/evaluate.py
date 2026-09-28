"""
Retrieval evaluation against the gold chunks.

    python -m scripts.evaluate

What gets reported is the CHUNK LEVEL metric. The source level metric is
printed as well, but labeled clearly as a historical comparison: with
four documents and k=8 its value is about 1.0 by construction, so it
cannot fail and therefore measures nothing.

The script also compares this run against the previous one - but only
when the questions and the gold chunks are identical. If the dataset or
the annotation changed, the comparison is refused outright, because a
number that goes up after gold was corrected is not evidence that the
system got better.
"""

import json
from pathlib import Path

from src import config
from src.evaluation import (
    evaluate_retrieval,
    load_evaluation_dataset,
)


RESULT_PATH = Path("evaluation/results/retrieval_results.json")


def _baca_hasil_sebelumnya():
    if not RESULT_PATH.exists():
        return None

    try:
        with RESULT_PATH.open(encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return None


def _cetak_metrik(results):
    chunk = results["chunk_level"]
    ks = results.get("ks") or list(config.EVALUATION_K_VALUES)

    print()
    print("=== RETRIEVAL EVALUATION (level chunk) ===")
    print(f"  Pertanyaan total       : {results['total_questions']}")
    print(f"  Pertanyaan in-scope    : {results['in_scope_questions']}")
    print(f"  Pertanyaan dinilai     : {results['scored_questions']}")
    print(f"  Gold chunk total       : {results['total_gold_chunks']}")

    audit = results["gold_audit"]
    print(f"  Gold mode strict       : {audit['strict']}")
    print(f"  Gold mode relaxed      : {audit['relaxed']}")
    print(f"  Tanpa gold             : {audit['without_gold']}")
    print(f"  Rata-rata gold/soal    : {audit['average_gold_chunks']}")

    print()
    print(f"  {'k':>3}  {'HitRate':>9}  {'Recall':>9}  {'Precision':>9}")

    for k in ks:
        print(
            f"  {k:>3}  "
            f"{chunk[f'hit_rate@{k}']:>9.4f}  "
            f"{chunk[f'recall@{k}']:>9.4f}  "
            f"{chunk[f'precision@{k}']:>9.4f}"
        )

    print()
    print(f"  MRR                    : {chunk['mrr']:.4f}")

    legacy = results["legacy_source_level"]
    print()
    print("  --- level sumber (pembanding historis, JANGAN dikutip) ---")
    print(f"  HitRate@{results['k']} (sumber)     : {legacy['hit_rate_at_k']:.4f}")
    print(f"  Recall@{results['k']} (sumber)      : {legacy['recall_at_k']:.4f}")
    print(f"  MRR (sumber)           : {legacy['mrr']:.4f}")


def _cetak_kegagalan(results):
    k = results["k"]

    gagal = [
        row
        for row in results["details"]
        if row["type"] == "in_scope"
        and row["chunk_level"] is not None
        and row["chunk_level"][f"hit@{k}"] == 0.0
    ]

    tanpa_gold = [
        row
        for row in results["details"]
        if row["type"] == "in_scope" and row["chunk_level"] is None
    ]

    print()
    print(f"  Gagal HitRate@{k}        : {len(gagal)}")

    for row in gagal:
        peringkat = next(
            (
                index
                for index, chunk_id in enumerate(
                    row["retrieved_chunk_ids"], start=1
                )
                if chunk_id in set(row["gold_chunk_ids"])
            ),
            None,
        )

        sebab = (
            f"gold pertama di peringkat {peringkat} (RANKING)"
            if peringkat
            else f"gold tidak ada di top-{k}"
        )

        print(f"    {row['id']:<14} {sebab}")
        print(f"      {row['question']}")

    if tanpa_gold:
        print()
        print(
            f"  TANPA GOLD (tidak dinilai): {len(tanpa_gold)} -> "
            f"{[row['id'] for row in tanpa_gold]}"
        )


def _cetak_perbandingan(results, sebelumnya):
    print()
    print("  --- perbandingan dengan run sebelumnya ---")

    if sebelumnya is None:
        print("  Tidak ada hasil sebelumnya. Ini baseline pertama.")
        return

    sidik_lama = sebelumnya.get("dataset_fingerprint")
    sidik_baru = results["dataset_fingerprint"]

    if sidik_lama != sidik_baru:
        print(
            "  TIDAK COMPARABLE.\n"
            f"    identitas dataset lama : {sidik_lama or '(tidak dicatat)'}\n"
            f"    identitas dataset baru : {sidik_baru}\n"
            "    Pertanyaan dan/atau gold chunk berbeda, jadi kedua angka\n"
            "    mengukur soal yang berbeda. Angka lama TIDAK boleh dipakai\n"
            "    sebagai pembanding; run ini adalah baseline baru."
        )
        return

    lama = sebelumnya.get("chunk_level", {})
    baru = results["chunk_level"]

    print(f"  Dataset evaluasi sama ({sidik_baru}). Perbandingan dapat dilakukan.")
    print(f"  {'metrik':<14}{'lama':>9}{'baru':>9}{'selisih':>10}")

    for nama in sorted(baru):
        if nama not in lama:
            continue

        selisih = baru[nama] - lama[nama]
        print(
            f"  {nama:<14}{lama[nama]:>9.4f}{baru[nama]:>9.4f}"
            f"{selisih:>+10.4f}"
        )


def run_evaluation():
    print("Memuat evaluation dataset...")
    print(f"  Sumber: {config.EVAL_QUESTIONS_PATH}")

    dataset = load_evaluation_dataset()

    print(f"Jumlah pertanyaan: {len(dataset)}")
    print("Menjalankan retrieval evaluation...")

    sebelumnya = _baca_hasil_sebelumnya()

    results = evaluate_retrieval(dataset, k=config.RETRIEVER_K)

    _cetak_metrik(results)
    _cetak_kegagalan(results)
    _cetak_perbandingan(results, sebelumnya)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print()
    print(f"Hasil disimpan ke: {RESULT_PATH}")


if __name__ == "__main__":
    run_evaluation()
