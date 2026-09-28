"""
Derive and audit the gold chunk annotation.

Run this after every change to the knowledge base or to the chunking
configuration, because chunk ids are content-addressed and therefore
change when the text changes.

    python -m scripts.build_gold_chunks
"""

from src import config
from src.evaluation import load_evaluation_dataset, load_expected_keywords
from src.gold_chunks import (
    SATURATION_THRESHOLD,
    audit_gold_chunks,
    build_gold_chunks,
    save_gold_chunks,
)
from src.preprocessing import load_and_split_documents, new_ingestion_report


def run():
    print("Memuat & membersihkan knowledge base...")

    report = new_ingestion_report()
    chunks = load_and_split_documents(report=report)

    print(f"  PDF                       : {report['pdf_files']}")
    print(f"  Halaman total             : {report['pages_total']}")
    print(f"  Halaman dipakai           : {report['pages_kept']}")
    print(f"  Halaman dibuang (pustaka) : {report['pages_dropped_reference']}")
    print(f"  Halaman dipotong          : {report['pages_truncated']}")
    print(f"  Marker sitasi dibuang     : {report['citation_markers_removed']}")
    print(f"  Kata patah diperbaiki     : {len(report['broken_words_repaired'])}")
    print(f"  Chunk                     : {len(chunks)}")

    dataset = load_evaluation_dataset()
    keywords = load_expected_keywords()

    print("\nMenurunkan gold chunk dari expected_keywords...")
    gold = build_gold_chunks(dataset, keywords, chunks=chunks)
    path = save_gold_chunks(gold)

    audit = audit_gold_chunks(gold)

    print("\n=== AUDIT GOLD CHUNK ===")
    print(f"  Pertanyaan in-scope       : {audit['total_items']}")
    print(f"  Gold mode strict          : {audit['strict']}")
    print(f"  Gold mode relaxed         : {audit['relaxed']}")
    print(f"  Tanpa gold chunk          : {audit['without_gold']}")
    print(f"  Rata-rata gold per soal   : {audit['average_gold_chunks']}")
    print(f"  Gold terbanyak            : {audit['max_gold_chunks']}")

    if audit["without_gold"]:
        print(
            "\n  PERHATIAN: ada pertanyaan yang kata kuncinya tidak "
            "ditemukan di dokumen mana pun."
        )
        for question_id, entry in gold.items():
            if entry["gold_mode"] == "none":
                print(f"    - {question_id}: {entry['keywords']}")
        print(
            "    Artinya pertanyaan itu tidak bisa dinilai di lapis "
            f"retrieval. Perbaiki kata kuncinya di "
            f"{config.EVAL_QUESTIONS_PATH.name}."
        )

    if audit["saturated_items"]:
        print(
            f"\n  PERHATIAN: kata kunci terlalu umum (> "
            f"{SATURATION_THRESHOLD} gold chunk). Recall@k untuk soal ini "
            f"hampir pasti 1,0 dan tidak mengukur apa pun:"
        )
        for question_id, size in audit["saturated_items"]:
            print(
                f"    - {question_id}: {size} gold chunk "
                f"({gold[question_id]['keywords']})"
            )

    print(f"\nGold annotation disimpan ke: {path}")
    print(
        "File ini boleh dikoreksi manual. Metrik akan memakai isinya "
        "apa adanya."
    )


if __name__ == "__main__":
    run()
