"""
Evaluate the router and the entity resolver against a labeled dataset.

    python -m scripts.evaluate_router              # patterns only, no LLM
    python -m scripts.evaluate_router --llm        # with the LLM router
    python -m scripts.evaluate_router --llm --semantik   # + LLM resolver

Without this script, router accuracy is just a claim. Three failure
modes are measured:

    1. wrong intent          RAG taken for LIVE, or the other way round
    2. wrong entity          the ticker recognized is not the one meant
    3. router false refusal  a valid question marked OUT_OF_SCOPE

Number 3 is reported separately because it costs the most: the refusal
happens before a single document has been looked at.
"""

import argparse
import json
from collections import Counter

from src import config
from src.entity_resolver import resolve_ticker
from src.router import INTENT_OUT_OF_SCOPE, classify_router_intent


ROUTER_EVAL_PATH = config.EVAL_DATASET_DIR / "router_eval.json"
HASIL_PATH = config.EVAL_RESULTS_DIR / "router_results.json"


def muat_dataset():
    with ROUTER_EVAL_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def jalankan(pakai_llm_router=False, pakai_semantik=False):
    data = muat_dataset()
    items = data["items"]

    baris = []
    panggilan_llm = 0

    for item in items:
        router = classify_router_intent(
            item["question"],
            izinkan_llm=pakai_llm_router,
        )
        panggilan_llm += router["llm_calls"]

        intent_benar = router["intent"] == item["expected_intent"]

        resolusi = None
        entity_benar = None
        dilewati = False

        # A row that can ONLY be settled through PgVector + LLM must not
        # be counted as wrong when that mode is switched off. What would
        # be measured in that case is the evaluator's own switch, not
        # the resolver - the same class of mistake as a metric that
        # cannot fail.
        if item.get("requires_semantic") and not pakai_semantik:
            dilewati = True

        # The resolver only matters for questions that need an entity.
        elif item.get("expected_resolution") or item.get("expected_ticker"):
            resolusi = resolve_ticker(
                item["question"],
                session_context=item.get("session_context"),
                izinkan_semantik=pakai_semantik,
            )
            panggilan_llm += resolusi["llm_calls"]

            diharapkan_status = item.get("expected_resolution")
            diharapkan_ticker = item.get("expected_ticker")

            cocok_status = (
                diharapkan_status is None
                or resolusi["status"] == diharapkan_status
            )
            cocok_ticker = (
                diharapkan_ticker is None
                or resolusi["ticker"] == diharapkan_ticker
            )

            entity_benar = bool(cocok_status and cocok_ticker)

        baris.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_intent": item["expected_intent"],
                "intent": router["intent"],
                "intent_benar": intent_benar,
                "method": router["method"],
                "expected_ticker": item.get("expected_ticker"),
                "ticker": resolusi["ticker"] if resolusi else None,
                "resolution_status": resolusi["status"] if resolusi else None,
                "resolution_level": resolusi["level"] if resolusi else None,
                "entity_benar": entity_benar,
                "dilewati": dilewati,
            }
        )

    return data, baris, panggilan_llm


def ringkas(baris, panggilan_llm, pakai_semantik):
    total = len(baris)
    intent_benar = sum(1 for b in baris if b["intent_benar"])

    dinilai_entity = [b for b in baris if b["entity_benar"] is not None]
    entity_benar = sum(1 for b in dinilai_entity if b["entity_benar"])
    dilewati = [b for b in baris if b.get("dilewati")]

    false_refusal = [
        b
        for b in baris
        if b["intent"] == INTENT_OUT_OF_SCOPE
        and b["expected_intent"] != INTENT_OUT_OF_SCOPE
    ]

    metode = Counter(b["method"] for b in baris)

    print("=" * 64)
    print("EVALUASI ROUTER + ENTITY RESOLVER")
    print("=" * 64)
    print(f"  Pertanyaan            : {total}")
    print(
        f"  Intent benar          : {intent_benar}/{total} "
        f"({intent_benar / total:.1%})"
    )

    if dinilai_entity:
        print(
            f"  Entity benar          : {entity_benar}/{len(dinilai_entity)} "
            f"({entity_benar / len(dinilai_entity):.1%})"
        )

    if dilewati:
        print(
            f"  Dilewati (butuh --semantik): {len(dilewati)} "
            f"-> {[b['id'] for b in dilewati]}"
        )
        print(
            "    Baris ini tidak dihitung benar maupun salah. Jalankan\n"
            "    `--llm --semantik` untuk menilainya."
        )

    print(f"  False refusal router  : {len(false_refusal)}")
    print(f"  Panggilan LLM total   : {panggilan_llm}")
    print(f"  Metode router         : {dict(metode)}")

    # The entity number cannot be read without knowing whether the
    # exchange was reachable. Without this line, "23%" reads as a bad
    # resolver when what actually failed was the network.
    belum_terverifikasi = [
        b for b in dinilai_entity if b["resolution_status"] == "unverified"
    ]

    if belum_terverifikasi:
        print(
            f"\n  PERINGATAN: {len(belum_terverifikasi)} baris berstatus "
            "'unverified' — Yahoo Finance tidak bisa dihubungi dari mesin\n"
            "  ini, jadi ticker yang benar pun tidak bisa dikonfirmasi. "
            "Angka entity\n  di atas BUKAN ukuran kualitas resolver "
            "selama ini terjadi."
        )

    salah_intent = [b for b in baris if not b["intent_benar"]]

    if salah_intent:
        print("\n  Intent yang salah:")
        for b in salah_intent:
            print(
                f"    {b['id']:<16} harap {b['expected_intent']:<13} "
                f"dapat {b['intent']:<13} ({b['method']})"
            )

    salah_entity = [
        b for b in dinilai_entity if not b["entity_benar"]
    ]

    if salah_entity:
        print("\n  Entity yang salah:")
        for b in salah_entity:
            print(
                f"    {b['id']:<16} harap {str(b['expected_ticker']):<8} "
                f"dapat {str(b['ticker']):<8} "
                f"status={b['resolution_status']}"
            )

    if false_refusal:
        print("\n  FALSE REFUSAL (paling merugikan):")
        for b in false_refusal:
            print(f"    {b['id']}: {b['question']}")

    return {
        "total": total,
        "intent_accuracy": intent_benar / total if total else 0.0,
        "entity_accuracy": (
            entity_benar / len(dinilai_entity) if dinilai_entity else None
        ),
        "entity_skipped": [b["id"] for b in dilewati],
        "false_refusal_count": len(false_refusal),
        "llm_calls": panggilan_llm,
        "router_methods": dict(metode),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        action="store_true",
        help="izinkan router memanggil LLM saat pola tidak tegas",
    )
    parser.add_argument(
        "--semantik",
        action="store_true",
        help="izinkan resolver memakai PgVector + LLM",
    )
    args = parser.parse_args()

    data, baris, panggilan_llm = jalankan(
        pakai_llm_router=args.llm,
        pakai_semantik=args.semantik,
    )

    ringkasan = ringkas(baris, panggilan_llm, args.semantik)

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with HASIL_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_version": data.get("version"),
                "config": {
                    "router_llm": args.llm,
                    "resolver_semantik": args.semantik,
                },
                "summary": ringkasan,
                "rows": baris,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDisimpan ke: {HASIL_PATH}")


if __name__ == "__main__":
    main()
