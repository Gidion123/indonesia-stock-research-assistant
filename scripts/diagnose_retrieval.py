"""
Take apart every question that fails HitRate@k.

    python -m scripts.diagnose_retrieval
    python -m scripts.diagnose_retrieval --semua       # including passes
    python -m scripts.diagnose_retrieval --k 8

An average says nothing about how to fix anything. For each failure this
prints the expected gold chunks with their text, the chunks that were
actually retrieved, their similarity scores, and the rank at which the
first gold chunk turns up if k is widened. Retrieval is inspected down
to KEDALAMAN_PERIKSA = 30, so with the default k=8 the ranks that come
into view are 9 to 30.

That separates the causes:

    gold at rank 9-30            -> RANKING problem, not recall
    gold nowhere in the top-30   -> EMBEDDING or CHUNKING problem
    gold retrieved but its text
    does not answer the question -> GOLD ANNOTATION problem
    question too generic         -> QUERY PHRASING problem

This script only reads; it changes nothing.
"""

import argparse
import json

from src import config
from src.gold_chunks import load_gold_chunks


KEDALAMAN_PERIKSA = 30


def _muat_dataset():
    with config.EVAL_QUESTIONS_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    return data["items"] if isinstance(data, dict) and "items" in data else data


def _cari_dengan_skor(query, kedalaman):
    """
    Fetch chunks together with their similarity scores when the vector
    store supports it; otherwise fall back to the plain retriever and
    return no scores.
    """
    from src.vector_store import get_vector_store

    store = get_vector_store()

    try:
        pasangan = store.similarity_search_with_score(query, k=kedalaman)
        return [(doc, skor) for doc, skor in pasangan]

    except Exception:
        from src.retriever import retrieve_documents

        return [(doc, None) for doc in retrieve_documents(query, k=kedalaman)]


def _ringkas(teks, panjang=90):
    return " ".join(str(teks or "").split())[:panjang]


def diagnosa(k=None, tampilkan_semua=False):
    k = k or config.RETRIEVER_K

    dataset = _muat_dataset()
    gold = load_gold_chunks()

    by_id = {}

    for chunk in _semua_chunk():
        by_id[chunk.metadata["chunk_id"]] = chunk

    gagal = []
    lolos = []

    for item in dataset:
        entry = gold.get(item["id"])

        if not entry or not entry.get("gold_chunk_ids"):
            continue

        emas = set(entry["gold_chunk_ids"])

        hasil = _cari_dengan_skor(item["question"], KEDALAMAN_PERIKSA)
        ids = [d.metadata.get("chunk_id") for d, _ in hasil]

        kena_di_k = bool(emas & set(ids[:k]))

        peringkat_pertama = next(
            (i + 1 for i, cid in enumerate(ids) if cid in emas),
            None,
        )

        baris = {
            "item": item,
            "entry": entry,
            "hasil": hasil,
            "ids": ids,
            "emas": emas,
            "peringkat_pertama": peringkat_pertama,
        }

        (lolos if kena_di_k else gagal).append(baris)

    _cetak(gagal, lolos, k, tampilkan_semua, by_id)

    return gagal, lolos


def _semua_chunk():
    from src.preprocessing import load_and_split_documents

    return load_and_split_documents()


def _cetak(gagal, lolos, k, tampilkan_semua, by_id):
    total = len(gagal) + len(lolos)

    print("=" * 72)
    print(f"DIAGNOSA RETRIEVAL  (k={k})")
    print("=" * 72)
    print(f"  Pertanyaan bergold : {total}")
    print(f"  Lolos HitRate@{k}   : {len(lolos)}")
    print(f"  GAGAL HitRate@{k}   : {len(gagal)}")

    if gagal:
        print()
        print("  Ringkasan penyebab:")

        for b in gagal:
            p = b["peringkat_pertama"]

            if p is None:
                sebab = (
                    f"gold TIDAK ADA di top-{KEDALAMAN_PERIKSA} "
                    "-> embedding/chunking"
                )
            else:
                sebab = f"gold pertama di peringkat {p} -> RANKING"

            print(f"    {b['item']['id']:<12} {sebab}")

    daftar = gagal + (lolos if tampilkan_semua else [])

    for b in daftar:
        item = b["item"]
        status = "GAGAL" if b in gagal else "lolos"

        print()
        print("-" * 72)
        print(f"[{item['id']}] {status}")
        print(f"  query      : {item['question']}")
        print(f"  keyword    : {item.get('expected_keywords')}")
        print(f"  gold mode  : {b['entry'].get('gold_mode')}")
        print(f"  jumlah gold: {len(b['emas'])}")

        print(f"\n  GOLD CHUNK yang diharapkan:")

        for cid in sorted(b["emas"])[:6]:
            chunk = by_id.get(cid)

            if chunk is None:
                print(f"    {cid}  <- TIDAK ADA di knowledge base sekarang!")
                continue

            print(
                f"    {cid}  {chunk.metadata.get('source', '?')[:34]} "
                f"hal.{chunk.metadata.get('page')}"
            )
            print(f"        {_ringkas(chunk.page_content)}")

        if len(b["emas"]) > 6:
            print(f"    … dan {len(b['emas']) - 6} lagi")

        print(f"\n  TOP-{k} YANG TERAMBIL:")

        for peringkat, (doc, skor) in enumerate(b["hasil"][:k], 1):
            cid = doc.metadata.get("chunk_id")
            tanda = "  <-- GOLD" if cid in b["emas"] else ""
            nilai = f"{skor:.4f}" if skor is not None else "n/a"

            print(
                f"    {peringkat:>2}. {cid}  skor={nilai}  "
                f"{doc.metadata.get('source', '?')[:30]} "
                f"hal.{doc.metadata.get('page')}{tanda}"
            )
            print(f"        {_ringkas(doc.page_content, 84)}")

        p = b["peringkat_pertama"]

        if p and p > k:
            print(
                f"\n  -> Gold pertama sebenarnya ada di peringkat {p}. "
                f"Ini masalah RANKING;\n     recall akan naik dengan "
                "reranker, bukan dengan menaikkan k."
            )
        elif p is None:
            print(
                f"\n  -> Tidak ada gold di top-{KEDALAMAN_PERIKSA}. "
                "Periksa apakah chunk gold-nya\n     benar-benar menjawab "
                "pertanyaan ini, atau embedding tidak menangkap\n"
                "     hubungannya."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--semua", action="store_true")
    args = parser.parse_args()

    diagnosa(k=args.k, tampilkan_semua=args.semua)


if __name__ == "__main__":
    main()
