"""
Who contacts Hugging Face, when, and how many times.

    python -m scripts.diagnose_embeddings --jalur retrieval
    python -m scripts.diagnose_embeddings --jalur end_to_end
    python -m scripts.diagnose_embeddings --jalur end_to_end --pertanyaan 3

Run once per path, in a separate process, exactly the way the two
evaluators run. Without that, "evaluate.py does not touch Hugging Face"
is only a guess: both scripts share the same lru cache, so whichever one
runs first looks like the only one that went to the network.

What is counted:

    HTTP requests to huggingface.co     caught in requests.Session
    embedding model construction        counted in src/embeddings.py
    LLM calls                           from the assistant.jawab() trace
    Yahoo Finance requests              from statistik_market_data()

This script only reads and counts; it changes nothing.
"""

import argparse
import time


PERMINTAAN_HF = []


def pasang_penghitung_jaringan():
    """
    Record every HTTP request heading for a Hugging Face host.

    Patched onto `requests.sessions.Session.request`, which both
    huggingface_hub and transformers go through, so no path escapes the
    count.
    """
    import requests.sessions

    asli = requests.sessions.Session.request

    def terbungkus(self, method, url, *args, **kwargs):
        if "huggingface.co" in str(url) or "hf.co" in str(url):
            PERMINTAAN_HF.append((method, str(url)))

        return asli(self, method, url, *args, **kwargs)

    requests.sessions.Session.request = terbungkus


def _garis(judul):
    print()
    print("=" * 68)
    print(judul)
    print("=" * 68)


def laporkan_model():
    from src import embeddings as emb

    print(f"  sumber model            : {emb.sumber_model()}")

    if emb.alasan_unduh():
        print(f"  sebab tidak dari cache  : {emb.alasan_unduh()}")

    print(f"  cache get_embeddings    : {emb.get_embeddings.cache_info()}")

    from src.vector_store import get_vector_store
    from src.retriever import get_retriever

    print(f"  cache get_vector_store  : {get_vector_store.cache_info()}")
    print(f"  cache get_retriever     : {get_retriever.cache_info()}")

    store = get_vector_store()
    sama = getattr(store, "embedding_service", None) is emb.get_embeddings()

    print(f"  store memakai model yang sama : {sama}")


def laporkan_hf(tahap):
    print(f"  permintaan huggingface.co ({tahap}) : {len(PERMINTAAN_HF)}")

    for method, url in PERMINTAAN_HF[:8]:
        print(f"      {method} {url[:96]}")

    if len(PERMINTAAN_HF) > 8:
        print(f"      … dan {len(PERMINTAAN_HF) - 8} lagi")


def jalur_retrieval(jumlah):
    """
    The path scripts/evaluate.py uses.
    """
    from src.evaluation import load_evaluation_dataset
    from src.retriever import retrieve_documents

    dataset = [
        item
        for item in load_evaluation_dataset()
        if item["type"] == "in_scope"
    ][:jumlah]

    _garis("JALUR RETRIEVAL (seperti scripts/evaluate.py)")

    mulai = time.perf_counter()
    sebelum = len(PERMINTAAN_HF)

    for item in dataset:
        t0 = time.perf_counter()
        documents = retrieve_documents(item["question"], k=8)
        delta = len(PERMINTAAN_HF) - sebelum
        sebelum = len(PERMINTAAN_HF)

        print(
            f"  {item['id']:<14} chunk={len(documents):<3} "
            f"{time.perf_counter() - t0:6.2f}s  "
            f"permintaan_hf_baru={delta}"
        )

    print(f"\n  total waktu             : {time.perf_counter() - mulai:.2f}s")

    laporkan_model()
    laporkan_hf("total")


def jalur_end_to_end(jumlah):
    """
    The path scripts/evaluate_answers.py --end-to-end uses.
    """
    from src.assistant import jawab
    from src.evaluation import load_evaluation_dataset
    from src.market_data import (
        reset_statistik_market_data,
        statistik_market_data,
    )

    dataset = load_evaluation_dataset()[:jumlah]

    _garis("JALUR END-TO-END (seperti evaluate_answers --end-to-end)")

    reset_statistik_market_data()

    mulai = time.perf_counter()
    sebelum = len(PERMINTAAN_HF)
    llm_total = 0

    for item in dataset:
        t0 = time.perf_counter()
        hasil = jawab(item["question"])

        jejak = hasil.get("trace") or {}
        llm = (
            jejak.get("router_llm_calls", 0)
            + jejak.get("resolution_llm_calls", 0)
            + jejak.get("answer_llm_calls", 0)
        )
        llm_total += llm

        delta = len(PERMINTAAN_HF) - sebelum
        sebelum = len(PERMINTAAN_HF)

        print(
            f"  {item['id']:<14} {str(hasil.get('intent')):<13} "
            f"{hasil.get('status'):<18} chunk={len(hasil.get('documents') or []):<3} "
            f"{time.perf_counter() - t0:6.2f}s  llm={llm}  "
            f"permintaan_hf_baru={delta}"
        )

    print(f"\n  total waktu             : {time.perf_counter() - mulai:.2f}s")
    print(f"  panggilan LLM           : {llm_total}")

    statistik = statistik_market_data()
    print(
        f"  permintaan Yahoo        : {statistik['requests']} "
        f"(cache hit: {statistik['cache_hits']})"
    )

    laporkan_model()
    laporkan_hf("total")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jalur",
        choices=["retrieval", "end_to_end"],
        default="retrieval",
    )
    parser.add_argument("--pertanyaan", type=int, default=3)
    args = parser.parse_args()

    pasang_penghitung_jaringan()

    _garis("SEBELUM MODEL DIBANGUN")
    laporkan_hf("sejauh ini")

    from src.embeddings import get_embeddings

    t0 = time.perf_counter()
    get_embeddings()
    durasi = time.perf_counter() - t0

    _garis("SESUDAH get_embeddings()")
    print(f"  waktu membangun model   : {durasi:.2f}s")
    laporkan_hf("saat membangun model")
    laporkan_model()

    if args.jalur == "retrieval":
        jalur_retrieval(args.pertanyaan)
    else:
        jalur_end_to_end(args.pertanyaan)


if __name__ == "__main__":
    main()
