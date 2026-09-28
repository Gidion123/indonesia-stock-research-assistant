"""
Diagnostic, not an assertion: prints similarity scores so the gap between
in-scope and out-of-scope questions can be inspected by eye.

    pytest tests/test_retrieval_scores.py -s
"""

import pytest

from src.vector_store import get_vector_store


QUERIES = [
    "Apa risiko dilusi yang dihadapi BIPI?",
    "Bagaimana kondisi working capital BIPI?",
    "Mengapa harga saham Barito Group relatif resilient?",
    "Bagaimana analisis saham BBRI?",
    "Berapa harga emas dunia hari ini?",
    "Bagaimana cara memasak nasi goreng?",
    "Bagaimana cara memperbaiki AC yang rusak?",
]


@pytest.mark.db
@pytest.mark.model
def test_show_retrieval_scores():
    vector_store = get_vector_store()

    for query in QUERIES:
        print(f"\n{'=' * 70}\nQUERY: {query}\n{'=' * 70}")

        for rank, (document, score) in enumerate(
            vector_store.similarity_search_with_score(query, k=3),
            start=1,
        ):
            print(
                f"\nRank {rank} | score {score:.4f} | "
                f"{document.metadata.get('source')} "
                f"hal.{document.metadata.get('page')} | "
                f"chunk {document.metadata.get('chunk_id')}"
            )
            print(
                "  "
                + document.page_content[:150].replace("\n", " ")
                + "..."
            )
