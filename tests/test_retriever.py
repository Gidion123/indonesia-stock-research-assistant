import pytest

from src import config
from src.retriever import RETRIEVER_K, get_retriever, retrieve_documents


QUERY_BARITO = "Apa analisis mengenai saham Barito Group?"


def test_retriever_k_matches_frozen_configuration():
    """
    k=8 with plain similarity is the configuration frozen in
    "Phase 1 - Retrieval Optimization" of the notebook.
    """
    assert RETRIEVER_K == 8
    assert config.RETRIEVER_SEARCH_TYPE == "similarity"


@pytest.mark.db
@pytest.mark.model
def test_retriever_is_cached():
    assert get_retriever() is get_retriever()


@pytest.mark.db
@pytest.mark.model
def test_retriever_returns_documents():
    assert len(retrieve_documents(QUERY_BARITO)) > 0


@pytest.mark.db
@pytest.mark.model
def test_retriever_returns_at_most_k_documents():
    assert len(retrieve_documents(QUERY_BARITO)) <= RETRIEVER_K


@pytest.mark.db
@pytest.mark.model
def test_retrieved_documents_have_metadata():
    for document in retrieve_documents(QUERY_BARITO):
        assert document.page_content.strip() != ""
        assert "source" in document.metadata
        assert "page" in document.metadata


@pytest.mark.db
@pytest.mark.model
def test_retrieved_documents_carry_chunk_id():
    """
    chunk_id must survive the round trip through PgVector, otherwise
    retrieval cannot be evaluated at chunk level.
    """
    for document in retrieve_documents(QUERY_BARITO):
        assert document.metadata.get("chunk_id"), (
            "Chunk yang tersimpan di PgVector belum punya chunk_id. "
            "Ini bukan kesalahan kode retrieval: isi database berasal "
            "dari ingestion versi lama, sebelum chunk_id ada. "
            "Jalankan ulang:\n"
            "    python -m scripts.ingest_knowledge_base\n"
            "    python -m scripts.build_gold_chunks"
        )
        assert len(document.metadata["chunk_id"]) == 12


@pytest.mark.db
@pytest.mark.model
def test_retrieved_documents_have_valid_page_metadata():
    for document in retrieve_documents(QUERY_BARITO):
        assert isinstance(document.metadata["page"], int)
        assert document.metadata["page"] >= 1


@pytest.mark.db
@pytest.mark.model
@pytest.mark.parametrize(
    "query, expected_fragment",
    [
        ("Apa analisis mengenai saham Barito Group?", "Barito"),
        ("Bagaimana analisis IHSG 2026?", "IHSG"),
        ("Apa analisis saham BBRI?", "Equity"),
    ],
)
def test_topical_retrieval_relevance(query, expected_fragment):
    documents = retrieve_documents(query)

    assert len(documents) > 0
    assert any(
        expected_fragment in document.metadata["source"]
        for document in documents
    )


@pytest.mark.db
@pytest.mark.model
@pytest.mark.skipif(
    not config.INCLUDE_ADDITIONAL_DOCUMENTS,
    reason=(
        "BIPI ada di data/knowledge_base/additional dan bersifat opt-in. "
        "Aktifkan dengan INCLUDE_ADDITIONAL_DOCUMENTS=true lalu ingest ulang."
    ),
)
def test_bipi_retrieval_relevance():
    """
    Only meaningful once the additional documents are ingested. The
    previous version of this test asserted that BIPI chunks come back
    while test_preprocessing asserted BIPI is not part of the knowledge
    base — the two could never pass together.
    """
    documents = retrieve_documents("Apa risiko investasi BIPI?")

    assert any(
        "BIPI" in document.metadata["source"] for document in documents
    )
