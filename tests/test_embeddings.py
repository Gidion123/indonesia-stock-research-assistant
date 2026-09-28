import pytest

from src import config
from src.embeddings import MODEL_NAME, get_embeddings


pytestmark = pytest.mark.model


EXPECTED_DIMENSION = config.EMBEDDING_DIMENSION


def test_embedding_model_name():
    assert MODEL_NAME == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert MODEL_NAME == config.EMBEDDING_MODEL_NAME


def test_embedding_model_is_cached():
    """
    The model must be built once. Without the cache, every retrieval and
    every question re-loaded it from disk.
    """
    assert get_embeddings() is get_embeddings()


def test_embedding_dimension():
    vector = get_embeddings().embed_query("Aku ingin menukar ukuran baju.")

    assert len(vector) == EXPECTED_DIMENSION


def test_embedding_returns_numeric_vector():
    vector = get_embeddings().embed_query(
        "Informasi mengenai kebijakan pengembalian produk."
    )

    assert len(vector) > 0
    assert all(isinstance(value, float) for value in vector)


def test_semantically_similar_texts_have_higher_similarity():
    embeddings = get_embeddings()

    vector_a = embeddings.embed_query("Saya ingin menukar ukuran pakaian.")
    vector_b = embeddings.embed_query(
        "Apakah baju yang salah size bisa ditukar?"
    )
    vector_c = embeddings.embed_query(
        "Bagaimana cara melakukan pembayaran menggunakan QRIS?"
    )

    similarity_ab = sum(a * b for a, b in zip(vector_a, vector_b))
    similarity_ac = sum(a * c for a, c in zip(vector_a, vector_c))

    assert similarity_ab > similarity_ac
