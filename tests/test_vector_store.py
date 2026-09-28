import psycopg
import pytest

from src import config
from src.preprocessing import load_and_split_documents
from src.vector_store import (
    DATABASE_URL,
    TABLE_NAME,
    engine,
    get_psycopg_connection_url,
    get_vector_store,
)


def get_row_count():
    with psycopg.connect(get_psycopg_connection_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
            result = cursor.fetchone()

    return result[0]


def test_database_url_is_loaded():
    assert DATABASE_URL
    assert DATABASE_URL == config.DATABASE_URL


def test_vector_table_name():
    assert TABLE_NAME == "stock_knowledge"


def test_pgengine_is_created():
    assert engine is not None


@pytest.mark.db
@pytest.mark.model
def test_vector_store_is_cached():
    assert get_vector_store() is get_vector_store()


@pytest.mark.db
def test_vector_table_contains_documents():
    assert get_row_count() > 0


@pytest.mark.db
def test_vector_table_matches_current_configuration():
    """
    The stored chunk count must equal what the current preprocessing
    configuration produces. A mismatch means the database is stale and
    every evaluation number read from it is meaningless.

    Fix by re-running:
        python -m scripts.ingest_knowledge_base
    """
    expected = len(load_and_split_documents())

    assert get_row_count() == expected, (
        f"PgVector berisi {get_row_count()} chunk, sedangkan konfigurasi "
        f"sekarang menghasilkan {expected}. Jalankan ulang ingestion."
    )
