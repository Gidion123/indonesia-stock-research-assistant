"""
PostgreSQL + PgVector vector store.

The engine is built once at import time and the store is cached, because
creating a PGVectorStore also builds the embedding model. An uncached
call would pay for that every time.
"""

from functools import lru_cache

from langchain_postgres import PGEngine, PGVectorStore

from src import config
from src.embeddings import get_embeddings


DATABASE_URL = config.DATABASE_URL
TABLE_NAME = config.TABLE_NAME


if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL tidak ditemukan di file .env.\n"
        "Contoh:\n"
        "  DATABASE_URL=postgresql+psycopg://postgres:password"
        "@localhost:5433/rag_stock_assistant"
    )


engine = PGEngine.from_connection_string(url=DATABASE_URL)


def get_psycopg_connection_url():
    """
    Convert the SQLAlchemy URL used by LangChain into the plain URL that
    psycopg expects.
    """
    return DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )


@lru_cache(maxsize=1)
def ensure_primary_category_column():
    """Backfill the old JSON category so public queries can filter in SQL."""
    import psycopg

    with psycopg.connect(get_psycopg_connection_url(), connect_timeout=5) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(81304590);")
            cursor.execute(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS "
                "category TEXT;"
            )
            cursor.execute(
                f"UPDATE {TABLE_NAME} SET category = "
                "langchain_metadata ->> 'category' "
                "WHERE category IS NULL;"
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_category "
                f"ON {TABLE_NAME} (category);"
            )


@lru_cache(maxsize=1)
def get_vector_store():
    """
    Create (once) and return the PgVector store used by the application.
    """
    ensure_primary_category_column()
    embeddings = get_embeddings()

    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=config.TABLE_NAME,
        embedding_service=embeddings,
        metadata_columns=["category"],
    )

    return vector_store


@lru_cache(maxsize=1)
def get_session_vector_store():
    """Shared connection/model, with ownership enforced on every search."""
    from src.session_documents import ensure_schema

    ensure_schema()
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=config.SESSION_TABLE_NAME,
        embedding_service=get_embeddings(),
        metadata_columns=["owner_id", "upload_id", "expires_at"],
    )
