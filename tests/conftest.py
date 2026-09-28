"""
Shared pytest configuration.

The test suite has three tiers:

* plain tests  - pure functions. Always run, no network, no database.
* `db` tests   - need PostgreSQL + PgVector running and populated.
* `llm` tests  - need a working LLM API key and network.

Tests in the last two tiers skip themselves automatically when their
dependency is unavailable, so `pytest` gives a useful result on any
machine instead of a wall of connection errors.
"""

import os

import pytest

from src import config as app_config


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "db: memerlukan PostgreSQL + PgVector yang sudah terisi data",
    )
    config.addinivalue_line(
        "markers",
        "llm: memerlukan API key LLM dan koneksi internet",
    )
    config.addinivalue_line(
        "markers",
        "live: memanggil layanan pihak ketiga (yfinance). Jalankan manual.",
    )
    config.addinivalue_line(
        "markers",
        "model: memerlukan model embedding sentence-transformers",
    )


def _database_is_available():
    if not app_config.DATABASE_URL:
        return False

    try:
        import psycopg

        from src.vector_store import get_psycopg_connection_url

        with psycopg.connect(
            get_psycopg_connection_url(),
            connect_timeout=3,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1;")

        return True

    except Exception:
        return False


def _llm_is_available():
    if app_config.LLM_PROVIDER == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "")
    else:
        key = os.getenv("GROQ_API_KEY", "")

    return bool(key) and not key.startswith("sk-dummy")


def _embedding_model_is_available():
    if os.getenv("SKIP_MODEL_TESTS", "").strip().lower() in {"1", "true"}:
        return False

    try:
        import sentence_transformers  # noqa: F401

        return True

    except Exception:
        return False


DATABASE_AVAILABLE = _database_is_available()
LLM_AVAILABLE = _llm_is_available()
EMBEDDING_MODEL_AVAILABLE = _embedding_model_is_available()


def pytest_collection_modifyitems(config, items):
    skip_db = pytest.mark.skip(
        reason="PostgreSQL/PgVector tidak tersedia atau belum terisi data"
    )
    skip_llm = pytest.mark.skip(
        reason="API key LLM tidak tersedia"
    )
    skip_live = pytest.mark.skip(
        reason="test live dijalankan manual: pytest -m live"
    )
    skip_model = pytest.mark.skip(
        reason="sentence-transformers / model embedding tidak tersedia"
    )

    run_live = config.getoption("-m") == "live"

    for item in items:
        if "db" in item.keywords and not DATABASE_AVAILABLE:
            item.add_marker(skip_db)

        if "llm" in item.keywords and not LLM_AVAILABLE:
            item.add_marker(skip_llm)

        if "live" in item.keywords and not run_live:
            item.add_marker(skip_live)

        if "model" in item.keywords and not EMBEDDING_MODEL_AVAILABLE:
            item.add_marker(skip_model)


@pytest.fixture(scope="session")
def chunks():
    """
    The active knowledge base, cleaned and chunked. Built once per session
    because reading five PDFs is not cheap.
    """
    from src.preprocessing import load_and_split_documents

    return load_and_split_documents()


@pytest.fixture(scope="session")
def documents():
    from src.preprocessing import load_all_pdfs

    return load_all_pdfs()
