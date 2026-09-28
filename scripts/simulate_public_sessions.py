"""Exercise real pgvector isolation on a disposable `ragtest` database.

Example: DATABASE_URL=postgresql+psycopg://.../ragtest \
    python -m scripts.simulate_public_sessions
This script refuses to run on any database other than `ragtest`.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src import config
from src.ingestion import IngestionError
from src.retriever import get_session_retriever
from src.session_documents import (
    QuotaExceeded,
    add_session_pdf,
    cleanup_expired,
    consume_question,
    ensure_schema,
    list_session_pdfs,
    new_owner_id,
)


class TinyEmbedding(Embeddings):
    """Fast fixed embedding; isolation does not depend on model quality."""

    def embed_documents(self, texts):
        return [[1.0] + [0.0] * (config.EMBEDDING_DIMENSION - 1) for _ in texts]

    def embed_query(self, text):
        return [1.0] + [0.0] * (config.EMBEDDING_DIMENSION - 1)


def main():
    from src import vector_store
    from src.session_documents import _connection, _jakarta_day

    with _connection() as connection:
        if connection.info.dbname != "ragtest":
            raise RuntimeError("Simulasi hanya boleh berjalan pada database ragtest.")
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {config.TABLE_NAME} ("
                "langchain_id UUID PRIMARY KEY, content TEXT NOT NULL, "
                f"embedding VECTOR({config.EMBEDDING_DIMENSION}) NOT NULL, "
                "langchain_metadata JSONB);"
            )

    ensure_schema()
    vector_store.get_embeddings = lambda: TinyEmbedding()
    vector_store.get_vector_store.cache_clear()
    vector_store.get_session_vector_store.cache_clear()

    base_ids = vector_store.get_vector_store().add_documents(
        [Document(
            page_content="Dokumen riset utama yang dibaca semua pengunjung.",
            metadata={"source": "simulasi-utama.pdf", "page": 1,
                      "category": "primary"},
        ), Document(
            page_content="Dokumen tambahan lama tidak boleh tampil untuk publik.",
            metadata={"source": "legacy-additional.pdf", "page": 1,
                      "category": "additional"},
        )]
    )

    owners = [new_owner_id(), new_owner_id()]
    source_pdf = Path("data/knowledge_base/primary/Blueprint Investasi Presisi Chaos Scenario.pdf")
    data = source_pdf.read_bytes()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            uploaded = list(pool.map(
                lambda item: add_session_pdf(data, f"private-user-{item[0]}.pdf", item[1]),
                enumerate(owners),
            ))

        assert all(entry["chunks"] > 0 for entry in uploaded)

        def search(index):
            docs = get_session_retriever(owners[index]).invoke(
                "Apa isi dokumen riset ini?"
            )
            return {doc.metadata.get("source") for doc in docs}

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(search, [index % 2 for index in range(100)]))

        for index, names in enumerate(results):
            owner_index = index % 2
            assert "simulasi-utama.pdf" in names
            assert f"private-user-{owner_index}.pdf" in names
            assert f"private-user-{1 - owner_index}.pdf" not in names
            assert "legacy-additional.pdf" not in names

        for index, owner in enumerate(owners):
            own_list = [doc["name"] for doc in list_session_pdfs(owner)]
            assert own_list == [f"private-user-{index}.pdf"]

        try:
            add_session_pdf(data, "private-user-0.pdf", owners[0])
        except IngestionError:
            pass
        else:
            raise AssertionError("Duplicate upload was accepted")

        second_pdf = Path(
            "data/knowledge_base/primary/Analisis Mendalam Saham Barito Group.pdf"
        ).read_bytes()
        add_session_pdf(second_pdf, "second-private.pdf", owners[0])
        try:
            add_session_pdf(data, "third-private.pdf", owners[0])
        except IngestionError as error:
            assert "Maksimal 2 PDF" in str(error)
        else:
            raise AssertionError("Third upload was accepted")

        original_limit = config.SESSION_DAILY_QUESTION_LIMIT
        config.SESSION_DAILY_QUESTION_LIMIT = 2
        try:
            consume_question(owners[0])
            consume_question(owners[0])
            try:
                consume_question(owners[0])
            except QuotaExceeded:
                pass
            else:
                raise AssertionError("Question quota was exceeded")
        finally:
            config.SESSION_DAILY_QUESTION_LIMIT = original_limit

        with _connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT used FROM {config.DAILY_USAGE_TABLE_NAME} "
                    "WHERE day = %s AND kind = 'question' "
                    "AND scope = 'global';",
                    (_jakarta_day(),),
                )
                global_used = cursor.fetchone()[0]

        original_global_limit = config.GLOBAL_DAILY_QUESTION_LIMIT
        config.GLOBAL_DAILY_QUESTION_LIMIT = global_used + 1
        try:
            def try_question(owner):
                try:
                    consume_question(owner)
                    return True
                except QuotaExceeded:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(try_question, owners))
            assert sorted(outcomes) == [False, True]
        finally:
            config.GLOBAL_DAILY_QUESTION_LIMIT = original_global_limit

        with _connection() as connection:
            with connection.cursor() as cursor:
                expired = datetime.now(timezone.utc) - timedelta(seconds=1)
                cursor.execute(
                    f"UPDATE {config.SESSION_TABLE_NAME} "
                    "SET expires_at = %s WHERE owner_id = %s;",
                    (expired, owners[0]),
                )
                cursor.execute(
                    f"UPDATE {config.SESSION_UPLOAD_TABLE_NAME} "
                    "SET expires_at = %s WHERE owner_id = %s;",
                    (expired, owners[0]),
                )
        assert "private-user-0.pdf" not in search(0)
        assert list_session_pdfs(owners[0]) == []
        assert cleanup_expired() == 2

        print("PASS: 2 concurrent uploads, 100 isolated searches, primary-only filter, private sidebar, duplicate and two-PDF caps, atomic global and session question quotas, expiry")
    finally:
        with _connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {config.SESSION_UPLOAD_TABLE_NAME} "
                    "SET expires_at = %s WHERE owner_id = ANY(%s);",
                    (datetime.now(timezone.utc) - timedelta(seconds=1), owners),
                )
        cleanup_expired()
        vector_store.get_vector_store().delete(base_ids)


if __name__ == "__main__":
    main()
