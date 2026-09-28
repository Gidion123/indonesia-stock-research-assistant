"""Temporary, isolated PDF uploads for the public Streamlit application.

The curated corpus and visitor uploads use different vector tables. Every
visitor query against the temporary table requires a server-generated owner
ID; raw uploaded PDFs are processed in memory and never written to disk.
"""

import hashlib
from io import BytesIO
import logging
import re
import secrets
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from src import config
from src.ingestion import IngestionError, validasi_pdf


logger = logging.getLogger(__name__)
_OWNER_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_cleanup_lock = threading.Lock()
_last_cleanup = 0.0


class QuotaExceeded(Exception):
    """A public usage limit was reached before doing expensive work."""


def new_owner_id():
    """Return an unguessable identifier for one browser-tab session."""
    return secrets.token_hex(16)


def validate_owner_id(owner_id):
    if not isinstance(owner_id, str) or not _OWNER_PATTERN.fullmatch(owner_id):
        raise ValueError("ID sesi tidak valid.")
    return owner_id


def _connection():
    import psycopg

    from src.vector_store import get_psycopg_connection_url

    return psycopg.connect(get_psycopg_connection_url(), connect_timeout=5)


@lru_cache(maxsize=1)
def ensure_schema():
    """Create public-upload tables without changing the curated table."""
    with _connection() as connection:
        with connection.cursor() as cursor:
            # The transaction lock makes first-run setup safe when two
            # visitors arrive together (or two app workers start at once).
            cursor.execute("SELECT pg_advisory_xact_lock(81304591);")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {config.SESSION_TABLE_NAME} (
                    langchain_id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR({config.EMBEDDING_DIMENSION}) NOT NULL,
                    owner_id TEXT NOT NULL,
                    upload_id UUID NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    langchain_metadata JSONB
                );
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{config.SESSION_TABLE_NAME}_owner "
                f"ON {config.SESSION_TABLE_NAME} (owner_id);"
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{config.SESSION_TABLE_NAME}_upload "
                f"ON {config.SESSION_TABLE_NAME} (upload_id);"
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {config.SESSION_UPLOAD_TABLE_NAME} (
                    upload_id UUID PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    pages INTEGER,
                    chunks INTEGER,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'ready')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    expires_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{config.SESSION_UPLOAD_TABLE_NAME}_owner "
                f"ON {config.SESSION_UPLOAD_TABLE_NAME} (owner_id, expires_at);"
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {config.DAILY_USAGE_TABLE_NAME} (
                    day DATE NOT NULL,
                    kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, kind, scope)
                );
                """
            )


def _jakarta_day():
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Jakarta")).date()


def _consume_daily(cursor, kind, owner_id, owner_limit, global_limit):
    day = _jakarta_day()
    limits = (("global", global_limit), (owner_id, owner_limit))

    for scope, _ in limits:
        cursor.execute(
            f"INSERT INTO {config.DAILY_USAGE_TABLE_NAME} "
            "(day, kind, scope, used) VALUES (%s, %s, %s, 0) "
            "ON CONFLICT DO NOTHING;",
            (day, kind, scope),
        )

    # Lock both rows before either increment, so concurrent workers cannot
    # overshoot a limit. Always lock in the same order to avoid deadlocks.
    for scope, limit in limits:
        cursor.execute(
            f"SELECT used FROM {config.DAILY_USAGE_TABLE_NAME} "
            "WHERE day = %s AND kind = %s AND scope = %s FOR UPDATE;",
            (day, kind, scope),
        )
        if cursor.fetchone()[0] >= limit:
            raise QuotaExceeded(
                "Batas penggunaan hari ini sudah tercapai. Silakan coba besok."
            )

    for scope, _ in limits:
        cursor.execute(
            f"UPDATE {config.DAILY_USAGE_TABLE_NAME} SET used = used + 1 "
            "WHERE day = %s AND kind = %s AND scope = %s;",
            (day, kind, scope),
        )


def consume_question(owner_id):
    """Atomically reserve one question before the assistant calls an API."""
    validate_owner_id(owner_id)
    ensure_schema()
    with _connection() as connection:
        with connection.cursor() as cursor:
            _consume_daily(
                cursor,
                "question",
                owner_id,
                config.SESSION_DAILY_QUESTION_LIMIT,
                config.GLOBAL_DAILY_QUESTION_LIMIT,
            )


def _safe_filename(name):
    name = unicodedata.normalize("NFKC", str(name or ""))
    name = Path(name.replace("\\", "/")).name
    name = "".join(
        char if char.isalnum() or char in " ._-()" else "_"
        for char in name
    ).strip(" .")[:120]
    if not name.lower().endswith(".pdf") or name.lower() == ".pdf":
        raise IngestionError("Pilih berkas dengan nama dan ekstensi PDF yang valid.")
    return name


def _reserve_upload(owner_id, upload_id, file_name, digest, size_bytes):
    ensure_schema()
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s));", (owner_id,)
            )
            cursor.execute(
                f"SELECT file_name, file_sha256 FROM "
                f"{config.SESSION_UPLOAD_TABLE_NAME} "
                "WHERE owner_id = %s AND expires_at > now() FOR UPDATE;",
                (owner_id,),
            )
            existing = cursor.fetchall()
            if len(existing) >= config.SESSION_MAX_PDFS:
                raise IngestionError(
                    f"Maksimal {config.SESSION_MAX_PDFS} PDF per sesi. "
                    "Dokumen sementara akan dihapus otomatis setelah 24 jam."
                )
            if any(name == file_name or sha == digest for name, sha in existing):
                raise IngestionError("PDF ini sudah diunggah dalam sesi Anda.")

            _consume_daily(
                cursor,
                "upload",
                owner_id,
                config.SESSION_MAX_PDFS,
                config.GLOBAL_DAILY_UPLOAD_LIMIT,
            )
            expires = datetime.now(timezone.utc) + timedelta(
                hours=config.SESSION_UPLOAD_TTL_HOURS
            )
            cursor.execute(
                f"INSERT INTO {config.SESSION_UPLOAD_TABLE_NAME} "
                "(upload_id, owner_id, file_name, file_sha256, size_bytes, "
                "status, expires_at) VALUES (%s, %s, %s, %s, %s, 'pending', %s);",
                (upload_id, owner_id, file_name, digest, size_bytes, expires),
            )
            return expires


def _discard_upload(owner_id, upload_id):
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {config.SESSION_TABLE_NAME} "
                "WHERE owner_id = %s AND upload_id = %s;",
                (owner_id, upload_id),
            )
            cursor.execute(
                f"DELETE FROM {config.SESSION_UPLOAD_TABLE_NAME} "
                "WHERE owner_id = %s AND upload_id = %s "
                "RETURNING (created_at AT TIME ZONE 'Asia/Jakarta')::date;",
                (owner_id, upload_id),
            )
            removed = cursor.fetchone()
            if removed:
                day = removed[0]
                for scope in ("global", owner_id):
                    cursor.execute(
                        f"UPDATE {config.DAILY_USAGE_TABLE_NAME} "
                        "SET used = GREATEST(used - 1, 0) "
                        "WHERE day = %s AND kind = 'upload' AND scope = %s;",
                        (day, scope),
                    )


def add_session_pdf(data, original_name, owner_id):
    """Validate, embed, and keep only this session's temporary vectors."""
    validate_owner_id(owner_id)
    file_name = _safe_filename(original_name)
    data = bytes(data)
    if not data:
        raise IngestionError("Berkasnya kosong.")
    if len(data) > config.SESSION_MAX_PDF_MB * 1024 * 1024:
        raise IngestionError(
            f"PDF maksimal {config.SESSION_MAX_PDF_MB} MB."
        )

    from src.preprocessing import load_pdf, split_documents

    upload_id = uuid4()
    reserved = False
    try:
        info = validasi_pdf(
            data,
            max_mb=config.SESSION_MAX_PDF_MB,
            max_pages=config.SESSION_MAX_PDF_PAGES,
            file_name=file_name,
        )
        digest = hashlib.sha256(data).hexdigest()
        expires = _reserve_upload(
            owner_id, upload_id, file_name, digest, len(data)
        )
        reserved = True

        pages = load_pdf(
            BytesIO(data), source_name=file_name, category="session"
        )
        if sum(len(page.page_content) for page in pages) > config.SESSION_MAX_EXTRACTED_CHARS:
            raise IngestionError("Teks PDF terlalu panjang untuk demo publik.")
        for page in pages:
            page.metadata.update(
                owner_id=owner_id,
                upload_id=upload_id,
                expires_at=expires,
            )
        chunks = split_documents(pages)
        if len(chunks) > config.SESSION_MAX_CHUNKS:
            raise IngestionError("PDF menghasilkan terlalu banyak potongan teks.")
        if not chunks:
            raise IngestionError(
                "Tidak ada teks yang dapat dibaca. PDF hasil pindaian "
                "mungkin belum memiliki lapisan teks."
            )

        from src.vector_store import get_session_vector_store

        get_session_vector_store().add_documents(chunks)
        with _connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE {config.SESSION_UPLOAD_TABLE_NAME} "
                    "SET status = 'ready', pages = %s, chunks = %s "
                    "WHERE upload_id = %s AND owner_id = %s "
                    "AND status = 'pending';",
                    (info["pages"], len(chunks), upload_id, owner_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Reservasi unggahan tidak ditemukan.")

        return {
            "name": file_name,
            "pages": info["pages"],
            "chunks": len(chunks),
            "size_mb": info["size_mb"],
        }
    except Exception:
        if reserved:
            try:
                _discard_upload(owner_id, upload_id)
            except Exception:
                logger.exception("Gagal membersihkan unggahan yang tidak selesai")
        raise


def list_session_pdfs(owner_id):
    validate_owner_id(owner_id)
    ensure_schema()
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT file_name, size_bytes, pages FROM "
                f"{config.SESSION_UPLOAD_TABLE_NAME} "
                "WHERE owner_id = %s AND status = 'ready' "
                "AND expires_at > now() ORDER BY created_at;",
                (owner_id,),
            )
            return [
                {
                    "name": name,
                    "folder": "session",
                    "size_mb": round(size / (1024 * 1024), 2),
                    "pages": pages,
                }
                for name, size, pages in cursor.fetchall()
            ]


def cleanup_expired():
    """Delete expired vectors and manifests; safe to run from cron too."""
    ensure_schema()
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {config.SESSION_TABLE_NAME} "
                f"WHERE upload_id IN (SELECT upload_id FROM "
                f"{config.SESSION_UPLOAD_TABLE_NAME} "
                "WHERE expires_at <= now());"
            )
            cursor.execute(
                f"DELETE FROM {config.SESSION_UPLOAD_TABLE_NAME} "
                "WHERE expires_at <= now();"
            )
            removed = cursor.rowcount
            cursor.execute(
                f"DELETE FROM {config.DAILY_USAGE_TABLE_NAME} "
                "WHERE day < %s;",
                (_jakarta_day() - timedelta(days=7),),
            )
            return removed


def maybe_cleanup():
    """Run lightweight cleanup at most once per hour per app process."""
    global _last_cleanup
    now = time.monotonic()
    with _cleanup_lock:
        if now - _last_cleanup < 3600:
            return
        _last_cleanup = now
    cleanup_expired()
