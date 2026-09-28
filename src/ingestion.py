"""
Knowledge-base ingestion service. Shared by the CLI and the UI.

The curated corpus is rebuilt by `scripts/ingest_knowledge_base.py`.
`tambah_pdf()` remains for explicit offline additions to primary/ or
additional/. The public Streamlit upload path is in session_documents.py:
it shares the PDF cleaning and chunking functions, but writes to a
different vector table so visitors cannot change the curated corpus.
"""

import hashlib
from io import BytesIO
import shutil
import tempfile
import time
from pathlib import Path

from src import config
from src.preprocessing import (
    load_and_split_documents,
    new_ingestion_report,
    split_documents,
)


UKURAN_MAKS_PDF_MB = 25

MINIMUM_CHUNK_BARU = 1


class IngestionError(Exception):
    """A failure that can be explained to the user without jargon."""


# ============================================================
# 1. VALIDATION
# ============================================================

def validasi_pdf(path, max_mb=None, max_pages=None, file_name=None):
    """
    Check the file before doing any work on it.

    Checking early is much cheaper than finding out the PDF is broken
    halfway through embedding.
    """
    if isinstance(path, (bytes, bytearray)):
        if not file_name or Path(file_name).suffix.lower() != ".pdf":
            raise IngestionError("Hanya berkas PDF yang didukung.")
        ukuran_mb = len(path) / (1024 * 1024)
        sumber_pdf = BytesIO(path)
    else:
        path = Path(path)
        if not path.exists():
            raise IngestionError("Berkasnya tidak ditemukan.")
        if path.suffix.lower() != ".pdf":
            raise IngestionError("Hanya berkas PDF yang didukung.")
        ukuran_mb = path.stat().st_size / (1024 * 1024)
        sumber_pdf = str(path)

    if ukuran_mb == 0:
        raise IngestionError("Berkasnya kosong.")

    batas_mb = max_mb if max_mb is not None else UKURAN_MAKS_PDF_MB
    if ukuran_mb > batas_mb:
        raise IngestionError(
            f"Berkasnya terlalu besar ({ukuran_mb:.1f} MB). "
            f"Maksimal {batas_mb} MB."
        )

    try:
        from pypdf import PdfReader

        reader = PdfReader(sumber_pdf)
        jumlah_halaman = len(reader.pages)

    except Exception as exception:
        raise IngestionError(
            "Berkas PDF-nya tidak bisa dibaca — mungkin rusak atau "
            "terkunci kata sandi."
        ) from exception

    if jumlah_halaman == 0:
        raise IngestionError("PDF-nya tidak punya halaman.")

    if max_pages is not None and jumlah_halaman > max_pages:
        raise IngestionError(
            f"PDF memiliki {jumlah_halaman} halaman. "
            f"Maksimal {max_pages} halaman per unggahan."
        )

    return {"pages": jumlah_halaman, "size_mb": round(ukuran_mb, 2)}


def sidik_jari_berkas(path):
    """
    SHA-1 of the file contents, to catch the same document uploaded twice
    under different names.
    """
    digest = hashlib.sha1()

    with open(path, "rb") as file:
        for blok in iter(lambda: file.read(65536), b""):
            digest.update(blok)

    return digest.hexdigest()[:16]


_JEDA_SETELAH_GAGAL = 60
_gagal_terakhir = None


def sumber_di_vector_store(timeout=2):
    """
    The document names the vector store actually holds.

    A PDF sitting in a folder is not the same thing as a PDF the
    assistant can answer from: a file copied in by hand is on disk but
    has never been embedded. Asking the database is the only way to tell
    them apart.

    Returns a set of file names, or None when the database cannot be
    asked. None means "no opinion" - the caller then falls back to
    listing files, which is better than showing nothing.

    A failed attempt is remembered for a minute so that a database which
    is down does not cost every page render a connection timeout.
    """
    global _gagal_terakhir

    if (
        _gagal_terakhir is not None
        and time.monotonic() - _gagal_terakhir < _JEDA_SETELAH_GAGAL
    ):
        return None

    try:
        import psycopg

        from src.vector_store import TABLE_NAME, get_psycopg_connection_url

        with psycopg.connect(
            get_psycopg_connection_url(),
            connect_timeout=timeout,
        ) as connection:
            with connection.cursor() as cursor:
                # The metadata column name comes from langchain_postgres
                # and is not ours to assume, so it is looked up rather
                # than hard-coded.
                cursor.execute(
                    "SELECT column_name, data_type "
                    "FROM information_schema.columns "
                    "WHERE table_name = %s;",
                    (TABLE_NAME,),
                )
                kolom = dict(cursor.fetchall())

                if "source" in kolom:
                    cursor.execute(
                        f"SELECT DISTINCT source FROM {TABLE_NAME} "
                        f"WHERE source IS NOT NULL;"
                    )
                else:
                    json_kolom = next(
                        (
                            nama
                            for nama, tipe in kolom.items()
                            if tipe in ("jsonb", "json")
                        ),
                        None,
                    )

                    if not json_kolom:
                        return None

                    cursor.execute(
                        f"SELECT DISTINCT {json_kolom} ->> 'source' "
                        f"FROM {TABLE_NAME} "
                        f"WHERE {json_kolom} ->> 'source' IS NOT NULL;"
                    )

                _gagal_terakhir = None

                return {baris[0] for baris in cursor.fetchall()}

    except Exception:
        _gagal_terakhir = time.monotonic()

        return None


def daftar_dokumen(include_additional=None, verifikasi=False):
    """
    The PDFs that currently make up the knowledge base.

    With `verifikasi=True` the list is narrowed to documents the vector
    store really contains. That is what the sidebar wants: it answers
    "what can this assistant answer from", and a file that was copied
    into the folder by hand has never been embedded, so listing it
    promises something the assistant cannot deliver.

    If the database cannot be reached the file listing is returned
    unfiltered, because a sidebar that shows a little too much is better
    than a sidebar that shows nothing.
    """
    from src.preprocessing import get_pdf_files

    if include_additional is None:
        include_additional = config.INCLUDE_ADDITIONAL_DOCUMENTS

    berkas = get_pdf_files(include_additional=include_additional)

    if verifikasi:
        sumber = sumber_di_vector_store()

        if sumber is not None:
            berkas = [p for p in berkas if p.name in sumber]

    return [
        {
            "name": p.name,
            "folder": p.parent.name,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        }
        for p in berkas
    ]


def _sudah_ada(path, tujuan):
    """
    Duplicate by name OR by contents. Name alone is not enough: the same
    file often gets re-uploaded under a different name.
    """
    if (tujuan / Path(path).name).exists():
        return f"Dokumen dengan nama '{Path(path).name}' sudah ada."

    sidik_baru = sidik_jari_berkas(path)

    for lama in tujuan.glob("*.pdf"):
        if sidik_jari_berkas(lama) == sidik_baru:
            return f"Isi dokumen ini sama dengan '{lama.name}'."

    return None


# ============================================================
# 2. EMBEDDING INTO PGVECTOR
# ============================================================

def _simpan_ke_vector_store(chunks):
    from src.vector_store import get_vector_store

    if not chunks:
        raise IngestionError(
            "Tidak ada teks yang bisa diambil dari dokumen itu. "
            "Kemungkinan isinya berupa hasil pindaian tanpa teks."
        )

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return len(chunks)


# ============================================================
# 3. ADD ONE DOCUMENT (offline/legacy path, never public UI)
# ============================================================

def tambah_pdf(
    sumber,
    nama_berkas=None,
    ke_folder_tambahan=False,
):
    """
    Add one PDF to the knowledge base.

    `sumber` may be a path, bytes, or a file object. Public visitors must
    use `session_documents.add_session_pdf()` instead.

    The flow is exactly the CLI's: clean -> chunk -> stable chunk_id ->
    embed -> PgVector. There is no second processing path.

    Returns a summary; raises `IngestionError` with a message that is
    safe to show to the user.
    """
    tujuan = (
        config.ADDITIONAL_DIR if ke_folder_tambahan else config.PRIMARY_DIR
    )
    tujuan.mkdir(parents=True, exist_ok=True)

    sementara = None

    try:
        # ---- write to a temporary file first ----
        if isinstance(sumber, (str, Path)):
            sementara = Path(sumber)
            hapus_sementara = False
            nama_berkas = nama_berkas or sementara.name
        else:
            isi = sumber.read() if hasattr(sumber, "read") else sumber
            nama_berkas = nama_berkas or getattr(sumber, "name", "dokumen.pdf")

            handle = tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            )
            handle.write(isi)
            handle.close()

            sementara = Path(handle.name)
            hapus_sementara = True

        info = validasi_pdf(sementara)

        nama_aman = Path(nama_berkas).name

        if not nama_aman.lower().endswith(".pdf"):
            nama_aman += ".pdf"

        duplikat = _sudah_ada(sementara, tujuan)

        if duplikat:
            raise IngestionError(duplikat)

        # ---- copy it under the final name, then process ----
        #
        # `load_pdf` takes `source` and `category` from the file path, and
        # that metadata is what shows up as the citation [name.pdf hal.N].
        # Processing straight from the temporary file would produce
        # citations reading "tmpXXXX.pdf". So the file is copied first,
        # and deleted again if processing fails - the knowledge base must
        # not keep a document whose chunks never made it in.
        from src.preprocessing import load_pdf

        final = tujuan / nama_aman
        shutil.copyfile(sementara, final)

        try:
            laporan = new_ingestion_report()
            halaman = load_pdf(final, collect_report=laporan)

            chunks = split_documents(halaman)

            if len(chunks) < MINIMUM_CHUNK_BARU:
                raise IngestionError(
                    "Tidak ada teks yang bisa diambil dari dokumen itu. "
                    "Kemungkinan isinya berupa hasil pindaian tanpa teks."
                )

            jumlah = _simpan_ke_vector_store(chunks)

        except Exception:
            final.unlink(missing_ok=True)
            raise

        return {
            "file_name": nama_aman,
            "pages": info["pages"],
            "size_mb": info["size_mb"],
            "chunks": jumlah,
            "folder": tujuan.name,
        }

    finally:
        if sementara is not None and hapus_sementara:
            try:
                sementara.unlink(missing_ok=True)
            except OSError:
                pass


# ============================================================
# 4. FULL REBUILD (used by the CLI)
# ============================================================

def bangun_ulang(include_additional=None, hapus_data_lama=True):
    """
    Rebuild the whole knowledge base from the PDF folders.

    This is what `scripts/ingest_knowledge_base.py` calls.
    """
    import psycopg

    from src.vector_store import (
        TABLE_NAME,
        get_psycopg_connection_url,
        get_vector_store,
    )

    if include_additional is None:
        include_additional = config.INCLUDE_ADDITIONAL_DOCUMENTS

    laporan = new_ingestion_report()
    chunks = load_and_split_documents(
        include_additional=include_additional,
        report=laporan,
    )

    if hapus_data_lama:
        with psycopg.connect(get_psycopg_connection_url()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {TABLE_NAME};")

            connection.commit()

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    laporan["chunks"] = len(chunks)

    return laporan


__all__ = [
    "IngestionError",
    "UKURAN_MAKS_PDF_MB",
    "validasi_pdf",
    "sidik_jari_berkas",
    "daftar_dokumen",
    "sumber_di_vector_store",
    "tambah_pdf",
    "bangun_ulang",
]
