"""Public-upload isolation and validation without a live LLM or database."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import BytesIO
from threading import Lock

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from src import config
from src.ingestion import IngestionError
from src.retriever import SessionRetriever
from src.session_documents import (
    _safe_filename,
    add_session_pdf,
    new_owner_id,
    validate_owner_id,
)


def _pdf_bytes(pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_owner_ids_are_random_and_required():
    ids = {new_owner_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(validate_owner_id(value) == value for value in ids)
    for bad in (None, "", "user_1", "0" * 31, "0" * 32 + " OR TRUE"):
        with pytest.raises(ValueError):
            SessionRetriever(bad)


def test_filename_cannot_escape_or_inject_markup():
    assert _safe_filename("../../laporan.pdf") == "laporan.pdf"
    assert _safe_filename(r"C:\\temp\\laporan.pdf") == "laporan.pdf"
    assert _safe_filename("<script>.pdf") == "_script_.pdf"
    with pytest.raises(IngestionError):
        _safe_filename("catatan.txt")


def test_public_size_and_page_limits_run_before_database(monkeypatch):
    import src.session_documents as sessions

    monkeypatch.setattr(
        sessions, "_reserve_upload", lambda *args: pytest.fail("DB contacted")
    )
    owner = new_owner_id()

    with pytest.raises(IngestionError, match="5 MB"):
        add_session_pdf(b"x" * (config.SESSION_MAX_PDF_MB * 1024 * 1024 + 1),
                        "besar.pdf", owner)

    with pytest.raises(IngestionError, match="30 halaman"):
        add_session_pdf(_pdf_bytes(config.SESSION_MAX_PDF_PAGES + 1),
                        "terlalu-panjang.pdf", owner)


def test_upload_is_validated_in_memory_when_reservation_fails(monkeypatch):
    import src.session_documents as sessions

    seen_bytes = []
    real_validate = sessions.validasi_pdf

    def record_bytes(data, **kwargs):
        seen_bytes.append(data)
        return real_validate(data, **kwargs)

    monkeypatch.setattr(sessions, "validasi_pdf", record_bytes)
    monkeypatch.setattr(
        sessions,
        "_reserve_upload",
        lambda *args: (_ for _ in ()).throw(IngestionError("penuh")),
    )
    data = _pdf_bytes()
    with pytest.raises(IngestionError, match="penuh"):
        add_session_pdf(data, "nota.pdf", new_owner_id())
    assert seen_bytes == [data]


def test_oversized_extracted_text_discards_reservation(monkeypatch):
    import src.preprocessing as preprocessing
    import src.session_documents as sessions

    discarded = []
    monkeypatch.setattr(
        sessions,
        "_reserve_upload",
        lambda *args: datetime.now(timezone.utc) + timedelta(hours=1),
    )
    monkeypatch.setattr(
        sessions,
        "_discard_upload",
        lambda owner, upload: discarded.append((owner, upload)),
    )
    monkeypatch.setattr(
        preprocessing,
        "load_pdf",
        lambda *args, **kwargs: [Document(
            page_content="x" * (config.SESSION_MAX_EXTRACTED_CHARS + 1),
            metadata={"source": "besar.pdf", "page": 1},
        )],
    )

    owner = new_owner_id()
    with pytest.raises(IngestionError, match="terlalu panjang"):
        add_session_pdf(_pdf_bytes(), "besar.pdf", owner)
    assert len(discarded) == 1
    assert discarded[0][0] == owner


def test_two_simultaneous_sessions_never_retrieve_each_others_uploads(
    monkeypatch,
):
    import src.retriever as retriever_module

    owners = [new_owner_id(), new_owner_id()]
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    primary = Document(
        page_content="Shared curated research",
        metadata={"source": "utama.pdf"},
    )
    personal = {
        owner: Document(
            page_content=f"Private research {index}",
            metadata={"source": f"rahasia-{index}.pdf", "owner_id": owner},
        )
        for index, owner in enumerate(owners)
    }

    class PrimaryStore:
        def similarity_search_with_score(self, query, k, filter):
            assert filter == {"category": "primary"}
            return [(primary, 0.2)]

    class SessionStore:
        def __init__(self):
            self.calls = []
            self.lock = Lock()

        def similarity_search_with_score(self, query, k, filter):
            # This fake intentionally rejects an unscoped search. It
            # mirrors the SQL WHERE that the real PGVectorStore performs.
            assert set(filter) == {"$and"}
            owner_filter, expiry_filter = filter["$and"]
            assert set(owner_filter) == {"owner_id"}
            assert set(expiry_filter) == {"expires_at"}
            assert expiry_filter["expires_at"]["$gt"] < expires
            owner = owner_filter["owner_id"]
            with self.lock:
                self.calls.append(owner)
            return [(personal[owner], 0.1)]

    session_store = SessionStore()
    monkeypatch.setattr(retriever_module, "get_vector_store", PrimaryStore)
    monkeypatch.setattr(
        retriever_module, "get_session_vector_store", lambda: session_store
    )

    def ask(owner):
        docs = SessionRetriever(owner).invoke("Tampilkan semua dokumen")
        return {doc.metadata["source"] for doc in docs}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ask, owners[i % 2]) for i in range(200)]
        results = [future.result() for future in futures]

    for index, result in enumerate(results):
        own_index = index % 2
        assert result == {"utama.pdf", f"rahasia-{own_index}.pdf"}
    assert len(session_store.calls) == 200


def test_ticker_powr_prioritizes_matching_chunks_without_unscoped_search(monkeypatch):
    import src.retriever as retriever_module

    owner = new_owner_id()
    unrelated = Document(page_content="Target harga emiten lain.", metadata={"source": "a.pdf"})
    powr = Document(page_content="Target harga POWR menurut riset.", metadata={"source": "b.pdf"})
    requested = []

    class PrimaryStore:
        def similarity_search_with_score(self, query, k, filter):
            assert filter == {"category": "primary"}
            requested.append(k)
            return [(unrelated, 0.1), (powr, 0.4)]

    class SessionStore:
        def similarity_search_with_score(self, query, k, filter):
            assert filter["$and"][0] == {"owner_id": owner}
            return []

    monkeypatch.setattr(retriever_module, "get_vector_store", PrimaryStore)
    monkeypatch.setattr(retriever_module, "get_session_vector_store", SessionStore)
    docs = SessionRetriever(owner).invoke("Apa target harga POWR?")

    assert docs[0] is powr
    assert requested == [config.RETRIEVER_K * 6]
