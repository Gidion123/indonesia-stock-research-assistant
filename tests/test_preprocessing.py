from src import config
from src.preprocessing import (
    BROKEN_WORD_PATTERN,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CITATION_MARKER_PATTERN,
    KNOWLEDGE_BASE_DIR,
    clean_text,
    get_pdf_files,
    is_reference_page,
    load_all_pdfs,
    load_and_split_documents,
    new_ingestion_report,
    split_documents,
    stable_chunk_id,
    truncate_at_bibliography,
)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

def test_knowledge_base_directory_exists():
    assert KNOWLEDGE_BASE_DIR.exists()
    assert KNOWLEDGE_BASE_DIR.is_dir()


def test_pdf_files_are_found():
    pdf_files = get_pdf_files()

    assert len(pdf_files) > 0
    assert all(file.suffix.lower() == ".pdf" for file in pdf_files)


def test_primary_knowledge_base_has_five_pdfs():
    pdf_files = get_pdf_files(include_additional=False)

    assert len(pdf_files) == 5
    assert any(p.name == "Riset Saham Data Center Indonesia.pdf" for p in pdf_files)


def test_additional_documents_are_opt_in(monkeypatch, tmp_path):
    """
    Documents in additional/ stay out of the baseline unless asked for.

    This used to assert that a specific PDF (BIPI) sat in additional/,
    which tied the test suite to the contents of a data folder the user
    is supposed to change. Uploading or removing one document broke the
    build. What actually matters is the rule, so the rule is what gets
    tested, on folders this test owns.
    """
    primary_dir = tmp_path / "primary"
    additional_dir = tmp_path / "additional"
    primary_dir.mkdir()
    additional_dir.mkdir()

    (primary_dir / "baseline.pdf").write_bytes(b"%PDF")
    (additional_dir / "tambahan.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(config, "PRIMARY_DIR", primary_dir)
    monkeypatch.setattr(config, "ADDITIONAL_DIR", additional_dir)

    primary = [path.name for path in get_pdf_files(include_additional=False)]
    extended = [path.name for path in get_pdf_files(include_additional=True)]

    assert primary == ["baseline.pdf"]
    assert extended == ["baseline.pdf", "tambahan.pdf"]


def test_additional_folder_may_be_empty(monkeypatch, tmp_path):
    """
    An empty additional/ is a normal state, not a failure.

    It is what a fresh clone looks like, and what the folder looks like
    again after a rebuild removes an uploaded document.
    """
    primary_dir = tmp_path / "primary"
    additional_dir = tmp_path / "additional"
    primary_dir.mkdir()
    additional_dir.mkdir()

    (primary_dir / "baseline.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(config, "PRIMARY_DIR", primary_dir)
    monkeypatch.setattr(config, "ADDITIONAL_DIR", additional_dir)

    assert get_pdf_files(include_additional=True) == [
        primary_dir / "baseline.pdf"
    ]


def test_missing_additional_folder_is_not_an_error(monkeypatch, tmp_path):
    """The folder may not exist at all; that must not raise."""
    primary_dir = tmp_path / "primary"
    primary_dir.mkdir()
    (primary_dir / "baseline.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(config, "PRIMARY_DIR", primary_dir)
    monkeypatch.setattr(config, "ADDITIONAL_DIR", tmp_path / "tidak-ada")

    assert len(get_pdf_files(include_additional=True)) == 1


# ============================================================
# CLEANING
# ============================================================

def test_clean_text_removes_citation_markers():
    raw = (
        "Harga mengalami koreksi tajam. 21 BMRI menawarkan potensi "
        "rebound. 8 Saham cadangan kedua adalah MEDC."
    )

    cleaned, report = clean_text(raw)

    assert report["citation_markers_removed"] == 2
    assert "21 BMRI" not in cleaned
    assert "BMRI menawarkan" in cleaned
    assert "MEDC" in cleaned


def test_clean_text_keeps_real_numbers():
    """
    The marker pattern must not eat data. Numbers that are part of a
    sentence stay untouched.
    """
    raw = "Target TP 1 berada di Rp3.400 dan dividend yield 13,75%."

    cleaned, report = clean_text(raw)

    assert report["citation_markers_removed"] == 0
    assert "Rp3.400" in cleaned
    assert "13,75%" in cleaned


def test_clean_text_repairs_broken_words():
    raw = "Sektor perbanka n menjadi penerima manfaat utama."

    cleaned, report = clean_text(raw)

    assert "perbankan" in cleaned
    assert "perbanka n" not in cleaned
    assert len(report["broken_words_repaired"]) == 1


def test_english_article_and_pronoun_are_not_broken_words():
    text = "Indonesia with a total capacity; where i work. Sektor perbanka n."
    cleaned, report = clean_text(text)
    assert "with a total" in cleaned
    assert "where i work" in cleaned
    assert "perbankan" in cleaned
    assert report["broken_words_repaired"] == ["'perbanka n' -> 'perbankan'"]


def test_clean_text_normalises_whitespace():
    cleaned, _ = clean_text("baris satu\n  baris   dua\xa0tiga")

    assert "\n" not in cleaned
    assert "  " not in cleaned


def test_is_reference_page():
    reference = (
        "1. https://example.com/a accessed June 2026 "
        "2. https://example.com/b diakses Juni 2026"
    )
    content = "Analisis IHSG menunjukkan tekanan koreksi pada awal 2026."

    assert is_reference_page(reference)
    assert is_reference_page(" ".join(f"{n}. https://example.com/{n}" for n in range(6)))
    assert not is_reference_page(content)
    assert not is_reference_page("Analisis bisnis https://a.test dan https://b.test menjelaskan kontrak.")


def test_truncate_at_bibliography():
    text = "Isi analisis yang penting. Works cited 1. https://contoh.com"

    truncated, was_truncated = truncate_at_bibliography(text)

    assert was_truncated
    assert "Works cited" not in truncated
    assert "Isi analisis yang penting." in truncated


def test_cleaning_actually_removes_something_from_real_pdfs():
    """
    Regression guard: pembersihan tidak boleh diam-diam menjadi no-op.

    Yang dijaga di sini hanya efek yang TIDAK bergantung versi pustaka
    ekstraksi PDF. Perbaikan kata patah sengaja TIDAK diikutkan —
    penjelasannya ada di test berikutnya.
    """
    report = new_ingestion_report()
    load_all_pdfs(include_additional=False, report=report)

    assert report["citation_markers_removed"] > 100
    assert report["pages_dropped_reference"] > 0
    assert report["pages_kept"] < report["pages_total"]


def test_perbaikan_kata_patah_bergantung_versi_pypdf():
    """
    Jumlah kata patah pada korpus nyata BUKAN ukuran yang stabil, dan
    karena itu tidak boleh dijadikan assertion.

    Diukur pada knowledge base yang sama:

        pypdf 3.17.4  ->  16 kata patah  ('perbanka n', 'Septembe r', …)
        pypdf 5.4.0   ->   0 kata patah  (semua kata keluar utuh)

    Ekstraksi teks pypdf 5.x tidak lagi memecah kata di batas kolom,
    jadi artefaknya hilang di hulu. Mekanisme perbaikannya sendiri tetap
    benar dan tetap diuji oleh `test_clean_text_repairs_broken_words`
    dengan teks sintetis.

    Kodenya sengaja dipertahankan: PDF dari sumber lain — termasuk yang
    diunggah pengguna lewat UI — masih bisa membawa artefak yang sama.

    JANGAN mengembalikan assertion `> 0` di sini. Itu akan gagal pada
    versi pypdf yang dipatok project ini, dan membuat seluruh test suite
    merah karena alasan yang bukan kesalahan kode.
    """
    report = new_ingestion_report()
    load_all_pdfs(include_additional=False, report=report)

    jumlah = len(report["broken_words_repaired"])

    # Dicatat, bukan di-assert: nilainya sah pada 0 maupun belasan.
    assert jumlah >= 0

    for entry in report["broken_words_repaired"]:
        assert "->" in entry


# ============================================================
# METADATA
# ============================================================

def test_documents_have_required_metadata(documents):
    document = documents[0]

    assert document.page_content.strip() != ""

    for key in (
        "source",
        "file_name",
        "page",
        "category",
        "document_type",
    ):
        assert key in document.metadata


def test_page_metadata_is_valid(documents):
    for document in documents:
        assert isinstance(document.metadata["page"], int)
        assert document.metadata["page"] >= 1


# ============================================================
# CHUNKING
# ============================================================

def test_chunk_configuration():
    assert CHUNK_SIZE == 500
    assert CHUNK_OVERLAP == 80
    assert CHUNK_SIZE == config.CHUNK_SIZE
    assert CHUNK_OVERLAP == config.CHUNK_OVERLAP


def test_documents_are_split_into_chunks(documents):
    assert len(split_documents(documents)) > len(documents)


def test_chunks_have_content_and_metadata(chunks):
    for chunk in chunks:
        assert chunk.page_content.strip() != ""

        for key in (
            "source",
            "file_name",
            "page",
            "category",
            "document_type",
            "chunk_id",
        ):
            assert key in chunk.metadata


def test_chunk_ids_are_unique(chunks):
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_id_is_stable_across_runs():
    """
    Chunk ids are content-addressed, so building the knowledge base twice
    must produce exactly the same ids. Evaluation depends on this.
    """
    first = load_and_split_documents(include_additional=False)
    second = load_and_split_documents(include_additional=False)

    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]


def test_chunk_id_changes_when_text_changes():
    from langchain_core.documents import Document

    base = Document(
        page_content="Target TP 1 BBRI berada di Rp3.400.",
        metadata={"source": "a.pdf", "page": 1},
    )
    changed = Document(
        page_content="Target TP 1 BBRI berada di Rp3.800.",
        metadata={"source": "a.pdf", "page": 1},
    )

    assert stable_chunk_id(base) != stable_chunk_id(changed)
    assert len(stable_chunk_id(base)) == 12


def test_load_and_split_documents(chunks):
    assert len(chunks) > 0
