"""
Knowledge-base loading, cleaning, and chunking.

The cleaning stage is ported from `kurasi_halaman()` in
notebooks/01_experiment.ipynb. Without it the raw PDF text goes straight
into the vector store. Without it, citation markers and bibliography
pages become searchable content. Page counts depend on the active PDFs
and are reported during ingestion.

Broken words split across table columns are repaired in the same pass,
but how many there are depends on how the installed pypdf lays out table
text. On the current version it is zero, so there is no number worth
quoting for it.
"""

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src import config


# Backwards-compatible aliases (older tests and scripts import these).
KNOWLEDGE_BASE_DIR = config.PRIMARY_DIR
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP


# ============================================================
# CLEANING PATTERNS
# ============================================================

# "...jenuh jual yang pekat. 21 BMRI menawarkan..." -> the 21 is a Deep
# Research footnote marker flattened into the text, not a data point.
# Dangerous here because the grounding prompt tells the model to copy
# numbers exactly as written.
CITATION_MARKER_PATTERN = re.compile(r"(?<=[.!?])\s+\d{1,3}\s+(?=[A-Z])")
CITATION_MARKER_TAIL_PATTERN = re.compile(r"(?<=[.!?])\s+\d{1,3}\s*$")

# "perbanka n" -> "perbankan": a word split by a table column boundary.
# Such a word can never match either semantic or lexical search.
BROKEN_WORD_PATTERN = re.compile(r"\b([A-Za-z]{4,})\s([a-z])\b(?![\w-])")


# ============================================================
# CLEANING
# ============================================================

def clean_text(text):
    """
    Normalise one page of extracted PDF text.

    Returns (cleaned_text, report) where report counts what was changed,
    so the effect of cleaning is measurable instead of invisible.
    """
    report = {
        "citation_markers_removed": 0,
        "broken_words_repaired": [],
    }

    text = str(text or "").replace("\xa0", " ")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r" {2,}", " ", text)

    if config.CLEAN_REMOVE_CITATION_MARKERS:
        report["citation_markers_removed"] = (
            len(CITATION_MARKER_PATTERN.findall(text))
            + len(CITATION_MARKER_TAIL_PATTERN.findall(text))
        )
        text = CITATION_MARKER_PATTERN.sub(" ", text)
        text = CITATION_MARKER_TAIL_PATTERN.sub("", text)

    if config.CLEAN_REPAIR_BROKEN_WORDS:
        def _join(match):
            # English article/pronoun: "with a" and "where i" can be
            # legitimate text, not the tail of a broken table word.
            if match.group(2) in {"a", "i"}:
                return match.group(0)
            report["broken_words_repaired"].append(
                f"{match.group(0)!r} -> {match.group(1) + match.group(2)!r}"
            )
            return match.group(1) + match.group(2)

        text = BROKEN_WORD_PATTERN.sub(_join, text)

    text = re.sub(r" {2,}", " ", text)

    return text.strip(), report


def is_reference_page(text):
    """
    Detect pages made of bibliography links, with or without access dates.
    Such pages are not answers but compete for retrieval slots.
    """
    lower = text.lower()
    links = lower.count("http")
    return (
        links >= 5
        or (links >= 2 and (lower.count("accessed") + lower.count("diakses")) >= 1)
    )


def truncate_at_bibliography(text):
    """
    Cut a page at the earliest bibliography marker.
    Returns (text, was_truncated).
    """
    positions = [
        text.find(marker)
        for marker in config.BIBLIOGRAPHY_MARKERS
        if text.find(marker) >= 0
    ]

    if not positions:
        return text, False

    position = min(positions)

    if position == 0:
        return "", True

    return text[:position].strip(), True


# ============================================================
# CHUNK IDENTITY
# ============================================================

def stable_chunk_id(document):
    """
    Content-addressed chunk id: sha1(source|page|text)[:12].

    Ported from the P0 harness in the notebook. It is what makes
    retrieval measurable at chunk level, because the same chunk gets the
    same id whether it is built locally for evaluation or read back from
    PgVector.
    """
    source = str(document.metadata.get("source", "dokumen"))
    page = str(document.metadata.get("page", ""))
    text = document.page_content.strip()

    payload = f"{source}|{page}|{text}".encode("utf-8")

    return hashlib.sha1(payload).hexdigest()[:12]


# ============================================================
# LOADING
# ============================================================

def get_pdf_files(include_additional=None):
    """
    Return the PDF files that make up the active knowledge base.
    """
    if include_additional is None:
        include_additional = config.INCLUDE_ADDITIONAL_DOCUMENTS

    pdf_files = sorted(config.PRIMARY_DIR.glob("*.pdf"))

    if include_additional and config.ADDITIONAL_DIR.exists():
        pdf_files += sorted(config.ADDITIONAL_DIR.glob("*.pdf"))

    return pdf_files


def load_pdf(file_path, collect_report=None, source_name=None, category=None):
    """
    Load one PDF from disk or a file-like object using the same pipeline.

    In-memory visitor uploads supply their display name and category;
    curated files derive both from their path as before.
    """
    if hasattr(file_path, "read"):
        if not source_name or not category:
            raise ValueError("PDF dalam memori memerlukan nama dan kategori.")
        reader = PdfReader(file_path)
    else:
        file_path = Path(file_path)
        reader = PdfReader(file_path)
        source_name = file_path.name
        category = file_path.parent.name

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""

        text, report = clean_text(raw_text)

        if collect_report is not None:
            collect_report["citation_markers_removed"] += report[
                "citation_markers_removed"
            ]
            collect_report["broken_words_repaired"].extend(
                report["broken_words_repaired"]
            )

        if config.CLEAN_TRUNCATE_AT_BIBLIOGRAPHY:
            text, truncated = truncate_at_bibliography(text)

            if truncated and collect_report is not None:
                collect_report["pages_truncated"] += 1

        if not text.strip() or len(text) < config.MINIMUM_PAGE_LENGTH:
            if collect_report is not None:
                collect_report["pages_dropped_short"] += 1
            continue

        if config.CLEAN_DROP_REFERENCE_PAGES and is_reference_page(text):
            if collect_report is not None:
                collect_report["pages_dropped_reference"] += 1
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "file_name": source_name,
                    "page": page_number,
                    "category": category,
                    "document_type": "stock_research",
                },
            )
        )

    return documents


def new_ingestion_report():
    return {
        "pdf_files": 0,
        "pages_total": 0,
        "pages_kept": 0,
        "pages_dropped_short": 0,
        "pages_dropped_reference": 0,
        "pages_truncated": 0,
        "citation_markers_removed": 0,
        "broken_words_repaired": [],
    }


def load_all_pdfs(include_additional=None, report=None):
    """
    Load every PDF of the active knowledge base as cleaned Documents.
    """
    pdf_files = get_pdf_files(include_additional)

    documents = []

    for file_path in pdf_files:
        if report is not None:
            report["pdf_files"] += 1
            report["pages_total"] += len(PdfReader(file_path).pages)

        documents.extend(load_pdf(file_path, collect_report=report))

    if report is not None:
        report["pages_kept"] = len(documents)

    return documents


# ============================================================
# SPLITTING
# ============================================================

def get_text_splitter():
    """
    Create the text splitter used by the application.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def split_documents(documents):
    """
    Split documents into chunks, preserving metadata and attaching a
    stable chunk_id to each chunk.
    """
    chunks = get_text_splitter().split_documents(documents)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = stable_chunk_id(chunk)

    return chunks


def load_and_split_documents(include_additional=None, report=None):
    """
    Load, clean, and split the whole knowledge base.
    """
    documents = load_all_pdfs(
        include_additional=include_additional,
        report=report,
    )

    chunks = split_documents(documents)

    if report is not None:
        report["chunks"] = len(chunks)

    return chunks
