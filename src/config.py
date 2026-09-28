"""
Central configuration for the Stock Research Customer Assistant.

Every value that affects retrieval and generation quality lives here, so
an experiment result can always be traced back to the configuration that
produced it. Modules must import from this file instead of defining their
own constants.

Not literally every constant in the project is here. `src/ingestion.py`
keeps UKURAN_MAKS_PDF_MB and MINIMUM_CHUNK_BARU, and `src/rag_chain.py`
keeps TRANSIENT_ERROR_HINTS. Those are operational limits and do not
change the numbers an evaluation produces.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_ROOT = PROJECT_ROOT / "data" / "knowledge_base"
PRIMARY_DIR = KNOWLEDGE_BASE_ROOT / "primary"
ADDITIONAL_DIR = KNOWLEDGE_BASE_ROOT / "additional"

GENERATED_DIR = PROJECT_ROOT / "data" / "generated"

# Entity resolution happens at query time (src/entity_resolver.py), so
# there is no registry file built up front.

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
EVAL_DATASET_DIR = EVALUATION_DIR / "dataset"
EVAL_RESULTS_DIR = EVALUATION_DIR / "results"

# One file, one truth.
#
# There used to be two: retrieval_eval.json supplied the questions and
# answer_eval.json supplied expected_keywords. Gold chunks were derived
# from the keywords in the second file, but retrieval was run with the
# questions from the first, so the two could drift apart without a single
# test failing - and they did. Two questions (equity_01, equity_04) had
# different wording, so gold was derived for a question that was never
# asked. That is where the HitRate@8 gap of 0.80 vs 0.90 between
# evaluate.py and diagnose_retrieval came from.
#
# Both now read the same file, so that drift cannot happen again. This is
# guarded by tests/test_eval_dataset.py.
EVAL_QUESTIONS_PATH = EVAL_DATASET_DIR / "eval_questions.json"

# The old names are kept so existing callers do not break; both now point
# at the same file.
RETRIEVAL_EVAL_PATH = EVAL_QUESTIONS_PATH
ANSWER_EVAL_PATH = EVAL_QUESTIONS_PATH

GOLD_CHUNKS_PATH = EVAL_DATASET_DIR / "gold_chunks.json"

RETRIEVAL_RESULTS_PATH = EVAL_RESULTS_DIR / "retrieval_results.json"
ANSWER_RESULTS_PATH = EVAL_RESULTS_DIR / "answer_results.json"


# ============================================================
# KNOWLEDGE BASE SCOPE
# ============================================================

# The primary folder is the curator-managed knowledge base. Historical
# evaluation numbers were measured against its original four documents;
# adding new primary PDFs requires a fresh evaluation before publishing
# new scores.
#
# The additional folder is opt-in for historical offline experiments.
# Public visitor uploads never use it.
# It mirrors "Phase 6 - menambah dokumen tambahan & re-validasi" in the
# notebook: documents are added deliberately and the system is
# re-evaluated, never silently mixed into the baseline. It is often
# empty, which is a normal state and not a problem.
#
# Enable with:  INCLUDE_ADDITIONAL_DOCUMENTS=true  in .env
INCLUDE_ADDITIONAL_DOCUMENTS = os.getenv(
    "INCLUDE_ADDITIONAL_DOCUMENTS",
    "false",
).strip().lower() in {"1", "true", "yes"}


# ============================================================
# PREPROCESSING
# ============================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Text cleaning (ported from `kurasi_halaman` in notebooks/01_experiment.ipynb).
# Every flag can be turned off to measure its individual contribution.
CLEAN_REMOVE_CITATION_MARKERS = True
CLEAN_REPAIR_BROKEN_WORDS = True
CLEAN_DROP_REFERENCE_PAGES = True
CLEAN_TRUNCATE_AT_BIBLIOGRAPHY = True

MINIMUM_PAGE_LENGTH = 50

BIBLIOGRAPHY_MARKERS = (
    "Works cited",
    "Karya yang dikutip",
    "Daftar Pustaka",
    "Bibliography",
    "Referensi",
)


# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

EMBEDDING_DIMENSION = 384

# Kept as "cpu" by default: changing the device changes nothing about
# quality but would require re-ingesting to keep the stored vectors
# bit-identical. Override with EMBEDDING_DEVICE=mps in .env if wanted.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

EMBEDDING_NORMALIZE = True


# ============================================================
# VECTOR STORE
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")
TABLE_NAME = "stock_knowledge"
SESSION_TABLE_NAME = "stock_session_knowledge"
SESSION_UPLOAD_TABLE_NAME = "stock_session_uploads"
DAILY_USAGE_TABLE_NAME = "stock_daily_usage"
PGVECTOR_COLLECTION = os.getenv("PGVECTOR_COLLECTION", "stock_research")

# Public demo limits. Anonymous browser tabs are separate sessions;
# global caps also bound usage when visitors open new tabs.
SESSION_UPLOAD_TTL_HOURS = 24
SESSION_MAX_PDFS = 2
SESSION_MAX_PDF_MB = 5
SESSION_MAX_PDF_PAGES = 30
SESSION_MAX_EXTRACTED_CHARS = 200_000
SESSION_MAX_CHUNKS = 500
GLOBAL_DAILY_UPLOAD_LIMIT = 30
SESSION_DAILY_QUESTION_LIMIT = 10
GLOBAL_DAILY_QUESTION_LIMIT = 100


# ============================================================
# RETRIEVAL
# ============================================================

# Frozen in "Phase 1 - Retrieval Optimization" (notebook): SIM_k8 beat
# every MMR configuration that was swept.
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_K = 8


# ============================================================
# LLM
# ============================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()

GROQ_MODEL = "openai/gpt-oss-20b"

DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT_SECONDS = 90

# ============================================================
# MARKET DATA
# ============================================================

# Yahoo Finance tetap menjadi sumber utama.
# EODHD hanya dipakai sebagai fallback ketika Yahoo gagal karena
# rate limit / gangguan jaringan.
MARKET_DATA_FALLBACK = os.getenv(
    "MARKET_DATA_FALLBACK",
    "eodhd",
).strip().lower()

EODHD_API_KEY = os.getenv("EODHD_API_KEY", "").strip()

EODHD_BASE_URL = os.getenv(
    "EODHD_BASE_URL",
    "https://eodhd.com/api",
).rstrip("/")

MARKET_DATA_TIMEOUT_SECONDS = int(
    os.getenv("MARKET_DATA_TIMEOUT_SECONDS", "10")
)


# ============================================================
# EVALUATION
# ============================================================

REFUSAL_MESSAGE = "Maaf, informasi itu tidak ada di dokumen saya."

# A run is only valid when almost every row produced a real answer.
# Anything above this ratio of failed rows means the run must be repeated,
# not reported.
MAX_FAILED_ROW_RATIO = 0.05

# Hit rate, recall and precision are reported at each of these cut-offs.
# MRR is not: it is computed once per question from the full ranking, so
# there is a single MRR number, not one per k (see src/evaluation.py).
EVALUATION_K_VALUES = (1, 3, 5, 8)

ROUTER_EVAL_PATH = EVAL_DATASET_DIR / "router_eval.json"


def describe() -> dict:
    """
    Return the configuration as a plain dict, to be stored alongside
    evaluation results for reproducibility.
    """
    return {
        "include_additional_documents": INCLUDE_ADDITIONAL_DOCUMENTS,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "clean_remove_citation_markers": CLEAN_REMOVE_CITATION_MARKERS,
        "clean_repair_broken_words": CLEAN_REPAIR_BROKEN_WORDS,
        "clean_drop_reference_pages": CLEAN_DROP_REFERENCE_PAGES,
        "clean_truncate_at_bibliography": CLEAN_TRUNCATE_AT_BIBLIOGRAPHY,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_device": EMBEDDING_DEVICE,
        "retriever_search_type": RETRIEVER_SEARCH_TYPE,
        "retriever_k": RETRIEVER_K,
        "llm_provider": LLM_PROVIDER,
        "llm_model": (
            DEEPSEEK_MODEL
            if LLM_PROVIDER == "deepseek"
            else GROQ_MODEL
        ),
        "table_name": TABLE_NAME,
    }
