"""
Privacy-first Langfuse helpers for application observability.

Tracing is intentionally opt-in. The application keeps working when tracing is
not configured or when Langfuse is temporarily unavailable. Raw document
contents, API keys, owner IDs, and full user queries are never added here.
"""

import hashlib
import os
from typing import Any

from langfuse import get_client


_TRUE_VALUES = {"1", "true", "yes", "on"}


def langfuse_tracing_enabled():
    """Return True only when tracing is explicitly enabled and configured."""
    # Unit tests should stay deterministic and must not emit external telemetry.
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False

    enabled = os.getenv("LANGFUSE_TRACING_ENABLED", "false").strip().lower()
    if enabled not in _TRUE_VALUES:
        return False

    return all(
        os.getenv(name)
        for name in (
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_BASE_URL",
        )
    )


def get_langfuse_client_if_enabled():
    """Return the Langfuse singleton client, or None when tracing is disabled."""
    if not langfuse_tracing_enabled():
        return None

    try:
        return get_client()
    except Exception:
        # Observability must never become a dependency for serving answers.
        return None


def text_fingerprint(value):
    """Stable short fingerprint for correlation without storing the raw text."""
    normalized = str(value or "").strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def safe_document_reference(document, rank=None):
    """Return only non-content metadata that is useful for retrieval debugging."""
    metadata = getattr(document, "metadata", {}) or {}

    reference = {
        "chunk_id": _json_safe_scalar(metadata.get("chunk_id")),
        "source": _json_safe_scalar(metadata.get("source")),
        "page": _json_safe_scalar(metadata.get("page")),
        "category": _json_safe_scalar(metadata.get("category")),
    }

    if rank is not None:
        reference["rank"] = rank

    return {key: value for key, value in reference.items() if value is not None}


def safe_observation_update(observation, **kwargs):
    """Best-effort Langfuse update that never breaks the application path."""
    if observation is None:
        return

    try:
        observation.update(**kwargs)
    except Exception:
        pass


def safe_error_type(error):
    """Keep only the exception class name; provider payloads are not exported."""
    if error is None:
        return None

    if isinstance(error, BaseException):
        return type(error).__name__

    text = str(error).strip()
    if not text:
        return None

    return text.split(":", 1)[0][:120]


def _json_safe_scalar(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)
