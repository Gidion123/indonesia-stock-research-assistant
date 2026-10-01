"""
Privacy-first Langfuse helpers for application observability.

Tracing is intentionally opt-in. The application keeps working when tracing is
not configured or when Langfuse is temporarily unavailable. Raw document
contents, API keys, owner IDs, full user queries, prompts, model context and
full model answers are never added here.
"""

import hashlib
import os
import time
from datetime import datetime, timezone
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


def configured_llm_identity(provider=None, model=None):
    """Return the configured provider/model without exposing credentials."""
    from src import config

    provider_name = str(provider or config.LLM_PROVIDER or "unknown").strip().lower()

    if model:
        model_name = str(model).strip()
    elif provider_name == "deepseek":
        model_name = config.DEEPSEEK_MODEL
    elif provider_name == "groq":
        model_name = config.GROQ_MODEL
    else:
        model_name = "unknown"

    return provider_name, model_name


def payload_character_count(value):
    """Count prompt/message characters locally without exporting their contents."""
    if value is None:
        return 0

    if isinstance(value, str):
        return len(value)

    if isinstance(value, bytes):
        return len(value)

    if isinstance(value, dict):
        return sum(payload_character_count(item) for item in value.values())

    if isinstance(value, (list, tuple, set)):
        return sum(payload_character_count(item) for item in value)

    content = getattr(value, "content", None)
    if content is not None:
        return payload_character_count(content)

    # This fallback is used only to calculate an integer length locally. The
    # resulting string is never attached to telemetry.
    return len(str(value))


def _first_int(*values):
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_llm_usage(response, provider=None):
    """Normalize provider token usage into mutually-exclusive billing buckets.

    DeepSeek exposes prompt cache hit/miss counts through its OpenAI-compatible
    response metadata. Those buckets are kept separate because they have
    different prices. Reasoning tokens are returned as diagnostic metadata only:
    they are already included in output tokens and must not be billed twice.
    """
    provider_name, _ = configured_llm_identity(provider=provider)

    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}

    if not isinstance(usage_metadata, dict):
        usage_metadata = {}
    if not isinstance(response_metadata, dict):
        response_metadata = {}

    token_usage = response_metadata.get("token_usage") or {}
    if not isinstance(token_usage, dict):
        token_usage = {}

    input_details = usage_metadata.get("input_token_details") or {}
    output_details = usage_metadata.get("output_token_details") or {}
    prompt_details = token_usage.get("prompt_tokens_details") or {}
    completion_details = token_usage.get("completion_tokens_details") or {}

    if not isinstance(input_details, dict):
        input_details = {}
    if not isinstance(output_details, dict):
        output_details = {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    if not isinstance(completion_details, dict):
        completion_details = {}

    input_tokens = _first_int(
        usage_metadata.get("input_tokens"),
        token_usage.get("prompt_tokens"),
    )
    output_tokens = _first_int(
        usage_metadata.get("output_tokens"),
        token_usage.get("completion_tokens"),
    )
    total_tokens = _first_int(
        usage_metadata.get("total_tokens"),
        token_usage.get("total_tokens"),
    )

    cache_hit_tokens = _first_int(
        token_usage.get("prompt_cache_hit_tokens"),
        prompt_details.get("cached_tokens"),
        input_details.get("cache_read"),
    )
    cache_miss_tokens = _first_int(token_usage.get("prompt_cache_miss_tokens"))

    if (
        cache_miss_tokens is None
        and input_tokens is not None
        and cache_hit_tokens is not None
    ):
        cache_miss_tokens = max(input_tokens - cache_hit_tokens, 0)

    reasoning_tokens = _first_int(
        completion_details.get("reasoning_tokens"),
        output_details.get("reasoning"),
    )

    usage_details = {}

    if provider_name == "deepseek" and (
        cache_hit_tokens is not None or cache_miss_tokens is not None
    ):
        # DeepSeek bills cached and uncached input at different rates, so do not
        # also send the overlapping aggregate input bucket.
        if cache_hit_tokens is not None:
            usage_details["input_cache_hit"] = cache_hit_tokens
        if cache_miss_tokens is not None:
            usage_details["input_cache_miss"] = cache_miss_tokens
    elif input_tokens is not None:
        usage_details["input"] = input_tokens

    if output_tokens is not None:
        usage_details["output"] = output_tokens
    if total_tokens is not None:
        usage_details["total"] = total_tokens

    reported_model = response_metadata.get("model_name") or response_metadata.get("model")
    if reported_model is not None:
        reported_model = str(reported_model)[:160]

    return {
        "usage_details": usage_details,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "reasoning_tokens": reasoning_tokens,
        "reported_model": reported_model,
    }


def _safe_metadata(metadata):
    """Keep caller-supplied generation metadata to scalar, non-content values."""
    cleaned = {}
    for key, value in (metadata or {}).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            cleaned[str(key)[:120]] = value
    return cleaned


def invoke_llm_observed(
    llm,
    request,
    *,
    purpose,
    question=None,
    provider=None,
    model=None,
    safe_metadata=None,
    observation_name="llm-generation",
):
    """Invoke an LLM exactly once and emit privacy-safe Langfuse telemetry.

    Raw prompts/messages and model output are deliberately never attached to the
    observation. If Langfuse fails before the call starts, the original LLM path
    runs once. If export fails after a successful call, the successful response
    is returned without repeating the paid/provider request. Original LLM
    exceptions are re-raised unchanged so each caller keeps its existing error
    handling semantics.
    """
    provider_name, model_name = configured_llm_identity(provider=provider, model=model)
    langfuse = get_langfuse_client_if_enabled()

    if langfuse is None:
        return llm.invoke(request)

    response = None
    llm_error = None
    started = time.perf_counter()

    try:
        now_utc = datetime.now(timezone.utc)
        question_text = str(question or "").strip()
        input_summary = {
            "purpose": str(purpose),
            "request_length": payload_character_count(request),
            "message_count": len(request) if isinstance(request, (list, tuple)) else 1,
        }
        if question_text:
            input_summary.update(
                {
                    "question_fingerprint": text_fingerprint(question_text),
                    "question_length": len(question_text),
                }
            )

        base_metadata = {
            "purpose": str(purpose),
            "provider": provider_name,
            "request_utc_weekday": now_utc.weekday(),
            "request_utc_hour": now_utc.hour,
        }
        base_metadata.update(_safe_metadata(safe_metadata))

        with langfuse.start_as_current_observation(
            as_type="generation",
            name=observation_name,
            model=model_name,
            input=input_summary,
            metadata=base_metadata,
        ) as observation:
            try:
                response = llm.invoke(request)
            except Exception as exception:
                llm_error = exception
                safe_observation_update(
                    observation,
                    level="ERROR",
                    status_message=type(exception).__name__,
                    output={
                        "status": "error",
                        "has_output": False,
                        "error_type": type(exception).__name__,
                    },
                    metadata={
                        **base_metadata,
                        "latency_seconds": round(time.perf_counter() - started, 4),
                    },
                )
                raise

            content = getattr(response, "content", "")
            output_text = str(content or "")
            telemetry = extract_llm_usage(response, provider=provider_name)
            latency_seconds = round(time.perf_counter() - started, 4)

            update_kwargs = {
                "output": {
                    "status": "ok",
                    "has_output": bool(output_text.strip()),
                    "output_length": len(output_text),
                },
                "metadata": {
                    **base_metadata,
                    "latency_seconds": latency_seconds,
                    "input_tokens": telemetry["input_tokens"],
                    "output_tokens": telemetry["output_tokens"],
                    "total_tokens": telemetry["total_tokens"],
                    "cache_hit_tokens": telemetry["cache_hit_tokens"],
                    "cache_miss_tokens": telemetry["cache_miss_tokens"],
                    "reasoning_tokens": telemetry["reasoning_tokens"],
                    "reported_model": telemetry["reported_model"],
                },
                "level": "DEFAULT" if output_text.strip() else "WARNING",
                "status_message": None if output_text.strip() else "empty_output",
            }

            if telemetry["usage_details"]:
                update_kwargs["usage_details"] = telemetry["usage_details"]

            safe_observation_update(observation, **update_kwargs)

        return response

    except Exception:
        if llm_error is not None:
            # Preserve the provider/application error exactly as if tracing did
            # not exist. Each caller retains its own fallback/error semantics.
            raise llm_error

        if response is not None:
            # The paid/provider call already succeeded. Never issue it again
            # because an observability context/export failed during exit.
            return response

        # Langfuse failed before the provider call began. Run the original path
        # once without making observability a serving dependency.
        return llm.invoke(request)


def _json_safe_scalar(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)
