"""
Tests for privacy-first Langfuse observability.

These tests are intentionally network-free:
- no Langfuse Cloud request,
- no PostgreSQL/PgVector request,
- no DeepSeek/Groq request.

They protect the contract that observability must never change application
behaviour or export raw user/document/prompt/answer content from retrieval or
LLM generation paths.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src import observability, rag_chain, retriever as retriever_module


RAW_QUESTION = "Pertanyaan rahasia untuk test observability"
RAW_CONTENT = "CONTENT INI TIDAK BOLEH MASUK KE LANGFUSE"
RAW_ANSWER = "Jawaban observability test."


class CountingRetriever:
    """Small fake retriever that records how many times it is invoked."""

    def __init__(self, documents=None, error=None, k=8):
        self.documents = list(documents or [])
        self.error = error
        self.k = k
        self.calls = 0
        self.queries = []

    def invoke(self, query):
        self.calls += 1
        self.queries.append(query)

        if self.error is not None:
            raise self.error

        return list(self.documents)


class FakeLLM:
    def __init__(
        self,
        answer=RAW_ANSWER,
        error=None,
        usage_metadata=None,
        response_metadata=None,
    ):
        self.answer = answer
        self.error = error
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            content=self.answer,
            usage_metadata=self.usage_metadata,
            response_metadata=self.response_metadata or {},
        )


class FakeObservation:
    """Capture Langfuse observation creation/update without network I/O."""

    def __init__(self, create_kwargs, fail_update=False, fail_exit=False):
        self.create_kwargs = create_kwargs
        self.fail_update = fail_update
        self.fail_exit = fail_exit
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fail_exit and exc_type is None:
            raise RuntimeError("fake tracing export failure")
        return False

    def update(self, **kwargs):
        if self.fail_update:
            raise RuntimeError("fake observation update failure")
        self.updates.append(kwargs)


class FakeLangfuse:
    """Minimal stand-in for the Langfuse client used by our code."""

    def __init__(self, fail_start=False, fail_update=False, fail_exit=False):
        self.fail_start = fail_start
        self.fail_update = fail_update
        self.fail_exit = fail_exit
        self.observations = []

    def start_as_current_observation(self, **kwargs):
        if self.fail_start:
            raise RuntimeError("fake tracing start failure")

        observation = FakeObservation(
            create_kwargs=kwargs,
            fail_update=self.fail_update,
            fail_exit=self.fail_exit,
        )
        self.observations.append(observation)
        return observation


def make_document():
    return Document(
        page_content=RAW_CONTENT,
        metadata={
            "chunk_id": "test-chunk-001",
            "source": "test-document.pdf",
            "page": 7,
            "category": "primary",
            "owner_id": "private-owner-id",
            "expires_at": "private-expiry",
        },
    )


def flatten_text(value):
    """Convert nested captured telemetry to text for leakage assertions."""
    if isinstance(value, dict):
        return " ".join(
            f"{flatten_text(key)} {flatten_text(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def test_text_fingerprint_is_stable_and_does_not_contain_raw_text():
    first = observability.text_fingerprint(RAW_QUESTION)
    second = observability.text_fingerprint(RAW_QUESTION)

    assert first == second
    assert len(first) == 16
    assert RAW_QUESTION not in first


def test_safe_document_reference_contains_only_allowlisted_metadata():
    reference = observability.safe_document_reference(make_document(), rank=1)

    assert reference == {
        "chunk_id": "test-chunk-001",
        "source": "test-document.pdf",
        "page": 7,
        "category": "primary",
        "rank": 1,
    }

    serialized = flatten_text(reference)
    assert RAW_CONTENT not in serialized
    assert "private-owner-id" not in serialized
    assert "private-expiry" not in serialized


def test_tracing_is_disabled_during_pytest_even_when_credentials_exist(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_observability.py::example")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "dummy-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "dummy-secret-key")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://jp.cloud.langfuse.com")

    assert observability.langfuse_tracing_enabled() is False


def test_tracing_requires_explicit_enable_and_credentials(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    for name in (
        "LANGFUSE_TRACING_ENABLED",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    assert observability.langfuse_tracing_enabled() is False

    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    assert observability.langfuse_tracing_enabled() is False

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "dummy-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "dummy-secret-key")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://jp.cloud.langfuse.com")

    assert observability.langfuse_tracing_enabled() is True


def test_retriever_without_tracing_invokes_original_retriever_once(monkeypatch):
    fake_retriever = CountingRetriever([make_document()])

    monkeypatch.setattr(
        retriever_module,
        "get_langfuse_client_if_enabled",
        lambda: None,
    )

    documents = retriever_module.invoke_retriever_observed(
        fake_retriever,
        RAW_QUESTION,
    )

    assert fake_retriever.calls == 1
    assert fake_retriever.queries == [RAW_QUESTION]
    assert documents[0].page_content == RAW_CONTENT


def test_retrieval_trace_exports_fingerprint_and_safe_document_metadata(monkeypatch):
    fake_langfuse = FakeLangfuse()
    fake_retriever = CountingRetriever([make_document()])

    monkeypatch.setattr(
        retriever_module,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    documents = retriever_module.invoke_retriever_observed(
        fake_retriever,
        RAW_QUESTION,
    )

    assert fake_retriever.calls == 1
    assert documents[0].page_content == RAW_CONTENT
    assert len(fake_langfuse.observations) == 1

    observation = fake_langfuse.observations[0]
    created = observation.create_kwargs

    assert created["as_type"] == "retriever"
    assert created["name"] == "rag-retrieval"
    assert created["input"]["query_fingerprint"] == observability.text_fingerprint(
        RAW_QUESTION
    )
    assert created["input"]["query_length"] == len(RAW_QUESTION)

    assert observation.updates
    captured = flatten_text(
        {
            "created": created,
            "updates": observation.updates,
        }
    )

    assert "test-chunk-001" in captured
    assert "test-document.pdf" in captured
    assert RAW_QUESTION not in captured
    assert RAW_CONTENT not in captured
    assert "private-owner-id" not in captured


def test_langfuse_start_failure_falls_back_to_one_retrieval(monkeypatch):
    fake_langfuse = FakeLangfuse(fail_start=True)
    fake_retriever = CountingRetriever([make_document()])

    monkeypatch.setattr(
        retriever_module,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    documents = retriever_module.invoke_retriever_observed(
        fake_retriever,
        RAW_QUESTION,
    )

    assert fake_retriever.calls == 1
    assert len(documents) == 1


def test_langfuse_exit_failure_does_not_repeat_successful_retrieval(monkeypatch):
    fake_langfuse = FakeLangfuse(fail_exit=True)
    fake_retriever = CountingRetriever([make_document()])

    monkeypatch.setattr(
        retriever_module,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    documents = retriever_module.invoke_retriever_observed(
        fake_retriever,
        RAW_QUESTION,
    )

    assert fake_retriever.calls == 1
    assert len(documents) == 1


def test_retrieval_error_is_preserved_and_not_retried(monkeypatch):
    original_error = RuntimeError("database unavailable")
    fake_langfuse = FakeLangfuse()
    fake_retriever = CountingRetriever(error=original_error)

    monkeypatch.setattr(
        retriever_module,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    with pytest.raises(RuntimeError, match="database unavailable") as exc_info:
        retriever_module.invoke_retriever_observed(
            fake_retriever,
            RAW_QUESTION,
        )

    assert exc_info.value is original_error
    assert fake_retriever.calls == 1


def test_rag_chain_trace_does_not_export_raw_question_context_or_answer(monkeypatch):
    document = make_document()
    fake_retriever = CountingRetriever([document])
    fake_llm = FakeLLM()
    fake_langfuse = FakeLangfuse()

    # ask_question() creates the parent observation in rag_chain, while
    # invoke_retriever_observed() creates the child observation in retriever.py.
    monkeypatch.setattr(
        rag_chain,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )
    monkeypatch.setattr(
        retriever_module,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )
    monkeypatch.setattr(
        observability,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    result = rag_chain.ask_question(
        RAW_QUESTION,
        retriever=fake_retriever,
        llm=fake_llm,
    )

    assert result["status"] == "ok"
    assert result["answer"] == RAW_ANSWER
    assert result["context"]
    assert fake_retriever.calls == 1
    assert fake_llm.calls == 1

    assert len(fake_langfuse.observations) == 3
    assert [item.create_kwargs["name"] for item in fake_langfuse.observations] == [
        "rag-answer",
        "rag-retrieval",
        "llm-generation",
    ]

    captured = flatten_text(
        [
            {
                "created": item.create_kwargs,
                "updates": item.updates,
            }
            for item in fake_langfuse.observations
        ]
    )

    # Privacy contract: only fingerprints, lengths, counters, safe references,
    # provider/config metadata, and error class names may leave this layer.
    assert RAW_QUESTION not in captured
    assert RAW_CONTENT not in captured
    assert RAW_ANSWER not in captured
    assert "private-owner-id" not in captured

    assert observability.text_fingerprint(RAW_QUESTION) in captured
    assert "test-chunk-001" in captured
    assert "test-document.pdf" in captured
    assert "retrieved_count" in captured


def test_rag_result_is_unchanged_when_langfuse_parent_fails_before_pipeline(monkeypatch):
    fake_retriever = CountingRetriever([make_document()])
    fake_llm = FakeLLM()
    fake_langfuse = FakeLangfuse(fail_start=True)

    monkeypatch.setattr(
        rag_chain,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )
    # The fallback path can still call the retrieval wrapper; disable its own
    # telemetry here so this test isolates failure of the parent trace.
    monkeypatch.setattr(
        retriever_module,
        "get_langfuse_client_if_enabled",
        lambda: None,
    )

    result = rag_chain.ask_question(
        RAW_QUESTION,
        retriever=fake_retriever,
        llm=fake_llm,
    )

    assert result["status"] == "ok"
    assert result["answer"] == RAW_ANSWER
    assert fake_retriever.calls == 1
    assert fake_llm.calls == 1


def test_rag_result_is_not_recomputed_when_langfuse_parent_exit_fails(monkeypatch):
    fake_retriever = CountingRetriever([make_document()])
    fake_llm = FakeLLM()
    fake_langfuse = FakeLangfuse(fail_exit=True)

    monkeypatch.setattr(
        rag_chain,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )
    monkeypatch.setattr(
        retriever_module,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    result = rag_chain.ask_question(
        RAW_QUESTION,
        retriever=fake_retriever,
        llm=fake_llm,
    )

    assert result["status"] == "ok"
    assert result["answer"] == RAW_ANSWER
    assert fake_retriever.calls == 1
    assert fake_llm.calls == 1


def test_deepseek_usage_is_normalized_without_double_counting_reasoning():
    response = SimpleNamespace(
        content=RAW_ANSWER,
        usage_metadata={
            "input_tokens": 38,
            "output_tokens": 52,
            "total_tokens": 90,
            "input_token_details": {"cache_read": 0},
            "output_token_details": {"reasoning": 50},
        },
        response_metadata={
            "model_name": "deepseek-flash",
            "token_usage": {
                "completion_tokens": 52,
                "prompt_tokens": 38,
                "total_tokens": 90,
                "completion_tokens_details": {"reasoning_tokens": 50},
                "prompt_tokens_details": {"cached_tokens": 0},
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 38,
            },
        },
    )

    telemetry = observability.extract_llm_usage(response, provider="deepseek")

    assert telemetry["usage_details"] == {
        "input_cache_hit": 0,
        "input_cache_miss": 38,
        "output": 52,
        "total": 90,
    }
    assert telemetry["reasoning_tokens"] == 50
    assert "reasoning" not in telemetry["usage_details"]
    assert telemetry["reported_model"] == "deepseek-flash"


def test_llm_generation_trace_exports_usage_but_not_raw_prompt_or_answer(monkeypatch):
    fake_langfuse = FakeLangfuse()
    fake_llm = FakeLLM(
        usage_metadata={
            "input_tokens": 38,
            "output_tokens": 52,
            "total_tokens": 90,
            "input_token_details": {"cache_read": 0},
            "output_token_details": {"reasoning": 50},
        },
        response_metadata={
            "model_name": "deepseek-flash",
            "token_usage": {
                "prompt_tokens": 38,
                "completion_tokens": 52,
                "total_tokens": 90,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 38,
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
        },
    )
    raw_prompt = f"SYSTEM {RAW_CONTENT} USER {RAW_QUESTION}"

    monkeypatch.setattr(
        observability,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    response = observability.invoke_llm_observed(
        fake_llm,
        raw_prompt,
        purpose="rag_answer",
        question=RAW_QUESTION,
        provider="deepseek",
        model="deepseek-flash",
        safe_metadata={"context_length": len(RAW_CONTENT)},
    )

    assert response.content == RAW_ANSWER
    assert fake_llm.calls == 1
    assert len(fake_langfuse.observations) == 1

    observation = fake_langfuse.observations[0]
    assert observation.create_kwargs["as_type"] == "generation"
    assert observation.create_kwargs["name"] == "llm-generation"
    assert observation.create_kwargs["model"] == "deepseek-flash"
    assert observation.create_kwargs["input"]["purpose"] == "rag_answer"
    assert observation.create_kwargs["input"]["request_length"] == len(raw_prompt)

    usage_updates = [
        update for update in observation.updates if "usage_details" in update
    ]
    assert usage_updates
    assert usage_updates[-1]["usage_details"] == {
        "input_cache_hit": 0,
        "input_cache_miss": 38,
        "output": 52,
        "total": 90,
    }

    captured = flatten_text(
        {
            "created": observation.create_kwargs,
            "updates": observation.updates,
        }
    )
    assert RAW_QUESTION not in captured
    assert RAW_CONTENT not in captured
    assert RAW_ANSWER not in captured
    assert observability.text_fingerprint(RAW_QUESTION) in captured
    assert "reasoning_tokens 50" in captured


def test_llm_without_token_metadata_still_returns_response(monkeypatch):
    fake_langfuse = FakeLangfuse()
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        observability,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    response = observability.invoke_llm_observed(
        fake_llm,
        "private prompt",
        purpose="router",
        question=RAW_QUESTION,
        provider="groq",
        model="openai/gpt-oss-20b",
    )

    assert response.content == RAW_ANSWER
    assert fake_llm.calls == 1
    assert fake_langfuse.observations[0].updates
    assert not any(
        "usage_details" in update
        for update in fake_langfuse.observations[0].updates
    )


def test_llm_langfuse_start_failure_calls_provider_once(monkeypatch):
    fake_langfuse = FakeLangfuse(fail_start=True)
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        observability,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    response = observability.invoke_llm_observed(
        fake_llm,
        "private prompt",
        purpose="rag_answer",
        question=RAW_QUESTION,
    )

    assert response.content == RAW_ANSWER
    assert fake_llm.calls == 1


def test_llm_langfuse_exit_failure_does_not_repeat_paid_call(monkeypatch):
    fake_langfuse = FakeLangfuse(fail_exit=True)
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        observability,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    response = observability.invoke_llm_observed(
        fake_llm,
        "private prompt",
        purpose="live_compare",
        question=RAW_QUESTION,
    )

    assert response.content == RAW_ANSWER
    assert fake_llm.calls == 1


def test_llm_original_exception_is_preserved_and_not_retried(monkeypatch):
    original_error = RuntimeError("provider unavailable")
    fake_langfuse = FakeLangfuse()
    fake_llm = FakeLLM(error=original_error)

    monkeypatch.setattr(
        observability,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    with pytest.raises(RuntimeError, match="provider unavailable") as exc_info:
        observability.invoke_llm_observed(
            fake_llm,
            "private prompt",
            purpose="entity_resolution",
            question=RAW_QUESTION,
        )

    assert exc_info.value is original_error
    assert fake_llm.calls == 1


def test_generation_safe_metadata_drops_nested_values(monkeypatch):
    fake_langfuse = FakeLangfuse()
    fake_llm = FakeLLM()

    monkeypatch.setattr(
        observability,
        "get_langfuse_client_if_enabled",
        lambda: fake_langfuse,
    )

    observability.invoke_llm_observed(
        fake_llm,
        "private prompt",
        purpose="live_price",
        question=RAW_QUESTION,
        safe_metadata={
            "safe_counter": 3,
            "unsafe_nested": {"raw": RAW_CONTENT},
        },
    )

    created = fake_langfuse.observations[0].create_kwargs
    assert created["metadata"]["safe_counter"] == 3
    assert "unsafe_nested" not in created["metadata"]
    assert RAW_CONTENT not in flatten_text(created)
