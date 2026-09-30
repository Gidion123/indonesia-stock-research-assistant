"""
Tests for privacy-first Langfuse observability.

These tests are intentionally network-free:
- no Langfuse Cloud request,
- no PostgreSQL/PgVector request,
- no DeepSeek/Groq request.

They protect the contract that observability must never change application
behaviour or export raw user/document content from the RAG retrieval path.
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
    def __init__(self, answer=RAW_ANSWER, error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        if self.error is not None:
            raise self.error

        return SimpleNamespace(content=self.answer)


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

    assert len(fake_langfuse.observations) == 2
    assert [item.create_kwargs["name"] for item in fake_langfuse.observations] == [
        "rag-answer",
        "rag-retrieval",
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
