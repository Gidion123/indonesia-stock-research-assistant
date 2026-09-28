import pytest
from langchain_core.documents import Document

from src import config
from src.rag_chain import (
    DEEPSEEK_MODEL,
    GROQ_MODEL,
    _classify_error,
    ask_question,
    create_rag_chain,
    format_documents,
    get_llm,
    get_prompt,
)


# ============================================================
# PURE FUNCTIONS
# ============================================================

def test_model_names_come_from_config():
    assert GROQ_MODEL == config.GROQ_MODEL == "openai/gpt-oss-20b"
    assert DEEPSEEK_MODEL == config.DEEPSEEK_MODEL == "deepseek-flash"


def test_format_documents_carries_source_and_page():
    documents = [
        Document(
            page_content="  TP 1 BBRI Rp3.400.  ",
            metadata={"source": "ihsg.pdf", "page": 12},
        )
    ]

    context = format_documents(documents)

    assert context.startswith("[ihsg.pdf hal.12]")
    assert "TP 1 BBRI Rp3.400." in context


def test_format_documents_handles_missing_metadata():
    context = format_documents(
        [Document(page_content="x", metadata={})]
    )

    assert "[Unknown hal.Unknown]" in context


def test_prompt_is_cached():
    assert get_prompt() is get_prompt()


def test_prompt_exposes_expected_variables():
    assert set(get_prompt().input_variables) == {"context", "question"}


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        ask_question("   ")


def test_transient_errors_are_classified_separately():
    """
    A rate limit is an infrastructure failure, not a wrong answer.
    """
    assert _classify_error(Exception("Error code: 429")) == "transient_error"
    assert _classify_error(Exception("Read timed out")) == "transient_error"
    assert _classify_error(Exception("invalid schema")) == "error"


# ============================================================
# END TO END
# ============================================================

@pytest.mark.db
@pytest.mark.model
@pytest.mark.llm
def test_llm_and_chain_can_be_created():
    retriever, prompt, llm = create_rag_chain()

    assert retriever is not None
    assert prompt is not None
    assert llm is not None
    assert get_llm() is get_llm()


@pytest.mark.db
@pytest.mark.model
@pytest.mark.llm
def test_ask_question_returns_full_trace():
    result = ask_question("Apa analisis mengenai saham Barito Group?")

    assert result["status"] in {"ok", "empty_retrieval"}
    assert result["answer"].strip() != ""
    assert len(result["documents"]) > 0
    assert result["context"].strip() != ""
    assert len(result["chunk_ids"]) == len(result["documents"])
    assert result["latency_seconds"] >= 0


@pytest.mark.db
@pytest.mark.model
@pytest.mark.llm
def test_in_scope_answer_is_cited_and_citation_is_valid():
    from src.answer_evaluation import citation_accuracy, contains_citation

    result = ask_question(
        "Mengapa harga saham Barito Group relatif resilient?"
    )

    assert result["status"] == "ok"
    assert contains_citation(result["answer"])
    assert citation_accuracy(result["answer"], result["documents"]) == 1.0


@pytest.mark.db
@pytest.mark.model
@pytest.mark.llm
def test_out_of_scope_question_is_refused():
    from src.answer_evaluation import is_refusal

    result = ask_question("Siapa presiden Indonesia tahun 2035?")

    assert result["status"] == "ok"
    assert is_refusal(result["answer"])
