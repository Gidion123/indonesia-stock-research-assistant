"""
Test sederhana untuk memastikan provider LLM yang digunakan aplikasi.

Prioritas provider:
1. DeepSeek jika tersedia
2. Groq sebagai fallback
"""

from src import config
from src.rag_chain import get_llm


def test_llm_provider_selected():
    """
    Memastikan aplikasi memilih provider LLM yang valid
    dan get_llm() menggunakan provider tersebut.
    """

    provider = config.LLM_PROVIDER.lower()

    assert provider in {"deepseek", "groq"}

    llm = get_llm()

    if provider == "deepseek":
        assert llm.__class__.__name__ == "ChatOpenAI"
        model = config.DEEPSEEK_MODEL

    else:
        assert llm.__class__.__name__ == "ChatGroq"
        model = config.GROQ_MODEL

    assert model

    print(f"\nLLM yang digunakan : {provider}")
    print(f"Model              : {model}")