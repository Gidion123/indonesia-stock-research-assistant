"""Exercise real retrieval and session memory with a fake LLM/market source.

Run against a disposable, populated database named ragtest. This checks
orchestration and retrieval, not the quality of a real model's answers.
"""

import re
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy.engine import make_url

from src import assistant, config, live_compare, live_price, memory
from src.retriever import get_session_retriever
from src.session_documents import new_owner_id


class RecordingRetriever:
    def __init__(self):
        self.inner = get_session_retriever(new_owner_id())
        self.documents = []
        self.query = None

    def invoke(self, query):
        self.query = query
        self.documents = self.inner.invoke(query)
        return self.documents


class FakeLLM:
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, prompt):
        document = self.retriever.documents[0]
        source = document.metadata["source"]
        page = document.metadata["page"]
        return SimpleNamespace(
            content=f"Simulasi konteks riset [{source} hal.{page}]."
        )


def main():
    if make_url(config.DATABASE_URL).database != "ragtest":
        raise SystemExit("Gunakan database simulasi terpisah bernama ragtest.")

    requested_symbols = []

    def fake_price(symbol, **kwargs):
        requested_symbols.append(symbol)
        return {
            "symbol": symbol, "price": 1000.0, "previous_close": 1000.0,
            "change": 0.0, "change_percent": 0.0, "currency": "IDR",
            "exchange": "JKT", "short_name": None,
            "as_of": "2026-09-26T00:00:00+00:00", "status": "ok",
            "error": None, "disclaimer": "Angka tiruan untuk simulasi.",
        }

    states = {name: memory.memori_baru() for name in ("A", "B")}
    retrievers = {name: RecordingRetriever() for name in states}
    scenarios = [
        ("A", "Bagaimana prospek PT Cikarang Listrindo Tbk menurut riset?", "POWR"),
        ("B", "Bagaimana prospek BBRI menurut riset?", "BBRI"),
        ("A", "Apa target harga?", "POWR"),
        ("B", "Bagaimana valuasi?", "BBRI"),
        ("A", "Bagaimana laba?", "POWR"),
        ("A", "Berapa harganya sekarang?", "POWR"),
        ("A", "Apakah sekarang sudah mencapai target?", "POWR"),
        ("A", "Bagaimana prospek TLKM menurut riset?", "TLKM"),
        ("A", "Apa target harga?", "TLKM"),
    ]
    with patch.object(live_price, "ambil_harga", fake_price), patch.object(
        live_compare, "ambil_harga", fake_price
    ):
        for owner, question, expected in scenarios:
            state, retriever = states[owner], retrievers[owner]
            query = memory.kueri_pencarian(question, state)
            result = assistant.jawab(
                question,
                session_context_data=memory.konteks_identitas(state),
                riwayat=memory.riwayat_teks(state),
                kueri_retrieval=query,
                retriever=retriever,
                llm=FakeLLM(retriever),
                izinkan_llm_router=False,
            )
            assert result["status"] == "ok", (question, result["status"])
            assert result["resolution"]["ticker"] == expected, (question, result["resolution"])
            if result["intent"] == "RAG":
                assert re.search(rf"\b{expected}\b", result["documents"][0].page_content)
                if expected == "POWR":
                    assert any(
                        doc.metadata["source"] == "Riset Saham Data Center Indonesia.pdf"
                        for doc in result["documents"]
                    )
            states[owner] = memory.perbarui(state, question, result)
            print(f"PASS sesi {owner}: {question} -> {expected} ({result['intent']})")

    assert requested_symbols == ["POWR.JK", "POWR.JK"], requested_symbols
    assert memory.konteks_identitas(states["B"])["ticker"] == "BBRI"
    assert memory.konteks_identitas(memory.memori_baru()) is None
    print("PASS 9 giliran, 2 memori sesi terpisah; LLM dan harga memakai tiruan.")


if __name__ == "__main__":
    main()
