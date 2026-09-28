"""
Bagaimana model embedding dimuat — bukan seberapa bagus vektornya.

Test di `test_embeddings.py` butuh model sungguhan dan karena itu diberi
marker `model`. Test di sini sengaja tidak: konstruktornya diganti stub,
jadi ia berjalan di mesin mana pun, termasuk mesin tanpa internet. Itu
penting, karena justru perilaku "apa yang terjadi saat jaringan lambat"
yang sedang dijaga.

Latar: satu run end-to-end menghabiskan ~20 detik di pertanyaan pertama
karena huggingface_hub menghubungi huggingface.co untuk memeriksa revisi
model yang sudah ada di cache, lalu ReadTimeout.
"""

import pytest

import src.embeddings as emb


class EmbeddingStub:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_query(self, teks):
        return [0.0] * 4

    def embed_documents(self, daftar):
        return [[0.0] * 4 for _ in daftar]


@pytest.fixture
def pabrik(monkeypatch):
    """
    Catat setiap percobaan konstruksi model, dan biarkan test menentukan
    percobaan mana yang gagal.
    """
    percobaan = []
    gagal_saat = {"local_files_only_true": False, "semua": False}

    def buat(**kwargs):
        percobaan.append(kwargs)

        lokal = kwargs["model_kwargs"].get("local_files_only")

        if gagal_saat["semua"]:
            raise OSError("model tidak bisa dimuat sama sekali")

        if lokal and gagal_saat["local_files_only_true"]:
            raise OSError(
                "We couldn't connect to huggingface.co and the model is "
                "not cached locally"
            )

        return EmbeddingStub(**kwargs)

    monkeypatch.setattr(emb, "HuggingFaceEmbeddings", buat)
    emb.reset_cache_embeddings()

    yield percobaan, gagal_saat

    emb.reset_cache_embeddings()


# ============================================================
# CACHE HANGAT: JALUR NORMAL
# ============================================================

def test_percobaan_pertama_meminta_local_files_only(pabrik):
    percobaan, _ = pabrik

    emb.get_embeddings()

    assert percobaan[0]["model_kwargs"]["local_files_only"] is True


def test_cache_hangat_hanya_satu_percobaan(pabrik):
    percobaan, _ = pabrik

    emb.get_embeddings()

    assert len(percobaan) == 1, (
        "Model dibangun lebih dari sekali padahal percobaan pertama "
        "berhasil."
    )
    assert emb.sumber_model() == emb.SUMBER_CACHE_LOKAL
    assert emb.alasan_unduh() is None


def test_model_dibangun_sekali_meski_dipanggil_berkali_kali(pabrik):
    percobaan, _ = pabrik

    pertama = emb.get_embeddings()

    for _ in range(10):
        assert emb.get_embeddings() is pertama

    assert len(percobaan) == 1


def test_device_dan_normalisasi_tetap_diteruskan(pabrik):
    percobaan, _ = pabrik

    from src import config

    emb.get_embeddings()

    kwargs = percobaan[0]

    assert kwargs["model_name"] == config.EMBEDDING_MODEL_NAME
    assert kwargs["model_kwargs"]["device"] == config.EMBEDDING_DEVICE
    assert (
        kwargs["encode_kwargs"]["normalize_embeddings"]
        == config.EMBEDDING_NORMALIZE
    )


# ============================================================
# CACHE DINGIN: MASIH BOLEH MENGUNDUH
# ============================================================

def test_cache_dingin_jatuh_ke_unduh(pabrik):
    percobaan, gagal_saat = pabrik
    gagal_saat["local_files_only_true"] = True

    emb.get_embeddings()

    assert len(percobaan) == 2
    assert percobaan[0]["model_kwargs"]["local_files_only"] is True
    assert percobaan[1]["model_kwargs"]["local_files_only"] is False
    assert emb.sumber_model() == emb.SUMBER_UNDUH


def test_alasan_jatuh_ke_unduh_tidak_disembunyikan(pabrik):
    _, gagal_saat = pabrik
    gagal_saat["local_files_only_true"] = True

    emb.get_embeddings()

    alasan = emb.alasan_unduh()

    assert alasan
    assert "huggingface.co" in alasan


def test_kegagalan_sebenarnya_tetap_naik(pabrik):
    """
    Fallback bukan penelan error. Kalau model memang tidak bisa dimuat,
    pemanggil harus melihat kegagalannya, bukan mendapat objek setengah
    jadi yang gagal jauh di kemudian hari.
    """
    percobaan, gagal_saat = pabrik
    gagal_saat["semua"] = True

    with pytest.raises(OSError):
        emb.get_embeddings()

    assert len(percobaan) == 2


# ============================================================
# SATU MODEL UNTUK SELURUH APLIKASI
# ============================================================

def test_vector_store_dan_retriever_memakai_model_yang_sama(
    pabrik, monkeypatch
):
    """
    Inti dari audit ini: jalur RAG dan jalur entity resolver tidak boleh
    masing-masing membangun model sendiri. Kalau iya, setiap jalur
    membayar satu pemeriksaan jaringan.
    """
    percobaan, _ = pabrik

    import src.vector_store as vs
    from src import retriever as rt

    monkeypatch.setattr(vs, "ensure_primary_category_column", lambda: None)

    class StoreStub:
        def __init__(self, embedding_service):
            self.embedding_service = embedding_service

        def as_retriever(self, **kwargs):
            return ("retriever", kwargs)

    monkeypatch.setattr(
        vs,
        "PGVectorStore",
        type(
            "PGVectorStoreStub",
            (),
            {
                "create_sync": staticmethod(
                    lambda engine=None, table_name=None,
                    embedding_service=None, **kwargs: StoreStub(embedding_service)
                )
            },
        ),
    )

    vs.get_vector_store.cache_clear()
    rt.get_retriever.cache_clear()

    try:
        store = vs.get_vector_store()

        rt.get_retriever(8)
        rt.get_retriever(None)
        vs.get_vector_store()

        assert len(percobaan) == 1, (
            f"Model dibangun {len(percobaan)} kali; seharusnya satu."
        )
        assert store.embedding_service is emb.get_embeddings()

    finally:
        vs.get_vector_store.cache_clear()
        rt.get_retriever.cache_clear()
