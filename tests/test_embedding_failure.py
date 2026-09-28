"""
What happens when the embedding model cannot be loaded at all.

Two different failures used to look identical to the user: a cold cache
(fixable by downloading) and a broken install (not fixable by anything
the app can do). The second one was reported as the first, so the
terminal said "downloading..." while the real cause - a package that
would not import - stayed buried under a second identical traceback.

These tests pin the distinction down.
"""

import pytest

from src import embeddings


@pytest.fixture(autouse=True)
def bersihkan_cache():
    embeddings.reset_cache_embeddings()
    yield
    embeddings.reset_cache_embeddings()


def test_import_error_tidak_memicu_unduhan(monkeypatch):
    """
    A broken import must fail immediately, not trigger a download.

    This is the exact shape of the real failure: sentence-transformers is
    installed, but importing it blows up somewhere deep in its own
    dependencies. Retrying with a download runs the same import again.
    """
    percobaan = []

    def bangun_gagal(local_files_only):
        percobaan.append(local_files_only)
        raise ImportError("Could not import sentence_transformers")

    monkeypatch.setattr(embeddings, "_bangun", bangun_gagal)

    with pytest.raises(embeddings.EmbeddingTidakTersedia):
        embeddings.get_embeddings()

    assert percobaan == [True], (
        "setelah ImportError, unduhan tidak boleh dicoba - "
        f"percobaan yang terjadi: {percobaan}"
    )


def test_penyebab_akar_ikut_dilaporkan(monkeypatch):
    """
    The message must name the deepest cause, not just the top wrapper.

    langchain wraps the real error in a generic "please install
    sentence-transformers", which sends the reader off installing a
    package that is already there. The root cause is what identifies the
    actual broken dependency.
    """
    akar = ImportError("dlopen(_spropack...) gagal: scipy rusak")

    def bangun_gagal(local_files_only):
        try:
            raise akar
        except ImportError as sebab:
            raise ImportError(
                "Could not import sentence_transformers python package."
            ) from sebab

    monkeypatch.setattr(embeddings, "_bangun", bangun_gagal)

    with pytest.raises(embeddings.EmbeddingTidakTersedia) as kegagalan:
        embeddings.get_embeddings()

    pesan = str(kegagalan.value)

    assert "scipy rusak" in pesan, (
        "penyebab akar harus ikut disebut, bukan hanya pembungkusnya"
    )
    assert "diagnose_embeddings" in pesan, "pesan harus bisa ditindaklanjuti"


def test_cache_dingin_tetap_mengunduh(monkeypatch):
    """
    A cold cache is a different failure and must still download.

    This is the behaviour the fix above must not break: nothing is
    installed wrong, the model file simply is not on disk yet.
    """
    percobaan = []

    def bangun(local_files_only):
        percobaan.append(local_files_only)

        if local_files_only:
            raise OSError("model tidak ditemukan di cache lokal")

        return "model"

    monkeypatch.setattr(embeddings, "_bangun", bangun)

    assert embeddings.get_embeddings() == "model"
    assert percobaan == [True, False], (
        "cache dingin harus mencoba lokal dulu, lalu mengunduh"
    )
    assert embeddings.sumber_model() == embeddings.SUMBER_UNDUH


def test_rantai_kesalahan_tidak_berputar():
    """
    A self-referencing exception chain must not loop forever.

    `raise X from X` and cycles built by exception handlers do occur in
    the wild, and walking the chain is done on a failure path where a
    hang is the worst possible outcome.
    """
    kesalahan = ValueError("berputar")
    kesalahan.__cause__ = kesalahan

    rantai = embeddings._rantai_kesalahan(kesalahan)

    assert rantai == [kesalahan]


def test_kegagalan_biasa_bukan_masalah_environment():
    """
    Only import failures count as environment failures.

    A timeout or a missing file must keep the download path, otherwise
    the fix above turns a recoverable situation into a hard stop.
    """
    assert not embeddings._masalah_environment(OSError("tidak ada berkas"))
    assert not embeddings._masalah_environment(TimeoutError("lambat"))
    assert embeddings._masalah_environment(ImportError("tidak bisa import"))
    assert embeddings._masalah_environment(
        ModuleNotFoundError("tidak ada modul")
    )
