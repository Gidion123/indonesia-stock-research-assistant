"""
Curated folders and the public upload path stay separate.

`additional/` remains an opt-in offline experiment folder. Public uploads
use session_documents and a different vector table; app.py must never
list the global additional folder or call the legacy add function.
"""

import ast
from pathlib import Path

import pytest


AKAR = Path(__file__).resolve().parent.parent


def _panggilan(nama_fungsi, berkas="app.py"):
    """Every call to `nama_fungsi` in `berkas`, as AST nodes."""
    pohon = ast.parse((AKAR / berkas).read_text())

    return [
        simpul
        for simpul in ast.walk(pohon)
        if isinstance(simpul, ast.Call)
        and isinstance(simpul.func, ast.Name)
        and simpul.func.id == nama_fungsi
    ]


def test_ui_memakai_unggahan_per_sesi():
    assert _panggilan("add_session_pdf")
    assert not _panggilan("tambah_pdf")


def test_daftar_dokumen_tidak_menampilkan_unggahan_global():
    """
    The sidebar must combine curated PDFs with only this session's uploads.
    """
    panggilan = _panggilan("daftar_dokumen")

    assert panggilan, "app.py harus memanggil daftar_dokumen"

    for satu in panggilan:
        argumen = {kw.arg: kw.value for kw in satu.keywords}

        assert "include_additional" in argumen, (
            "daftar_dokumen dipanggil tanpa include_additional, jadi dokumen "
            "yang baru diunggah tidak akan muncul di panel samping"
        )

        nilai = argumen["include_additional"]

        assert isinstance(nilai, ast.Constant) and nilai.value is False

    assert _panggilan("list_session_pdfs")


def test_folder_tambahan_tidak_ikut_baseline():
    """
    The additional folder stays out of a rebuild by default.

    This is the other half of the guarantee: putting uploads in
    additional/ only protects the baseline as long as additional/ is
    genuinely excluded unless asked for.
    """
    from src import config
    from src.preprocessing import get_pdf_files

    assert config.INCLUDE_ADDITIONAL_DOCUMENTS is False, (
        "baseline harus tertutup secara bawaan"
    )

    berkas = get_pdf_files(include_additional=False)
    folder = {p.parent.name for p in berkas}

    assert "additional" not in folder, (
        f"additional/ tidak boleh ikut secara bawaan; folder terbaca: {folder}"
    )


def test_tambah_pdf_menghormati_tujuannya(monkeypatch, tmp_path):
    """
    tambah_pdf must write where it was told, not where it prefers.
    """
    from src import config, ingestion

    primary = tmp_path / "primary"
    tambahan = tmp_path / "additional"
    primary.mkdir()
    tambahan.mkdir()

    monkeypatch.setattr(config, "PRIMARY_DIR", primary)
    monkeypatch.setattr(config, "ADDITIONAL_DIR", tambahan)

    dicatat = {}

    def palsu_validasi(path):
        return {"pages": 1, "size_mb": 0.1}

    def palsu_load(path, collect_report=None):
        dicatat["tujuan"] = Path(path).parent
        return ["halaman"]

    def palsu_split(halaman):
        return ["chunk"]

    def palsu_simpan(chunks):
        return len(chunks)

    monkeypatch.setattr(ingestion, "validasi_pdf", palsu_validasi)
    monkeypatch.setattr(ingestion, "split_documents", palsu_split)
    monkeypatch.setattr(ingestion, "_simpan_ke_vector_store", palsu_simpan)
    monkeypatch.setattr(ingestion, "_sudah_ada", lambda *a, **k: None)

    import src.preprocessing as preprocessing

    monkeypatch.setattr(preprocessing, "load_pdf", palsu_load)

    sumber = tmp_path / "uji.pdf"
    sumber.write_bytes(b"%PDF-1.4 uji")

    ringkas = ingestion.tambah_pdf(sumber, ke_folder_tambahan=True)

    assert ringkas["folder"] == "additional"
    assert dicatat["tujuan"] == tambahan
    assert (tambahan / "uji.pdf").exists()
    assert not (primary / "uji.pdf").exists(), (
        "korpus beku tidak boleh ikut berubah"
    )


# ============================================================
# The sidebar must list what is searchable, not what is on disk
# ============================================================

def test_daftar_dokumen_menyaring_lewat_vector_store(monkeypatch, tmp_path):
    """
    A PDF on disk that was never embedded must not be listed.

    This is the exact situation that caused confusion: a file copied
    into additional/ by hand showed up in the sidebar, but asking about
    it produced "the information is not in my documents".
    """
    from src import config, ingestion

    primary = tmp_path / "primary"
    tambahan = tmp_path / "additional"
    primary.mkdir()
    tambahan.mkdir()

    (primary / "sudah-masuk.pdf").write_bytes(b"%PDF")
    (tambahan / "belum-masuk.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(config, "PRIMARY_DIR", primary)
    monkeypatch.setattr(config, "ADDITIONAL_DIR", tambahan)
    monkeypatch.setattr(
        ingestion, "sumber_di_vector_store", lambda: {"sudah-masuk.pdf"}
    )

    nama = [
        d["name"]
        for d in ingestion.daftar_dokumen(
            include_additional=True, verifikasi=True
        )
    ]

    assert nama == ["sudah-masuk.pdf"], (
        "dokumen yang belum ter-embed tidak boleh muncul di daftar"
    )


def test_database_mati_tidak_mengosongkan_daftar(monkeypatch, tmp_path):
    """
    An unreachable database must not blank the sidebar.

    `sumber_di_vector_store` returns None for "cannot tell", which is a
    different answer from an empty set. Treating None as "nothing is
    embedded" would make the app look broken whenever Postgres is down.
    """
    from src import config, ingestion

    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "satu.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(config, "PRIMARY_DIR", primary)
    monkeypatch.setattr(config, "ADDITIONAL_DIR", tmp_path / "additional")
    monkeypatch.setattr(ingestion, "sumber_di_vector_store", lambda: None)

    nama = [
        d["name"]
        for d in ingestion.daftar_dokumen(
            include_additional=True, verifikasi=True
        )
    ]

    assert nama == ["satu.pdf"]


def test_kegagalan_database_tidak_diulang_terus(monkeypatch):
    """
    One failed connection must not cost every page render a timeout.

    Streamlit re-runs the whole script on each interaction, so an
    unreachable database would otherwise stall the sidebar for the
    connect timeout every single time.
    """
    from src import ingestion

    percobaan = []

    def gagal(*args, **kwargs):
        percobaan.append(1)
        raise OSError("database tidak bisa dihubungi")

    monkeypatch.setattr(ingestion, "_gagal_terakhir", None)
    monkeypatch.setitem(__import__("sys").modules, "psycopg", None)

    import src.vector_store as vs

    monkeypatch.setattr(vs, "get_psycopg_connection_url", gagal)

    assert ingestion.sumber_di_vector_store() is None
    assert ingestion.sumber_di_vector_store() is None

    assert len(percobaan) <= 1, (
        "percobaan koneksi harus dijeda setelah gagal, "
        f"tapi terjadi {len(percobaan)} kali"
    )


def test_app_meminta_verifikasi():
    """app.py must ask for the verified list, not the raw file listing."""
    panggilan = _panggilan("daftar_dokumen")

    for satu in panggilan:
        argumen = {kw.arg: kw.value for kw in satu.keywords}

        assert "verifikasi" in argumen, (
            "daftar_dokumen dipanggil tanpa verifikasi, jadi panel samping "
            "bisa menampilkan dokumen yang belum ter-embed"
        )

        nilai = argumen["verifikasi"]

        assert isinstance(nilai, ast.Constant) and nilai.value is True
