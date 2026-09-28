"""
Test layanan ingestion.

Yang dijaga: CLI dan UI memakai pipeline yang sama, berkas buruk
ditolak dengan pesan yang bisa dibaca pengguna, dan dokumen gagal tidak
meninggalkan berkas nyasar di knowledge base.
"""

import ast
import inspect
from pathlib import Path

import pytest

from src import ingestion
from src.ingestion import (
    IngestionError,
    UKURAN_MAKS_PDF_MB,
    daftar_dokumen,
    sidik_jari_berkas,
    validasi_pdf,
)


PDF_ASLI = sorted(Path("data/knowledge_base/primary").glob("*.pdf"))


# ============================================================
# SATU PIPELINE, BUKAN DUA
# ============================================================

def test_cli_memanggil_layanan_yang_sama():
    """
    Duplikasi pipeline ingestion adalah cara termudah membuat gold chunk
    tidak lagi sebanding dengan isi database: cukup satu jalur memakai
    ukuran chunk berbeda.
    """
    import scripts.ingest_knowledge_base as cli

    sumber = inspect.getsource(cli)

    assert "bangun_ulang" in sumber
    assert "add_documents" not in sumber
    assert "RecursiveCharacter" not in sumber


def test_ui_tidak_menjalankan_perintah_shell():
    """
    Unggah dari UI harus memanggil fungsi Python, bukan men-spawn
    proses ingestion.
    """
    sumber = Path("app.py").read_text(encoding="utf-8")
    pohon = ast.parse(sumber)

    dilarang = {"subprocess", "os.system", "popen"}

    for node in ast.walk(pohon):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in dilarang, alias.name

        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module not in dilarang, node.module

    assert "add_session_pdf" in sumber


def test_kedua_jalur_memakai_split_documents_yang_sama():
    sumber = inspect.getsource(ingestion)

    assert "split_documents" in sumber
    assert "load_and_split_documents" in sumber


# ============================================================
# VALIDASI
# ============================================================

def test_pdf_asli_lolos_validasi():
    assert PDF_ASLI, "knowledge base kosong"

    info = validasi_pdf(PDF_ASLI[0])

    assert info["pages"] > 0
    assert info["size_mb"] > 0


def test_berkas_tidak_ada_ditolak():
    with pytest.raises(IngestionError) as kendala:
        validasi_pdf("tidak-ada-sama-sekali.pdf")

    assert "tidak ditemukan" in str(kendala.value).lower()


def test_bukan_pdf_ditolak(tmp_path):
    berkas = tmp_path / "catatan.txt"
    berkas.write_text("bukan pdf")

    with pytest.raises(IngestionError) as kendala:
        validasi_pdf(berkas)

    assert "PDF" in str(kendala.value)


def test_berkas_kosong_ditolak(tmp_path):
    berkas = tmp_path / "kosong.pdf"
    berkas.write_bytes(b"")

    with pytest.raises(IngestionError) as kendala:
        validasi_pdf(berkas)

    assert "kosong" in str(kendala.value).lower()


def test_pdf_rusak_ditolak_dengan_pesan_ramah(tmp_path):
    berkas = tmp_path / "rusak.pdf"
    berkas.write_bytes(b"%PDF-1.4 ini bukan struktur pdf yang sah")

    with pytest.raises(IngestionError) as kendala:
        validasi_pdf(berkas)

    pesan = str(kendala.value)

    # Pesan untuk pengguna, bukan traceback.
    assert "rusak" in pesan.lower() or "tidak bisa dibaca" in pesan.lower()
    assert "Traceback" not in pesan


def test_batas_ukuran_masuk_akal():
    assert 1 <= UKURAN_MAKS_PDF_MB <= 100


# ============================================================
# DUPLIKAT
# ============================================================

def test_sidik_jari_sama_untuk_isi_sama(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"

    a.write_bytes(b"isi yang sama persis")
    b.write_bytes(b"isi yang sama persis")

    assert sidik_jari_berkas(a) == sidik_jari_berkas(b)


def test_sidik_jari_beda_untuk_isi_beda(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"

    a.write_bytes(b"isi pertama")
    b.write_bytes(b"isi kedua")

    assert sidik_jari_berkas(a) != sidik_jari_berkas(b)


def test_duplikat_nama_terdeteksi(tmp_path):
    (tmp_path / "laporan.pdf").write_bytes(b"lama")

    sumber = tmp_path / "sub"
    sumber.mkdir()
    baru = sumber / "laporan.pdf"
    baru.write_bytes(b"baru")

    assert "sudah ada" in ingestion._sudah_ada(baru, tmp_path)


def test_duplikat_isi_terdeteksi_walau_nama_berbeda(tmp_path):
    (tmp_path / "riset-q1.pdf").write_bytes(b"isi identik")

    sumber = tmp_path / "sub"
    sumber.mkdir()
    baru = sumber / "salinan-riset.pdf"
    baru.write_bytes(b"isi identik")

    pesan = ingestion._sudah_ada(baru, tmp_path)

    assert pesan is not None
    assert "riset-q1.pdf" in pesan


# ============================================================
# DAFTAR DOKUMEN
# ============================================================

def test_daftar_dokumen_menyebut_pdf_aktif():
    dokumen = daftar_dokumen(include_additional=False)

    assert len(dokumen) == 5
    assert any(d["name"] == "Riset Saham Data Center Indonesia.pdf" for d in dokumen)

    for d in dokumen:
        assert d["name"].lower().endswith(".pdf")
        assert d["size_mb"] > 0


def test_dokumen_tambahan_tetap_opt_in(monkeypatch, tmp_path):
    """
    `daftar_dokumen` honours the opt-in, whatever is on disk.

    The earlier version asserted `primary < extended`, a strict subset,
    which silently required additional/ to hold at least one file. That
    made an empty additional/ - a perfectly normal state - look like a
    bug, and it broke the moment a test document was moved out.
    """
    from src import config

    primary_dir = tmp_path / "primary"
    additional_dir = tmp_path / "additional"
    primary_dir.mkdir()
    additional_dir.mkdir()

    (primary_dir / "baseline.pdf").write_bytes(b"%PDF")
    (additional_dir / "tambahan.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(config, "PRIMARY_DIR", primary_dir)
    monkeypatch.setattr(config, "ADDITIONAL_DIR", additional_dir)

    primary = {d["name"] for d in daftar_dokumen(include_additional=False)}
    extended = {d["name"] for d in daftar_dokumen(include_additional=True)}

    assert primary == {"baseline.pdf"}
    assert extended == {"baseline.pdf", "tambahan.pdf"}
    assert primary < extended


def test_daftar_dokumen_dengan_folder_tambahan_kosong(monkeypatch, tmp_path):
    """
    An empty additional/ must give the same list as primary alone.

    Not an error, and not an empty list - this is the state the project
    sits in after a rebuild, and the state a reviewer clones into.
    """
    from src import config

    primary_dir = tmp_path / "primary"
    additional_dir = tmp_path / "additional"
    primary_dir.mkdir()
    additional_dir.mkdir()

    (primary_dir / "baseline.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(config, "PRIMARY_DIR", primary_dir)
    monkeypatch.setattr(config, "ADDITIONAL_DIR", additional_dir)

    primary = {d["name"] for d in daftar_dokumen(include_additional=False)}
    extended = {d["name"] for d in daftar_dokumen(include_additional=True)}

    assert primary == extended == {"baseline.pdf"}
