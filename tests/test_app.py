"""
Test antarmuka Streamlit sebagai KODE, tanpa menjalankan server.

Dua hal yang dijaga:

1.  Halaman ini milik pelanggan. Konfigurasi teknis — nama model,
    embedding, nilai k, URL database — tidak boleh muncul di sana.
2.  Mengimpor app.py tidak boleh memicu permintaan jaringan. Sebuah
    modul yang menembak Yahoo Finance saat di-import akan melakukannya
    pada setiap rerun Streamlit, yaitu setiap kali pengguna mengetik.
"""

import ast
from pathlib import Path

import pytest


SUMBER_APP = Path("app.py").read_text(encoding="utf-8")
POHON_APP = ast.parse(SUMBER_APP)


# ============================================================
# TIDAK ADA KONFIGURASI TEKNIS DI LAYAR PELANGGAN
# ============================================================

def _teks_yang_ditampilkan():
    """
    Kumpulkan string literal yang menjadi argumen pemanggilan st.*.
    Komentar dan docstring tidak ikut — penjelasan di dalam kode bukan
    sesuatu yang dilihat pengguna.
    """
    teks = []

    for node in ast.walk(POHON_APP):
        if not isinstance(node, ast.Call):
            continue

        fungsi = node.func

        if not (
            isinstance(fungsi, ast.Attribute)
            and isinstance(fungsi.value, ast.Name)
            and fungsi.value.id == "st"
        ):
            continue

        for argumen in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(argumen, ast.Constant) and isinstance(
                argumen.value, str
            ):
                teks.append(argumen.value)

    return " ".join(teks).lower()


@pytest.mark.parametrize(
    "istilah",
    [
        "deepseek",
        "groq",
        "minilm",
        "sentence-transformers",
        "embedding",
        "temperature",
        "database_url",
        "postgres",
        "pgvector",
        "api_key",
        "api key",
        "base_url",
        "retriever",
        "provider",
    ],
)
def test_konfigurasi_teknis_tidak_ditampilkan(istilah):
    assert istilah not in _teks_yang_ditampilkan(), istilah


def test_nilai_k_tidak_ditampilkan():
    ditampilkan = _teks_yang_ditampilkan()

    assert "k=" not in ditampilkan
    assert "retrieval k" not in ditampilkan


def test_jejak_keputusan_tidak_dibuka_ke_pelanggan():
    """
    Router method, jumlah panggilan LLM, dan level resolver berguna saat
    evaluasi — bukan di layar orang yang bertanya soal saham.
    """
    ditampilkan = _teks_yang_ditampilkan()

    for istilah in ("jejak keputusan", "llm calls", "intent", "router"):
        assert istilah not in ditampilkan, istilah


def test_config_describe_tidak_dipanggil_di_ui():
    """
    `config.describe()` memuat nama model dan nama tabel. Berguna untuk
    laporan evaluasi, tidak untuk halaman pelanggan.
    """
    assert "describe()" not in SUMBER_APP


def test_tidak_ada_akses_environment_langsung():
    assert "os.environ" not in SUMBER_APP
    assert "getenv" not in SUMBER_APP


# ============================================================
# IMPORT TIDAK MEMICU JARINGAN
# ============================================================

def test_import_app_tidak_memanggil_yahoo():
    """
    Rerun Streamlit terjadi pada setiap interaksi. Modul yang menembak
    Yahoo saat di-import akan melakukannya berkali-kali per percakapan.
    """
    modul_diimpor = set()

    for node in ast.walk(POHON_APP):
        if isinstance(node, ast.Import):
            modul_diimpor.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modul_diimpor.add(node.module)

    assert "yfinance" not in modul_diimpor

    # Pemanggilan di tingkat modul (bukan di dalam fungsi) tidak boleh
    # berisi pengambilan harga.
    assert "ambil_harga" not in SUMBER_APP
    assert "jaringan_sehat" not in SUMBER_APP


def test_app_bisa_diparse_dan_dikompilasi():
    compile(SUMBER_APP, "app.py", "exec")


# ============================================================
# BENTUK ANTARMUKA PERCAKAPAN
# ============================================================

def test_memakai_komponen_chat_streamlit():
    assert "st.chat_message" in SUMBER_APP
    assert "st.chat_input" in SUMBER_APP


def test_memakai_session_state_bukan_state_global():
    """
    Memori percakapan harus terisolasi antar pengguna. State global di
    tingkat modul akan dibagi oleh semua sesi.
    """
    assert "st.session_state" in SUMBER_APP

    # Tidak ada dict/list mutable tingkat modul yang dipakai sebagai
    # penyimpan percakapan.
    for node in POHON_APP.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "RIWAYAT",
                    "KONTEKS",
                    "MEMORY",
                    "SESSION",
                }:
                    raise AssertionError(
                        f"state percakapan global: {target.id}"
                    )


def test_ada_area_unggah_knowledge():
    assert "file_uploader" in SUMBER_APP
    assert "add_session_pdf" in SUMBER_APP


def test_pesan_sukses_dan_gagal_ramah_pengguna():
    ditampilkan = _teks_yang_ditampilkan()

    assert "knowledge berhasil ditambahkan" in ditampilkan
    assert "knowledge gagal ditambahkan" in ditampilkan


def test_disclaimer_pasar_ditampilkan():
    ditampilkan = _teks_yang_ditampilkan()

    assert "tertunda" in ditampilkan
    assert "bukan nasihat investasi" in ditampilkan


def test_exception_ditangkap_agar_ui_tidak_crash():
    """
    PgVector mati, API key habis, PDF rusak — tidak satu pun boleh
    menjatuhkan halaman.
    """
    jumlah_try = sum(
        1 for node in ast.walk(POHON_APP) if isinstance(node, ast.Try)
    )

    assert jumlah_try >= 3
