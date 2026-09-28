"""
Asisten Research Bursa Saham - the conversational interface.

    streamlit run app.py

This page belongs to the USER, not to the developer. No model names, no
technical values, no database addresses, no raw decision traces on
screen. All of that still exists - in the result of `jawab()` and in the
evaluation scripts - just not in front of someone who is asking about a
stock.

The conversation layout follows the st-chat pattern (AI-Yash/st-chat):
user bubbles on the right, assistant bubbles on the left, avatars on the
outside. Only the pattern is borrowed. The components are still
Streamlit's own `st.chat_message`, so markdown, spinners and history
keep working without an extra dependency.
"""

import logging
from html import escape

import streamlit as st

from src import memory
from src.assistant import jawab
from src.embeddings import EmbeddingTidakTersedia
from src.ingestion import IngestionError, daftar_dokumen
from src.retriever import get_session_retriever
from src.session_documents import (
    QuotaExceeded,
    add_session_pdf,
    consume_question,
    list_session_pdfs,
    maybe_cleanup,
    new_owner_id,
)


logger = logging.getLogger("asisten_bursa")


# Product name lives in one place so it is easy to change.
NAMA = "Asisten Research Bursa Saham"
TAGLINE = "Riset saham Indonesia, dijawab dari dokumen riset milik Gidion Depari"

LAMBANG_ASISTEN = "📈"
LAMBANG_PENGGUNA = "🧑"

st.set_page_config(
    page_title=NAMA,
    page_icon=LAMBANG_ASISTEN,
    layout="centered",
    initial_sidebar_state="expanded",
)


SAPAAN = (
    "Halo! Saya asisten riset bursa saham Anda.\n\n"
    "Saya bisa membantu memahami isi laporan riset, mengecek harga "
    "pasar terkini, atau membandingkan keduanya. Mulai dari salah satu "
    "contoh di bawah, atau tanya apa saja."
)

CONTOH = [
    ("Prospek saham", "Bagaimana prospek BBRI menurut riset?"),
    ("Harga terkini", "Harga BBRI sekarang berapa?"),
    ("Bandingkan target", "BBRI sudah mencapai target belum?"),
    ("Hitung posisi", "Saya beli BBRI di 4.000, sekarang untung berapa?"),
]

PESAN_GAGAL_UMUM = (
    "Maaf, ada kendala saat memproses pertanyaan Anda. "
    "Silakan coba lagi sebentar lagi."
)

# A broken embedding install is not a temporary glitch, so it must not be
# answered with "try again later" - that sends the reader off looking for
# a network problem that is not there. The terminal already carries the
# root cause; this message only has to say that waiting will not help.
PESAN_MODEL_TIDAK_SIAP = (
    "Pencarian dokumen riset sedang tidak bisa dijalankan karena model "
    "pencarinya gagal dimuat di environment ini. Ini bukan gangguan "
    "sementara — mencoba lagi tidak akan menolong. Penyebabnya tercetak "
    "di terminal, dan `python -m scripts.diagnose_embeddings` "
    "memeriksanya lebih rinci."
)


# ============================================================
# PRESENTATION
# ============================================================
#
# Which side a bubble sits on is decided by `st.container(key=...)`.
# Streamlit puts an `st-key-<key>` class on the container, so the CSS can
# target user or assistant without depending on Streamlit's internal
# class names, which change from release to release.
#
# The first attempt used a hidden <span> marker INSIDE the bubble. That
# was wrong, and it failed in a subtle way: Streamlit pins an explicit
# pixel height on the element container based on its own measurement.
# Once the marker was hidden with CSS, the vertical block's `gap: 16px`
# disappeared from view but the height had already been fixed - so the
# last line was clipped 16px below the bubble. A keyed container adds
# nothing inside the bubble, so that problem does not exist.

def kunci_gelembung(peran, nomor):
    return f"bi-{peran}-{nomor}"


GAYA = """
<style>
  /* Hide Streamlit's built-in chrome. This is a product, not a notebook. */
  #MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
  .stDeployButton {display: none;}
  header {height: 0;}

  :root {
    --bi-ink:       #101f2d;
    --bi-ink-soft:  #5d6d7c;
    --bi-line:      #e2e8ee;
    --bi-surface:   #ffffff;
    --bi-raised:    #f2f6f9;
    --bi-canvas:    #f6f9fb;
    --bi-brand:     #0b6b55;
    --bi-brand-ink: #ffffff;
    --bi-brand-dim: #e6f3ef;
    --bi-shadow:    0 1px 2px rgba(16, 31, 45, .06);
    --bi-warn:      #9a6200;
    --bi-warn-on:   #ffffff;
    --bi-warn-ink:  #6b4400;
    --bi-warn-dim:  #fff7e8;
    --bi-warn-line: #f0dfbc;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --bi-ink:       #e9eff5;
      --bi-ink-soft:  #93a5b6;
      --bi-line:      #1f2d3a;
      --bi-surface:   #131f2a;
      --bi-raised:    #16232f;
      --bi-canvas:    #0b131b;
      --bi-brand:     #2f9e82;
      --bi-brand-ink: #f2fffb;
      --bi-brand-dim: #12302a;
      --bi-shadow:    none;
      --bi-warn:      #e0a94f;
      --bi-warn-on:   #1a1205;
      --bi-warn-ink:  #f0cea0;
      --bi-warn-dim:  #241c0f;
      --bi-warn-line: #3d3018;
    }
  }

  /* The rule above forces Streamlit's default header to zero height.
     Without this padding the title row slides under the window edge. */
  .block-container {
    padding-top: 3.2rem;
    padding-bottom: 8rem;
    max-width: 48rem;
  }

  /* ---------- Page header ---------- */

  .bi-head {
    display: flex; align-items: center; gap: .85rem;
    padding: .2rem 0 1rem;
    margin-bottom: .6rem;
    border-bottom: 1px solid var(--bi-line);
  }
  .bi-mark {
    width: 44px; height: 44px; flex: 0 0 44px;
    border-radius: 14px;
    background: linear-gradient(145deg, var(--bi-brand), var(--bi-brand-dim));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: var(--bi-shadow);
  }
  .bi-name {
    font-size: 1.12rem; font-weight: 650; line-height: 1.25;
    color: var(--bi-ink); letter-spacing: -.015em;
  }
  .bi-tag {
    font-size: .8rem; color: var(--bi-ink-soft); margin-top: .12rem;
  }
  .bi-live {
    margin-left: auto; font-size: .72rem; color: var(--bi-ink-soft);
    display: flex; align-items: center; gap: .4rem; white-space: nowrap;
    padding: .28rem .6rem;
    border: 1px solid var(--bi-line); border-radius: 999px;
  }
  .bi-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--bi-brand);
    box-shadow: 0 0 0 0 var(--bi-brand);
    animation: bi-pulse 2.4s ease-out infinite;
  }
  @keyframes bi-pulse {
    0%   {box-shadow: 0 0 0 0 rgba(47, 158, 130, .5);}
    70%  {box-shadow: 0 0 0 6px rgba(47, 158, 130, 0);}
    100% {box-shadow: 0 0 0 0 rgba(47, 158, 130, 0);}
  }

  /* ---------- Chat bubbles ---------- */

  [data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: .3rem 0;
    gap: .7rem;
    align-items: flex-start;
  }

  /* Avatar: a small ringed circle, not a plain square. */
  [data-testid="stChatMessage"] > img,
  [data-testid="stChatMessage"] > div:first-child {
    border-radius: 50%;
  }

  /* A bubble hugs its text instead of filling the whole column. That
     is what makes the page read like a conversation.

     `margin-inline: 0` is not optional. Streamlit gives this box
     automatic left and right margins, which CENTRE the bubble: the
     shorter the message the bigger the margin, so short replies float
     in the middle, far from their avatar. Zeroing the margins pins
     each bubble to its own side - left for the assistant, right for
     the user. */
  [data-testid="stChatMessageContent"] {
    background: var(--bi-surface);
    border: 1px solid var(--bi-line);
    border-radius: 16px 16px 16px 4px;
    padding: .7rem 1rem;
    box-shadow: var(--bi-shadow);

    /* The assistant bubble is allowed to run full width. */
    flex: 1 1 auto;
    width: auto;
    max-width: none;
    min-width: 0;

    margin-left: 0 !important;
    margin-right: 0 !important;

    box-sizing: border-box;
    }
    /* This is what clipped the bubbles, and the cause is not where you
     would look for it. Streamlit puts `margin-bottom: -16px` on every
     stMarkdownContainer to cancel the surrounding vertical block's
     `gap: 16px`. Inside a chat bubble there is only ONE child, so
     there is no gap to cancel - the negative margin just makes the box
     16px shorter than its text, and the last line spills below the
     bubble. Cancelled here, inside bubbles only. */
  [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
    margin-bottom: 0 !important;
  }

  [data-testid="stChatMessageContent"] p {
    line-height: 1.65; margin-bottom: .55rem;
  }
  [data-testid="stChatMessageContent"] p:last-child {margin-bottom: 0;}
  [data-testid="stChatMessageContent"] ul,
  [data-testid="stChatMessageContent"] ol {margin-bottom: .3rem;}
  [data-testid="stChatMessageContent"] li {line-height: 1.6;}

  /* User side: aligned right, in the brand colour. */
  [class*="st-key-bi-user-"] [data-testid="stChatMessage"] {
    flex-direction: row-reverse;
  }
  [class*="st-key-bi-user-"] [data-testid="stChatMessageContent"] {
    /* The user bubble stays as short as its text. */
    flex: 0 1 auto;
    width: fit-content;
    max-width: 78%;
    min-width: 0;

    background: var(--bi-brand);
    border-color: var(--bi-brand);
    border-radius: 16px 16px 4px 16px;

    margin-left: 0 !important;
    margin-right: 0 !important;
}
  [class*="st-key-bi-user-"] [data-testid="stChatMessageContent"],
  [class*="st-key-bi-user-"] [data-testid="stChatMessageContent"] p {
    color: var(--bi-brand-ink);
  }

  /* ---------- Example question cards ---------- */

  .bi-hint {
    font-size: .72rem; font-weight: 650; letter-spacing: .07em;
    text-transform: uppercase; color: var(--bi-ink-soft);
    margin: 1.4rem 0 .6rem;
  }
  .stButton > button {
    width: 100%;
    text-align: left;
    border: 1px solid var(--bi-line);
    background: var(--bi-raised);
    color: var(--bi-ink);
    border-radius: 14px;
    padding: .7rem .9rem;
    transition: border-color .16s ease, background .16s ease,
                transform .16s ease;
  }
  .stButton > button:hover {
    border-color: var(--bi-brand);
    background: var(--bi-surface);
    color: var(--bi-ink);
    transform: translateY(-1px);
  }
  .stButton > button:focus:not(:active) {
    border-color: var(--bi-brand); color: var(--bi-ink);
  }

  /* The button label is written as two paragraphs: a title, then the
     question. Streamlit lays labels out on one line with an ellipsis,
     so this puts them back to two lines that are allowed to wrap. */
  .stButton > button [data-testid="stMarkdownContainer"] {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
  }

  /* Buttons in one column row share a height, so a card with longer
     text does not leave the row looking crooked. */
  [data-testid="stColumn"] .stButton,
  [data-testid="stColumn"] .stButton > button {height: 100%;}

  .stButton > button div,
  .stButton > button p {
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
    overflow-wrap: anywhere;
  }
  .stButton > button p {
    margin: 0; font-size: .85rem; line-height: 1.45;
  }

  /* Only the example buttons hold TWO paragraphs. Single-paragraph
     buttons in the sidebar must not shrink and change colour too, so
     the title is selected as a first-child that is not also a
     last-child. */
  .stButton > button p:first-child:not(:last-child) {
    font-size: .73rem; font-weight: 650; letter-spacing: .04em;
    color: var(--bi-brand); margin-bottom: .22rem;
  }
  .stButton > button p:last-child:not(:first-child) {
    color: var(--bi-ink-soft);
  }
  .stButton > button:hover p:last-child:not(:first-child) {
    color: var(--bi-ink);
  }

  /* ---------- Input row ---------- */

  [data-testid="stChatInput"] {
    border-radius: 14px;
    border-color: var(--bi-line);
  }
  [data-testid="stChatInput"] textarea {font-size: .94rem;}
  [data-testid="stChatInput"]:focus-within {border-color: var(--bi-brand);}

  /* ---------- Footer ---------- */

  .bi-foot {
    margin-top: 2.4rem; padding-top: 1.1rem;
    border-top: 1px solid var(--bi-line);
  }

  /* The disclaimer gets more contrast and a border of its own. It
     used to be faint grey at footnote size: the most important
     sentence on the page was the hardest one to read. */
  .bi-catatan {
    display: flex; align-items: flex-start; gap: .6rem;
    max-width: 34rem; margin: 0 auto;
    padding: .7rem .9rem;
    background: var(--bi-warn-dim);
    border: 1px solid var(--bi-warn-line);
    border-radius: 12px;
    font-size: .82rem; line-height: 1.55;
    color: var(--bi-warn-ink);
  }
  .bi-catatan-tanda {
    flex: 0 0 auto;
    font-size: .64rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase;
    padding: .18rem .45rem; margin-top: .08rem;
    border-radius: 6px;
    background: var(--bi-warn); color: var(--bi-warn-on);
    white-space: nowrap;
  }
  .bi-catatan strong {font-weight: 700;}

  .bi-foot-kecil {
    text-align: center; font-size: .73rem; color: var(--bi-ink-soft);
    margin-top: .8rem; line-height: 1.7;
  }

  /* ---------- Sidebar ---------- */

  [data-testid="stSidebar"] {
    background: var(--bi-canvas);
    border-right: 1px solid var(--bi-line);
  }
  [data-testid="stSidebar"] .stButton > button {text-align: center;}

  .bi-doc {
    position: relative;
    font-size: .82rem; color: var(--bi-ink);
    padding: .55rem .7rem .55rem 1rem;
    margin-bottom: .4rem;
    background: var(--bi-surface);
    border: 1px solid var(--bi-line);
    border-radius: 11px;
    overflow-wrap: anywhere;
    transition: border-color .16s ease;
  }
  .bi-doc::before {
    content: ""; position: absolute;
    left: .45rem; top: .7rem; bottom: .7rem;
    width: 2px; border-radius: 2px;
    background: var(--bi-brand);
    opacity: .55;
  }
  .bi-doc:hover {border-color: var(--bi-brand);}

  .bi-label {
    font-size: .7rem; font-weight: 650; letter-spacing: .08em;
    text-transform: uppercase; color: var(--bi-ink-soft);
    margin: .2rem 0 .6rem;
  }
  .bi-gap {height: 1.6rem;}
</style>
"""

st.markdown(GAYA, unsafe_allow_html=True)


# ============================================================
# STATE - isolated per session
# ============================================================

if "pesan" not in st.session_state:
    st.session_state.pesan = [{"role": "assistant", "content": SAPAAN}]

# Conversation memory: which stock is being discussed, plus the last few
# turns. The contents belong to `src/memory.py`; this page only keeps it
# in session_state so it is never shared between users.
if "memori" not in st.session_state:
    st.session_state.memori = memory.memori_baru()

if "antrian" not in st.session_state:
    st.session_state.antrian = None

if "owner_id" not in st.session_state:
    st.session_state.owner_id = new_owner_id()

try:
    maybe_cleanup()
except Exception:
    logger.exception("Pembersihan dokumen sementara gagal")


def tampilkan(peran, isi, nomor):
    """
    One chat bubble. `nomor` is what keeps the container key unique.
    """
    lambang = LAMBANG_ASISTEN if peran == "assistant" else LAMBANG_PENGGUNA

    with st.container(key=kunci_gelembung(peran, nomor)):
        with st.chat_message(peran, avatar=lambang):
            st.markdown(isi)


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    f"""
    <div class="bi-head">
      <div class="bi-mark">{LAMBANG_ASISTEN}</div>
      <div>
        <div class="bi-name">{NAMA}</div>
        <div class="bi-tag">{TAGLINE}</div>
      </div>
      <div class="bi-live"><span class="bi-dot"></span>Siap membantu</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR - documents, not settings
# ============================================================

with st.sidebar:
    st.markdown('<div class="bi-label">Dokumen riset</div>', True)

    try:
        # The sidebar answers "what can this assistant answer from", so
        # The curated corpus is shared; visitor uploads are listed only
        # for the current browser session, never from the global folder.
        dokumen = daftar_dokumen(include_additional=False, verifikasi=True)
        dokumen += list_session_pdfs(st.session_state.owner_id)
    except Exception:
        logger.exception("Gagal membaca daftar dokumen")
        dokumen = []

    if dokumen:
        for d in dokumen:
            judul = (
                d["name"][:-4]
                if d["name"].lower().endswith(".pdf")
                else d["name"]
            )
            st.markdown(f'<div class="bi-doc">{escape(judul)}</div>', True)
    else:
        st.caption("Belum ada dokumen riset.")

    st.markdown('<div class="bi-gap"></div>', True)
    st.markdown('<div class="bi-label">Tambah dokumen</div>', True)
    st.caption("Maksimal 2 PDF per sesi, 5 MB dan 30 halaman per PDF. "
               "Hanya Anda yang dapat menanyakannya; data dihapus otomatis "
               "setelah 24 jam.")

    berkas = st.file_uploader(
        "Pilih berkas PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Dokumen ini hanya tersedia dalam sesi browser Anda.",
    )

    if berkas is not None:
        if st.button("Tambahkan ke koleksi", use_container_width=True):
            with st.spinner("Membaca dan menyiapkan dokumen…"):
                try:
                    ringkas = add_session_pdf(
                        berkas.getvalue(),
                        berkas.name,
                        st.session_state.owner_id,
                    )

                except IngestionError as kendala:
                    # IngestionError messages are written for the user.
                    st.warning(str(kendala))

                except QuotaExceeded as kendala:
                    st.warning(str(kendala))

                except Exception:
                    # Anything else must not leak as-is: it can carry
                    # the database address or an internal trace.
                    logger.exception("Penambahan dokumen gagal")
                    st.error(
                        "Knowledge gagal ditambahkan. Silakan coba lagi."
                    )

                else:
                    st.success(
                        "Knowledge berhasil ditambahkan dan siap digunakan."
                    )
                    st.caption(f"{ringkas['pages']} halaman siap dicari.")

    st.markdown('<div class="bi-gap"></div>', True)

    if st.button("Percakapan baru", use_container_width=True):
        st.session_state.pesan = [{"role": "assistant", "content": SAPAAN}]
        st.session_state.memori = memory.lupakan()
        st.rerun()


# ============================================================
# CONVERSATION
# ============================================================

for nomor, pesan in enumerate(st.session_state.pesan):
    tampilkan(pesan["role"], pesan["content"], nomor)


# ============================================================
# INPUT
# ============================================================

pertanyaan = st.chat_input("Tanyakan sesuatu tentang saham…")

if st.session_state.antrian:
    pertanyaan = st.session_state.antrian
    st.session_state.antrian = None

if pertanyaan:
    st.session_state.pesan.append({"role": "user", "content": pertanyaan})

    tampilkan("user", pertanyaan, len(st.session_state.pesan) - 1)

    nomor_jawaban = len(st.session_state.pesan)

    with st.container(key=kunci_gelembung("assistant", nomor_jawaban)):
        with st.chat_message("assistant", avatar=LAMBANG_ASISTEN):
            with st.spinner("Sedang menelusuri dokumen riset…"):
                kegagalan = PESAN_GAGAL_UMUM

                try:
                    memori = st.session_state.memori

                    consume_question(st.session_state.owner_id)

                    hasil = jawab(
                        pertanyaan,
                        retriever=get_session_retriever(
                            st.session_state.owner_id
                        ),
                        session_context_data=memory.konteks_identitas(memori),
                        riwayat=memory.riwayat_teks(memori),
                        kueri_retrieval=memory.kueri_pencarian(
                            pertanyaan, memori
                        ),
                    )

                except QuotaExceeded as kendala:
                    hasil = None
                    kegagalan = str(kendala)

                except EmbeddingTidakTersedia:
                    # The install is broken, not the network. Say so
                    # instead of suggesting a retry that cannot work.
                    logger.exception("Model embedding tidak bisa dimuat")
                    hasil = None
                    kegagalan = PESAN_MODEL_TIDAK_SIAP

                except Exception:
                    # PgVector down, an expired service key, a corrupt
                    # document - none of it may take the page down.
                    logger.exception("Gagal menjawab pertanyaan")
                    hasil = None
                    kegagalan = PESAN_GAGAL_UMUM

            if hasil is None:
                jawaban = kegagalan
            else:
                jawaban = hasil.get("answer") or PESAN_GAGAL_UMUM

                if hasil.get("error"):
                    logger.warning(
                        "Jawaban selesai dengan catatan: %s", hasil["error"]
                    )

            st.markdown(jawaban)

    st.session_state.pesan.append({"role": "assistant", "content": jawaban})

    if hasil is not None:
        try:
            st.session_state.memori = memory.perbarui(
                st.session_state.memori, pertanyaan, hasil
            )
        except Exception:
            # Memory must not take the conversation down. If it breaks,
            # the conversation goes on without it instead of stopping.
            logger.exception("Gagal memperbarui memori percakapan")
            st.session_state.memori = memory.memori_baru()


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================
#
# Drawn AFTER the input is handled, not before. Drawn first, the question
# and its answer would land underneath them on the same turn - so the
# example cards would end up wedged in the middle of the conversation and
# only disappear on the next turn. By this point `pesan` has already
# grown, so the "conversation is still empty" test is right immediately.

if len(st.session_state.pesan) == 1:
    st.markdown('<div class="bi-hint">Coba tanyakan</div>', True)

    baris = st.columns(2)

    for indeks, (judul, isi) in enumerate(CONTOH):
        if baris[indeks % 2].button(
            f"{judul}\n\n{isi}",
            key=f"contoh_{indeks}",
            use_container_width=True,
        ):
            st.session_state.antrian = isi
            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="bi-foot">
      <div class="bi-catatan">
        <span class="bi-catatan-tanda">Penting</span>
        <span>
          Yang Anda baca di sini adalah <strong>ringkasan riset dari Gidion Depari</strong>,
          <strong>bukan nasihat investasi</strong>. Keputusan jual dan beli
          tetap ada di tangan Anda.
        </span>
      </div>
      <div class="bi-foot-kecil">
        Jawaban riset berasal dari dokumen koleksi Gidion Depari.
        Harga pasar dapat tertunda beberapa menit.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
