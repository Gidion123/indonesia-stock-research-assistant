"""Create a fresh curated index once; rebuild only when explicitly requested.

    docker compose --env-file .env.vps -f compose.vps.yaml exec app \
        python -m scripts.bootstrap_vps
    docker compose --env-file .env.vps -f compose.vps.yaml exec app \
        python -m scripts.bootstrap_vps --rebuild
"""

import argparse

import psycopg

from src import config
from src.ingestion import bangun_ulang
from src.vector_store import get_psycopg_connection_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true", help="Replace the curated index from primary PDFs"
    )
    args = parser.parse_args()

    pdfs = sorted(config.PRIMARY_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("Tidak ada PDF di data/knowledge_base/primary/.")
    print(f"PDF primer di image: {len(pdfs)}")
    for pdf in pdfs:
        print(f"  {pdf.name}")

    with psycopg.connect(get_psycopg_connection_url(), connect_timeout=10) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (f"public.{config.TABLE_NAME}",))
            exists = cursor.fetchone()[0] is not None

    if not exists:
        from scripts.create_database import create_vector_table

        create_vector_table()

    with psycopg.connect(get_psycopg_connection_url(), connect_timeout=10) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {config.TABLE_NAME}")
            existing_chunks = cursor.fetchone()[0]

    if existing_chunks and not args.rebuild:
        print(
            f"Indeks sudah berisi {existing_chunks} chunk. "
            "Lewati rebuild; gunakan --rebuild jika korpus primer berubah."
        )
        return

    print("Membangun indeks primer dari PDF...")
    report = bangun_ulang()
    print(
        f"Selesai: {report['pdf_files']} PDF, "
        f"{report['pages_kept']} halaman dipakai, {report['chunks']} chunk."
    )


if __name__ == "__main__":
    main()
