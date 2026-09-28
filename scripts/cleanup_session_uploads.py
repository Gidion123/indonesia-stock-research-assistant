"""Remove expired anonymous-upload vectors (run hourly with cron/systemd)."""

from src.session_documents import cleanup_expired


if __name__ == "__main__":
    print(f"Expired uploads removed: {cleanup_expired()}")
