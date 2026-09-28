FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    XDG_CACHE_HOME=/home/app/.cache \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/huggingface /app \
    && chown -R app:app /home/app /app

# The default Linux torch wheel also pulls large CUDA libraries. This VPS
# has no GPU, so install the official CPU wheel before other dependencies.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

WORKDIR /app
COPY requirements-vps.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-vps.txt

COPY --chown=app:app app.py ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app .streamlit/config.toml ./.streamlit/config.toml
COPY --chown=app:app data/knowledge_base/primary/ ./data/knowledge_base/primary/

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
