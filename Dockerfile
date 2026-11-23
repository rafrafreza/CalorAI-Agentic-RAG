# syntax=docker/dockerfile:1
# One image, one origin: FastAPI serves the API and the built React app on :8000.

# ── 1. Build the web app ────────────────────────────────────────────────────
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── 2. Runtime ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CALORAI_DB_PATH=/data/calorai.db \
    CALORAI_UPLOAD_DIR=/data/uploads \
    CALORAI_CACHE_DIR=/data/cache \
    CALORAI_STATIC_DIR=/app/frontend/dist
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/ backend/
COPY --from=web /web/dist frontend/dist

RUN useradd --create-home --uid 10001 calorai && mkdir -p /data && chown calorai:calorai /data
USER calorai
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
