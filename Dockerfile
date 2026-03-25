# AMP Memory Server — Production Docker Image
# Multi-stage build: builder → runtime (minimal)
#
# Usage:
#   docker build -t amp-memory .
#   docker run -p 8765:8765 -v $(pwd)/data:/data \
#     -e AMP_AGENT_ID=agent-1 \
#     -e AMP_USER_ID=default-user \
#     -e AMP_DB_PATH=/data/amp.db \
#     amp-memory
#
# With PostgreSQL + pgvector:
#   docker run -p 8765:8765 \
#     -e AMP_DSN="postgresql://amp:pass@postgres:5432/amp" \
#     -e OPENAI_API_KEY="sk-..." \
#     amp-memory

# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for scipy/numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY amp/ amp/

RUN pip install --no-cache-dir --upgrade pip build \
    && pip install --no-cache-dir ".[server]" \
    && pip wheel --no-cache-dir --wheel-dir=/wheels ".[server]"


# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title       = "AMP Memory Server"
LABEL org.opencontainers.image.description = "Agent Memory Protocol — open standard for AI agent memory"
LABEL org.opencontainers.image.url         = "https://amp-protocol.org"
LABEL org.opencontainers.image.source      = "https://github.com/amp-protocol/amp-python"
LABEL org.opencontainers.image.licenses    = "MIT"

# Runtime deps (for scipy/numpy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 amp
WORKDIR /app
RUN mkdir -p /data && chown amp:amp /data

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

# Copy application
COPY amp/ amp/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${AMP_PORT:-8765}/health')"

USER amp

# ── Environment ───────────────────────────────────────────────────────────
ENV AMP_AGENT_ID   = "amp-server"
ENV AMP_USER_ID    = "default-user"
ENV AMP_DB_PATH    = "/data/amp.db"
ENV AMP_HOST       = "0.0.0.0"
ENV AMP_PORT       = "8765"

EXPOSE 8765
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "amp.server.http_server"]
