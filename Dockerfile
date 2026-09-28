# Multi-stage build: slim runtime, fast dependency layer.

# ---------- Stage 1: dependencies ----------
FROM python:3.12-slim AS deps

WORKDIR /build
RUN pip install --no-cache-dir pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

# Non-root user: a container escape should not yield root.
RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/

RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8001

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    DEBUG=false \
    PORT=8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/api/v1/health').status==200 else 1)"

# One worker per container; scale horizontally with replicas instead.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
