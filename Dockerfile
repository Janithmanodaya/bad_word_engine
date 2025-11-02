# Minimal Dockerfile for the FastAPI bad-words service
# Allow overriding the base Python image at build time for stability testing
ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# System dependencies: CA certs for HTTPS downloads
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and model artifacts
COPY app.py /app/app.py
COPY model /app/model

# Default server binding; override via SERVER_URL env var
ENV SERVER_URL=0.0.0.0:8000 \
    # Tame native BLAS/threads to reduce segfault risk in slim containers
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    # Enable ML by default; can disable via ML_DISABLE=1 at runtime
    ML_DISABLE=0 \
    # Low-resource defaults
    LOW_RESOURCE_MODE=1 \
    RUNTIME_DEPS_INSTALL=0 \
    # Subprocess isolation disabled by default for 0.2 vCPU stability (enable with PREDICT_SUBPROCESS=1 or PREDICT_IN_SUBPROCESS=1)
    PREDICT_SUBPROCESS=0 \
    PREDICT_IN_SUBPROCESS=0 \
    # Timeout only applies when subprocess mode is enabled
    PREDICT_TIMEOUT_SEC=30

EXPOSE 8000

# Uvicorn tuned for constrained CPU/RAM:
# - single worker
# - no access log
# - short keep-alive timeout
# - limited concurrency
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --loop asyncio --no-access-log --timeout-keep-alive 5 --limit-concurrency 16"]
