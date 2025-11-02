# Minimal Dockerfile for the FastAPI bad-words service
FROM python:3.11-slim

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
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # Enable ML by default; can disable via ML_DISABLE=1 at runtime
    ML_DISABLE=0 \
    # Enable subprocess isolation for predictions to prevent main-process segfaults
    PREDICT_SUBPROCESS=1 \
    # Timeout only applies when subprocess mode is enabled
    PREDICT_TIMEOUT_SEC=30

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
