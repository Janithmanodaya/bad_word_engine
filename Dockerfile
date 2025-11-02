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
    # Disable native C-extensions in text matching by default; can override at runtime
    DISABLE_NATIVE=1 \
    # Disable ML model by default to avoid potential native segfaults from scipy/sklearn in minimal containers.
    # You can re-enable by setting ML_DISABLE=0 at deploy/runtime if your base image provides stable BLAS/ SciPy wheels.
    ML_DISABLE=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
