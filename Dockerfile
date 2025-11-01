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
ENV SERVER_URL=0.0.0.0:8000

EXPOSE 8000

CMD ["python", "app.py"]
