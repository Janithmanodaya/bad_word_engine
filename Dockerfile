# Minimal Dockerfile for the FastAPI bad-words service
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Optional: install Hugging Face datasets if you want SOLD-derived tokens
# RUN pip install --no-cache-dir datasets

# Copy application code
COPY app.py /app/app.py

# Default server binding; override via SERVER_URL env var
ENV SERVER_URL=0.0.0.0:8000

EXPOSE 8000

CMD ["python", "app.py"]