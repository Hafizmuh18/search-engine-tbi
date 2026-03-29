FROM python:3.11-slim

WORKDIR /app

# System deps untuk scipy/numpy build
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps dulu (layer ini di-cache selama requirements.txt tidak berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (kecuali yang ada di .dockerignore)
COPY . .

# Buat direktori yang dibutuhkan jika belum ada
RUN mkdir -p index tmp

EXPOSE 8000

# Healthcheck bawaan container
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]