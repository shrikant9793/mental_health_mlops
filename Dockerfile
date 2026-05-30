# ─── Stage 1: Base Image ───────────────────────────────────────
FROM python:3.12.7-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Stage 2: Dependencies ────────────────────────────────────
FROM base AS dependencies

# Copy requirements first — leverage Docker cache
COPY requirements/base.txt requirements/base.txt
COPY requirements/prod.txt requirements/prod.txt

# Install production dependencies only
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements/prod.txt

# ─── Stage 3: Final Image ─────────────────────────────────────
FROM dependencies AS final

# Copy source code
COPY src/ src/
COPY configs/ configs/

# Create necessary directories
RUN mkdir -p data/raw data/processed models/artifacts logs reports

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI app
CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]