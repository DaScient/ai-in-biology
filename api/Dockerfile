# syntax=docker/dockerfile:1.6
# Multi-stage build for the API service.

# ---- Builder stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and application source
COPY pyproject.toml ./
COPY api/ ./api/
COPY data/ ./data/

# Install the project and its runtime dependencies into the system site-packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---- Final stage ----
FROM python:3.11-slim

WORKDIR /app

# Runtime system dependencies (curl is used by the healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and console scripts from the builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and data
COPY --from=builder /app/api ./api
COPY --from=builder /app/data ./data

# Create a non-root user
RUN useradd -m -u 1000 api-user && chown -R api-user:api-user /app
USER api-user

EXPOSE 8000

# Health check against the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
