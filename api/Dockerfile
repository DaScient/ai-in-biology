FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Copy application source before installing the package
COPY api/ api/

# Install Python dependencies needed at runtime
RUN pip install --no-cache-dir .

# Expose API port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
