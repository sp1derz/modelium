# Multi-stage build for Modelium server
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./

# Install Poetry and dependencies
RUN pip install poetry==1.7.0 && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY modelium/ ./modelium/
COPY configs/ ./configs/
COPY examples/ ./examples/

# Create directories for models and logs
RUN mkdir -p /models/incoming /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODELIUM_CONFIG=/app/modelium.yaml
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create non-root user for security
RUN useradd -m -u 1000 modelium && \
    chown -R modelium:modelium /app /models

USER modelium

# Default command
CMD ["python", "-m", "modelium.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]

