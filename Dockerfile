# Multi-stage build for Modelium server
# Using latest stable CUDA 12.6 with Ubuntu 24.04 (Nov 2025)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base

# Install system dependencies including Python 3.12
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Remove PEP 668 restriction (safe in Docker, not on host OS)
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# Upgrade pip to latest (--ignore-installed to avoid conflict with Debian packages)
RUN python -m pip install --upgrade --ignore-installed pip setuptools wheel

WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./

# Install Poetry 1.8.5 (latest)
RUN pip install poetry==1.8.5

# Configure Poetry and install dependencies (using --only main instead of deprecated --no-dev)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --extras all --no-interaction --no-ansi

# Clean up to save space (separate step so install errors aren't hidden)
RUN poetry cache clear pypi --all -n || true && \
    poetry cache clear _default_cache --all -n || true && \
    pip cache purge || true && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/lib/python3.12 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -type f -name '*.pyc' -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -type f -name '*.pyo' -delete 2>/dev/null || true

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

# Create non-root user for security (UID 10000 to avoid conflicts)
RUN useradd -m -u 10000 modelium && \
    chown -R modelium:modelium /app /models

USER modelium

# Default command (use defaults from CLI, no args needed)
CMD ["python", "-m", "modelium.cli", "serve"]

