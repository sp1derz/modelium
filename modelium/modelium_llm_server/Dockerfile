FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY pyproject.toml /app/

# Install dependencies
RUN pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root

# Copy application code
COPY modelium/ /app/modelium/
COPY forge_llm/ /app/forge_llm/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/forge-llm

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "-m", "forge.forge_llm.server.inference"]

