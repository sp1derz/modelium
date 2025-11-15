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

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/modelium-llm

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "-m", "modelium.modelium_llm.server.inference"]

