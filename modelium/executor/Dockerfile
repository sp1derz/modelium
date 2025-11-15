FROM python:3.10-slim

# Install Docker CLI for Docker-in-Docker
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY pyproject.toml /app/

# Install dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root

# Copy application code
COPY modelium/ /app/modelium/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run executor service
CMD ["uvicorn", "modelium.executor.service:app", "--host", "0.0.0.0", "--port", "8000"]

