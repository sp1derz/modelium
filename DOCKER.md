# Docker Quick Start

> **Note**: The Docker image is ~8-10GB due to CUDA runtime + PyTorch + vLLM dependencies. 
> This is normal for GPU-accelerated ML workloads. Build time: 10-15 minutes.

## Build and Run Locally

```bash
# 1. Build the image
docker build -t modelium:latest .

# 2. Run with GPU
docker run --gpus all -p 8000:8000 -p 9090:9090 \
  -v $(pwd)/modelium.yaml:/app/modelium.yaml:ro \
  -v $(pwd)/models:/models \
  modelium:latest

# 3. Check it's running
curl http://localhost:8000/health
```

## Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## Configuration

Mount your config file:
```bash
docker run --gpus all \
  -v /path/to/your/modelium.yaml:/app/modelium.yaml:ro \
  -p 8000:8000 \
  modelium:latest
```

## Model Storage

Mount model directory:
```bash
docker run --gpus all \
  -v /path/to/models:/models \
  -p 8000:8000 \
  modelium:latest
```

Then drop models into `/path/to/models/incoming/`

## Environment Variables

```bash
docker run --gpus all \
  -e LOG_LEVEL=DEBUG \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -p 8000:8000 \
  modelium:latest
```

## GPU Selection

```bash
# Use specific GPUs
docker run --gpus '"device=0,1"' -p 8000:8000 modelium:latest

# Use all GPUs
docker run --gpus all -p 8000:8000 modelium:latest
```

## Health Check

```bash
# Manual check
docker exec modelium-server curl http://localhost:8000/health

# Docker healthcheck (automatic)
docker ps  # Shows health status
```

## Logs

```bash
# Follow logs
docker logs -f modelium-server

# Last 100 lines
docker logs --tail 100 modelium-server
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Permission Denied

```bash
# Run as your user
docker run --user $(id -u):$(id -g) \
  --gpus all -p 8000:8000 \
  -v $(pwd)/models:/models \
  modelium:latest
```

### Out of Memory

```bash
# Limit memory
docker run --gpus all --memory=32g --memory-swap=64g \
  -p 8000:8000 modelium:latest
```

## Production Deployment

Use docker-compose with restart policies:

```yaml
services:
  modelium-server:
    image: modelium:latest
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]
```

## Image Registry

### Push to GitHub Container Registry

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag
docker tag modelium:latest ghcr.io/sp1derz/modelium:latest

# Push
docker push ghcr.io/sp1derz/modelium:latest
```

### Pull from Registry

```bash
docker pull ghcr.io/sp1derz/modelium:latest
docker run --gpus all -p 8000:8000 ghcr.io/sp1derz/modelium:latest
```

## Image Size Optimization

Current image: ~8-10GB (CUDA runtime + PyTorch + ML dependencies)

**Already optimized**:
- ✅ Clears Poetry cache after install
- ✅ Removes pip cache
- ✅ Deletes __pycache__ and .pyc files
- ✅ Uses runtime-only CUDA image (not devel)

**Why so large?**
```
CUDA runtime:         ~2GB
PyTorch + dependencies: ~4GB
vLLM dependencies:     ~2GB
Modelium code:         ~100MB
Total:                ~8-10GB
```

**Can't reduce much more** without losing functionality. CPU-only would be ~2GB but defeats the purpose.

## Multi-Stage Build (Future)

For even smaller images, could use multi-stage:

```dockerfile
FROM base AS builder
RUN poetry install

FROM base AS runtime  
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
# Saves ~500MB
```

Trade-off: More complex, harder to debug.

## CI/CD Integration

The GitHub Actions workflow automatically:
1. Builds image on push to main
2. Tags with commit SHA and `latest`
3. Pushes to ghcr.io/sp1derz/modelium

Use in production:
```bash
docker pull ghcr.io/sp1derz/modelium:SHA
docker run --gpus all -p 8000:8000 ghcr.io/sp1derz/modelium:SHA
```

