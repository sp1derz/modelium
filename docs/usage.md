# Usage Guide

## Quick Reference

```bash
# Start server
python -m modelium.cli serve

# Drop model
cp my-model /models/incoming/

# Use it
curl http://localhost:8000/predict/my-model \
  -d '{"prompt": "Hello", "max_tokens": 50}'

# That's it!
```

## CLI Commands

### Start Server

```bash
# Default (uses modelium.yaml)
python -m modelium.cli serve

# Custom config
python -m modelium.cli serve --config custom.yaml

# Custom host/port
python -m modelium.cli serve --host 0.0.0.0 --port 8080

# Background
nohup python -m modelium.cli serve > modelium.log 2>&1 &
```

## API Endpoints

### Health Check

```bash
GET /health

# Returns:
{"status": "healthy"}
```

### System Status

```bash
GET /status

# Returns:
{
  "status": "running",
  "organization": "my-company",
  "gpu_count": 4,
  "gpu_enabled": true,
  "brain_enabled": true,
  "models_loaded": 3,
  "models_discovered": 5
}
```

### List Models

```bash
GET /models

# Returns:
{
  "models": [
    {
      "name": "gpt2",
      "status": "loaded",
      "runtime": "vllm",
      "gpu": 0,
      "path": "/models/incoming/gpt2"
    },
    {
      "name": "bert",
      "status": "unloaded",
      "runtime": "ray",
      "gpu": null
    }
  ]
}
```

### Run Inference

```bash
POST /predict/{model_name}
Content-Type: application/json

{
  "prompt": "Your input text",
  "max_tokens": 100,
  "temperature": 0.7,
  "organizationId": "your-org"  # Optional, for multi-tenancy
}

# Example:
curl -X POST http://localhost:8000/predict/gpt2 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Response:
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1699...,
  "model": "gpt2",
  "choices": [{
    "text": " in a land far away...",
    "index": 0,
    "finish_reason": "length"
  }]
}
```

## Configuration

### Minimal Config

```yaml
# modelium.yaml

# Enable at least one runtime
vllm:
  enabled: true

# Watch directory
orchestration:
  model_discovery:
    watch_directories:
      - /models/incoming
```

### Full Config

```yaml
# Organization (multi-tenancy)
organization:
  id: "my-company"
  name: "My Company"

# Runtimes
vllm:
  enabled: true
triton:
  enabled: false
ray_serve:
  enabled: false

# Model discovery
orchestration:
  enabled: true
  model_discovery:
    watch_directories:
      - /models/incoming
    scan_interval_seconds: 30
  
  # Unload policy
  policies:
    evict_after_idle_seconds: 300  # 5 min idle = unload
    always_loaded: []  # Models that never unload

# Metrics
metrics:
  enabled: true
  port: 9090

# Brain (optional)
modelium_brain:
  enabled: true
  fallback_to_rules: true

# GPU
gpu:
  enabled: auto  # auto-detect
```

## Examples

### Deploy LLM

```bash
# 1. Drop model
git clone https://huggingface.co/gpt2 models/incoming/gpt2

# Modelium automatically:
# - Detects it's an LLM
# - Chooses vLLM runtime
# - Loads to GPU
# - Exposes /predict/gpt2

# 2. Use it
curl http://localhost:8000/predict/gpt2 \
  -d '{"prompt": "Hello world", "max_tokens": 50}'
```

### Deploy Multiple Models

```bash
# Drop multiple models
cp model1/ models/incoming/
cp model2/ models/incoming/
cp model3/ models/incoming/

# Modelium loads them all
# Unloads idle ones automatically
# Maximum GPU utilization!
```

### Monitor Usage

```bash
# Prometheus metrics
curl http://localhost:9090/metrics

# Key metrics:
# modelium_requests_total{model="gpt2",runtime="vllm"} 1234
# modelium_latency_seconds{model="gpt2",p="50"} 0.123
# modelium_model_idle_seconds{model="gpt2"} 45.2
```

## Python Client (Optional)

```python
import requests

# Run inference
response = requests.post(
    "http://localhost:8000/predict/gpt2",
    json={
        "prompt": "Hello, my name is",
        "max_tokens": 50,
        "temperature": 0.7
    }
)

print(response.json())

# List models
models = requests.get("http://localhost:8000/models").json()
for model in models["models"]:
    print(f"{model['name']}: {model['status']}")
```

## Docker

```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f modelium-server

# Stop
docker-compose down

# Drop model (from host)
cp my-model models/incoming/
```

## Kubernetes

```bash
# Deploy
kubectl apply -k infra/k8s/

# Status
kubectl get pods -n modelium

# Logs
kubectl logs -f deployment/modelium-server -n modelium

# Drop model
kubectl cp my-model modelium-server:/models/incoming/
```

## Common Patterns

### Always-Loaded Models

```yaml
# For critical models that should never unload
orchestration:
  policies:
    always_loaded:
      - "production-llm"
      - "main-model"
```

### Custom Unload Time

```yaml
# Unload after 10 minutes idle
orchestration:
  policies:
    evict_after_idle_seconds: 600
```

### Multi-Tenancy

```yaml
# Track usage per organization
organization:
  id: "company-a"

# In requests:
curl http://localhost:8000/predict/model \
  -d '{"prompt": "...", "organizationId": "company-a"}'
```

## Troubleshooting

### Model Not Loading

```bash
# 1. Check model directory
ls -la models/incoming/your-model/

# 2. Must have config.json
ls models/incoming/your-model/config.json

# 3. Check logs
tail -f modelium.log
```

### Inference Timeout

```bash
# Check if model is loaded
curl http://localhost:8000/models | jq '.models[] | select(.name=="your-model")'

# If status != "loaded", wait for loading to complete
```

### GPU Memory Full

Models will queue if GPU is full. Check:

```bash
# GPU usage
nvidia-smi

# Loaded models
curl http://localhost:8000/models

# Force unload
# (Coming soon: Manual unload API)
```

## Best Practices

1. **Model Names**: Use descriptive names (not just "model.pt")
2. **Directory Structure**: One model per directory
3. **Config Files**: Always include `config.json` from HuggingFace
4. **Monitoring**: Set up Prometheus + Grafana
5. **Always-Loaded**: For critical models with constant traffic

## Performance Tips

- **Small models** (<2GB): Load instantly
- **Large models** (>10GB): 30-120s load time
- **vLLM**: Best for LLMs (continuous batching)
- **Ray**: Best for custom Python models
- **Triton**: Best for ONNX/TensorRT

## Next Steps

- [Architecture](architecture.md) - Understand how it works
- [Getting Started](getting-started.md) - Installation guide
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Production deployment
- [Examples](../examples/) - Sample code
