# Testing Guide

## Quick Test

### 1. Start Modelium

```bash
python -m modelium.cli serve
```

### 2. Drop a Model

```bash
# Create directory
mkdir -p models/incoming

# Download GPT-2 from HuggingFace
git clone https://huggingface.co/gpt2 models/incoming/gpt2
```

### 3. Watch the Logs

You should see:

```
📋 New model discovered: gpt2
   Analyzing model...
   Architecture: GPT2LMHeadModel
   Type: causal_lm
   Size: 0.55GB
   🎯 Brain decision: vllm
   Loading model...
   ✅ gpt2 loaded successfully!
```

### 4. Test Inference

```bash
# Check model is loaded
curl http://localhost:8000/models | jq

# Run inference
curl -X POST http://localhost:8000/predict/gpt2 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq
```

Expected response:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "model": "gpt2",
  "choices": [{
    "text": " in a land far away, there lived a...",
    "finish_reason": "length"
  }]
}
```

## Test Checklist

### Health Check

```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### System Status

```bash
curl http://localhost:8000/status | jq
# Should show:
# - status: "running"
# - models_loaded: 1
# - gpu_count: N (if GPUs available)
```

### List Models

```bash
curl http://localhost:8000/models | jq
# Should show gpt2 with status "loaded"
```

### Inference

```bash
curl -X POST http://localhost:8000/predict/gpt2 \
  -d '{"prompt": "Hello", "max_tokens": 20}' | jq
# Should return completion
```

### Metrics

```bash
curl http://localhost:9090/metrics | grep modelium
# Should show prometheus metrics
```

## Test Scenarios

### Scenario 1: Single Model

```bash
# 1. Drop one model
cp model1/ models/incoming/

# 2. Wait for detection (max 30s)
tail -f modelium.log

# 3. Verify loaded
curl http://localhost:8000/models

# 4. Run inference
curl -X POST http://localhost:8000/predict/model1 -d '{"prompt":"test"}'
```

### Scenario 2: Multiple Models

```bash
# 1. Drop 3 models
cp model1/ models/incoming/
cp model2/ models/incoming/
cp model3/ models/incoming/

# 2. Verify all detected
curl http://localhost:8000/models | jq '.models | length'
# Should show 3

# 3. Use them
curl -X POST http://localhost:8000/predict/model1 -d '{"prompt":"test1"}'
curl -X POST http://localhost:8000/predict/model2 -d '{"prompt":"test2"}'
curl -X POST http://localhost:8000/predict/model3 -d '{"prompt":"test3"}'
```

### Scenario 3: Idle Unload

```bash
# 1. Load model
cp model/ models/incoming/

# 2. Wait 5+ minutes without requests
sleep 360

# 3. Check status
curl http://localhost:8000/models | jq '.models[] | select(.name=="model")'
# Status should be "unloaded"

# 4. Make request (will reload)
curl -X POST http://localhost:8000/predict/model -d '{"prompt":"test"}'
# Will trigger reload
```

## Docker Testing

```bash
# 1. Build
docker build -t modelium:latest .

# 2. Run
docker-compose up -d

# 3. Check
curl http://localhost:8000/health

# 4. Drop model
cp model/ models/incoming/

# 5. Test inference
curl -X POST http://localhost:8000/predict/model -d '{"prompt":"test"}'

# 6. View logs
docker-compose logs -f modelium-server

# 7. Stop
docker-compose down
```

## Troubleshooting Tests

### Model Not Detected

**Check**:
```bash
# 1. Directory exists?
ls models/incoming/

# 2. Model has config.json?
ls models/incoming/model/config.json

# 3. Watcher running?
grep "Model Watcher" modelium.log
```

### Model Not Loading

**Check**:
```bash
# 1. Runtime enabled?
grep "vllm:" modelium.yaml

# 2. GPU available?
nvidia-smi

# 3. Errors in logs?
grep "ERROR" modelium.log
```

### Inference Fails

**Check**:
```bash
# 1. Model loaded?
curl http://localhost:8000/models | jq '.models[] | select(.name=="model") | .status'

# 2. Runtime healthy?
# For vLLM:
curl http://localhost:8001/health

# 3. Check logs
tail -f modelium.log
```

## Performance Testing

### Load Test

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Run load test
hey -n 1000 -c 10 \
  -m POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","max_tokens":20}' \
  http://localhost:8000/predict/model

# Check:
# - Requests per second
# - Latency (p50, p95, p99)
# - Success rate
```

### Metrics Validation

```bash
# Watch metrics in real-time
watch -n 1 'curl -s http://localhost:9090/metrics | grep modelium'

# Should see:
# - modelium_requests_total increasing
# - modelium_latency_seconds tracking
# - modelium_model_idle_seconds updating
```

## Automated Testing

```python
#!/usr/bin/env python3
"""Automated Modelium test"""

import requests
import time

BASE_URL = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.json()["status"] == "healthy"
    print("✅ Health check passed")

def test_status():
    r = requests.get(f"{BASE_URL}/status")
    data = r.json()
    assert data["status"] == "running"
    print(f"✅ Status check passed (GPUs: {data['gpu_count']})")

def test_models():
    r = requests.get(f"{BASE_URL}/models")
    models = r.json()["models"]
    print(f"✅ Found {len(models)} models")
    return models

def test_inference(model_name):
    r = requests.post(
        f"{BASE_URL}/predict/{model_name}",
        json={"prompt": "test", "max_tokens": 10}
    )
    assert r.status_code == 200
    print(f"✅ Inference on {model_name} passed")

if __name__ == "__main__":
    print("Running Modelium tests...")
    
    test_health()
    test_status()
    models = test_models()
    
    if models:
        test_inference(models[0]["name"])
    
    print("\n✅ All tests passed!")
```

## Common Issues

### "No runtimes enabled"
**Fix**: Enable at least one in `modelium.yaml`

### "Model not found"
**Fix**: Check model directory and `config.json`

### "GPU not detected"
**Fix**: Install PyTorch with CUDA support

### "Runtime timeout"
**Fix**: Increase timeout in `modelium.yaml`

## Next Steps

- [Usage Guide](usage.md) - Full API documentation
- [Architecture](architecture.md) - How it works
- [Examples](../examples/) - Sample code
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Production deployment
