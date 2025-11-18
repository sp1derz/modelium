# Getting Started with Modelium

The simplest AI model serving platform. Drop models, they load automatically.

## Prerequisites

- **Python 3.11+**
- **NVIDIA GPU** (optional, but recommended)
- **50GB disk space**

## Installation

### Quick Start

```bash
# 1. Clone
git clone https://github.com/sp1derz/modelium.git
cd modelium

# 2. Install
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"

# 3. Configure
cp modelium.yaml.example modelium.yaml
nano modelium.yaml  # Set which runtimes you want

# 4. Start
python -m modelium.cli serve
```

## Configuration

Edit `modelium.yaml`:

```yaml
# Which runtimes to use?
vllm:
  enabled: true    # ← For LLMs

triton:
  enabled: false   # ← For all models

ray_serve:
  enabled: false   # ← For Python models

# Where to watch for models?
orchestration:
  model_discovery:
    watch_directories:
      - /models/incoming  # ← Drop models here

# When to unload idle models?
  policies:
    evict_after_idle_seconds: 300  # 5 minutes

# Metrics
metrics:
  enabled: true
  port: 9090  # Prometheus at http://localhost:9090/metrics
```

That's it! No other configuration needed.

## First Model

```bash
# 1. Create directory
mkdir -p models/incoming

# 2. Drop a HuggingFace model
git clone https://huggingface.co/gpt2 models/incoming/gpt2

# 3. Watch Modelium logs
# You'll see:
# 📋 New model discovered: gpt2
# 🎯 Brain decision: vllm
# 🚀 Loading model...
# ✅ gpt2 loaded successfully!

# 4. Use it
curl http://localhost:8000/predict/gpt2 \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

## Docker (Recommended)

```bash
# Build (first time only, ~10 min)
docker build -t modelium:latest .

# Run
docker-compose up -d

# Check
curl http://localhost:8000/health
```

## Kubernetes

```bash
# Deploy
kubectl apply -k infra/k8s/

# Check
kubectl get pods -n modelium
```

See [DEPLOYMENT.md](../DEPLOYMENT.md) for full Kubernetes guide.

## Common Issues

### "No runtimes enabled"

Edit `modelium.yaml` and set at least one runtime to `enabled: true`.

### "Model not showing up"

1. Check model is in correct directory: `ls models/incoming/`
2. Model must have `config.json` file
3. Check Modelium logs for errors

### "GPU not detected"

```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

- [Usage Guide](usage.md) - All features and APIs
- [Architecture](architecture.md) - How it works
- [Examples](../examples/) - Sample code

## That's It!

The system is designed to be **simple**:
1. Configure which runtimes you want
2. Drop models in a folder
3. Use them via HTTP API

Modelium handles everything else:
- Auto-detection
- Runtime selection
- GPU allocation
- Load balancing
- Automatic unloading

**Maximum GPU utilization with minimum effort.**
