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

# 2. Install Modelium
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"

# 3. Install Runtime (choose ONE or more)

# Option A: vLLM (for LLMs) - Linux+CUDA only
pip install vllm

# Option B: Ray Serve (for general models)
pip install ray[serve]

# Option C: Triton (for all models) - see below for setup

# 4. Configure
cp modelium.yaml.example modelium.yaml
nano modelium.yaml  # Enable the runtime(s) you installed

# 5. Start
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

## Runtime Setup Details

### vLLM (Recommended for LLMs)

**What**: High-performance LLM serving with continuous batching  
**When**: For GPT, Llama, Mistral, Qwen, and other LLMs  
**Requirements**: Linux + CUDA

```bash
# Install vLLM
pip install vllm

# Configure
# Edit modelium.yaml:
vllm:
  enabled: true

# That's it!
```

**✅ AUTO-SPAWNED**: Modelium automatically spawns vLLM processes when loading models. You do **NOT** need to:
- ❌ Start vLLM containers manually
- ❌ Run `vllm serve` commands
- ❌ Configure vLLM endpoints

**How it works**:
1. You drop a model (e.g., GPT-2) in `/models/incoming/`
2. Modelium detects it's an LLM
3. Modelium spawns: `python -m vllm.entrypoints.openai.api_server --model gpt2 --port 8100`
4. One vLLM process per model, automatically managed

**You only install vLLM once, Modelium handles the rest!**

### Ray Serve (For General Models)

**What**: Scalable model serving for Python models  
**When**: For PyTorch models, custom models, non-LLMs  
**Requirements**: Any OS

```bash
# Install Ray
pip install ray[serve]

# Configure
# Edit modelium.yaml:
ray_serve:
  enabled: true

# Done!
```

**✅ AUTO-INITIALIZED**: Modelium can initialize Ray automatically if not running. You do **NOT** need to:
- ❌ Start Ray cluster manually (unless you want multi-node)
- ❌ Run `ray start` commands (for single-node)

**How it works**:
1. You drop a model in `/models/incoming/`
2. Modelium detects it needs Ray
3. Modelium initializes Ray (if not running): `ray.init()`
4. Modelium deploys model as Ray Serve deployment

**For single-node (local): No manual setup needed!**  
**For multi-node: Start Ray cluster first, then Modelium connects to it.**

### Triton (For All Models)

**What**: NVIDIA's inference server, supports ONNX, TensorRT, PyTorch  
**When**: For maximum performance, TensorRT optimizations  
**Requirements**: Docker + NVIDIA GPU

```bash
# Start Triton server (REQUIRED - Modelium doesn't start this)
docker run -d --gpus all \
  -p 8003:8000 -p 8004:8001 -p 8005:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models/triton-repository

# Verify running
curl http://localhost:8003/v2/health/ready

# Configure
# Edit modelium.yaml:
triton:
  enabled: true

# Done!
```

**Important**: Unlike vLLM and Ray, Triton must be started manually before Modelium.

### Which Runtime Should I Use?

| Model Type | Recommended | Alternative |
|------------|-------------|-------------|
| **LLMs** (GPT, Llama, etc.) | vLLM | Ray |
| **Vision** (ResNet, YOLO) | Triton (TensorRT) | Ray |
| **Text** (BERT, etc.) | vLLM | Ray |
| **Custom Python** | Ray | - |
| **ONNX models** | Triton | - |

**Simplest Setup**: Just use vLLM for everything (if you have Linux+CUDA)

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

### "No runtimes available" on startup

**Problem**: Modelium can't find any installed runtimes

**Solution**:
```bash
# Install at least one:
pip install vllm        # For LLMs
pip install ray[serve]  # For general models

# Or start Triton:
docker run --gpus all -p 8003:8000 nvcr.io/nvidia/tritonserver:latest
```

### "vLLM not installed" when loading model

**Problem**: vLLM is enabled in config but not installed

**Solution**:
```bash
# Option 1: Install vLLM
pip install vllm

# Option 2: Disable in config
# Edit modelium.yaml:
vllm:
  enabled: false
```

### "Triton not ready" when loading model

**Problem**: Triton is enabled but server is not running

**Solution**:
```bash
# Start Triton server first
docker run -d --gpus all -p 8003:8000 \
  nvcr.io/nvidia/tritonserver:24.01-py3
  
# Then start Modelium
python -m modelium.cli serve
```

### "Model not showing up"

**Problem**: Model not detected by watcher

**Solution**:
1. Check model is in correct directory: `ls models/incoming/`
2. Model directory must contain `config.json` file
3. Check Modelium logs for errors: `tail -f modelium.log`

### "GPU not detected"

**Problem**: PyTorch can't see CUDA GPUs

**Solution**:
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### "Model status: error" after loading

**Problem**: Runtime failed to load model

**Solution**:
1. Check runtime is actually installed/running
2. Check GPU has enough memory: `nvidia-smi`
3. Check Modelium logs for specific error
4. Try with smaller model first (e.g., GPT-2)

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
