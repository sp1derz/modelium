# Getting Started with Modelium

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.0+ (recommended)
- 8GB+ RAM (16GB+ for production)
- Linux, macOS, or Windows (WSL2)
- Docker (optional, for containerized deployment)

## Deployment Options

Choose your deployment method:

1. **🐳 Docker Compose** (Recommended) - [DOCKER.md](../DOCKER.md)
   - Fastest setup
   - GPU support included
   - Good for local testing and small production

2. **☸️ Kubernetes** (Production) - [DEPLOYMENT.md](../DEPLOYMENT.md)
   - Enterprise-grade
   - Auto-scaling
   - Multi-node support

3. **🐍 Python CLI** (Development) - Continue below
   - Full control
   - Easy debugging
   - Ideal for development

## Installation (Python CLI)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourorg/modelium.git
cd modelium
```

### Step 2: Install Dependencies

**Basic installation:**
```bash
pip install -e .
```

**With specific runtimes:**
```bash
# For LLM support (vLLM)
pip install -e ".[vllm]"

# For general models (Ray Serve)
pip install -e ".[ray]"

# Everything (all runtimes)
pip install -e ".[all]"
```

### Step 3: Initialize Configuration

```bash
modelium init
```

This creates `modelium.yaml` with default settings.

## Quick Test

### 1. Check Your System

```bash
python -m modelium.cli check --verbose
```

Should show all green checkmarks:
```
✅ Python 3.11.x
✅ PyTorch: 2.x
✅ FastAPI: 0.104.x
✅ CUDA available: 4 GPU(s)
✅ vLLM: 0.x.x
✅ Config found: modelium.yaml
```

### 2. Start the Server

```bash
python -m modelium.cli serve --config modelium.yaml
```

You should see:
```
🧠 Starting Modelium Server...
📝 Loading configuration...
   Organization: my-company
   GPUs: 4
🧠 Loading Modelium Brain...
   ✅ Brain loaded successfully
🔧 Initializing services...
🚀 Starting background services...
✅ Server ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Check Status

```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status | jq .

# List models (initially empty)
curl http://localhost:8000/models | jq .
```

### 4. Drop a Model

Copy any model to the watched directory:

```bash
cp your_model.pt /models/incoming/
```

Watch the logs:
```bash
# In another terminal
tail -f modelium.log

# You'll see:
📁 Discovered model: your_model
🔍 Analyzing your_model...
   Framework: pytorch
   ✅ Analysis complete
📋 Planning deployment for your_model...
   Plan: vllm on GPU 2
🔼 Loading your_model to GPU 2...
   ✅ your_model loaded
```

### 5. Make Requests

```bash
# Check model status
curl http://localhost:8000/models | jq '.models[] | {name, status, runtime, gpu}'

# Run inference
curl -X POST http://localhost:8000/predict/your_model \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "organizationId": "test-org",
    "max_tokens": 50
  }'
```

## What's Next?

**Deployment**:
- [Docker Guide](../DOCKER.md) - Docker Compose for local/testing
- [Deployment Guide](../DEPLOYMENT.md) - Kubernetes and Helm for production

**Understanding the System**:
- [Architecture](architecture.md) - How Modelium works
- [The Brain](brain.md) - AI orchestration explained
- [Usage Guide](usage.md) - Complete CLI and API reference

**Advanced**:
- [Configuration Examples](../configs/README.md) - Multi-instance, workload separation
- [Testing Guide](../TESTING_TOMORROW.md) - End-to-end testing

## Common Issues

**Issue**: `modelium: command not found`  
**Fix**: Use `python -m modelium.cli serve` instead

**Issue**: Models not detected  
**Fix**: 
1. Check `watch_directories` in `modelium.yaml`
2. Verify directory exists: `ls -la /models/incoming/`
3. Check logs: `tail -f modelium.log`

**Issue**: GPU not found  
**Fix**: Set `gpu.enabled: false` for CPU-only mode (slower)

**Issue**: vLLM fails to load  
**Fix**: Lower GPU memory: `vllm.gpu_memory_utilization: 0.7` in config

**Issue**: Brain fails to load  
**Expected**: System falls back to rule-based mode automatically

**Issue**: Port already in use  
**Fix**: `pkill -f "modelium.cli serve"` or change port

For detailed troubleshooting, see [Testing Guide](../TESTING_TOMORROW.md).

