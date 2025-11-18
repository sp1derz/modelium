# Getting Started with Modelium

Complete guide to install and run Modelium in any environment.

## Prerequisites

### Required
- **Python 3.11+**
- **8GB+ RAM**
- **50GB+ disk space**

### Recommended
- **NVIDIA GPU** with CUDA 12.1+ (for GPU inference)
- **Ubuntu 20.04+** or similar Linux (for production)
- **16GB+ GPU memory** (for larger models)

### Runtime Requirements

Modelium connects to external runtime services. **You need at least one**:

- **vLLM** (for LLMs): https://github.com/vllm-project/vllm
- **Triton** (for all models): https://github.com/triton-inference-server/server
- **Ray Serve** (for Python models): https://docs.ray.io/en/latest/serve/

## Installation Options

### Option 1: Python Virtual Environment (Recommended for Development)

```bash
# 1. Clone repository
git clone https://github.com/sp1derz/modelium.git
cd modelium

# 2. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Modelium with all dependencies
pip install --upgrade pip
pip install -e ".[all]"

# 4. Verify installation
python -m modelium.cli check
```

### Option 2: Docker (Recommended for Production)

```bash
# 1. Clone repository
git clone https://github.com/sp1derz/modelium.git
cd modelium

# 2. Build Docker image (one-time, 10-15 minutes)
docker build -t modelium:latest .

# 3. Run with Docker Compose
docker-compose up -d

# 4. Check status
curl http://localhost:8000/health
```

See [DOCKER.md](../DOCKER.md) for GPU setup and advanced Docker configuration.

### Option 3: Kubernetes (Production at Scale)

See [DEPLOYMENT.md](../DEPLOYMENT.md) for complete Kubernetes deployment guide.

## Setting Up Runtimes

Modelium requires at least one runtime service running. Here's how to start each:

### vLLM (For LLMs)

**Using Docker (easiest):**

```bash
# Start vLLM with GPT-2 (for testing)
docker run --gpus all -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model gpt2 \
  --dtype auto

# For larger models (e.g., Llama-2-7B)
docker run --gpus all -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-hf \
  --dtype float16 \
  --max-model-len 2048
```

**From source:**

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model gpt2 \
  --host 0.0.0.0 \
  --port 8001
```

**Verify vLLM is running:**

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

### Triton (For All Model Types)

**Using Docker:**

```bash
# Create model repository
mkdir -p triton-models

# Start Triton
docker run --gpus all -p 8003:8000 -p 8004:8001 -p 8005:8002 \
  -v $(pwd)/triton-models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

**Verify Triton is running:**

```bash
curl http://localhost:8003/v2/health/ready
curl http://localhost:8003/v2/models
```

### Ray Serve (For Python Models)

**Using pip:**

```bash
pip install ray[serve]

# Start Ray cluster
ray start --head --port=6379

# Deploy a model (example)
python your_ray_deployment.py
```

**Using Docker:**

```bash
docker run --gpus all -p 8002:8000 -p 8265:8265 \
  rayproject/ray:latest \
  ray start --head --dashboard-host=0.0.0.0
```

**Verify Ray is running:**

```bash
curl http://localhost:8002/-/healthz
```

## Configuring Modelium

### Initialize Configuration

```bash
# Create default config file
python -m modelium.cli init

# This creates modelium.yaml
```

### Edit Runtime Endpoints

Edit `modelium.yaml` to match your runtime ports:

```yaml
# Runtime Configuration
vllm:
  enabled: true
  endpoint: "http://localhost:8001"  # Update if vLLM is on different host/port
  health_check_path: "/health"
  timeout: 300

triton:
  enabled: false  # Set to true if you're running Triton
  endpoint: "http://localhost:8003"
  health_check_path: "/v2/health/ready"
  timeout: 300

ray_serve:
  enabled: false  # Set to true if you're running Ray Serve
  endpoint: "http://localhost:8002"
  health_check_path: "/-/healthz"
  timeout: 300

# Model Discovery
orchestration:
  model_discovery:
    watch_directories:
      - "/models/incoming"  # Where to watch for new models
    scan_interval_seconds: 5
    supported_formats:
      - "safetensors"
      - "bin"
      - "pt"
      - "pth"

# Organization (Multi-Tenancy)
organization:
  id: "my-company"
  name: "My Company"

# Modelium Brain (Optional)
modelium_brain:
  enabled: true
  model_name: "modelium/brain-v1"
  fallback_to_rules: true  # Use rule-based if brain unavailable
```

### Create Model Directory

```bash
# Create directory for incoming models
mkdir -p models/incoming

# Modelium will watch this directory for new models
```

## Starting Modelium

### CLI Mode

```bash
# Activate your virtual environment
source venv/bin/activate

# Start Modelium server
python -m modelium.cli serve

# Or with custom config
python -m modelium.cli serve --config my-config.yaml --host 0.0.0.0 --port 8000
```

You should see:

```
🧠 Starting Modelium Server...
Config: modelium.yaml
Host: 0.0.0.0
Port: 8000

📝 Loading configuration...
   Organization: my-company
   GPUs: auto-detect

🧠 Loading Modelium Brain...
   ✅ Brain loaded successfully

🚀 Server starting...
   API: http://0.0.0.0:8000
   Status: http://0.0.0.0:8000/status
   Metrics: http://0.0.0.0:9090/metrics

🔧 Connecting to runtimes...
   ✅ vLLM connected

🚀 Starting background services...
✅ Server ready!
```

### Docker Mode

```bash
docker-compose up -d

# View logs
docker-compose logs -f modelium-server
```

### Kubernetes Mode

```bash
kubectl apply -k infra/k8s/

# Check status
kubectl get pods -n modelium
kubectl logs -f deployment/modelium-server -n modelium
```

## First Deployment

### 1. Prepare a Model

Download a HuggingFace model:

```bash
cd models/incoming

# Option A: Clone from HuggingFace
git clone https://huggingface.co/gpt2 gpt2-model

# Option B: Symlink your local model
ln -s /path/to/your/model my-model

# Option C: Download with Python
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2', cache_dir='./gpt2-model')"
```

**Important**: Model directory must contain `config.json` for Modelium to analyze it.

### 2. Watch Modelium Detect It

In the Modelium logs, you'll see:

```
📋 Analyzing gpt2-model...
   Detected: GPT2LMHeadModel
   Runtime: vllm
   Size: 0.55GB
🔼 Checking if gpt2-model is loaded in vllm...
   ✅ gpt2-model available in vLLM
```

### 3. Test Inference

```bash
# Check available models
curl http://localhost:8000/models | jq

# Run inference
curl -X POST http://localhost:8000/predict/gpt2-model \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "organizationId": "my-company",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq
```

Expected response:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1699...,
  "model": "gpt2-model",
  "choices": [{
    "text": " in a land far away...",
    "index": 0,
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
}
```

## Verification & Testing

### Check System Status

```bash
# Health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status | jq

# List models
curl http://localhost:8000/models | jq
```

### Run Diagnostics

```bash
python -m modelium.cli check --verbose
```

This checks:
- Python version
- Dependencies
- GPU availability
- Runtime connectivity
- Configuration validity
- Disk space

## Common Issues

### "No runtimes available"

**Problem**: Modelium can't connect to any runtime.

**Solution**:
1. Make sure at least one runtime is running
2. Check endpoints in `modelium.yaml` match actual ports
3. Test runtime health directly:
   ```bash
   curl http://localhost:8001/health  # vLLM
   curl http://localhost:8003/v2/health/ready  # Triton
   curl http://localhost:8002/-/healthz  # Ray
   ```

### "Model not found"

**Problem**: Model not showing in `/models` endpoint.

**Solution**:
1. Check model directory contains `config.json`
2. Model must be in watched directory (`/models/incoming`)
3. Check Modelium logs for analysis errors
4. Verify model is loaded in runtime:
   ```bash
   curl http://localhost:8001/v1/models  # vLLM
   ```

### "Model not loaded (status: error)"

**Problem**: Model detected but failed to load in runtime.

**Solution**:
1. Check runtime logs for errors
2. Verify runtime has enough GPU memory
3. Check model format is supported by runtime
4. For vLLM: Model must be started with `--model` flag matching model name

### GPU not detected

**Problem**: `gpu_count: 0` in `/status`.

**Solution**:
1. Install CUDA toolkit
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

## Next Steps

- **[Usage Guide](usage.md)** - Learn about all features and APIs
- **[Architecture](architecture.md)** - Understand how Modelium works
- **[Examples](../examples/)** - See practical deployment examples
- **[DEPLOYMENT.md](../DEPLOYMENT.md)** - Production deployment guides

## Environment-Specific Guides

### AWS EC2

```bash
# Launch g6e.12xlarge (4x L40S GPUs)
# Install CUDA
sudo yum install -y cuda-toolkit-12-1

# Install Docker with GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum install -y nvidia-docker2
sudo systemctl restart docker

# Continue with standard installation...
```

### Google Cloud (GCP)

```bash
# Use Deep Learning VM with CUDA pre-installed
# Continue with standard installation...
```

### Azure

```bash
# Use NC-series VMs with NVIDIA GPUs
# Install CUDA: https://docs.microsoft.com/azure/virtual-machines/linux/n-series-driver-setup
# Continue with standard installation...
```

## Getting Help

- **Documentation**: Full docs in `docs/` directory
- **Examples**: See `examples/` for working code
- **Issues**: https://github.com/sp1derz/modelium/issues
- **Discussions**: https://github.com/sp1derz/modelium/discussions
