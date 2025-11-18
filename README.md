# Modelium 🧠

**AI-Orchestrated Multi-Runtime Model Serving Platform**

Modelium is an intelligent orchestrator that connects your ML models to the best inference runtimes (vLLM, Triton, Ray Serve). Drop a model in a folder, and Modelium automatically analyzes it, chooses the optimal runtime, and routes requests - all while maximizing GPU utilization.

## ✨ What Makes Modelium Different

- **🤖 AI Brain**: Optional fine-tuned LLM makes intelligent runtime selection and orchestration decisions
- **🔌 Runtime Agnostic**: Connects to vLLM, Triton, and Ray Serve via HTTP - no reimplementation
- **🔄 Auto-Discovery**: Drop HuggingFace models in a folder → automatic analysis → smart routing
- **📊 Smart Routing**: Routes inference requests to the correct runtime based on model type
- **🎯 Unified API**: Single API endpoint for all your models, regardless of runtime
- **🔒 Enterprise-Ready**: Multi-tenant, usage tracking, rate limiting built-in

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Modelium Server                        │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Analyzer  │→ │ Modelium     │→ │  Request Router    │  │
│  │  (config.  │  │  Brain       │  │  (runtime-aware)   │  │
│  │   json)    │  │  (optional)  │  │                    │  │
│  └────────────┘  └──────────────┘  └────────────────────┘  │
│         ↑               ↑                     ↓              │
│         │               │        ┌────────────────────────┐ │
│  ┌──────────────────────┴────────┤   Runtime Connectors   │ │
│  │  Model Discovery (watches    │   (HTTP clients)      │ │
│  │  /models/incoming/)           └────────────────────────┘ │
│  └────────────────────────────────────┬─────┬───────┬──────┘
└───────────────────────────────────────┼─────┼───────┼───────┘
                                        ↓     ↓       ↓
         ┌──────────────────────────────┴─────┴───────┴─────┐
         │        External Runtime Services (User-Run)     │
         ├─────────────────┬───────────────┬────────────────┤
         │   vLLM Server   │ Triton Server │ Ray Serve      │
         │   (LLMs)        │ (All Models)  │ (General)      │
         │   Port 8001     │ Port 8003     │ Port 8002      │
         └─────────────────┴───────────────┴────────────────┘
```

**How It Works:**
1. **You start** your preferred runtimes (vLLM/Triton/Ray) with your preferred configuration
2. **Modelium connects** to them via HTTP APIs
3. **Drop a model** in `/models/incoming/`
4. **Modelium analyzes** it (reads `config.json`, detects architecture)
5. **Brain decides** which runtime is best (or uses rules)
6. **Requests route** automatically to the correct runtime

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **CUDA 12.1+** (for GPU)
- **At least one runtime** (vLLM, Triton, or Ray Serve)

### Step 1: Start Your Runtime(s)

Choose one or more runtimes:

```bash
# Option A: vLLM (Best for LLMs like GPT, Llama, Mistral)
docker run --gpus all -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model gpt2 \
  --dtype auto

# Option B: Triton (NVIDIA's high-performance server)
docker run --gpus all -p 8003:8000 \
  nvcr.io/nvidia/tritonserver:latest \
  tritonserver --model-repository=/models

# Option C: Ray Serve (For general Python models)
docker run --gpus all -p 8002:8000 \
  rayproject/ray:latest \
  ray start --head

# You can run all three simultaneously on different ports!
```

### Step 2: Install Modelium

```bash
# Clone
git clone https://github.com/sp1derz/modelium.git
cd modelium

# Install
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"

# Check setup
python -m modelium.cli check
```

### Step 3: Configure Runtime Endpoints

```bash
# Initialize config
python -m modelium.cli init

# Edit modelium.yaml to match your runtime endpoints
# Default ports: vLLM=8001, Ray=8002, Triton=8003
```

Example `modelium.yaml`:

```yaml
vllm:
  enabled: true
  endpoint: "http://localhost:8001"  # Where vLLM is running

triton:
  enabled: false  # Set to true if you're running Triton
  endpoint: "http://localhost:8003"

ray_serve:
  enabled: false  # Set to true if you're running Ray Serve
  endpoint: "http://localhost:8002"
```

### Step 4: Start Modelium

```bash
python -m modelium.cli serve
```

You'll see:

```
🧠 Starting Modelium Server...
📝 Loading configuration...
🔧 Connecting to runtimes...
   ✅ vLLM connected
🚀 Server starting...
   API: http://0.0.0.0:8000
```

### Step 5: Drop a Model & Test

```bash
# Create directory for incoming models
mkdir -p models/incoming

# Download a HuggingFace model (or symlink your local models)
cd models/incoming
git clone https://huggingface.co/gpt2 gpt2-model
cd ../..

# Modelium will auto-detect it! Watch the logs:
# 📋 Analyzing gpt2-model...
#    Detected: GPT2LMHeadModel
#    Runtime: vllm

# Test inference
curl -X POST http://localhost:8000/predict/gpt2-model \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world! This is",
    "organizationId": "my-org",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq
```

## 📖 Documentation

- **[Getting Started](docs/getting-started.md)** - Complete installation guide
- **[Architecture](docs/architecture.md)** - System design and components
- **[Usage](docs/usage.md)** - API reference and advanced features
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Kubernetes, Helm, cloud deployment
- **[DOCKER.md](DOCKER.md)** - Docker setup and GPU configuration

## 💡 Examples

See `examples/` directory:

- **`vllm_deployment.py`** - Deploy LLMs with vLLM
- **`triton_deployment.py`** - Deploy with Triton
- **`ray_deployment.py`** - Deploy with Ray Serve
- **`multi_runtime.py`** - Use multiple runtimes together

## 🎯 Use Cases

### 1. **Simplified LLM Serving**
Run vLLM once with your model, let Modelium handle discovery and routing:

```bash
# Start vLLM with your model
docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-hf

# Drop more models in /models/incoming, Modelium detects them
# All accessible via single API: http://localhost:8000/predict/{model}
```

### 2. **Multi-Model Serving**
Run multiple models across different runtimes:

```yaml
# Modelium automatically routes:
# - GPT models → vLLM
# - Vision models → Ray Serve  
# - Optimized models → Triton
```

### 3. **Enterprise Deployment**
Multi-tenant with usage tracking:

```bash
curl -X POST http://localhost:8000/predict/gpt2 \
  -d '{"organizationId": "acme-corp", "prompt": "..."}'

# Automatic usage tracking per organization
# Rate limiting, billing metrics built-in
```

## 🔧 Configuration

Modelium uses `modelium.yaml` for all configuration:

```yaml
# Runtime endpoints (required)
vllm:
  enabled: true
  endpoint: "http://localhost:8001"
  
triton:
  enabled: false
  endpoint: "http://localhost:8003"

ray_serve:
  enabled: false
  endpoint: "http://localhost:8002"

# Model discovery
orchestration:
  model_discovery:
    watch_directories: ["/models/incoming"]
    scan_interval_seconds: 5

# AI Brain (optional - for advanced orchestration)
modelium_brain:
  enabled: true
  model_name: "modelium/brain-v1"  # HuggingFace model
  fallback_to_rules: true  # Use rules if brain unavailable

# Multi-tenancy
organization:
  id: "my-company"
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
```

## 🧠 The Modelium Brain (Optional)

The Modelium Brain is a fine-tuned Qwen-2.5-1.5B model that makes intelligent decisions:

- **Runtime Selection**: Chooses the best runtime for each model
- **GPU Allocation**: Optimizes which GPUs to use
- **Load/Unload Decisions**: Dynamic model swapping for max utilization

**Training your own brain:**

```bash
cd modelium/modelium_llm/training
python train.py --dataset your_dataset.json --output modelium-brain-custom
```

See [Modelium Brain on HuggingFace](https://huggingface.co/modelium) (coming soon).

## 🐳 Docker & Kubernetes

### Docker Compose (Local Development)

```bash
# Start all services
docker-compose up -d

# Modelium + vLLM + Prometheus + Grafana
```

### Kubernetes (Production)

```bash
# Using Kustomize
kubectl apply -k infra/k8s/

# Using Helm
helm install modelium ./infra/helm/modelium \
  --set image.tag=latest \
  --set vllm.endpoint=http://vllm-service:8000
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guide.

## 📊 What Works Now

✅ **Core Features:**
- HuggingFace model analysis (`config.json` parsing)
- Runtime connectors (vLLM, Triton, Ray Serve via HTTP)
- Auto-discovery and model watching
- Request routing to correct runtime
- Health checks and startup validation
- Multi-tenant tracking (`organizationId`)

✅ **Deployment:**
- Docker (single container)
- Docker Compose (with GPU support)
- Kubernetes manifests
- Helm charts
- CI/CD (GitHub Actions)

## 🚧 Coming Soon

- **Modelium Brain Training**: Fine-tuned model on HuggingFace
- **GPUDirect Storage**: Ultra-fast model loading
- **Advanced Orchestration**: Dynamic load/unload based on demand
- **Prometheus Metrics**: Detailed monitoring
- **More Runtimes**: TensorRT, OpenVINO, ONNX Runtime
- **Web UI**: Dashboard for monitoring and management

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority areas:**
- Testing on different GPUs (A100, H100, L40S, etc.)
- Runtime-specific optimizations
- Model format support (GGUF, AWQ, GPTQ)
- Documentation improvements

## 📝 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## 🙋 FAQ

**Q: Do I need to use the AI Brain?**  
A: No! Set `modelium_brain.enabled: false` to use rule-based runtime selection.

**Q: Can I run Modelium without Docker?**  
A: Yes! Use the Python CLI (see Quick Start Option 3).

**Q: Which runtime should I use?**  
A: 
- **vLLM**: Best for LLMs (GPT, Llama, Mistral, etc.)
- **Triton**: Best for production, optimized models, multi-framework
- **Ray Serve**: Best for custom Python models, easy deployment

**Q: Does Modelium replace vLLM/Triton/Ray?**  
A: No! Modelium **connects to** them. You still run the runtimes; Modelium orchestrates and routes.

**Q: Can I use multiple runtimes simultaneously?**  
A: Yes! Modelium will route each model to the most appropriate runtime.

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/sp1derz/modelium/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sp1derz/modelium/discussions)

---

**Built with ❤️ for the ML community. Star ⭐ if you find it useful!**
