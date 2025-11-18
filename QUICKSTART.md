# Modelium Quick Start 🚀

Get Modelium running in **5 minutes**.

## TL;DR

```bash
# 1. Start vLLM with a model
docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest --model gpt2

# 2. Install Modelium
git clone https://github.com/sp1derz/modelium.git && cd modelium
python3 -m venv venv && source venv/bin/activate
pip install -e ".[all]"

# 3. Configure & Start
python -m modelium.cli init
python -m modelium.cli serve

# 4. Drop a model & test
mkdir -p models/incoming
cd models/incoming && git clone https://huggingface.co/gpt2 gpt2-model && cd ../..
curl http://localhost:8000/models | jq
```

## Step-by-Step

### 1. Start a Runtime (Choose One)

**vLLM (for LLMs - easiest to start):**
```bash
docker run --gpus all -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model gpt2 \
  --dtype auto
```

**Triton (for production):**
```bash
mkdir triton-models
docker run --gpus all -p 8003:8000 \
  -v $(pwd)/triton-models:/models \
  nvcr.io/nvidia/tritonserver:latest \
  tritonserver --model-repository=/models
```

**Ray Serve (for Python models):**
```bash
pip install ray[serve]
ray start --head
```

### 2. Install Modelium

```bash
# Clone repo
git clone https://github.com/sp1derz/modelium.git
cd modelium

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install --upgrade pip
pip install -e ".[all]"

# Verify
python -m modelium.cli check
```

### 3. Configure

```bash
# Create config
python -m modelium.cli init

# Edit modelium.yaml (optional - defaults work for local testing)
# Just make sure the enabled runtime matches what you started in Step 1
```

Example `modelium.yaml`:
```yaml
vllm:
  enabled: true  # Set to true if you started vLLM
  endpoint: "http://localhost:8001"

triton:
  enabled: false  # Set to true if you started Triton
  endpoint: "http://localhost:8003"

ray_serve:
  enabled: false  # Set to true if you started Ray
  endpoint: "http://localhost:8002"
```

### 4. Start Modelium

```bash
python -m modelium.cli serve
```

You should see:
```
🧠 Starting Modelium Server...
🔧 Connecting to runtimes...
   ✅ vLLM connected
✅ Server ready!
   API: http://0.0.0.0:8000
```

### 5. Deploy a Model

```bash
# Create directory
mkdir -p models/incoming

# Download model from HuggingFace
cd models/incoming
git clone https://huggingface.co/gpt2 gpt2-model
cd ../..
```

Watch Modelium logs - you'll see:
```
📋 Analyzing gpt2-model...
   Detected: GPT2LMHeadModel
   Runtime: vllm
   Size: 0.55GB
```

### 6. Test Inference

```bash
# List models
curl http://localhost:8000/models | jq

# Run inference
curl -X POST http://localhost:8000/predict/gpt2-model \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "organizationId": "my-org",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq
```

Expected response:
```json
{
  "id": "cmpl-...",
  "choices": [{
    "text": " in a land far away, there lived a...",
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
}
```

## Troubleshooting

**"No runtimes available"**
- Make sure you started a runtime in Step 1
- Check it's healthy: `curl http://localhost:8001/health` (vLLM)

**"Model not found"**
- Model directory must contain `config.json`
- Check logs for analysis errors
- Verify model is in watched directory

**GPU not detected**
- Install CUDA Toolkit 12.1+
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

## What's Next?

- **[Full Documentation](docs/getting-started.md)** - Complete guide
- **[Examples](examples/)** - Working code samples
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment

## Common Commands

```bash
# Check system
python -m modelium.cli check --verbose

# Start server
python -m modelium.cli serve

# Custom config
python -m modelium.cli serve --config my-config.yaml

# API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/status | jq
curl http://localhost:8000/models | jq
```

## Architecture in 30 Seconds

```
Your Models → Modelium → Routes to → vLLM / Triton / Ray
              (watches)  (analyzes)   (best runtime)
                         (unified API)
```

1. **You**: Start runtime(s) with models
2. **Modelium**: Watches folder, analyzes models, routes requests
3. **You**: Access all models via single API

That's it! 🎉

For advanced features (AI Brain, multi-GPU, Kubernetes), see [docs/](docs/).

