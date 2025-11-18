# Modelium Testing Guide

Complete testing instructions for both automated and manual testing.

## Quick Start: Automated Testing

### Option 1: Test with Virtual Environment

```bash
# Make script executable
chmod +x test_modelium_venv.sh

# Run tests (takes ~5-10 minutes)
./test_modelium_venv.sh
```

**What it tests:**
- ✅ Environment setup
- ✅ Installation
- ✅ Configuration
- ✅ Server startup
- ✅ Model detection
- ✅ Model loading (GPT-2)
- ✅ Inference
- ✅ Metrics
- ✅ Load testing
- ✅ Intelligent orchestration

### Option 2: Test with Docker

```bash
# Make script executable
chmod +x test_modelium_docker.sh

# Run tests (takes ~15-20 minutes first time for build)
./test_modelium_docker.sh
```

**What it tests:**
- ✅ Docker environment
- ✅ Image build
- ✅ Container startup
- ✅ Health checks
- ✅ Model detection & loading
- ✅ Inference
- ✅ Metrics
- ✅ Load testing
- ✅ Resource usage
- ✅ Container logs

---

## Manual Testing: Step-by-Step

### Prerequisites

```bash
# For venv testing
- Python 3.11+
- 8GB+ RAM
- 50GB+ disk space
- (Optional) NVIDIA GPU with CUDA

# For Docker testing
- Docker 20.10+
- docker-compose 1.29+
- 16GB+ RAM
- 50GB+ disk space
```

---

## Manual Test 1: Virtual Environment

### Step 1: Setup

```bash
# Clone repo
git clone https://github.com/sp1derz/modelium
cd modelium

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install Modelium
pip install -e ".[all]"

# Install runtime (choose based on OS)
pip install vllm        # Linux+CUDA only
# OR
pip install ray[serve]  # Any OS
```

### Step 2: Configure

```bash
# Copy config
cp modelium.yaml.example modelium.yaml

# Edit config
nano modelium.yaml

# Set:
# vllm.enabled: true  (if Linux+CUDA)
# ray_serve.enabled: true  (if Mac or no CUDA)
```

### Step 3: Start Server

```bash
# Start in foreground (to see logs)
python -m modelium.cli serve

# OR start in background
nohup python -m modelium.cli serve > modelium.log 2>&1 &

# Check it started
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

### Step 4: Check Status

```bash
curl http://localhost:8000/status | jq
```

**Expected output:**
```json
{
  "status": "running",
  "organization": "my-company",
  "gpu_count": 0,  // or N if GPUs available
  "models_loaded": 0,
  "models_discovered": 0
}
```

### Step 5: Drop a Model

```bash
# Create directory
mkdir -p models/incoming

# Download GPT-2 (small, ~500MB)
git clone https://huggingface.co/gpt2 models/incoming/gpt2

# OR use transformers
python3 << 'EOF'
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("gpt2", cache_dir="./models/incoming/gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./models/incoming/gpt2")
EOF
```

### Step 6: Watch Detection

```bash
# Watch logs (if running in background)
tail -f modelium.log

# OR watch models endpoint
watch -n 5 'curl -s http://localhost:8000/models | jq'
```

**Expected logs:**
```
📋 New model discovered: gpt2
   Analyzing model...
   Architecture: GPT2LMHeadModel
   Type: causal_lm
   Size: 0.55GB
   🎯 Brain decision: vllm
   Loading model...
   Spawning vLLM on port 8100, GPU 0
   Waiting for vLLM to start...
   ✅ gpt2 loaded successfully!
```

### Step 7: Check Model Status

```bash
curl http://localhost:8000/models | jq
```

**Expected output:**
```json
{
  "models": [
    {
      "name": "gpt2",
      "status": "loaded",
      "runtime": "vllm",
      "gpu": 0,
      "qps": 0.0,
      "idle_seconds": 0
    }
  ]
}
```

### Step 8: Test Inference

```bash
curl -X POST http://localhost:8000/predict/gpt2 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "organizationId": "test-company",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq
```

**Expected output:**
```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1699...,
  "model": "gpt2",
  "choices": [{
    "text": " in a land far away, there lived a...",
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

### Step 9: Test Metrics

```bash
curl http://localhost:9090/metrics | grep modelium
```

**Expected metrics:**
```
modelium_requests_total{model="gpt2",runtime="vllm",status="success"} 1.0
modelium_latency_seconds{model="gpt2",runtime="vllm",percentile="50"} 0.123
modelium_model_idle_seconds{model="gpt2",runtime="vllm"} 0.0
```

### Step 10: Test Intelligent Orchestration

```bash
# Make several requests
for i in {1..5}; do
  curl -s -X POST http://localhost:8000/predict/gpt2 \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"Test $i\", \"max_tokens\": 10}" | jq '.choices[0].text'
  sleep 1
done

# Check QPS is tracked
curl http://localhost:8000/models | jq '.models[] | {name, qps, idle_seconds}'
```

**Expected:**
```json
{
  "name": "gpt2",
  "qps": 0.5,  // Non-zero after requests
  "idle_seconds": 0
}
```

### Step 11: Test Idle Unloading

```bash
# Wait 6+ minutes without requests
sleep 360

# Check status
curl http://localhost:8000/models | jq '.models[] | {name, status, idle_seconds}'
```

**Expected (if GPU has space):**
```json
{
  "name": "gpt2",
  "status": "loaded",  // Still loaded (GPU has space)
  "idle_seconds": 360
}
```

**Expected (if GPU under pressure):**
```json
{
  "name": "gpt2",
  "status": "unloaded",  // Unloaded to free GPU
  "idle_seconds": 360
}
```

---

## Manual Test 2: Docker

### Step 1: Build Image

```bash
cd modelium

# Build (first time takes 10-15 minutes)
docker build -t modelium:latest .

# Check image
docker images | grep modelium
```

### Step 2: Start Container

```bash
# Using docker-compose
docker-compose up -d

# OR using docker run
docker run -d --name modelium-server \
  --gpus all \
  -p 8000:8000 -p 9090:9090 \
  -v $(pwd)/models:/models \
  modelium:latest

# Check it started
docker ps | grep modelium
```

### Step 3: Check Logs

```bash
# Follow logs
docker-compose logs -f

# OR with docker
docker logs -f modelium-server
```

### Step 4: Test Endpoints

```bash
# Health
curl http://localhost:8000/health

# Status
curl http://localhost:8000/status | jq

# Models
curl http://localhost:8000/models | jq
```

### Step 5: Drop Model

```bash
# Create directory (on host)
mkdir -p models/incoming

# Download GPT-2
git clone https://huggingface.co/gpt2 models/incoming/gpt2

# Container will detect it automatically
docker-compose logs -f | grep "discovered"
```

### Step 6: Test Inference

```bash
# Wait for model to load
sleep 60

# Test inference
curl -X POST http://localhost:8000/predict/gpt2 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello world",
    "max_tokens": 20
  }' | jq
```

### Step 7: Check Container Resources

```bash
# Stats
docker stats --no-stream modelium-server

# Disk usage
docker system df

# Container size
docker inspect modelium-server | jq '.[0].SizeRootFs'
```

### Step 8: Enter Container

```bash
# Bash shell
docker exec -it modelium-server bash

# Inside container:
ls /models/incoming/
cat /app/modelium.yaml
ps aux | grep vllm
exit
```

### Step 9: Test Updates

```bash
# Make code changes, then rebuild
docker build -t modelium:latest .

# Restart container
docker-compose down
docker-compose up -d

# Verify changes
docker-compose logs
```

---

## Troubleshooting

### Issue: "No runtimes available"

**Solution:**
```bash
# Check what's enabled
grep "enabled:" modelium.yaml

# Install missing runtime
pip install vllm        # Linux+CUDA
pip install ray[serve]  # Any OS
```

### Issue: "Model not detected"

**Solution:**
```bash
# Check directory
ls -la models/incoming/

# Check model has config.json
ls models/incoming/gpt2/config.json

# Check watcher is running
curl http://localhost:8000/status | jq '.orchestration_enabled'

# Check logs
grep "watcher" modelium.log
```

### Issue: "Model status: error"

**Solution:**
```bash
# Check runtime logs
tail -100 modelium.log | grep -A 10 "error"

# Check runtime is installed
python -c "import vllm; print('vLLM OK')"
python -c "import ray; print('Ray OK')"

# Check GPU
nvidia-smi  # For CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "Inference timeout"

**Solution:**
```bash
# Check model is loaded
curl http://localhost:8000/models | jq '.models[] | {name, status}'

# Check runtime is responding
# For vLLM:
curl http://localhost:8100/health

# Check resources
top  # CPU/RAM
nvidia-smi  # GPU
df -h  # Disk
```

---

## Performance Testing

### Load Test with `hey`

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Run load test
hey -n 100 -c 5 \
  -m POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","max_tokens":10}' \
  http://localhost:8000/predict/gpt2

# Check results:
# - Requests/sec
# - Latency (p50, p95, p99)
# - Success rate
```

### Monitor Metrics

```bash
# Watch metrics
watch -n 1 'curl -s http://localhost:9090/metrics | grep modelium'

# Export to file
curl http://localhost:9090/metrics > metrics_$(date +%s).txt

# Analyze
grep "modelium_requests_total" metrics_*.txt
grep "modelium_latency" metrics_*.txt
```

---

## Success Criteria

✅ **Basic Functionality:**
- Health endpoint returns "healthy"
- Status endpoint returns system info
- Server starts without errors

✅ **Model Management:**
- Models detected within 30s
- Models load within 120s
- Models show status "loaded"

✅ **Inference:**
- Inference returns valid responses
- Latency < 5 seconds for GPT-2
- Success rate > 95%

✅ **Metrics:**
- Prometheus endpoint accessible
- QPS tracked correctly
- Idle time updated
- No gaps in metrics

✅ **Intelligent Orchestration:**
- Models with QPS > 0.01 stay loaded
- Idle models unload after threshold
- Always-loaded models never unload
- GPU pressure considered

---

## Next Steps

After testing:

1. **Production Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
2. **Kubernetes**: See [infra/k8s/README.md](infra/k8s/README.md)
3. **Monitoring**: Set up Grafana dashboards
4. **Scaling**: Add more GPUs, test multi-model

---

## Getting Help

- **Logs**: Check `modelium.log` or `docker-compose logs`
- **Issues**: https://github.com/sp1derz/modelium/issues
- **Docs**: See `docs/` directory
- **Examples**: See `examples/` directory

