# Testing Guide for Tomorrow

## Pre-Test Setup (On AWS g6e Instance)

### 1. Pull Latest Code
```bash
cd /home/demo/modelium
git pull origin main
```

### 2. Install/Update Dependencies
```bash
source venv/bin/activate
pip install -e .[all]
pip install vllm  # Separate install
```

### 3. Run System Check
```bash
python -m modelium.cli check --verbose
```

**Expected Output:**
- ✅ Python 3.11.x
- ✅ PyTorch with CUDA
- ✅ 4 GPUs detected (L40S)
- ✅ FastAPI, Uvicorn
- ✅ vLLM installed
- ✅ Config file found
- ✅ Watch directories exist

### 4. Update Config
```bash
nano modelium.yaml
```

Key settings:
```yaml
# Set this to use real Qwen model
modelium_brain:
  enabled: true
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  device: "cuda:0"
  fallback_to_rules: true

orchestration:
  enabled: true
  model_discovery:
    watch_directories:
      - "/home/demo/models/incoming"  # Update path

gpu:
  count: 4  # You have 4x L40S
```

### 5. Create Watch Directory
```bash
mkdir -p /home/demo/models/incoming
```

## Test Scenario 1: Basic Server Startup

### Start Server
```bash
cd /home/demo/modelium
source venv/bin/activate

# Stop old process if running
pkill -f "modelium.cli serve"

# Start new
nohup python -m modelium.cli serve --config modelium.yaml > modelium.log 2>&1 &

# Wait 30 seconds for brain to load
sleep 30
```

### Check Logs
```bash
tail -50 modelium.log
```

**Expected:**
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
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Status
curl http://localhost:8000/status | jq .

# List models (should be empty)
curl http://localhost:8000/models | jq .
```

**Expected Status:**
```json
{
  "status": "running",
  "organization": "my-company",
  "gpu_count": 4,
  "models_loaded": 0,
  "models_discovered": 0
}
```

## Test Scenario 2: Model Discovery & Loading

### Option A: Test with Small Model (Recommended First)

```bash
# Create a tiny test model
cd /home/demo/models
python << EOF
import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

model = TinyModel()
torch.save(model.state_dict(), 'tiny_test_model.pt')
print("Created tiny_test_model.pt")
EOF

# Drop it in watched directory
cp tiny_test_model.pt /home/demo/models/incoming/
```

### Watch What Happens
```bash
# Watch logs
tail -f /home/demo/modelium/modelium.log

# In another terminal, check status every 5s
watch -n 5 'curl -s http://localhost:8000/models | jq .'
```

**Expected Log Output:**
```
📁 Discovered model: tiny_test_model at /home/demo/models/incoming/tiny_test_model.pt
🔍 Analyzing tiny_test_model...
   Framework: pytorch
   ✅ Analysis complete
📋 Planning deployment for tiny_test_model...
   Plan: ray_serve on GPU 0
🔼 Loading tiny_test_model to GPU 0...
   ✅ tiny_test_model loaded
```

### Option B: Test with Real HuggingFace Model

```bash
# Download a small model
python << EOF
from transformers import AutoModel, AutoTokenizer
model_name = "microsoft/phi-2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained("/home/demo/models/incoming/phi-2")
tokenizer.save_pretrained("/home/demo/models/incoming/phi-2")
print("Downloaded phi-2")
EOF
```

**Note**: This will take 5-10 minutes to download and 2-3 minutes to load with vLLM.

## Test Scenario 3: Inference Request

### Once Model is Loaded
```bash
# Check model status
curl http://localhost:8000/models | jq '.models[] | {name, status, runtime, gpu}'

# Make prediction (if model loaded)
curl -X POST http://localhost:8000/predict/tiny_test_model \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "organizationId": "my-company",
    "max_tokens": 50
  }'
```

## Test Scenario 4: Orchestration

### Drop Multiple Models
```bash
# Create 3 test models
for i in 1 2 3; do
  cp tiny_test_model.pt /home/demo/models/incoming/test_model_$i.pt
  sleep 5  # Space them out
done
```

### Watch Orchestration
```bash
# Check what brain decides every 10 seconds
watch -n 2 'curl -s http://localhost:8000/status | jq .'
```

**Observe:**
- Models get discovered
- Brain decides which GPU for each
- Models load
- After 5 minutes idle, models may get evicted

## Expected Issues & Solutions

### Issue 1: vLLM Fails to Load
**Error**: "CUDA out of memory"

**Solution**:
```yaml
# In modelium.yaml
vllm:
  gpu_memory_utilization: 0.7  # Lower from 0.9
```

### Issue 2: Model Not Detected
**Check**:
```bash
ls -la /home/demo/models/incoming/
# Verify file is there and readable
```

**Solution**: Check watch directory in config matches actual path

### Issue 3: Brain Fails to Load
**Error**: "Failed to load brain: modelium/brain-v1..."

**Expected**: This is NORMAL - falls back to rules
**Check logs**: Should see "✅ Using rule-based fallback"

### Issue 4: Port Already in Use
**Error**: "Address already in use"

**Solution**:
```bash
pkill -f "modelium.cli serve"
# Or change port
python -m modelium.cli serve --port 8001
```

## Success Criteria

### ✅ Phase 1 Complete If:
1. Server starts without errors
2. Brain loads (or falls back gracefully)
3. `/status` endpoint returns 200
4. GPU count shows 4

### ✅ Phase 2 Complete If:
1. Model gets discovered in logs
2. Model shows in `/models` endpoint
3. Model status changes: discovered → analyzing → loading → loaded
4. Can make inference request

### ✅ Phase 3 Complete If:
1. Multiple models can be loaded
2. Orchestration runs (check logs every 10s)
3. Idle models get evicted (after 5 min)

## Troubleshooting Commands

```bash
# Check if server running
ps aux | grep modelium

# Check GPU usage
nvidia-smi

# Check GPU memory from Python
python -c "import torch; print([torch.cuda.memory_allocated(i)/1e9 for i in range(4)])"

# Check ports
netstat -tuln | grep 8000

# View full logs
tail -200 modelium.log

# Restart server
pkill -f "modelium.cli serve"
python -m modelium.cli serve --config modelium.yaml > modelium.log 2>&1 &
```

## Next Steps After Successful Test

1. **Document what worked**
   - Model types tested
   - Load times
   - Memory usage

2. **Identify gaps**
   - Which placeholders hit?
   - What failed?
   - Performance issues?

3. **Plan production fixes**
   - Docker containers
   - K8s manifests
   - Prometheus metrics
   - Request queueing

## Quick Reference

**Start**: `python -m modelium.cli serve --config modelium.yaml`  
**Check**: `python -m modelium.cli check -v`  
**Status**: `curl http://localhost:8000/status`  
**Models**: `curl http://localhost:8000/models`  
**Logs**: `tail -f modelium.log`  
**Stop**: `pkill -f "modelium.cli serve"`

Good luck testing! 🚀

