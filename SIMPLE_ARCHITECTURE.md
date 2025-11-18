# 🎯 Modelium - Simple Architecture

## The Vision (Your Words)
> **"Watch folder → Drop model → Brain decides runtime → Load it → Track metrics → Unload if idle"**

## The Reality (What We Built)

```
User drops model in /models/incoming/
           ↓
    ModelWatcher detects it
           ↓
    Orchestrator.on_model_discovered()
           ↓
    HuggingFaceAnalyzer reads config.json
           ↓
    Brain decides: vLLM? Triton? Ray?
           ↓
    RuntimeManager.load_model(runtime=chosen)
           ↓
    Inference available at /predict/{model}
           ↓
    Prometheus tracks QPS, latency, idle time
           ↓
    If idle > threshold: RuntimeManager.unload_model()
           ↓
    GPU freed for next model!
```

## Directory Structure (SIMPLE)

```
modelium/
├── runtime_manager.py        ← ONE file for ALL runtimes (vLLM/Triton/Ray)
├── brain/                    
│   └── unified_brain.py      ← LLM decision maker
├── services/
│   ├── orchestrator.py       ← Watches → Decides → Loads
│   ├── model_watcher.py      ← Monitors /models/incoming/
│   └── model_registry.py     ← Tracks what's loaded
├── metrics/
│   └── prometheus_exporter.py ← Tracks everything
├── core/analyzers/
│   └── huggingface_analyzer.py ← Reads config.json
└── cli.py                    ← python -m modelium.cli serve
```

**That's it!** No connectors, no managers, no repository - just 7 core files.

## User Configuration (SIMPLE)

```yaml
# modelium.yaml - ONLY WHAT USERS NEED TO CONFIGURE

# Which runtimes to use? (Enable what you have)
vllm:
  enabled: true          # ← Set to true if you want to use vLLM
triton:
  enabled: false
ray_serve:
  enabled: false

# Watch this folder for new models
orchestration:
  model_discovery:
    watch_directories: 
      - /models/incoming  # ← Drop models here

# When to unload idle models?
  policies:
    evict_after_idle_seconds: 300  # ← 5 minutes idle = unload

# Metrics
metrics:
  enabled: true
  port: 9090  # ← Prometheus at http://localhost:9090/metrics
```

**That's all the user needs to configure!** Everything else is automatic.

## How It Works (THE CORE)

### 1. RuntimeManager (ONE FILE - 450 lines)

```python
# modelium/runtime_manager.py

class RuntimeManager:
    """Handles vLLM, Triton, AND Ray in one place."""
    
    def load_model(self, model_name, model_path, runtime, gpu_id):
        """
        Load model into specified runtime.
        
        - vLLM: Spawns process (vllm.entrypoints.openai.api_server)
        - Triton: Calls /v2/repository/models/{name}/load API
        - Ray: serve.run(deployment)
        """
        if runtime == "vllm":
            return self._load_vllm(...)
        elif runtime == "triton":
            return self._load_triton(...)
        elif runtime == "ray":
            return self._load_ray(...)
    
    def unload_model(self, model_name):
        """
        Unload model from its runtime.
        
        - vLLM: Kill process
        - Triton: /v2/repository/models/{name}/unload
        - Ray: serve.delete()
        """
    
    def inference(self, model_name, prompt, **kwargs):
        """Route inference to correct runtime automatically."""
```

**Why ONE file?**
- No confusion: Everything related to runtimes is in ONE place
- Easy to understand: Read 450 lines, you know how EVERYTHING works
- Easy to extend: Want to add TGI? Add `_load_tgi()` method. Done.

### 2. Orchestrator (THE BRAIN)

```python
# modelium/services/orchestrator.py

class Orchestrator:
    """Watches → Analyzes → Decides → Loads."""
    
    def on_model_discovered(self, model_name, model_path):
        """Called when watcher detects new model."""
        
        # 1. ANALYZE: Read config.json
        analysis = self.analyzer.analyze(model_path)
        
        # 2. BRAIN DECIDES: Which runtime?
        runtime = self._choose_runtime(analysis)  # GPT? vLLM. Vision? Ray.
        
        # 3. LOAD: Use RuntimeManager
        self.runtime_manager.load_model(
            model_name=model_name,
            model_path=model_path,
            runtime=runtime,
            gpu_id=self._choose_gpu()  # Pick least used GPU
        )
        
        # 4. DONE! Model is now loaded and ready.
    
    def _check_for_idle_models(self):
        """Background loop: Unload models idle > 5 minutes."""
        for model in self.registry.get_loaded_models():
            if model.idle_seconds > threshold:
                self.runtime_manager.unload_model(model.name)
```

**Why THIS design?**
- Clear flow: 1 → 2 → 3 → 4, no magic
- Brain makes ONE decision: Which runtime?
- RuntimeManager handles the rest

### 3. Metrics (PROMETHEUS)

```python
# modelium/metrics/prometheus_exporter.py

class ModeliumMetrics:
    """Track everything that matters."""
    
    def record_request(self, model, runtime, latency_ms, status, gpu):
        """Track each inference request."""
        # QPS, latency, errors
    
    def get_model_idle_seconds(self, model, runtime):
        """How long since last request?"""
        # Used by orchestrator to decide when to unload
```

**What's tracked?**
- Requests per second (QPS)
- Latency (P50, P95, P99)
- Idle time (for unload decisions)
- GPU memory (optional)
- Brain decisions (load/unload reasons)

## The Flow (VISUAL)

```
┌─────────────────────────────────────────────────────────┐
│  User: cp gpt2/ /models/incoming/                      │
└────────────────────┬────────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │   ModelWatcher        │  (Scans /models/incoming/ every 30s)
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │   Orchestrator        │  
         │  .on_model_discovered │
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │  HuggingFaceAnalyzer  │  (Read config.json → GPT2)
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │   Brain               │  (GPT2 = LLM → vLLM is best)
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │  RuntimeManager       │  
         │  .load_model(vllm)    │  (Spawns: vllm --model /models/incoming/gpt2)
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │  vLLM Process         │  (Listening on http://localhost:8100)
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │  Modelium Server      │  
         │  /predict/gpt2        │  (Routes to vLLM)
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │  User: curl POST      │  
         │  /predict/gpt2        │  (Inference!)
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │  Prometheus Metrics   │  (Tracks QPS, latency, idle time)
         └───────────┬───────────┘
                     ↓ (5 minutes of no requests)
         ┌───────────────────────┐
         │  Orchestrator         │  (Idle detected → Unload)
         │  .unload_model(gpt2)  │
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │  RuntimeManager       │  (Kill vLLM process)
         │  .unload_model()      │  (GPU freed!)
         └───────────────────────┘
```

## Why This is SIMPLE

### Before (COMPLEX):
```
modelium/
├── connectors/        ← 4 files (800 lines) - HTTP clients
│   ├── vllm_connector.py
│   ├── triton_connector.py
│   └── ray_connector.py
├── managers/          ← 4 files (1300 lines) - Process managers
│   ├── vllm_manager.py
│   ├── triton_manager.py
│   └── ray_manager.py
└── repository/        ← 2 files (400 lines) - File restructuring
    └── model_repository.py

Total: 10 files, ~2,500 lines
```

**Problem**: Where do I look to understand how vLLM loading works?
- `vllm_connector.py`? 
- `vllm_manager.py`? 
- Both? 
- What's the difference?

### After (SIMPLE):
```
modelium/
└── runtime_manager.py  ← 1 file (450 lines) - EVERYTHING

Total: 1 file, 450 lines
```

**Solution**: ONE place. Read `runtime_manager.py`, understand EVERYTHING.

## File Size Comparison

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| Runtime handling | 2,500 lines (10 files) | 450 lines (1 file) | **-82%** |
| Directories | 6 (services, connectors, managers, repository, metrics, brain) | 4 (services, metrics, brain, core) | **-33%** |
| User confusion | High | **Zero** | ✅ |

## What Users See (THE EXPERIENCE)

### Step 1: Install
```bash
git clone https://github.com/sp1derz/modelium
cd modelium
pip install -e ".[all]"
```

### Step 2: Configure (ONE FILE)
```bash
nano modelium.yaml
# Set vllm.enabled: true
# Done!
```

### Step 3: Start
```bash
python -m modelium.cli serve
# 🧠 Modelium Server starting...
# ✅ Server ready at http://0.0.0.0:8000
```

### Step 4: Drop Model
```bash
cp -r my-gpt2-model /models/incoming/
# Server logs:
# 📋 New model discovered: my-gpt2-model
# 🎯 Brain decision: vllm
# 🚀 Loading model...
# ✅ my-gpt2-model loaded successfully!
```

### Step 5: Use It
```bash
curl http://localhost:8000/predict/my-gpt2-model \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

### Step 6: Metrics
```bash
# http://localhost:9090/metrics
# modelium_requests_total{model="my-gpt2-model",runtime="vllm"} 1
# modelium_latency_seconds{model="my-gpt2-model"} 0.123
```

### Step 7: Automatic Unload
```
# (After 5 minutes of no requests)
# Server logs:
# 🔽 Unloading idle model: my-gpt2-model (idle: 300s, QPS: 0.00)
# ✅ GPU freed!
```

## Summary

**THE GOAL**: Maximum GPU utilization with minimum user effort

**THE SOLUTION**: 
- Watch folder
- Analyze model (config.json)
- Brain decides runtime (vLLM/Triton/Ray)
- Load automatically
- Track metrics (Prometheus)
- Unload idle models
- **ALL IN 7 FILES**

**NO MORE**:
- ❌ Separate connectors directory
- ❌ Separate managers directory
- ❌ Separate repository directory
- ❌ Confusion about what goes where

**JUST**:
- ✅ `runtime_manager.py` - Handles ALL runtimes
- ✅ `orchestrator.py` - Makes decisions
- ✅ `model_watcher.py` - Watches folder
- ✅ `prometheus_exporter.py` - Tracks metrics
- ✅ `unified_brain.py` - Chooses runtime

**USER EXPERIENCE**:
1. Enable runtimes in `modelium.yaml`
2. Drop models in `/models/incoming/`
3. That's it!

**The complexity is GONE. The functionality remains.**

