# Modelium Architecture

## The Vision

**"Maximum GPU utilization with minimum user effort"**

Users drop models → Modelium loads them → Tracks usage → Unloads idle → Repeat

## The Flow

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

## Directory Structure

```
modelium/
├── runtime_manager.py        # ONE file for ALL runtimes
├── cli.py                    # FastAPI server + CLI
├── config.py                 # Configuration
│
├── brain/                    
│   └── unified_brain.py      # Decision maker
│
├── core/analyzers/           
│   └── huggingface_analyzer.py  # Reads config.json
│
├── services/
│   ├── orchestrator.py       # Watches → Decides → Loads
│   ├── model_watcher.py      # Monitors folder
│   └── model_registry.py     # Tracks models
│
└── metrics/
    └── prometheus_exporter.py  # Tracks everything
```

**That's it!** 8 directories, ~2000 lines of code.

## Core Components

### 1. RuntimeManager (`runtime_manager.py`)

**Purpose**: Handle ALL runtimes in ONE place

**Methods**:
- `load_model(name, path, runtime, gpu_id)` - Load to vLLM/Triton/Ray
- `unload_model(name)` - Unload from runtime
- `inference(name, prompt)` - Run inference

**Why ONE file?**
- No confusion about connectors vs managers
- Easy to understand (read 450 lines, know everything)
- Easy to extend (add `_load_tgi()` method)

### 2. Orchestrator (`services/orchestrator.py`)

**Purpose**: The INTELLIGENT brain of the system

**Key Method**:
```python
def on_model_discovered(model_name, model_path):
    """Called when watcher finds new model"""
    
    # 1. Analyze
    analysis = self.analyzer.analyze(model_path)
    
    # 2. Brain decides runtime
    runtime = self._choose_runtime(analysis)  # vLLM/Triton/Ray
    
    # 3. Load it
    self.runtime_manager.load_model(
        model_name, model_path, runtime, gpu_id
    )
    
    # Done!
```

**INTELLIGENT Background Loop** (Every 10s):
```python
def _check_for_idle_models():
    """
    Smart decisions considering:
    1. Policies (from config)
    2. Prometheus metrics (real-time)
    3. GPU memory state
    """
    
    for model in loaded_models:
        # Get metrics
        qps = metrics.get_model_qps(model.name)
        idle_seconds = metrics.get_model_idle_seconds(model.name)
        gpu_pressure = self._get_gpu_memory_pressure()
        
        # RULE 1: Never unload if in always_loaded list
        if model.name in always_loaded:
            continue
        
        # RULE 2: Keep if actively used (QPS > 0.5)
        if qps > 0.5:
            continue
        
        # RULE 3: Keep if has ANY usage (QPS > 0.01)
        # Even 1 req/100s = someone using it!
        if qps > 0.01:
            continue
        
        # RULE 4: Keep if recently used
        if idle_seconds < idle_threshold:
            continue
        
        # RULE 5: Unload if TRULY idle AND (GPU needs space OR idle > 2x threshold)
        if qps == 0 and idle_seconds > idle_threshold:
            if gpu_pressure or idle_seconds > (idle_threshold * 2):
                runtime_manager.unload_model(model.name)
            else:
                # Idle but GPU has space - keep it (might be used soon!)
                pass
```

**Why This is INTELLIGENT**:
- ✅ Scenario 1: 5 models, 1 hot (100 QPS), 4 warm (1-5 QPS) → **Keeps all 5** (all being used!)
- ✅ Scenario 2: Model with 80 QPS at 60% GPU → **No scaling** (sufficient)
- ✅ Scenario 3: Model with 0 QPS for 10+ min → **Unloads only if GPU needs space**
- ✅ Never aggressive: Prefers keeping models loaded unless necessary
- ✅ GPU-aware: Only unloads when memory pressure exists

### 3. ModelWatcher (`services/model_watcher.py`)

**Purpose**: Watch `/models/incoming/` for new files

**How**: Scans directory every 30 seconds, calls `orchestrator.on_model_discovered()` for new models.

### 4. ModelRegistry (`services/model_registry.py`)

**Purpose**: Track all models (discovered, loading, loaded, unloaded)

**Methods**:
- `register_model()` - Add new model
- `update_model()` - Update status/metrics
- `get_model()` - Retrieve model info
- `get_loaded_models()` - All loaded models

### 5. Brain (`brain/unified_brain.py`)

**Purpose**: Choose best runtime for each model

**Logic**:
```python
def _choose_runtime(analysis):
    # LLMs → vLLM (if enabled)
    if "gpt" in arch or "llama" in arch or "qwen" in arch:
        if vllm_enabled:
            return "vllm"
    
    # Ray for everything else (if enabled)
    if ray_enabled:
        return "ray"
    
    # Triton as fallback
    return "triton"
```

**Future**: Fine-tuned LLM for smarter decisions (HuggingFace: `modelium/brain-v1`)

### 6. Metrics (`metrics/prometheus_exporter.py`)

**Purpose**: Track everything for monitoring and unload decisions

**Tracks**:
- Requests per second (QPS)
- Latency (P50, P95, P99)
- Idle time (time since last request)
- Model load/unload events
- GPU memory (optional)

**Used by**: Orchestrator to decide when to unload models

## Data Flow

### New Model Dropped

```
1. User: cp gpt2/ /models/incoming/
2. Watcher: "New model: gpt2"
3. Orchestrator: Analyze it
4. Analyzer: "GPT2LMHeadModel, 0.5GB"
5. Brain: "LLM → vLLM"
6. RuntimeManager: Spawn vLLM process
7. Registry: Status = LOADED
8. Done! /predict/gpt2 ready
```

### Inference Request

```
1. User: POST /predict/gpt2
2. Registry: Check status (LOADED)
3. RuntimeManager: Route to vLLM port
4. vLLM: Generate response
5. Metrics: Track QPS, latency
6. Return response
```

### Idle Model Unloaded

```
1. Orchestrator loop (every 10s)
2. Check: gpt2 idle for 320s, QPS=0
3. Decision: Unload (idle > 300s)
4. RuntimeManager: Kill vLLM process
5. Registry: Status = UNLOADED
6. GPU freed!
```

## Configuration

```yaml
# modelium.yaml - ONLY WHAT USERS NEED

# Enable runtimes
vllm:
  enabled: true

# Watch folder
orchestration:
  model_discovery:
    watch_directories: ["/models/incoming"]

# Unload policy
  policies:
    evict_after_idle_seconds: 300  # 5 minutes

# Metrics
metrics:
  enabled: true
  port: 9090
```

## Technology Stack

**Core**:
- Python 3.11+
- FastAPI (API server)
- PyTorch (GPU management)
- Pydantic (config validation)

**Runtimes** (user installs separately):
- vLLM (for LLMs)
- Triton (for all models)
- Ray Serve (for Python models)

**Monitoring**:
- Prometheus (metrics)
- watchdog (file monitoring)

## Performance

**Latency**:
- Model discovery: <1s
- Analysis: 1-2s
- Load decision: <0.1s
- Model loading: 10-120s (depends on model size)
- Inference: Depends on runtime

**Throughput**:
- Limited by runtime (vLLM: 50-200 QPS per model)
- Orchestrator: 1 decision per 10s (lightweight)

## Deployment

### Local (Development)
```bash
python -m modelium.cli serve
```

### Docker (Production)
```bash
docker-compose up -d
```

### Kubernetes (Scale)
```bash
kubectl apply -k infra/k8s/
```

See [DEPLOYMENT.md](../DEPLOYMENT.md) for details.

## Why This Design?

### Simple for Users
- Drop model → It loads automatically
- No complex configuration
- No manual load/unload

### Simple for Developers
- ONE file for runtimes (`runtime_manager.py`)
- Clear flow: Watch → Analyze → Decide → Load
- Easy to extend

### Maximum GPU Utilization
- Automatic loading on demand
- Automatic unloading when idle
- Metrics-driven decisions

**The complexity is hidden. The user experience is simple.**
