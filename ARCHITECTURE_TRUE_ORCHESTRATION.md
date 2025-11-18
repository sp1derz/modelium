# True Orchestration Architecture

## Core Principle

**Modelium actively manages models, not just discovers them.**

```
User drops model → Modelium decides where/how to load it → 
Actively tells runtime to load → Monitors usage → 
Dynamically unloads idle models → Loads high-demand models
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Modelium Orchestrator                         │
│                                                                   │
│  ┌────────────────┐   ┌──────────────────┐   ┌───────────────┐ │
│  │ Model Watcher  │──→│   Analyzer       │──→│  Brain        │ │
│  │ (file system)  │   │ (config.json)    │   │ (AI decisions)│ │
│  └────────────────┘   └──────────────────┘   └───────────────┘ │
│                                                      ↓            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Decision Engine                                 ││
│  │  - Which runtime? (vLLM/Triton/Ray)                         ││
│  │  - Which GPU?                                                ││
│  │  - What settings? (batch size, tensor parallel, etc.)       ││
│  │  - Load now or queue?                                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                          ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Runtime Managers                                ││
│  │  - Triton: Dynamic load/unload via API                      ││
│  │  - vLLM: Process manager (spawn/kill instances)             ││
│  │  - Ray: Deployment manager (serve.run)                      ││
│  └─────────────────────────────────────────────────────────────┘│
│                          ↓                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Metrics Collector (Prometheus)                  ││
│  │  - GPU utilization per model                                 ││
│  │  - QPS, latency, throughput                                  ││
│  │  - Idle time                                                  ││
│  │  - Memory usage                                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                          ↑                                        │
└──────────────────────────┼───────────────────────────────────────┘
                           │ (feedback loop)
                           │
┌──────────────────────────┼───────────────────────────────────────┐
│                  Shared Model Repository                          │
│  /models/repository/                                              │
│    ├── gpt2/                                                      │
│    │   ├── config.json                                           │
│    │   ├── model.safetensors                                     │
│    │   └── tokenizer.json                                        │
│    └── llama-2-7b/                                               │
│        └── ...                                                    │
└───────────────────────────────────────────────────────────────────┘
                           ↑
                           │ (mounted as volume)
                           │
┌──────────────────────────┴───────────────────────────────────────┐
│                  Runtimes (User-Managed)                          │
│                                                                   │
│  Triton:     Points to /models/repository                        │
│              Modelium calls load/unload API                       │
│                                                                   │
│  vLLM:       Modelium spawns instances on-demand                 │
│              Each instance: vllm --model /models/repository/X    │
│                                                                   │
│  Ray Serve:  Modelium deploys via Python API                     │
│              serve.run(..., model_path=/models/repository/X)     │
└───────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Shared Model Repository Pattern

**All models in ONE place:**
```
/models/
  ├── repository/          # Source of truth
  │   ├── gpt2/
  │   ├── llama-2-7b/
  │   └── stable-diffusion/
  └── incoming/            # Drop zone (watcher monitors)
```

**All runtimes mount this:**
```bash
# Triton
-v /models/repository:/models

# vLLM instances (spawned by Modelium)
vllm --model /models/repository/gpt2

# Ray Serve (Modelium passes path)
serve.run(model_path="/models/repository/gpt2")
```

### 2. Runtime Managers (NEW)

#### Triton Manager
- Runtime has model repository mounted
- Modelium calls Triton API to load/unload
- True dynamic orchestration

```python
class TritonManager:
    def load_model(self, model_name: str, model_path: str):
        # Copy/symlink model to Triton's repository
        # Call: POST /v2/repository/models/{model}/load
        
    def unload_model(self, model_name: str):
        # Call: POST /v2/repository/models/{model}/unload
```

#### vLLM Manager
- Spawn vLLM process per model
- Kill process to unload
- Port management

```python
class VLLMManager:
    def load_model(self, model_name: str, model_path: str, gpu_id: int):
        # Spawn: vllm --model /models/repository/{model} --gpu {gpu_id}
        # Track process, port
        
    def unload_model(self, model_name: str):
        # Kill process
        # Free GPU memory
```

#### Ray Serve Manager
- Deploy via Python API
- Use Ray's deployment management

```python
class RayManager:
    def load_model(self, model_name: str, model_path: str, gpu_id: int):
        # Create deployment:
        # @serve.deployment(ray_actor_options={"num_gpus": 1})
        # serve.run(deployment, model_path=model_path)
        
    def unload_model(self, model_name: str):
        # serve.delete(deployment_name)
```

### 3. Brain Decision Engine (ENHANCED)

**Inputs:**
- Model analysis (size, type, architecture)
- Current GPU state (memory, utilization)
- Historical metrics (QPS, latency)
- Policy (always-loaded models, priorities)

**Outputs:**
- Runtime selection
- GPU allocation
- Loading parameters
- Load immediately or queue

**Example Decision:**
```python
{
    "action": "load",
    "model": "llama-2-7b",
    "runtime": "vllm",
    "gpu_id": 2,
    "settings": {
        "tensor_parallel_size": 1,
        "max_batch_size": 32,
        "dtype": "float16"
    },
    "priority": "high",
    "reason": "High QPS model, GPU 2 has space"
}
```

### 4. Prometheus Metrics

**Collected:**
- `modelium_model_qps{model, runtime, gpu}`
- `modelium_model_latency{model, runtime, gpu}`
- `modelium_gpu_memory_used{gpu}`
- `modelium_gpu_utilization{gpu}`
- `modelium_model_idle_seconds{model}`

**Used by Brain:**
```python
def should_unload(model):
    if model.idle_seconds > threshold:
        if model.qps_last_hour < min_qps:
            if not model in always_loaded:
                return True
    return False

def should_load(model):
    if model.pending_requests > threshold:
        if gpu_has_space():
            return True
    return False
```

## Orchestration Flows

### Flow 1: Model Discovery & Loading

```
1. User drops model in /models/incoming/gpt2/
2. Watcher detects it
3. Analyzer parses config.json
   - Architecture: GPT2LMHeadModel
   - Size: 548MB
   - Type: NLP/LLM
4. Brain decides:
   - Best runtime: vLLM (it's an LLM)
   - GPU allocation: GPU 1 (has 20GB free)
   - Settings: fp16, max_batch=32
5. Orchestrator executes:
   - Copy to /models/repository/gpt2/
   - Call VLLMManager.load_model()
   - VLLMManager spawns: vllm --model /models/repository/gpt2 --gpu 1
6. Model ready for inference
7. Metrics start collecting
```

### Flow 2: Dynamic Unloading (High Utilization)

```
1. Prometheus shows: GPU 0 at 95% memory
2. Brain analyzes loaded models on GPU 0:
   - model-a: idle 3600s, 0 QPS
   - model-b: idle 30s, 10 QPS
   - model-c: idle 10s, 100 QPS
3. Brain decides: Unload model-a
4. Orchestrator executes:
   - Call runtime.unload_model("model-a")
   - GPU 0 memory freed: 15GB
5. New model can now load on GPU 0
```

### Flow 3: Smart Loading (Queue Management)

```
1. 5 new models dropped
2. Only 2 GPUs available
3. Brain ranks by priority:
   - High: Production models with known traffic
   - Medium: New models from important teams
   - Low: Experimental models
4. Loads top 2 immediately
5. Queues remaining 3
6. As GPU space frees up, loads from queue
```

## Configuration

```yaml
orchestration:
  enabled: true
  
  # Model Repository
  model_repository: "/models/repository"
  incoming_directory: "/models/incoming"
  
  # Decision Policies
  policies:
    # Auto-unload after idle time
    auto_unload_after_seconds: 3600
    min_qps_threshold: 0.1
    
    # Always keep these loaded
    always_loaded:
      - "production-gpt"
      - "chat-model"
    
    # GPU management
    max_gpu_memory_utilization: 0.9
    enable_model_swapping: true
    
    # Priority
    high_priority_teams: ["prod", "research"]
  
  # Metrics
  metrics:
    prometheus_enabled: true
    prometheus_port: 9090
    collection_interval_seconds: 10

# Runtime configurations
triton:
  enabled: true
  model_repository: "/models/repository"
  endpoint: "http://localhost:8003"
  
vllm:
  enabled: true
  base_port: 8100  # Spawn instances on 8100, 8101, 8102...
  default_settings:
    dtype: "float16"
    max_model_len: 2048
    
ray_serve:
  enabled: true
  endpoint: "http://localhost:8002"
```

## Comparison: Old vs New

### Old Architecture (What I Built - WRONG)
```
User starts vLLM with model loaded
     ↓
Modelium discovers it
     ↓
Modelium routes requests
```
**Problem:** Not orchestrating, just discovering!

### New Architecture (TRUE ORCHESTRATION)
```
User drops model
     ↓
Modelium analyzes
     ↓
Brain decides (runtime, GPU, settings)
     ↓
Modelium actively loads model
     ↓
Monitors usage (Prometheus)
     ↓
Dynamically unloads idle models
     ↓
High GPU utilization achieved!
```

## Benefits

✅ **True Orchestration**: Actively manages model lifecycle
✅ **High GPU Utilization**: Dynamic load/unload based on demand
✅ **Intelligent**: Brain uses metrics to optimize
✅ **Flexible**: Works with multiple runtimes
✅ **Scalable**: Queue management for many models
✅ **Production-Ready**: Prometheus, metrics, policies

## Implementation Priority

1. **Triton Manager** (easiest - has all APIs)
2. **Prometheus Metrics** (essential for Brain)
3. **Enhanced Brain** (decision engine with metrics)
4. **vLLM Manager** (process spawning)
5. **Ray Manager** (deployment API)
6. **Dynamic Orchestration Loop** (load/unload)

---

This is the REAL Modelium architecture you envisioned.

