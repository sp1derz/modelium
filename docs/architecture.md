# Modelium Architecture

## System Overview

Modelium is an **AI-orchestrated multi-model serving platform** that maximizes GPU utilization through intelligent, automated model lifecycle management. The system continuously monitors resource usage and dynamically loads/unloads models based on demand, ensuring optimal GPU utilization while maintaining low latency.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User / External API                       │
│         (Drop Models, HTTP Requests, Monitoring)             │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                  FastAPI Server (Port 8000)                  │
│   Endpoints: /status /models /predict/<model> /health        │
└─┬─────────────────────────────────────────────────────────┬─┘
  │                                                           │
  │  ┌────────────────────────────────────────────────────┐  │
  │  │          Modelium Core Services Layer              │  │
  │  │                                                    │  │
  │  │  ┌──────────────────┐  ┌─────────────────────┐   │  │
  │  │  │  Model Watcher   │  │  Model Registry     │   │  │
  │  │  │  (Filesystem)    │──▶ (Singleton Store)   │   │  │
  │  │  │  Scan: 30s       │  │  Thread-safe        │   │  │
  │  │  └──────────────────┘  └─────────────────────┘   │  │
  │  │           │                       │               │  │
  │  │           └───────┐       ┌───────┘               │  │
  │  │                   ▼       ▼                       │  │
  │  │          ┌─────────────────────────┐              │  │
  │  │          │   Orchestrator Service  │              │  │
  │  │          │   - Every 10s loop      │              │  │
  │  │          │   - Uses brain          │              │  │
  │  │          │   - Load/unload         │              │  │
  │  │          └──────────┬──────────────┘              │  │
  │  │                     │                             │  │
  │  └─────────────────────┼─────────────────────────────┘  │
  │                        │                                 │
  │  ┌─────────────────────▼─────────────────────────────┐  │
  │  │      Modelium Brain (Qwen-2.5-1.8B)               │  │
  │  │                                                    │  │
  │  │  Task 1: Deployment Planning                      │  │
  │  │   • Analyze model descriptor                      │  │
  │  │   • Choose runtime (vLLM/Ray/TRT)                 │  │
  │  │   • Select GPU                                    │  │
  │  │                                                    │  │
  │  │  Task 2: Orchestration (Every 10s)                │  │
  │  │   • Gather metrics (QPS, latency, idle)           │  │
  │  │   • Evaluate all models                           │  │
  │  │   • Decide: keep/evict/load                       │  │
  │  │                                                    │  │
  │  │  Fallback: Rule-based if LLM unavailable          │  │
  │  └────────────────────────────────────────────────────┘  │
  │                        │                                 │
  └────────────────────────┼─────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  Runtime Execution Layer                     │
│                                                              │
│  ┌──────────────────┐  ┌─────────────────┐                 │
│  │  vLLM Service    │  │  Ray Serve      │                 │
│  │  • For LLMs      │  │  • General      │                 │
│  │  • OpenAI API    │  │  • Custom API   │                 │
│  │  • Dynamic       │  │  • Flexible     │                 │
│  └────────┬─────────┘  └────────┬────────┘                 │
│           │                     │                           │
└───────────┼─────────────────────┼───────────────────────────┘
            │                     │
┌───────────▼─────────────────────▼───────────────────────────┐
│                     GPU Resources                            │
│    GPU 0     GPU 1     GPU 2     GPU 3    ...               │
│   [Model A] [Model B]  [Idle]  [Loading]                    │
│   80GB Used  45GB     0GB       30GB                         │
└──────────────────────────────────────────────────────────────┘
```

## Core Services

### 1. Model Watcher

**File**: `modelium/services/model_watcher.py`

**Purpose**: Monitors directories for new model files and triggers discovery

**Implementation**:
```python
from watchdog.observers import Observer

class ModelWatcher:
    def __init__(self, watch_dirs, registry, orchestrator):
        self.watch_dirs = watch_dirs
        self.registry = registry
        self.orchestrator = orchestrator
    
    def on_created(self, event):
        # New file detected
        if event.src_path.endswith(('.pt', '.pth', '.onnx')):
            model_name = extract_name(event.src_path)
            
            # Register
            self.registry.register_model(
                name=model_name,
                path=event.src_path,
                status=ModelStatus.DISCOVERED
            )
            
            # Notify orchestrator
            self.orchestrator.on_model_discovered(model_name, event.src_path)
```

**Configuration**:
```yaml
orchestration:
  model_discovery:
    watch_directories: ["/models/incoming"]
    scan_interval_seconds: 30
    supported_extensions: [".pt", ".pth", ".onnx", ".safetensors"]
```

**Key Features**:
- Background thread with watchdog
- Supports multiple directories
- File extension filtering
- Debouncing for large files

### 2. Model Registry

**File**: `modelium/services/model_registry.py`

**Purpose**: Central source of truth for all models

**Data Model**:
```python
class ModelStatus(str, Enum):
    DISCOVERED = "discovered"
    ANALYZING = "analyzing"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    ERROR = "error"

class ModelEntry(BaseModel):
    name: str
    path: str
    status: ModelStatus
    runtime: Optional[str]  # "vllm", "ray_serve"
    gpu_id: Optional[int]
    
    # Metrics
    qps: float = 0.0
    avg_latency_ms: float = 0.0
    last_request_time: Optional[float]
    loaded_at: Optional[float]
    unloaded_at: Optional[float]
    
    # Metadata
    model_type: Optional[str]  # "llm", "vision", "text"
    framework: Optional[str]  # "pytorch", "onnx"
    size_gb: Optional[float]
```

**Key Methods**:
```python
class ModelRegistry:
    _instance = None
    
    def register_model(name, path, status):
        # Add to registry
    
    def update_model(name, **kwargs):
        # Update attributes
    
    def get_model(name) -> ModelEntry:
        # Thread-safe retrieval
    
    def get_all_models() -> List[ModelEntry]:
        # All models
    
    def update_metrics(name, qps, latency):
        # Called on each request
```

**Features**:
- Thread-safe singleton
- In-memory (fast lookups)
- Metrics tracking
- Status lifecycle management

### 3. Modelium Brain

**File**: `modelium/brain/unified_brain.py`

**Purpose**: AI-powered decision making for both deployment planning and runtime orchestration

**Model**: Qwen-2.5-1.5B-Instruct (downloads from HuggingFace)

#### Task 1: Deployment Planning

**Input**:
```json
{
  "model_name": "my-llm",
  "framework": "pytorch",
  "model_type": "llm",
  "size_gb": 14.5,
  "parameters_millions": 7000,
  "available_gpus": [
    {"id": 0, "used_gb": 20, "total_gb": 80},
    {"id": 1, "used_gb": 60, "total_gb": 80}
  ]
}
```

**Output**:
```json
{
  "runtime": "vllm",
  "gpu_id": 0,
  "confidence": 0.92,
  "reasoning": "LLM detected, vLLM optimal. GPU 0 has 60GB free."
}
```

#### Task 2: Orchestration Decisions

**Input** (every 10s):
```json
{
  "models": [
    {
      "name": "qwen-7b",
      "status": "loaded",
      "gpu": 1,
      "qps": 45.2,
      "idle_seconds": 0
    },
    {
      "name": "bert-base",
      "status": "loaded",
      "gpu": 2,
      "qps": 0,
      "idle_seconds": 320
    }
  ],
  "gpu_state": {
    "gpu_0": {"used": 0, "total": 80},
    "gpu_1": {"used": 45, "total": 80},
    "gpu_2": {"used": 12, "total": 80}
  }
}
```

**Output**:
```json
{
  "qwen-7b": {"action": "keep", "reason": "High traffic"},
  "bert-base": {"action": "evict", "reason": "Idle >5min"}
}
```

#### Fallback Mode

If brain fails to load or is disabled:
```python
# Rule-based fallback
if model_type == "llm":
    runtime = "vllm"
elif model_type == "vision":
    runtime = "ray_serve"
else:
    runtime = "ray_serve"

# Always load on first request
# Evict after 5 min idle
```

### 4. Orchestrator

**File**: `modelium/services/orchestrator.py`

**Purpose**: Execute brain decisions and manage model lifecycle

**Main Loop**:
```python
class Orchestrator:
    async def orchestration_loop(self):
        while True:
            # 1. Gather state
            models = self.registry.get_all_models()
            gpu_state = self._get_gpu_memory_state()
            
            # 2. Ask brain
            if self.config.orchestration.mode == "intelligent":
                decisions = self.brain.make_orchestration_decision(
                    models=models,
                    gpu_state=gpu_state,
                    config=self.config.orchestration.policies
                )
            else:
                decisions = self._rule_based_decisions(models)
            
            # 3. Execute
            for model_name, decision in decisions.items():
                if decision["action"] == "evict":
                    await self._evict_model(model_name)
                elif decision["action"] == "load":
                    await self._load_model(model_name)
            
            # 4. Wait
            await asyncio.sleep(
                self.config.orchestration.decision_interval_seconds
            )
```

**Actions**:
- **keep**: Do nothing, model stays loaded
- **evict**: Unload model, free GPU memory
- **load**: Load model to selected GPU

**GPU Memory Tracking**:
```python
def _get_gpu_memory_state(self) -> dict:
    import torch
    
    gpu_state = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        
        gpu_state[f"gpu_{i}"] = {
            "used": reserved,
            "total": total
        }
    return gpu_state
```

### 5. vLLM Service

**File**: `modelium/services/vllm_service.py`

**Purpose**: Manage vLLM instances for LLM serving

**Methods**:

**Load Model**:
```python
def load_model(self, model_path: str, gpu_id: int):
    # Launch vLLM subprocess
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.9",
        "--port", str(8000 + gpu_id)
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    process = subprocess.Popen(cmd, env=env)
    
    # Wait for health
    await self._wait_for_health(port)
    
    # Store
    self.processes[model_name] = {
        "process": process,
        "port": port,
        "gpu": gpu_id
    }
```

**Unload Model**:
```python
def unload_model(self, model_name: str):
    if model_name in self.processes:
        process = self.processes[model_name]["process"]
        process.terminate()
        process.wait(timeout=30)
        del self.processes[model_name]
```

**Inference**:
```python
async def predict(self, model_name: str, prompt: str, params: dict):
    port = self.processes[model_name]["port"]
    
    # OpenAI-compatible API
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:{port}/v1/completions",
            json={"prompt": prompt, **params}
        ) as resp:
            return await resp.json()
```

**Current Implementation**: Subprocess-based (works, but not containerized)
**TODO**: Docker containers for isolation and better resource management

### 6. FastAPI Server

**File**: `modelium/cli.py` (in `serve` command)

**Endpoints**:

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/status")
async def status():
    return {
        "status": "running",
        "organization": config.organization.id,
        "gpu_count": torch.cuda.device_count(),
        "models_loaded": len([m for m in registry.get_all_models() if m.status == ModelStatus.LOADED]),
        "models_discovered": len(registry.get_all_models()),
    }

@app.get("/models")
async def list_models():
    models = registry.get_all_models()
    return {"models": [m.dict() for m in models]}

@app.post("/predict/{model_name}")
async def predict(model_name: str, request: dict):
    model = registry.get_model(model_name)
    
    # Check status
    if model.status != ModelStatus.LOADED:
        # Trigger loading
        orchestrator.on_model_discovered(model_name, model.path)
        return {"status": "loading", "message": "Model is being loaded"}
    
    # Route to correct runtime
    if model.runtime == "vllm":
        result = await vllm_service.predict(model_name, request["prompt"], request)
    elif model.runtime == "ray_serve":
        result = await ray_service.predict(model_name, request)
    
    # Update metrics
    registry.update_metrics(model_name, qps=..., latency=...)
    
    return result
```

## Data Flow

### Scenario 1: New Model Dropped

```
1. User: cp model.pt /models/incoming/
   ↓
2. Watcher: Detects file, fires event
   ↓
3. Registry: Status = DISCOVERED
   ↓
4. Orchestrator: on_model_discovered()
   ↓
5. Analyzer: Extract metadata (framework, type, size)
   ↓
6. Registry: Status = ANALYZING
   ↓
7. Brain: Generate deployment plan (runtime + GPU)
   ↓
8. Registry: Status = LOADING
   ↓
9. vLLM Service: Load model to GPU
   ↓
10. Registry: Status = LOADED
   ↓
11. API: /predict/model available
```

### Scenario 2: Orchestration Loop (Every 10s)

```
1. Orchestrator: Gather all models + metrics
   ↓
2. GPU Query: Get memory usage (CUDA)
   ↓
3. Brain: Analyze all models
   - qwen-7b: 50 QPS → Keep
   - bert: 0 QPS, idle 320s → Evict
   ↓
4. Execute: Unload bert from GPU 2
   ↓
5. Registry: bert.status = UNLOADED
   ↓
6. Wait 10 seconds, repeat
```

### Scenario 3: Inference Request

```
1. User: POST /predict/qwen-7b
   ↓
2. Registry: Check status
   - If LOADED → Continue
   - If UNLOADED → Trigger load
   ↓
3. vLLM Service: Forward to correct port
   ↓
4. vLLM: Generate response
   ↓
5. Track Metrics: QPS++, latency
   ↓
6. Return: Response to user
```

## Configuration

See `modelium.yaml` for full configuration. Key sections:

**Brain**:
```yaml
modelium_brain:
  enabled: true
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  device: "cuda:0"
  fallback_to_rules: true
```

**Orchestration**:
```yaml
orchestration:
  enabled: true
  mode: "intelligent"  # or "simple"
  decision_interval_seconds: 10
  
  model_discovery:
    watch_directories: ["/models/incoming"]
    scan_interval_seconds: 30
  
  policies:
    evict_after_idle_seconds: 300
    evict_when_memory_above_percent: 85
    always_loaded: []
```

**Runtimes**:
```yaml
vllm:
  enabled: true
  gpu_memory_utilization: 0.9
  port: 8000

ray_serve:
  enabled: true
  num_gpus_per_replica: 1.0
  port: 8001
```

## Technology Stack

**Core**:
- Python 3.10+
- FastAPI + Uvicorn (API server)
- PyTorch (GPU management)
- Pydantic (data validation)

**AI**:
- Transformers (brain model)
- Qwen-2.5-1.8B (decision making)

**Runtimes**:
- vLLM (LLM serving)
- Ray Serve (general models)
- TensorRT (future)

**Monitoring**:
- watchdog (file system)
- threading (background tasks)
- asyncio (async I/O)

## Performance Characteristics

**Latency**:
- Model discovery: <1s (watchdog event)
- Analysis: 2-5s (depends on model size)
- Brain decision: 0.5-1s (LLM inference)
- vLLM loading: 30-120s (depends on model size)
- Orchestration cycle: 10s interval

**Throughput**:
- vLLM: 50-200 QPS per model (continuous batching)
- Brain: 1 decision per 10s (orchestration)
- Registry: Thread-safe, handles concurrent updates

**Resource Usage**:
- Brain: 3-4GB GPU memory (Qwen-1.8B)
- vLLM: 10-80GB per model (depends on size)
- Python overhead: ~500MB RAM

## Current Status

**Production Ready**:
- ✅ Model discovery & watching
- ✅ Model registry (thread-safe)
- ✅ Brain (LLM + fallback)
- ✅ Orchestrator (10s loop)
- ✅ vLLM service (subprocess)
- ✅ Real GPU tracking
- ✅ FastAPI endpoints

**Work In Progress**:
- ⏳ Prometheus metrics
- ⏳ Docker containers
- ⏳ Kubernetes manifests
- ⏳ Request queueing
- ⏳ Ray Serve integration
- ⏳ Fine-tuned brain model

See [STATUS.md](../STATUS.md) for detailed progress.

## Security Considerations

**Current**:
- Sandboxed model analysis (safe loading)
- Pickle detection in security scanner
- No external network access during analysis

**Future**:
- Model signing & verification
- Rate limiting per organization
- API authentication
- Network policies (K8s)

## Scalability

**Current (Single Node)**:
- 4-8 GPUs per machine
- 10-50 models (with eviction)
- 1000s of requests per second

**Future (Multi-Node)**:
- Kubernetes deployment
- Multiple Modelium instances
- Distributed orchestration
- Cross-cluster routing

## Monitoring & Observability

**Current**:
- Status endpoint (`/status`)
- Models endpoint (`/models`)
- Log files (`modelium.log`)

**Planned**:
- Prometheus metrics export
- Grafana dashboards
- Distributed tracing (OpenTelemetry)
- Alert manager integration

## References

- [Getting Started](getting-started.md)
- [The Brain](brain.md)
- [Usage Guide](usage.md)
- [Testing](../TESTING_TOMORROW.md)
- [Status](../STATUS.md)
