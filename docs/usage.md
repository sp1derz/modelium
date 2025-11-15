# Usage Guide

## CLI Commands

### Check System

```bash
# Basic check
python -m modelium.cli check

# Verbose (shows GPU details)
python -m modelium.cli check --verbose
```

### Initialize Configuration

```bash
# Create default config
python -m modelium.cli init

# Force overwrite existing
python -m modelium.cli init --force
```

### Start Server

```bash
# Default config (modelium.yaml)
python -m modelium.cli serve

# Custom config
python -m modelium.cli serve --config my-config.yaml

# Custom host/port
python -m modelium.cli serve --host 0.0.0.0 --port 8080

# Run in background
nohup python -m modelium.cli serve > modelium.log 2>&1 &
```

## Basic Workflow

### 1. Drop Models

```bash
# Copy model to watched directory
cp your_model.pt /models/incoming/

# Or use any supported format
cp model.onnx /models/incoming/
cp model.safetensors /models/incoming/
```

### 2. Monitor Discovery

```bash
# Watch logs in real-time
tail -f modelium.log

# Check models API
curl http://localhost:8000/models | jq .
```

### 3. Make Requests

```bash
# Via curl
curl -X POST http://localhost:8000/predict/your_model \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your input text",
    "organizationId": "your-org",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Via Python
import requests

response = requests.post(
    "http://localhost:8000/predict/your_model",
    json={
        "prompt": "Your input text",
        "organizationId": "your-org",
        "max_tokens": 100
    }
)
print(response.json())
```

## Configuration

### Minimal Config

```yaml
organization:
  id: "my-company"

orchestration:
  model_discovery:
    watch_directories: ["/models/incoming"]

gpu:
  enabled: true
```

### With Brain Enabled

```yaml
modelium_brain:
  enabled: true
  fallback_to_rules: true

orchestration:
  enabled: true
  policies:
    evict_after_idle_seconds: 300
```

## Examples

### Deploy LLM

```bash
# Drop Qwen model
cp qwen-7b.pt /models/incoming/

# Modelium automatically:
# - Detects it's an LLM
# - Chooses vLLM runtime
# - Deploys to optimal GPU
# - Exposes API endpoint
```

### Deploy Vision Model

```bash
# Drop ResNet model
cp resnet50.pt /models/incoming/

# Modelium automatically:
# - Detects it's a vision model
# - Chooses TensorRT for max performance
# - Deploys alongside other models
```

## API Endpoints

### Predict

```bash
POST /predict/{model_name}
Content-Type: application/json

{
  "input": "data",
  "organizationId": "org-id"
}
```

### Status

```bash
GET /status

# Returns:
{
  "status": "running",
  "organization": "my-company",
  "gpu_count": 4,
  "gpu_enabled": true,
  "brain_enabled": true,
  "orchestration_enabled": true,
  "models_loaded": 3,
  "models_discovered": 5,
  "models_loading": 1
}
```

### List Models

```bash
GET /models

# Returns:
{
  "models": [
    {
      "name": "qwen-7b",
      "status": "loaded",
      "runtime": "vllm",
      "gpu": 1,
      "qps": 50.2,
      "idle_seconds": 0
    },
    {
      "name": "bert-base",
      "status": "unloaded",
      "runtime": "ray_serve",
      "gpu": null,
      "qps": 0,
      "idle_seconds": 650
    }
  ]
}
```

### Health Check

```bash
GET /health

# Returns:
{
  "status": "healthy"
}
```

## Advanced Usage

### Custom Runtime Selection

```yaml
runtime:
  default: "auto"  # Let brain decide
  overrides:
    llm: "vllm"        # Force LLMs to vLLM
    vision: "tensorrt"  # Force vision to TensorRT
    text: "ray_serve"   # Force text models to Ray
```

### Orchestration Policies

```yaml
orchestration:
  enabled: true
  mode: "intelligent"  # or "simple" for rule-based only
  
  decision_interval_seconds: 10  # How often brain checks
  
  policies:
    # Eviction rules
    evict_after_idle_seconds: 300  # 5 minutes
    evict_when_memory_above_percent: 85
    
    # Never evict these models
    always_loaded: ["critical-model-v1", "main-llm"]
    
    # Priority settings
    priority_by_qps: true
    priority_custom:
      "team-a": 10  # Higher priority
      "team-b": 5
    
    # Loading behavior
    preload_on_first_request: true
    max_concurrent_loads: 2
```

### Workload Separation (Multi-Instance)

For high-traffic deployments, run separate instances:

```yaml
workload_separation:
  enabled: true
  instances:
    llm_instance:
      description: "Dedicated for LLMs"
      model_types: ["llm"]
      runtime: "vllm"
      gpu_count: 2
      port_offset: 0  # vLLM at 8000
    
    vision_instance:
      description: "Vision models"
      model_types: ["vision", "image"]
      runtime: "tensorrt"
      gpu_count: 1
      port_offset: 100  # TensorRT at 8100
```

### Fast Loading (Advanced Hardware)

```yaml
orchestration:
  fast_loading:
    enabled: true
    use_gpu_direct_storage: true  # Requires A100/H100 + NVMe
    nvme_cache_path: "/mnt/nvme/models"
    quantize_on_load: false  # FP32 → INT8 during load
```

### Multi-Tenancy & Rate Limiting

```yaml
# Per-organization limits
rate_limiting:
  enabled: true
  per_organization:
    enabled: true
    default_rpm: 1000
    overrides:
      "premium-org": 10000
      "enterprise-org": 50000

# Usage tracking for billing
usage_tracking:
  enabled: true
  track_inference_calls: true
  track_gpu_hours: true
  export_to: "prometheus"
```

See [configuration examples](../configs/README.md) for complete setups.

