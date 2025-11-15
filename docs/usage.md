# Usage Guide

## Basic Usage

### 1. Start Modelium

```bash
modelium serve --config modelium.yaml
```

### 2. Drop Models

Copy models to the watched directory:

```bash
cp your_model.pt /models/incoming/
```

### 3. Use Models

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/your_model",
    json={
        "input": "your data",
        "organizationId": "your-org"
    }
)
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

{
  "total_gpus": 4,
  "models_loaded": 6,
  "gpu_utilization": "72%"
}
```

### Metrics

```bash
GET /metrics  # Prometheus format
```

## Advanced Usage

### Custom Runtime Selection

```yaml
runtime:
  default: "auto"
  overrides:
    llm: "vllm"
    vision: "tensorrt"
```

### Workload Separation

```yaml
workload_separation:
  enabled: true
  instances:
    llm_instance:
      model_types: ["llm"]
      gpu_count: 2
```

See [configuration examples](../configs/README.md) for more.

