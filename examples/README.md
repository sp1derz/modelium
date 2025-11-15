# Modelium Examples

This directory contains example scripts and workflows for using Modelium.

## Quick Start

### 1. Basic PyTorch Model

Deploy a simple CNN model:

```bash
python examples/quickstart.py
```

This example:
- Creates a simple PyTorch CNN
- Uploads it to Modelium
- Monitors the conversion
- Makes a test prediction

### 2. HuggingFace Models

Deploy models from HuggingFace Hub:

```bash
python examples/huggingface-model.py
```

This example deploys:
- BERT-base-uncased
- DistilBERT
- GPT-2 Small

### 3. Custom Conversion Plan

Use a custom conversion plan:

```bash
python examples/custom-plan.py
```

### 4. Batch Processing

Convert multiple models in parallel:

```bash
python examples/batch-convert.py models/
```

## Example Workflows

### Computer Vision

#### ResNet50 from torchvision

```python
import torch
import torchvision
import requests

# Load pretrained model
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# Save
torch.save(model, "resnet50.pt")

# Upload to Modelium
with open("resnet50.pt", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/models/upload",
        files={"file": f},
        data={"name": "resnet50", "framework": "pytorch"}
    )

model_id = response.json()["id"]
print(f"Model uploaded: {model_id}")
```

#### YOLO from Ultralytics

```python
from ultralytics import YOLO

# Export YOLO model to ONNX
model = YOLO("yolov8n.pt")
model.export(format="onnx")

# Upload ONNX model to Modelium
# Modelium will convert to TensorRT automatically
```

### Natural Language Processing

#### Fine-tuned BERT

```bash
# After fine-tuning your BERT model
python -m transformers.onnx \
  --model=./my-finetuned-bert \
  --feature=sequence-classification \
  ./bert-onnx/

# Upload to Modelium
curl -X POST http://localhost:8000/api/v1/models/upload \
  -F "file=@bert-onnx/model.onnx" \
  -F "name=my-bert" \
  -F "framework=onnx"
```

#### T5 for Translation

```python
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Save
model.save_pretrained("./t5-model")

# Upload directory to Modelium
# It will detect HuggingFace format automatically
```

### Large Language Models

#### Llama-2 7B

```python
import requests

# Deploy Llama-2 from HuggingFace
response = requests.post(
    "http://localhost:8000/api/v1/models/from-hub",
    json={
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "name": "llama2-7b",
        "deployment_config": {
            "precision": "fp16",
            "target_format": "trt_llm",  # Use TRT-LLM for best performance
            "gpu_count": 1,
        }
    }
)
```

#### Mistral 7B with 4-bit quantization

```python
response = requests.post(
    "http://localhost:8000/api/v1/models/from-hub",
    json={
        "repo_id": "mistralai/Mistral-7B-v0.1",
        "name": "mistral-7b",
        "deployment_config": {
            "precision": "int4",  # 4-bit quantization
            "target_format": "trt_llm",
            "max_batch_size": 32,
        }
    }
)
```

## Custom Conversion Plans

### Example: PyTorch → ONNX → TensorRT INT8

```json
{
  "target_format": "tensorrt_int8",
  "steps": [
    {
      "name": "export_onnx",
      "script": "import torch; torch.onnx.export(...)",
      "timeout": 600
    },
    {
      "name": "calibrate",
      "script": "# INT8 calibration code",
      "timeout": 1800
    },
    {
      "name": "convert_tensorrt",
      "command": "trtexec --onnx=model.onnx --saveEngine=model.plan --int8 --calib=cache.bin",
      "timeout": 3600
    }
  ],
  "triton_config": {
    "name": "my-model",
    "platform": "tensorrt_plan",
    "max_batch_size": 32
  }
}
```

## Monitoring Examples

### Get model status

```python
import requests

model_id = "model-abc123"
response = requests.get(f"http://localhost:8000/api/v1/models/{model_id}")
model = response.json()

print(f"Status: {model['status']}")
print(f"Created: {model['created_at']}")
print(f"Converted: {model.get('converted_at', 'N/A')}")
```

### Watch conversion logs

```python
import requests
import time

model_id = "model-abc123"

while True:
    response = requests.get(f"http://localhost:8000/api/v1/models/{model_id}/logs")
    logs = response.json()
    
    for log in logs["entries"]:
        print(f"[{log['timestamp']}] {log['message']}")
    
    if logs["status"] in ["deployed", "failed"]:
        break
    
    time.sleep(5)
```

### Performance metrics

```python
# Get inference metrics
response = requests.get(
    f"http://localhost:8000/api/v1/models/{model_id}/metrics"
)

metrics = response.json()
print(f"Throughput: {metrics['throughput_qps']} QPS")
print(f"Latency P50: {metrics['latency_p50_ms']} ms")
print(f"Latency P95: {metrics['latency_p95_ms']} ms")
print(f"GPU Utilization: {metrics['gpu_utilization']}%")
```

## Testing

Run example tests:

```bash
# Run all examples
pytest examples/test_examples.py

# Run specific example
pytest examples/test_examples.py::test_quickstart
```

## Troubleshooting

### Connection refused

Make sure Modelium is running:
```bash
docker-compose ps
```

### Model conversion failed

Check logs:
```bash
curl http://localhost:8000/api/v1/models/{model_id}/logs
```

### Deployment timeout

Increase timeout for large models:
```python
deployment_config = {
    "timeout": 3600,  # 1 hour
}
```

## More Examples

See the [documentation](../docs/) for more examples:
- [User Guide](../docs/user-guide.md)
- [API Reference](../docs/api-reference.md)
- [Advanced Workflows](../docs/advanced-workflows.md)

