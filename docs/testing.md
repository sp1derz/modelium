# Testing Guide

## Quick Test

### 1. Create Test Model

```python
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

model = TestModel()
torch.save(model, "test_model.pt")
```

### 2. Drop Model

```bash
cp test_model.pt /models/incoming/
```

### 3. Check Logs

```bash
# Watch for:
# 🔍 Analyzing test_model.pt...
# 🧠 Brain decision: runtime=ray_serve, gpu=0
# ✅ test_model ready at http://localhost:8000/predict/test_model
```

### 4. Make Request

```bash
curl -X POST http://localhost:8000/predict/test_model \
  -H "Content-Type: application/json" \
  -d '{"input": [[1,2,3,4,5,6,7,8,9,10]], "organizationId": "test"}'
```

## Run Examples

```bash
# Brain demo
python examples/brain_demo.py

# Quickstart
python examples/quickstart.py

# Real deployment test
python examples/real_deployment_test.py
```

## Troubleshooting

### Model Not Detected

Check watch directory:
```bash
grep "watch_directories" modelium.yaml
ls -la /models/incoming/
```

### GPU Not Found

Test GPU availability:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

### Brain Fails to Load

Check fallback:
```yaml
modelium_brain:
  fallback_to_rules: true  # Should be enabled
```

## Monitoring

View metrics:
```bash
curl http://localhost:9090/metrics | grep modelium
```

Check status:
```bash
curl http://localhost:8000/status | jq .
```

