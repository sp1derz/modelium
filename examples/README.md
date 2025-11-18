# Modelium Examples

## Simple Usage (START HERE)

The most important example:

```bash
python 01_simple_usage.py
```

This walkthrough shows you:
- How to configure Modelium
- How to drop models
- How inference works
- How metrics are tracked
- How idle models are auto-unloaded

## Real Usage

After understanding the basics, just:

### 1. Configure
```yaml
# modelium.yaml
vllm:
  enabled: true
```

### 2. Start Server
```bash
python -m modelium.cli serve
```

### 3. Drop Model
```bash
cp -r my-model /models/incoming/
```

### 4. Use It
```bash
curl http://localhost:8000/predict/my-model \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

## That's It!

No complex API, no configuration files per model, no manual loading.

**Drop model → It loads → You use it → It unloads when idle**

Maximum GPU utilization with minimum effort.
