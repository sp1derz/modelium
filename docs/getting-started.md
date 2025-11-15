# Getting Started with Modelium

## Prerequisites

- Python 3.9+
- NVIDIA GPU (optional, can run on CPU)
- 8GB+ RAM
- Linux or macOS

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourorg/modelium.git
cd modelium
```

### Step 2: Install Dependencies

**Basic installation:**
```bash
pip install -e .
```

**With specific runtimes:**
```bash
# For LLM support (vLLM)
pip install -e ".[vllm]"

# For general models (Ray Serve)
pip install -e ".[ray]"

# Everything (all runtimes)
pip install -e ".[all]"
```

### Step 3: Initialize Configuration

```bash
modelium init
```

This creates `modelium.yaml` with default settings.

## Quick Test

### 1. Start the Server

```bash
modelium serve --config modelium.yaml
```

You should see:
```
🧠 Modelium Brain loading...
✅ Brain loaded (rule-based mode)
👁️  Watching /models/incoming for new models
🚀 Server running on http://0.0.0.0:8000
```

### 2. Drop a Model

Copy any PyTorch model to the watched directory:

```bash
cp your_model.pt /models/incoming/
```

Modelium will automatically:
- Detect the model
- Analyze it
- Choose the best runtime
- Deploy it
- Make it available via API

### 3. Make a Request

```bash
curl -X POST http://localhost:8000/predict/your_model \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "organizationId": "test-org"}'
```

## What's Next?

- Read [Architecture](architecture.md) to understand how it works
- Check [The Brain](brain.md) to learn about AI orchestration
- See [Usage Guide](usage.md) for complete examples
- Explore [Configuration](../configs/README.md) for advanced setups

## Common Issues

**Issue**: `modelium: command not found`  
**Fix**: Use `python -m modelium.cli serve` instead

**Issue**: Models not detected  
**Fix**: Check `watch_directories` in `modelium.yaml`

**Issue**: GPU not found  
**Fix**: Set `gpu.enabled: false` for CPU-only mode

For more help, see [Testing Guide](testing.md).

