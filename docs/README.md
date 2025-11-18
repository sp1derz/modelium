# Modelium Documentation

## Start Here

New to Modelium? Start with these docs in order:

1. **[Getting Started](getting-started.md)** - Install and run Modelium
2. **[Usage Guide](usage.md)** - All features and APIs
3. **[Architecture](architecture.md)** - How it works
4. **[Testing](testing.md)** - Verify everything works

## Quick Links

- **Installation**: [Getting Started](getting-started.md#installation)
- **First Model**: [Getting Started](getting-started.md#first-model)
- **API Endpoints**: [Usage Guide](usage.md#api-endpoints)
- **Configuration**: [Usage Guide](usage.md#configuration)
- **Troubleshooting**: [Testing](testing.md#troubleshooting-tests)

## Documentation Structure

```
docs/
├── README.md              ← You are here
├── getting-started.md     ← Installation & first steps
├── usage.md               ← Complete API reference
├── architecture.md        ← How Modelium works
├── brain.md               ← Decision-making system
└── testing.md             ← Testing & validation
```

## The 30-Second Overview

**What**: AI model serving platform that maximizes GPU utilization

**How**: Drop models → Auto-load → Track usage → Auto-unload when idle

**Why**: Maximum GPU efficiency with minimum user effort

## Quick Start

```bash
# 1. Install
git clone https://github.com/sp1derz/modelium
cd modelium
pip install -e ".[all]"

# 2. Configure
cp modelium.yaml.example modelium.yaml
nano modelium.yaml  # Enable vLLM

# 3. Start
python -m modelium.cli serve

# 4. Drop model
cp my-model models/incoming/

# 5. Use it
curl http://localhost:8000/predict/my-model \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

## Core Concepts

### Simple Flow

```
Drop Model → Watcher Detects → Analyzer Reads config.json → 
Brain Decides Runtime → Load to GPU → Inference Ready → 
Track Metrics → Unload if Idle → Repeat
```

### Three Key Features

1. **Auto-Detection**: Drop models, they load automatically
2. **Runtime Selection**: vLLM for LLMs, Ray for others, Triton fallback
3. **Smart Unloading**: Free GPU memory by unloading idle models

### Configuration

One file: `modelium.yaml`

```yaml
# What runtimes to use?
vllm:
  enabled: true

# Where to watch?
orchestration:
  model_discovery:
    watch_directories: ["/models/incoming"]

# When to unload?
  policies:
    evict_after_idle_seconds: 300  # 5 minutes
```

That's all you need!

## Key Files

- `modelium/runtime_manager.py` - Handles ALL runtimes (vLLM/Triton/Ray)
- `modelium/services/orchestrator.py` - Watches → Decides → Loads
- `modelium/services/model_watcher.py` - Monitors folder
- `modelium/services/model_registry.py` - Tracks models
- `modelium/brain/unified_brain.py` - Chooses runtime
- `modelium/metrics/prometheus_exporter.py` - Tracks everything

## Architecture at a Glance

```
User
  ↓ (drops model)
ModelWatcher
  ↓ (detects)
Orchestrator
  ↓ (analyzes)
Brain
  ↓ (decides)
RuntimeManager
  ↓ (loads)
vLLM/Triton/Ray
  ↓ (serves)
User gets inference!
```

## Support

- **Issues**: https://github.com/sp1derz/modelium/issues
- **Discussions**: https://github.com/sp1derz/modelium/discussions
- **Examples**: [../examples/](../examples/)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) (if it exists, otherwise open an issue!)

## License

See [LICENSE](../LICENSE)

---

**Remember: Simplicity is the goal. If something feels complex, it probably needs to be simpler.**

