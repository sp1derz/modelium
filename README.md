# Modelium 🧠

**AI-Powered Model Serving Platform with Intelligent GPU Orchestration**

Modelium is an open-source library that automatically discovers, analyzes, and deploys ML models with maximum GPU utilization. Just drop your models in a folder, and let the AI brain handle everything.

## ✨ Key Features

- 🤖 **AI Brain**: Fine-tuned LLM makes intelligent deployment decisions
- 🔄 **Auto-Discovery**: Drop models in a folder, automatic deployment
- 🚀 **Multi-Runtime**: Supports vLLM, Ray Serve, TensorRT, Triton
- 📊 **Smart Orchestration**: Dynamic model loading/unloading for max GPU utilization
- ⚡ **Fast Swapping**: GPUDirect Storage support for rapid model loading
- 🎯 **Zero Config**: Minimal configuration, maximum automation
- 🔒 **Multi-Tenant**: Built-in organization tracking and usage monitoring

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourorg/modelium.git
cd modelium

# Install dependencies
pip install -e .

# Or with specific runtimes
pip install -e ".[vllm]"      # For LLMs
pip install -e ".[ray]"        # For general models
pip install -e ".[all]"        # Everything
```

### Usage

```bash
# 1. Initialize configuration
modelium init

# 2. Start the server
modelium serve --config modelium.yaml

# 3. Drop your models
cp your_model.pt /models/incoming/

# That's it! Modelium handles the rest.
```

### Make Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/your_model",
    json={"input": "your data", "organizationId": "your-org"}
)
```

## 🧠 How It Works

1. **Model Discovery**: Watches directories for new models
2. **Analysis**: Extracts metadata (size, type, architecture)
3. **AI Decision**: Brain chooses optimal runtime and GPU
4. **Deployment**: Automatically deploys with best configuration
5. **Orchestration**: Dynamically manages GPU resources based on traffic

```
Drop Model → Analyze → Brain Decides → Deploy → Serve
                ↓
         Continuous optimization every 10s
```

## 📊 Example Scenario

**Setup**: 4 GPUs, 10 models, varying traffic patterns

**Without Modelium**: 20% GPU utilization, manual configuration  
**With Modelium**: 70% GPU utilization, zero manual work

The AI brain:
- Keeps high-traffic models loaded (qwen-7b: 50 QPS)
- Evicts idle models (bert: 0 QPS for 10 minutes)
- Loads models on-demand (mistral-7b: 3 pending requests)
- Optimizes GPU packing (small models share GPUs)

## 📚 Documentation

- [Getting Started](docs/getting-started.md) - Detailed setup guide
- [Architecture](docs/architecture.md) - System design and components
- [The Brain](docs/brain.md) - How the AI decision-making works
- [Usage Guide](docs/usage.md) - Complete user guide with examples
- [Testing Guide](docs/testing.md) - How to test your deployment

## 🔧 Configuration

Edit `modelium.yaml`:

```yaml
# Minimal config
organization:
  id: "my-company"

modelium_brain:
  enabled: true
  fallback_to_rules: true

orchestration:
  enabled: true
  model_discovery:
    watch_directories: ["/models/incoming"]

gpu:
  enabled: true
  count: 4
```

See [configuration examples](configs/) for more.

## 🎯 Use Cases

- **AI Startups**: Maximize GPU ROI, serve multiple models efficiently
- **ML Teams**: Zero-config model deployment, focus on models not ops
- **Research Labs**: Dynamic resource allocation, experiment freely
- **Enterprises**: Multi-tenant serving, usage tracking, cost optimization

## 🤝 Contributing

We welcome contributions! Areas of interest:

- Fine-tuning the brain on more diverse workloads
- Adding support for new runtimes
- Improving fast loading strategies
- Documentation and examples

## 📝 License

Apache-2.0 License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [Ray Serve](https://docs.ray.io/en/latest/serve/) - Scalable model serving
- [TensorRT](https://developer.nvidia.com/tensorrt) - High-performance inference
- [Qwen](https://github.com/QwenLM/Qwen) - Base model for the brain

---

**Status**: Active development | **Version**: 0.1.0 | **Python**: 3.9+

Star ⭐ this repo if you find it useful!
