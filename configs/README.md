# Modelium Configuration Examples

This directory contains example configurations for different deployment scenarios.

## Quick Start

1. **Copy the config that matches your use case:**
```bash
# For single server deployments
cp configs/single-instance.yaml modelium.yaml

# For dedicated LLM serving
cp configs/multi-instance-llms.yaml modelium.yaml

# For enterprise with multiple workload types
cp configs/enterprise-multi-workload.yaml modelium.yaml
```

2. **Edit the config:**
```bash
nano modelium.yaml
# Update organization.id, storage paths, etc.
```

3. **Use it:**
```bash
# Modelium automatically reads modelium.yaml from current directory
python -m modelium deploy

# Or specify config path
export MODELIUM_CONFIG=/path/to/your/config.yaml
python -m modelium deploy
```

---

## Configuration Files

### 1. `single-instance.yaml`
**Use case:** Small teams, startups, development

- **Setup:** 1 server with 1-2 GPUs
- **Runtimes:** Auto-select (vLLM for LLMs, Ray for others)
- **Scaling:** Up to 5 replicas per model
- **Perfect for:** Getting started, proof-of-concept

**Deploy:**
```bash
# On your single GPU server
cp configs/single-instance.yaml modelium.yaml
python -m modelium start
```

---

### 2. `multi-instance-llms.yaml`
**Use case:** Companies serving multiple LLMs

- **Setup:** 1 powerful server with 4-8 GPUs dedicated to LLMs
- **Runtime:** vLLM only (optimized for LLMs)
- **Features:** 
  - Tensor parallelism (2 GPUs per large model)
  - AWQ quantization
  - S3 storage
  - Usage tracking & billing
- **Perfect for:** LLM-focused businesses, chatbot platforms

**Deploy:**
```bash
# On your LLM GPU server (e.g., 4x A100)
cp configs/multi-instance-llms.yaml modelium.yaml
# Edit: Update organization ID and S3 bucket
python -m modelium start --workload llm
```

---

### 3. `multi-instance-vision.yaml`
**Use case:** Computer vision workloads

- **Setup:** 1 server with 2 GPUs dedicated to vision models
- **Runtime:** Ray Serve (better for high-throughput image processing)
- **Features:**
  - Auto-scaling up to 20 replicas
  - Optimized for batch image processing
- **Perfect for:** Image classification, object detection services

**Deploy:**
```bash
# On your vision GPU server (e.g., 2x A100)
cp configs/multi-instance-vision.yaml modelium.yaml
python -m modelium start --workload vision
```

---

### 4. `enterprise-multi-workload.yaml`
**Use case:** Large enterprises with diverse model types

- **Setup:** Multiple servers, each specialized for different workloads:
  - **gpu-llm-01:** 8 GPUs for LLMs
  - **gpu-vision-01:** 4 GPUs for vision models
  - **gpu-text-01:** 2 GPUs for BERT, embeddings
  - **gpu-audio-01:** 2 GPUs for audio models

- **Features:**
  - Workload routing (automatically routes models to the right instance)
  - Per-customer rate limiting
  - Enterprise security
  - Full usage tracking & billing
  - Multi-tenant support

**Deploy:**
```bash
# On each specialized server:

# LLM server (gpu-llm-01)
cp configs/enterprise-multi-workload.yaml modelium.yaml
python -m modelium start --instance llm_instance

# Vision server (gpu-vision-01)
cp configs/enterprise-multi-workload.yaml modelium.yaml
python -m modelium start --instance vision_instance

# Text server (gpu-text-01)
cp configs/enterprise-multi-workload.yaml modelium.yaml
python -m modelium start --instance text_instance

# Audio server (gpu-audio-01)
cp configs/enterprise-multi-workload.yaml modelium.yaml
python -m modelium start --instance audio_instance
```

---

## Configuration Sections Explained

### `organization`
Multi-tenant organization tracking:
```yaml
organization:
  id: "my-company"  # Used for billing/tracking
  name: "My Company"
  enable_usage_tracking: true
```

### `runtime`
Control which runtime to use:
```yaml
runtime:
  default: "auto"  # Let Modelium decide
  overrides:
    llm: "vllm"      # Force LLMs to use vLLM
    vision: "ray_serve"  # Vision models use Ray
```

### `workload_separation`
Split workloads across multiple servers (for high traffic):
```yaml
workload_separation:
  enabled: true
  instances:
    llm_instance:
      model_types: ["llm"]
      runtime: "vllm"
      gpu_count: 8
      port_offset: 0  # vLLM at port 8000
    
    vision_instance:
      model_types: ["vision"]
      runtime: "ray_serve"
      gpu_count: 4
      port_offset: 100  # Ray at port 8101
```

**How it works:**
1. Deploy `llm_instance` on server A (8 GPUs)
2. Deploy `vision_instance` on server B (4 GPUs)
3. When user drops an LLM → routes to server A
4. When user drops vision model → routes to server B
5. Each server scales independently!

### `rate_limiting`
Control API usage per organization:
```yaml
rate_limiting:
  per_organization:
    enabled: true
    default_rpm: 1000  # 1000 requests/minute default
    overrides:
      "premium-customer": 50000  # Give premium customers more
```

### `usage_tracking`
Track usage for billing:
```yaml
usage_tracking:
  enabled: true
  track_inference_calls: true
  track_gpu_hours: true
  export_to: "prometheus"
```

---

## Common Patterns

### Pattern 1: Start Simple, Scale Later

**Phase 1:** Single instance (1 server)
```bash
cp configs/single-instance.yaml modelium.yaml
```

**Phase 2:** Add dedicated LLM server (when LLM traffic increases)
```bash
# On new LLM server
cp configs/multi-instance-llms.yaml modelium.yaml
```

**Phase 3:** Add specialized servers (when you have diverse workloads)
```bash
# Use enterprise config and deploy to multiple servers
cp configs/enterprise-multi-workload.yaml modelium.yaml
```

### Pattern 2: Development vs Production

**Development:**
```yaml
# modelium-dev.yaml
deployment:
  environment: "development"
monitoring:
  logging:
    level: "DEBUG"
security:
  enable_sandbox: false  # Faster iteration
```

**Production:**
```yaml
# modelium-prod.yaml
deployment:
  environment: "production"
monitoring:
  logging:
    level: "INFO"
security:
  enable_sandbox: true
  enable_model_scanning: true
```

### Pattern 3: Multi-Cloud

Run different workloads on different clouds:

**AWS (LLMs):**
```yaml
# On AWS EC2 P5 instances
workload_separation:
  instances:
    llm_instance:
      model_types: ["llm"]
      gpu_count: 8
storage:
  backend: "s3"
```

**GCP (Vision):**
```yaml
# On GCP with A100s
workload_separation:
  instances:
    vision_instance:
      model_types: ["vision"]
      gpu_count: 4
storage:
  backend: "gcs"
```

---

## Testing Your Config

1. **Validate syntax:**
```bash
python -c "
from modelium.config import load_config
config = load_config('modelium.yaml')
print('✅ Config valid!')
print(f'Organization: {config.organization.id}')
print(f'Default runtime: {config.runtime.default}')
"
```

2. **Test runtime selection:**
```bash
python -c "
from modelium.config import get_config
config = get_config()
print(config.get_runtime_for_model('llm', 'my-org'))
# Output: vllm
print(config.get_runtime_for_model('vision', 'my-org'))
# Output: ray_serve
"
```

3. **Test port routing:**
```bash
python -c "
from modelium.config import get_config
config = get_config()
print(config.get_port_for_runtime('vllm', 'llm'))
# Output: 8000 (or 8000 + port_offset if workload separation enabled)
"
```

---

## Environment Variables

You can override config values with environment variables:

```bash
export MODELIUM_CONFIG=/path/to/config.yaml
export MODELIUM_ORGANIZATION_ID=my-override-org
export MODELIUM_GPU_ENABLED=true
export MODELIUM_VLLM_PORT=9000

python -m modelium start
```

---

## Tips

1. **Start with `single-instance.yaml`** - It's the simplest
2. **Use `auto` runtime** - Let Modelium choose (it's smart!)
3. **Enable usage tracking early** - You'll need it for billing
4. **Plan for scaling** - Design your org IDs to be scalable
5. **Test on cheap instances first** - Use g4dn.xlarge before p5.48xlarge

---

## Need Help?

See main README.md for:
- Complete architecture explanation
- API documentation
- Deployment guides
- Troubleshooting

---

**Your config is the heart of Modelium - configure once, scale forever!** 🚀

