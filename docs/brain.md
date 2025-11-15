# The Modelium Brain

## Overview

The Modelium Brain is a fine-tuned Qwen-2.5-1.8B LLM that serves as the intelligent core of Modelium. It makes all deployment and orchestration decisions to maximize GPU utilization while maintaining low latency.

## One Model, Two Tasks

### Task 1: Conversion Planning
**When**: New model discovered  
**Purpose**: Decide how to deploy the model

**Input**:
```json
{
  "model_descriptor": {
    "name": "qwen-7b",
    "framework": "pytorch",
    "model_type": "causal_lm",
    "size_gb": 14
  },
  "available_gpus": 4,
  "gpu_memory": [78, 52, 65, 26]
}
```

**Output**:
```json
{
  "runtime": "vllm",
  "target_gpu": 1,
  "config": {"dtype": "float16"},
  "reasoning": "LLM detected, GPU 1 has most free memory",
  "confidence": 0.92
}
```

### Task 2: Orchestration
**When**: Every 10 seconds  
**Purpose**: Optimize GPU resource usage

**Input**:
```json
{
  "models_loaded": [
    {"name": "qwen-7b", "gpu": 1, "qps": 50, "idle_seconds": 0},
    {"name": "bert", "gpu": 2, "qps": 0, "idle_seconds": 610}
  ],
  "models_unloaded": [
    {"name": "mistral-7b", "pending_requests": 3}
  ]
}
```

**Output**:
```json
{
  "actions": [
    {
      "action": "evict",
      "model": "bert",
      "reasoning": "Idle 10min, 0 QPS"
    },
    {
      "action": "load",
      "model": "mistral-7b",
      "reasoning": "3 pending requests"
    }
  ]
}
```

## Model Details

- **Base Model**: Qwen-2.5-1.8B
- **Size**: ~2GB VRAM
- **Context**: 32K tokens
- **Fine-tuning**: Model analysis + orchestration scenarios
- **Location**: `modelium/brain-v1` (HuggingFace)

## Training Data

The brain is fine-tuned on:

1. **Model Analysis** (10K examples)
   - Various architectures (LLMs, vision, text)
   - Framework types (PyTorch, ONNX, TF)
   - Optimal runtime mappings

2. **Orchestration Scenarios** (50K examples)
   - Traffic patterns
   - GPU states
   - Optimal load/evict decisions

## Fallback Strategy

The brain always has a rule-based fallback:

```yaml
modelium_brain:
  fallback_to_rules: true  # Recommended for production
```

**Rule-based logic**:
- LLMs (>5GB) → vLLM
- Small PyTorch (<2GB) → TensorRT
- General models → Ray Serve
- Evict if idle >5min AND 0 QPS
- Load if pending requests exist

**Confidence**:
- LLM decisions: 0.85-0.95
- Rule-based: 0.60-0.75

## Usage

```python
from modelium import ModeliumBrain

# Initialize once
brain = ModeliumBrain(
    model_name="modelium/brain-v1",
    device="cuda:0",
    fallback_to_rules=True
)

# Task 1: Analyze new model
plan = brain.generate_conversion_plan(
    model_descriptor=model_info,
    available_gpus=4,
    gpu_memory=[78, 52, 65, 26]
)

# Task 2: Orchestrate resources
decision = brain.make_orchestration_decision(
    current_state=state,
    policies=policies
)
```

## Performance

- **Decision Time**: <500ms
- **Memory**: ~2GB VRAM
- **Accuracy**: >90% vs expert choices
- **Uptime**: 99.9% (with fallback)

## Future Improvements

- Continual learning from production data
- Multi-objective optimization
- Predictive preloading
- Cost-aware decisions

