# The Modelium Brain

## Overview

The Modelium Brain is a simple decision-making system that chooses the best runtime for each model.

## Current Implementation: Rule-Based

**Simple rules that work**:

```python
def choose_runtime(model_analysis):
    """Choose best runtime based on model type"""
    
    arch = model_analysis.architecture.lower()
    
    # LLM? Use vLLM (if enabled)
    if any(k in arch for k in ["gpt", "llama", "mistral", "qwen", "falcon"]):
        if vllm_enabled:
            return "vllm"
    
    # Ray for everything else (if enabled)
    if ray_enabled:
        return "ray"
    
    # Triton as fallback
    return "triton"
```

**That's it!** No complex AI, no training data, just simple rules.

## Why Rule-Based?

1. **Fast**: <1ms decision time
2. **Reliable**: No model download, no GPU memory needed
3. **Transparent**: Easy to understand and debug
4. **Sufficient**: 95% of cases are straightforward

## GPU Selection

**Simple strategy: Pick GPU with most free memory**

```python
def choose_gpu():
    """Pick GPU with lowest memory usage"""
    
    best_gpu = 0
    min_allocated = float('inf')
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)
        if allocated < min_allocated:
            min_allocated = allocated
            best_gpu = i
    
    return best_gpu
```

## Unload Decisions

**Simple policy: Unload if idle > 5 minutes**

```python
def should_unload(model):
    """Check if model should be unloaded"""
    
    # Get metrics
    idle_seconds = metrics.get_model_idle_seconds(model.name)
    qps = metrics.get_model_qps(model.name)
    
    # Simple rule
    threshold = 300  # 5 minutes
    if idle_seconds > threshold and qps < 0.1:
        return True
    
    # Check always-loaded list
    if model.name in config.always_loaded:
        return False
    
    return False
```

## Future: LLM-Powered Brain (Optional)

**Goal**: Fine-tuned small LLM for smarter decisions

**Model**: Qwen-2.5-1.5B-Instruct (2GB VRAM)

**Decisions**:
1. **Runtime Selection**: Consider model type, size, hardware
2. **GPU Packing**: Optimal model placement across GPUs
3. **Predictive Loading**: Load models before requests arrive
4. **Cost-Aware**: Balance latency vs GPU cost

**Training Data**:
- Model architectures + optimal runtimes
- Traffic patterns + load/unload decisions
- GPU states + packing strategies

**Status**: Planned (not required for core functionality)

**HuggingFace**: `modelium/brain-v1` (when ready)

## Configuration

```yaml
# Current (rule-based)
modelium_brain:
  enabled: true  # Uses rule-based logic
  fallback_to_rules: true

# Future (LLM-powered)
modelium_brain:
  enabled: true
  model_name: "modelium/brain-v1"  # Fine-tuned LLM
  device: "cuda:0"  # Needs GPU
  fallback_to_rules: true  # Still use rules if brain fails
```

## Why Not Start with LLM?

1. **Complexity**: Adds dependency on LLM, GPU memory, model download
2. **Not Required**: Rule-based works for 95% of cases
3. **Simplicity**: Keeping it simple is the goal
4. **Future**: Can add later without breaking anything

## Benefits of Current Approach

**For Users**:
- ✅ No model download
- ✅ No GPU memory for brain
- ✅ Instant decisions
- ✅ Easy to understand

**For Developers**:
- ✅ No training pipeline
- ✅ No LLM serving
- ✅ Simple to test
- ✅ Easy to debug

## When to Use LLM Brain?

**Consider LLM brain if**:
- You have 100+ models
- Complex multi-GPU setups
- Predictive loading needs
- Cost optimization required

**For most users**: Rule-based is perfect

## Example Decisions

```python
# GPT-2 → vLLM
model = "gpt2"
runtime = "vllm"  # Correct!

# ResNet → Ray
model = "resnet50"
runtime = "ray"  # Correct!

# BERT → vLLM (if enabled)
model = "bert-base"
runtime = "vllm"  # Works for transformers!

# Unknown → Ray (fallback)
model = "custom-model"
runtime = "ray"  # Safe default
```

## Summary

**Current Brain**:
- Rule-based
- Fast (<1ms)
- Reliable
- Simple

**Future Brain** (optional):
- LLM-powered
- Smarter decisions
- Predictive loading
- Cost-aware

**For 95% of users: Rule-based is perfect!**

The complexity is not needed. Simple rules work great.
