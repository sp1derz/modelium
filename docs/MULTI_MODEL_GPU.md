# Multiple Models on Same GPU - Architecture

## Can Multiple Models Share a GPU?

**YES!** This is actually **desirable** for maximum GPU utilization. Modelium supports packing multiple models on the same GPU.

## How It Works

### 1. **QPS Tracking is Per-Model (Not Per-GPU)**

Prometheus metrics use the `model` label to track QPS independently:

```prometheus
modelium_model_qps{model="gpt2",runtime="ray",gpu="1"} = 1.87
modelium_model_qps{model="gpt2-small",runtime="ray",gpu="1"} = 0.5
modelium_model_qps{model="gpt2-medium",runtime="ray",gpu="2"} = 0.05
```

**Key Point**: Even if `gpt2` and `gpt2-small` are both on GPU 1, they have **separate QPS metrics**. The `model` label ensures each model is tracked independently.

### 2. **GPU Selection Logic**

The `_choose_gpu()` function:
1. **First**: Tries to find an unused GPU (preferred)
2. **Fallback**: If all GPUs are in use, chooses GPU with lowest memory utilization
3. **Result**: Multiple models can end up on the same GPU

```python
# From orchestrator.py
if not found_unused:
    logger.warning("⚠️  All GPUs are in use, choosing GPU with lowest utilization")
    # Chooses GPU with least memory used
```

### 3. **Brain Eviction Decisions**

The brain receives **per-model metrics**, not per-GPU:

```json
{
  "models_loaded": [
    {
      "name": "gpt2",
      "gpu": 1,
      "qps": 1.87,
      "idle_seconds": 146.6
    },
    {
      "name": "gpt2-small",
      "gpu": 1,  // Same GPU!
      "qps": 0.0,
      "idle_seconds": 300.0
    }
  ],
  "gpu_memory_pressure": false
}
```

**The brain can evict individual models** based on:
- **QPS per model** (not GPU total)
- **Idle time per model**
- **GPU memory pressure** (affects all models on that GPU)

### 4. **Eviction Scenarios**

#### Scenario A: Multiple Models on Same GPU, One is Idle

```
GPU 1:
  - gpt2: QPS=1.87, idle=10s → KEEP (active)
  - gpt2-small: QPS=0.0, idle=300s → EVICT (idle)
```

**Result**: Brain evicts `gpt2-small`, `gpt2` stays on GPU 1.

#### Scenario B: GPU Memory Pressure

```
GPU 1:
  - gpt2: QPS=0.5, idle=200s
  - gpt2-small: QPS=0.0, idle=400s
  - gpu_memory_pressure: true
```

**Result**: Brain evicts the most idle model (`gpt2-small`) to free memory.

#### Scenario C: All Models Active

```
GPU 1:
  - gpt2: QPS=2.0, idle=5s → KEEP
  - gpt2-small: QPS=1.5, idle=8s → KEEP
```

**Result**: Both models stay loaded (high utilization).

## Current Limitations

### 1. **No Model Size in Brain Prompt**

The brain doesn't know model memory footprint, so it can't make optimal packing decisions.

**Future Enhancement**: Add model memory size to brain prompt:
```json
{
  "name": "gpt2",
  "gpu": 1,
  "qps": 1.87,
  "memory_gb": 0.5,  // NEW
  "gpu_total_memory_gb": 40.0  // NEW
}
```

### 2. **No Memory Check Before Packing**

`_choose_gpu()` doesn't verify if a GPU has enough free memory before assigning a model.

**Future Enhancement**: Check available memory:
```python
def _choose_gpu(self, model_memory_gb: float) -> int:
    for gpu_id in range(gpu_count):
        free_memory = get_free_memory(gpu_id)
        if free_memory >= model_memory_gb:
            return gpu_id  # Can fit!
    # No GPU has space, evict something first
```

### 3. **GPU Memory Pressure is Binary**

Currently, `gpu_memory_pressure` is just `true/false`. The brain doesn't know:
- How much memory is used per GPU
- Which models are using the most memory
- How much memory would be freed by evicting a model

**Future Enhancement**: Send detailed GPU memory metrics:
```json
{
  "gpu_memory": [
    {
      "gpu": 1,
      "used_gb": 35.0,
      "total_gb": 40.0,
      "utilization_percent": 87.5
    }
  ]
}
```

## Best Practices

1. **Small Models**: Pack together on same GPU (e.g., GPT-2 variants)
2. **Large LLMs**: Isolate on dedicated GPU (e.g., Llama-70B)
3. **Mixed Workloads**: Balance active models across GPUs
4. **Memory Monitoring**: Use DCGM exporter for detailed GPU metrics

## Example: 5 Models on 2 GPUs

```
GPU 0:
  - gpt2: QPS=2.0, memory=0.5GB → KEEP
  - gpt2-small: QPS=0.0, memory=0.3GB, idle=400s → EVICT
  - bert-base: QPS=1.5, memory=0.4GB → KEEP

GPU 1:
  - llama-7b: QPS=0.8, memory=14GB → KEEP
  - qwen-1.5b: QPS=0.0, memory=3GB, idle=500s → EVICT
```

**Brain Decision**:
- Evict `gpt2-small` (idle, low memory)
- Evict `qwen-1.5b` (idle, frees 3GB)
- Keep others (active traffic)

## Conclusion

✅ **Multiple models CAN share a GPU** - This is supported and desirable  
✅ **QPS is tracked per-model** - Each model has independent metrics  
✅ **Brain evicts per-model** - Not per-GPU, based on individual QPS/idle  
⚠️ **Memory-aware packing** - Needs enhancement (future work)

The current implementation works correctly for per-model eviction, but could be enhanced with memory-aware packing for optimal GPU utilization.

