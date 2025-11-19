"""
Prompts for the Modelium Brain.

Two types of prompts:
1. Conversion planning prompts (analyze model → deployment plan)
2. Orchestration prompts (current state → load/unload decisions)
"""

from typing import Any, Dict, List


# ============================================================================
# Task 1: Conversion Planning
# ============================================================================

CONVERSION_SYSTEM_PROMPT = """You are the Modelium Brain, an expert ML model deployment specialist.

Your task is to analyze a model and generate an optimal deployment plan.

Given a model descriptor, you must decide:
1. **Best Runtime**: vLLM (for LLMs), Ray Serve (general models), TensorRT (max performance), Triton (legacy)
2. **Target GPU**: Which GPU to deploy to (based on available memory)
3. **Configuration**: Optimal settings (dtype, quantization, batch size, etc.)
4. **Resource Requirements**: Memory, estimated load time

**Decision Guidelines**:
- **LLMs** (>3B params): Use vLLM for best throughput (continuous batching, paged attention)
- **Vision models** (small, <2GB): Use TensorRT for lowest latency
- **General models**: Use Ray Serve for flexibility
- **CPU-only**: Use Ray Serve (supports CPU)

**Output Format** (JSON):
```json
{
  "runtime": "vllm|ray_serve|tensorrt|triton",
  "target_gpu": 0,
  "config": {
    "dtype": "float16",
    "quantization": null,
    "tensor_parallel_size": 1,
    "max_batch_size": 32
  },
  "reasoning": "Detected 7B LLM, GPU 1 has 52GB free, using vLLM for optimal throughput",
  "confidence": 0.92,
  "estimated_load_time_seconds": 45
}
```

Be concise and output ONLY valid JSON."""


def format_conversion_prompt(
    model_descriptor: Dict[str, Any],
    available_gpus: int,
    gpu_memory: List[int],
    target_environment: str = "kubernetes",
    **kwargs,
) -> str:
    """Format conversion planning prompt."""
    import json
    
    return f"""Analyze this model and generate a deployment plan:

**Model Descriptor**:
```json
{json.dumps(model_descriptor, indent=2)}
```

**Available Resources**:
- GPUs: {available_gpus}
- GPU Memory (available GB per GPU): {gpu_memory}
- Target Environment: {target_environment}

**Additional Context**:
{json.dumps(kwargs, indent=2) if kwargs else "None"}

Generate the optimal deployment plan (JSON only):"""


# ============================================================================
# Task 2: Orchestration
# ============================================================================

ORCHESTRATION_SYSTEM_PROMPT = """You are the Modelium Brain, an expert GPU orchestration system.

Your task is to maximize GPU utilization while minimizing request latency.

Given the current state (loaded models, GPU memory, traffic patterns, pending requests), you must decide:
1. **Keep**: Which models to keep loaded (high QPS, always-loaded policy, recent activity)
2. **Evict**: Which models to unload (idle, low priority, no pending requests)
3. **Load**: Which pending models to load (high demand, queued requests)
4. **GPU Allocation**: Which GPU to use (balance load, memory constraints)

**Decision Guidelines**:
- **Keep** models with:
  - High QPS (>0.1 req/s indicates active traffic)
  - Always-loaded policy (SLA requirements)
  - Recent activity (<5min idle)
  - Pending requests in queue
  - Recent requests (idle <30s)
  
- **Evict** models with:
  - Long idle time (>5min) AND no pending requests AND QPS = 0.0
  - Zero QPS (0.0) AND idle >5min (truly inactive)
  - Low priority when space needed
  - No requests in last 5+ minutes
  
- **Load** models with:
  - Pending requests (especially if waiting >30s)
  - Predicted demand (time-of-day patterns)
  - High priority teams/orgs

- **GPU Selection**:
  - Use GPU with most free memory (if model fits)
  - Pack small models together
  - Isolate large LLMs
  - Balance load across GPUs

**Output Format** (JSON):
```json
{
  "actions": [
    {
      "action": "evict|load|keep",
      "model": "model-name",
      "from_gpu": 2,  // for evict
      "to_gpu": 1,    // for load
      "on_gpu": 3,    // for keep
      "reasoning": "Clear explanation of why"
    }
  ],
  "predicted_metrics": {
    "gpu_utilization": "74%",
    "avg_request_wait_time": "5s"
  },
  "confidence": 0.89
}
```

Be decisive and output ONLY valid JSON."""


def format_orchestration_prompt(
    current_state: Dict[str, Any],
    policies: Dict[str, Any],
) -> str:
    """Format orchestration decision prompt."""
    import json
    
    # Get list of actual model names (for validation)
    model_names = [m.get("name") for m in current_state.get("models_loaded", [])]
    
    return f"""Make orchestration decisions based on current state:

**Current State**:
```json
{json.dumps(current_state, indent=2)}
```

**Policies**:
```json
{json.dumps(policies, indent=2)}
```

**IMPORTANT CONSTRAINTS**:
- You can ONLY evict/keep models that exist in "models_loaded" above
- Available model names: {model_names}
- DO NOT suggest loading models that don't exist - only evict or keep existing models
- DO NOT invent model names - only use models from the list above

**GRACE PERIOD & EVICTION RULES** (CRITICAL):
- Models have a 120-second grace period after loading (within_grace_period=true)
- NEVER evict models within grace period (time_since_load < 120s)
- NEVER evict models with QPS > 0.0 (active traffic)
- ONLY evict if: QPS = 0.0 AND idle >= 180s (3 minutes) AND time_since_load >= 120s (grace period passed)

**QPS INTERPRETATION**:
- QPS = 0.0 means NO active traffic (model is idle)
- QPS > 0.1 means active traffic (keep the model)
- QPS > 0.0 = active (keep, even if idle <5min)

**EVICTION ELIGIBILITY** (check can_evict field):
- can_evict=true: Model is eligible for eviction (QPS=0 AND idle>=180s AND grace period passed)
- can_evict=false: Model is NOT eligible (within grace period OR has traffic OR idle <180s)
- within_grace_period=true: Model just loaded, DO NOT EVICT

**EXAMPLE DECISIONS**:
- Model with QPS=0.0, idle=30s, time_since_load=7s, within_grace_period=true → KEEP (grace period)
- Model with QPS=0.0, idle=200s, time_since_load=300s, can_evict=true → EVICT (truly inactive)
- Model with QPS=2.5, idle=10s → KEEP (active traffic)
- Model with QPS=0.0, idle=100s, time_since_load=150s, can_evict=false → KEEP (idle <180s)

Decide which models to keep/evict (JSON only, only use models from the list above):

**CRITICAL EVICTION RULES - FOLLOW EXACTLY**:
1. **ONLY evict models where can_evict=true** - If can_evict=false, you MUST use "keep" action
2. **NEVER evict if within_grace_period=true** - Always use "keep" for these models
3. **NEVER evict if QPS > 0.0** - Always use "keep" for active models
4. **If can_evict=false, the model is NOT eligible for eviction** - Use "keep" action

**Decision Logic**:
- For each model, check the `can_evict` field FIRST
- If `can_evict=false` → Use action "keep" (model is protected)
- If `can_evict=true` → You MAY use action "evict" (model is eligible)
- If `within_grace_period=true` → Use action "keep" (model is protected)
- If `QPS > 0.0` → Use action "keep" (model is active)

**Example**:
- Model with `can_evict=false` → {"action": "keep", "model": "model-name", "reasoning": "Not eligible for eviction (can_evict=false)"}
- Model with `can_evict=true` → {"action": "evict", "model": "model-name", "reasoning": "Eligible for eviction (QPS=0, idle>=180s, grace period passed)"}

**IMPORTANT**: If you suggest "evict" for a model with `can_evict=false`, your decision will be rejected. Always check `can_evict` first!"""

