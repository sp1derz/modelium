"""
Orchestrator Service - SIMPLIFIED

Watches folder → Analyzes model → Brain decides → Loads model → Monitors → Unloads idle
"""

import time
import threading
import logging
import os
from typing import Optional
from pathlib import Path

from modelium.brain import ModeliumBrain
from modelium.services.model_registry import ModelRegistry, ModelStatus
from modelium.config import ModeliumConfig
from modelium.core.analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from modelium.runtime_manager import RuntimeManager
from modelium.metrics import ModeliumMetrics

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Simple orchestrator: Watches → Analyzes → Decides → Loads → Monitors.
    
    Flow:
        1. User drops model in /models/incoming/
        2. Watcher calls on_model_discovered()
        3. Analyze config.json
        4. Brain decides which runtime (vLLM/Triton/Ray)
        5. RuntimeManager loads it
        6. Metrics track usage
        7. Unload if idle too long
    """
    
    def __init__(
        self,
        brain: Optional[ModeliumBrain],
        runtime_manager: RuntimeManager,
        registry: ModelRegistry,
        metrics: ModeliumMetrics,
        config: ModeliumConfig,
    ):
        """
        Initialize orchestrator.
        
        Args:
            brain: Modelium brain for decision making
            runtime_manager: Runtime manager for loading models
            registry: Model registry for tracking
            metrics: Prometheus metrics
            config: Configuration
        """
        self.brain = brain
        self.runtime_manager = runtime_manager
        self.registry = registry
        self.metrics = metrics
        self.config = config
        self.analyzer = HuggingFaceAnalyzer()
        
        self._running = False
        self._thread = None
        self._decision_interval = config.orchestration.decision_interval_seconds
        
        logger.info("Orchestrator initialized")
    
    def start(self):
        """Start orchestration loop."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self._thread.start()
        logger.info(f"🧠 Orchestrator started (checks every {self._decision_interval}s)")
    
    def stop(self):
        """Stop orchestration loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _orchestration_loop(self):
        """Background loop: Check for idle models and unload them."""
        while self._running:
            try:
                self._check_for_idle_models()
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
            
            time.sleep(self._decision_interval)
    
    def _check_for_idle_models(self):
        """
        INTELLIGENT orchestration decisions.
        
        Uses the Brain (Qwen LLM) if available, otherwise falls back to rules.
        
        Considers:
        1. Policies (always_loaded, idle threshold)
        2. Prometheus metrics (QPS, latency, idle time)
        3. Current state (GPU memory, what's running)
        
        Makes smart decisions about:
        - Keep actively used models (even low QPS)
        - Unload truly idle models
        - Respect GPU memory pressure
        - Never unload if pending requests
        """
        loaded_models = self.registry.get_loaded_models()
        
        if not loaded_models:
            return
        
        # Get GPU memory state
        gpu_memory_pressure = self._get_gpu_memory_pressure()
        
        # Get policies
        policies = self.config.orchestration.policies
        idle_threshold = policies.evict_after_idle_seconds
        always_loaded = policies.always_loaded
        
        # Use Brain (Qwen LLM) for intelligent decisions - MANDATORY
        if self.brain is None:
            logger.error("❌ FATAL: Brain is None! Orchestration cannot proceed.")
            raise RuntimeError("Brain is required but not initialized")
        
        if self.brain.model is None:
            logger.error("❌ FATAL: Brain model is None! Orchestration cannot proceed.")
            raise RuntimeError("Brain model is required but not loaded")
        
        # Brain is available - use it (MANDATORY)
        logger.debug("🧠 Using Brain (Qwen) for orchestration decision...")
        
        # Build current state for brain (ONLY relevant Prometheus metrics)
           # We only send what the brain needs, not everything
        # CRITICAL: Enforce grace period - don't even consider models for eviction if within grace period
        grace_period = 120  # 120 seconds grace period for newly loaded models
        min_idle_for_eviction = 180  # 3 minutes of zero QPS before eviction
        
        models_data = []
        for m in loaded_models:
            # Get relevant metrics from Prometheus
            # Pass GPU ID to ensure we get QPM for the specific model on its specific GPU
            # This is critical when multiple models are on different GPUs
            # Use QPM (queries per minute) instead of QPS for more stable orchestration decisions
            model_gpu = m.target_gpu if hasattr(m, 'target_gpu') and m.target_gpu is not None else None
            qpm = self.metrics.get_model_qpm(m.name, m.runtime, gpu=model_gpu)
            qps = qpm / 60.0  # Convert QPM to QPS for display/compatibility
            idle_seconds = self.metrics.get_model_idle_seconds(m.name, m.runtime)
            
            # Calculate time since load
            time_since_load = None
            if m.loaded_at:
                time_since_load = time.time() - m.loaded_at
            
            # Check if model is within grace period
            within_grace_period = time_since_load is not None and time_since_load < grace_period
            
            # Check if model meets eviction criteria (QPM=0 AND idle >= min_idle AND grace period passed)
            # CRITICAL: All three conditions must be true:
            # 1. QPM must be exactly 0.0 (no active traffic in last 60s)
            # 2. Idle time must be >= 180s (3 minutes of inactivity)
            # 3. Grace period must have passed (time_since_load >= 120s)
            # Use QPM instead of QPS for more stable decisions (60s window vs 10s)
            can_evict = (
                qpm == 0.0 and 
                idle_seconds >= min_idle_for_eviction and 
                not within_grace_period and
                time_since_load is not None and
                time_since_load >= grace_period
            )
            
            model_data = {
                "name": m.name,
                "runtime": m.runtime,
                "gpu": m.target_gpu if hasattr(m, 'target_gpu') else None,
                "qps": qps,  # Converted from QPM (qpm/60.0) for display
                "qpm": qpm,  # From Prometheus: modelium_model_qpm (stable, for orchestration)
                "idle_seconds": idle_seconds,  # From Prometheus: modelium_model_idle_seconds
                "loaded_at": m.loaded_at,
                "time_since_load_seconds": time_since_load,
                "within_grace_period": within_grace_period,  # NEW: Tell brain about grace period
                "can_evict": can_evict,  # NEW: Pre-calculated eviction eligibility
            }
            models_data.append(model_data)
            
            # Only log detailed metrics at DEBUG level (too verbose for INFO)
            if within_grace_period:
                logger.debug(f"   📊 {m.name}: QPM={qpm:.2f}, idle={idle_seconds:.1f}s, since_load={time_since_load:.1f}s [GRACE PERIOD]")
            elif can_evict:
                logger.debug(f"   📊 {m.name}: QPM={qpm:.2f}, idle={idle_seconds:.1f}s, since_load={time_since_load:.1f}s [ELIGIBLE FOR EVICTION]")
            else:
                logger.debug(f"   📊 {m.name}: QPM={qpm:.2f}, idle={idle_seconds:.1f}s, since_load={time_since_load:.1f}s")
        
        current_state = {
            "models_loaded": models_data,
            "gpu_memory_pressure": gpu_memory_pressure,  # From PyTorch/nvidia-smi
            "total_gpus": self.config.gpu.count if self.config.gpu.count else 1,
        }
        
        # Get policies dict
        policies_dict = {
            "evict_after_idle_seconds": idle_threshold,
            "always_loaded": always_loaded,
            "evict_when_memory_above_percent": policies.evict_when_memory_above_percent,
        }
        
        # Log concise summary at INFO level, detailed data at DEBUG
        logger.info(f"📤 Brain decision: {len(models_data)} models loaded, GPU pressure: {gpu_memory_pressure}")
        logger.debug("📤 Sending to Brain (Qwen):")
        logger.debug(f"   Models: {len(models_data)} loaded")
        for m in models_data:
            logger.debug(f"      - {m['name']}: QPM={m.get('qpm', 0):.2f} (QPS={m['qps']:.2f}), idle={m['idle_seconds']:.1f}s, GPU={m['gpu']}, since_load={m.get('time_since_load_seconds', 0):.1f}s")
        logger.debug(f"   GPU memory pressure: {gpu_memory_pressure}")
        logger.debug(f"   Policies: evict_after_idle={idle_threshold}s, always_loaded={always_loaded}")
        
        # Ask brain for decisions (MANDATORY - no fallback)
        try:
            brain_decision = self.brain.make_orchestration_decision(current_state, policies_dict)
        except Exception as e:
            logger.error(f"❌ Brain decision failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Brain is mandatory - don't fallback, raise error
            raise RuntimeError(f"Brain orchestration failed: {e}")
        
        if brain_decision and "actions" in brain_decision:
            # Decision summary already logged by brain, just log count at DEBUG
            logger.debug(f"🧠 Brain returned {len(brain_decision.get('actions', []))} actions")
            
            # Get list of actual model names for validation
            actual_model_names = {m.name for m in loaded_models}
            
            # Execute brain's decisions
            for action in brain_decision.get("actions", []):
                action_type = action.get("action")
                model_name = action.get("model")
                reasoning = action.get("reasoning", "")
                
                # Validate model exists (prevent brain from hallucinating models)
                if model_name and model_name not in actual_model_names:
                    logger.warning(f"🧠 Brain suggested action for non-existent model '{model_name}' - ignoring")
                    logger.warning(f"   Available models: {list(actual_model_names)}")
                    continue
                
                if action_type == "evict" and model_name:
                    # PRE-FILTER: Check can_evict from models_data before doing expensive validation
                    model_data = next((m for m in models_data if m.get("name") == model_name), None)
                    if model_data and not model_data.get("can_evict", False):
                        logger.warning(f"🧠 Brain suggested evicting '{model_name}' but can_evict=false - IGNORING (pre-filter)")
                        logger.warning(f"   Model data: QPS={model_data.get('qps', 0):.2f}, idle={model_data.get('idle_seconds', 0):.1f}s, within_grace={model_data.get('within_grace_period', False)}")
                        logger.warning(f"   Brain should have used 'keep' action for this model")
                        continue
                    # CRITICAL: Double-check eviction eligibility before executing
                    model_info = self.registry.get_model(model_name)
                    if not model_info:
                        logger.warning(f"🧠 Brain suggested evicting '{model_name}' but model not found in registry")
                        continue
                    
                    # Check grace period
                    if model_info.loaded_at:
                        time_since_load = time.time() - model_info.loaded_at
                        if time_since_load < grace_period:
                            logger.warning(f"🧠 Brain suggested evicting '{model_name}' but it's within grace period ({time_since_load:.1f}s < {grace_period}s) - IGNORING")
                            logger.warning(f"   Model must be loaded for at least {grace_period}s before eviction")
                            continue
                    
                    # Check QPS and idle time
                    qps = self.metrics.get_model_qps(model_name, model_info.runtime)
                    idle_seconds = self.metrics.get_model_idle_seconds(model_name, model_info.runtime)
                    
                    if qps > 0.0:
                        logger.warning(f"🧠 Brain suggested evicting '{model_name}' but QPS={qps:.2f} > 0 - IGNORING (model is active)")
                        continue
                    
                    if idle_seconds < min_idle_for_eviction:
                        logger.warning(f"🧠 Brain suggested evicting '{model_name}' but idle={idle_seconds:.1f}s < {min_idle_for_eviction}s - IGNORING (too recent)")
                        continue
                    
                    logger.info(f"🧠 Brain decision: Unload {model_name} - {reasoning}")
                    logger.info(f"   ✅ Eviction validated: QPS={qps:.2f}, idle={idle_seconds:.1f}s, since_load={time_since_load:.1f}s")
                    
                    # Get runtime before unloading (for metrics)
                    runtime = model_info.runtime
                    
                    success = self.runtime_manager.unload_model(model_name)
                    if success:
                        self.registry.update_model(model_name, status=ModelStatus.UNLOADED)
                        self.metrics.record_model_unload(runtime, "success")
                        self.metrics.record_orchestration_decision("unload", f"brain_{reasoning}")
                        logger.info(f"   ✅ {model_name} unloaded successfully (removed from RuntimeManager)")
                    else:
                        logger.error(f"   ❌ Failed to unload {model_name} (may already be unloaded)")
                        self.metrics.record_model_unload(runtime, "error")
                elif action_type == "keep" and model_name:
                    logger.debug(f"🧠 Brain decision: Keep {model_name} - {reasoning}")
                elif action_type == "load" and model_name:
                    logger.warning(f"🧠 Brain suggested loading '{model_name}' - load actions handled by on_model_discovered")
                    # Load actions are handled when models are discovered, not here
                # Note: "load" actions are handled by on_model_discovered
        else:
            logger.warning(f"🧠 Brain returned invalid decision (no actions)")
            raise RuntimeError("Brain decision returned no actions")
    
    def _get_gpu_memory_pressure(self) -> bool:
        """
        Check if GPUs are under memory pressure.
        
        Returns True if any GPU is > 85% full (default threshold).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            
            threshold = self.config.orchestration.policies.evict_when_memory_above_percent / 100.0
            
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                
                if allocated / total > threshold:
                    logger.debug(f"GPU {i} under pressure: {allocated/total*100:.1f}% used")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return False  # Assume no pressure if we can't check
    
    def on_model_discovered(self, model_name: str, model_path: str):
        """
        Called when watcher discovers a new model.
        
        This is THE CORE METHOD - everything happens here!
        
        Args:
            model_name: Name of discovered model
            model_path: Path to model directory
        """
        logger.info(f"")
        logger.info(f"=" * 60)
        logger.info(f"📋 ORCHESTRATOR CALLBACK: {model_name}")
        logger.info(f"   Path: {model_path}")
        logger.info(f"   Orchestrator callback triggered!")
        logger.info(f"=" * 60)
        
        try:
            path = Path(model_path).resolve()  # Resolve to absolute path
            logger.info(f"   ✅ Resolved path: {path}")
            logger.info(f"   Path exists: {path.exists()}")
            
            # 1. ANALYZE: Read config.json
            config_path = path / "config.json"
            if not config_path.exists():
                logger.warning(f"   ⚠️  No config.json found at {config_path}, skipping")
                return
            
            logger.info(f"   ✅ Found config.json at {config_path}")
            
            logger.info(f"   Analyzing model...")
            analysis = self.analyzer.analyze(path)
            
            if not analysis.architecture:
                logger.warning(f"   ⚠️  Could not determine architecture")
                return
            
            logger.info(f"   Architecture: {analysis.architecture}")
            logger.info(f"   Type: {analysis.model_type.value if analysis.model_type else 'unknown'}")
            
            # Get size
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(path)
                for filename in filenames
            )
            logger.info(f"   Size: {size_bytes / 1e9:.2f}GB")
            
            # 2. BRAIN DECIDES: Which runtime?
            runtime = self._choose_runtime(analysis, model_name=model_name)
            logger.info(f"   🎯 Brain decision: {runtime}")
            
            # Register in registry (if not already registered by watcher)
            existing_model = self.registry.get_model(model_name)
            if not existing_model:
                self.registry.register_model(
                    name=model_name,
                    path=str(path),
                )
            
            # Update with analysis results
            self.registry.update_model(
                model_name,
                framework="pytorch",
                model_type=analysis.model_type.value if analysis.model_type else "unknown",
                runtime=runtime,
                size_bytes=size_bytes,
                parameters=analysis.resources.parameters if analysis.resources else 0,
            )
            
            # 3. LOAD: Use RuntimeManager to actually load it
            logger.info(f"   🚀 Starting model load...")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Path: {path}")
            logger.info(f"   Runtime: {runtime}")
            
            self.registry.update_model(model_name, status=ModelStatus.LOADING)
            
            gpu_id = self._choose_gpu()
            logger.info(f"   Selected GPU: {gpu_id}")
            
            # Set GPU immediately (even before loading) so it shows in API
            self.registry.update_model(model_name, target_gpu=gpu_id)
            logger.info(f"   ✅ GPU {gpu_id} assigned to {model_name} (will load now)")
            
            try:
                logger.info(f"   📞 Calling runtime_manager.load_model()...")
                logger.info(f"   📋 Load parameters:")
                logger.info(f"      - model_name: {model_name}")
                logger.info(f"      - model_path: {path}")
                logger.info(f"      - runtime: {runtime}")
                logger.info(f"      - gpu_id: {gpu_id}")
                logger.info(f"      - path.exists(): {path.exists()}")
                
                success = self.runtime_manager.load_model(
                    model_name=model_name,
                    model_path=path,
                    runtime=runtime,
                    gpu_id=gpu_id
                )
                
                logger.info(f"   📞 load_model() returned: {success}")
                
                if success:
                    logger.info(f"   ✅ {model_name} loaded successfully on GPU {gpu_id}!")
                    self.registry.update_model(
                        model_name,
                        status=ModelStatus.LOADED,
                        target_gpu=gpu_id,  # Ensure it's set
                        loaded_at=time.time()
                    )
                    logger.info(f"   ✅ Registry updated: {model_name} -> LOADED")
                    self.metrics.record_model_load(runtime, "success")
                    self.metrics.record_orchestration_decision("load", f"new_model_{runtime}")
                else:
                    logger.error(f"   ❌ load_model() returned False for {model_name}")
                    logger.error(f"   Model path: {path}")
                    logger.error(f"   Path exists: {path.exists()}")
                    logger.error(f"   Runtime: {runtime}")
                    logger.error(f"   GPU: {gpu_id}")
                    logger.error(f"   ⚠️  Check RuntimeManager logs above for detailed error messages")
                    logger.error(f"   ⚠️  Check modelium.log for full traceback")
                    # Keep GPU assigned even on error (for debugging)
                    self.registry.update_model(
                        model_name, 
                        status=ModelStatus.ERROR,
                        target_gpu=gpu_id,  # Keep GPU visible
                        error=f"Failed to load with {runtime} (check logs for details)"
                    )
                    logger.error(f"   ❌ Registry updated: {model_name} -> ERROR")
                    self.metrics.record_model_load(runtime, "error")
            except Exception as load_error:
                logger.error(f"   ❌ Exception during model load: {load_error}", exc_info=True)
                self.registry.update_model(
                    model_name, 
                    status=ModelStatus.ERROR,
                    error=f"Exception: {str(load_error)}"
                )
                self.metrics.record_model_load(runtime, "error")
                
        except Exception as e:
            logger.error(f"❌ Error processing {model_name}: {e}", exc_info=True)
            logger.error(f"   Model path: {model_path}")
            logger.error(f"   Full traceback above")
            self.registry.update_model(
                model_name, 
                status=ModelStatus.ERROR,
                error=str(e)
            )
    
    def _choose_runtime(self, analysis, model_name: str = None) -> str:
        """
        Choose best runtime based on model analysis and enabled runtimes.
        
        Simple rules:
        - GPT-2 → Ray Serve (vLLM doesn't support GPT-2 OpenAI API)
        - Modern LLMs (Llama, Mistral, Qwen, etc.) → vLLM if enabled
        - Vision/Other → Ray if enabled
        - Fallback to whatever is enabled
        """
        arch = (analysis.architecture or "").lower()
        model_name_lower = (model_name or "").lower()
        
        # Check what's enabled
        enabled_runtimes = []
        if self.config.vllm.enabled:
            enabled_runtimes.append("vllm")
        if self.config.triton.enabled:
            enabled_runtimes.append("triton")
        if self.config.ray_serve.enabled:
            enabled_runtimes.append("ray")
        
        # CRITICAL: GPT-2 doesn't work with vLLM OpenAI API endpoints
        # Route GPT-2 to Ray Serve instead
        is_gpt2 = (
            "gpt2" in arch or 
            "gpt2" in model_name_lower or
            arch == "gpt2" or
            model_name_lower == "gpt2"
        )
        
        if is_gpt2:
            if "ray" in enabled_runtimes:
                logger.info(f"   🎯 GPT-2 detected → Routing to Ray Serve (vLLM doesn't support GPT-2 OpenAI API)")
                return "ray"
            elif "triton" in enabled_runtimes:
                logger.info(f"   🎯 GPT-2 detected → Routing to Triton (vLLM doesn't support GPT-2 OpenAI API)")
                return "triton"
            else:
                logger.warning(f"   ⚠️  GPT-2 detected but Ray/Triton not enabled. vLLM may not work!")
        
        # Modern LLM architectures (but NOT GPT-2)
        llm_keywords = ["llama", "mistral", "qwen", "falcon", "bloom", "opt", "gpt-3", "gpt-4", "gpt3", "gpt4"]
        is_modern_llm = any(k in arch for k in llm_keywords)
        
        if is_modern_llm and "vllm" in enabled_runtimes:
            return "vllm"
        
        if "ray" in enabled_runtimes:
            return "ray"
        
        if "triton" in enabled_runtimes:
            return "triton"
        
        # Default to first enabled
        return enabled_runtimes[0] if enabled_runtimes else "vllm"
    
    def _choose_gpu(self) -> int:
        """Choose GPU with most free memory, avoiding GPUs already in use."""
        # Get GPUs already in use by loaded models
        loaded_models = self.registry.get_loaded_models()
        used_gpus = set()
        for m in loaded_models:
            if hasattr(m, 'target_gpu') and m.target_gpu is not None:
                used_gpus.add(m.target_gpu)
                logger.debug(f"   GPU {m.target_gpu} is in use by {m.name}")
        
        # Try PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"   🔍 PyTorch detected {gpu_count} GPU(s)")
                logger.info(f"   GPUs in use: {list(used_gpus) if used_gpus else 'none'}")
                
                if gpu_count > 0:
                    # Prefer unused GPUs, then choose GPU with lowest utilization
                    best_gpu = 0
                    min_allocated = float('inf')
                    found_unused = False
                    
                    # First pass: Find unused GPU with lowest utilization
                    for i in range(gpu_count):
                        if i not in used_gpus:
                            allocated = torch.cuda.memory_allocated(i)
                            reserved = torch.cuda.memory_reserved(i)
                            logger.debug(f"   GPU {i} (unused): {allocated / 1e9:.2f}GB allocated, {reserved / 1e9:.2f}GB reserved")
                            if allocated < min_allocated:
                                min_allocated = allocated
                                best_gpu = i
                                found_unused = True
                    
                    # If no unused GPU found, use the one with lowest utilization
                    if not found_unused:
                        logger.warning(f"   ⚠️  All GPUs are in use, choosing GPU with lowest utilization")
                        min_allocated = float('inf')
                        for i in range(gpu_count):
                            allocated = torch.cuda.memory_allocated(i)
                            reserved = torch.cuda.memory_reserved(i)
                            logger.debug(f"   GPU {i}: {allocated / 1e9:.2f}GB allocated, {reserved / 1e9:.2f}GB reserved")
                            if allocated < min_allocated:
                                min_allocated = allocated
                                best_gpu = i
                    
                    if best_gpu in used_gpus:
                        logger.warning(f"   ⚠️  Selected GPU {best_gpu} is already in use (may cause conflicts)")
                    else:
                        logger.info(f"   ✅ Selected GPU {best_gpu} (unused, lowest utilization: {min_allocated / 1e9:.2f}GB)")
                    return best_gpu
        except Exception as e:
            logger.warning(f"   ⚠️  PyTorch GPU detection failed: {e}")
        
        # Fallback: Use nvidia-smi if PyTorch fails
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_count = len(result.stdout.strip().split('\n'))
                if gpu_count > 0:
                    logger.info(f"   🔍 nvidia-smi detected {gpu_count} GPU(s) (PyTorch CUDA unavailable)")
                    logger.info(f"   ✅ Selected GPU 0 via nvidia-smi fallback")
                    return 0
        except Exception as e:
            logger.warning(f"   ⚠️  nvidia-smi fallback failed: {e}")
        
        # Last resort: Return 0 (assume GPU 0 exists)
        logger.warning("   ⚠️  Could not detect GPUs, assuming GPU 0 exists")
        return 0
