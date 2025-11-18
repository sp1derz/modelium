"""
Orchestrator Service

Continuously optimizes GPU resources using the Modelium Brain.
"""

import time
import threading
import logging
from typing import Optional

from modelium.brain import ModeliumBrain
from modelium.services.model_registry import ModelRegistry, ModelStatus
from modelium.config import ModeliumConfig
from modelium.core.analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Intelligent orchestrator that uses the brain to manage GPU resources.
    
    Runs continuously in background, making decisions every N seconds.
    Connects to external runtimes (vLLM, Triton, Ray) via HTTP.
    """
    
    def __init__(
        self,
        brain: Optional[ModeliumBrain],
        connectors: Dict[str, Any],
        registry: ModelRegistry,
        config: ModeliumConfig,
    ):
        """
        Initialize orchestrator.
        
        Args:
            brain: Modelium brain for decision making
            connectors: Dict of runtime connectors {"vllm": VLLMConnector, ...}
            registry: Model registry for tracking models
            config: Modelium configuration
        """
        self.brain = brain
        self.connectors = connectors
        self.registry = registry
        self.config = config
        self.analyzer = HuggingFaceAnalyzer()
        
        self._running = False
        self._thread = None
        self._decision_interval = config.orchestration.decision_interval_seconds
    
    def start(self):
        """Start orchestration loop."""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self._thread.start()
        logger.info(f"🧠 Orchestrator started (decisions every {self._decision_interval}s)")
    
    def stop(self):
        """Stop orchestration loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Orchestrator stopped")
    
    def _orchestration_loop(self):
        """Main orchestration loop (runs in background thread)."""
        while self._running:
            try:
                self._make_decision()
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
            
            time.sleep(self._decision_interval)
    
    def _make_decision(self):
        """Make orchestration decision using the brain."""
        # Get current state
        loaded_models = self.registry.get_loaded_models()
        unloaded_models = self.registry.get_unloaded_models()
        
        # Skip if no models
        if not loaded_models and not unloaded_models:
            return
        
        # Build state for brain
        current_state = {
            "models_loaded": [
                {
                    "name": m.name,
                    "gpu": m.target_gpu or 0,
                    "qps": m.qps,
                    "idle_seconds": m.idle_seconds,
                }
                for m in loaded_models
            ],
            "models_unloaded": [
                {
                    "name": m.name,
                    "pending_requests": 0,  # TODO: Track pending requests
                }
                for m in unloaded_models
            ],
            "gpu_memory": self._get_gpu_memory_state(),
        }
        
        # Get policies
        policies = {
            "evict_after_idle_seconds": self.config.orchestration.policies.evict_after_idle_seconds,
            "always_loaded": self.config.orchestration.policies.always_loaded,
        }
        
        # Make decision
        try:
            decision = self.brain.make_orchestration_decision(
                current_state=current_state,
                policies=policies,
            )
            
            # Execute actions
            self._execute_actions(decision.get("actions", []))
            
        except Exception as e:
            logger.error(f"Error making orchestration decision: {e}")
    
    def _execute_actions(self, actions: list):
        """Execute orchestration actions."""
        for action in actions:
            action_type = action.get("action")
            model_name = action.get("model")
            
            try:
                if action_type == "load":
                    self._load_model(model_name, action.get("to_gpu", 0))
                elif action_type == "evict":
                    self._unload_model(model_name)
                elif action_type == "keep":
                    pass  # Do nothing, keep loaded
                    
            except Exception as e:
                logger.error(f"Error executing action {action_type} for {model_name}: {e}")
    
    def _load_model(self, model_name: str, gpu_id: int):
        """
        Mark model as loaded if it's available in the runtime.
        
        NOTE: Actual model loading happens in the runtime (vLLM/Triton/Ray).
        Users should start their runtime with models pre-loaded.
        This method just verifies the model is accessible and marks it as loaded.
        """
        model = self.registry.get_model(model_name)
        if not model:
            logger.error(f"Model {model_name} not found in registry")
            return
        
        if model.status == ModelStatus.LOADED:
            logger.info(f"{model_name} already loaded")
            return
        
        logger.info(f"🔼 Checking if {model_name} is loaded in {model.runtime}...")
        self.registry.update_model(model_name, status=ModelStatus.LOADING)
        
        # Check if model is available in runtime
        runtime = model.runtime
        if runtime not in self.connectors:
            logger.error(f"Runtime {runtime} not available")
            self.registry.update_model(
                model_name,
                status=ModelStatus.ERROR,
                error=f"Runtime {runtime} not connected",
            )
            return
        
        connector = self.connectors[runtime]
        
        # Check if model is loaded in runtime
        try:
            if runtime == "vllm":
                models = connector.list_models()
                if model_name in models or any(model_name in m for m in models):
                    self.registry.update_model(
                        model_name,
                        status=ModelStatus.LOADED,
                        target_gpu=gpu_id,
                        loaded_at=time.time(),
                    )
                    logger.info(f"   ✅ {model_name} available in vLLM")
                else:
                    self.registry.update_model(
                        model_name,
                        status=ModelStatus.ERROR,
                        error=f"Model not found in vLLM. Please start vLLM with: --model {model.path}",
                    )
                    logger.warning(f"   ⚠️  {model_name} not loaded in vLLM yet")
            
            elif runtime == "triton":
                if connector.get_model_ready(model_name):
                    self.registry.update_model(
                        model_name,
                        status=ModelStatus.LOADED,
                        target_gpu=gpu_id,
                        loaded_at=time.time(),
                    )
                    logger.info(f"   ✅ {model_name} ready in Triton")
                else:
                    # Try to load it via Triton's API
                    if connector.load_model(model_name):
                        self.registry.update_model(
                            model_name,
                            status=ModelStatus.LOADED,
                            target_gpu=gpu_id,
                            loaded_at=time.time(),
                        )
                        logger.info(f"   ✅ {model_name} loaded in Triton")
                    else:
                        self.registry.update_model(
                            model_name,
                            status=ModelStatus.ERROR,
                            error="Failed to load in Triton",
                        )
                        logger.error(f"   ❌ Failed to load {model_name} in Triton")
            
            elif runtime == "ray_serve":
                deployments = connector.list_deployments()
                if model_name in deployments:
                    self.registry.update_model(
                        model_name,
                        status=ModelStatus.LOADED,
                        target_gpu=gpu_id,
                        loaded_at=time.time(),
                    )
                    logger.info(f"   ✅ {model_name} deployed in Ray Serve")
                else:
                    self.registry.update_model(
                        model_name,
                        status=ModelStatus.ERROR,
                        error=f"Model not deployed in Ray Serve",
                    )
                    logger.warning(f"   ⚠️  {model_name} not deployed in Ray Serve yet")
                    
        except Exception as e:
            logger.error(f"Error checking {model_name}: {e}")
            self.registry.update_model(
                model_name,
                status=ModelStatus.ERROR,
                error=str(e),
            )
    
    def _unload_model(self, model_name: str):
        """
        Unload a model from runtime (if supported).
        
        NOTE: Not all runtimes support dynamic unloading.
        """
        model = self.registry.get_model(model_name)
        if not model or model.status != ModelStatus.LOADED:
            return
        
        runtime = model.runtime
        if runtime not in self.connectors:
            return
        
        connector = self.connectors[runtime]
        
        logger.info(f"🔽 Unloading {model_name} from {runtime}...")
        self.registry.update_model(model_name, status=ModelStatus.UNLOADING)
        
        try:
            # Only Triton supports easy unloading
            if runtime == "triton":
                if connector.unload_model(model_name):
                    self.registry.update_model(
                        model_name,
                        status=ModelStatus.UNLOADED,
                        unloaded_at=time.time(),
                    )
                    logger.info(f"   ✅ {model_name} unloaded")
                else:
                    self.registry.update_model(model_name, status=ModelStatus.ERROR)
            else:
                logger.warning(f"   Runtime {runtime} doesn't support dynamic unloading")
                # Mark as unloaded anyway (user needs to restart runtime)
                self.registry.update_model(
                    model_name,
                    status=ModelStatus.UNLOADED,
                    unloaded_at=time.time(),
                )
        except Exception as e:
            logger.error(f"Error unloading {model_name}: {e}")
            self.registry.update_model(model_name, status=ModelStatus.ERROR)
    
    def _get_gpu_memory_state(self) -> dict:
        """Get GPU memory state from actual GPUs."""
        import torch
        
        gpu_state = {}
        
        if not torch.cuda.is_available():
            # No GPU, return empty
            return {}
        
        for i in range(torch.cuda.device_count()):
            try:
                # Get memory in GB
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1e9
                
                # Get allocated/reserved memory
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                
                gpu_state[f"gpu_{i}"] = {
                    "used": reserved,  # Use reserved as "used"
                    "total": total,
                    "allocated": allocated,
                }
            except Exception as e:
                logger.error(f"Error getting GPU {i} memory: {e}")
                gpu_state[f"gpu_{i}"] = {"used": 0, "total": 0}
        
        return gpu_state
    
    def on_model_discovered(self, model_name: str, model_path: str):
        """
        Callback when a new model is discovered.
        
        Analyzes model and decides which runtime to use.
        """
        logger.info(f"📋 Analyzing {model_name}...")
        
        try:
            # Analyze HuggingFace model
            path = Path(model_path)
            if (path / "config.json").exists():
                analysis = self.analyzer.analyze(path)
                
                # Determine best runtime based on model type
                runtime = self._choose_runtime(analysis)
                
                # Get model size
                import os
                size_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(path)
                    for filename in filenames
                )
                
                # Register model with analysis results
                self.registry.register_model(
                    name=model_name,
                    path=str(path),
                    framework="pytorch" if analysis.architecture else "unknown",
                    model_type=analysis.model_type.value if analysis.model_type else "unknown",
                    runtime=runtime,
                    size_bytes=size_bytes,
                    parameters=analysis.resources.parameters if analysis.resources else 0,
                )
                
                logger.info(f"   Detected: {analysis.architecture or 'Unknown'}")
                logger.info(f"   Runtime: {runtime}")
                logger.info(f"   Size: {size_bytes / 1e9:.2f}GB")
                
                # Auto-load if enabled and runtime is available
                if self.config.deployment.auto_deploy and runtime in self.connectors:
                    self._load_model(model_name, 0)
            else:
                logger.warning(f"   No config.json found, skipping")
                
        except Exception as e:
            logger.error(f"Error analyzing {model_name}: {e}")
    
    def _choose_runtime(self, analysis) -> str:
        """
        Choose best runtime based on model analysis and available connectors.
        
        Args:
            analysis: HuggingFace analysis result
        
        Returns:
            Runtime name ("vllm", "triton", "ray_serve")
        """
        # Priority: vLLM for LLMs, Ray for general models, Triton as fallback
        arch = (analysis.architecture or "").lower()
        
        # LLM architectures - prefer vLLM
        if any(k in arch for k in ["gpt", "llama", "mistral", "qwen", "t5", "bert"]):
            if "vllm" in self.connectors:
                return "vllm"
        
        # Vision/other models - prefer Ray Serve
        if "ray_serve" in self.connectors:
            return "ray_serve"
        
        # Fallback to Triton
        if "triton" in self.connectors:
            return "triton"
        
        # Default to first available
        return list(self.connectors.keys())[0] if self.connectors else "vllm"

