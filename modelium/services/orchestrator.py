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
from modelium.services.vllm_service import VLLMService
from modelium.config import ModeliumConfig

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Intelligent orchestrator that uses the brain to manage GPU resources.
    
    Runs continuously in background, making decisions every N seconds.
    """
    
    def __init__(
        self,
        brain: Optional[ModeliumBrain],
        vllm_service: VLLMService,
        config: ModeliumConfig,
    ):
        """
        Initialize orchestrator.
        
        Args:
            brain: Modelium brain for decision making
            vllm_service: vLLM service for model management
            config: Modelium configuration
        """
        self.brain = brain
        self.vllm_service = vllm_service
        self.config = config
        self.registry = ModelRegistry()
        
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
        """Load a model using vLLM."""
        model = self.registry.get_model(model_name)
        if not model:
            logger.error(f"Model {model_name} not found in registry")
            return
        
        if model.status == ModelStatus.LOADED:
            logger.info(f"{model_name} already loaded")
            return
        
        logger.info(f"🔼 Loading {model_name} to GPU {gpu_id}...")
        self.registry.update_model(model_name, status=ModelStatus.LOADING)
        
        # Load with vLLM
        result = self.vllm_service.load_model(
            model_name=model_name,
            model_path=model.path,
            gpu_id=gpu_id,
        )
        
        if result.get("status") == "loaded":
            self.registry.update_model(
                model_name,
                status=ModelStatus.LOADED,
                target_gpu=gpu_id,
                port=result.get("port"),
                loaded_at=time.time(),
            )
            logger.info(f"   ✅ {model_name} loaded")
        else:
            self.registry.update_model(
                model_name,
                status=ModelStatus.ERROR,
                error=result.get("error"),
            )
            logger.error(f"   ❌ Failed to load {model_name}")
    
    def _unload_model(self, model_name: str):
        """Unload a model."""
        model = self.registry.get_model(model_name)
        if not model or model.status != ModelStatus.LOADED:
            return
        
        logger.info(f"🔽 Unloading {model_name}...")
        self.registry.update_model(model_name, status=ModelStatus.UNLOADING)
        
        if self.vllm_service.unload_model(model_name):
            self.registry.update_model(
                model_name,
                status=ModelStatus.UNLOADED,
                unloaded_at=time.time(),
            )
        else:
            self.registry.update_model(model_name, status=ModelStatus.ERROR)
    
    def _get_gpu_memory_state(self) -> dict:
        """Get GPU memory state."""
        # TODO: Query actual GPU memory from CUDA
        # For now, return dummy data
        gpu_count = self.config.gpu.count or 4
        return {
            f"gpu_{i}": {"used": 0, "total": 80}
            for i in range(gpu_count)
        }
    
    def on_model_discovered(self, model_name: str, model_path: str):
        """
        Callback when a new model is discovered.
        
        Triggers brain to decide if/how to load it.
        """
        logger.info(f"📋 Planning deployment for {model_name}...")
        
        model = self.registry.get_model(model_name)
        if not model:
            return
        
        # Get model descriptor
        model_descriptor = {
            "name": model_name,
            "framework": model.framework or "unknown",
            "model_type": model.model_type or "unknown",
            "parameters": model.parameters,
            "resources": {"memory_bytes": model.size_bytes},
        }
        
        # Ask brain for deployment plan
        try:
            plan = self.brain.generate_conversion_plan(
                model_descriptor=model_descriptor,
                available_gpus=self.config.gpu.count or 4,
                gpu_memory=[70, 75, 78, 80],  # TODO: Query actual GPU memory
            )
            
            # Update model with plan
            self.registry.update_model(
                model_name,
                runtime=plan.get("runtime"),
                target_gpu=plan.get("target_gpu"),
            )
            
            logger.info(f"   Plan: {plan.get('runtime')} on GPU {plan.get('target_gpu')}")
            
            # Auto-load if enabled
            if self.config.deployment.auto_deploy:
                self._load_model(model_name, plan.get("target_gpu", 0))
                
        except Exception as e:
            logger.error(f"Error planning deployment for {model_name}: {e}")

