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
        """Check loaded models and unload idle ones."""
        loaded_models = self.registry.get_loaded_models()
        
        for model in loaded_models:
            # Get metrics
            idle_seconds = self.metrics.get_model_idle_seconds(model.name, model.runtime)
            qps = self.metrics.get_model_qps(model.name, model.runtime)
            
            # Should unload?
            threshold = self.config.orchestration.policies.evict_after_idle_seconds
            min_qps = 0.1
            
            if idle_seconds > threshold and qps < min_qps:
                # Check if it's in always_loaded list
                always_loaded = self.config.orchestration.policies.always_loaded
                if model.name not in always_loaded:
                    logger.info(f"🔽 Unloading idle model: {model.name} (idle: {idle_seconds:.0f}s, QPS: {qps:.2f})")
                    self.runtime_manager.unload_model(model.name)
                    self.registry.update_model(model.name, status=ModelStatus.UNLOADED)
                    self.metrics.record_model_unload(model.runtime, "success")
                    self.metrics.record_orchestration_decision("unload", "idle_timeout")
    
    def on_model_discovered(self, model_name: str, model_path: str):
        """
        Called when watcher discovers a new model.
        
        This is THE CORE METHOD - everything happens here!
        
        Args:
            model_name: Name of discovered model
            model_path: Path to model directory
        """
        logger.info(f"📋 New model discovered: {model_name}")
        
        try:
            path = Path(model_path)
            
            # 1. ANALYZE: Read config.json
            if not (path / "config.json").exists():
                logger.warning(f"   ⚠️  No config.json found, skipping")
                return
            
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
            runtime = self._choose_runtime(analysis)
            logger.info(f"   🎯 Brain decision: {runtime}")
            
            # Register in registry
            self.registry.register_model(
                name=model_name,
                path=str(path),
                framework="pytorch",
                model_type=analysis.model_type.value if analysis.model_type else "unknown",
                runtime=runtime,
                size_bytes=size_bytes,
                parameters=analysis.resources.parameters if analysis.resources else 0,
            )
            
            # 3. LOAD: Use RuntimeManager to actually load it
            logger.info(f"   Loading model...")
            self.registry.update_model(model_name, status=ModelStatus.LOADING)
            
            gpu_id = self._choose_gpu()
            success = self.runtime_manager.load_model(
                model_name=model_name,
                model_path=path,
                runtime=runtime,
                gpu_id=gpu_id
            )
            
            if success:
                logger.info(f"   ✅ {model_name} loaded successfully!")
                self.registry.update_model(
                    model_name,
                    status=ModelStatus.LOADED,
                    target_gpu=gpu_id,
                    loaded_at=time.time()
                )
                self.metrics.record_model_load(runtime, "success")
                self.metrics.record_orchestration_decision("load", f"new_model_{runtime}")
            else:
                logger.error(f"   ❌ Failed to load {model_name}")
                self.registry.update_model(model_name, status=ModelStatus.ERROR)
                self.metrics.record_model_load(runtime, "error")
                
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            self.registry.update_model(model_name, status=ModelStatus.ERROR)
    
    def _choose_runtime(self, analysis) -> str:
        """
        Choose best runtime based on model analysis and enabled runtimes.
        
        Simple rules:
        - LLMs (GPT, Llama, Mistral, etc.) → vLLM if enabled
        - Vision/Other → Ray if enabled
        - Fallback to whatever is enabled
        """
        arch = (analysis.architecture or "").lower()
        
        # Check what's enabled
        enabled_runtimes = []
        if self.config.vllm.enabled:
            enabled_runtimes.append("vllm")
        if self.config.triton.enabled:
            enabled_runtimes.append("triton")
        if self.config.ray_serve.enabled:
            enabled_runtimes.append("ray")
        
        # LLM architectures
        llm_keywords = ["gpt", "llama", "mistral", "qwen", "falcon", "bloom", "opt"]
        is_llm = any(k in arch for k in llm_keywords)
        
        if is_llm and "vllm" in enabled_runtimes:
            return "vllm"
        
        if "ray" in enabled_runtimes:
            return "ray"
        
        if "triton" in enabled_runtimes:
            return "triton"
        
        # Default to first enabled
        return enabled_runtimes[0] if enabled_runtimes else "vllm"
    
    def _choose_gpu(self) -> int:
        """Choose GPU with most free memory."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            
            # Simple: Choose GPU with lowest utilization
            best_gpu = 0
            min_allocated = float('inf')
            
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                if allocated < min_allocated:
                    min_allocated = allocated
                    best_gpu = i
            
            return best_gpu
        except:
            return 0
