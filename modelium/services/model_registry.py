"""
Model Registry

Tracks all discovered and loaded models in the system.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    DISCOVERED = "discovered"
    ANALYZING = "analyzing"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a model in the registry."""
    
    # Identity
    name: str
    path: str
    
    # Status
    status: ModelStatus = ModelStatus.DISCOVERED
    error: Optional[str] = None
    
    # Model metadata (from analyzer)
    framework: Optional[str] = None
    model_type: Optional[str] = None
    size_bytes: int = 0
    parameters: int = 0
    
    # Deployment info (from brain)
    runtime: Optional[str] = None
    target_gpu: Optional[int] = None
    port: Optional[int] = None
    
    # Runtime metrics
    qps: float = 0.0
    latency_p99: float = 0.0
    idle_seconds: float = 0.0
    last_request_time: float = 0.0
    total_requests: int = 0
    
    # Timestamps
    discovered_at: float = field(default_factory=time.time)
    loaded_at: Optional[float] = None
    unloaded_at: Optional[float] = None


class ModelRegistry:
    """
    Central registry for all models in the system.
    
    Thread-safe singleton that tracks:
    - Discovered models
    - Loaded models
    - Model metadata and metrics
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._models: Dict[str, ModelInfo] = {}
            self._lock = threading.RLock()
            self._initialized = True
    
    def register_model(self, name: str, path: str) -> ModelInfo:
        """Register a newly discovered model."""
        with self._lock:
            if name in self._models:
                return self._models[name]
            
            model = ModelInfo(name=name, path=path)
            self._models[name] = model
            return model
    
    def update_model(self, name: str, **kwargs) -> Optional[ModelInfo]:
        """Update model information."""
        with self._lock:
            if name not in self._models:
                return None
            
            model = self._models[name]
            for key, value in kwargs.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            return model
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model by name."""
        with self._lock:
            return self._models.get(name)
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelInfo]:
        """List all models, optionally filtered by status."""
        with self._lock:
            models = list(self._models.values())
            if status:
                models = [m for m in models if m.status == status]
            return models
    
    def get_loaded_models(self) -> List[ModelInfo]:
        """Get all loaded models."""
        return self.list_models(status=ModelStatus.LOADED)
    
    def get_unloaded_models(self) -> List[ModelInfo]:
        """Get all unloaded models."""
        return self.list_models(status=ModelStatus.UNLOADED)
    
    def update_metrics(self, name: str, qps: float = None, latency_p99: float = None):
        """Update model runtime metrics."""
        with self._lock:
            if name not in self._models:
                return
            
            model = self._models[name]
            if qps is not None:
                model.qps = qps
            if latency_p99 is not None:
                model.latency_p99 = latency_p99
            
            # Update idle time
            if model.last_request_time > 0:
                model.idle_seconds = time.time() - model.last_request_time
    
    def record_request(self, name: str):
        """Record an inference request for a model."""
        with self._lock:
            if name not in self._models:
                return
            
            model = self._models[name]
            model.last_request_time = time.time()
            model.total_requests += 1
            model.idle_seconds = 0.0
    
    def get_stats(self) -> Dict:
        """Get overall registry statistics."""
        with self._lock:
            return {
                "total_models": len(self._models),
                "loaded": len(self.get_loaded_models()),
                "unloaded": len(self.get_unloaded_models()),
                "discovering": len(self.list_models(ModelStatus.DISCOVERING)),
                "loading": len(self.list_models(ModelStatus.LOADING)),
                "errors": len(self.list_models(ModelStatus.ERROR)),
            }

