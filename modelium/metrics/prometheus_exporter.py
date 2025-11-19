"""
Prometheus Metrics Exporter

Collects and exposes metrics for Modelium's orchestration decisions.
"""

import logging
import time
from typing import Dict, List, Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)


class ModeliumMetrics:
    """
    Prometheus metrics for Modelium orchestration.
    
    Exposes metrics that the Brain uses for intelligent decisions:
    - GPU utilization and memory
    - Model QPS (queries per second)
    - Model latency
    - Model idle time
    - Load/unload events
    
    Usage:
        metrics = ModeliumMetrics()
        metrics.start_server(port=9090)
        
        # Record metrics
        metrics.record_request("gpt2", latency_ms=150)
        metrics.update_gpu_memory(gpu_id=0, used_gb=15.2, total_gb=40.0)
    """
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Request metrics
        self.requests_total = Counter(
            'modelium_requests_total',
            'Total inference requests',
            ['model', 'runtime', 'status']
        )
        
        self.request_latency = Histogram(
            'modelium_request_latency_seconds',
            'Request latency in seconds',
            ['model', 'runtime'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Model metrics
        self.model_qps = Gauge(
            'modelium_model_qps',
            'Queries per second for each model',
            ['model', 'runtime', 'gpu']
        )
        
        self.model_idle_seconds = Gauge(
            'modelium_model_idle_seconds',
            'Seconds since last request for each model',
            ['model', 'runtime']
        )
        
        self.models_loaded = Gauge(
            'modelium_models_loaded',
            'Number of currently loaded models',
            ['runtime']
        )
        
        # GPU metrics
        self.gpu_memory_used_gb = Gauge(
            'modelium_gpu_memory_used_gb',
            'GPU memory used in GB',
            ['gpu']
        )
        
        self.gpu_memory_total_gb = Gauge(
            'modelium_gpu_memory_total_gb',
            'GPU total memory in GB',
            ['gpu']
        )
        
        self.gpu_utilization_percent = Gauge(
            'modelium_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu']
        )
        
        # Orchestration metrics
        self.model_loads_total = Counter(
            'modelium_model_loads_total',
            'Total model load operations',
            ['runtime', 'status']
        )
        
        self.model_unloads_total = Counter(
            'modelium_model_unloads_total',
            'Total model unload operations',
            ['runtime', 'status']
        )
        
        self.orchestration_decisions_total = Counter(
            'modelium_orchestration_decisions_total',
            'Total orchestration decisions made',
            ['action', 'reason']
        )
        
        # Model state tracking (for QPS calculation)
        self._model_request_counts: Dict[str, int] = {}
        self._model_last_request: Dict[str, float] = {}
        self._last_qps_update: Dict[str, float] = {}
        
        self.logger.info("Prometheus metrics initialized")
    
    def start_server(self, port: int = 9090):
        """
        Start Prometheus metrics HTTP server.
        
        Args:
            port: Port to expose metrics on (default: 9090)
        """
        try:
            start_http_server(port)
            self.logger.info(f"📊 Prometheus metrics server started on port {port}")
            self.logger.info(f"   Metrics: http://localhost:{port}/metrics")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
    
    def record_request(
        self,
        model: str,
        runtime: str,
        latency_ms: float,
        status: str = "success",
        gpu: Optional[int] = None
    ):
        """
        Record an inference request.
        
        Args:
            model: Model name
            runtime: Runtime used (vllm, triton, ray)
            latency_ms: Request latency in milliseconds
            status: success or error
            gpu: GPU ID (optional)
        """
        # Record request
        self.requests_total.labels(
            model=model,
            runtime=runtime,
            status=status
        ).inc()
        
        # Record latency
        self.request_latency.labels(
            model=model,
            runtime=runtime
        ).observe(latency_ms / 1000.0)  # Convert to seconds
        
        # Update request tracking for QPS calculation
        model_key = f"{model}:{runtime}"
        self._model_request_counts[model_key] = self._model_request_counts.get(model_key, 0) + 1
        self._model_last_request[model_key] = time.time()
        
        # Initialize last_update if not set
        if model_key not in self._last_qps_update:
            self._last_qps_update[model_key] = time.time()
        
        # Update QPS gauge periodically (every 1+ seconds) but DON'T reset counter
        # Counter accumulates for 10-second window, then resets
        last_update = self._last_qps_update.get(model_key, time.time())
        if time.time() - last_update >= 1.0:
            self._update_model_qps(model, runtime, gpu)
    
    def _update_model_qps(self, model: str, runtime: str, gpu: Optional[int]):
        """
        Calculate and update QPS gauge for a model.
        
        This updates the Prometheus gauge but does NOT reset the counter.
        The counter accumulates requests for a 10-second window.
        Only resets counter when window expires (>10 seconds).
        """
        model_key = f"{model}:{runtime}"
        now = time.time()
        last_update = self._last_qps_update.get(model_key, now)
        elapsed = now - last_update
        
        if elapsed > 0:
            count = self._model_request_counts.get(model_key, 0)
            
            # Calculate QPS over the elapsed time
            qps = count / elapsed if elapsed > 0 else 0.0
            
            # Update Prometheus gauge
            self.model_qps.labels(
                model=model,
                runtime=runtime,
                gpu=str(gpu) if gpu is not None else "unknown"
            ).set(qps)
            
            # Only reset counter if window expired (10 seconds)
            # This allows get_model_qps() to see recent requests
            if elapsed >= 10.0:
                # Window expired - reset for next window
                self._model_request_counts[model_key] = 0
                self._last_qps_update[model_key] = now
            else:
                # Window not expired - keep accumulating, just update timestamp
                # Don't reset counter - let it accumulate for the full window
                self._last_qps_update[model_key] = now
    
    def update_model_idle_time(self, model: str, runtime: str):
        """Update idle time for a model."""
        model_key = f"{model}:{runtime}"
        last_request = self._model_last_request.get(model_key)
        
        if last_request:
            idle_seconds = time.time() - last_request
            self.model_idle_seconds.labels(
                model=model,
                runtime=runtime
            ).set(idle_seconds)
    
    def update_gpu_memory(self, gpu_id: int, used_gb: float, total_gb: float):
        """
        Update GPU memory metrics.
        
        Args:
            gpu_id: GPU index
            used_gb: Used memory in GB
            total_gb: Total memory in GB
        """
        self.gpu_memory_used_gb.labels(gpu=str(gpu_id)).set(used_gb)
        self.gpu_memory_total_gb.labels(gpu=str(gpu_id)).set(total_gb)
    
    def update_gpu_utilization(self, gpu_id: int, utilization_percent: float):
        """
        Update GPU utilization.
        
        Args:
            gpu_id: GPU index
            utilization_percent: Utilization percentage (0-100)
        """
        self.gpu_utilization_percent.labels(gpu=str(gpu_id)).set(utilization_percent)
    
    def record_model_load(self, runtime: str, status: str = "success"):
        """Record a model load operation."""
        self.model_loads_total.labels(
            runtime=runtime,
            status=status
        ).inc()
    
    def record_model_unload(self, runtime: str, status: str = "success"):
        """Record a model unload operation."""
        self.model_unloads_total.labels(
            runtime=runtime,
            status=status
        ).inc()
    
    def record_orchestration_decision(self, action: str, reason: str):
        """
        Record an orchestration decision.
        
        Args:
            action: load, unload, keep, queue
            reason: Why this decision was made
        """
        self.orchestration_decisions_total.labels(
            action=action,
            reason=reason
        ).inc()
    
    def update_models_loaded_count(self, runtime: str, count: int):
        """Update count of loaded models for a runtime."""
        self.models_loaded.labels(runtime=runtime).set(count)
    
    def get_model_qps(self, model: str, runtime: str) -> float:
        """
        Get current QPS for a model.
        
        Calculates from request counter over a 10-second sliding window.
        This is called by the orchestrator to get real-time QPS for brain decisions.
        """
        model_key = f"{model}:{runtime}"
        now = time.time()
        
        # Get current count and last update time
        count = self._model_request_counts.get(model_key, 0)
        last_update = self._last_qps_update.get(model_key, now)
        elapsed = now - last_update
        
        # If we have recent requests, calculate QPS
        if elapsed > 0 and count > 0:
            # Use a 10-second sliding window
            # If window hasn't expired, calculate QPS from current count
            if elapsed < 10.0:
                # Calculate QPS over the elapsed time
                qps = count / elapsed if elapsed > 0 else 0.0
                return qps
            else:
                # Window expired (>10s), counter should be reset by _update_model_qps
                # But if it hasn't been reset yet, return 0
                return 0.0
        
        # No requests or counter was reset
        return 0.0
    
    def get_model_idle_seconds(self, model: str, runtime: str) -> float:
        """Get idle seconds for a model."""
        model_key = f"{model}:{runtime}"
        last_request = self._model_last_request.get(model_key)
        
        if last_request:
            return time.time() - last_request
        return float('inf')  # Never had a request

