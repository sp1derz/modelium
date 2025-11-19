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
        now = time.time()
        
        # Increment request counter
        old_count = self._model_request_counts.get(model_key, 0)
        self._model_request_counts[model_key] = old_count + 1
        self._model_last_request[model_key] = now
        
        # Initialize last_update if not set (start of window)
        is_new_window = model_key not in self._last_qps_update
        if is_new_window:
            self._last_qps_update[model_key] = now
            self.logger.info(f"📊 QPS: Initialized window for {model_key} at {now}")
        
        # Update QPS gauge periodically (every 1+ seconds) but DON'T reset counter
        # Counter accumulates for 10-second window, then resets
        last_update = self._last_qps_update.get(model_key, now)
        elapsed_since_update = now - last_update
        
        # Log request recording
        self.logger.info(f"📊 QPS: Recorded request for {model_key}: count={self._model_request_counts[model_key]} (was {old_count}), elapsed={elapsed_since_update:.2f}s, gpu={gpu}")
        
        if elapsed_since_update >= 1.0:
            self.logger.info(f"📊 QPS: Updating gauge for {model_key} (elapsed={elapsed_since_update:.2f}s >= 1.0s)")
            self._update_model_qps(model, runtime, gpu)
        else:
            self.logger.debug(f"📊 QPS: Skipping gauge update for {model_key} (elapsed={elapsed_since_update:.2f}s < 1.0s)")
    
    def _update_model_qps(self, model: str, runtime: str, gpu: Optional[int]):
        """
        Calculate and update QPS gauge for a model.
        
        This updates the Prometheus gauge but does NOT reset the counter.
        The counter accumulates requests for a 10-second window.
        Only resets counter when window expires (>10 seconds).
        
        IMPORTANT: We don't update _last_qps_update here unless window expired.
        This allows get_model_qps() to calculate QPS from the full window.
        """
        model_key = f"{model}:{runtime}"
        now = time.time()
        last_update = self._last_qps_update.get(model_key, now)
        elapsed = now - last_update
        
        if elapsed > 0:
            count = self._model_request_counts.get(model_key, 0)
            
            # Calculate QPS over the elapsed time
            qps = count / elapsed if elapsed > 0 else 0.0
            
            # Log before updating gauge
            gpu_label = str(gpu) if gpu is not None else "unknown"
            self.logger.info(f"📊 QPS: Updating gauge for {model_key}: count={count}, elapsed={elapsed:.2f}s, qps={qps:.2f}, gpu={gpu_label}")
            
            # Update Prometheus gauge
            self.model_qps.labels(
                model=model,
                runtime=runtime,
                gpu=gpu_label
            ).set(qps)
            
            # Verify gauge was updated
            try:
                verify_value = self.model_qps.labels(
                    model=model,
                    runtime=runtime,
                    gpu=gpu_label
                )._value.get()
                self.logger.info(f"📊 QPS: Gauge updated successfully for {model_key}: {verify_value:.2f}")
            except Exception as e:
                self.logger.error(f"📊 QPS: Failed to verify gauge update for {model_key}: {e}")
            
            # Only reset counter and update timestamp if window expired (10 seconds)
            # This allows get_model_qps() to see recent requests in the current window
            if elapsed >= 10.0:
                # Window expired - reset for next window
                self.logger.info(f"📊 QPS: Window expired for {model_key} (elapsed={elapsed:.2f}s >= 10.0s), resetting counter")
                self._model_request_counts[model_key] = 0
                self._last_qps_update[model_key] = now
            # If window not expired, DON'T update _last_qps_update
            # This allows get_model_qps() to calculate from the start of the window
        else:
            self.logger.warning(f"📊 QPS: Cannot update gauge for {model_key} (elapsed={elapsed:.2f}s <= 0)")
    
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
        
        IMPORTANT: Reads from Prometheus gauge first (most accurate), then falls back to counter.
        """
        model_key = f"{model}:{runtime}"
        now = time.time()
        
        # First, try to read from Prometheus gauge (updated by _update_model_qps)
        # This is the most accurate value, updated every 1+ seconds
        gauge_value = None
        gauge_labels_tried = []
        best_gauge_value = None
        best_gpu_label = None
        
        # Try multiple GPU label values (unknown, actual GPU ID if available)
        # IMPORTANT: Try all labels and pick the one with highest value (most recent)
        # Don't return early on first match, as "unknown" might be 0.0 while actual GPU has value
        for gpu_label in ["unknown", "0", "1", "2", "3"]:
            try:
                gauge_value = self.model_qps.labels(
                    model=model,
                    runtime=runtime,
                    gpu=gpu_label
                )._value.get()
                
                if gauge_value is not None and gauge_value >= 0:
                    gauge_labels_tried.append(f"{gpu_label}:{gauge_value:.2f}")
                    # Track the best (highest) value - this is likely the actual GPU label used
                    if best_gauge_value is None or gauge_value > best_gauge_value:
                        best_gauge_value = gauge_value
                        best_gpu_label = gpu_label
                else:
                    gauge_labels_tried.append(f"{gpu_label}:None/Invalid")
            except Exception as e:
                gauge_labels_tried.append(f"{gpu_label}:ERROR({str(e)[:50]})")
                continue
        
        # Return the best gauge value found (highest, likely the actual GPU)
        if best_gauge_value is not None and best_gauge_value > 0:
            self.logger.info(f"📊 QPS: Read from Prometheus gauge for {model_key} (gpu={best_gpu_label}): {best_gauge_value:.2f} (tried: {gauge_labels_tried})")
            return float(best_gauge_value)
        
        # Gauge read failed for all labels - log and fall back to counter
        self.logger.warning(f"📊 QPS: Could not read gauge for {model_key}, tried labels: {gauge_labels_tried}, falling back to counter")
        
        # Fallback: Calculate from request counter
        count = self._model_request_counts.get(model_key, 0)
        last_update = self._last_qps_update.get(model_key, now)
        elapsed = now - last_update
        
        self.logger.info(f"📊 QPS: Counter calculation for {model_key}: count={count}, elapsed={elapsed:.2f}s, last_update={last_update}")
        
        # If we have requests, calculate QPS
        if count > 0 and elapsed > 0:
            # Use a 10-second sliding window
            if elapsed < 10.0:
                # Calculate QPS over the elapsed time (requests per second)
                qps = count / elapsed
                self.logger.info(f"📊 QPS: Calculated from counter for {model_key}: {qps:.2f} (count={count}, elapsed={elapsed:.2f}s)")
                return qps
            else:
                # Window expired (>10s), counter should be reset by _update_model_qps
                # But calculate from what we have anyway (average over 10s)
                qps = count / 10.0
                self.logger.info(f"📊 QPS: Calculated from counter (expired) for {model_key}: {qps:.2f} (count={count}, window expired)")
                return qps
        
        # No requests recorded or elapsed is 0
        self.logger.warning(f"📊 QPS: Returning 0.0 for {model_key} (count={count}, elapsed={elapsed:.2f}s)")
        return 0.0
    
    def get_model_idle_seconds(self, model: str, runtime: str) -> float:
        """Get idle seconds for a model."""
        model_key = f"{model}:{runtime}"
        last_request = self._model_last_request.get(model_key)
        
        if last_request:
            return time.time() - last_request
        return float('inf')  # Never had a request

