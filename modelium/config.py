"""
Modelium Configuration System

Central configuration management for all Modelium deployments.
Supports multi-tenant, multi-instance, and enterprise deployments.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml
from pydantic import BaseModel, Field


class OrganizationConfig(BaseModel):
    """Organization settings for multi-tenancy."""
    id: str = "default"
    name: str = "Default Organization"
    enable_usage_tracking: bool = True


class RuntimeConfig(BaseModel):
    """Runtime preferences."""
    default: str = "auto"  # auto, vllm, ray_serve, tensorrt, triton
    overrides: Dict[str, str] = Field(default_factory=dict)


class GPUConfig(BaseModel):
    """GPU configuration."""
    enabled: bool = True
    count: Optional[int] = None  # None = auto-detect
    type: str = "nvidia-a100"
    memory_fraction: float = 0.9


class VLLMConfig(BaseModel):
    """vLLM settings - connects to external vLLM server."""
    enabled: bool = True
    endpoint: str = "http://localhost:8001"  # vLLM server endpoint
    health_check_path: str = "/health"
    model_load_path: str = "/v1/models"
    inference_path: str = "/v1/completions"
    timeout: int = 300  # seconds


class RayServeConfig(BaseModel):
    """Ray Serve settings - connects to external Ray Serve cluster."""
    enabled: bool = True
    endpoint: str = "http://localhost:8002"  # Ray Serve endpoint
    health_check_path: str = "/-/healthz"
    inference_path: str = "/predict"
    timeout: int = 300  # seconds


class TensorRTConfig(BaseModel):
    """TensorRT settings."""
    enabled: bool = True
    use_fp16: bool = True
    use_int8: bool = False
    workspace_size_gb: int = 4
    max_batch_size: int = 32
    dynamic_shapes: bool = True


class TritonConfig(BaseModel):
    """Triton Inference Server settings - connects to external Triton server."""
    enabled: bool = True
    endpoint: str = "http://localhost:8003"  # Triton server endpoint
    health_check_path: str = "/v2/health/ready"
    model_repository_path: str = "/v2/repository/models"
    inference_path: str = "/v2/models/{model}/infer"  # KServe v2 protocol
    timeout: int = 300  # seconds


class ConversionConfig(BaseModel):
    """Model conversion settings."""
    timeout_seconds: int = 3600
    max_retries: int = 3
    enable_validation: bool = True
    save_intermediate_artifacts: bool = True
    cleanup_on_success: bool = False
    cleanup_on_failure: bool = True


class DeploymentConfig(BaseModel):
    """Deployment settings."""
    environment: str = "production"
    namespace: str = "modelium"
    auto_deploy: bool = True
    health_check_timeout: int = 300
    resources: Dict[str, Any] = Field(default_factory=lambda: {
        "requests": {"cpu": "2", "memory": "8Gi", "gpu": "1"},
        "limits": {"cpu": "8", "memory": "32Gi", "gpu": "1"}
    })


class StorageConfig(BaseModel):
    """Storage settings."""
    models_dir: str = "/models/incoming"
    artifacts_dir: str = "/models/artifacts"
    logs_dir: str = "/models/logs"
    backend: str = "local"  # local, s3, gcs, azure
    s3: Dict[str, Any] = Field(default_factory=dict)
    gcs: Dict[str, Any] = Field(default_factory=dict)
    azure: Dict[str, Any] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    """Monitoring & observability settings."""
    enabled: bool = True
    prometheus: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "port": 9090})
    grafana: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "port": 3000})
    logging: Dict[str, Any] = Field(default_factory=lambda: {"level": "INFO", "format": "json"})


class SecurityConfig(BaseModel):
    """Security settings."""
    enable_sandbox: bool = True
    enable_model_scanning: bool = True
    allowed_model_sources: List[str] = Field(default_factory=lambda: ["huggingface.co", "local"])
    block_external_network: bool = True


class WorkloadInstance(BaseModel):
    """Configuration for a separate workload instance."""
    description: str
    model_types: List[str]
    runtime: str
    gpu_count: int = 1
    port_offset: int = 0


class WorkloadSeparationConfig(BaseModel):
    """Workload separation for high-traffic deployments."""
    enabled: bool = False
    instances: Dict[str, WorkloadInstance] = Field(default_factory=dict)


class RateLimitingConfig(BaseModel):
    """Rate limiting settings."""
    enabled: bool = True
    requests_per_minute: int = 1000
    requests_per_day: int = 100000
    per_organization: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "default_rpm": 1000,
        "overrides": {}
    })


class UsageTrackingConfig(BaseModel):
    """Usage tracking for billing."""
    enabled: bool = True
    track_inference_calls: bool = True
    track_conversion_time: bool = True
    track_gpu_hours: bool = True
    track_storage_bytes: bool = True
    export_to: str = "prometheus"
    retention_days: int = 90


class APIConfig(BaseModel):
    """API settings."""
    enable_cors: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    enable_authentication: bool = False
    auth_type: str = "jwt"
    require_organization_id: bool = True


class ExperimentalConfig(BaseModel):
    """Experimental features."""
    enable_model_caching: bool = True
    enable_batching_optimization: bool = True
    enable_quantization_auto_tuning: bool = False


class ModeliumBrainConfig(BaseModel):
    """Modelium Brain (AI decision engine) settings."""
    enabled: bool = True
    model_name: str = "modelium/brain-v1"
    device: str = "cuda:0"
    dtype: str = "float16"
    max_new_tokens: int = 2048
    temperature: float = 0.3
    timeout_seconds: int = 5
    fallback_to_rules: bool = True


class ModelDiscoveryConfig(BaseModel):
    """Model discovery settings."""
    watch_directories: List[str] = Field(default_factory=lambda: ["/models/incoming"])
    auto_register: bool = True
    scan_interval_seconds: int = 30
    supported_formats: List[str] = Field(default_factory=lambda: [".pt", ".pth", ".onnx", ".safetensors", ".bin"])


class OrchestrationPoliciesConfig(BaseModel):
    """Orchestration policy settings."""
    evict_after_idle_seconds: int = 300
    evict_when_memory_above_percent: int = 85
    always_loaded: List[str] = Field(default_factory=list)
    priority_by_qps: bool = True
    priority_by_organization: bool = True
    priority_custom: Dict[str, int] = Field(default_factory=dict)
    preload_on_first_request: bool = True
    max_concurrent_loads: int = 2
    target_load_time_seconds: int = 60


class FastLoadingConfig(BaseModel):
    """Fast loading settings."""
    enabled: bool = False
    use_gpu_direct_storage: bool = False
    nvme_cache_path: str = "/mnt/nvme/models"
    use_memory_mapping: bool = True
    quantize_on_load: bool = False


class RequestRoutingConfig(BaseModel):
    """Request routing settings."""
    queue_when_unloaded: bool = True
    max_queue_size: int = 1000
    max_queue_wait_seconds: int = 120
    notify_on_load_complete: bool = True


class OrchestrationConfig(BaseModel):
    """Orchestration system settings."""
    enabled: bool = True
    mode: str = "intelligent"  # intelligent or simple
    model_discovery: ModelDiscoveryConfig = Field(default_factory=ModelDiscoveryConfig)
    decision_interval_seconds: int = 10
    policies: OrchestrationPoliciesConfig = Field(default_factory=OrchestrationPoliciesConfig)
    fast_loading: FastLoadingConfig = Field(default_factory=FastLoadingConfig)
    request_routing: RequestRoutingConfig = Field(default_factory=RequestRoutingConfig)


class MetricsConfig(BaseModel):
    """Metrics collection settings."""
    enabled: bool = True
    port: int = 9090  # Prometheus port
    collection_interval_seconds: int = 10
    retention_hours: int = 24
    exporters: Dict[str, Any] = Field(default_factory=lambda: {
        "prometheus": {"enabled": True, "port": 9090, "path": "/metrics"},
        "storage": {"enabled": True, "backend": "local", "path": "/models/metrics"}
    })
    track: Dict[str, bool] = Field(default_factory=lambda: {
        "model_requests": True,
        "model_latency": True,
        "model_idle_time": True,
        "gpu_memory": True,
        "gpu_utilization": True,
        "orchestration_decisions": True,
        "brain_confidence": True
    })


class ModeliumConfig(BaseModel):
    """Complete Modelium configuration."""
    
    organization: OrganizationConfig = Field(default_factory=OrganizationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    
    # Intelligent orchestration
    modelium_brain: ModeliumBrainConfig = Field(default_factory=ModeliumBrainConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    
    # Runtimes
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    ray_serve: RayServeConfig = Field(default_factory=RayServeConfig)
    tensorrt: TensorRTConfig = Field(default_factory=TensorRTConfig)
    triton: TritonConfig = Field(default_factory=TritonConfig)
    
    # Conversion & Deployment
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Monitoring & Security
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Advanced features
    workload_separation: WorkloadSeparationConfig = Field(default_factory=WorkloadSeparationConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    usage_tracking: UsageTrackingConfig = Field(default_factory=UsageTrackingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)
    
    def get_runtime_for_model(self, model_type: str, organizationId: str) -> str:
        """
        Get the appropriate runtime for a model type and organization.
        
        Args:
            model_type: Type of model (llm, vision, text, etc.)
            organizationId: Organization ID for tracking
            
        Returns:
            Runtime name: vllm, ray_serve, tensorrt, or triton
        """
        # Check if workload separation is enabled
        if self.workload_separation.enabled:
            # Find instance that handles this model type
            for instance_name, instance in self.workload_separation.instances.items():
                if model_type in instance.model_types:
                    return instance.runtime
        
        # Check runtime overrides
        if model_type in self.runtime.overrides:
            return self.runtime.overrides[model_type]
        
        # Use default
        return self.runtime.default
    
    def get_port_for_runtime(self, runtime: str, model_type: Optional[str] = None) -> int:
        """
        Get the port for a runtime, accounting for workload separation.
        
        Args:
            runtime: Runtime name (vllm, ray_serve, etc.)
            model_type: Optional model type for workload separation
            
        Returns:
            Port number
        """
        base_ports = {
            "vllm": self.vllm.port,
            "ray_serve": self.ray_serve.port,
            "triton": self.triton.port,
            "tensorrt": self.triton.port,  # TensorRT uses Triton
        }
        
        base_port = base_ports.get(runtime, 8000)
        
        # Check workload separation
        if self.workload_separation.enabled and model_type:
            for instance_name, instance in self.workload_separation.instances.items():
                if model_type in instance.model_types:
                    return base_port + instance.port_offset
        
        return base_port
    
    def is_runtime_enabled(self, runtime: str) -> bool:
        """Check if a runtime is enabled."""
        runtime_enabled = {
            "vllm": self.vllm.enabled,
            "ray_serve": self.ray_serve.enabled,
            "tensorrt": self.tensorrt.enabled,
            "triton": self.triton.enabled,
        }
        return runtime_enabled.get(runtime, False)


# Global config instance
_config: Optional[ModeliumConfig] = None


def load_config(config_path: Optional[str] = None) -> ModeliumConfig:
    """
    Load Modelium configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: ./modelium.yaml)
        
    Returns:
        ModeliumConfig instance
    """
    global _config
    
    if config_path is None:
        # Try to find config file
        possible_paths = [
            "modelium.yaml",
            "modelium.yml",
            "/etc/modelium/modelium.yaml",
            os.path.expanduser("~/.modelium/config.yaml"),
            os.environ.get("MODELIUM_CONFIG", ""),
        ]
        
        for path in possible_paths:
            if path and Path(path).exists():
                config_path = path
                break
        
        if not config_path:
            # Use defaults
            _config = ModeliumConfig()
            return _config
    
    # Load from YAML
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    _config = ModeliumConfig(**config_dict)
    return _config


def get_config() -> ModeliumConfig:
    """
    Get the current Modelium configuration.
    
    Returns:
        ModeliumConfig instance (loads from file if not already loaded)
    """
    global _config
    
    if _config is None:
        _config = load_config()
    
    return _config


def reload_config(config_path: Optional[str] = None) -> ModeliumConfig:
    """
    Reload configuration from file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        New ModeliumConfig instance
    """
    global _config
    _config = None
    return load_config(config_path)


# Convenience functions
def get_runtime_for_model(model_type: str, organizationId: str) -> str:
    """Get runtime for a model type."""
    return get_config().get_runtime_for_model(model_type, organizationId)


def get_organization_id() -> str:
    """Get the configured organization ID."""
    return get_config().organization.id


def is_gpu_enabled() -> bool:
    """Check if GPU is enabled."""
    return get_config().gpu.enabled


def get_vllm_config() -> VLLMConfig:
    """Get vLLM configuration."""
    return get_config().vllm


def get_ray_config() -> RayServeConfig:
    """Get Ray Serve configuration."""
    return get_config().ray_serve

