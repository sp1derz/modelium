"""
KServe InferenceService manifest generator.

Generates Kubernetes manifests for KServe deployment.
"""

import logging
from typing import Dict, Any, Optional, List

import yaml

logger = logging.getLogger(__name__)


class KServeManifestGenerator:
    """
    Generates KServe InferenceService manifests.
    
    Creates Kubernetes YAML for deploying models with KServe.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_inference_service(
        self,
        name: str,
        namespace: str = "default",
        predictor: Dict[str, Any] = None,
        transformer: Optional[Dict[str, Any]] = None,
        explainer: Optional[Dict[str, Any]] = None,
        min_replicas: int = 1,
        max_replicas: int = 5,
        scale_target: int = 100,
        scale_metric: str = "concurrency",
        canary_traffic_percent: Optional[int] = None,
        annotations: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate InferenceService manifest.
        
        Args:
            name: Service name
            namespace: Kubernetes namespace
            predictor: Predictor specification
            transformer: Optional transformer specification
            explainer: Optional explainer specification
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas
            scale_target: Scaling target value
            scale_metric: Scaling metric (concurrency, rps, cpu, memory)
            canary_traffic_percent: Canary traffic percentage
            annotations: Additional annotations
            labels: Additional labels
            
        Returns:
            YAML manifest as string
        """
        self.logger.info(f"Generating InferenceService manifest for {name}")
        
        manifest = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": name,
                "namespace": namespace,
            },
            "spec": {},
        }
        
        # Add annotations and labels
        if annotations:
            manifest["metadata"]["annotations"] = annotations
        if labels:
            manifest["metadata"]["labels"] = labels
        
        # Add predictor
        if predictor:
            manifest["spec"]["predictor"] = predictor
        else:
            raise ValueError("Predictor specification is required")
        
        # Add autoscaling
        if "minReplicas" not in manifest["spec"]["predictor"]:
            manifest["spec"]["predictor"]["minReplicas"] = min_replicas
        if "maxReplicas" not in manifest["spec"]["predictor"]:
            manifest["spec"]["predictor"]["maxReplicas"] = max_replicas
        
        # Add scaling target
        manifest["spec"]["predictor"]["scaleTarget"] = scale_target
        manifest["spec"]["predictor"]["scaleMetric"] = scale_metric
        
        # Add transformer if specified
        if transformer:
            manifest["spec"]["transformer"] = transformer
        
        # Add explainer if specified
        if explainer:
            manifest["spec"]["explainer"] = explainer
        
        # Add canary if specified
        if canary_traffic_percent is not None:
            manifest["spec"]["canaryTrafficPercent"] = canary_traffic_percent
        
        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)
    
    def create_triton_predictor(
        self,
        model_uri: str,
        runtime_version: str = "23.10-py3",
        resources: Optional[Dict[str, Any]] = None,
        env: Optional[List[Dict[str, str]]] = None,
        tolerations: Optional[List[Dict[str, Any]]] = None,
        node_selector: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create Triton predictor specification.
        
        Args:
            model_uri: URI to model (s3://, gs://, pvc://)
            runtime_version: Triton runtime version
            resources: Resource requests/limits
            env: Environment variables
            tolerations: Pod tolerations
            node_selector: Node selector
            
        Returns:
            Predictor specification dictionary
        """
        predictor = {
            "triton": {
                "storageUri": model_uri,
                "runtimeVersion": runtime_version,
            }
        }
        
        # Add resources
        if resources:
            predictor["triton"]["resources"] = resources
        else:
            # Default resources
            predictor["triton"]["resources"] = {
                "requests": {
                    "cpu": "1",
                    "memory": "4Gi",
                    "nvidia.com/gpu": "1",
                },
                "limits": {
                    "cpu": "2",
                    "memory": "8Gi",
                    "nvidia.com/gpu": "1",
                },
            }
        
        # Add environment variables
        if env:
            predictor["triton"]["env"] = env
        
        # Add tolerations
        if tolerations:
            predictor["triton"]["tolerations"] = tolerations
        
        # Add node selector
        if node_selector:
            predictor["triton"]["nodeSelector"] = node_selector
        
        return predictor
    
    def create_pytorch_predictor(
        self,
        model_uri: str,
        protocol_version: str = "v2",
        resources: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create PyTorch predictor specification."""
        predictor = {
            "pytorch": {
                "storageUri": model_uri,
                "protocolVersion": protocol_version,
            }
        }
        
        if resources:
            predictor["pytorch"]["resources"] = resources
        
        return predictor
    
    def create_onnx_predictor(
        self,
        model_uri: str,
        protocol_version: str = "v2",
        resources: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create ONNX predictor specification."""
        predictor = {
            "onnx": {
                "storageUri": model_uri,
                "protocolVersion": protocol_version,
            }
        }
        
        if resources:
            predictor["onnx"]["resources"] = resources
        
        return predictor
    
    def generate_service_mesh_config(
        self,
        name: str,
        namespace: str = "default",
        timeout: int = 300,
        retry_attempts: int = 3,
    ) -> str:
        """
        Generate service mesh configuration (Istio VirtualService).
        
        Args:
            name: Service name
            namespace: Namespace
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            
        Returns:
            VirtualService YAML manifest
        """
        manifest = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{name}-vs",
                "namespace": namespace,
            },
            "spec": {
                "hosts": [f"{name}.{namespace}.svc.cluster.local"],
                "http": [
                    {
                        "timeout": f"{timeout}s",
                        "retries": {
                            "attempts": retry_attempts,
                            "perTryTimeout": f"{timeout // retry_attempts}s",
                        },
                        "route": [
                            {
                                "destination": {
                                    "host": f"{name}-predictor-default.{namespace}.svc.cluster.local",
                                }
                            }
                        ],
                    }
                ],
            },
        }
        
        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)
    
    def generate_hpa(
        self,
        name: str,
        namespace: str = "default",
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_cpu_utilization: int = 80,
        target_memory_utilization: Optional[int] = None,
    ) -> str:
        """
        Generate HorizontalPodAutoscaler manifest.
        
        Args:
            name: Service name
            namespace: Namespace
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas
            target_cpu_utilization: Target CPU utilization percentage
            target_memory_utilization: Target memory utilization percentage
            
        Returns:
            HPA YAML manifest
        """
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{name}-hpa",
                "namespace": namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "serving.kserve.io/v1beta1",
                    "kind": "InferenceService",
                    "name": name,
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": target_cpu_utilization,
                            },
                        },
                    }
                ],
            },
        }
        
        # Add memory metric if specified
        if target_memory_utilization:
            manifest["spec"]["metrics"].append({
                "type": "Resource",
                "resource": {
                    "name": "memory",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": target_memory_utilization,
                    },
                },
            })
        
        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

