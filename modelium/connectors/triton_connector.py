"""
Triton Inference Server HTTP Connector

Connects to an external Triton server (KServe v2 protocol).
"""

import logging
import requests
import numpy as np
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)


class TritonConnector:
    """
    HTTP client for NVIDIA Triton Inference Server.
    
    Uses KServe v2 inference protocol.
    Users should start Triton separately:
    
        docker run --gpus all -p 8003:8000 \\
            nvcr.io/nvidia/tritonserver:latest \\
            tritonserver --model-repository=/models
    
    Then Modelium connects to it for inference.
    """
    
    def __init__(self, endpoint: str, timeout: int = 300):
        """
        Initialize Triton connector.
        
        Args:
            endpoint: Base URL of Triton server (e.g., "http://localhost:8003")
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def health_check(self) -> bool:
        """
        Check if Triton server is healthy and ready.
        
        Returns:
            True if server is ready
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/v2/health/ready",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Triton health check failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """
        List models available in Triton model repository.
        
        Returns:
            List of model names
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/v2/models",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            self.logger.error(f"Failed to list Triton models: {e}")
            return []
    
    def get_model_metadata(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata from Triton.
        
        Args:
            model: Model name
        
        Returns:
            Model metadata dict or None
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/v2/models/{model}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.debug(f"Failed to get Triton model metadata: {e}")
            return None
    
    def get_model_config(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration from Triton.
        
        Args:
            model: Model name
        
        Returns:
            Model config dict or None
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/v2/models/{model}/config",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.debug(f"Failed to get Triton model config: {e}")
            return None
    
    def inference(
        self,
        model: str,
        inputs: List[Dict[str, Any]],
        outputs: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference using Triton's KServe v2 protocol.
        
        Args:
            model: Model name
            inputs: List of input tensors [{"name": "INPUT", "shape": [...], "datatype": "FP32", "data": [...]}]
            outputs: Optional output specification
            **kwargs: Additional parameters
        
        Returns:
            Dict with inference results (KServe v2 format)
        """
        try:
            payload = {
                "inputs": inputs
            }
            if outputs:
                payload["outputs"] = outputs
            
            response = self.session.post(
                f"{self.endpoint}/v2/models/{model}/infer",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Triton inference timeout after {self.timeout}s")
            return {"error": "Inference timeout", "model": model}
        except Exception as e:
            self.logger.error(f"Triton inference failed: {e}")
            return {"error": str(e), "model": model}
    
    def load_model(self, model: str) -> bool:
        """
        Load a model into Triton (if dynamic loading is enabled).
        
        Args:
            model: Model name
        
        Returns:
            True if successful
        """
        try:
            response = self.session.post(
                f"{self.endpoint}/v2/repository/models/{model}/load",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to load Triton model {model}: {e}")
            return False
    
    def unload_model(self, model: str) -> bool:
        """
        Unload a model from Triton.
        
        Args:
            model: Model name
        
        Returns:
            True if successful
        """
        try:
            response = self.session.post(
                f"{self.endpoint}/v2/repository/models/{model}/unload",
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to unload Triton model {model}: {e}")
            return False
    
    def get_model_ready(self, model: str) -> bool:
        """
        Check if a specific model is ready for inference.
        
        Args:
            model: Model name
        
        Returns:
            True if model is ready
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/v2/models/{model}/ready",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Triton model {model} not ready: {e}")
            return False
    
    def get_server_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get Triton server metadata.
        
        Returns:
            Server metadata dict or None
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/v2",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.debug(f"Failed to get Triton server metadata: {e}")
            return None

