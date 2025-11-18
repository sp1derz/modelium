"""
Ray Serve HTTP Connector

Connects to an external Ray Serve deployment.
"""

import logging
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class RayConnector:
    """
    HTTP client for Ray Serve.
    
    Ray Serve provides flexible model serving for Python models.
    Users should start Ray Serve separately and deploy their models.
    
    Then Modelium connects to it for inference.
    """
    
    def __init__(self, endpoint: str, timeout: int = 300):
        """
        Initialize Ray Serve connector.
        
        Args:
            endpoint: Base URL of Ray Serve (e.g., "http://localhost:8002")
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def health_check(self) -> bool:
        """
        Check if Ray Serve is healthy.
        
        Returns:
            True if Ray Serve is healthy
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/-/healthz",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ray Serve health check failed: {e}")
            return False
    
    def list_deployments(self) -> List[str]:
        """
        List active Ray Serve deployments.
        
        Returns:
            List of deployment names
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/-/routes",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return list(data.keys()) if isinstance(data, dict) else []
        except Exception as e:
            self.logger.error(f"Failed to list Ray Serve deployments: {e}")
            return []
    
    def inference(
        self,
        deployment: str,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference on a Ray Serve deployment.
        
        Args:
            deployment: Deployment name/route
            input_data: Input data (format depends on deployment)
            **kwargs: Additional parameters
        
        Returns:
            Dict with inference results
        """
        try:
            # Ray Serve typically uses custom routes
            # Format depends on how the deployment was configured
            route = f"/{deployment}" if not deployment.startswith("/") else deployment
            
            payload = {
                "input": input_data,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.endpoint}{route}",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Ray Serve inference timeout after {self.timeout}s")
            return {"error": "Inference timeout", "deployment": deployment}
        except Exception as e:
            self.logger.error(f"Ray Serve inference failed: {e}")
            return {"error": str(e), "deployment": deployment}
    
    def get_deployment_info(self, deployment: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific deployment.
        
        Args:
            deployment: Deployment name
        
        Returns:
            Deployment info dict or None
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/-/routes",
                timeout=10
            )
            response.raise_for_status()
            routes = response.json()
            return routes.get(deployment)
        except Exception as e:
            self.logger.debug(f"Failed to get Ray Serve deployment info: {e}")
            return None
    
    def predict(
        self,
        model_name: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Standard prediction endpoint (common Ray Serve pattern).
        
        Args:
            model_name: Model name
            data: Input data dict
            **kwargs: Additional parameters
        
        Returns:
            Prediction results
        """
        try:
            payload = {
                "data": data,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.endpoint}/predict",
                json=payload,
                timeout=self.timeout,
                params={"model": model_name}
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Ray Serve prediction failed: {e}")
            return {"error": str(e), "model": model_name}

