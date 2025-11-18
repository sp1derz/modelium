"""
Ray Serve Runtime Manager

Manages model deployments in Ray Serve programmatically.
Uses Ray's Python API to deploy/undeploy models dynamically.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class RayManager:
    """
    Manages models in Ray Serve.
    
    Uses Ray Serve's Python API for dynamic deployment.
    
    Prerequisites:
        - Ray cluster running: ray start --head
        - ray[serve] installed
    
    Usage:
        manager = RayManager()
        manager.load_model("gpt2", "/models/repository/gpt2", gpu_id=0)
        manager.unload_model("gpt2")
    """
    
    def __init__(self):
        """Initialize Ray Serve manager."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._deployments: Dict[str, Dict] = {}
        
        # Check if Ray Serve is available
        try:
            import ray
            from ray import serve
            self.ray = ray
            self.serve = serve
            self._available = True
            self.logger.info("Ray Serve manager initialized")
        except ImportError:
            self.logger.warning("Ray Serve not available (pip install ray[serve])")
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Ray Serve is available."""
        return self._available
    
    def load_model(
        self,
        model_name: str,
        model_path: Path,
        gpu_id: int = 0,
        settings: Optional[Dict] = None
    ) -> bool:
        """
        Deploy a model in Ray Serve.
        
        Args:
            model_name: Name for the deployment
            model_path: Path to model files
            gpu_id: GPU to use (via ray_actor_options)
            settings: Ray Serve deployment settings
        
        Returns:
            True if successful
        """
        if not self._available:
            self.logger.error("Ray Serve not available")
            return False
        
        try:
            self.logger.info(f"🚀 Deploying {model_name} with Ray Serve...")
            
            # Import here to avoid issues when Ray not installed
            from ray import serve
            
            # Check if Ray is initialized
            if not self.ray.is_initialized():
                self.logger.info("   Initializing Ray...")
                self.ray.init(ignore_reinit_error=True)
            
            # Start Serve if not started
            try:
                serve.start(detached=True)
            except:
                pass  # Already started
            
            # Define deployment configuration
            ray_actor_options = {
                "num_gpus": 1,
                "num_cpus": 2,
            }
            
            if gpu_id is not None:
                # Pin to specific GPU via CUDA_VISIBLE_DEVICES
                ray_actor_options["runtime_env"] = {
                    "env_vars": {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
                }
            
            # Create deployment
            # Note: This is a simplified example
            # In production, you'd create a proper Ray Serve deployment class
            
            @serve.deployment(
                name=model_name,
                ray_actor_options=ray_actor_options,
                num_replicas=1,
            )
            class ModelDeployment:
                def __init__(self, model_path_str: str):
                    self.model_path = model_path_str
                    self.model = None
                    # Load model here
                    # from transformers import AutoModel
                    # self.model = AutoModel.from_pretrained(model_path)
                
                def __call__(self, request: Dict) -> Dict:
                    # Inference logic here
                    return {"model": self.model_path, "result": "placeholder"}
            
            # Deploy
            deployment = ModelDeployment.bind(str(model_path))
            serve.run(deployment, name=model_name, route_prefix=f"/{model_name}")
            
            self._deployments[model_name] = {
                "model_path": str(model_path),
                "gpu_id": gpu_id,
            }
            
            self.logger.info(f"   ✅ {model_name} deployed via Ray Serve")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying {model_name} with Ray Serve: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """
        Undeploy a model from Ray Serve.
        
        Args:
            model_name: Name of deployment to remove
        
        Returns:
            True if successful
        """
        if not self._available:
            return False
        
        try:
            if model_name not in self._deployments:
                self.logger.warning(f"{model_name} not deployed")
                return False
            
            self.logger.info(f"🛑 Undeploying {model_name} from Ray Serve...")
            
            # Delete deployment
            self.serve.delete(model_name)
            
            del self._deployments[model_name]
            
            self.logger.info(f"   ✅ {model_name} undeployed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error undeploying {model_name}: {e}")
            return False
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently deployed."""
        if not self._available:
            return False
        
        try:
            # Check deployment status
            status = self.serve.status()
            applications = status.applications
            return model_name in applications
        except:
            return model_name in self._deployments
    
    def list_loaded_models(self) -> List[str]:
        """List all currently deployed models."""
        if not self._available:
            return []
        
        try:
            status = self.serve.status()
            return list(status.applications.keys())
        except:
            return list(self._deployments.keys())
    
    def shutdown_all(self):
        """Shutdown all deployments."""
        if not self._available:
            return
        
        self.logger.info("Shutting down all Ray Serve deployments...")
        for model_name in list(self._deployments.keys()):
            self.unload_model(model_name)
        self.logger.info("All deployments stopped")

