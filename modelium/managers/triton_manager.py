"""
Triton Runtime Manager

Manages model loading/unloading in Triton Inference Server.
Uses Triton's model repository and management API.
"""

import logging
import requests
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class TritonManager:
    """
    Manages models in Triton Inference Server.
    
    Triton supports dynamic model loading/unloading via its management API.
    
    Prerequisites:
        - Triton running with model repository: tritonserver --model-repository=/models
        - Model repository mounted at same path Modelium uses
    
    Usage:
        manager = TritonManager(
            endpoint="http://localhost:8003",
            model_repository="/models/repository"
        )
        
        # Load a model
        manager.load_model("gpt2", "/models/repository/gpt2")
        
        # Unload when idle
        manager.unload_model("gpt2")
    """
    
    def __init__(self, endpoint: str, model_repository: str, timeout: int = 300):
        """
        Initialize Triton manager.
        
        Args:
            endpoint: Triton server endpoint (e.g., "http://localhost:8003")
            model_repository: Path to Triton's model repository
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.model_repository = Path(model_repository)
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._loaded_models: Dict[str, Dict] = {}  # Track loaded models
    
    def health_check(self) -> bool:
        """Check if Triton is healthy and ready."""
        try:
            response = self.session.get(
                f"{self.endpoint}/v2/health/ready",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Triton health check failed: {e}")
            return False
    
    def load_model(
        self,
        model_name: str,
        model_path: Path,
        config: Optional[Dict] = None
    ) -> bool:
        """
        Load a model into Triton.
        
        This creates the proper Triton model directory structure if needed,
        then calls Triton's load API.
        
        Args:
            model_name: Name for the model in Triton
            model_path: Path to model files (HuggingFace format)
            config: Optional Triton config overrides
        
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"📦 Loading {model_name} into Triton...")
            
            # Triton model path
            triton_model_path = self.model_repository / model_name
            
            # Check if already in repository
            if not triton_model_path.exists():
                self.logger.info(f"   Preparing Triton model structure...")
                
                # Create Triton model directory structure
                # /models/repository/
                #   └── model_name/
                #       ├── config.pbtxt  (Triton config)
                #       └── 1/            (version directory)
                #           └── model.plan or model files
                
                triton_model_path.mkdir(parents=True, exist_ok=True)
                version_dir = triton_model_path / "1"
                version_dir.mkdir(exist_ok=True)
                
                # Copy model files to version directory
                for file in model_path.iterdir():
                    if file.is_file():
                        shutil.copy2(file, version_dir / file.name)
                    elif file.is_dir() and file.name not in ['.git']:
                        shutil.copytree(file, version_dir / file.name, dirs_exist_ok=True)
                
                # Generate Triton config.pbtxt
                self._generate_triton_config(
                    triton_model_path,
                    model_name,
                    config or {}
                )
                
                self.logger.info(f"   ✅ Model structure ready at {triton_model_path}")
            
            # Call Triton's load API
            self.logger.info(f"   Calling Triton load API...")
            response = self.session.post(
                f"{self.endpoint}/v2/repository/models/{model_name}/load",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Wait for model to be ready
                self.logger.info(f"   Waiting for model to be ready...")
                if self._wait_for_model_ready(model_name, timeout=60):
                    self._loaded_models[model_name] = {
                        "path": str(model_path),
                        "loaded_at": time.time(),
                    }
                    self.logger.info(f"   ✅ {model_name} loaded and ready in Triton")
                    return True
                else:
                    self.logger.error(f"   ❌ {model_name} loaded but not ready")
                    return False
            else:
                self.logger.error(f"   ❌ Triton load failed: {response.status_code}")
                self.logger.error(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading {model_name} in Triton: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from Triton.
        
        Args:
            model_name: Name of model to unload
        
        Returns:
            True if successful
        """
        try:
            if model_name not in self._loaded_models:
                self.logger.warning(f"{model_name} not tracked as loaded")
                return False
            
            self.logger.info(f"📤 Unloading {model_name} from Triton...")
            
            response = self.session.post(
                f"{self.endpoint}/v2/repository/models/{model_name}/unload",
                timeout=30
            )
            
            if response.status_code == 200:
                del self._loaded_models[model_name]
                self.logger.info(f"   ✅ {model_name} unloaded from Triton")
                return True
            else:
                self.logger.error(f"   ❌ Triton unload failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unloading {model_name}: {e}")
            return False
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        try:
            response = self.session.get(
                f"{self.endpoint}/v2/models/{model_name}/ready",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def list_loaded_models(self) -> List[str]:
        """List all models currently loaded in Triton."""
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
    
    def get_model_stats(self, model_name: str) -> Optional[Dict]:
        """Get statistics for a loaded model."""
        try:
            response = self.session.get(
                f"{self.endpoint}/v2/models/{model_name}/stats",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get stats for {model_name}: {e}")
            return None
    
    def _wait_for_model_ready(self, model_name: str, timeout: int = 60) -> bool:
        """Wait for model to be ready after loading."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_model_loaded(model_name):
                return True
            time.sleep(1)
        
        return False
    
    def _generate_triton_config(
        self,
        model_dir: Path,
        model_name: str,
        config: Dict
    ):
        """
        Generate Triton config.pbtxt for a model.
        
        This creates a basic config that works for PyTorch/ONNX models.
        For production, users should customize this.
        """
        # Basic config for PyTorch backend
        config_content = f"""
name: "{model_name}"
backend: "pytorch"
max_batch_size: {config.get('max_batch_size', 32)}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [-1]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, -1]
  }}
]

dynamic_batching {{
  preferred_batch_size: [1, 2, 4, 8, 16, 32]
  max_queue_delay_microseconds: {config.get('max_queue_delay', 100)}
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [{config.get('gpu_id', 0)}]
  }}
]
"""
        
        with open(model_dir / "config.pbtxt", "w") as f:
            f.write(config_content.strip())
        
        self.logger.debug(f"Generated config.pbtxt for {model_name}")

