"""
Framework detection from model artifacts.

Analyzes file extensions, directory structures, and file contents
to determine the ML framework used.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from modelium.core.descriptor import Framework

logger = logging.getLogger(__name__)


class FrameworkDetector:
    """
    Detects the ML framework from model artifacts.
    
    Supports PyTorch, TensorFlow, ONNX, JAX/Flax, HuggingFace, and more.
    """
    
    # File extension mappings
    FRAMEWORK_EXTENSIONS: Dict[str, Framework] = {
        ".pt": Framework.PYTORCH,
        ".pth": Framework.PYTORCH,
        ".ckpt": Framework.PYTORCH,
        ".pb": Framework.TENSORFLOW,
        ".h5": Framework.KERAS,
        ".keras": Framework.KERAS,
        ".onnx": Framework.ONNX,
        ".msgpack": Framework.JAXFLAX,
        ".safetensors": Framework.HUGGINGFACE,
    }
    
    # Config file indicators
    CONFIG_INDICATORS: Dict[str, Framework] = {
        "config.json": Framework.HUGGINGFACE,
        "pytorch_model.bin": Framework.PYTORCH,
        "tf_model.h5": Framework.TENSORFLOW,
        "model.onnx": Framework.ONNX,
    }
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect(self, artifact_path: Path) -> Tuple[Framework, Optional[str]]:
        """
        Detect framework from artifact path.
        
        Args:
            artifact_path: Path to model artifact (file or directory)
            
        Returns:
            Tuple of (Framework, version) where version may be None
        """
        if artifact_path.is_file():
            return self._detect_from_file(artifact_path)
        elif artifact_path.is_dir():
            return self._detect_from_directory(artifact_path)
        else:
            self.logger.warning(f"Artifact path does not exist: {artifact_path}")
            return Framework.UNKNOWN, None
    
    def _detect_from_file(self, file_path: Path) -> Tuple[Framework, Optional[str]]:
        """Detect framework from a single file."""
        extension = file_path.suffix.lower()
        
        # Check extension mapping
        if extension in self.FRAMEWORK_EXTENSIONS:
            framework = self.FRAMEWORK_EXTENSIONS[extension]
            version = self._extract_version(file_path, framework)
            self.logger.info(f"Detected {framework.value} from extension {extension}")
            return framework, version
        
        # For .bin files, need deeper inspection
        if extension == ".bin":
            framework = self._inspect_binary_file(file_path)
            return framework, None
        
        self.logger.warning(f"Unknown file extension: {extension}")
        return Framework.UNKNOWN, None
    
    def _detect_from_directory(self, dir_path: Path) -> Tuple[Framework, Optional[str]]:
        """Detect framework from directory structure."""
        files = list(dir_path.rglob("*"))
        file_names = [f.name for f in files if f.is_file()]
        
        # Check for HuggingFace structure
        if "config.json" in file_names:
            config_path = dir_path / "config.json"
            if self._is_huggingface_config(config_path):
                self.logger.info("Detected HuggingFace model from config.json")
                return Framework.HUGGINGFACE, self._extract_hf_version(config_path)
        
        # Check for specific model files
        for indicator, framework in self.CONFIG_INDICATORS.items():
            if indicator in file_names:
                self.logger.info(f"Detected {framework.value} from {indicator}")
                return framework, None
        
        # Count files by extension
        extension_counts: Dict[Framework, int] = {}
        for file in files:
            if file.is_file():
                ext = file.suffix.lower()
                if ext in self.FRAMEWORK_EXTENSIONS:
                    fw = self.FRAMEWORK_EXTENSIONS[ext]
                    extension_counts[fw] = extension_counts.get(fw, 0) + 1
        
        # Return most common framework
        if extension_counts:
            most_common = max(extension_counts.items(), key=lambda x: x[1])
            self.logger.info(f"Detected {most_common[0].value} from file extensions")
            return most_common[0], None
        
        self.logger.warning(f"Could not detect framework from directory: {dir_path}")
        return Framework.UNKNOWN, None
    
    def _is_huggingface_config(self, config_path: Path) -> bool:
        """Check if config.json is a HuggingFace config."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # HuggingFace configs typically have these keys
            hf_keys = {"architectures", "model_type", "transformers_version"}
            return bool(hf_keys & set(config.keys()))
        except Exception as e:
            self.logger.warning(f"Error reading config.json: {e}")
            return False
    
    def _extract_hf_version(self, config_path: Path) -> Optional[str]:
        """Extract transformers version from HuggingFace config."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("transformers_version")
        except Exception:
            return None
    
    def _extract_version(self, file_path: Path, framework: Framework) -> Optional[str]:
        """Extract framework version from model file."""
        # This is framework-specific and would require loading the model
        # For now, return None (can be enhanced later)
        return None
    
    def _inspect_binary_file(self, file_path: Path) -> Framework:
        """Inspect binary file to determine framework."""
        try:
            # Read first few bytes to detect magic numbers
            with open(file_path, "rb") as f:
                header = f.read(1024)
            
            # PyTorch pickle files
            if b"PK\x03\x04" in header[:10]:  # ZIP header (PyTorch)
                return Framework.PYTORCH
            
            # TensorFlow SavedModel
            if b"tensorflow" in header.lower():
                return Framework.TENSORFLOW
            
            # ONNX has protobuf header
            if b"\x08\x03" in header[:10]:
                return Framework.ONNX
            
        except Exception as e:
            self.logger.warning(f"Error inspecting binary file: {e}")
        
        return Framework.UNKNOWN
    
    def get_all_model_files(self, artifact_path: Path) -> List[Path]:
        """Get all model-related files from artifact path."""
        if artifact_path.is_file():
            return [artifact_path]
        
        model_files = []
        for ext in self.FRAMEWORK_EXTENSIONS.keys():
            model_files.extend(artifact_path.rglob(f"*{ext}"))
        
        # Also include common config files
        for pattern in ["config.json", "*.yaml", "*.yml", "*.txt"]:
            model_files.extend(artifact_path.rglob(pattern))
        
        return model_files

