"""
PyTorch-specific model analyzer.

Extracts architecture information, operations, and metadata from PyTorch models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from modelium.core.descriptor import (
    ModelType,
    OpInfo,
    LayerInfo,
    ResourceEstimate,
)

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Container for PyTorch analysis results."""
    
    def __init__(self) -> None:
        self.model_type: Optional[ModelType] = None
        self.architecture: Optional[str] = None
        self.operations: List[OpInfo] = []
        self.layers: List[LayerInfo] = []
        self.input_info: Optional[Dict[str, Any]] = None
        self.output_info: Optional[Dict[str, Any]] = None
        self.resources: Optional[ResourceEstimate] = None
        self.dependencies: Dict[str, str] = {}


class PyTorchAnalyzer:
    """
    Analyzer for PyTorch models.
    
    Supports .pt, .pth, and .ckpt files.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(self, model_path: Path) -> AnalysisResult:
        """
        Analyze PyTorch model.
        
        Args:
            model_path: Path to PyTorch model file
            
        Returns:
            AnalysisResult with extracted information
        """
        result = AnalysisResult()
        
        try:
            # Load model
            self.logger.info(f"Loading PyTorch model from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Extract information based on checkpoint structure
            if isinstance(checkpoint, dict):
                result = self._analyze_checkpoint_dict(checkpoint)
            elif isinstance(checkpoint, torch.nn.Module):
                result = self._analyze_module(checkpoint)
            else:
                self.logger.warning(f"Unknown checkpoint type: {type(checkpoint)}")
            
            # Add PyTorch version as dependency
            result.dependencies["torch"] = torch.__version__
            
        except Exception as e:
            self.logger.error(f"Error analyzing PyTorch model: {e}")
        
        return result
    
    def _analyze_checkpoint_dict(self, checkpoint: Dict[str, Any]) -> AnalysisResult:
        """Analyze checkpoint dictionary."""
        result = AnalysisResult()
        
        # Check for state_dict
        state_dict = None
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        
        if state_dict:
            result = self._analyze_state_dict(state_dict)
        
        # Extract architecture name if available
        if "arch" in checkpoint:
            result.architecture = str(checkpoint["arch"])
        elif "model_name" in checkpoint:
            result.architecture = str(checkpoint["model_name"])
        
        # Extract metadata
        if "config" in checkpoint:
            result = self._analyze_config(checkpoint["config"], result)
        
        return result
    
    def _analyze_module(self, module: torch.nn.Module) -> AnalysisResult:
        """Analyze PyTorch module."""
        result = AnalysisResult()
        
        # Get architecture name
        result.architecture = module.__class__.__name__
        
        # Extract layers
        result.layers = self._extract_layers(module)
        
        # Count parameters
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Estimate memory (4 bytes per float32 parameter)
        memory_bytes = total_params * 4
        
        result.resources = ResourceEstimate(
            parameters=total_params,
            memory_bytes=memory_bytes,
        )
        
        # Infer model type from architecture
        result.model_type = self._infer_model_type(result.architecture)
        
        return result
    
    def _analyze_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> AnalysisResult:
        """Analyze state dictionary."""
        result = AnalysisResult()
        
        # Extract layer information from parameter names
        layer_names = set()
        total_params = 0
        memory_bytes = 0
        
        for name, tensor in state_dict.items():
            # Extract layer name (before the last dot)
            if "." in name:
                layer_name = name.rsplit(".", 1)[0]
                layer_names.add(layer_name)
            
            # Count parameters
            params = tensor.numel()
            total_params += params
            memory_bytes += params * tensor.element_size()
        
        # Create layer info
        for layer_name in sorted(layer_names):
            layer_type = self._infer_layer_type(layer_name)
            layer_params = sum(
                t.numel() for n, t in state_dict.items()
                if n.startswith(layer_name + ".")
            )
            
            result.layers.append(LayerInfo(
                name=layer_name,
                type=layer_type,
                params=layer_params,
            ))
        
        result.resources = ResourceEstimate(
            parameters=total_params,
            memory_bytes=memory_bytes,
        )
        
        # Infer architecture from layer patterns
        result.architecture = self._infer_architecture(list(layer_names))
        result.model_type = self._infer_model_type(result.architecture or "")
        
        return result
    
    def _extract_layers(self, module: torch.nn.Module) -> List[LayerInfo]:
        """Extract layer information from module."""
        layers = []
        
        for name, child in module.named_children():
            layer_type = child.__class__.__name__
            params = sum(p.numel() for p in child.parameters())
            
            layers.append(LayerInfo(
                name=name,
                type=layer_type,
                params=params,
            ))
        
        return layers
    
    def _infer_layer_type(self, layer_name: str) -> str:
        """Infer layer type from name."""
        name_lower = layer_name.lower()
        
        if "conv" in name_lower:
            return "Conv"
        elif "bn" in name_lower or "batchnorm" in name_lower:
            return "BatchNorm"
        elif "linear" in name_lower or "fc" in name_lower:
            return "Linear"
        elif "lstm" in name_lower:
            return "LSTM"
        elif "gru" in name_lower:
            return "GRU"
        elif "attention" in name_lower or "attn" in name_lower:
            return "Attention"
        elif "embed" in name_lower:
            return "Embedding"
        elif "pool" in name_lower:
            return "Pooling"
        else:
            return "Unknown"
    
    def _infer_architecture(self, layer_names: List[str]) -> str:
        """Infer model architecture from layer names."""
        names_str = " ".join(layer_names).lower()
        
        # Common architectures
        if "resnet" in names_str:
            return "ResNet"
        elif "vgg" in names_str:
            return "VGG"
        elif "efficientnet" in names_str:
            return "EfficientNet"
        elif "mobilenet" in names_str:
            return "MobileNet"
        elif "bert" in names_str:
            return "BERT"
        elif "gpt" in names_str:
            return "GPT"
        elif "t5" in names_str:
            return "T5"
        elif "vit" in names_str:
            return "ViT"
        elif "unet" in names_str:
            return "UNet"
        else:
            return "Unknown"
    
    def _infer_model_type(self, architecture: str) -> ModelType:
        """Infer model type from architecture."""
        arch_lower = architecture.lower()
        
        # Vision models
        vision_keywords = ["resnet", "vgg", "efficientnet", "mobilenet", "vit", "convnet"]
        if any(k in arch_lower for k in vision_keywords):
            return ModelType.VISION
        
        # NLP models
        nlp_keywords = ["bert", "gpt", "t5", "transformer", "llama", "mistral"]
        if any(k in arch_lower for k in nlp_keywords):
            return ModelType.NLP
        
        # Audio models
        audio_keywords = ["wav2vec", "whisper", "audio"]
        if any(k in arch_lower for k in audio_keywords):
            return ModelType.AUDIO
        
        return ModelType.UNKNOWN
    
    def _analyze_config(self, config: Dict[str, Any], result: AnalysisResult) -> AnalysisResult:
        """Extract information from config dictionary."""
        if "model_type" in config:
            model_type_str = config["model_type"].lower()
            if "vision" in model_type_str:
                result.model_type = ModelType.VISION
            elif "nlp" in model_type_str or "language" in model_type_str:
                result.model_type = ModelType.NLP
        
        return result

