"""
HuggingFace-specific model analyzer.

Extracts information from HuggingFace model repositories.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from modelium.core.descriptor import (
    ModelType,
    TokenizerInfo,
    ResourceEstimate,
    LicenseInfo,
)

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Container for HuggingFace analysis results."""
    
    def __init__(self) -> None:
        self.model_type: Optional[ModelType] = None
        self.architecture: Optional[str] = None
        self.tokenizer: Optional[TokenizerInfo] = None
        self.resources: Optional[ResourceEstimate] = None
        self.license_info: Optional[LicenseInfo] = None
        self.metadata: Dict[str, Any] = {}


class HuggingFaceAnalyzer:
    """
    Analyzer for HuggingFace models.
    
    Reads config.json, tokenizer files, and model cards.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(self, model_path: Path) -> AnalysisResult:
        """
        Analyze HuggingFace model directory.
        
        Args:
            model_path: Path to HuggingFace model directory
            
        Returns:
            AnalysisResult with extracted information
        """
        result = AnalysisResult()
        
        try:
            # Read config.json
            config_path = model_path / "config.json"
            if config_path.exists():
                result = self._analyze_config(config_path, result)
            
            # Read tokenizer config
            tokenizer_config = model_path / "tokenizer_config.json"
            if tokenizer_config.exists():
                result = self._analyze_tokenizer(tokenizer_config, result)
            
            # Read model card
            readme_path = model_path / "README.md"
            if readme_path.exists():
                result = self._analyze_model_card(readme_path, result)
            
            # Estimate resources from config
            if config_path.exists():
                result.resources = self._estimate_resources(config_path)
            
        except Exception as e:
            self.logger.error(f"Error analyzing HuggingFace model: {e}")
        
        return result
    
    def _analyze_config(self, config_path: Path, result: AnalysisResult) -> AnalysisResult:
        """Analyze config.json."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Extract architecture
            if "architectures" in config and config["architectures"]:
                result.architecture = config["architectures"][0]
            
            # Extract model type
            if "model_type" in config:
                result.model_type = self._map_model_type(config["model_type"])
            
            # Store full config as metadata
            result.metadata["config"] = config
            
            self.logger.info(f"Detected architecture: {result.architecture}")
            
        except Exception as e:
            self.logger.error(f"Error reading config.json: {e}")
        
        return result
    
    def _analyze_tokenizer(
        self,
        tokenizer_path: Path,
        result: AnalysisResult
    ) -> AnalysisResult:
        """Analyze tokenizer configuration."""
        try:
            with open(tokenizer_path, "r") as f:
                tokenizer_config = json.load(f)
            
            # Extract tokenizer info
            vocab_size = tokenizer_config.get("vocab_size", 0)
            tokenizer_type = tokenizer_config.get("tokenizer_class", "unknown")
            
            # Extract special tokens
            special_tokens = {}
            for key in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token"]:
                if key in tokenizer_config:
                    special_tokens[key] = tokenizer_config[key]
            
            result.tokenizer = TokenizerInfo(
                type=tokenizer_type,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                config=tokenizer_config,
            )
            
            self.logger.info(f"Detected tokenizer: {tokenizer_type}")
            
        except Exception as e:
            self.logger.error(f"Error reading tokenizer config: {e}")
        
        return result
    
    def _analyze_model_card(self, readme_path: Path, result: AnalysisResult) -> AnalysisResult:
        """Analyze README.md model card."""
        try:
            with open(readme_path, "r") as f:
                content = f.read()
            
            # Extract license information (simple pattern matching)
            license_type = None
            if "license:" in content.lower():
                # Try to find license in YAML frontmatter
                lines = content.split("\n")
                for line in lines:
                    if line.strip().startswith("license:"):
                        license_type = line.split(":", 1)[1].strip()
                        break
            
            if license_type:
                result.license_info = LicenseInfo(
                    license_type=license_type,
                    commercial_use=self._is_commercial_license(license_type),
                )
            
            # Store model card in metadata
            result.metadata["model_card"] = content
            
        except Exception as e:
            self.logger.error(f"Error reading model card: {e}")
        
        return result
    
    def _estimate_resources(self, config_path: Path) -> ResourceEstimate:
        """Estimate resource requirements from config."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Rough parameter estimation based on config
            # This is very approximate and model-specific
            params = 0
            
            # For transformer models
            if "num_layers" in config or "n_layer" in config:
                num_layers = config.get("num_layers", config.get("n_layer", 12))
                hidden_size = config.get("hidden_size", config.get("n_embd", 768))
                vocab_size = config.get("vocab_size", 50000)
                
                # Very rough estimate
                params_per_layer = hidden_size * hidden_size * 12  # Approximate
                params = num_layers * params_per_layer + vocab_size * hidden_size
            
            # Estimate memory (4 bytes per float32 parameter)
            memory_bytes = params * 4 if params > 0 else 0
            
            return ResourceEstimate(
                parameters=params,
                memory_bytes=memory_bytes,
            )
            
        except Exception as e:
            self.logger.error(f"Error estimating resources: {e}")
            return ResourceEstimate(parameters=0, memory_bytes=0)
    
    def _map_model_type(self, model_type_str: str) -> ModelType:
        """Map HuggingFace model type to ModelType enum."""
        model_type_lower = model_type_str.lower()
        
        # Vision models
        if any(k in model_type_lower for k in ["vit", "deit", "beit", "clip"]):
            return ModelType.VISION
        
        # NLP models
        if any(k in model_type_lower for k in ["bert", "gpt", "t5", "llama", "mistral", "roberta"]):
            return ModelType.NLP
        
        # Audio models
        if any(k in model_type_lower for k in ["wav2vec", "whisper", "hubert"]):
            return ModelType.AUDIO
        
        # Multimodal
        if any(k in model_type_lower for k in ["clip", "blip", "flamingo"]):
            return ModelType.MULTIMODAL
        
        return ModelType.UNKNOWN
    
    def _is_commercial_license(self, license_type: str) -> bool:
        """Check if license allows commercial use."""
        commercial_licenses = [
            "apache-2.0", "mit", "bsd", "cc-by", "openrail",
        ]
        
        license_lower = license_type.lower()
        return any(lic in license_lower for lic in commercial_licenses)

