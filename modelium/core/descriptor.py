"""
Model descriptor schema and builder.

The descriptor is a comprehensive JSON representation of a model artifact
that includes structure, operations, shapes, dependencies, and metadata.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Framework(str, Enum):
    """Supported ML frameworks."""
    
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    JAXFLAX = "jax-flax"
    HUGGINGFACE = "huggingface"
    KERAS = "keras"
    MXNET = "mxnet"
    PADDLE = "paddle"
    UNKNOWN = "unknown"


class ModelType(str, Enum):
    """Model architecture categories."""
    
    VISION = "vision"
    NLP = "nlp"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    TABULAR = "tabular"
    REINFORCEMENT = "reinforcement"
    GENERATIVE = "generative"
    UNKNOWN = "unknown"


class OpInfo(BaseModel):
    """Information about a single operation in the model."""
    
    name: str
    type: str
    input_shapes: List[List[Optional[int]]] = Field(default_factory=list)
    output_shapes: List[List[Optional[int]]] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    custom: bool = False


class LayerInfo(BaseModel):
    """Information about a model layer."""
    
    name: str
    type: str
    params: int = 0
    trainable: bool = True


class TokenizerInfo(BaseModel):
    """Tokenizer configuration."""
    
    type: str
    vocab_size: int
    special_tokens: Dict[str, Union[str, int]] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class ResourceEstimate(BaseModel):
    """Estimated resource requirements."""
    
    parameters: int
    memory_bytes: int
    compute_flops: Optional[int] = None
    gpu_memory_bytes: Optional[int] = None


class LicenseInfo(BaseModel):
    """License and compliance information."""
    
    license_type: Optional[str] = None
    license_url: Optional[str] = None
    attribution_required: bool = False
    commercial_use: bool = True
    derived_from: List[str] = Field(default_factory=list)


class SecurityScan(BaseModel):
    """Security scan results."""
    
    has_pickle: bool = False
    has_custom_ops: bool = False
    suspicious_imports: List[str] = Field(default_factory=list)
    binary_files: List[str] = Field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    warnings: List[str] = Field(default_factory=list)


class ModelDescriptor(BaseModel):
    """
    Complete descriptor for a model artifact.
    
    This is the core data structure that flows through the Forge pipeline.
    """
    
    # Identification
    id: str
    name: str
    version: str = "unknown"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Framework and type
    framework: Framework
    framework_version: Optional[str] = None
    model_type: ModelType = ModelType.UNKNOWN
    
    # Architecture
    architecture: Optional[str] = None
    operations: List[OpInfo] = Field(default_factory=list)
    layers: List[LayerInfo] = Field(default_factory=list)
    
    # Input/Output specifications
    input_names: List[str] = Field(default_factory=list)
    output_names: List[str] = Field(default_factory=list)
    input_shapes: Dict[str, List[Optional[int]]] = Field(default_factory=dict)
    output_shapes: Dict[str, List[Optional[int]]] = Field(default_factory=dict)
    input_dtypes: Dict[str, str] = Field(default_factory=dict)
    output_dtypes: Dict[str, str] = Field(default_factory=dict)
    
    # Tokenizer (for NLP models)
    tokenizer: Optional[TokenizerInfo] = None
    
    # Resources
    resources: Optional[ResourceEstimate] = None
    
    # Dependencies
    dependencies: Dict[str, str] = Field(default_factory=dict)
    python_version: Optional[str] = None
    cuda_version: Optional[str] = None
    
    # Files
    file_paths: List[str] = Field(default_factory=list)
    primary_file: Optional[str] = None
    config_files: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    # License and security
    license_info: Optional[LicenseInfo] = None
    security_scan: Optional[SecurityScan] = None
    
    # Source information
    source_url: Optional[str] = None
    source_commit: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "resnet50-v1",
                "name": "ResNet50",
                "version": "1.0",
                "framework": "pytorch",
                "model_type": "vision",
                "architecture": "ResNet",
                "input_names": ["input"],
                "output_names": ["output"],
                "input_shapes": {"input": [1, 3, 224, 224]},
                "output_shapes": {"output": [1, 1000]},
            }
        }


class DescriptorBuilder:
    """
    Builder class for constructing ModelDescriptor objects.
    
    Coordinates various analyzers to build a comprehensive descriptor.
    """
    
    def __init__(self) -> None:
        self.descriptor: Optional[ModelDescriptor] = None
    
    def create_new(
        self,
        model_id: str,
        name: str,
        framework: Framework,
    ) -> "DescriptorBuilder":
        """Initialize a new descriptor."""
        self.descriptor = ModelDescriptor(
            id=model_id,
            name=name,
            framework=framework,
        )
        return self
    
    def set_framework_version(self, version: str) -> "DescriptorBuilder":
        """Set framework version."""
        if self.descriptor:
            self.descriptor.framework_version = version
        return self
    
    def set_model_type(self, model_type: ModelType) -> "DescriptorBuilder":
        """Set model type."""
        if self.descriptor:
            self.descriptor.model_type = model_type
        return self
    
    def set_architecture(self, architecture: str) -> "DescriptorBuilder":
        """Set architecture name."""
        if self.descriptor:
            self.descriptor.architecture = architecture
        return self
    
    def add_operations(self, operations: List[OpInfo]) -> "DescriptorBuilder":
        """Add operations."""
        if self.descriptor:
            self.descriptor.operations.extend(operations)
        return self
    
    def add_layers(self, layers: List[LayerInfo]) -> "DescriptorBuilder":
        """Add layers."""
        if self.descriptor:
            self.descriptor.layers.extend(layers)
        return self
    
    def set_inputs(
        self,
        names: List[str],
        shapes: Dict[str, List[Optional[int]]],
        dtypes: Dict[str, str],
    ) -> "DescriptorBuilder":
        """Set input specifications."""
        if self.descriptor:
            self.descriptor.input_names = names
            self.descriptor.input_shapes = shapes
            self.descriptor.input_dtypes = dtypes
        return self
    
    def set_outputs(
        self,
        names: List[str],
        shapes: Dict[str, List[Optional[int]]],
        dtypes: Dict[str, str],
    ) -> "DescriptorBuilder":
        """Set output specifications."""
        if self.descriptor:
            self.descriptor.output_names = names
            self.descriptor.output_shapes = shapes
            self.descriptor.output_dtypes = dtypes
        return self
    
    def set_tokenizer(self, tokenizer: TokenizerInfo) -> "DescriptorBuilder":
        """Set tokenizer information."""
        if self.descriptor:
            self.descriptor.tokenizer = tokenizer
        return self
    
    def set_resources(self, resources: ResourceEstimate) -> "DescriptorBuilder":
        """Set resource estimates."""
        if self.descriptor:
            self.descriptor.resources = resources
        return self
    
    def add_dependencies(self, dependencies: Dict[str, str]) -> "DescriptorBuilder":
        """Add dependencies."""
        if self.descriptor:
            self.descriptor.dependencies.update(dependencies)
        return self
    
    def add_files(self, files: List[str], primary: Optional[str] = None) -> "DescriptorBuilder":
        """Add file paths."""
        if self.descriptor:
            self.descriptor.file_paths.extend(files)
            if primary:
                self.descriptor.primary_file = primary
        return self
    
    def add_metadata(self, metadata: Dict[str, Any]) -> "DescriptorBuilder":
        """Add metadata."""
        if self.descriptor:
            self.descriptor.metadata.update(metadata)
        return self
    
    def set_license(self, license_info: LicenseInfo) -> "DescriptorBuilder":
        """Set license information."""
        if self.descriptor:
            self.descriptor.license_info = license_info
        return self
    
    def set_security_scan(self, security_scan: SecurityScan) -> "DescriptorBuilder":
        """Set security scan results."""
        if self.descriptor:
            self.descriptor.security_scan = security_scan
        return self
    
    def build(self) -> ModelDescriptor:
        """Build and return the descriptor."""
        if not self.descriptor:
            raise ValueError("Descriptor not initialized")
        return self.descriptor
    
    def to_json(self, filepath: Path) -> None:
        """Save descriptor to JSON file."""
        if not self.descriptor:
            raise ValueError("Descriptor not initialized")
        
        with open(filepath, "w") as f:
            f.write(self.descriptor.model_dump_json(indent=2))
    
    @staticmethod
    def from_json(filepath: Path) -> ModelDescriptor:
        """Load descriptor from JSON file."""
        with open(filepath, "r") as f:
            return ModelDescriptor.model_validate_json(f.read())

