"""
Main model analyzer that coordinates framework-specific analyzers.
"""

import logging
from pathlib import Path
from typing import Optional

from modelium.core.descriptor import Framework, ModelDescriptor, DescriptorBuilder
from modelium.core.analyzers.framework_detector import FrameworkDetector
from modelium.core.analyzers.pytorch_analyzer import PyTorchAnalyzer
from modelium.core.analyzers.onnx_analyzer import ONNXAnalyzer
from modelium.core.analyzers.huggingface_analyzer import HuggingFaceAnalyzer
from modelium.core.analyzers.security_scanner import SecurityScanner

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """
    Main analyzer that orchestrates model analysis.
    
    Detects framework and delegates to framework-specific analyzers.
    """
    
    def __init__(self) -> None:
        self.detector = FrameworkDetector()
        self.security_scanner = SecurityScanner()
        
        # Framework-specific analyzers
        self.pytorch_analyzer = PyTorchAnalyzer()
        self.onnx_analyzer = ONNXAnalyzer()
        self.huggingface_analyzer = HuggingFaceAnalyzer()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(
        self,
        artifact_path: Path,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> ModelDescriptor:
        """
        Analyze a model artifact and generate a descriptor.
        
        Args:
            artifact_path: Path to model artifact
            model_id: Optional model ID (generated if not provided)
            model_name: Optional model name (derived from path if not provided)
            
        Returns:
            Complete ModelDescriptor
        """
        self.logger.info(f"Analyzing model artifact: {artifact_path}")
        
        # Generate ID and name if not provided
        if not model_id:
            model_id = self._generate_id(artifact_path)
        if not model_name:
            model_name = artifact_path.stem
        
        # Detect framework
        framework, fw_version = self.detector.detect(artifact_path)
        self.logger.info(f"Detected framework: {framework.value}")
        
        # Create descriptor builder
        builder = DescriptorBuilder().create_new(
            model_id=model_id,
            name=model_name,
            framework=framework,
        )
        
        if fw_version:
            builder.set_framework_version(fw_version)
        
        # Add file information
        model_files = self.detector.get_all_model_files(artifact_path)
        builder.add_files(
            [str(f) for f in model_files],
            primary=str(artifact_path),
        )
        
        # Run security scan
        security_scan = self.security_scanner.scan(artifact_path)
        builder.set_security_scan(security_scan)
        
        # Delegate to framework-specific analyzer
        try:
            if framework == Framework.PYTORCH:
                self._analyze_pytorch(artifact_path, builder)
            elif framework == Framework.ONNX:
                self._analyze_onnx(artifact_path, builder)
            elif framework == Framework.HUGGINGFACE:
                self._analyze_huggingface(artifact_path, builder)
            elif framework == Framework.TENSORFLOW:
                self._analyze_tensorflow(artifact_path, builder)
            else:
                self.logger.warning(f"No specific analyzer for {framework.value}")
        except Exception as e:
            self.logger.error(f"Error during framework-specific analysis: {e}")
            # Continue with basic descriptor
        
        descriptor = builder.build()
        self.logger.info(f"Analysis complete for {model_name}")
        
        return descriptor
    
    def _generate_id(self, artifact_path: Path) -> str:
        """Generate a unique model ID."""
        import hashlib
        import time
        
        # Use path and timestamp for uniqueness
        content = f"{artifact_path.absolute()}{time.time()}"
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]
    
    def _analyze_pytorch(self, artifact_path: Path, builder: DescriptorBuilder) -> None:
        """Analyze PyTorch model."""
        self.logger.info("Running PyTorch-specific analysis")
        result = self.pytorch_analyzer.analyze(artifact_path)
        
        if result.model_type:
            builder.set_model_type(result.model_type)
        if result.architecture:
            builder.set_architecture(result.architecture)
        if result.operations:
            builder.add_operations(result.operations)
        if result.layers:
            builder.add_layers(result.layers)
        if result.input_info:
            builder.set_inputs(
                result.input_info["names"],
                result.input_info["shapes"],
                result.input_info["dtypes"],
            )
        if result.output_info:
            builder.set_outputs(
                result.output_info["names"],
                result.output_info["shapes"],
                result.output_info["dtypes"],
            )
        if result.resources:
            builder.set_resources(result.resources)
        if result.dependencies:
            builder.add_dependencies(result.dependencies)
    
    def _analyze_onnx(self, artifact_path: Path, builder: DescriptorBuilder) -> None:
        """Analyze ONNX model."""
        self.logger.info("Running ONNX-specific analysis")
        result = self.onnx_analyzer.analyze(artifact_path)
        
        if result.model_type:
            builder.set_model_type(result.model_type)
        if result.operations:
            builder.add_operations(result.operations)
        if result.input_info:
            builder.set_inputs(
                result.input_info["names"],
                result.input_info["shapes"],
                result.input_info["dtypes"],
            )
        if result.output_info:
            builder.set_outputs(
                result.output_info["names"],
                result.output_info["shapes"],
                result.output_info["dtypes"],
            )
        if result.resources:
            builder.set_resources(result.resources)
    
    def _analyze_huggingface(self, artifact_path: Path, builder: DescriptorBuilder) -> None:
        """Analyze HuggingFace model."""
        self.logger.info("Running HuggingFace-specific analysis")
        result = self.huggingface_analyzer.analyze(artifact_path)
        
        if result.model_type:
            builder.set_model_type(result.model_type)
        if result.architecture:
            builder.set_architecture(result.architecture)
        if result.tokenizer:
            builder.set_tokenizer(result.tokenizer)
        if result.resources:
            builder.set_resources(result.resources)
        if result.license_info:
            builder.set_license(result.license_info)
        if result.metadata:
            builder.add_metadata(result.metadata)
    
    def _analyze_tensorflow(self, artifact_path: Path, builder: DescriptorBuilder) -> None:
        """Analyze TensorFlow model."""
        self.logger.info("Running TensorFlow-specific analysis")
        # TensorFlow analyzer would be implemented similarly
        # For now, just log
        self.logger.warning("TensorFlow analyzer not yet implemented")

