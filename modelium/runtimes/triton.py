"""
Triton Inference Server configuration generator.

Generates config.pbtxt files for Triton model repository.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TritonConfigGenerator:
    """
    Generates Triton Inference Server configurations.
    
    Creates config.pbtxt files and model repository structure.
    """
    
    # Platform mappings
    PLATFORM_MAP = {
        "pytorch": "pytorch_libtorch",
        "torchscript": "pytorch_libtorch",
        "onnx": "onnxruntime_onnx",
        "tensorrt": "tensorrt_plan",
        "tensorflow": "tensorflow_savedmodel",
    }
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_config(
        self,
        model_name: str,
        platform: str,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        max_batch_size: int = 8,
        dynamic_batching: Optional[Dict[str, Any]] = None,
        instance_group: Optional[List[Dict[str, Any]]] = None,
        optimization: Optional[Dict[str, Any]] = None,
        version_policy: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate Triton config.pbtxt content.
        
        Args:
            model_name: Name of the model
            platform: Model platform (pytorch, onnx, tensorrt, etc.)
            inputs: List of input specifications
            outputs: List of output specifications
            max_batch_size: Maximum batch size
            dynamic_batching: Dynamic batching configuration
            instance_group: Instance group configuration
            optimization: Optimization configuration
            version_policy: Version policy configuration
            
        Returns:
            config.pbtxt content as string
        """
        self.logger.info(f"Generating Triton config for {model_name}")
        
        # Map platform to Triton platform
        triton_platform = self.PLATFORM_MAP.get(platform.lower(), platform)
        
        config = []
        config.append(f'name: "{model_name}"')
        config.append(f'platform: "{triton_platform}"')
        config.append(f'max_batch_size: {max_batch_size}')
        config.append('')
        
        # Add inputs
        for inp in inputs:
            config.append('input {')
            config.append(f'  name: "{inp["name"]}"')
            config.append(f'  data_type: {self._map_dtype(inp["dtype"])}')
            config.append(f'  dims: [{self._format_dims(inp["shape"])}]')
            if inp.get("optional"):
                config.append('  optional: true')
            config.append('}')
        config.append('')
        
        # Add outputs
        for out in outputs:
            config.append('output {')
            config.append(f'  name: "{out["name"]}"')
            config.append(f'  data_type: {self._map_dtype(out["dtype"])}')
            config.append(f'  dims: [{self._format_dims(out["shape"])}]')
            config.append('}')
        config.append('')
        
        # Add dynamic batching if specified
        if dynamic_batching:
            config.append('dynamic_batching {')
            if "max_queue_delay_microseconds" in dynamic_batching:
                config.append(f'  max_queue_delay_microseconds: {dynamic_batching["max_queue_delay_microseconds"]}')
            if "preferred_batch_size" in dynamic_batching:
                for size in dynamic_batching["preferred_batch_size"]:
                    config.append(f'  preferred_batch_size: {size}')
            config.append('}')
            config.append('')
        
        # Add instance group if specified
        if instance_group:
            for group in instance_group:
                config.append('instance_group {')
                config.append(f'  count: {group.get("count", 1)}')
                config.append(f'  kind: {group.get("kind", "KIND_GPU")}')
                if "gpus" in group:
                    config.append(f'  gpus: [{", ".join(map(str, group["gpus"]))}]')
                config.append('}')
            config.append('')
        else:
            # Default instance group
            config.append('instance_group {')
            config.append('  count: 1')
            config.append('  kind: KIND_GPU')
            config.append('}')
            config.append('')
        
        # Add optimization if specified
        if optimization:
            config.append('optimization {')
            if "graph" in optimization:
                config.append('  graph {')
                config.append(f'    level: {optimization["graph"].get("level", 0)}')
                config.append('  }')
            if "cuda" in optimization:
                config.append('  cuda {')
                if "graphs" in optimization["cuda"]:
                    config.append(f'    graphs: {optimization["cuda"]["graphs"]}')
                config.append('  }')
            config.append('}')
            config.append('')
        
        # Add version policy if specified
        if version_policy:
            config.append('version_policy {')
            if version_policy.get("latest"):
                config.append(f'  latest {{')
                config.append(f'    num_versions: {version_policy["latest"]["num_versions"]}')
                config.append(f'  }}')
            config.append('}')
            config.append('')
        
        return '\n'.join(config)
    
    def _map_dtype(self, dtype: str) -> str:
        """Map dtype string to Triton data type."""
        dtype_map = {
            "float32": "TYPE_FP32",
            "float16": "TYPE_FP16",
            "int32": "TYPE_INT32",
            "int64": "TYPE_INT64",
            "int8": "TYPE_INT8",
            "uint8": "TYPE_UINT8",
            "bool": "TYPE_BOOL",
            "string": "TYPE_STRING",
        }
        
        dtype_lower = dtype.lower()
        return dtype_map.get(dtype_lower, "TYPE_FP32")
    
    def _format_dims(self, shape: List[int]) -> str:
        """Format shape dimensions for config."""
        # Skip batch dimension (first dim)
        dims = shape[1:] if len(shape) > 1 else shape
        
        # Replace -1 or None with -1 for dynamic dimensions
        formatted = []
        for dim in dims:
            if dim is None or dim < 0:
                formatted.append("-1")
            else:
                formatted.append(str(dim))
        
        return ", ".join(formatted)
    
    def create_model_repository(
        self,
        repository_path: Path,
        model_name: str,
        model_file: Path,
        config_content: str,
        version: int = 1,
    ) -> Path:
        """
        Create Triton model repository structure.
        
        Args:
            repository_path: Path to model repository root
            model_name: Name of the model
            model_file: Path to model file
            config_content: config.pbtxt content
            version: Model version
            
        Returns:
            Path to created model directory
        """
        self.logger.info(f"Creating model repository for {model_name}")
        
        # Create directory structure
        model_dir = repository_path / model_name
        version_dir = model_dir / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config.pbtxt
        config_path = model_dir / "config.pbtxt"
        config_path.write_text(config_content)
        
        # Copy model file to version directory
        # Determine model filename based on platform
        if model_file.suffix == ".onnx":
            dest_name = "model.onnx"
        elif model_file.suffix in [".pt", ".pth"]:
            dest_name = "model.pt"
        elif model_file.suffix in [".plan", ".engine"]:
            dest_name = "model.plan"
        else:
            dest_name = model_file.name
        
        import shutil
        dest_path = version_dir / dest_name
        shutil.copy2(model_file, dest_path)
        
        self.logger.info(f"Model repository created at {model_dir}")
        
        return model_dir
    
    def generate_ensemble_config(
        self,
        ensemble_name: str,
        steps: List[Dict[str, Any]],
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
    ) -> str:
        """
        Generate ensemble model configuration.
        
        Args:
            ensemble_name: Name of ensemble model
            steps: List of pipeline steps
            inputs: Ensemble inputs
            outputs: Ensemble outputs
            
        Returns:
            Ensemble config.pbtxt content
        """
        config = []
        config.append(f'name: "{ensemble_name}"')
        config.append('platform: "ensemble"')
        config.append('')
        
        # Add inputs
        for inp in inputs:
            config.append('input {')
            config.append(f'  name: "{inp["name"]}"')
            config.append(f'  data_type: {self._map_dtype(inp["dtype"])}')
            config.append(f'  dims: [{self._format_dims(inp["shape"])}]')
            config.append('}')
        config.append('')
        
        # Add outputs
        for out in outputs:
            config.append('output {')
            config.append(f'  name: "{out["name"]}"')
            config.append(f'  data_type: {self._map_dtype(out["dtype"])}')
            config.append(f'  dims: [{self._format_dims(out["shape"])}]')
            config.append('}')
        config.append('')
        
        # Add ensemble scheduling
        config.append('ensemble_scheduling {')
        for i, step in enumerate(steps):
            config.append('  step {')
            config.append(f'    model_name: "{step["model_name"]}"')
            config.append(f'    model_version: {step.get("version", -1)}')
            
            # Input mappings
            for inp_map in step.get("input_map", []):
                config.append('    input_map {')
                config.append(f'      key: "{inp_map["key"]}"')
                config.append(f'      value: "{inp_map["value"]}"')
                config.append('    }')
            
            # Output mappings
            for out_map in step.get("output_map", []):
                config.append('    output_map {')
                config.append(f'      key: "{out_map["key"]}"')
                config.append(f'      value: "{out_map["value"]}"')
                config.append('    }')
            
            config.append('  }')
        config.append('}')
        
        return '\n'.join(config)

