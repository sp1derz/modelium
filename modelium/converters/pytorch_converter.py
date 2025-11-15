"""
PyTorch model converter.

Converts PyTorch models to TorchScript and ONNX formats.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.onnx

logger = logging.getLogger(__name__)


class PyTorchConverter:
    """
    Converter for PyTorch models.
    
    Supports:
    - PyTorch → TorchScript
    - PyTorch → ONNX
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_model(self, model_path: Path) -> torch.nn.Module:
        """Load PyTorch model from file."""
        self.logger.info(f"Loading PyTorch model from {model_path}")
        
        # Try different loading methods
        try:
            # Try loading as full model
            model = torch.load(model_path, map_location="cpu")
            
            # If it's a dict, extract the model
            if isinstance(model, dict):
                if "state_dict" in model:
                    # This is a checkpoint, need model architecture
                    raise ValueError("Checkpoint detected, model architecture required")
                elif "model" in model:
                    model = model["model"]
            
            # Ensure it's a module
            if not isinstance(model, torch.nn.Module):
                raise ValueError(f"Loaded object is not a torch.nn.Module: {type(model)}")
            
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def to_torchscript(
        self,
        model: torch.nn.Module,
        output_path: Path,
        example_inputs: Optional[torch.Tensor] = None,
        use_tracing: bool = True,
    ) -> Path:
        """
        Convert PyTorch model to TorchScript.
        
        Args:
            model: PyTorch model
            output_path: Path to save TorchScript model
            example_inputs: Example inputs for tracing
            use_tracing: Use tracing (True) or scripting (False)
            
        Returns:
            Path to saved TorchScript model
        """
        self.logger.info(f"Converting to TorchScript (tracing={use_tracing})")
        
        model.eval()
        
        try:
            if use_tracing:
                if example_inputs is None:
                    # Create default example inputs (assumes image input)
                    example_inputs = torch.randn(1, 3, 224, 224)
                
                traced_model = torch.jit.trace(model, example_inputs)
            else:
                traced_model = torch.jit.script(model)
            
            # Save
            traced_model.save(str(output_path))
            
            self.logger.info(f"Saved TorchScript model to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error converting to TorchScript: {e}")
            raise
    
    def to_onnx(
        self,
        model: torch.nn.Module,
        output_path: Path,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Path:
        """
        Convert PyTorch model to ONNX.
        
        Args:
            model: PyTorch model
            output_path: Path to save ONNX model
            input_names: List of input names
            output_names: List of output names
            input_shapes: Dictionary of input shapes
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            
        Returns:
            Path to saved ONNX model
        """
        self.logger.info(f"Converting to ONNX (opset {opset_version})")
        
        model.eval()
        
        # Set defaults
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]
        if input_shapes is None:
            input_shapes = {"input": [1, 3, 224, 224]}
        
        # Create dummy inputs
        dummy_inputs = []
        for name in input_names:
            shape = input_shapes.get(name, [1, 3, 224, 224])
            dummy_inputs.append(torch.randn(*shape))
        
        # If single input, unpack
        if len(dummy_inputs) == 1:
            dummy_inputs = dummy_inputs[0]
        else:
            dummy_inputs = tuple(dummy_inputs)
        
        # Setup dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {}
            for name in input_names:
                dynamic_axes[name] = {0: "batch_size"}
            for name in output_names:
                dynamic_axes[name] = {0: "batch_size"}
        
        try:
            torch.onnx.export(
                model,
                dummy_inputs,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
            )
            
            self.logger.info(f"Saved ONNX model to {output_path}")
            
            # Verify the model
            self._verify_onnx(output_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error converting to ONNX: {e}")
            raise
    
    def _verify_onnx(self, onnx_path: Path) -> None:
        """Verify ONNX model."""
        try:
            import onnx
            
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            
            self.logger.info("ONNX model verification passed")
            
        except Exception as e:
            self.logger.warning(f"ONNX verification failed: {e}")
    
    def convert_from_file(
        self,
        input_path: Path,
        output_path: Path,
        target_format: str = "onnx",
        **kwargs: Any,
    ) -> Path:
        """
        Convenience method to convert from file.
        
        Args:
            input_path: Path to PyTorch model
            output_path: Path to save converted model
            target_format: Target format ("onnx" or "torchscript")
            **kwargs: Additional arguments for conversion
            
        Returns:
            Path to converted model
        """
        model = self.load_model(input_path)
        
        if target_format.lower() == "onnx":
            return self.to_onnx(model, output_path, **kwargs)
        elif target_format.lower() == "torchscript":
            return self.to_torchscript(model, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

