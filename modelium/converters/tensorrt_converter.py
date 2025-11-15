"""
TensorRT converter.

Converts ONNX models to TensorRT engines with optimization.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class TensorRTConverter:
    """
    Converter for TensorRT engines.
    
    Uses trtexec to build optimized TensorRT engines from ONNX models.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Check if trtexec is available
        self._check_trtexec()
    
    def _check_trtexec(self) -> None:
        """Check if trtexec is available."""
        try:
            result = subprocess.run(
                ["trtexec", "--help"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                self.logger.warning("trtexec not available or not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("trtexec not found in PATH")
    
    def convert(
        self,
        onnx_path: Path,
        output_path: Path,
        precision: str = "fp32",
        min_shapes: Optional[Dict[str, List[int]]] = None,
        opt_shapes: Optional[Dict[str, List[int]]] = None,
        max_shapes: Optional[Dict[str, List[int]]] = None,
        max_batch_size: Optional[int] = None,
        workspace_size_gb: int = 4,
        calibration_cache: Optional[Path] = None,
        verbose: bool = False,
    ) -> Path:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            precision: Precision mode ("fp32", "fp16", "int8")
            min_shapes: Minimum input shapes for dynamic shapes
            opt_shapes: Optimal input shapes for dynamic shapes
            max_shapes: Maximum input shapes for dynamic shapes
            max_batch_size: Maximum batch size
            workspace_size_gb: Workspace size in GB
            calibration_cache: Path to INT8 calibration cache
            verbose: Enable verbose logging
            
        Returns:
            Path to TensorRT engine
        """
        self.logger.info(f"Converting ONNX to TensorRT ({precision})")
        
        # Build trtexec command
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={output_path}",
        ]
        
        # Add precision flags
        if precision.lower() == "fp16":
            cmd.append("--fp16")
        elif precision.lower() == "int8":
            cmd.append("--int8")
            if calibration_cache:
                cmd.append(f"--calib={calibration_cache}")
        
        # Add dynamic shapes if provided
        if min_shapes:
            min_str = self._format_shapes(min_shapes)
            cmd.append(f"--minShapes={min_str}")
        
        if opt_shapes:
            opt_str = self._format_shapes(opt_shapes)
            cmd.append(f"--optShapes={opt_str}")
        
        if max_shapes:
            max_str = self._format_shapes(max_shapes)
            cmd.append(f"--maxShapes={max_str}")
        
        # Add batch size
        if max_batch_size:
            cmd.append(f"--maxBatch={max_batch_size}")
        
        # Add workspace size (convert GB to MB)
        workspace_mb = workspace_size_gb * 1024
        cmd.append(f"--workspace={workspace_mb}")
        
        # Add verbose flag
        if verbose:
            cmd.append("--verbose")
        
        # Run conversion
        try:
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"TensorRT conversion failed: {result.stderr}")
                raise RuntimeError(f"TensorRT conversion failed: {result.stderr}")
            
            self.logger.info("TensorRT conversion successful")
            self.logger.debug(f"Output: {result.stdout}")
            
            return output_path
            
        except subprocess.TimeoutExpired:
            self.logger.error("TensorRT conversion timed out")
            raise
        except Exception as e:
            self.logger.error(f"Error during TensorRT conversion: {e}")
            raise
    
    def _format_shapes(self, shapes: Dict[str, List[int]]) -> str:
        """Format shapes for trtexec."""
        shape_strs = []
        for name, shape in shapes.items():
            shape_str = "x".join(map(str, shape))
            shape_strs.append(f"{name}:{shape_str}")
        return ",".join(shape_strs)
    
    def profile(
        self,
        engine_path: Path,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        duration: int = 10,
    ) -> Dict[str, float]:
        """
        Profile TensorRT engine performance.
        
        Args:
            engine_path: Path to TensorRT engine
            input_shapes: Input shapes for profiling
            duration: Profiling duration in seconds
            
        Returns:
            Dictionary with profiling results
        """
        self.logger.info(f"Profiling TensorRT engine for {duration}s")
        
        cmd = [
            "trtexec",
            f"--loadEngine={engine_path}",
            f"--duration={duration}",
        ]
        
        if input_shapes:
            shapes_str = self._format_shapes(input_shapes)
            cmd.append(f"--shapes={shapes_str}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 60,
            )
            
            if result.returncode != 0:
                self.logger.error(f"Profiling failed: {result.stderr}")
                raise RuntimeError(f"Profiling failed: {result.stderr}")
            
            # Parse output
            metrics = self._parse_profiling_output(result.stdout)
            
            self.logger.info(f"Profiling complete: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during profiling: {e}")
            raise
    
    def _parse_profiling_output(self, output: str) -> Dict[str, float]:
        """Parse trtexec profiling output."""
        metrics = {}
        
        # Look for key metrics in output
        for line in output.split("\n"):
            if "mean:" in line.lower() and "ms" in line.lower():
                # Extract latency
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() == "mean:":
                        try:
                            metrics["mean_latency_ms"] = float(parts[i + 1])
                        except (IndexError, ValueError):
                            pass
            
            if "throughput:" in line.lower():
                # Extract throughput
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() == "throughput:":
                        try:
                            metrics["throughput_qps"] = float(parts[i + 1])
                        except (IndexError, ValueError):
                            pass
        
        return metrics
    
    def calibrate_int8(
        self,
        onnx_path: Path,
        calibration_data_path: Path,
        calibration_cache_path: Path,
        batch_size: int = 1,
    ) -> Path:
        """
        Generate INT8 calibration cache.
        
        Args:
            onnx_path: Path to ONNX model
            calibration_data_path: Path to calibration dataset
            calibration_cache_path: Path to save calibration cache
            batch_size: Batch size for calibration
            
        Returns:
            Path to calibration cache
        """
        self.logger.info("Generating INT8 calibration cache")
        
        # This would require implementing calibration logic
        # For now, just log a warning
        self.logger.warning(
            "INT8 calibration requires custom implementation with calibration dataset"
        )
        
        return calibration_cache_path

