"""
Sandboxed execution environment for conversion plans.

Executes conversion steps in isolated Docker containers with resource limits.
"""

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import docker
from docker.types import Mount

from modelium.modelium_llm.schemas import ConversionPlan, ConversionStep

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when step execution fails."""
    pass


class ExecutionResult:
    """Result of executing a conversion step."""
    
    def __init__(
        self,
        step_name: str,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        duration_seconds: float = 0.0,
        error_message: Optional[str] = None,
    ) -> None:
        self.step_name = step_name
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.duration_seconds = duration_seconds
        self.error_message = error_message


class PlanExecutionResult:
    """Result of executing an entire conversion plan."""
    
    def __init__(self, plan_id: str) -> None:
        self.plan_id = plan_id
        self.step_results: List[ExecutionResult] = []
        self.success = False
        self.total_duration_seconds = 0.0
        self.artifacts: List[str] = []
        self.error_message: Optional[str] = None


class SandboxExecutor:
    """
    Executes conversion plans in isolated Docker containers.
    
    Features:
    - Network isolation
    - Resource limits (CPU, memory, GPU)
    - Filesystem isolation
    - Timeout enforcement
    - Artifact collection
    """
    
    # Base image for conversions (includes PyTorch, ONNX, TensorRT)
    BASE_IMAGE = "nvcr.io/nvidia/pytorch:23.10-py3"
    
    def __init__(
        self,
        workspace_dir: Path,
        memory_limit: str = "32g",
        enable_gpu: bool = True,
        network_mode: str = "none",
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_limit = memory_limit
        self.enable_gpu = enable_gpu
        self.network_mode = network_mode
        
        # Initialize Docker client
        self.docker_client = docker.from_env()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute_plan(
        self,
        plan: ConversionPlan,
        input_files: Dict[str, Path],
    ) -> PlanExecutionResult:
        """
        Execute a complete conversion plan.
        
        Args:
            plan: The conversion plan to execute
            input_files: Dictionary mapping file names to paths
            
        Returns:
            PlanExecutionResult with execution details
        """
        result = PlanExecutionResult(plan_id=plan.plan_id)
        start_time = time.time()
        
        self.logger.info(f"Executing plan {plan.plan_id} with {len(plan.steps)} steps")
        
        # Create workspace for this execution
        exec_workspace = self.workspace_dir / plan.plan_id
        exec_workspace.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy input files to workspace
            for name, path in input_files.items():
                dest = exec_workspace / name
                if path.is_file():
                    shutil.copy2(path, dest)
                elif path.is_dir():
                    shutil.copytree(path, dest, dirs_exist_ok=True)
            
            # Execute each step
            for step in plan.steps:
                self.logger.info(f"Executing step: {step.name}")
                
                # Check dependencies
                if not self._check_dependencies(step, result):
                    error_msg = f"Dependencies not met for step {step.name}"
                    result.step_results.append(ExecutionResult(
                        step_name=step.name,
                        success=False,
                        error_message=error_msg,
                    ))
                    result.error_message = error_msg
                    break
                
                # Execute step
                step_result = self._execute_step(step, exec_workspace)
                result.step_results.append(step_result)
                
                if not step_result.success:
                    self.logger.error(f"Step {step.name} failed: {step_result.error_message}")
                    result.error_message = f"Step {step.name} failed"
                    break
            
            # Check if all steps succeeded
            result.success = all(r.success for r in result.step_results)
            
            # Collect artifacts
            if result.success:
                result.artifacts = self._collect_artifacts(exec_workspace)
                self.logger.info(f"Collected {len(result.artifacts)} artifacts")
            
        except Exception as e:
            self.logger.error(f"Error executing plan: {e}")
            result.error_message = str(e)
        
        result.total_duration_seconds = time.time() - start_time
        
        self.logger.info(f"Plan execution {'succeeded' if result.success else 'failed'} in {result.total_duration_seconds:.2f}s")
        
        return result
    
    def _execute_step(
        self,
        step: ConversionStep,
        workspace: Path,
    ) -> ExecutionResult:
        """Execute a single conversion step."""
        start_time = time.time()
        
        try:
            # Prepare command
            if step.command:
                command = ["bash", "-c", step.command]
            elif step.script:
                # Write script to file and execute
                script_file = workspace / f"{step.name}_script.py"
                script_file.write_text(step.script)
                command = ["python3", f"{step.name}_script.py"]
            else:
                return ExecutionResult(
                    step_name=step.name,
                    success=False,
                    error_message="No command or script provided",
                )
            
            # Setup device requests for GPU
            device_requests = None
            if self.enable_gpu:
                device_requests = [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]
            
            # Run container
            container = self.docker_client.containers.run(
                image=self.BASE_IMAGE,
                command=command,
                working_dir="/workspace",
                mounts=[
                    Mount(target="/workspace", source=str(workspace.absolute()), type="bind")
                ],
                mem_limit=self.memory_limit,
                network_mode=self.network_mode,
                device_requests=device_requests,
                detach=True,
                remove=False,
            )
            
            # Wait for completion with timeout
            try:
                exit_code = container.wait(timeout=step.timeout)
                if isinstance(exit_code, dict):
                    exit_code = exit_code.get("StatusCode", 0)
            except Exception as e:
                # Timeout or error
                container.kill()
                container.remove()
                return ExecutionResult(
                    step_name=step.name,
                    success=False,
                    error_message=f"Timeout or error: {e}",
                    duration_seconds=time.time() - start_time,
                )
            
            # Get logs
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
            
            # Clean up
            container.remove()
            
            # Determine success
            success = exit_code == 0
            
            duration = time.time() - start_time
            
            return ExecutionResult(
                step_name=step.name,
                success=success,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_seconds=duration,
                error_message=stderr if not success else None,
            )
            
        except Exception as e:
            self.logger.error(f"Error executing step {step.name}: {e}")
            return ExecutionResult(
                step_name=step.name,
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )
    
    def _check_dependencies(
        self,
        step: ConversionStep,
        result: PlanExecutionResult,
    ) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True
        
        completed_steps = {r.step_name for r in result.step_results if r.success}
        
        return all(dep in completed_steps for dep in step.dependencies)
    
    def _collect_artifacts(self, workspace: Path) -> List[str]:
        """Collect generated artifacts from workspace."""
        artifacts = []
        
        # Common artifact patterns
        patterns = [
            "*.onnx",
            "*.plan",
            "*.engine",
            "*.pt",
            "*.pth",
            "*.pb",
            "config.pbtxt",
            "*.json",
        ]
        
        for pattern in patterns:
            artifacts.extend([str(f.relative_to(workspace)) for f in workspace.rglob(pattern)])
        
        return artifacts

