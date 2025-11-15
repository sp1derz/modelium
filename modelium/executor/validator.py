"""
Plan validator for conversion plans.

Validates that plans are safe to execute and conform to expected schema.
"""

import logging
import re
from typing import List, Tuple, Optional

from modelium.modelium_llm.schemas import ConversionPlan, ConversionStep

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when plan validation fails."""
    pass


class PlanValidator:
    """
    Validates conversion plans before execution.
    
    Checks for:
    - Schema conformance
    - Dangerous commands
    - Resource limits
    - Dependency validity
    - Script safety
    """
    
    # Dangerous command patterns
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",  # Recursive delete from root
        r"dd\s+if=.*\s+of=/dev/",  # Disk operations
        r":\(\)\{\s*:\|:&\s*\};:",  # Fork bomb
        r"curl\s+.*\|\s*bash",  # Pipe to bash
        r"wget\s+.*\|\s*bash",
        r"chmod\s+777",  # Overly permissive permissions
        r"shutdown",  # System shutdown
        r"reboot",  # System reboot
    ]
    
    # Allowed commands (whitelist for production)
    ALLOWED_COMMANDS = {
        "python", "python3",
        "pip", "pip3",
        "torch", "onnx",
        "trtexec",  # TensorRT
        "polygraphy",  # NVIDIA tool
        "onnxruntime",
        "trtllm-build",  # TRT-LLM
        "nvidia-smi",
        "ls", "cd", "pwd", "mkdir", "cp", "mv",
    }
    
    def __init__(self, strict: bool = True) -> None:
        self.strict = strict
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, plan: ConversionPlan) -> Tuple[bool, List[str]]:
        """
        Validate a conversion plan.
        
        Args:
            plan: The conversion plan to validate
            
        Returns:
            Tuple of (is_valid, list of warnings/errors)
        """
        errors = []
        warnings = []
        
        # Validate basic structure
        if not plan.model_id:
            errors.append("Missing model_id")
        
        if not plan.steps:
            errors.append("No conversion steps provided")
        
        # Validate steps
        for i, step in enumerate(plan.steps):
            step_errors, step_warnings = self._validate_step(step, i)
            errors.extend(step_errors)
            warnings.extend(step_warnings)
        
        # Validate resource requirements
        resource_warnings = self._validate_resources(plan)
        warnings.extend(resource_warnings)
        
        # Validate dependencies between steps
        dep_errors = self._validate_dependencies(plan.steps)
        errors.extend(dep_errors)
        
        # Check for dangerous patterns in all commands/scripts
        danger_errors = self._check_dangerous_patterns(plan)
        errors.extend(danger_errors)
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.logger.error(f"Plan validation failed: {errors}")
        
        if warnings:
            self.logger.warning(f"Plan validation warnings: {warnings}")
        
        return is_valid, errors + warnings
    
    def _validate_step(
        self,
        step: ConversionStep,
        index: int,
    ) -> Tuple[List[str], List[str]]:
        """Validate a single conversion step."""
        errors = []
        warnings = []
        
        # Check that step has either command or script
        if not step.command and not step.script:
            errors.append(f"Step {index} ({step.name}): Must have either command or script")
        
        if step.command and step.script:
            warnings.append(f"Step {index} ({step.name}): Has both command and script, command will be used")
        
        # Validate timeout
        if step.timeout < 10:
            warnings.append(f"Step {index} ({step.name}): Very short timeout ({step.timeout}s)")
        elif step.timeout > 7200:  # 2 hours
            warnings.append(f"Step {index} ({step.name}): Very long timeout ({step.timeout}s)")
        
        # Validate command if present
        if step.command:
            cmd_errors, cmd_warnings = self._validate_command(step.command, step.name)
            errors.extend(cmd_errors)
            warnings.extend(cmd_warnings)
        
        # Validate script if present
        if step.script:
            script_errors, script_warnings = self._validate_script(step.script, step.name)
            errors.extend(script_errors)
            warnings.extend(script_warnings)
        
        return errors, warnings
    
    def _validate_command(
        self,
        command: str,
        step_name: str,
    ) -> Tuple[List[str], List[str]]:
        """Validate a shell command."""
        errors = []
        warnings = []
        
        # Extract the base command
        base_command = command.split()[0] if command.strip() else ""
        
        # In strict mode, check whitelist
        if self.strict:
            if base_command not in self.ALLOWED_COMMANDS:
                errors.append(f"Step {step_name}: Command '{base_command}' not in whitelist")
        
        # Check for pipes to shell
        if "|" in command and ("bash" in command or "sh" in command):
            errors.append(f"Step {step_name}: Piping to shell is not allowed")
        
        # Check for command substitution
        if "$(" in command or "`" in command:
            warnings.append(f"Step {step_name}: Command substitution detected")
        
        return errors, warnings
    
    def _validate_script(
        self,
        script: str,
        step_name: str,
    ) -> Tuple[List[str], List[str]]:
        """Validate a Python script."""
        errors = []
        warnings = []
        
        # Check for dangerous imports
        dangerous_imports = ["os.system", "subprocess.call", "eval", "exec", "__import__"]
        for danger in dangerous_imports:
            if danger in script:
                warnings.append(f"Step {step_name}: Script contains '{danger}'")
        
        # Check for file operations outside working directory
        if "../" in script or "/.." in script:
            warnings.append(f"Step {step_name}: Script accesses parent directories")
        
        # Check for network operations
        if any(lib in script for lib in ["requests", "urllib", "http", "socket"]):
            warnings.append(f"Step {step_name}: Script contains network operations")
        
        return errors, warnings
    
    def _validate_resources(self, plan: ConversionPlan) -> List[str]:
        """Validate resource requirements."""
        warnings = []
        
        # Check memory requirements
        if plan.required_memory_gb > 128:
            warnings.append(f"Very high memory requirement: {plan.required_memory_gb}GB")
        
        if plan.required_gpu_memory_gb and plan.required_gpu_memory_gb > 80:
            warnings.append(f"Very high GPU memory requirement: {plan.required_gpu_memory_gb}GB")
        
        # Check conversion time
        if plan.estimated_conversion_time_minutes > 120:
            warnings.append(f"Very long estimated conversion time: {plan.estimated_conversion_time_minutes} minutes")
        
        return warnings
    
    def _validate_dependencies(self, steps: List[ConversionStep]) -> List[str]:
        """Validate dependencies between steps."""
        errors = []
        
        step_names = {step.name for step in steps}
        
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(f"Step '{step.name}' depends on unknown step '{dep}'")
        
        return errors
    
    def _check_dangerous_patterns(self, plan: ConversionPlan) -> List[str]:
        """Check for dangerous patterns in all commands and scripts."""
        errors = []
        
        for step in plan.steps:
            content = (step.command or "") + (step.script or "")
            
            for pattern in self.DANGEROUS_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    errors.append(f"Step '{step.name}' contains dangerous pattern: {pattern}")
        
        return errors

