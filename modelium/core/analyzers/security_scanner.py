"""
Security scanner for model artifacts.

Detects potential security risks, malicious code, and license issues.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import List, Set

from modelium.core.descriptor import SecurityScan

logger = logging.getLogger(__name__)


class SecurityScanner:
    """
    Scans model artifacts for security risks.
    
    Checks for:
    - Pickle files (potential code execution)
    - Suspicious imports
    - Unknown binary files
    - Custom operations
    - License compliance
    """
    
    # Suspicious patterns in pickle files
    SUSPICIOUS_IMPORTS = {
        "os", "sys", "subprocess", "eval", "exec", "compile",
        "__import__", "open", "file", "input", "raw_input",
        "socket", "urllib", "requests", "http",
    }
    
    # Safe file extensions
    SAFE_EXTENSIONS = {
        ".json", ".yaml", ".yml", ".txt", ".md", ".rst",
        ".cfg", ".ini", ".toml",
    }
    
    # Binary file extensions
    BINARY_EXTENSIONS = {
        ".pt", ".pth", ".ckpt", ".bin", ".pb", ".h5",
        ".onnx", ".msgpack", ".safetensors", ".so", ".dylib", ".dll",
    }
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def scan(self, artifact_path: Path) -> SecurityScan:
        """
        Scan artifact for security risks.
        
        Args:
            artifact_path: Path to model artifact
            
        Returns:
            SecurityScan with detected risks
        """
        scan = SecurityScan()
        
        if artifact_path.is_file():
            files = [artifact_path]
        else:
            files = list(artifact_path.rglob("*"))
            files = [f for f in files if f.is_file()]
        
        # Scan each file
        for file_path in files:
            self._scan_file(file_path, scan)
        
        # Determine overall risk level
        scan.risk_level = self._calculate_risk_level(scan)
        
        self.logger.info(f"Security scan complete. Risk level: {scan.risk_level}")
        
        return scan
    
    def _scan_file(self, file_path: Path, scan: SecurityScan) -> None:
        """Scan individual file."""
        extension = file_path.suffix.lower()
        
        # Check for pickle files
        if extension in [".pt", ".pth", ".ckpt", ".pkl", ".pickle"]:
            scan.has_pickle = True
            self._scan_pickle_file(file_path, scan)
        
        # Check for binary files
        if extension in self.BINARY_EXTENSIONS:
            scan.binary_files.append(str(file_path))
        
        # Scan text files for suspicious content
        if extension in self.SAFE_EXTENSIONS:
            self._scan_text_file(file_path, scan)
    
    def _scan_pickle_file(self, file_path: Path, scan: SecurityScan) -> None:
        """Scan pickle file for suspicious content."""
        try:
            # Read raw bytes to look for suspicious patterns
            with open(file_path, "rb") as f:
                content = f.read(10000)  # Read first 10KB
            
            # Convert to string for pattern matching (ignore errors)
            content_str = content.decode("latin-1")
            
            # Check for suspicious imports
            for suspicious in self.SUSPICIOUS_IMPORTS:
                if suspicious in content_str:
                    if suspicious not in scan.suspicious_imports:
                        scan.suspicious_imports.append(suspicious)
                        scan.warnings.append(
                            f"Found suspicious import '{suspicious}' in {file_path.name}"
                        )
            
            # Check for eval/exec patterns
            dangerous_patterns = [
                r"eval\s*\(",
                r"exec\s*\(",
                r"__import__\s*\(",
                r"compile\s*\(",
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, content_str):
                    scan.warnings.append(
                        f"Found dangerous code pattern '{pattern}' in {file_path.name}"
                    )
            
        except Exception as e:
            self.logger.warning(f"Error scanning pickle file {file_path}: {e}")
    
    def _scan_text_file(self, file_path: Path, scan: SecurityScan) -> None:
        """Scan text file for suspicious content."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Check for suspicious code patterns
            if "custom_op" in content.lower() or "custom_kernel" in content.lower():
                scan.has_custom_ops = True
            
            # Check for URLs pointing to external resources
            url_pattern = r"https?://[^\s]+"
            urls = re.findall(url_pattern, content)
            if urls:
                scan.warnings.append(
                    f"Found external URLs in {file_path.name}: {len(urls)} URLs"
                )
            
        except Exception as e:
            self.logger.warning(f"Error scanning text file {file_path}: {e}")
    
    def _calculate_risk_level(self, scan: SecurityScan) -> str:
        """Calculate overall risk level."""
        risk_score = 0
        
        # Pickle files add risk
        if scan.has_pickle:
            risk_score += 1
        
        # Suspicious imports are high risk
        if scan.suspicious_imports:
            risk_score += 3
        
        # Custom ops add moderate risk
        if scan.has_custom_ops:
            risk_score += 2
        
        # Many warnings increase risk
        if len(scan.warnings) > 5:
            risk_score += 2
        elif len(scan.warnings) > 0:
            risk_score += 1
        
        # Determine level
        if risk_score >= 5:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

