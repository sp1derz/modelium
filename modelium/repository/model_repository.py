"""
Model Repository Manager

Manages the shared model repository that all runtimes use.
Handles moving models from incoming/ to repository/, creating proper structure.
"""

import logging
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RepositoryModel:
    """Model in the repository."""
    name: str
    path: Path
    size_bytes: int
    architecture: Optional[str] = None
    framework: str = "pytorch"
    added_at: Optional[datetime] = None
    config: Optional[Dict] = None


class ModelRepository:
    """
    Manages the centralized model repository.
    
    Structure:
        /models/
          ├── repository/      # All models here (mounted by runtimes)
          │   ├── gpt2/
          │   │   ├── config.json
          │   │   ├── model.safetensors
          │   │   └── tokenizer.json
          │   └── llama-2-7b/
          └── incoming/        # Drop zone (watcher monitors)
              └── new-model/
    
    Usage:
        repo = ModelRepository("/models/repository")
        repo.add_model_from_incoming("/models/incoming/gpt2", "gpt2")
    """
    
    def __init__(self, repository_path: str):
        """
        Initialize model repository.
        
        Args:
            repository_path: Path to repository directory (e.g., "/models/repository")
        """
        self.repository_path = Path(repository_path)
        self.repository_path.mkdir(parents=True, exist_ok=True)
        
        self._models: Dict[str, RepositoryModel] = {}
        self._scan_repository()
        
        logger.info(f"Model repository initialized at {self.repository_path}")
        logger.info(f"Found {len(self._models)} existing models")
    
    def _scan_repository(self):
        """Scan repository directory and index existing models."""
        if not self.repository_path.exists():
            return
        
        for model_dir in self.repository_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Check if it has config.json (HuggingFace model)
            config_file = model_dir / "config.json"
            if not config_file.exists():
                logger.warning(f"Skipping {model_dir.name}: no config.json")
                continue
            
            # Parse config
            try:
                with open(config_file) as f:
                    config = json.load(f)
                
                # Calculate size
                size_bytes = sum(
                    f.stat().st_size
                    for f in model_dir.rglob("*")
                    if f.is_file()
                )
                
                architecture = None
                if "architectures" in config and config["architectures"]:
                    architecture = config["architectures"][0]
                
                self._models[model_dir.name] = RepositoryModel(
                    name=model_dir.name,
                    path=model_dir,
                    size_bytes=size_bytes,
                    architecture=architecture,
                    config=config,
                )
                
                logger.debug(f"Indexed model: {model_dir.name} ({size_bytes / 1e9:.2f}GB)")
                
            except Exception as e:
                logger.error(f"Error indexing {model_dir.name}: {e}")
    
    def add_model_from_incoming(
        self,
        incoming_path: str,
        model_name: str,
        move: bool = True
    ) -> Optional[RepositoryModel]:
        """
        Add a model from incoming directory to repository.
        
        Args:
            incoming_path: Path to model in incoming directory
            model_name: Name for the model in repository
            move: If True, move (delete source). If False, copy.
        
        Returns:
            RepositoryModel if successful, None otherwise
        """
        incoming_path = Path(incoming_path)
        
        if not incoming_path.exists():
            logger.error(f"Incoming path doesn't exist: {incoming_path}")
            return None
        
        # Check if model has config.json
        config_file = incoming_path / "config.json"
        if not config_file.exists():
            logger.error(f"Model missing config.json: {incoming_path}")
            return None
        
        # Destination in repository
        repo_model_path = self.repository_path / model_name
        
        if repo_model_path.exists():
            logger.warning(f"Model {model_name} already exists in repository")
            return self._models.get(model_name)
        
        try:
            logger.info(f"Adding {model_name} to repository...")
            
            # Copy or move
            if move:
                shutil.move(str(incoming_path), str(repo_model_path))
                logger.info(f"  Moved from {incoming_path}")
            else:
                shutil.copytree(str(incoming_path), str(repo_model_path))
                logger.info(f"  Copied from {incoming_path}")
            
            # Parse config
            with open(repo_model_path / "config.json") as f:
                config = json.load(f)
            
            # Calculate size
            size_bytes = sum(
                f.stat().st_size
                for f in repo_model_path.rglob("*")
                if f.is_file()
            )
            
            architecture = None
            if "architectures" in config and config["architectures"]:
                architecture = config["architectures"][0]
            
            # Create repository model
            repo_model = RepositoryModel(
                name=model_name,
                path=repo_model_path,
                size_bytes=size_bytes,
                architecture=architecture,
                config=config,
                added_at=datetime.now(),
            )
            
            self._models[model_name] = repo_model
            
            logger.info(f"  ✅ {model_name} added to repository")
            logger.info(f"     Size: {size_bytes / 1e9:.2f}GB")
            logger.info(f"     Architecture: {architecture}")
            logger.info(f"     Path: {repo_model_path}")
            
            return repo_model
            
        except Exception as e:
            logger.error(f"Failed to add {model_name}: {e}")
            # Clean up partial copy
            if repo_model_path.exists():
                shutil.rmtree(repo_model_path)
            return None
    
    def get_model(self, model_name: str) -> Optional[RepositoryModel]:
        """Get model from repository by name."""
        return self._models.get(model_name)
    
    def list_models(self) -> List[RepositoryModel]:
        """List all models in repository."""
        return list(self._models.values())
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get absolute path to model in repository."""
        model = self._models.get(model_name)
        return model.path if model else None
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from repository.
        
        Args:
            model_name: Name of model to remove
        
        Returns:
            True if successful
        """
        if model_name not in self._models:
            logger.warning(f"Model {model_name} not in repository")
            return False
        
        try:
            model = self._models[model_name]
            logger.info(f"Removing {model_name} from repository...")
            
            # Delete directory
            shutil.rmtree(model.path)
            
            # Remove from index
            del self._models[model_name]
            
            logger.info(f"  ✅ {model_name} removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove {model_name}: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get repository statistics."""
        total_size = sum(m.size_bytes for m in self._models.values())
        
        # Group by architecture
        by_arch = {}
        for model in self._models.values():
            arch = model.architecture or "unknown"
            if arch not in by_arch:
                by_arch[arch] = 0
            by_arch[arch] += 1
        
        return {
            "total_models": len(self._models),
            "total_size_gb": total_size / 1e9,
            "by_architecture": by_arch,
            "models": [m.name for m in self._models.values()],
        }

