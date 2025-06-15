"""Configuration management."""
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load(self, name: str) -> Dict[str, Any]:
        """Load YAML config file."""
        if name not in self._configs:
            config_path = self.config_dir / f"{name}.yaml"
            with open(config_path) as f:
                self._configs[name] = yaml.safe_load(f)
        return self._configs[name]

# Global config instance
config = Config()