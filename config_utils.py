"""
YAML-based configuration loader
Loads ``config.yaml`` from the project root directory and resolves
relative paths to absolute paths.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG_FILE = "config.yaml"


class ConfigManager:
    """
    Load and access the YAML configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML config file.  Defaults to ``config.yaml`` in the
        project root directory.
    """

    def __init__(self, config_path: str = None) -> None:
        if config_path is None:
            config_path = PROJECT_ROOT / DEFAULT_CONFIG_FILE
        self.config_path = Path(config_path)
        self.raw: Dict[str, Any] = self._load()
        self._resolve_paths()

    def _load(self) -> Dict[str, Any]:
        """Load the YAML config file"""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}

    def _resolve_paths(self) -> None:
        """Convert relative paths in the 'paths' section to absolute."""
        paths = self.raw.get("paths", {})
        for key in ("data_root", "prebuilt_dir", "save_dir"):
            val = paths.get(key, "")
            if val and not os.path.isabs(val):
                paths[key] = str(PROJECT_ROOT / val)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Lookup ``cfg.get('phase1', 'epochs')``."""
        return self.raw.get(section, {}).get(key, default)

    def get_path(self, path_key: str) -> str:
        """Return an absolute path for a ``paths.*`` entry."""
        return self.raw.get("paths", {}).get(path_key, "")
