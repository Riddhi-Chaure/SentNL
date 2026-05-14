"""
SentNL — Configuration Manager

Loads YAML configs with inheritance from base.yaml.
Task-specific configs override base defaults via deep merge.
"""

import os
import copy
import yaml
from pathlib import Path


class ConfigManager:
    """Centralized configuration loader with base-config inheritance."""

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _CONFIGS_DIR = _PROJECT_ROOT / "configs"

    def __init__(self):
        self._base_config = self._load_yaml(self._CONFIGS_DIR / "base.yaml")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_base_config(self) -> dict:
        """Return a deep copy of the global base configuration."""
        return copy.deepcopy(self._base_config)

    def get_task_config(self, task_name: str) -> dict:
        """
        Load a task-specific config and deep-merge it over base.yaml.

        Args:
            task_name: One of 'spam', 'sentiment', 'topics'.

        Returns:
            Merged configuration dictionary.
        """
        task_path = self._CONFIGS_DIR / f"{task_name}.yaml"
        if not task_path.exists():
            raise FileNotFoundError(
                f"Task config not found: {task_path}"
            )
        task_config = self._load_yaml(task_path)
        merged = self._deep_merge(self.get_base_config(), task_config)
        return merged

    def get_project_root(self) -> Path:
        """Return the resolved project root path."""
        return self._PROJECT_ROOT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        Recursively merge *override* into *base*.
        Lists and scalars in override replace those in base entirely.
        """
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = ConfigManager._deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
