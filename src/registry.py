"""
SentNL — Task Registry

Simple dictionary-based registry mapping task names to predictor classes.
Both the CLI and Streamlit UI use this registry to dynamically load
the correct predictor — no hardcoded if-else chains.
"""

from typing import Dict, Type

from src.common.base.base_predictor import BasePredictor


# Global registry: task_name → Predictor class
_REGISTRY: Dict[str, Type[BasePredictor]] = {}


def register_task(task_name: str, predictor_class: Type[BasePredictor]):
    """Register a predictor class for a task name."""
    _REGISTRY[task_name] = predictor_class


def get_predictor_class(task_name: str) -> Type[BasePredictor]:
    """Look up and return the predictor class for a task."""
    if task_name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise KeyError(
            f"Unknown task '{task_name}'. Available: {available}"
        )
    return _REGISTRY[task_name]


def list_tasks() -> list[str]:
    """Return all registered task names."""
    return list(_REGISTRY.keys())


def get_registry() -> Dict[str, Type[BasePredictor]]:
    """Return the full registry dict."""
    return dict(_REGISTRY)
