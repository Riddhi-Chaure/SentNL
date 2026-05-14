"""
SentNL — Topics Predictor

Loads topic model artifacts and classifies text into topic categories.
"""

from pathlib import Path

from src.common.base.base_predictor import BasePredictor


class TopicsPredictor(BasePredictor):
    """Predictor for multi-class topic classification."""

    def __init__(self, project_root: Path, config: dict):
        super().__init__(project_root, config)

    def _get_task_name(self) -> str:
        return "topics"

    def _get_model_dir(self) -> str:
        return self.config.get("artifacts", {}).get(
            "model_dir", "models/topics"
        )
