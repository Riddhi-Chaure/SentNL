"""
SentNL — Spam Predictor

Loads spam model artifacts and classifies text as spam/ham.
"""

from pathlib import Path

from src.common.base.base_predictor import BasePredictor


class SpamPredictor(BasePredictor):
    """Predictor for spam/ham classification."""

    def __init__(self, project_root: Path, config: dict):
        super().__init__(project_root, config)

    def _get_task_name(self) -> str:
        return "spam"

    def _get_model_dir(self) -> str:
        return self.config.get("artifacts", {}).get(
            "model_dir", "models/spam"
        )
