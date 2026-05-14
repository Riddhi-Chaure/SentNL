"""
SentNL — Sentiment Predictor

Loads sentiment model artifacts and classifies text sentiment.
"""

from pathlib import Path

from src.common.base.base_predictor import BasePredictor


class SentimentPredictor(BasePredictor):
    """Predictor for sentiment analysis."""

    def __init__(self, project_root: Path, config: dict):
        super().__init__(project_root, config)

    def _get_task_name(self) -> str:
        return "sentiment"

    def _get_model_dir(self) -> str:
        return self.config.get("artifacts", {}).get(
            "model_dir", "models/sentiment"
        )
