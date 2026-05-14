"""
SentNL — Sentiment Data Loader

Loads and validates the IMDB sentiment analysis dataset.
"""

from pathlib import Path

from src.common.base.base_data_loader import BaseDataLoader


class SentimentDataLoader(BaseDataLoader):
    """Loader for the IMDB sentiment analysis dataset."""

    def __init__(self, project_root: Path, config: dict):
        dataset_cfg = config.get("dataset", {})
        super().__init__(
            dataset_path=project_root / dataset_cfg.get(
                "path", "datasets/sentiment/IMDB Dataset.csv"
            ),
            text_column=dataset_cfg.get("text_column", "review"),
            label_column=dataset_cfg.get("label_column", "sentiment"),
            expected_labels=dataset_cfg.get(
                "expected_labels", ["positive", "negative"]
            ),
        )
