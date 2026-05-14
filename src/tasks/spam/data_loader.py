"""
SentNL — Spam Data Loader

Loads and validates the spam/ham SMS dataset.
Handles latin-1 encoding and extra unnamed columns.
"""

from pathlib import Path

from src.common.base.base_data_loader import BaseDataLoader


class SpamDataLoader(BaseDataLoader):
    """Loader for the spam classification dataset (SMS Spam Collection)."""

    def __init__(self, project_root: Path, config: dict):
        dataset_cfg = config.get("dataset", {})
        super().__init__(
            dataset_path=project_root / dataset_cfg.get(
                "path", "datasets/spam/spam.csv"
            ),
            text_column=dataset_cfg.get("text_column", "v2"),
            label_column=dataset_cfg.get("label_column", "v1"),
            expected_labels=dataset_cfg.get("expected_labels", ["ham", "spam"]),
            encoding=dataset_cfg.get("encoding", "latin-1"),
            use_columns=dataset_cfg.get("use_columns", ["v1", "v2"]),
        )
