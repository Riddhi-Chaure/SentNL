"""
SentNL — Topics Data Loader

Loads and validates the AG News topic classification dataset.
Maps numeric class indices (1-4) to human-readable names.
"""

from pathlib import Path

import pandas as pd

from src.common.base.base_data_loader import BaseDataLoader
from src.common.logger import get_logger

logger = get_logger("data_loader", "system.log")


class TopicsDataLoader(BaseDataLoader):
    """
    Loader for the AG News topic classification dataset.

    AG News uses numeric class indices:
        1 = World, 2 = Sports, 3 = Business, 4 = Sci/Tech
    This loader maps them to human-readable strings.
    """

    # Default AG News mapping
    DEFAULT_CLASS_MAPPING = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech",
    }

    def __init__(self, project_root: Path, config: dict):
        dataset_cfg = config.get("dataset", {})
        self.class_mapping = dataset_cfg.get(
            "class_mapping", self.DEFAULT_CLASS_MAPPING
        )
        # Convert keys to int (YAML may load them as int or str)
        self.class_mapping = {
            int(k): v for k, v in self.class_mapping.items()
        }

        super().__init__(
            dataset_path=project_root / dataset_cfg.get(
                "path", "datasets/topics/test.csv"
            ),
            text_column=dataset_cfg.get("text_column", "Description"),
            label_column=dataset_cfg.get("label_column", "Class Index"),
            expected_labels=[],  # Dynamic
        )

    def load(self) -> pd.DataFrame:
        """Load dataset and map numeric class indices to string labels."""
        df = super().load()
        # Map numeric labels to human-readable names
        df["label"] = df["label"].map(self.class_mapping)
        # Drop any rows with unmapped labels
        unmapped = df["label"].isna().sum()
        if unmapped:
            logger.warning(f"Dropped {unmapped} rows with unmapped class indices")
            df = df.dropna(subset=["label"]).reset_index(drop=True)
        logger.info(f"Mapped class indices to: {list(self.class_mapping.values())}")
        return df
