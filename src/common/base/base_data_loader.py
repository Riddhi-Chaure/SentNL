"""
SentNL — Base Data Loader

Abstract base enforcing a strict schema contract:
  - Required columns exist
  - No null values in text/label
  - Expected label values (if specified)
  - Duplicate removal
  - Type validation
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.common.logger import get_logger

logger = get_logger("data_loader", "system.log")


class BaseDataLoader(ABC):
    """
    Abstract data loader that validates and cleans task datasets.

    Subclasses only need to specify:
        - dataset_path
        - text_column / label_column names
        - expected_labels (optional; empty = discover from data)
    """

    def __init__(
        self,
        dataset_path: str,
        text_column: str = "text",
        label_column: str = "label",
        expected_labels: Optional[List[str]] = None,
        encoding: str = "utf-8",
        use_columns: Optional[List[str]] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.text_column = text_column
        self.label_column = label_column
        self.expected_labels = expected_labels or []
        self.encoding = encoding
        self.use_columns = use_columns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load, validate, and clean the dataset.

        Returns:
            DataFrame with columns ['text', 'label'], no nulls,
            valid labels, duplicates removed.
        """
        logger.info(f"Loading dataset from {self.dataset_path}")
        df = self._read_raw()
        df = self._validate_schema(df)
        df = self._validate_nulls(df)
        df = self._validate_labels(df)
        df = self._remove_duplicates(df)
        df = self._standardise_columns(df)
        logger.info(
            f"Dataset loaded: {len(df)} samples, "
            f"{df['label'].nunique()} classes"
        )
        self._log_class_distribution(df)
        return df

    # ------------------------------------------------------------------
    # Validation steps
    # ------------------------------------------------------------------

    def _read_raw(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )
        kwargs = {"encoding": self.encoding}
        if self.use_columns:
            kwargs["usecols"] = self.use_columns
        return pd.read_csv(self.dataset_path, **kwargs)

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist."""
        for col in [self.text_column, self.label_column]:
            if col not in df.columns:
                raise ValueError(
                    f"Missing required column '{col}'. "
                    f"Available columns: {list(df.columns)}"
                )
        return df

    def _validate_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with null text or label, log count."""
        before = len(df)
        df = df.dropna(subset=[self.text_column, self.label_column])
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with null values")
        # Also drop empty-string texts
        df = df[df[self.text_column].astype(str).str.strip().astype(bool)]
        return df

    def _validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check labels against expected values if specified."""
        actual = set(df[self.label_column].unique())
        if self.expected_labels:
            expected = set(self.expected_labels)
            invalid = actual - expected
            if invalid:
                logger.warning(
                    f"Removing {len(invalid)} unexpected label(s): {invalid}"
                )
                df = df[df[self.label_column].isin(expected)]
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(
            subset=[self.text_column, self.label_column]
        )
        dropped = before - len(df)
        if dropped:
            logger.info(f"Removed {dropped} duplicate rows")
        return df

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to the canonical 'text' / 'label' names."""
        df = df.rename(
            columns={
                self.text_column: "text",
                self.label_column: "label",
            }
        )
        return df[["text", "label"]].reset_index(drop=True)

    def _log_class_distribution(self, df: pd.DataFrame):
        dist = df["label"].value_counts()
        for label, count in dist.items():
            pct = count / len(df) * 100
            logger.info(f"  {label}: {count} ({pct:.1f}%)")
