"""
SentNL — Base Vectorizer

Abstract interface for text vectorisation. Concrete implementations
(TF-IDF, BERT, etc.) are drop-in replacements.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseVectorizer(ABC):
    """
    Abstract vectorizer contract.

    Every vectorizer must implement:
        fit(texts)           — learn vocabulary from training data
        transform(texts)     — convert texts to feature matrix
        save(path)           — serialise fitted vectorizer
        load(path)           — deserialise vectorizer

    The fit→transform separation guarantees no feature leakage
    between train/val/test splits.
    """

    @abstractmethod
    def fit(self, texts: list[str]) -> "BaseVectorizer":
        """Learn vocabulary / embeddings from training texts."""
        ...

    @abstractmethod
    def transform(self, texts: list[str]) -> Any:
        """Transform texts into feature matrix (sparse or dense)."""
        ...

    def fit_transform(self, texts: list[str]) -> Any:
        """Convenience: fit then transform in one call."""
        return self.fit(texts).transform(texts)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialise the fitted vectorizer to disk."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> "BaseVectorizer":
        """Load a previously saved vectorizer from disk."""
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return the learned feature/vocabulary names."""
        ...
