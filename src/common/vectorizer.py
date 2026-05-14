"""
SentNL — TF-IDF Vectorizer Wrapper

Concrete implementation of BaseVectorizer using sklearn's TfidfVectorizer.
Supports serialisation via joblib for artifact persistence.
"""

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.base.base_vectorizer import BaseVectorizer
from src.common.logger import get_logger

logger = get_logger("vectorizer", "system.log")


class TfidfVectorizerWrapper(BaseVectorizer):
    """
    TF-IDF vectorizer producing sparse feature matrices.

    Config keys:
        max_features, ngram_range, min_df, max_df, sublinear_tf
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        ngram = cfg.get("ngram_range", [1, 2])
        self._vectorizer = TfidfVectorizer(
            max_features=cfg.get("max_features", 5000),
            ngram_range=tuple(ngram),
            min_df=cfg.get("min_df", 2),
            max_df=cfg.get("max_df", 0.95),
            sublinear_tf=cfg.get("sublinear_tf", True),
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfVectorizerWrapper":
        logger.info(
            f"Fitting TF-IDF vectorizer on {len(texts)} documents…"
        )
        self._vectorizer.fit(texts)
        self._fitted = True
        vocab_size = len(self._vectorizer.vocabulary_)
        logger.info(f"  Vocabulary size: {vocab_size}")
        return self

    def transform(self, texts: list[str]) -> Any:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")
        return self._vectorizer.transform(texts)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, path)
        logger.info(f"Vectorizer saved to {path}")

    def load(self, path: str | Path) -> "TfidfVectorizerWrapper":
        path = Path(path)
        self._vectorizer = joblib.load(path)
        self._fitted = True
        logger.info(f"Vectorizer loaded from {path}")
        return self

    def get_feature_names(self) -> list[str]:
        return list(self._vectorizer.get_feature_names_out())
