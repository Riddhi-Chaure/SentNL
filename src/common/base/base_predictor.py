"""
SentNL — Base Predictor

Provides predict(text) and predict_batch(texts).
Subclasses load the correct artifacts and specify label mapping.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.common.logger import get_logger
from src.common.preprocess import Preprocessor
from src.common.artifacts.artifact_manager import ArtifactManager

logger = get_logger("inference", "inference.log")


class BasePredictor(ABC):
    """
    Abstract predictor that loads artifacts and runs inference.

    Subclasses implement:
        _get_task_name()   → return the task identifier
        _get_model_dir()   → return the model directory path
    """

    def __init__(self, project_root: Path, config: dict):
        self.project_root = project_root
        self.config = config
        self.task_cfg = config.get("task", {})
        self.prep_cfg = config.get("preprocessing", {})

        self.preprocessor = Preprocessor(self.prep_cfg)
        self.artifact_manager = ArtifactManager(project_root)

        # Loaded during _load_artifacts()
        self.model = None
        self.vectorizer = None
        self.label_mapping = None
        self.metadata = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify a single text input.

        Args:
            text: Raw input text.

        Returns:
            Structured response dict with task, prediction,
            confidence, probabilities, and processing time.
        """
        self._ensure_loaded()
        start = time.time()

        # Preprocess
        clean = self.preprocessor.clean(text)
        if not clean.strip():
            return self._empty_response(text)

        # Vectorize
        features = self.vectorizer.transform([clean])

        # Predict
        prediction = self.model.predict(features)[0]
        label = self.label_mapping.get(int(prediction), str(prediction))

        # Probabilities
        probabilities = self._get_probabilities(features)

        elapsed_ms = round((time.time() - start) * 1000, 2)

        response = {
            "task": self._get_task_name(),
            "model_version": self.metadata.get("version", "unknown"),
            "prediction": {
                "label": label,
                "confidence": round(
                    float(max(probabilities.values())), 4
                ),
                "probabilities": probabilities,
            },
            "processing_time_ms": elapsed_ms,
        }

        logger.info(
            f"[{self._get_task_name()}] "
            f"'{text[:50]}...' -> {label} "
            f"({response['prediction']['confidence']:.2%})"
        )
        return response

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a batch of texts."""
        return [self.predict(t) for t in texts]

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_task_name(self) -> str:
        ...

    @abstractmethod
    def _get_model_dir(self) -> str:
        ...

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            self._load_artifacts()
            self._loaded = True

    def _load_artifacts(self):
        """Load model, vectorizer, label mapping, and metadata."""
        model_dir = self._get_model_dir()
        bundle = self.artifact_manager.load(model_dir)
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]
        self.label_mapping = bundle["label_mapping"]
        self.metadata = bundle["metadata"]
        logger.info(
            f"Loaded artifacts for {self._get_task_name()} "
            f"(version: {self.metadata.get('version', '?')})"
        )

    def _get_probabilities(self, features) -> Dict[str, float]:
        """
        Extract probability distribution.
        Falls back to decision_function for models without predict_proba.
        """
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
        elif hasattr(self.model, "decision_function"):
            decisions = self.model.decision_function(features)[0]
            # Softmax approximation for SVC
            if isinstance(decisions, np.ndarray):
                exp = np.exp(decisions - np.max(decisions))
                probs = exp / exp.sum()
            else:
                # Binary case
                prob = 1 / (1 + np.exp(-decisions))
                probs = np.array([1 - prob, prob])
        else:
            # Fallback: hard prediction only
            pred = self.model.predict(features)[0]
            probs = np.zeros(len(self.label_mapping))
            probs[pred] = 1.0

        return {
            self.label_mapping.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(probs)
        }

    def _empty_response(self, text: str) -> Dict[str, Any]:
        return {
            "task": self._get_task_name(),
            "model_version": (
                self.metadata.get("version", "unknown")
                if self.metadata
                else "unknown"
            ),
            "prediction": {
                "label": "unknown",
                "confidence": 0.0,
                "probabilities": {},
            },
            "processing_time_ms": 0.0,
            "warning": "Text was empty after preprocessing",
        }
