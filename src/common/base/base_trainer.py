"""
SentNL — Base Trainer

Orchestrates the complete training lifecycle:
    load → preprocess → encode → vectorize → split → build model →
    tune hyperparameters → evaluate → save artifacts → log experiment

Subclasses override only:
    _build_model()
    _tune_hyperparameters() (optional)
    _get_model_name()
"""

import json
import time
import hashlib
import csv
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.common.logger import get_logger
from src.common.preprocess import Preprocessor
from src.common.vectorizer import TfidfVectorizerWrapper
from src.common.artifacts.artifact_manager import ArtifactManager
from src.common.base.base_data_loader import BaseDataLoader

logger = get_logger("training", "training.log")


class BaseTrainer(ABC):
    """
    Abstract trainer that orchestrates the full ML lifecycle.

    Subclasses implement:
        _build_model()            → return an sklearn estimator
        _get_model_name()         → return a string identifier
        _tune_hyperparameters()   → optional grid search
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        config: dict,
        project_root: Path,
    ):
        self.data_loader = data_loader
        self.config = config
        self.project_root = project_root

        # Configuration sections
        self.task_cfg = config.get("task", {})
        self.train_cfg = config.get("training", {})
        self.vec_cfg = config.get("vectorizer", {})
        self.prep_cfg = config.get("preprocessing", {})
        self.artifact_cfg = config.get("artifacts", {})

        # Components
        self.preprocessor = Preprocessor(self.prep_cfg)
        self.label_encoder = LabelEncoder()
        self.artifact_manager = ArtifactManager(project_root)

        # Populated during training
        self.model = None
        self.vectorizer = None
        self.metrics = {}
        self.run_id = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Execute the full training pipeline.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = self.task_cfg.get("name", "unknown")
        logger.info(f"=== Training run {self.run_id} for task: {task_name} ===")

        # 1. Load data
        logger.info("Step 1/8: Loading dataset...")
        df = self.data_loader.load()
        dataset_hash = self._compute_dataset_hash(df)

        # 2. Preprocess
        logger.info("Step 2/8: Preprocessing text...")
        df["clean_text"] = self.preprocessor.batch_clean(df["text"].tolist())
        # Drop rows that became empty after cleaning
        df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

        # 3. Encode labels
        logger.info("Step 3/8: Encoding labels...")
        df["encoded_label"] = self.label_encoder.fit_transform(df["label"])
        label_mapping = dict(
            zip(
                self.label_encoder.transform(self.label_encoder.classes_),
                self.label_encoder.classes_,
            )
        )

        # 4. Vectorize
        logger.info("Step 4/8: Vectorizing features...")
        self.vectorizer = TfidfVectorizerWrapper(self.vec_cfg)
        X = self.vectorizer.fit_transform(df["clean_text"].tolist())
        y = df["encoded_label"].values

        # 5. Split
        logger.info("Step 5/8: Splitting dataset...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.train_cfg.get("test_size", 0.15)
            + self.train_cfg.get("validation_size", 0.15),
            random_state=self.train_cfg.get("random_state", 42),
            stratify=y,
        )
        # Split temp into val + test
        val_ratio = self.train_cfg.get("validation_size", 0.15) / (
            self.train_cfg.get("test_size", 0.15)
            + self.train_cfg.get("validation_size", 0.15)
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - val_ratio,
            random_state=self.train_cfg.get("random_state", 42),
            stratify=y_temp,
        )
        logger.info(
            f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, "
            f"Test: {X_test.shape[0]}"
        )

        # 6. Build & optionally tune model
        logger.info("Step 6/8: Building model...")
        self.model = self._build_model()
        model_name = self._get_model_name()
        logger.info(f"  Model: {model_name}")

        # Optional hyperparameter tuning
        self.model = self._tune_hyperparameters(
            self.model, X_train, y_train
        )

        # 7. Train
        logger.info("Step 7/8: Training...")
        start = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start
        logger.info(f"  Training completed in {train_time:.2f}s")

        # 8. Evaluate
        logger.info("Step 8/8: Evaluating...")
        y_pred = self.model.predict(X_test)
        self.metrics = self._compute_metrics(y_test, y_pred, label_mapping)
        self.metrics["training_time_seconds"] = round(train_time, 3)

        # Inference speed benchmark
        start = time.time()
        self.model.predict(X_test[:1])
        self.metrics["inference_time_ms"] = round(
            (time.time() - start) * 1000, 2
        )

        logger.info(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1: {self.metrics['macro_f1']:.4f}")

        # --- Save everything ---
        version = f"v_{self.run_id}"
        model_dir = self.artifact_cfg.get("model_dir", f"models/{task_name}")

        self.artifact_manager.save(
            model=self.model,
            vectorizer=self.vectorizer,
            label_mapping=label_mapping,
            metadata={
                "task": task_name,
                "model_name": model_name,
                "version": version,
                "dataset_hash": dataset_hash,
                "metrics": self.metrics,
                "preprocessing": self.prep_cfg,
                "vectorizer_config": self.vec_cfg,
                "timestamp": datetime.now().isoformat(),
                "model_params": self._get_model_params(),
            },
            model_dir=model_dir,
            version=version,
        )

        # Generate evaluation reports
        self._generate_reports(
            y_test, y_pred, label_mapping, task_name, version
        )

        # Append to experiments.csv
        self._log_experiment(
            task_name, model_name, version, dataset_hash, self.metrics
        )

        logger.info(f"=== Run {self.run_id} complete ===\n")
        return self.metrics

    # ------------------------------------------------------------------
    # Abstract methods — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self) -> Any:
        """Return an sklearn-compatible estimator."""
        ...

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return a human-readable model identifier."""
        ...

    def _tune_hyperparameters(self, model, X_train, y_train):
        """
        Optional hyperparameter tuning. Default: no-op.
        Override in subclass for GridSearchCV / RandomizedSearchCV.
        """
        return model

    # ------------------------------------------------------------------
    # Evaluation & reporting
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, y_true, y_pred, label_mapping
    ) -> Dict[str, float]:
        labels = sorted(label_mapping.keys())
        target_names = [label_mapping[l] for l in labels]
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision_weighted": round(
                precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                4,
            ),
            "recall_weighted": round(
                recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                4,
            ),
            "f1_weighted": round(
                f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                4,
            ),
            "macro_f1": round(
                f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
                4,
            ),
            "classification_report": classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0
            ),
        }

    def _generate_reports(
        self, y_test, y_pred, label_mapping, task_name, version
    ):
        report_dir = (
            self.project_root / "reports" / task_name / version
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        # metrics.json
        metrics_path = report_dir / "metrics.json"
        serialisable = {
            k: v
            for k, v in self.metrics.items()
            if k != "classification_report"
        }
        with open(metrics_path, "w") as f:
            json.dump(serialisable, f, indent=2)

        # classification_report.txt
        report_path = report_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(self.metrics["classification_report"])

        # confusion_matrix.png
        labels = sorted(label_mapping.keys())
        target_names = [label_mapping[l] for l in labels]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {task_name} ({version})")
        fig.tight_layout()
        fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        # feature_importance.csv (top TF-IDF terms per class)
        self._save_feature_importance(report_dir, label_mapping)

        logger.info(f"  Reports saved to {report_dir}")

    def _save_feature_importance(
        self, report_dir: Path, label_mapping: dict
    ):
        """Extract top features for linear models (coef_ attribute)."""
        if not hasattr(self.model, "coef_"):
            return

        feature_names = self.vectorizer.get_feature_names()
        rows = []
        coef = self.model.coef_

        if coef.ndim == 1:
            # Binary classification — single coef vector
            top_idx = np.argsort(np.abs(coef))[-20:][::-1]
            for idx in top_idx:
                rows.append(
                    {
                        "feature": feature_names[idx],
                        "weight": round(float(coef[idx]), 4),
                        "class": "global",
                    }
                )
        else:
            for class_idx, weights in enumerate(coef):
                class_name = label_mapping.get(class_idx, str(class_idx))
                top_idx = np.argsort(np.abs(weights))[-10:][::-1]
                for idx in top_idx:
                    rows.append(
                        {
                            "feature": feature_names[idx],
                            "weight": round(float(weights[idx]), 4),
                            "class": class_name,
                        }
                    )

        pd.DataFrame(rows).to_csv(
            report_dir / "feature_importance.csv", index=False
        )

    def _log_experiment(
        self, task, model_name, version, dataset_hash, metrics
    ):
        """Append a row to reports/experiments.csv."""
        csv_path = self.project_root / "reports" / "experiments.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "task",
            "model_name",
            "version",
            "dataset_hash",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "macro_f1",
            "training_time_seconds",
            "inference_time_ms",
        ]

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "task": task,
                    "model_name": model_name,
                    "version": version,
                    "dataset_hash": dataset_hash,
                    "accuracy": metrics.get("accuracy"),
                    "precision_weighted": metrics.get("precision_weighted"),
                    "recall_weighted": metrics.get("recall_weighted"),
                    "f1_weighted": metrics.get("f1_weighted"),
                    "macro_f1": metrics.get("macro_f1"),
                    "training_time_seconds": metrics.get(
                        "training_time_seconds"
                    ),
                    "inference_time_ms": metrics.get("inference_time_ms"),
                }
            )
        logger.info("  Experiment logged to experiments.csv")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dataset_hash(df: pd.DataFrame) -> str:
        """SHA-256 hash of the dataset for versioning."""
        content = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:12]

    def _get_model_params(self) -> dict:
        """Extract model parameters for metadata."""
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}
