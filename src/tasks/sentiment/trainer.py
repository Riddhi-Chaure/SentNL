"""
SentNL — Sentiment Trainer

Trains sentiment classification models (LogisticRegression / LinearSVC).
"""

from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from src.common.base.base_trainer import BaseTrainer
from src.tasks.sentiment.data_loader import SentimentDataLoader
from src.common.logger import get_logger

logger = get_logger("training", "training.log")


class SentimentTrainer(BaseTrainer):
    """Trainer for sentiment analysis."""

    def __init__(self, config: dict, project_root: Path, model_key: str = "primary"):
        loader = SentimentDataLoader(project_root, config)
        super().__init__(loader, config, project_root)
        self.model_key = model_key

    def _build_model(self) -> Any:
        models_cfg = self.config.get("models", {})
        model_cfg = models_cfg.get(self.model_key, {})
        model_class = model_cfg.get(
            "class", "sklearn.linear_model.LogisticRegression"
        )

        if "LogisticRegression" in model_class:
            return LogisticRegression(max_iter=1000, random_state=42)
        elif "LinearSVC" in model_class:
            return LinearSVC(max_iter=2000, random_state=42)
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

    def _get_model_name(self) -> str:
        models_cfg = self.config.get("models", {})
        return models_cfg.get(self.model_key, {}).get(
            "name", "LogisticRegression"
        )

    def _tune_hyperparameters(self, model, X_train, y_train):
        models_cfg = self.config.get("models", {})
        model_cfg = models_cfg.get(self.model_key, {})
        param_grid = model_cfg.get("hyperparameters", {})

        if not param_grid:
            return model

        logger.info(f"  Tuning hyperparameters: {param_grid}")
        grid = GridSearchCV(
            model,
            param_grid,
            cv=self.train_cfg.get("cv_folds", 5),
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        logger.info(f"  Best params: {grid.best_params_}")
        logger.info(f"  Best CV score: {grid.best_score_:.4f}")
        return grid.best_estimator_
