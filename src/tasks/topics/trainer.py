"""
SentNL — Topics Trainer

Trains multi-class topic classification models (LinearSVC / LogisticRegression).
"""

from pathlib import Path
from typing import Any

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.common.base.base_trainer import BaseTrainer
from src.tasks.topics.data_loader import TopicsDataLoader
from src.common.logger import get_logger

logger = get_logger("training", "training.log")


class TopicsTrainer(BaseTrainer):
    """Trainer for multi-class topic classification."""

    def __init__(self, config: dict, project_root: Path, model_key: str = "primary"):
        loader = TopicsDataLoader(project_root, config)
        super().__init__(loader, config, project_root)
        self.model_key = model_key

    def _build_model(self) -> Any:
        models_cfg = self.config.get("models", {})
        model_cfg = models_cfg.get(self.model_key, {})
        model_class = model_cfg.get("class", "sklearn.svm.LinearSVC")

        if "LinearSVC" in model_class:
            return LinearSVC(max_iter=2000, random_state=42)
        elif "LogisticRegression" in model_class:
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class="multinomial",
                solver="lbfgs",
            )
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

    def _get_model_name(self) -> str:
        models_cfg = self.config.get("models", {})
        return models_cfg.get(self.model_key, {}).get("name", "LinearSVC")

    def _tune_hyperparameters(self, model, X_train, y_train):
        models_cfg = self.config.get("models", {})
        model_cfg = models_cfg.get(self.model_key, {})
        param_grid = model_cfg.get("hyperparameters", {})

        if not param_grid:
            return model

        # Filter out non-tunable params for GridSearchCV
        tunable = {
            k: v for k, v in param_grid.items()
            if isinstance(v, list)
        }

        if not tunable:
            return model

        logger.info(f"  Tuning hyperparameters: {tunable}")
        grid = GridSearchCV(
            model,
            tunable,
            cv=self.train_cfg.get("cv_folds", 5),
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        logger.info(f"  Best params: {grid.best_params_}")
        logger.info(f"  Best CV score: {grid.best_score_:.4f}")
        return grid.best_estimator_
