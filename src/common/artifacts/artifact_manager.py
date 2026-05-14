"""
SentNL — Artifact Manager

Centralised model persistence layer. Saves and loads complete model bundles:
    - model.joblib          (trained estimator)
    - vectorizer.joblib     (fitted vectorizer)
    - label_mapping.json    (int → human-readable label)
    - metadata.json         (config, metrics, timestamp, dataset hash)
"""

import json
from pathlib import Path

import joblib

from src.common.logger import get_logger
from src.common.vectorizer import TfidfVectorizerWrapper

logger = get_logger("artifacts", "system.log")


class ArtifactManager:
    """Save and load self-contained, reproducible model bundles."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model,
        vectorizer,
        label_mapping: dict,
        metadata: dict,
        model_dir: str,
        version: str,
    ) -> Path:
        """
        Persist a complete model bundle to disk.

        Args:
            model:         Trained sklearn estimator.
            vectorizer:    Fitted BaseVectorizer instance.
            label_mapping: {int: str} mapping.
            metadata:      Training metadata dict.
            model_dir:     Relative directory (e.g. 'models/spam').
            version:       Version string (e.g. 'v_20250513_120000').

        Returns:
            Path to the saved bundle directory.
        """
        bundle_dir = self.project_root / model_dir / version
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Model
        model_path = bundle_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Vectorizer
        vec_path = bundle_dir / "vectorizer.joblib"
        vectorizer.save(vec_path)

        # Label mapping
        label_path = bundle_dir / "label_mapping.json"
        # Ensure keys are strings for JSON
        str_mapping = {str(k): v for k, v in label_mapping.items()}
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(str_mapping, f, indent=2)

        # Metadata
        meta_path = bundle_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Also save as "latest" symlink-equivalent
        latest_dir = self.project_root / model_dir / "latest"
        self._update_latest_pointer(latest_dir, version)

        logger.info(f"Artifacts saved to {bundle_dir}")
        return bundle_dir

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, model_dir: str, version: str | None = None) -> dict:
        """
        Load a complete model bundle.

        Args:
            model_dir: Relative directory (e.g. 'models/spam').
            version:   Specific version, or None for latest.

        Returns:
            Dict with keys: model, vectorizer, label_mapping, metadata.
        """
        base = self.project_root / model_dir

        if version is None:
            version = self._resolve_latest(base)

        bundle_dir = base / version
        if not bundle_dir.exists():
            raise FileNotFoundError(
                f"Model bundle not found: {bundle_dir}"
            )

        # Model
        model = joblib.load(bundle_dir / "model.joblib")

        # Vectorizer
        vectorizer = TfidfVectorizerWrapper()
        vectorizer.load(bundle_dir / "vectorizer.joblib")

        # Label mapping (convert string keys back to int)
        with open(
            bundle_dir / "label_mapping.json", "r", encoding="utf-8"
        ) as f:
            raw = json.load(f)
        label_mapping = {int(k): v for k, v in raw.items()}

        # Metadata
        with open(
            bundle_dir / "metadata.json", "r", encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        logger.info(f"Artifacts loaded from {bundle_dir}")
        return {
            "model": model,
            "vectorizer": vectorizer,
            "label_mapping": label_mapping,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def list_versions(self, model_dir: str) -> list[str]:
        """List all available versions for a model directory."""
        base = self.project_root / model_dir
        if not base.exists():
            return []
        return sorted(
            [
                d.name
                for d in base.iterdir()
                if d.is_dir() and d.name != "latest"
            ]
        )

    def _resolve_latest(self, base: Path) -> str:
        """Find the latest version by reading the pointer file."""
        pointer = base / "latest" / "version.txt"
        if pointer.exists():
            return pointer.read_text().strip()
        # Fallback: sort directories by name (timestamp-based)
        versions = sorted(
            [
                d.name
                for d in base.iterdir()
                if d.is_dir() and d.name != "latest"
            ]
        )
        if not versions:
            raise FileNotFoundError(
                f"No model versions found in {base}"
            )
        return versions[-1]

    def _update_latest_pointer(self, latest_dir: Path, version: str):
        """Write a version pointer file (cross-platform alternative to symlinks)."""
        latest_dir.mkdir(parents=True, exist_ok=True)
        (latest_dir / "version.txt").write_text(version)
