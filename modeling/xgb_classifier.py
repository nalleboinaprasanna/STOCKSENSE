"""
modeling/xgb_classifier.py

XGBoost classifier trained on the fused feature DataFrame.
Targets ~74% accuracy on held-out chronological test split.

Public contract (Architecture Bible §09):
    predict(fused_vector: pd.Series) -> tuple[str, float]
    Returns ("UP" | "DOWN", probability).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_MODEL_PATH = _MODEL_DIR / "xgb_v1.json"


class XGBPredictor:
    """Wraps XGBoost classifier with train / save / load / predict methods."""

    def __init__(self):
        self._model = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        from xgboost import XGBClassifier
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.utils.class_weight import compute_sample_weight

        sample_weights = compute_sample_weight("balanced", y)

        self._model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.6,
            colsample_bylevel=0.6,
            min_child_weight=5,
            gamma=1.0,
            reg_alpha=0.5,
            reg_lambda=2.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,  # n_jobs=-1 causes segfault on Python 3.14
        )

        logger.info("Training XGBoost on %d samples, %d features.", len(X), X.shape[1])
        self._model.fit(
            X, y,
            sample_weight=sample_weights,
            eval_set=[(X, y)],
            verbose=False,
        )

        y_pred = self._model.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        logger.info("XGBoost training accuracy: %.2f%%", acc * 100)
        return {"accuracy": acc, "report": report}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        """Save model to JSON format (XGBoost native)."""
        if self._model is None:
            raise RuntimeError("No model to save — call train() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("XGBoost saved → %s", path)

    def load(self, path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        """Load model from XGBoost JSON file."""
        from xgboost import XGBClassifier
        self._model = XGBClassifier()
        self._model.load_model(str(path))
        logger.info("XGBoost loaded ← %s", path)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, fused_vector: pd.Series) -> tuple[str, float]:
        """Predict next-day price direction.

        Parameters
        ----------
        fused_vector : pd.Series
            31-dimensional feature vector (output of ``fuse_features``).

        Returns
        -------
        tuple[str, float]
            (``"UP"`` or ``"DOWN"``, probability of UP)
        """
        if self._model is None:
            raise RuntimeError(
                "XGBoost model not loaded. "
                "Run the training pipeline or call load() first."
            )

        X = fused_vector.values.reshape(1, -1)
        prob_up = float(self._model.predict_proba(X)[0, 1])
        direction = "UP" if prob_up >= 0.5 else "DOWN"
        return direction, prob_up


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience (for dashboard quick-load)
# ──────────────────────────────────────────────────────────────────────────────

_predictor_singleton: Optional[XGBPredictor] = None


def get_predictor() -> XGBPredictor:
    """Return a cached XGBPredictor loaded from the default model path."""
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = XGBPredictor()
        if _DEFAULT_MODEL_PATH.exists():
            _predictor_singleton.load(_DEFAULT_MODEL_PATH)
        else:
            logger.warning(
                "XGBoost model not found at %s. Run the training pipeline first.",
                _DEFAULT_MODEL_PATH,
            )
    return _predictor_singleton
