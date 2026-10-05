"""
explainability/shap_explainer.py

SHAP explanations for the XGBoost model using TreeExplainer.

Public contract (Architecture Bible §09):
    explain(fused_vector: pd.Series) -> list[dict]
    Returns a ranked list of {"feature": str, "shap_value": float},
    sorted by |shap_value| descending.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_XGB_MODEL_PATH = _MODEL_DIR / "xgb_v1.json"


@lru_cache(maxsize=4)
def _load_explainer(model_path: str):
    """Load SHAP TreeExplainer once and cache it in memory."""
    import shap
    from xgboost import XGBClassifier

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {path}. "
            "Run the training pipeline first."
        )

    model = XGBClassifier()
    model.load_model(str(path))
    explainer = shap.TreeExplainer(model)
    logger.info("SHAP TreeExplainer loaded.")
    return explainer


def explain(
    fused_vector: pd.Series,
    model_path: str | Path = _XGB_MODEL_PATH,
) -> list[dict]:
    """Compute SHAP feature contributions for one prediction.

    Parameters
    ----------
    fused_vector : pd.Series
        31-dimensional feature vector (output of ``fuse_features``).
    model_path : str | Path
        XGBoost model to explain. Defaults to the generic v1 model.

    Returns
    -------
    list[dict]
        Each entry: ``{"feature": str, "shap_value": float}``,
        sorted by ``|shap_value|`` descending (most impactful first).
    """
    try:
        explainer = _load_explainer(str(Path(model_path).resolve()))
    except FileNotFoundError as exc:
        logger.warning("SHAP explainer unavailable: %s", exc)
        return []

    X = fused_vector.values.reshape(1, -1)
    shap_values = explainer.shap_values(X)

    # shap_values may be (1, 31) or list[(1,31), (1,31)] for binary
    if isinstance(shap_values, list):
        # Use class-1 (UP) SHAP values
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    contributions = [
        {"feature": name, "shap_value": float(val)}
        for name, val in zip(fused_vector.index, sv)
    ]

    # Sort by absolute SHAP value descending
    contributions.sort(key=lambda d: abs(d["shap_value"]), reverse=True)

    logger.debug("SHAP computed: %d features.", len(contributions))
    return contributions
