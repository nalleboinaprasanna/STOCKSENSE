"""SHAP explanations for XGBoost models using native TreeSHAP contributions.

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
def _load_booster(model_path: str):
    """Load an XGBoost booster once and cache it in memory."""
    import xgboost as xgb

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {path}. "
            "Run the training pipeline first."
        )

    booster = xgb.Booster()
    booster.load_model(str(path))
    logger.info("XGBoost booster loaded for native TreeSHAP contributions.")
    return booster


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
        booster = _load_booster(str(Path(model_path).resolve()))
    except FileNotFoundError as exc:
        logger.warning("SHAP explainer unavailable: %s", exc)
        return []

    import xgboost as xgb

    feature_names = booster.feature_names
    if feature_names and list(fused_vector.index) != feature_names:
        raise ValueError("Input feature names do not match the XGBoost model.")

    X = fused_vector.to_numpy(dtype=np.float32).reshape(1, -1)
    matrix = xgb.DMatrix(X, feature_names=feature_names)
    # The last column is the expected value (bias), not a feature contribution.
    sv = booster.predict(matrix, pred_contribs=True)[0, :-1]
    if len(sv) != len(fused_vector):
        raise ValueError(
            f"Expected {len(fused_vector)} SHAP values, received {len(sv)}."
        )

    contributions = [
        {"feature": name, "shap_value": float(val)}
        for name, val in zip(fused_vector.index, sv)
    ]

    # Sort by absolute SHAP value descending
    contributions.sort(key=lambda d: abs(d["shap_value"]), reverse=True)

    logger.debug("SHAP computed: %d features.", len(contributions))
    return contributions
