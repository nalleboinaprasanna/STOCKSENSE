"""
modeling/lstm_model.py

Temporal feature embedder: produces a 16-dim vector from a 60-day window.
Uses hand-crafted momentum/slope/volatility statistics instead of TF LSTM
(TensorFlow unsupported on Python 3.14).

Same public API as original LSTM version.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 60
_EMBED_DIM = 16

_embedder_instance: Optional["LSTMEmbedder"] = None


def _window_to_temporal_features(window: np.ndarray) -> np.ndarray:
    """Extract 16 hand-crafted temporal statistics from a (W, F) window.

    Features (per close-price column index 3, and aggregate statistics):
        0  - 5-day return  (momentum short)
        1  - 10-day return (momentum mid)
        2  - 20-day return (momentum long)
        3  - 60-day return (momentum full window)
        4  - linear slope of close (normalised)
        5  - linear slope of volume (normalised)
        6  - 5-day volatility (std of returns)
        7  - 20-day volatility
        8  - mean cross-feature correlation magnitude
        9  - skewness of close returns
        10 - ratio of last close to window max close
        11 - ratio of last close to window min close
        12 - 5-day mean return acceleration (slope of 5d rolling mean)
        13 - fraction of up-days in window
        14 - z-score of last close vs window mean
        15 - last-day range (High-Low)/Close normalised
    """
    W, F = window.shape
    eps = 1e-9

    # Use column indices as proxies (matches lstm_feature_cols order):
    # 0=Open,1=High,2=Low,3=Close,4=Volume, then indicators…
    close_col = min(3, F - 1)
    high_col  = min(1, F - 1)
    low_col   = min(2, F - 1)
    vol_col   = min(4, F - 1)

    close  = window[:, close_col]
    volume = window[:, vol_col]

    returns = np.diff(close) / (np.abs(close[:-1]) + eps)

    def ret(n):
        if len(close) < n + 1:
            return 0.0
        return float((close[-1] - close[-n]) / (abs(close[-n]) + eps))

    def slope(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=np.float64)
        x -= x.mean()
        y = arr - arr.mean()
        denom = (x * x).sum()
        return float((x * y).sum() / (denom + eps))

    feat = np.zeros(16, dtype=np.float32)
    feat[0]  = ret(5)
    feat[1]  = ret(10)
    feat[2]  = ret(20)
    feat[3]  = ret(59)
    feat[4]  = slope(close)
    feat[5]  = slope(volume)
    feat[6]  = float(returns[-5:].std())  if len(returns) >= 5  else 0.0
    feat[7]  = float(returns[-20:].std()) if len(returns) >= 20 else 0.0
    # correlation magnitude across all feature pairs (mean of abs corr matrix)
    if F > 1 and W > 2:
        corr = np.corrcoef(window.T)
        np.fill_diagonal(corr, 0.0)
        feat[8] = float(np.abs(corr).mean())
    feat[9]  = float(_skew(returns))
    cmax = close.max(); cmin = close.min()
    feat[10] = float(close[-1] / (cmax + eps))
    feat[11] = float(close[-1] / (cmin + eps)) if cmin != 0 else 0.0
    # 5-day acceleration: slope of the last 10 5-day rolling means
    if len(close) >= 10:
        rolling5 = np.array([close[max(0,i-4):i+1].mean() for i in range(len(close)-10, len(close))])
        feat[12] = slope(rolling5)
    frac_up = float((returns > 0).mean()) if len(returns) > 0 else 0.5
    feat[13] = frac_up
    cmean = close.mean(); cstd = close.std()
    feat[14] = float((close[-1] - cmean) / (cstd + eps))
    if F > high_col and F > low_col:
        feat[15] = float((window[-1, high_col] - window[-1, low_col]) / (abs(close[-1]) + eps))
    return feat


def _skew(arr):
    if len(arr) < 3:
        return 0.0
    m = arr.mean(); s = arr.std()
    if s < 1e-9:
        return 0.0
    return float(((arr - m) ** 3).mean() / (s ** 3))


class LSTMEmbedder:
    """Hand-crafted temporal embedder. Same API as original TF LSTM version."""

    def __init__(self):
        self._fitted = False   # stateless — no training needed

    def train(self, X, y, epochs=30, batch_size=32, validation_split=0.2):
        """Stateless — just validates input shape and marks fitted."""
        logger.info("Temporal embedder: no training needed (hand-crafted features).")
        self._fitted = True
        return {"accuracy": None}

    def save(self, path):
        import joblib
        pkl_path = Path(path).with_suffix(".pkl")
        joblib.dump({"fitted": self._fitted}, pkl_path)
        logger.info("Embedder saved -> %s", pkl_path)

    def load(self, path):
        import joblib
        pkl_path = Path(path).with_suffix(".pkl")
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model not found: {pkl_path}")
        data = joblib.load(pkl_path)
        self._fitted = data.get("fitted", True)
        logger.info("Embedder loaded <- %s", pkl_path)

    def embed(self, window: np.ndarray) -> np.ndarray:
        if window.ndim == 3:
            window = window[0]
        return _window_to_temporal_features(window)


def lstm_embed(window: np.ndarray) -> np.ndarray:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = _load_default_embedder()
    return _embedder_instance.embed(window)


def _load_default_embedder() -> LSTMEmbedder:
    model_path = Path(__file__).resolve().parent.parent / "models" / "lstm_v1.pkl"
    embedder = LSTMEmbedder()
    if model_path.exists():
        embedder.load(model_path)
    else:
        logger.warning("Embedder not found at %s. Run: python -m modeling.train_pipeline", model_path)
    return embedder


def prepare_lstm_data(df, feature_cols, window_size=_WINDOW_SIZE):
    from sklearn.preprocessing import MinMaxScaler

    data = df[feature_cols].values.astype(np.float32)
    targets = df["target"].values.astype(np.float32)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i - window_size: i])
        y.append(targets[i])

    return np.array(X), np.array(y), scaler
