"""
modeling/fusion.py

Feature Fusion Layer — concatenates temporal embedding, technical indicators,
and sentiment scores into a single flat pd.Series for XGBoost.

Enhanced: 16 embed + 35 indicators + 2 sentiment = 53 features
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LSTM_DIMS = 16
_INDICATOR_COLS = [
    "rsi_14", "rsi_7", "sma_20", "sma_50", "ema_12", "ema_26", "ema_20", "ema_9",
    "price_vs_sma20", "price_vs_sma50", "sma20_vs_sma50",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "volatility_10d", "volatility_20d",
    "volume_change", "volume_ratio",
    "atr_14", "atr_pct",
    "stoch_k", "stoch_d", "williams_r",
    "obv_change",
    "daily_return", "return_2d", "return_5d", "return_10d", "return_20d",
    "momentum_10", "roc_10",
]
_SENTIMENT_COLS = ["finbert_sentiment", "vader_sentiment"]

FEATURE_SCHEMA: list[str] = (
    [f"lstm_embed_{i}" for i in range(_LSTM_DIMS)]
    + _INDICATOR_COLS
    + _SENTIMENT_COLS
)


def fuse_features(embed, indicators, sentiment) -> pd.Series:
    embed = np.asarray(embed, dtype=np.float32).flatten()
    # Pad/trim to exactly _LSTM_DIMS
    emb_padded = np.zeros(_LSTM_DIMS, dtype=np.float32)
    n = min(len(embed), _LSTM_DIMS)
    emb_padded[:n] = embed[:n]

    if isinstance(indicators, pd.Series):
        indicators = indicators.to_dict()

    # Fill missing indicator cols with 0 (graceful fallback for legacy data)
    values = (
        emb_padded.tolist()
        + [float(indicators.get(c, 0.0)) for c in _INDICATOR_COLS]
        + [float(sentiment.get(c, 0.0)) for c in _SENTIMENT_COLS]
    )

    return pd.Series(values, index=FEATURE_SCHEMA, dtype=np.float64)
