"""
modeling/fusion.py

Feature Fusion Layer — concatenates temporal embedding, technical indicators,
and sentiment scores into a single flat pd.Series for XGBoost.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LSTM_DIMS = 16
_INDICATOR_COLS = [
    "rsi_14",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "ema_20",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "volatility_10d",
    "volume_change",
    "daily_return",
]
_SENTIMENT_COLS = ["finbert_sentiment", "vader_sentiment"]

FEATURE_SCHEMA: list[str] = (
    [f"lstm_embed_{i}" for i in range(_LSTM_DIMS)]
    + _INDICATOR_COLS
    + _SENTIMENT_COLS
)


def fuse_features(embed, indicators, sentiment) -> pd.Series:
    embed = np.asarray(embed, dtype=np.float32).flatten()
    if embed.size != _LSTM_DIMS:
        raise ValueError(f"embed shape must be exactly {_LSTM_DIMS}, got {embed.size}")

    if isinstance(indicators, pd.Series):
        indicators = indicators.to_dict()
    elif isinstance(indicators, pd.DataFrame):
        indicators = indicators.iloc[0].to_dict()

    missing_indicators = [c for c in _INDICATOR_COLS if c not in indicators]
    if missing_indicators:
        raise ValueError(f"Missing indicator: {missing_indicators[0]}")

    if isinstance(sentiment, pd.Series):
        sentiment = sentiment.to_dict()
    elif isinstance(sentiment, pd.DataFrame):
        sentiment = sentiment.iloc[0].to_dict()

    missing_sentiment = [c for c in _SENTIMENT_COLS if c not in sentiment]
    if missing_sentiment:
        raise ValueError(f"Missing sentiment: {missing_sentiment[0]}")

    values = (
        embed.tolist()
        + [float(indicators[c]) for c in _INDICATOR_COLS]
        + [float(sentiment[c]) for c in _SENTIMENT_COLS]
    )

    return pd.Series(values, index=FEATURE_SCHEMA, dtype=np.float64)
