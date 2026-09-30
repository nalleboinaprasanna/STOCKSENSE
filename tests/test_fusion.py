"""
tests/test_fusion.py

Unit tests for modeling.fusion — schema validation and correct concatenation.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.fusion import fuse_features, FEATURE_SCHEMA, _LSTM_DIMS, _INDICATOR_COLS, _SENTIMENT_COLS


def _make_embed() -> np.ndarray:
    return np.random.rand(_LSTM_DIMS).astype(np.float32)


def _make_indicators() -> dict:
    return {col: float(np.random.rand()) for col in _INDICATOR_COLS}


def _make_sentiment() -> dict:
    return {"finbert_sentiment": 0.3, "vader_sentiment": -0.1}


def test_schema_length():
    assert len(FEATURE_SCHEMA) == 31


def test_fuse_returns_series():
    fused = fuse_features(_make_embed(), _make_indicators(), _make_sentiment())
    assert isinstance(fused, pd.Series)


def test_fuse_correct_index():
    fused = fuse_features(_make_embed(), _make_indicators(), _make_sentiment())
    assert list(fused.index) == FEATURE_SCHEMA


def test_fuse_no_nan():
    fused = fuse_features(_make_embed(), _make_indicators(), _make_sentiment())
    assert not fused.isna().any()


def test_fuse_wrong_embed_shape_raises():
    with pytest.raises(ValueError, match="embed shape"):
        fuse_features(np.zeros(10), _make_indicators(), _make_sentiment())


def test_fuse_missing_indicator_raises():
    ind = _make_indicators()
    del ind["rsi_14"]
    with pytest.raises(ValueError, match="Missing indicator"):
        fuse_features(_make_embed(), ind, _make_sentiment())


def test_fuse_missing_sentiment_raises():
    sent = _make_sentiment()
    del sent["finbert_sentiment"]
    with pytest.raises(ValueError, match="Missing sentiment"):
        fuse_features(_make_embed(), _make_indicators(), sent)


def test_fuse_accepts_series_indicators():
    ind_series = pd.Series(_make_indicators())
    fused = fuse_features(_make_embed(), ind_series, _make_sentiment())
    assert isinstance(fused, pd.Series)
