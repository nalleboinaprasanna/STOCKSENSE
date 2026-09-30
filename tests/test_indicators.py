"""
tests/test_indicators.py

Unit tests for preprocessing.indicators — pure function, no I/O.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.indicators import compute_indicators


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open": close * (1 + np.random.randn(n) * 0.003),
        "High": close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "Low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)
    return df


def test_returns_dataframe():
    df = _make_ohlcv()
    result = compute_indicators(df)
    assert isinstance(result, pd.DataFrame)


def test_required_columns_present():
    df = _make_ohlcv()
    result = compute_indicators(df)
    required = [
        "rsi_14", "sma_20", "sma_50", "ema_12", "ema_26", "ema_20",
        "macd", "macd_signal", "bb_upper", "bb_lower",
        "volatility_10d", "volume_change", "daily_return", "target",
    ]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_no_nan_in_indicator_cols():
    df = _make_ohlcv()
    result = compute_indicators(df)
    indicator_cols = [
        "rsi_14", "sma_20", "sma_50",
        "volatility_10d", "volume_change", "daily_return",
    ]
    for col in indicator_cols:
        assert result[col].isna().sum() == 0, f"NaN found in {col}"


def test_rsi_bounds():
    df = _make_ohlcv()
    result = compute_indicators(df)
    assert result["rsi_14"].min() >= 0
    assert result["rsi_14"].max() <= 100


def test_target_binary():
    df = _make_ohlcv()
    result = compute_indicators(df)
    assert set(result["target"].unique()).issubset({0, 1})


def test_bollinger_upper_gt_lower():
    df = _make_ohlcv()
    result = compute_indicators(df)
    assert (result["bb_upper"] >= result["bb_lower"]).all()


def test_row_reduction_due_to_rolling():
    df = _make_ohlcv(200)
    result = compute_indicators(df)
    # Rolling 50 SMA needs 50 rows + target shift: result should be < 200
    assert len(result) < 200
    # But should have at least 100 rows for n=200
    assert len(result) >= 100
