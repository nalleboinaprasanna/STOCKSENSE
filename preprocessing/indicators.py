"""
preprocessing/indicators.py

Computes technical indicators from OHLCV data.
Enhanced with additional momentum, trend, and volume features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and next-day binary target to *ohlcv*."""
    df = ohlcv.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── Returns ───────────────────────────────────────────────────────────
    df["daily_return"]  = close.pct_change()
    df["return_2d"]     = close.pct_change(2)
    df["return_5d"]     = close.pct_change(5)
    df["return_10d"]    = close.pct_change(10)
    df["return_20d"]    = close.pct_change(20)

    # ── RSI (14 + 7) ──────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(close, 14)
    df["rsi_7"]  = _rsi(close, 7)

    # ── Moving Averages ───────────────────────────────────────────────────
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_9"]  = close.ewm(span=9,  adjust=False).mean()

    # Price vs moving averages (relative position)
    df["price_vs_sma20"] = (close - df["sma_20"]) / (df["sma_20"] + 1e-9)
    df["price_vs_sma50"] = (close - df["sma_50"]) / (df["sma_50"] + 1e-9)
    df["sma20_vs_sma50"] = (df["sma_20"] - df["sma_50"]) / (df["sma_50"] + 1e-9)

    # ── MACD ──────────────────────────────────────────────────────────────
    df["macd"]          = df["ema_12"] - df["ema_26"]
    df["macd_signal"]   = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]     = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ───────────────────────────────────────────────────
    std20            = close.rolling(20).std()
    df["bb_upper"]   = df["sma_20"] + 2 * std20
    df["bb_lower"]   = df["sma_20"] - 2 * std20
    df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / (df["sma_20"] + 1e-9)
    df["bb_pct"]     = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ── Volatility ────────────────────────────────────────────────────────
    df["volatility_10d"] = df["daily_return"].rolling(10).std()
    df["volatility_20d"] = df["daily_return"].rolling(20).std()

    # ── Volume features ───────────────────────────────────────────────────
    df["volume_change"]  = volume.pct_change()
    df["volume_sma20"]   = volume.rolling(20).mean()
    df["volume_ratio"]   = volume / (df["volume_sma20"] + 1e-9)

    # ── ATR (Average True Range) ──────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / (close + 1e-9)

    # ── Stochastic Oscillator %K/%D ───────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── Williams %R ───────────────────────────────────────────────────────
    df["williams_r"] = -100 * (high14 - close) / (high14 - low14 + 1e-9)

    # ── On-Balance Volume (OBV) ───────────────────────────────────────────
    obv = (np.sign(df["daily_return"]) * volume).cumsum()
    df["obv_change"] = obv.pct_change(5)

    # ── Momentum ──────────────────────────────────────────────────────────
    df["momentum_10"] = close - close.shift(10)
    df["roc_10"]      = close.pct_change(10)          # Rate of Change

    # ── Target ────────────────────────────────────────────────────────────
    df["target"] = (close.shift(-1) > close).astype(int)
    df = df.iloc[:-1]  # drop last row (target NaN)

    indicator_cols = [
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
    df = df.dropna(subset=indicator_cols)
    logger.info("Indicators computed: %d rows remain.", len(df))
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
