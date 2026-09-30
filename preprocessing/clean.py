"""
preprocessing/clean.py

Cleans and validates raw OHLCV data before indicator computation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a raw OHLCV DataFrame.

    Steps performed:
    1. Drop rows where Close is NaN.
    2. Forward-fill small gaps (≤ 5 consecutive missing days).
    3. Remove duplicate index entries, keeping the last.
    4. Sort the index ascending.
    5. Ensure numeric dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with columns ``Open``, ``High``, ``Low``,
        ``Close``, ``Volume`` and a ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Cleaned copy; original is never mutated.
    """
    df = df.copy()

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

    # Convert to numeric, coerce bad values to NaN
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no close price
    before = len(df)
    df = df[df["Close"].notna()]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with NaN Close.", dropped)

    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]

    # Sort chronologically
    df = df.sort_index()

    # Forward-fill volume gaps
    df["Volume"] = df["Volume"].ffill(limit=5).fillna(0)

    logger.info("Cleaned OHLCV: %d rows, %s → %s", len(df),
                df.index[0].date(), df.index[-1].date())
    return df
