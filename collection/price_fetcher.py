"""
collection/price_fetcher.py

Fetches OHLCV price history from yfinance and caches the result
to data/raw/prices/{ticker}.csv to avoid redundant network calls.

Public contract (Architecture Bible §09):
    fetch_price_history(ticker: str, years: int = 5) -> pd.DataFrame
    Returns raw OHLCV indexed by date (DatetimeIndex, tz-naive).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Resolve the project root relative to this file so the module works
# regardless of the cwd when it is imported.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "raw" / "prices"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DOWNLOAD_TIMEOUT_SECONDS = 5


def fetch_price_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """Download OHLCV history for *ticker* covering the last *years* years.

    If a cached CSV already exists and was written today, the local copy
    is returned immediately; otherwise the cache is refreshed from yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. ``"AAPL"``.
    years : int
        Number of calendar years of history to fetch (default 5).

    Returns
    -------
    pd.DataFrame
        Columns: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
        Index: ``DatetimeIndex`` (tz-naive, daily frequency).

    Raises
    ------
    RuntimeError
        If yfinance returns an empty response and no cache is available.
    """
    ticker = ticker.upper().strip()
    cache_path = _CACHE_DIR / f"{ticker}.csv"

    # Return cached data if it is still fresh (written today)
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if mtime.date() == datetime.today().date():
            logger.info("Cache hit for %s — loading from %s", ticker, cache_path)
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            df.index = df.index.tz_localize(None)
            required_start = datetime.today() - timedelta(days=years * 365 + 10)
            if not df.empty and df.index.min().date() <= required_start.date():
                return df
            logger.info(
                "Cached history for %s does not cover %d years; refreshing.",
                ticker,
                years,
            )

    # Fetch from yfinance
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 10)  # +10 for market holidays
    logger.info("Fetching %s from yfinance [%s → %s]", ticker, start.date(), end.date())

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        threads=False,
        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
    )

    if raw.empty:
        if cache_path.exists():
            logger.warning("yfinance returned empty; using stale cache for %s", ticker)
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            df.index = df.index.tz_localize(None)
            return df
        raise RuntimeError(
            f"yfinance returned no data for '{ticker}'. "
            "Check the ticker symbol and your internet connection."
        )

    # Flatten multi-level columns if present (yfinance ≥ 0.2.37)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    ohlcv.index = ohlcv.index.tz_localize(None)
    ohlcv.index.name = "Date"

    # Persist to cache
    ohlcv.to_csv(cache_path)
    logger.info("Cached %d rows → %s", len(ohlcv), cache_path)

    return ohlcv
