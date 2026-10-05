"""
collection/news_fetcher.py

Fetches financial news headlines for a ticker from NewsAPI and caches
the result to data/raw/news/{ticker}_{date}.json.

Public contract (Architecture Bible §09):
    fetch_news_headlines(ticker: str, days: int = 7) -> list[str]
    Returns a list of headline strings for the recent *days* days.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "raw" / "news"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_NEWSAPI_BASE = "https://newsapi.org/v2/everything"


def fetch_news_headlines(ticker: str, days: int = 30) -> list[str]:
    """Retrieve recent news headlines mentioning *ticker*.

    Results are cached per (ticker, date) to ``data/raw/news/``.
    If the NewsAPI key is absent or the request fails, the most recent
    cached file for the ticker is returned.  If no cache exists at all,
    an empty list is returned so the pipeline can continue with VADER/
    FinBERT falling back to neutral sentiment.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. ``"AAPL"``.
    days : int
        Number of calendar days of headlines to fetch (default 30).

    Returns
    -------
    list[str]
        Flat list of headline strings, newest first.
    """
    ticker = ticker.upper().strip()
    today_str = datetime.today().strftime("%Y-%m-%d")
    cache_path = _CACHE_DIR / f"{ticker}_{today_str}.json"

    api_key = os.getenv("NEWS_API_KEY", "")
    if cache_path.exists():
        with open(cache_path) as fh:
            cached_headlines = json.load(fh)
        if cached_headlines:
            logger.info("News cache hit for %s on %s", ticker, today_str)
            return cached_headlines
        logger.info("Today's news cache is empty for %s; checking NewsAPI.", ticker)

    if not api_key or api_key == "your_newsapi_key_here":
        logger.warning("NEWS_API_KEY not set — using cached headlines or empty list.")
        return _load_latest_cache(ticker)

    from_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": ticker,
        "from": from_date,
        "to": today_str,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(_NEWSAPI_BASE, params=params, timeout=5)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        headlines = [a["title"] for a in articles if a.get("title")]
    except Exception as exc:
        logger.error("NewsAPI request failed: %s — falling back to cache.", exc)
        return _load_latest_cache(ticker)

    # Cache the result
    with open(cache_path, "w") as fh:
        json.dump(headlines, fh)
    logger.info("Cached %d headlines for %s → %s", len(headlines), ticker, cache_path)

    return headlines


def _load_latest_cache(ticker: str) -> list[str]:
    """Return the most recent cached headlines for *ticker*, or an empty list."""
    candidates = sorted(_CACHE_DIR.glob(f"{ticker}_*.json"), reverse=True)
    if candidates:
        logger.info("Returning stale cache: %s", candidates[0])
        with open(candidates[0]) as fh:
            return json.load(fh)
    logger.warning("No cached headlines found for %s — returning [].", ticker)
    return []
