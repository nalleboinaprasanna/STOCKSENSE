"""
collection/__init__.py
Exposes the public API of the collection layer.
"""

from .price_fetcher import fetch_price_history
from .news_fetcher import fetch_news_headlines

__all__ = ["fetch_price_history", "fetch_news_headlines"]
