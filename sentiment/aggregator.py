"""
sentiment/aggregator.py

Aggregates VADER and FinBERT scores into a single daily sentiment dict.

Public contract (Architecture Bible §09):
    score_sentiment(headlines: list[str]) -> dict
    Returns {
        "finbert_sentiment": float,   # mean FinBERT score
        "vader_sentiment":   float,   # mean VADER compound
        "headline_count":    int,
    }
"""

from __future__ import annotations

import logging

import numpy as np

from .finbert_scorer import score_finbert
from .vader_scorer import score_vader

logger = logging.getLogger(__name__)


def score_sentiment(headlines: list[str]) -> dict:
    """Compute aggregated daily sentiment from a list of headlines.

    Parameters
    ----------
    headlines : list[str]
        All headlines for one trading day (may be empty).

    Returns
    -------
    dict
        Keys: ``finbert_sentiment``, ``vader_sentiment``, ``headline_count``.
    """
    if not headlines:
        logger.debug("No headlines — returning neutral sentiment.")
        return {"finbert_sentiment": 0.0, "vader_sentiment": 0.0, "headline_count": 0}

    vader_scores = score_vader(headlines)
    finbert_scores = score_finbert(headlines)

    result = {
        "finbert_sentiment": float(np.mean(finbert_scores)) if finbert_scores else 0.0,
        "vader_sentiment": float(np.mean(vader_scores)) if vader_scores else 0.0,
        "headline_count": len(headlines),
    }
    logger.debug("Sentiment aggregated: %s", result)
    return result
