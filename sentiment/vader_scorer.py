"""
sentiment/vader_scorer.py

Fast lexicon-based sentiment scoring via VADER.
Returns compound scores in [-1, +1].
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def score_vader(headlines: list[str]) -> list[float]:
    """Score each headline with VADER and return compound scores.

    Parameters
    ----------
    headlines : list[str]
        List of news headline strings.

    Returns
    -------
    list[float]
        VADER compound score per headline (range −1 to +1).
    """
    if not headlines:
        return []

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        logger.error("vaderSentiment not installed. Run: pip install vaderSentiment")
        return [0.0] * len(headlines)

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    logger.debug("VADER scored %d headlines.", len(scores))
    return scores
