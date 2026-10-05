"""
sentiment/finbert_scorer.py

FinBERT-based financial sentiment scoring via HuggingFace Transformers.
Model: ProsusAI/finbert  (~440 MB, downloaded once and cached by HF).

Falls back gracefully to 0.0 if the model is unavailable (no internet,
first-run download in progress, etc.).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "ProsusAI/finbert"
_LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load the FinBERT pipeline once and cache it in memory."""
    from transformers import pipeline  # noqa: PLC0415
    logger.info("Loading FinBERT model '%s' (one-time download ~440 MB)…", _MODEL_NAME)
    return pipeline(
        "text-classification",
        model=_MODEL_NAME,
        tokenizer=_MODEL_NAME,
        truncation=True,
        max_length=512,
    )


def score_finbert(headlines: list[str]) -> list[float]:
    """Score each headline with FinBERT.

    Maps labels → numeric: positive=+1, negative=−1, neutral=0.
    The returned score is label_score × confidence, giving a value in [−1, +1].

    Parameters
    ----------
    headlines : list[str]
        List of news headline strings.

    Returns
    -------
    list[float]
        Per-headline FinBERT sentiment score in [−1, +1].
    """
    if not headlines:
        return []

    try:
        pipe = _load_pipeline()
    except Exception as exc:
        logger.warning("FinBERT unavailable (%s); returning 0.0 for all headlines.", exc)
        return [0.0] * len(headlines)

    scores: list[float] = []
    for i in range(0, len(headlines), _BATCH_SIZE):
        batch = headlines[i : i + _BATCH_SIZE]
        try:
            results = pipe(batch, batch_size=_BATCH_SIZE)
            for r in results:
                direction = _LABEL_MAP.get(r["label"].lower(), 0.0)
                scores.append(direction * r["score"])
        except Exception as exc:
            logger.warning("FinBERT batch %d failed: %s", i // _BATCH_SIZE, exc)
            scores.extend([0.0] * len(batch))

    logger.debug("FinBERT scored %d headlines.", len(scores))
    return scores
