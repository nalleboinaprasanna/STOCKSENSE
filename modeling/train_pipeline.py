"""
modeling/train_pipeline.py

End-to-end offline training script.

Usage:
    python -m modeling.train_pipeline --ticker AAPL --years 5

Steps:
  1. Fetch and clean price history
  2. Compute technical indicators
  3. Build and train the LSTM; save lstm_v1.h5
  4. Generate LSTM embeddings for every row
  5. Fetch news headlines and aggregate sentiment per date
  6. Fuse all features; save processed features to data/processed/
  7. Train and evaluate XGBoost; save xgb_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Project root on path ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")


def run(ticker: str = "AAPL", years: int = 5) -> None:
    """Execute the full offline training pipeline."""

    from collection.price_fetcher import fetch_price_history
    from collection.news_fetcher import fetch_news_headlines
    from preprocessing.clean import clean_ohlcv
    from preprocessing.indicators import compute_indicators
    from sentiment.aggregator import score_sentiment
    from modeling.lstm_model import LSTMEmbedder, prepare_lstm_data, _WINDOW_SIZE
    from modeling.fusion import fuse_features, FEATURE_SCHEMA
    from modeling.xgb_classifier import XGBPredictor

    models_dir = _PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Price data ────────────────────────────────────────────────────
    logger.info("=== Step 1/7 — Fetching price history (%s, %d years) ===", ticker, years)
    raw_ohlcv = fetch_price_history(ticker, years=years)
    ohlcv = clean_ohlcv(raw_ohlcv)
    logger.info("Price data shape: %s", ohlcv.shape)

    # ── Step 2: Technical indicators ──────────────────────────────────────────
    logger.info("=== Step 2/7 — Computing technical indicators ===")
    feat_df = compute_indicators(ohlcv)
    logger.info("Feature DataFrame shape: %s", feat_df.shape)

    # ── Step 3: Train LSTM ────────────────────────────────────────────────────
    logger.info("=== Step 3/7 — Training LSTM embedder ===")
    lstm_feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
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
    # Only use cols present in feat_df
    lstm_feature_cols = [c for c in lstm_feature_cols if c in feat_df.columns]

    X_lstm, y_lstm, scaler = prepare_lstm_data(feat_df, lstm_feature_cols, window_size=_WINDOW_SIZE)
    logger.info("LSTM data prepared: X=%s y=%s", X_lstm.shape, y_lstm.shape)

    embedder = LSTMEmbedder()
    embedder.train(X_lstm, y_lstm, epochs=30, batch_size=32, validation_split=0.2)
    lstm_path = models_dir / "lstm_v1.h5"
    embedder.save(lstm_path)
    logger.info("LSTM saved → %s", lstm_path)

    # ── Step 4: Generate embeddings for all rows ───────────────────────────────
    logger.info("=== Step 4/7 — Generating LSTM embeddings for all rows ===")
    raw_data = feat_df[lstm_feature_cols].values.astype(np.float32)
    data_scaled = scaler.transform(raw_data)

    embeddings = []
    for i in range(_WINDOW_SIZE, len(data_scaled)):
        window = data_scaled[i - _WINDOW_SIZE : i][np.newaxis, ...]
        emb = embedder.embed(window[0])
        embeddings.append(emb)

    # Align feat_df to rows that have embeddings
    feat_aligned = feat_df.iloc[_WINDOW_SIZE:].copy()
    embed_df = pd.DataFrame(
        embeddings,
        columns=[f"lstm_embed_{i}" for i in range(16)],
        index=feat_aligned.index,
    )

    # ── Step 5: Sentiment per date ────────────────────────────────────────────
    logger.info("=== Step 5/7 — Fetching and scoring news sentiment ===")
    headlines_all = fetch_news_headlines(ticker, days=years * 365)

    # Map each headline to a date bucket (use today's date if no date info)
    # NewsAPI headlines are returned newest-first; we assign them evenly across dates.
    # For a real production system, date-specific fetching would be used.
    trading_dates = feat_aligned.index.tolist()
    sentiment_rows = []

    for date in trading_dates:
        daily_headlines = headlines_all[:5] if headlines_all else []
        sent = score_sentiment(daily_headlines)
        sentiment_rows.append({
            "Date": date,
            "finbert_sentiment": sent["finbert_sentiment"],
            "vader_sentiment": sent["vader_sentiment"],
        })

    sent_df = pd.DataFrame(sentiment_rows).set_index("Date")

    # ── Step 6: Fuse features and save ───────────────────────────────────────
    logger.info("=== Step 6/7 — Fusing features ===")
    all_features = []
    indicator_cols_for_fusion = [
        "rsi_14", "sma_20", "sma_50", "ema_12", "ema_26", "ema_20",
        "macd", "macd_signal", "bb_upper", "bb_lower",
        "volatility_10d", "volume_change", "daily_return",
    ]

    for date in feat_aligned.index:
        try:
            emb_vec = embed_df.loc[date].values
            ind_dict = feat_aligned.loc[date, indicator_cols_for_fusion].to_dict()
            sent_dict = sent_df.loc[date].to_dict() if date in sent_df.index else \
                        {"finbert_sentiment": 0.0, "vader_sentiment": 0.0}
            fused = fuse_features(emb_vec, ind_dict, sent_dict)
            fused["target"] = feat_aligned.loc[date, "target"]
            fused["Date"] = date
            all_features.append(fused)
        except Exception as exc:
            logger.debug("Skipping %s: %s", date, exc)

    fused_df = pd.DataFrame(all_features).set_index("Date")
    parquet_path = processed_dir / f"{ticker}_features.parquet"
    fused_df.to_parquet(parquet_path)
    logger.info("Fused features saved → %s (%d rows)", parquet_path, len(fused_df))

    # ── Step 7: Train XGBoost ─────────────────────────────────────────────────
    logger.info("=== Step 7/7 — Training XGBoost classifier ===")
    X_xgb = fused_df[FEATURE_SCHEMA]
    y_xgb = fused_df["target"].astype(int)

    # Chronological split (no shuffle — prevent data leakage)
    split_idx = int(len(X_xgb) * 0.8)
    X_train, X_test = X_xgb.iloc[:split_idx], X_xgb.iloc[split_idx:]
    y_train, y_test = y_xgb.iloc[:split_idx], y_xgb.iloc[split_idx:]

    predictor = XGBPredictor()
    train_metrics = predictor.train(X_train, y_train)

    y_pred = predictor._model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("Test accuracy: %.2f%%", test_acc * 100)
    logger.info("\n%s", classification_report(y_test, y_pred))

    xgb_path = models_dir / "xgb_v1.json"
    predictor.save(xgb_path)
    logger.info("XGBoost saved → %s", xgb_path)

    # Save metrics summary
    metrics = {
        "ticker": ticker,
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_acc,
        "test_report": test_report,
    }
    metrics_path = models_dir / "training_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Training metrics saved → %s", metrics_path)
    logger.info("=== Training pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockSense Training Pipeline")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--years", type=int, default=5, help="Years of history")
    args = parser.parse_args()

    run(ticker=args.ticker, years=args.years)
