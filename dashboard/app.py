"""
dashboard/app.py

StockSense — AI-Powered Stock Market Prediction Dashboard
=========================================================
Run with:
    streamlit run dashboard/app.py

Layout (Architecture Bible §08 Blueprint):
    Sidebar  — Ticker, date range, model info, refresh button
    Main     — 3 columns: [Price Trend (wide)] | [Prediction Card] | [Sentiment]
               Below   : SHAP panel
               Tab row : Trend · Sentiment · SHAP · Raw Data
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Add project root to path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load env vars from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("app")

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System CSS ─────────────────────────────────────────────────────────
from dashboard.theme import inject_css
inject_css()

# ── Component imports ─────────────────────────────────────────────────────────
from dashboard.components import (
    render_trend_panel,
    render_sentiment_panel,
    render_prediction_card,
    render_shap_panel,
)

# ──────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(ticker: str, years: int):
    from collection.price_fetcher import fetch_price_history
    from preprocessing.clean import clean_ohlcv
    raw = fetch_price_history(ticker, years=years)
    return clean_ohlcv(raw)


@st.cache_data(ttl=3600, show_spinner=False)
def load_indicators(ticker: str, years: int):
    from preprocessing.indicators import compute_indicators
    ohlcv = load_price_data(ticker, years)
    return compute_indicators(ohlcv)


@st.cache_data(ttl=43200, show_spinner=False)
def load_headlines(ticker: str):
    from collection.news_fetcher import fetch_news_headlines
    return fetch_news_headlines(ticker, days=30)


@st.cache_resource(show_spinner=False)
def load_lstm_embedder():
    from modeling.lstm_model import LSTMEmbedder
    model_path = _PROJECT_ROOT / "models" / "lstm_v1.pkl"
    embedder = LSTMEmbedder()
    if model_path.exists():
        embedder.load(model_path)
        return embedder
    return None


@st.cache_resource(show_spinner=False)
def load_xgb_predictor():
    from modeling.xgb_classifier import XGBPredictor
    model_path = _PROJECT_ROOT / "models" / "xgb_v1.json"
    predictor = XGBPredictor()
    if model_path.exists():
        predictor.load(model_path)
        return predictor
    return None


def load_training_metrics() -> dict:
    p = _PROJECT_ROOT / "models" / "training_metrics.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Inference pipeline (runs on demand)
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(ticker: str, years: int) -> dict:
    """Run the full prediction pipeline for *ticker*.

    Returns a dict with keys:
        direction, prob_up, shap_contributions, sentiment_series,
        headlines, ohlcv, indicators, error (optional)
    """
    result: dict = {}

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Fetching price data…"):
        try:
            ohlcv = load_price_data(ticker, years)
            indicators = load_indicators(ticker, years)
        except Exception as exc:
            return {"error": str(exc)}

    result["ohlcv"] = ohlcv
    result["indicators"] = indicators

    # ── Headlines + sentiment ─────────────────────────────────────────────────
    with st.spinner("Fetching news headlines…"):
        headlines = load_headlines(ticker)
    result["headlines"] = headlines

    from sentiment.aggregator import score_sentiment
    daily_sent = score_sentiment(headlines)

    # Build a simple single-day sentiment series for the chart
    sent_series = pd.DataFrame(
        [{"finbert_sentiment": daily_sent["finbert_sentiment"],
          "vader_sentiment": daily_sent["vader_sentiment"]}],
        index=[ohlcv.index[-1]],
    )
    result["sentiment_series"] = sent_series
    result["daily_sentiment"] = daily_sent

    # ── Models ────────────────────────────────────────────────────────────────
    embedder = load_lstm_embedder()
    predictor = load_xgb_predictor()

    models_ready = embedder is not None and predictor is not None

    if models_ready:
        with st.spinner("Running LSTM + XGBoost inference…"):
            try:
                # Build the last 60-day window
                from modeling.lstm_model import _WINDOW_SIZE
                from sklearn.preprocessing import MinMaxScaler

                lstm_feature_cols = [c for c in [
                    "Open", "High", "Low", "Close", "Volume",
                    "rsi_14", "sma_20", "sma_50", "ema_12", "ema_26", "ema_20",
                    "macd", "macd_signal", "bb_upper", "bb_lower",
                    "volatility_10d", "volume_change", "daily_return",
                ] if c in indicators.columns]

                raw_data = indicators[lstm_feature_cols].values.astype(np.float32)
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(raw_data)

                if len(data_scaled) >= _WINDOW_SIZE:
                    window = data_scaled[-_WINDOW_SIZE:]
                    embed = embedder.embed(window)
                else:
                    embed = np.zeros(16, dtype=np.float32)

                # Indicator values from latest row
                latest_ind = indicators.iloc[-1][
                    ["rsi_14", "sma_20", "sma_50", "ema_12", "ema_26", "ema_20",
                     "macd", "macd_signal", "bb_upper", "bb_lower",
                     "volatility_10d", "volume_change", "daily_return"]
                ].to_dict()

                from modeling.fusion import fuse_features
                fused = fuse_features(embed, latest_ind, daily_sent)

                direction, prob_up = predictor.predict(fused)
                result["direction"] = direction
                result["prob_up"] = prob_up
                result["fused"] = fused

                # SHAP
                from explainability.shap_explainer import explain
                result["shap_contributions"] = explain(fused)

            except Exception as exc:
                logger.error("Inference error: %s", exc)
                result["direction"] = "—"
                result["prob_up"] = 0.5
                result["shap_contributions"] = []
                result["inference_error"] = str(exc)
    else:
        result["direction"] = "—"
        result["prob_up"] = 0.5
        result["shap_contributions"] = []
        result["models_missing"] = True

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, int]:
    with st.sidebar:
        st.markdown(
            '<h1 style="font-size:18px;margin-bottom:4px;">📈 StockSense</h1>'
            '<p style="font-size:8.5pt;color:#767676;margin-top:0;">AI Stock Prediction</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        ticker = st.text_input(
            "Ticker",
            value=os.getenv("DEFAULT_TICKER", "AAPL"),
            max_chars=10,
            key="ticker_input",
            help="Enter a stock ticker symbol, e.g. AAPL, TSLA, MSFT",
        ).upper().strip()

        years = st.selectbox(
            "History window",
            options=[1, 2, 3, 5, 7, 10],
            index=3,  # default: 5Y
            key="years_select",
        )

        st.markdown("---")
        run_btn = st.button("▶ Run Prediction", use_container_width=True, key="run_btn")

        st.markdown("---")

        # Model status
        embedder = load_lstm_embedder()
        predictor = load_xgb_predictor()
        metrics = load_training_metrics()

        st.markdown('<h3 style="font-size:9pt;">Model Status</h3>', unsafe_allow_html=True)
        lstm_ok = embedder is not None
        xgb_ok = predictor is not None

        st.markdown(
            f'<div class="kpi-sub">LSTM &nbsp;{"✓ ready" if lstm_ok else "✗ not trained"}</div>'
            f'<div class="kpi-sub">XGBoost &nbsp;{"✓ ready" if xgb_ok else "✗ not trained"}</div>',
            unsafe_allow_html=True,
        )

        if metrics.get("test_accuracy"):
            acc = metrics["test_accuracy"]
            st.markdown(
                f'<div class="kpi-sub" style="margin-top:8px;">'
                f'Test accuracy: <b>{acc*100:.1f}%</b></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if not lstm_ok or not xgb_ok:
            st.info(
                "Models not found. Run the training pipeline:\n\n"
                "```\npython -m modeling.train_pipeline --ticker AAPL\n```"
            )

        # NewsAPI key notice
        api_key = os.getenv("NEWS_API_KEY", "")
        if not api_key or api_key == "your_newsapi_key_here":
            st.markdown(
                '<div class="stale-notice">No NewsAPI key — '
                'set NEWS_API_KEY in .env for live headlines.</div>',
                unsafe_allow_html=True,
            )

    return ticker, years, run_btn


# ──────────────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ticker, years, run_btn = render_sidebar()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<h1>StockSense &nbsp;<span style="color:#767676;font-weight:400;">'
        f'{ticker}</span></h1>',
        unsafe_allow_html=True,
    )

    # Session state: hold results across reruns
    if "result" not in st.session_state:
        st.session_state.result = None

    if run_btn or st.session_state.result is None:
        st.session_state.result = run_inference(ticker, years)

    result = st.session_state.result

    if "error" in result:
        st.error(f"Data fetch error: {result['error']}")
        st.stop()

    ohlcv = result["ohlcv"]
    indicators = result["indicators"]
    latest_close = float(ohlcv["Close"].iloc[-1])

    # ── Top-row KPIs ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Last Close</div>'
            f'<div class="kpi-value" style="font-size:20pt;">${latest_close:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        rsi_val = indicators["rsi_14"].iloc[-1] if "rsi_14" in indicators.columns else 0
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">RSI (14)</div>'
            f'<div class="kpi-value" style="font-size:20pt;">{rsi_val:.1f}</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        sent = result.get("daily_sentiment", {})
        fb = sent.get("finbert_sentiment", 0.0)
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">FinBERT Sentiment</div>'
            f'<div class="kpi-value" style="font-size:20pt;">{fb:+.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        direction = result.get("direction", "—")
        prob_up = result.get("prob_up", 0.5)
        conf = round(prob_up * 100 if direction == "UP" else (1 - prob_up) * 100, 1)
        badge_class = "badge-up" if direction == "UP" else "badge-down"
        arrow = "▲" if direction == "UP" else ("▼" if direction == "DOWN" else "●")
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Prediction</div>'
            f'<div class="kpi-value" style="font-size:20pt;">{arrow} {direction}</div>'
            f'<div style="margin-top:8px;">'
            f'<span class="badge {badge_class}">{conf}% confidence</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)

    # ── Missing models notice ──────────────────────────────────────────────────
    if result.get("models_missing"):
        st.warning(
            "**Models not trained yet.** Price charts and sentiment are live, "
            "but prediction and SHAP require training.\n\n"
            "Run: `python -m modeling.train_pipeline --ticker AAPL`"
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_trend, tab_sentiment, tab_prediction, tab_shap, tab_raw = st.tabs([
        "📈 Price Trend",
        "📰 Sentiment",
        "🤖 Prediction",
        "🔍 SHAP",
        "📊 Raw Data",
    ])

    with tab_trend:
        render_trend_panel(ohlcv, indicators, ticker)

    with tab_sentiment:
        render_sentiment_panel(
            result.get("sentiment_series", pd.DataFrame()),
            result.get("headlines", []),
        )

    with tab_prediction:
        if not result.get("models_missing") and direction != "—":
            render_prediction_card(direction, prob_up, latest_close, ticker)
        else:
            st.info(
                "Prediction not available. Train the models first.\n\n"
                "```\npython -m modeling.train_pipeline --ticker AAPL\n```"
            )

    with tab_shap:
        shap_contribs = result.get("shap_contributions", [])
        render_shap_panel(shap_contribs)

    with tab_raw:
        st.markdown('<h2>Raw OHLCV Data</h2>', unsafe_allow_html=True)
        st.dataframe(ohlcv.sort_index(ascending=False), use_container_width=True)
        st.markdown('<h2>Indicator Features</h2>', unsafe_allow_html=True)
        st.dataframe(indicators.sort_index(ascending=False), use_container_width=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<hr class="ss-divider">'
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8.5pt;'
        'color:#A3A3A3;text-align:center;">'
        'StockSense is a decision-support tool only — not financial advice. '
        'All data sourced from public markets via yfinance.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
