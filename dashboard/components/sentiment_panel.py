"""
dashboard/components/sentiment_panel.py

News sentiment panel (Tab 2).
Renders a daily sentiment trend chart (FinBERT + VADER) and top headlines.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_sentiment_panel(
    sentiment_series: pd.DataFrame,
    headlines: list[str],
) -> None:
    """Render the News Sentiment panel.

    Parameters
    ----------
    sentiment_series : pd.DataFrame
        Columns: ``finbert_sentiment``, ``vader_sentiment``, index: Date.
    headlines : list[str]
        Recent headlines (up to ~50 shown).
    """
    st.markdown('<h2>News Sentiment</h2>', unsafe_allow_html=True)

    if not headlines:
        st.info(
            "Sentiment is unavailable because no news headlines were found. "
            "Add NEWS_API_KEY in Streamlit Cloud app secrets to fetch live news."
        )
        return

    if sentiment_series.empty:
        st.info("No sentiment data available. Provide a NewsAPI key in the sidebar.")
        return

    # ── Sentiment trend chart ──────────────────────────────────────────────
    fig = go.Figure()

    if "finbert_sentiment" in sentiment_series.columns:
        fig.add_trace(
            go.Scatter(
                x=sentiment_series.index,
                y=sentiment_series["finbert_sentiment"].rolling(7).mean(),
                name="FinBERT (7-day MA)",
                line=dict(color="#0A0A0A", width=2),
                fill="tozeroy",
                fillcolor="rgba(10,10,10,0.06)",
            )
        )

    if "vader_sentiment" in sentiment_series.columns:
        fig.add_trace(
            go.Scatter(
                x=sentiment_series.index,
                y=sentiment_series["vader_sentiment"].rolling(7).mean(),
                name="VADER (7-day MA)",
                line=dict(color="#767676", width=1.5, dash="dash"),
            )
        )

    # Neutral baseline
    fig.add_hline(y=0, line_color="#E6E6E6", line_width=0.5)

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=300,
        margin=dict(l=0, r=0, t=8, b=0),
        font=dict(family="JetBrains Mono", size=9, color="#767676"),
        legend=dict(
            font=dict(family="JetBrains Mono", size=9),
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        yaxis=dict(
            range=[-1, 1],
            tickvals=[-1, -0.5, 0, 0.5, 1],
            gridcolor="#E6E6E6",
            gridwidth=0.5,
        ),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Headlines table ────────────────────────────────────────────────────
    st.markdown('<h3>Recent Headlines</h3>', unsafe_allow_html=True)
    for i, h in enumerate(headlines[:30]):
        st.markdown(
            f'<div class="ss-card" style="padding:12px 16px;margin-bottom:8px;">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9pt;'
            f'color:#4A4A4A;">{i+1}.</span>&nbsp;'
            f'<span style="font-size:10pt;">{h}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
