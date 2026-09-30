"""
dashboard/components/trend_panel.py

Price trend + technical indicator panel (Tab 1).
Renders a candlestick chart with SMA overlays and a volume sub-chart.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_trend_panel(ohlcv: pd.DataFrame, indicators: pd.DataFrame, ticker: str) -> None:
    """Render the Price Trend + Indicators panel."""
    st.markdown('<h2>Price Trend &amp; Indicators</h2>', unsafe_allow_html=True)

    # ── Main figure: candlestick + MAs ────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=("", "Volume"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            close=ohlcv["Close"],
            name=ticker,
            increasing_line_color="#0A0A0A",
            decreasing_line_color="#767676",
            increasing_fillcolor="#0A0A0A",
            decreasing_fillcolor="#FFFFFF",
        ),
        row=1, col=1,
    )

    # SMA 20 overlay
    if "sma_20" in indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators["sma_20"],
                name="SMA 20",
                line=dict(color="#4A4A4A", width=1.5, dash="solid"),
            ),
            row=1, col=1,
        )

    # SMA 50 overlay (dashed)
    if "sma_50" in indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators["sma_50"],
                name="SMA 50",
                line=dict(color="#A3A3A3", width=1.5, dash="dash"),
            ),
            row=1, col=1,
        )

    # Bollinger Bands — fill between
    if "bb_upper" in indicators.columns and "bb_lower" in indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators["bb_upper"],
                name="BB Upper",
                line=dict(color="#CFCFCF", width=1, dash="dot"),
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators["bb_lower"],
                name="BB Lower",
                line=dict(color="#CFCFCF", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(10,10,10,0.04)",
            ),
            row=1, col=1,
        )

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=ohlcv.index,
            y=ohlcv["Volume"],
            name="Volume",
            marker_color="#CFCFCF",
        ),
        row=2, col=1,
    )

    _apply_chart_style(fig)
    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(family="JetBrains Mono", size=9, color="#4A4A4A"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── RSI sub-chart ─────────────────────────────────────────────────────────
    if "rsi_14" in indicators.columns:
        st.markdown('<h3>RSI (14)</h3>', unsafe_allow_html=True)
        rsi_fig = go.Figure()
        rsi_fig.add_trace(
            go.Scatter(
                x=indicators.index,
                y=indicators["rsi_14"],
                name="RSI 14",
                line=dict(color="#0A0A0A", width=2),
            )
        )
        # Reference lines at 30 and 70
        for level in [30, 70]:
            rsi_fig.add_hline(
                y=level,
                line_dash="dot",
                line_color="#CFCFCF",
                line_width=1,
                annotation_text=str(level),
                annotation_font=dict(family="JetBrains Mono", size=9, color="#767676"),
            )
        _apply_chart_style(rsi_fig)
        rsi_fig.update_layout(height=200, showlegend=False)
        rsi_fig.update_yaxes(range=[0, 100])
        st.plotly_chart(rsi_fig, use_container_width=True)


def _apply_chart_style(fig: go.Figure) -> None:
    """Apply the StockSense chart grammar to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="JetBrains Mono", size=9, color="#767676"),
        margin=dict(l=0, r=0, t=8, b=0),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#E6E6E6",
        tickfont=dict(family="JetBrains Mono", size=9, color="#767676"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=0.5,
        zeroline=False,
        linecolor="#E6E6E6",
        tickfont=dict(family="JetBrains Mono", size=9, color="#767676"),
    )
