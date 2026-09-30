"""
dashboard/components/prediction_card.py

Prediction card component.
Renders the UP/DOWN prediction, confidence bar, and KPI stats.
"""

from __future__ import annotations

import streamlit as st


def render_prediction_card(
    direction: str,
    prob_up: float,
    latest_close: float,
    ticker: str,
) -> None:
    """Render the next-day prediction card.

    Parameters
    ----------
    direction : str
        ``"UP"`` or ``"DOWN"``.
    prob_up : float
        Probability of UP (0–1).
    latest_close : float
        Most recent closing price.
    ticker : str
        Stock ticker symbol.
    """
    is_up = direction == "UP"
    badge_class = "badge-up" if is_up else "badge-down"
    arrow = "▲" if is_up else "▼"
    confidence_pct = round(prob_up * 100 if is_up else (1 - prob_up) * 100, 1)

    st.markdown('<h2>Next-Day Prediction</h2>', unsafe_allow_html=True)

    # ── Main direction display ────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="kpi-card" style="border-width:2px;">
            <div class="kpi-label">{ticker} · NEXT-DAY DIRECTION</div>
            <div class="kpi-value">{arrow} {direction}</div>
            <div style="margin-top:16px;">
                <span class="badge {badge_class}">{arrow} {direction} · {confidence_pct}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Confidence bar ────────────────────────────────────────────────────────
    bar_fill = int(prob_up * 100)
    bar_empty = 100 - bar_fill
    st.markdown(
        f"""
        <div style="margin-top:16px;">
            <div class="kpi-label">Model probability — UP vs DOWN</div>
            <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;
                        border:1px solid #CFCFCF;margin-top:8px;">
                <div style="width:{bar_fill}%;background:#0A0A0A;"></div>
                <div style="width:{bar_empty}%;background:#F2F2F2;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px;">
                <span class="mono" style="font-size:9pt;color:#767676;">UP {bar_fill}%</span>
                <span class="mono" style="font-size:9pt;color:#767676;">DOWN {bar_empty}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Latest close KPI ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="kpi-card" style="margin-top:16px;">
            <div class="kpi-label">Last Close</div>
            <div class="kpi-value" style="font-size:20pt;">${latest_close:,.2f}</div>
            <div class="kpi-sub">{ticker}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
