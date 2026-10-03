"""
dashboard/theme.py

Design token constants and Streamlit CSS injection.
Exact tokens from StockSense Design System Document 02.

Usage:
    from dashboard.theme import inject_css, TOKENS
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Design Tokens (Document 02 §12)
# ──────────────────────────────────────────────────────────────────────────────

TOKENS = {
    "color.ink": "#0A0A0A",
    "color.gray.1": "#4A4A4A",
    "color.gray.2": "#767676",
    "color.gray.3": "#A3A3A3",
    "color.gray.4": "#CFCFCF",
    "color.gray.5": "#E6E6E6",
    "color.gray.6": "#F2F2F2",
    "color.paper": "#FFFFFF",
    "font.display": "Space Grotesk",
    "font.body": "Inter",
    "font.mono": "JetBrains Mono",
    "radius.card": "8px",
    "radius.pill": "999px",
    "border.default": "1px solid #CFCFCF",
}

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset / Base ─────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stHeader"], [data-testid="stBottom"],
section[data-testid="stSidebar"], .main, .block-container {
    font-family: 'Inter', sans-serif;
    color: #0A0A0A;
    background-color: #FFFFFF !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit chrome ────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page container ───────────────────────────────────────────────────── */
.main .block-container {
    max-width: 1200px;
    padding: 32px 32px 64px 32px;
    background: #FFFFFF !important;
}

/* ── White app shell / inputs ─────────────────────────────────────────── */
header[data-testid="stHeader"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"],
[data-testid="stBottom"] {
    background-color: #FFFFFF !important;
}
div[data-baseweb="select"], div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"], div[data-testid="stSelectbox"] > div,
.stSelectbox, div[data-testid="stDateInput"],
div[data-testid="stDateInput"] input {
    background: #FFFFFF;
    color: #0A0A0A;
}
div[role="listbox"], div[role="option"], .stSelectbox [role="combobox"],
div[data-baseweb="popover"], div[data-baseweb="calendar"],
div[data-baseweb="calendar"] * {
    background: #FFFFFF;
    color: #0A0A0A;
}

/* ── Typography ───────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif;
    color: #0A0A0A;
    font-weight: 700;
    letter-spacing: -0.01em;
}
h1 { font-size: 26px; line-height: 1.1; }
h2 { font-size: 18px; line-height: 1.15; }
h3 { font-size: 13px; line-height: 1.2; font-weight: 600; }

p, li, span { font-size: 10pt; line-height: 1.55; color: #0A0A0A; }

code, .mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10pt;
}

/* ── Cards ─────────────────────────────────────────────────────────────── */
.ss-card {
    background: #FFFFFF;
    border: 1px solid #CFCFCF;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 16px;
    transition: background 80ms linear;
}
.ss-card:hover { background: #F2F2F2; }

/* ── KPI card ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #CFCFCF;
    border-radius: 8px;
    padding: 24px 24px 20px;
    text-align: center;
}
.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 8.5pt;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #767676;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26pt;
    font-weight: 700;
    color: #0A0A0A;
    line-height: 1.1;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9pt;
    color: #767676;
    margin-top: 4px;
}

/* ── Direction badge ──────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 8.5pt;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid #0A0A0A;
    color: #0A0A0A;
    background: #FFFFFF;
}
.badge-up {
    background: #0A0A0A;
    color: #FFFFFF;
    border-color: #0A0A0A;
}
.badge-down {
    background: #FFFFFF;
    color: #0A0A0A;
    border-color: #0A0A0A;
}

/* ── SHAP bars ────────────────────────────────────────────────────────── */
.shap-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 0.5px solid #E6E6E6;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9pt;
}
.shap-feature { width: 180px; color: #4A4A4A; }
.shap-bar-wrap { flex: 1; background: #F2F2F2; border-radius: 2px; height: 8px; }
.shap-bar-pos  { height: 8px; border-radius: 2px; background: #0A0A0A; }
.shap-bar-neg  { height: 8px; border-radius: 2px; background: #FFFFFF;
                 border: 1px solid #0A0A0A; }
.shap-val { width: 60px; text-align: right; color: #4A4A4A; }

/* ── Loading bar ──────────────────────────────────────────────────────── */
div[data-stale] { animation: none; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    width: 320px !important;
    min-width: 320px !important;
    background: #FFFFFF;
    border-right: 1px solid #CFCFCF;
}
section[data-testid="stSidebar"] .block-container {
    padding: 24px 16px;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr.ss-divider {
    border: none;
    border-top: 0.5px solid #E6E6E6;
    margin: 24px 0;
}

/* ── Stale notice ────────────────────────────────────────────────────── */
.stale-notice {
    font-family: 'Inter', sans-serif;
    font-size: 8.5pt;
    color: #767676;
    background: #F2F2F2;
    border: 1px solid #CFCFCF;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
}
"""


def inject_css() -> None:
    """Inject the StockSense design system CSS into the Streamlit page."""
    import streamlit as st
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
