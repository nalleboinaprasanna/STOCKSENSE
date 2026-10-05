"""
dashboard/components/shap_panel.py

SHAP explanation panel (Tab 3).
Renders a horizontal bar chart of feature contributions.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_shap_panel(shap_contributions: list[dict]) -> None:
    """Render the SHAP feature explanation panel.

    Parameters
    ----------
    shap_contributions : list[dict]
        Each entry: ``{"feature": str, "shap_value": float}``,
        sorted by |shap_value| descending (output of ``explain()``).
    """
    st.markdown('<h2>SHAP — Feature Contributions</h2>', unsafe_allow_html=True)

    if not shap_contributions:
        st.info(
            "SHAP feature contributions could not be computed for this prediction. "
            "Check any error details above and try running the prediction again."
        )
        return

    # Top 15 features
    top = shap_contributions[:15]
    features = [d["feature"] for d in top]
    values = [d["shap_value"] for d in top]
    colors = ["#0A0A0A" if v >= 0 else "#CFCFCF" for v in values]

    # ── Plotly horizontal bar chart ───────────────────────────────────────────
    fig = go.Figure(
        go.Bar(
            y=features[::-1],
            x=values[::-1],
            orientation="h",
            marker=dict(
                color=colors[::-1],
                line=dict(color="#0A0A0A", width=1),
            ),
            text=[f"{v:+.3f}" for v in values[::-1]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=9, color="#4A4A4A"),
        )
    )

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=max(300, len(top) * 28),
        margin=dict(l=0, r=80, t=8, b=0),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E6E6E6",
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor="#CFCFCF",
            zerolinewidth=1,
            tickfont=dict(family="JetBrains Mono", size=9, color="#767676"),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(family="JetBrains Mono", size=9, color="#4A4A4A"),
        ),
        font=dict(family="JetBrains Mono", size=9, color="#767676"),
    )
    fig.add_vline(x=0, line_color="#CFCFCF", line_width=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Legend / caption ──────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8.5pt;color:#767676;">'
        "FIG — FILLED BARS PUSH THE PREDICTION TOWARD UP; "
        "OUTLINED BARS PULL IT TOWARD DOWN. BAR LENGTH IS PROPORTIONAL TO |SHAP VALUE|."
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Tabular readout ───────────────────────────────────────────────────────
    st.markdown('<h3>Top Contributing Features</h3>', unsafe_allow_html=True)
    df_display = pd.DataFrame(shap_contributions[:15])
    df_display.columns = ["Feature", "SHAP Value"]
    df_display["Direction"] = df_display["SHAP Value"].apply(
        lambda v: "▲ Pushes UP" if v >= 0 else "▼ Pulls DOWN"
    )
    df_display["SHAP Value"] = df_display["SHAP Value"].map(lambda v: f"{v:+.4f}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)
