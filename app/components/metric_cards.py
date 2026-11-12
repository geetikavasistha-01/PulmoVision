"""
PulmoVision — Reusable Scientific Metric Card Components
Formats key research indicators, confidence intervals, and cohort statistics with clinical precision.
"""

from typing import Optional, List, Dict
import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    ci: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "normal",  # "normal", "inverse", "off"
    caption: Optional[str] = None,
    border_accent: Optional[str] = None,
) -> None:
    """
    Renders an individual scientific metric card.
    """
    border_style = f"border-left: 4px solid {border_accent};" if border_accent else ""
    
    ci_html = f'<div style="font-size: 0.76rem; color: var(--pv-text-muted); font-family: var(--pv-font-mono); margin-top: 2px;">95% CI: {ci}</div>' if ci else ""
    delta_html = f'<div style="font-size: 0.78rem; font-weight: 600; color: var(--pv-accent-secondary); margin-top: 4px;">{delta}</div>' if delta else ""
    caption_html = f'<div style="font-size: 0.75rem; color: var(--pv-text-muted); margin-top: 4px;">{caption}</div>' if caption else ""

    card_html = f"""
    <div class="pv-card" style="{border_style} padding: 16px 18px; margin-bottom: 12px;">
        <div style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pv-text-secondary); margin-bottom: 6px;">
            {label}
        </div>
        <div style="font-size: 1.75rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-accent-primary); letter-spacing: -0.02em;">
            {value}
        </div>
        {ci_html}
        {delta_html}
        {caption_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_cohort_kpis() -> None:
    """
    Renders the four authoritative cohort volume metrics.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            label="Development Patients",
            value="122",
            caption="77 COVID-19 / 45 Normal",
            border_accent="var(--pv-accent-primary)"
        )

    with col2:
        render_metric_card(
            label="Acquisition Series",
            value="239",
            caption="1–3 CT series per patient",
            border_accent="var(--pv-accent-secondary)"
        )

    with col3:
        render_metric_card(
            label="Unique CT Slices",
            value="3,132",
            caption="604 duplicate entries resolved",
            border_accent="var(--pv-success)"
        )

    with col4:
        render_metric_card(
            label="Quarantined Test Set",
            value="255",
            caption="8,322 slices (100% sealed)",
            border_accent="var(--pv-error)"
        )


def render_leader_benchmark_cards() -> None:
    """
    Renders Experiment B (BiLSTM Mean) authoritative pooled out-of-fold metrics.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            label="Macro F1 Score",
            value="0.9461",
            ci="[0.8984, 0.9820]",
            delta="+0.1225 vs Atemporal",
            border_accent="var(--pv-accent-primary)"
        )

    with col2:
        render_metric_card(
            label="ROC-AUC (Pooled)",
            value="0.9688",
            delta="Fold Mean: 0.9804 ± 0.0358",
            border_accent="var(--pv-accent-secondary)"
        )

    with col3:
        render_metric_card(
            label="Patient Sensitivity",
            value="98.70%",
            ci="76 / 77 COVID detected",
            delta="1 False Negative",
            border_accent="var(--pv-success)"
        )

    with col4:
        render_metric_card(
            label="Patient Specificity",
            value="88.89%",
            ci="40 / 45 Normal detected",
            delta="5 False Positives",
            border_accent="var(--pv-info)"
        )

