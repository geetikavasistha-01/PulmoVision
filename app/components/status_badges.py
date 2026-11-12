"""
PulmoVision — Status Badge Component
Renders semantic clinical and research status badges.
"""

import streamlit as st


def render_badge(label: str, badge_type: str = "prototype") -> str:
    """Returns HTML for an individual semantic status badge."""
    css_class = f"pv-badge-{badge_type}"
    return f'<span class="pv-badge {css_class}">{label}</span>'


def render_quarantine_badge() -> str:
    """Returns HTML for quarantined status badge."""
    return render_badge("SEALED & QUARANTINED", "quarantine")


def render_system_status_bar(device_label: str = "Apple MPS (Metal)") -> None:
    """Renders persistent platform status chips."""
    badges_html = f"""
    <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
        <span class="pv-badge pv-badge-prototype">Research Prototype</span>
        <span class="pv-badge pv-badge-verified">Stage 2 CV Complete</span>
        <span class="pv-badge pv-badge-quarantine">Test Cohort Sealed</span>
        <span style="font-size: 0.76rem; font-family: var(--pv-font-mono); color: var(--pv-text-muted);">
            ACCELERATOR: <strong style="color: var(--pv-accent-secondary);">{device_label}</strong>
        </span>
    </div>
    """
    st.markdown(badges_html, unsafe_allow_html=True)

