"""
PulmoVision — Application Header Component
Renders persistent project identity, subtitle, status badges, and prototype disclaimers.
"""

from typing import Optional
import streamlit as st
import torch


def render_header(title: Optional[str] = None, subtitle: Optional[str] = None) -> None:
    """
    Renders top-level platform identity with status badges.
    """
    device_label = "Apple MPS (Metal)" if torch.backends.mps.is_available() else ("CUDA GPU" if torch.cuda.is_available() else "CPU")

    st.markdown(
        f"""
        <div style="margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid var(--pv-border);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                        <span style="font-size: 1.8rem;">🫁</span>
                        <span style="font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; color: var(--pv-text-primary);">
                            PulmoVision
                        </span>
                        <span class="pv-badge pv-badge-prototype">Research Prototype</span>
                        <span class="pv-badge pv-badge-verified">Stage 2 Frozen</span>
                    </div>
                    <div style="font-size: 0.92rem; color: var(--pv-text-secondary); margin-left: 2px;">
                        Thoracic CT Sequence Intelligence & Research Platform
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; font-family: var(--pv-font-mono);">
                        <span style="color: var(--pv-text-muted);">ACCEL:</span> <strong style="color: var(--pv-accent-secondary);">{device_label}</strong>
                    </div>
                    <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; font-family: var(--pv-font-mono);">
                        <span style="color: var(--pv-text-muted);">TEST COHORT:</span> <strong style="color: var(--pv-error);">SEALED (255 pts)</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if title:
        st.markdown(
            f"""
            <div style="margin-bottom: 16px;">
                <h2 style="margin: 0 0 4px 0; font-size: 1.45rem; font-weight: 600; color: var(--pv-text-primary);">{title}</h2>
                {f'<p style="margin: 0; font-size: 0.92rem; color: var(--pv-text-secondary);">{subtitle}</p>' if subtitle else ''}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_page_header(title: str, subtitle: Optional[str] = None, section: Optional[str] = None) -> None:
    """Renders page header with title and subtitle."""
    render_header(title=title, subtitle=subtitle)


def render_disclaimer() -> None:
    """
    Renders persistent non-diagnostic research disclaimer banner.
    """
    st.markdown(
        """
        <div class="pv-disclaimer-banner">
            <strong>⚠️ EDUCATIONAL RESEARCH PROTOTYPE ONLY:</strong>
            PulmoVision is an academic research instrument investigating deep sequence modeling across thoracic CT scans. 
            It is <strong>NOT</strong> a certified medical diagnostic device, clinical decision-support system, or medical diagnostic tool. 
            Dataset labels represent categorical archive annotations, not confirmed diagnoses.
        </div>
        """,
        unsafe_allow_html=True
    )

