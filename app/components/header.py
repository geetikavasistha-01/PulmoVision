"""app/components/header.py - Application Header Component.

Renders persistent project identity, subtitle, status badges, and prototype disclaimers.
Zero decorative emojis; uses crisp SVG iconography and Material Symbols.
"""

from typing import Optional
import streamlit as st
import torch


def render_header(title: Optional[str] = None, subtitle: Optional[str] = None) -> None:
    """Renders top-level platform identity with status badges."""
    device_label = (
        "Apple MPS (Metal)"
        if torch.backends.mps.is_available()
        else ("CUDA GPU" if torch.cuda.is_available() else "CPU")
    )

    st.markdown(
        f"""
        <div style="margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid var(--pv-border);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--pv-accent-primary);">
                            <path d="M7 3C5 5 4 8 4 12c0 4 2 8 5 9 1-3 1-6 1-9 0-4-1-8-3-12Z"/>
                            <path d="M17 3c2 2 3 5 3 9 0 4-2 8-5 9-1-3-1-6-1-9 0-4 1-8 3-12Z"/>
                            <path d="M12 3v18"/>
                        </svg>
                        <span style="font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; color: var(--pv-text-primary);">
                            PulmoVision
                        </span>
                        <span class="pv-badge pv-badge-prototype">Research Prototype</span>
                        <span class="pv-badge pv-badge-verified">Stage 2 Frozen</span>
                    </div>
                    <div style="font-size: 0.90rem; color: var(--pv-text-secondary); margin-left: 2px;">
                        Thoracic CT Sequence Intelligence & Research Platform
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 6px 12px; font-size: 0.80rem; font-family: var(--pv-font-mono);">
                        <span style="color: var(--pv-text-muted);">ACCELERATOR:</span> <strong style="color: var(--pv-accent-secondary);">{device_label}</strong>
                    </div>
                    <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 6px 12px; font-size: 0.80rem; font-family: var(--pv-font-mono);">
                        <span style="color: var(--pv-text-muted);">TEST COHORT:</span> <strong style="color: var(--pv-error);">SEALED (255 pts)</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if title:
        st.markdown(
            f"""
            <div style="margin-bottom: 16px;">
                <h2 style="margin: 0 0 4px 0; font-size: 1.40rem; font-weight: 600; color: var(--pv-text-primary);">{title}</h2>
                {f'<p style="margin: 0; font-size: 0.90rem; color: var(--pv-text-secondary);">{subtitle}</p>' if subtitle else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_header(title: str, subtitle: Optional[str] = None, section: Optional[str] = None) -> None:
    """Renders page header with title and subtitle."""
    render_header(title=title, subtitle=subtitle)


def render_disclaimer() -> None:
    """Renders persistent non-diagnostic research disclaimer banner."""
    st.markdown(
        """
        <div class="pv-disclaimer-banner">
            <div style="display: flex; align-items: flex-start; gap: 8px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--pv-warning); flex-shrink: 0; margin-top: 2px;">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
                <div>
                    <strong>EDUCATIONAL RESEARCH PROTOTYPE ONLY:</strong>
                    PulmoVision is an academic research instrument investigating deep sequence modeling across thoracic CT scans.
                    It is <strong>NOT</strong> a certified medical diagnostic device, clinical decision-support system, or medical diagnostic tool.
                    Dataset labels represent categorical archive annotations, not confirmed diagnoses.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
