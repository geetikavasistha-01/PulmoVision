"""
PulmoVision — Dynamic CSS Theme Injector
Injects semantic CSS custom properties and component rules based on the active theme.
"""

from typing import Dict, Any
import streamlit as st
from app.styles.tokens import get_theme


def apply_theme(theme_name: str = "dark") -> None:
    """
    Injects custom CSS matching the chosen theme token dictionary.
    Ensures accessibility, consistent typography, and preserved dark CT canvas.
    """
    t = get_theme(theme_name)

    css = f"""
    <style>
        /* CSS Variables derived from tokens */
        :root {{
            --pv-bg-app: {t['bg_app']};
            --pv-bg-surface: {t['bg_surface']};
            --pv-bg-elevated: {t['bg_elevated']};
            --pv-bg-card: {t['bg_card']};
            --pv-border: {t['border']};
            --pv-border-subtle: {t['border_subtle']};
            --pv-text-primary: {t['text_primary']};
            --pv-text-secondary: {t['text_secondary']};
            --pv-text-muted: {t['text_muted']};
            --pv-accent-primary: {t['accent_primary']};
            --pv-accent-secondary: {t['accent_secondary']};
            --pv-success: {t['success']};
            --pv-warning: {t['warning']};
            --pv-error: {t['error']};
            --pv-info: {t['info']};
            --pv-workstation-bg: {t['workstation_bg']};
            --pv-font-sans: {t['font_sans']};
            --pv-font-mono: {t['font_mono']};
            --pv-shadow-card: {t['shadow_card']};
        }}

        /* Global App Background & Text */
        .stApp {{
            background-color: var(--pv-bg-app);
            color: var(--pv-text-primary);
            font-family: var(--pv-font-sans);
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--pv-bg-surface);
            border-right: 1px solid var(--pv-border);
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--pv-text-primary);
        }}

        /* Headers & Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--pv-text-primary) !important;
            font-family: var(--pv-font-sans);
            font-weight: 600;
            letter-spacing: -0.015em;
        }}

        p, span, label, li {{
            color: var(--pv-text-secondary);
            font-family: var(--pv-font-sans);
        }}

        strong, b {{
            color: var(--pv-text-primary);
        }}

        code, pre, .stCodeBlock {{
            font-family: var(--pv-font-mono) !important;
        }}

        /* Reusable Card Components */
        .pv-card {{
            background-color: var(--pv-bg-card);
            border: 1px solid var(--pv-border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: var(--pv-shadow-card);
            transition: border-color 0.15s ease-in-out;
        }}
        .pv-card:hover {{
            border-color: var(--pv-accent-primary);
        }}

        .pv-card-highlight {{
            background-color: var(--pv-bg-elevated);
            border: 1px solid var(--pv-accent-primary);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }}

        /* Research Status Badge */
        .pv-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-right: 8px;
        }}
        .pv-badge-prototype {{
            background-color: rgba(245, 158, 11, 0.15);
            color: var(--pv-warning);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        .pv-badge-verified {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--pv-success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .pv-badge-quarantine {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--pv-error);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}

        /* Research Prototype Banner */
        .pv-disclaimer-banner {{
            background-color: var(--pv-bg-elevated);
            border-left: 4px solid var(--pv-warning);
            border-radius: 6px;
            padding: 12px 18px;
            margin-bottom: 24px;
            font-size: 0.88rem;
            line-height: 1.45;
            color: var(--pv-text-secondary);
        }}
        .pv-disclaimer-banner strong {{
            color: var(--pv-warning);
        }}

        /* Clinical Workstation Canvas (Preserved Dark Environment) */
        .pv-workstation-panel {{
            background-color: var(--pv-workstation-bg);
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 18px;
            color: #f1f5f9;
        }}
        .pv-workstation-panel p, .pv-workstation-panel span {{
            color: #94a3b8;
        }}
        .pv-workstation-panel strong {{
            color: #ffffff;
        }}

        /* Metric Indicators */
        [data-testid="stMetricValue"] {{
            color: var(--pv-accent-primary) !important;
            font-family: var(--pv-font-sans);
            font-weight: 700;
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--pv-text-secondary) !important;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            border-bottom: 1px solid var(--pv-border);
            gap: 16px;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 10px 16px;
            font-size: 0.92rem;
            font-weight: 500;
            color: var(--pv-text-secondary);
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--pv-accent-primary) !important;
            border-bottom-color: var(--pv-accent-primary) !important;
        }}

        /* Button customization */
        .stButton>button {{
            border-radius: 6px;
            font-weight: 600;
            padding: 8px 18px;
            transition: all 0.15s ease-in-out;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

