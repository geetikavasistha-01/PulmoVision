"""
PulmoVision — Sidebar Navigation Component
Manages persistent theme selection, destination routing, and platform provenance.
"""

from typing import Tuple
import streamlit as st


DESTINATIONS = [
    ("Research Overview", "🔬"),
    ("Dataset Explorer", "📂"),
    ("Model Laboratory", "🧠"),
    ("Experiment Analytics", "📊"),
    ("CT Workstation", "🩻"),
    ("Explainability", "💡"),
    ("Research Documentation", "📖"),
]


def render_sidebar() -> Tuple[str, str]:
    """
    Renders sidebar navigation, theme switcher, and model provenance.
    Returns:
        (selected_destination, active_theme)
    """
    # 1. Initialize session state defaults
    if "theme" not in st.session_state:
        # Support optional query parameter: ?theme=light or ?theme=dark
        query_theme = st.query_params.get("theme", "dark")
        st.session_state["theme"] = "light" if str(query_theme).lower() == "light" else "dark"

    query_page = st.query_params.get("page", None)
    if query_page and any(name == query_page for name, _ in DESTINATIONS):
        st.session_state["current_page"] = query_page
    elif "current_page" not in st.session_state:
        st.session_state["current_page"] = "Research Overview"

    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 14px;">
                <div style="font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em;">
                    🫁 PulmoVision
                </div>
                <div style="font-size: 0.78rem; color: var(--pv-text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">
                    Research Workstation v2.0
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Theme Toggle Switcher
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; color: var(--pv-text-muted);'>Interface Theme</p>", unsafe_allow_html=True)
        theme_options = ["Clinical Dark", "Laboratory Light"]
        theme_index = 0 if st.session_state["theme"] == "dark" else 1
        chosen_theme_label = st.radio(
            "Select Theme",
            options=theme_options,
            index=theme_index,
            label_visibility="collapsed",
            horizontal=True,
            key="theme_radio_selector"
        )
        active_theme = "dark" if chosen_theme_label == "Clinical Dark" else "light"
        st.session_state["theme"] = active_theme

        st.markdown("<div style='margin: 14px 0 10px 0; border-top: 1px solid var(--pv-border);'></div>", unsafe_allow_html=True)

        # 3. Destination Navigation
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: var(--pv-text-muted);'>Destinations</p>", unsafe_allow_html=True)
        nav_labels = [f"{icon}  {name}" for name, icon in DESTINATIONS]
        
        # Match current index
        current_label = next(
            (f"{icon}  {name}" for name, icon in DESTINATIONS if name == st.session_state["current_page"]),
            nav_labels[0]
        )
        selected_nav = st.radio(
            "Navigate",
            options=nav_labels,
            index=nav_labels.index(current_label),
            label_visibility="collapsed",
            key="main_nav_radio"
        )
        # Extract base destination name
        destination = selected_nav.split("  ", 1)[1]
        st.session_state["current_page"] = destination

        st.markdown("<div style='margin: 18px 0 12px 0; border-top: 1px solid var(--pv-border);'></div>", unsafe_allow_html=True)

        # 4. Provenance & Quarantine Metadata Box
        st.markdown(
            """
            <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; font-size: 0.78rem;">
                <div style="font-weight: 600; margin-bottom: 6px; color: var(--pv-text-primary);">Deployment Checkpoint</div>
                <div style="font-family: var(--pv-font-mono); color: var(--pv-accent-primary); margin-bottom: 4px;">effnet_b0_bilstm_mean</div>
                <div style="color: var(--pv-text-secondary); margin-bottom: 8px;">1,460,418 trainable params</div>
                <div style="border-top: 1px solid var(--pv-border); padding-top: 6px; font-family: var(--pv-font-mono); font-size: 0.72rem;">
                    <div>COHORT: 122 DEV PTS</div>
                    <div>TEST: 255 PTS (QUARANTINED)</div>
                    <div>CHECKSUM: <span style="color: var(--pv-accent-secondary);">0bd81731...</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="margin-top: 16px; font-size: 0.72rem; color: var(--pv-text-muted); line-height: 1.4;">
                PulmoVision Research Prototype.<br>Strictly for investigational study.
            </div>
            """,
            unsafe_allow_html=True
        )

    return destination, active_theme
