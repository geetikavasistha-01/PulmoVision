"""app/components/navigation.py - Sidebar Navigation and Platform Provenance.

Manages persistent theme selection, destination routing via Material Symbols icons,
and platform provenance metadata.
"""

from typing import Tuple
import streamlit as st


DESTINATIONS = [
    ("Research Overview", ":material/biotech:"),
    ("Dataset Explorer", ":material/dataset:"),
    ("Model Laboratory", ":material/model_training:"),
    ("Experiment Analytics", ":material/analytics:"),
    ("CT Workstation", ":material/medical_services:"),
    ("Explainability", ":material/visibility:"),
    ("Research Documentation", ":material/menu_book:"),
]


def render_sidebar() -> Tuple[str, str]:
    """Renders sidebar navigation, theme switcher, and model provenance.

    Returns:
        (selected_destination, active_theme)
    """
    # 1. Initialize session state defaults
    if "theme" not in st.session_state:
        query_theme = st.query_params.get("theme", "dark")
        st.session_state["theme"] = "light" if str(query_theme).lower() == "light" else "dark"

    query_page = st.query_params.get("page", None)
    if query_page and any(name == query_page for name, _ in DESTINATIONS):
        st.session_state["current_page"] = query_page
    elif "current_page" not in st.session_state:
        st.session_state["current_page"] = "Research Overview"

    with st.sidebar:
        # Header Brand Identity
        st.markdown(
            """
            <div style="margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--pv-border);">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--pv-accent-primary);">
                        <path d="M7 3C5 5 4 8 4 12c0 4 2 8 5 9 1-3 1-6 1-9 0-4-1-8-3-12Z"/>
                        <path d="M17 3c2 2 3 5 3 9 0 4-2 8-5 9-1-3-1-6-1-9 0-4 1-8 3-12Z"/>
                        <path d="M12 3v18"/>
                    </svg>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; color: var(--pv-text-primary); line-height: 1.2;">
                            PulmoVision
                        </div>
                        <div style="font-size: 0.70rem; color: var(--pv-text-muted); text-transform: uppercase; letter-spacing: 0.08em; font-family: var(--pv-font-mono);">
                            Research Workstation
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Integrated Theme Switcher
        st.markdown(
            "<p style='font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; color: var(--pv-text-muted);'>Appearance</p>",
            unsafe_allow_html=True
        )

        theme_options = ["Clinical Dark", "Laboratory Light"]
        curr_theme_idx = 0 if st.session_state["theme"] == "dark" else 1

        chosen_theme_label = st.segmented_control(
            "Appearance Mode",
            options=theme_options,
            default=theme_options[curr_theme_idx],
            format_func=lambda x: ":material/dark_mode: Dark" if "Dark" in x else ":material/light_mode: Light",
            label_visibility="collapsed",
            key="theme_segmented_control"
        )
        if chosen_theme_label:
            st.session_state["theme"] = "dark" if "Dark" in chosen_theme_label else "light"
        active_theme = st.session_state["theme"]

        st.markdown("<div style='margin: 14px 0 10px 0; border-top: 1px solid var(--pv-border);'></div>", unsafe_allow_html=True)

        # 3. Destination Navigation
        st.markdown(
            "<p style='font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; color: var(--pv-text-muted);'>Navigation</p>",
            unsafe_allow_html=True
        )

        nav_labels = [f"{icon} {name}" for name, icon in DESTINATIONS]
        
        # Match current index
        current_nav_label = next(
            (f"{icon} {name}" for name, icon in DESTINATIONS if name == st.session_state["current_page"]),
            nav_labels[0]
        )

        selected_nav = st.radio(
            "Destinations",
            options=nav_labels,
            index=nav_labels.index(current_nav_label),
            label_visibility="collapsed",
            key="main_nav_radio"
        )

        # Extract base destination name
        destination = next((name for name, _ in DESTINATIONS if name in selected_nav), "Research Overview")
        st.session_state["current_page"] = destination

        st.markdown("<div style='margin: 18px 0 12px 0; border-top: 1px solid var(--pv-border);'></div>", unsafe_allow_html=True)

        # 4. Provenance & Quarantine Metadata Box
        st.markdown(
            """
            <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; font-size: 0.76rem;">
                <div style="font-weight: 600; margin-bottom: 6px; color: var(--pv-text-primary); display: flex; align-items: center; justify-content: space-between;">
                    <span>Inference Checkpoint</span>
                    <span style="font-size: 0.68rem; padding: 2px 6px; background-color: var(--pv-bg-surface); border-radius: 4px; border: 1px solid var(--pv-border);">STAGE 2</span>
                </div>
                <div style="font-family: var(--pv-font-mono); color: var(--pv-accent-primary); margin-bottom: 4px; font-size: 0.78rem;">effnet_b0_bilstm_mean</div>
                <div style="color: var(--pv-text-secondary); margin-bottom: 8px;">1,460,418 trainable params</div>
                <div style="border-top: 1px solid var(--pv-border); padding-top: 6px; font-family: var(--pv-font-mono); font-size: 0.70rem; line-height: 1.5;">
                    <div>Development cohort: 122 patients</div>
                    <div style="color: var(--pv-warning);">Independent holdout cohort: 255 patients &mdash; SEALED, NOT EVALUATED</div>
                    <div style="margin-top: 4px;">SHA-256: <span title="Click to copy full SHA-256: 0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed" style="color: var(--pv-accent-secondary); cursor: pointer; text-decoration: underline dotted;" onclick="navigator.clipboard.writeText('0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed'); const orig = this.innerText; this.innerText = 'Copied!'; setTimeout(() => { this.innerText = orig; }, 1800);">0bd81731...</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="margin-top: 14px; font-size: 0.70rem; color: var(--pv-text-muted); line-height: 1.4;">
                PulmoVision Research Prototype.<br>Strictly for academic exploration.
            </div>
            """,
            unsafe_allow_html=True
        )

    return destination, active_theme
