"""app/components/navigation.py - Sidebar Navigation and Platform Provenance.

Manages persistent theme selection, destination routing via Material Symbols icons,
and platform provenance metadata.
"""

from typing import Tuple
import textwrap
import streamlit as st
import streamlit.components.v1 as components


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

        # Pre-synchronize radio state before widget instantiation
        if st.session_state.get("main_nav_radio") != current_nav_label:
            st.session_state["main_nav_radio"] = current_nav_label

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
        st.query_params["page"] = destination

        st.markdown("<div style='margin: 18px 0 12px 0; border-top: 1px solid var(--pv-border);'></div>", unsafe_allow_html=True)

        # 4. Provenance & Quarantine Metadata Box
        st.markdown(
            textwrap.dedent(
                """\
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; font-family: var(--pv-font-sans); font-size: 0.74rem;">
                  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 600; font-size: 0.80rem; color: var(--pv-text-primary); letter-spacing: -0.01em;">Inference Checkpoint</span>
                    <span style="font-size: 0.62rem; font-weight: 700; padding: 2px 6px; background-color: var(--pv-bg-surface); border-radius: 4px; border: 1px solid var(--pv-border); color: var(--pv-text-secondary); letter-spacing: 0.04em;">STAGE 2</span>
                  </div>
                  <div style="margin-bottom: 8px;">
                    <div style="font-family: var(--pv-font-mono); font-size: 0.74rem; font-weight: 600; color: var(--pv-accent-primary); letter-spacing: -0.01em; margin-bottom: 2px;">effnet_b0_bilstm_mean</div>
                    <div style="font-size: 0.68rem; color: var(--pv-text-muted);">1,460,418 trainable parameters</div>
                  </div>
                  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; font-size: 0.70rem; padding-bottom: 6px; border-bottom: 1px solid var(--pv-border-subtle);">
                    <span style="color: var(--pv-text-secondary);">Development cohort</span>
                    <span style="font-family: var(--pv-font-mono); font-weight: 600; color: var(--pv-text-primary);">122 patients</span>
                  </div>
                  <div style="background-color: var(--pv-bg-surface); border: 1px solid var(--pv-border); border-radius: 4px; padding: 8px 10px; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                      <span style="font-size: 0.70rem; font-weight: 600; color: var(--pv-text-primary);">Independent holdout dataset</span>
                      <span style="font-size: 0.60rem; font-weight: 700; letter-spacing: 0.05em; padding: 1px 5px; border-radius: 3px; background-color: rgba(245, 158, 11, 0.12); color: var(--pv-warning); border: 1px solid rgba(245, 158, 11, 0.35);">SEALED</span>
                    </div>
                    <div style="font-size: 0.74rem; font-family: var(--pv-font-mono); font-weight: 600; color: var(--pv-text-primary); margin-bottom: 2px;">
                      255 patients
                    </div>
                    <div style="font-size: 0.67rem; color: var(--pv-text-muted); line-height: 1.35;">
                      Not evaluated. Reserved for independent testing.
                    </div>
                  </div>
                  <div style="border-top: 1px solid var(--pv-border); padding-top: 8px;">
                    <div style="font-size: 0.64rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-muted); margin-bottom: 4px;">
                      CHECKPOINT SHA-256
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between; background-color: var(--pv-bg-surface); border: 1px solid var(--pv-border); border-radius: 4px; padding: 5px 8px; gap: 6px;">
                      <span style="font-family: var(--pv-font-mono); font-size: 0.68rem; color: var(--pv-text-secondary); letter-spacing: -0.01em; user-select: all;" title="Full SHA-256: 0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed">
                        0bd81731...ff1caed
                      </span>
                      <button id="pv-copy-hash-btn" type="button" aria-label="Copy SHA-256 hash" title="Copy full SHA-256 to clipboard" data-clipboard="0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed"
                        style="background: transparent; border: none; cursor: pointer; padding: 2px 4px; display: inline-flex; align-items: center; gap: 4px; color: var(--pv-text-muted); border-radius: 3px; font-family: var(--pv-font-sans); font-size: 0.65rem; transition: color 0.15s, background-color 0.15s; outline: none; margin: 0; min-height: unset; height: auto;"
                        onmouseover="this.style.color='var(--pv-text-primary)'; this.style.backgroundColor='var(--pv-bg-elevated)';"
                        onmouseout="this.style.color='var(--pv-text-muted)'; this.style.backgroundColor='transparent';">
                        <svg id="pv-copy-svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;">
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <svg id="pv-check-svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--pv-success)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="display: none; flex-shrink: 0;">
                          <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        <span id="pv-copy-status" style="font-weight: 500; font-size: 0.64rem;">Copy</span>
                      </button>
                    </div>
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # Non-rendering clipboard bridge
        components.html(
            """
            <script>
            (function() {
                const parentDoc = window.parent.document;
                function attachClipboard() {
                    const btn = parentDoc.getElementById('pv-copy-hash-btn');
                    if (!btn || btn.dataset.bound === 'true') return;
                    btn.dataset.bound = 'true';
                    btn.addEventListener('click', function(e) {
                        e.preventDefault();
                        const hash = btn.getAttribute('data-clipboard') || '0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed';
                        if (window.parent.navigator && window.parent.navigator.clipboard) {
                            window.parent.navigator.clipboard.writeText(hash).catch(function() {
                                const ta = parentDoc.createElement('textarea');
                                ta.value = hash;
                                parentDoc.body.appendChild(ta);
                                ta.select();
                                parentDoc.execCommand('copy');
                                parentDoc.body.removeChild(ta);
                            });
                        }
                        const copyIcon = parentDoc.getElementById('pv-copy-svg');
                        const checkIcon = parentDoc.getElementById('pv-check-svg');
                        const statusText = parentDoc.getElementById('pv-copy-status');
                        if (copyIcon && checkIcon) {
                            copyIcon.style.display = 'none';
                            checkIcon.style.display = 'inline-block';
                            if (statusText) {
                                statusText.innerText = 'Copied!';
                                statusText.style.color = 'var(--pv-success)';
                            }
                            setTimeout(function() {
                                copyIcon.style.display = 'inline-block';
                                checkIcon.style.display = 'none';
                                if (statusText) {
                                    statusText.innerText = 'Copy';
                                    statusText.style.color = 'var(--pv-text-muted)';
                                }
                            }, 2000);
                        }
                    });
                }
                attachClipboard();
                setInterval(attachClipboard, 800);
            })();
            </script>
            """,
            height=0,
            width=0,
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
