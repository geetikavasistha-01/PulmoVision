"""app/views/documentation_view.py - Research Documentation & Manuscript Viewer.

Renders the complete audited Stage 2 Research Report with:
- Document metadata header (Branch, Commit, Audit Status, Date)
- Table of contents with section filtering
- Interactive manuscript viewer
- Raw Markdown download button
"""

from pathlib import Path
import streamlit as st

from app.components.header import render_page_header
from app.utils.paths import REPO_ROOT


def render_documentation_view():
    """Render Page 7: Research Documentation."""
    render_page_header(
        title="Research Documentation",
        subtitle="Complete, audited scientific manuscript for PulmoVision Stage 2 sequence modeling",
        section="Scientific Research Suite",
    )

    report_path = REPO_ROOT / "reports" / "research" / "PULMOVISION_STAGE_2_RESEARCH_REPORT.md"

    if not report_path.is_file():
        st.error(f"Research report file not found at expected path: `{report_path}`")
        return

    content = report_path.read_text(encoding="utf-8")

    # Document Metadata Ribbon
    st.markdown("""
    <div class="pv-card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 20px;">
        <div>
            <strong style="color: var(--text-primary); font-size: 1.05rem;">PulmoVision Stage 2 Research Manuscript</strong><br>
            <span style="color: var(--text-muted); font-size: 0.85rem;">Branch: <code>feature/cnn-bilstm</code> &bull; Commit: <code>a38bdd3</code> &bull; Verified: Nov 12, 2026</span>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="pv-badge pv-badge-verified">AUDITED: PASS</span>
            <span class="pv-badge pv-badge-prototype">STAGE 2 ARCHIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action Toolbar: Download + Display Mode
    col_dl, col_mode, col_search = st.columns([1, 1, 2])

    with col_dl:
        st.download_button(
            label="📥 Download Markdown",
            data=content,
            file_name="PULMOVISION_STAGE_2_RESEARCH_REPORT.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_mode:
        view_mode = st.radio(
            "View Mode",
            ["Full Manuscript", "By Section"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with col_search:
        search_query = st.text_input("🔍 Search within manuscript", placeholder="e.g., Bahdanau, Macro F1, B0...", label_visibility="collapsed")

    st.markdown("<hr style='border: none; border-top: 1px solid var(--border-color); margin: 16px 0 24px 0;'>", unsafe_allow_html=True)

    if view_mode == "Full Manuscript":
        if search_query:
            # Highlight search matches
            lines = content.splitlines()
            matching_lines = [l for l in lines if search_query.lower() in l.lower()]
            st.info(f"Found {len(matching_lines)} matching lines for '{search_query}'.")
        st.markdown(content)
    else:
        # Split content into major numbered sections (## X.)
        sections = content.split("\n## ")
        intro_part = sections[0]
        st.markdown(intro_part)

        section_dict = {}
        for sec in sections[1:]:
            first_line = sec.split("\n", 1)[0]
            sec_title = f"Section: {first_line}"
            sec_body = "## " + sec
            section_dict[sec_title] = sec_body

        selected_sec = st.selectbox("Select Section to Review:", list(section_dict.keys()))
        if selected_sec:
            st.markdown(section_dict[selected_sec])

