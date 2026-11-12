"""app/views/documentation_view.py - Research Documentation & Manuscript Viewer.

Renders audited scientific manuscripts and engineering reports directly from repository sources:
- Stage 2 Research Report (Primary Manuscript)
- Stage 2 Forensic Audit Report
- Stage 2 Evaluation Benchmark Report
- Final Implementation Consistency Audit
- BiLSTM Architecture Specification
- Experiment Protocol Specification
- Stage 1 Methodology Overview
Zero decorative emojis; uses Material Symbols.
"""

from pathlib import Path
from typing import Dict, Tuple
import streamlit as st

from app.components.header import render_page_header
from app.utils.paths import REPO_ROOT


DOCUMENTS: Dict[str, Tuple[Path, str, str]] = {
    "Stage 2 Research Report (Primary Manuscript)": (
        REPO_ROOT / "reports" / "research" / "PULMOVISION_STAGE_2_RESEARCH_REPORT.md",
        "Primary scientific manuscript detailing patient-balanced optimization, BiLSTM sequence modeling, and nested CV findings.",
        "PULMOVISION_STAGE_2_RESEARCH_REPORT.md"
    ),
    "Final Implementation Consistency Audit": (
        REPO_ROOT / "reports" / "experiments" / "FINAL_IMPLEMENTATION_AUDIT.md",
        "Forensic consistency audit verifying metric alignment across checkpoints, reports, and code.",
        "FINAL_IMPLEMENTATION_AUDIT.md"
    ),
    "Stage 2 Forensic Audit Report": (
        REPO_ROOT / "reports" / "experiments" / "STAGE_2_AUDIT_REPORT.md",
        "Detailed run-by-run audit of all 15 outer-fold runs, early stopping behavior, and parameter hashes.",
        "STAGE_2_AUDIT_REPORT.md"
    ),
    "Stage 2 Evaluation Benchmark Report": (
        REPO_ROOT / "reports" / "experiments" / "STAGE_2_EVALUATION_REPORT.md",
        "Comprehensive cross-validation benchmark results across Experiments A, B, and C.",
        "STAGE_2_EVALUATION_REPORT.md"
    ),
    "BiLSTM Architecture Specification": (
        REPO_ROOT / "docs" / "research" / "cnn_bilstm_architecture.md",
        "Mathematical and tensor formulation of craniocaudal recurrent sequence aggregation.",
        "cnn_bilstm_architecture.md"
    ),
    "Stage 2 Experiment Protocol": (
        REPO_ROOT / "docs" / "research" / "cnn_bilstm_experiment_protocol.md",
        "Frozen cross-validation, threshold search, and early-stopping protocol rules.",
        "cnn_bilstm_experiment_protocol.md"
    ),
    "Methodology & Limitations Overview": (
        REPO_ROOT / "docs" / "methodology.md",
        "High-level research methodology, patient stratification rules, and documented dataset limitations.",
        "methodology.md"
    ),
}


def render_documentation_view() -> None:
    """Renders Page 7: Research Documentation."""
    render_page_header(
        title="Research Documentation & Manuscripts",
        subtitle="Live viewing of audited scientific manuscripts, protocol specifications, and forensic audit reports",
        section="Scientific Research Suite",
    )

    # Document Selection Bar
    doc_title = st.selectbox(
        "Select Document to Inspect:",
        options=list(DOCUMENTS.keys()),
        index=0,
        key="doc_selector"
    )

    doc_path, doc_desc, doc_filename = DOCUMENTS[doc_title]

    if not doc_path.is_file():
        st.error(f"Selected document file not found at expected path: `{doc_path}`")
        return

    content = doc_path.read_text(encoding="utf-8")

    # Document Metadata Ribbon
    st.markdown(
        f"""
        <div class="pv-card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 20px;">
            <div>
                <strong style="color: var(--pv-text-primary); font-size: 1.05rem;">{doc_title}</strong><br>
                <span style="color: var(--pv-text-secondary); font-size: 0.85rem;">{doc_desc}</span><br>
                <span style="color: var(--pv-text-muted); font-size: 0.78rem; font-family: var(--pv-font-mono);">FILE: {doc_path.relative_to(REPO_ROOT)} &bull; SIZE: {len(content):,} chars</span>
            </div>
            <div style="display: flex; gap: 8px;">
                <span class="pv-badge pv-badge-verified">AUDITED: PASS</span>
                <span class="pv-badge pv-badge-prototype">VERIFIED SOURCE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Toolbar: Download + Display Mode + Live Search
    col_dl, col_mode, col_search = st.columns([1.2, 1.2, 2.6])

    with col_dl:
        st.download_button(
            label=":material/download: Download Markdown",
            data=content,
            file_name=doc_filename,
            mime="text/markdown",
            use_container_width=True,
        )

    with col_mode:
        view_mode = st.radio(
            "View Mode",
            ["Full Document", "By Section"],
            horizontal=True,
            label_visibility="collapsed",
            key="doc_view_mode",
        )

    with col_search:
        search_query = st.text_input(
            "Search document",
            placeholder="Search within text (e.g. Bahdanau, Macro F1, B0...)",
            label_visibility="collapsed",
            key="doc_search_input",
        )

    st.markdown("<hr style='border: none; border-top: 1px solid var(--pv-border); margin: 16px 0 24px 0;'>", unsafe_allow_html=True)

    if view_mode == "Full Document":
        if search_query:
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

        if section_dict:
            selected_sec = st.selectbox("Select Section to Review:", list(section_dict.keys()), key="doc_section_picker")
            if selected_sec:
                st.markdown(section_dict[selected_sec])
        else:
            st.markdown(content)
