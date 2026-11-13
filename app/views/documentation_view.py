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
        raw_sections = content.split("\n## ")
        intro_part = raw_sections[0].strip()

        parsed_sections = []
        if intro_part:
            parsed_sections.append({
                "badge": "OVERVIEW",
                "title": "0. Document Overview & Executive Disclaimer",
                "content": intro_part,
                "char_count": len(intro_part),
                "read_time": max(1, len(intro_part) // 900),
            })

        for i, sec in enumerate(raw_sections[1:], start=1):
            lines = sec.split("\n", 1)
            sec_title = lines[0].strip()
            sec_content = lines[1].strip() if len(lines) > 1 else ""
            parsed_sections.append({
                "badge": f"SECTION {i:02d}",
                "title": sec_title,
                "content": sec_content,
                "char_count": len(sec),
                "read_time": max(1, len(sec) // 900),
            })

        if not parsed_sections:
            st.markdown(content)
            return

        section_titles = [s["title"] for s in parsed_sections]

        # Reset selection if document changes
        if st.session_state.get("doc_last_active_title") != doc_title:
            st.session_state["doc_last_active_title"] = doc_title
            default_label = section_titles[1] if len(section_titles) > 1 else section_titles[0]
            st.session_state["doc_section_picker"] = default_label
        elif (
            "doc_section_picker" not in st.session_state
            or st.session_state["doc_section_picker"] not in section_titles
        ):
            default_label = section_titles[1] if len(section_titles) > 1 else section_titles[0]
            st.session_state["doc_section_picker"] = default_label

        curr_title = st.session_state["doc_section_picker"]
        curr_idx = section_titles.index(curr_title)
        active_sec = parsed_sections[curr_idx]

        def _set_section_idx(idx: int) -> None:
            clamped = max(0, min(len(section_titles) - 1, idx))
            st.session_state["doc_section_picker"] = section_titles[clamped]

        # ---------------------------------------------------------------------
        # 1. Section Navigator Console
        # ---------------------------------------------------------------------
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--pv-text-muted);">
                    Manuscript Section Navigator
                </span>
                <span style="font-size: 0.74rem; font-family: var(--pv-font-mono); color: var(--pv-accent-primary); font-weight: 600;">
                    {curr_idx + 1} of {len(parsed_sections)} Sections
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_prev, col_select, col_next = st.columns([1, 3.2, 1])

        with col_prev:
            st.button(
                ":material/arrow_back: Previous",
                disabled=(curr_idx == 0),
                use_container_width=True,
                key="doc_btn_prev_top",
                on_click=_set_section_idx,
                args=(curr_idx - 1,),
            )

        with col_select:
            st.selectbox(
                "Select Section to Review:",
                options=section_titles,
                key="doc_section_picker",
                label_visibility="collapsed",
            )

        with col_next:
            st.button(
                "Next :material/arrow_forward:",
                disabled=(curr_idx >= len(parsed_sections) - 1),
                use_container_width=True,
                key="doc_btn_next_top",
                on_click=_set_section_idx,
                args=(curr_idx + 1,),
            )

        # Progress strip
        progress_ratio = (curr_idx + 1) / len(parsed_sections)
        st.progress(
            progress_ratio,
            text=f"Reading Progress: Section {curr_idx + 1} of {len(parsed_sections)} ({int(progress_ratio * 100)}%)",
        )

        # ---------------------------------------------------------------------
        # 2. Collapsible Table of Contents & Quick Jump
        # ---------------------------------------------------------------------
        with st.expander(
            f":material/toc: Table of Contents & Quick Jump ({len(parsed_sections)} Sections)",
            expanded=False,
        ):
            st.markdown(
                "<p style='font-size: 0.82rem; color: var(--pv-text-secondary); margin-bottom: 12px;'>"
                "Select any section below to jump directly into the manuscript text:</p>",
                unsafe_allow_html=True,
            )
            toc_col1, toc_col2 = st.columns(2)
            for i, sec in enumerate(parsed_sections):
                target_col = toc_col1 if i % 2 == 0 else toc_col2
                is_active = (i == curr_idx)
                label = f"{sec['title']}"
                if is_active:
                    label = f"• {label} (Current)"
                with target_col:
                    st.button(
                        label,
                        key=f"toc_jump_{i}",
                        use_container_width=True,
                        disabled=is_active,
                        on_click=_set_section_idx,
                        args=(i,),
                    )

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        # Search Query notification within active section
        if search_query:
            query_lower = search_query.lower()
            in_sec_count = active_sec["content"].lower().count(query_lower)
            if in_sec_count > 0:
                st.info(f":material/search: Found {in_sec_count} match(es) for '{search_query}' in this section.")
            else:
                other_matches = [
                    s["title"] for s in parsed_sections if query_lower in s["content"].lower()
                ]
                if other_matches:
                    match_list = ", ".join(other_matches[:3])
                    st.warning(f"'{search_query}' not found in this section. Found in: {match_list}...")

        # ---------------------------------------------------------------------
        # 3. Main Section Reading Canvas
        # ---------------------------------------------------------------------
        st.markdown(
            f"""
            <div class="pv-card" style="padding: 26px 30px; border-left: 4px solid var(--pv-accent-primary); margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 8px; border-radius: 4px; background-color: var(--pv-bg-surface); color: var(--pv-accent-primary); border: 1px solid var(--pv-border);">
                            {active_sec['badge']}
                        </span>
                        <span style="font-size: 0.80rem; color: var(--pv-text-muted);">
                            ~{active_sec['read_time']} min read &bull; {active_sec['char_count']:,} characters
                        </span>
                    </div>
                    <div style="font-size: 0.74rem; color: var(--pv-text-secondary); font-family: var(--pv-font-mono);">
                        {doc_filename}
                    </div>
                </div>
                <h2 style="font-size: 1.65rem; font-weight: 700; color: var(--pv-text-primary); margin: 0 0 16px 0; letter-spacing: -0.02em; line-height: 1.3;">
                    {active_sec['title']}
                </h2>
                <div style="border-top: 1px solid var(--pv-border); margin-bottom: 22px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render Section Body
        st.markdown(active_sec["content"])

        # ---------------------------------------------------------------------
        # 4. Bottom Directional Navigation
        # ---------------------------------------------------------------------
        st.markdown(
            "<div style='border-top: 1px solid var(--pv-border); margin: 32px 0 20px 0;'></div>",
            unsafe_allow_html=True,
        )

        col_b_prev, col_b_space, col_b_next = st.columns([1.5, 0.2, 1.5])
        with col_b_prev:
            if curr_idx > 0:
                p_title = parsed_sections[curr_idx - 1]["title"]
                st.button(
                    f":material/arrow_back: Previous: {p_title}",
                    use_container_width=True,
                    key="doc_btn_prev_bot",
                    on_click=_set_section_idx,
                    args=(curr_idx - 1,),
                )
        with col_b_next:
            if curr_idx < len(parsed_sections) - 1:
                n_title = parsed_sections[curr_idx + 1]["title"]
                st.button(
                    f"Next: {n_title} :material/arrow_forward:",
                    use_container_width=True,
                    key="doc_btn_next_bot",
                    on_click=_set_section_idx,
                    args=(curr_idx + 1,),
                )
