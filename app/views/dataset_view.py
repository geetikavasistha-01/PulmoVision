"""app/views/dataset_view.py - Dataset Explorer View (Page 2).

Provides interactive data exploration over approved development cohort metadata (122 patients, 3,132 slices).
Utilizes ManifestAdapter for robust schema normalization, search, filtering, and provenance tracking.
Strict quarantine: Zero access to sealed held-out test partition.
Zero decorative emojis.
"""

from typing import Optional
import streamlit as st
import pandas as pd

from app.utils.data_loader import load_dev_manifest
from app.utils.manifest_adapter import ManifestAdapter, REQUIRED_COLUMNS
from app.components.charts import (
    render_class_distribution_chart,
    render_sequence_length_histogram,
)
from app.components.header import render_page_header


def render_dataset_view() -> None:
    """Renders the comprehensive dataset exploration page."""
    render_page_header(
        title="COVID-CTset Development Cohort Explorer",
        subtitle="Exploring cohort demographics, anatomical deduplication provenance, and 16-bit radiodensity preprocessing",
        section="Scientific Research Suite",
    )

    df_manifest = load_dev_manifest()

    if df_manifest.empty:
        st.error(
            "Development manifest could not be loaded. Please ensure "
            "`data/metadata/splits/stage2_dev_deduplicated_manifest.csv` or `.json` exists."
        )
        return

    # Validate schema
    is_valid, missing_cols = ManifestAdapter.validate_manifest(df_manifest)
    if not is_valid:
        st.error(
            f"Manifest schema validation failure: Missing required columns {missing_cols}. "
            "Please check data integrity."
        )
        return

    # Compute verified cohort statistics via ManifestAdapter
    summary = ManifestAdapter.get_manifest_summary(df_manifest)

    # 1. High-Level Cohort Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="pv-card" style="padding: 14px 16px;">
                <div style="font-size: 0.74rem; text-transform: uppercase; color: var(--pv-text-secondary); font-weight: 600;">Development Cohort</div>
                <div style="font-size: 1.55rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-accent-primary);">{summary['unique_patients']} Patients</div>
                <div style="font-size: 0.74rem; color: var(--pv-text-muted);">{summary['covid_patients']} COVID-19 / {summary['normal_patients']} Normal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="pv-card" style="padding: 14px 16px;">
                <div style="font-size: 0.74rem; text-transform: uppercase; color: var(--pv-text-secondary); font-weight: 600;">Acquisition Series</div>
                <div style="font-size: 1.55rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-accent-secondary);">{summary['unique_series']} Series</div>
                <div style="font-size: 0.74rem; color: var(--pv-text-muted);">1–3 series per patient</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="pv-card" style="padding: 14px 16px;">
                <div style="font-size: 0.74rem; text-transform: uppercase; color: var(--pv-text-secondary); font-weight: 600;">Canonical Slices</div>
                <div style="font-size: 1.55rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-success);">{summary['total_slices']:,} Slices</div>
                <div style="font-size: 0.74rem; color: var(--pv-text-muted);">{summary['total_duplicates_resolved']} duplicate aliases resolved</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="pv-card" style="padding: 14px 16px;">
                <div style="font-size: 0.74rem; text-transform: uppercase; color: var(--pv-text-secondary); font-weight: 600;">Test Set Quarantine</div>
                <div style="font-size: 1.55rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-error);">255 Patients</div>
                <div style="font-size: 0.74rem; color: var(--pv-text-muted);">8,322 slices (100% sealed partition)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Interactive Charts Row
    st.markdown("### Cohort Characteristics & Sequence Lengths")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(
            "<p style='font-size: 0.88rem; font-weight: 600; color: var(--pv-text-primary); margin-bottom: 6px;'>Patient Class Distribution</p>",
            unsafe_allow_html=True,
        )
        render_class_distribution_chart(df_manifest)

    with col_chart2:
        st.markdown(
            "<p style='font-size: 0.88rem; font-weight: 600; color: var(--pv-text-primary); margin-bottom: 6px;'>Slices per Series (Sequence Length T)</p>",
            unsafe_allow_html=True,
        )
        render_sequence_length_histogram(df_manifest)

    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # 3. Anatomical De-Duplication Audit Panel
    st.markdown("### Forensic Anatomical De-Duplication Protocol")
    st.markdown(
        """
        <div class="pv-card">
            <div style="font-weight: 600; font-size: 0.95rem; color: var(--pv-text-primary); margin-bottom: 8px;">
                Resolution of 604 Redundant Archive Entries
            </div>
            <p style="font-size: 0.88rem; color: var(--pv-text-secondary); line-height: 1.5; margin-bottom: 12px;">
                During forensic dataset validation, the raw development cohort manifest contained 3,736 rows. 
                Grouping records by the composite anatomical key <code>(patient_id, series_id, slice_idx)</code> revealed 
                604 redundant duplicate slice instances. A byte-level cryptographic SHA256 audit confirmed that all duplicate 
                TIFF files were bitwise identical copies. Rather than allowing duplicate slices to artificially inflate 
                sequence lengths, deterministic canonical filenames were selected, and all 604 redundant aliases were 
                preserved in a provenance audit manifest.
            </p>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-family: var(--pv-font-mono); font-size: 0.80rem; background-color: var(--pv-bg-elevated); padding: 12px; border-radius: 6px;">
                <div><span style="color: var(--pv-text-muted);">ORIGINAL ROWS:</span> <strong>3,736</strong></div>
                <div><span style="color: var(--pv-text-muted);">RESOLVED ALIASES:</span> <strong style="color: var(--pv-warning);">604</strong></div>
                <div><span style="color: var(--pv-text-muted);">CANONICAL UNIQUE:</span> <strong style="color: var(--pv-success);">3,132</strong></div>
                <div><span style="color: var(--pv-text-muted);">HASH CONSISTENCY:</span> <strong>100% SHA256</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. CT Preprocessing Pipeline
    st.markdown("### 16-Bit Radiodensity Preprocessing Pipeline")
    st.markdown(
        """
        <div class="pv-card">
            <p style="font-size: 0.88rem; color: var(--pv-text-secondary); line-height: 1.5; margin-bottom: 12px;">
                Raw CT scans in the COVID-CTset distribution are stored as 16-bit uncompressed TIFF images with intensity values 
                ranging up to &approx; 2,204. Because accompanying DICOM header metadata (rescale slope and rescale intercept) 
                were omitted from the public release, these values represent <strong>uncalibrated radiodensity counts</strong>, 
                not standardized Hounsfield Units (HU).
            </p>
            <div style="background-color: var(--pv-bg-app); border: 1px solid var(--pv-border); border-radius: 6px; padding: 14px; font-family: var(--pv-font-mono); font-size: 0.80rem; line-height: 1.6;">
                <div>1. <strong>Intensity Windowing:</strong> Clip pixel radiodensity to [200, 1800] counts to isolate pulmonary parenchyma.</div>
                <div>2. <strong>Dynamic Range Scaling:</strong> Linear normalization: I_norm = (I_clip - 200.0) / 1600.0 &rarr; [0.0, 1.0].</div>
                <div>3. <strong>Spatial Standardisation:</strong> Bilinear interpolation resizing from original 512&times;512 to 224&times;224 pixels.</div>
                <div>4. <strong>Channel Replication:</strong> Replicate single-channel grayscale across 3 channels (R = G = B).</div>
                <div>5. <strong>ImageNet Normalisation:</strong> Standardize channels using official ImageNet parameters (&mu; = [0.485, 0.456, 0.406], &sigma; = [0.229, 0.224, 0.225]).</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Searchable Manifest Explorer (Approved Development Data Only)
    st.markdown("### Approved Development Manifest Explorer")
    st.caption("Inspect, filter, and search metadata for all 3,132 deduplicated development slices. Zero test data exposed.")

    col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
    with col_f1:
        label_filter = st.selectbox(
            "Filter by Class Label",
            options=["All Classes", "COVID-19 (Label 1)", "Normal (Label 0)"],
            key="dataset_label_filter",
        )
    with col_f2:
        search_query = st.text_input(
            "Search by Patient ID, Series ID, or Filename",
            placeholder="e.g. patient100, SR_2, IM00010",
            key="dataset_search_input",
        )
    with col_f3:
        show_aliases = st.checkbox("Show Duplicate Aliases", value=False, key="dataset_show_aliases")

    # Safe filtering via ManifestAdapter
    filtered_df = ManifestAdapter.filter_manifest(
        df_manifest,
        label_filter=label_filter,
        search_query=search_query,
    )

    # Define columns to display safely
    display_cols = ["patient_id", "series_id", "slice_idx", "class_name", "canonical_filename"]
    if show_aliases:
        if "duplicate_count" in filtered_df.columns:
            display_cols.append("duplicate_count")
        if "duplicate_aliases" in filtered_df.columns:
            display_cols.append("duplicate_aliases")

    st.dataframe(
        filtered_df[display_cols],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Displaying **{len(filtered_df):,}** matching slices out of {len(df_manifest):,} total canonical development slices."
    )
