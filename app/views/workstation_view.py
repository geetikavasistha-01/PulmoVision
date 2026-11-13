"""app/views/workstation_view.py - Clinical CT Imaging Workstation View (Page 5).

Provides a 3-column workstation layout:
- Left: CT sequence upload, format validator, and pre-loaded sample series picker
- Center: Pitch-black viewport (#000000) with isotropic aspect ratio preservation and axial slider
- Right: Real-time inference using the verified deployment BiLSTM checkpoint on raw 16-bit TIFFs
Zero decorative emojis; uses Material Symbols.
"""

from typing import List, Dict, Any, Optional
import os
import glob
import streamlit as st
import numpy as np

from app.utils.paths import RAW_DEV_DATA_DIR, DEMO_SAMPLES_DIR
from app.utils.data_loader import get_deployment_predictor
from app.utils.validation import validate_ct_uploads
from app.components.ct_viewer import render_ct_viewport
from app.components.header import render_page_header
from src.inference.sequence_validator import CTSequenceValidator


def render_workstation_view() -> None:
    """Renders the professional 3-column clinical CT workstation."""
    render_page_header(
        title="Clinical CT Imaging Workstation",
        subtitle="Volumetric axial slice inspection, patient/series validation, and research inference using the verified deployment BiLSTM model",
        section="Clinical Workstation Suite",
    )

    # Initialize predictor
    predictor = get_deployment_predictor()

    # Session state initialization for workstation files and inference results
    if "ws_slices" not in st.session_state:
        st.session_state["ws_slices"] = []
    if "ws_slice_indices" not in st.session_state:
        st.session_state["ws_slice_indices"] = []
    if "ws_patient_id" not in st.session_state:
        st.session_state["ws_patient_id"] = "None"
    if "ws_series_id" not in st.session_state:
        st.session_state["ws_series_id"] = "None"
    if "ws_inference_res" not in st.session_state:
        st.session_state["ws_inference_res"] = None
    if "ws_warning" not in st.session_state:
        st.session_state["ws_warning"] = None

    # Load default sample scan if workspace is empty
    if not st.session_state["ws_slices"]:
        sample_pattern = str(RAW_DEV_DATA_DIR / "*patient100_SR_2*.tif")
        sample_files = sorted(glob.glob(sample_pattern))[:5]
        if not sample_files:
            demo_pattern = str(DEMO_SAMPLES_DIR / "*patient100_SR_2*.tif")
            sample_files = sorted(glob.glob(demo_pattern))[:5]
        if sample_files:
            try:
                ordered_inputs, meta = CTSequenceValidator.validate_and_order_sequence(sample_files)
                st.session_state["ws_patient_id"] = meta["patient_id"]
                st.session_state["ws_series_id"] = meta["series_id"]
                st.session_state["ws_slice_indices"] = meta["slice_indices"]
                st.session_state["ws_warning"] = meta["warning"]
                st.session_state["ws_slices"] = [predictor.load_image_as_array(f) for f in ordered_inputs]
                st.session_state["ws_raw_inputs"] = ordered_inputs
            except Exception:
                pass

    # Three-Column Layout
    col_input, col_viewer, col_analysis = st.columns([1, 1.8, 1.2])

    # -------------------------------------------------------------
    # LEFT PANEL: PATIENT & SERIES INPUT
    # -------------------------------------------------------------
    with col_input:
        st.markdown(
            """
            <div style="font-weight: 600; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pv-text-secondary); margin-bottom: 8px;">
                1. Input CT Sequence
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Upload 16-bit TIFF cross-sections belonging to a single scan series:")
        uploaded_files = st.file_uploader(
            "Upload CT Slices",
            type=["tif", "tiff", "png", "jpg", "jpeg", "npy"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            is_valid, err_msg, meta = validate_ct_uploads(uploaded_files)
            if is_valid and meta:
                st.success(f"Verified: {meta['patient_id']} &bull; {meta['series_id']} ({meta['num_slices']} slices)")
                if st.button("Load Sequence into Viewport", use_container_width=True, type="secondary"):
                    ordered_inputs, _ = CTSequenceValidator.validate_and_order_sequence(
                        uploaded_files, filenames=[f.name for f in uploaded_files]
                    )
                    st.session_state["ws_patient_id"] = meta["patient_id"]
                    st.session_state["ws_series_id"] = meta["series_id"]
                    st.session_state["ws_slice_indices"] = meta["slice_indices"]
                    st.session_state["ws_warning"] = meta["warning"]
                    st.session_state["ws_slices"] = [predictor.load_image_as_array(f) for f in ordered_inputs]
                    st.session_state["ws_raw_inputs"] = ordered_inputs
                    st.session_state["ws_inference_res"] = None
                    st.rerun()
            else:
                st.error(f"Validation Error: {err_msg}")

        st.divider()

        # One-Click Real Development Sample Buttons
        st.markdown(
            "<p style='font-size: 0.82rem; font-weight: 600; color: var(--pv-text-primary); margin-bottom: 6px;'>Pre-Loaded Development Scans:</p>",
            unsafe_allow_html=True,
        )

        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("COVID Scan (pt100)", use_container_width=True):
                files = sorted(glob.glob(str(RAW_DEV_DATA_DIR / "*patient100_SR_2*.tif")))[:5]
                if files:
                    ordered_inputs, meta = CTSequenceValidator.validate_and_order_sequence(files)
                    st.session_state["ws_patient_id"] = meta["patient_id"]
                    st.session_state["ws_series_id"] = meta["series_id"]
                    st.session_state["ws_slice_indices"] = meta["slice_indices"]
                    st.session_state["ws_warning"] = meta["warning"]
                    st.session_state["ws_slices"] = [predictor.load_image_as_array(f) for f in ordered_inputs]
                    st.session_state["ws_raw_inputs"] = ordered_inputs
                    st.session_state["ws_inference_res"] = None
                    st.rerun()

        with c_btn2:
            if st.button("Normal Scan (pt169)", use_container_width=True):
                files = sorted(glob.glob(str(RAW_DEV_DATA_DIR / "*patient169_SR_2*.tif")))[:5]
                if files:
                    ordered_inputs, meta = CTSequenceValidator.validate_and_order_sequence(files)
                    st.session_state["ws_patient_id"] = meta["patient_id"]
                    st.session_state["ws_series_id"] = meta["series_id"]
                    st.session_state["ws_slice_indices"] = meta["slice_indices"]
                    st.session_state["ws_warning"] = meta["warning"]
                    st.session_state["ws_slices"] = [predictor.load_image_as_array(f) for f in ordered_inputs]
                    st.session_state["ws_raw_inputs"] = ordered_inputs
                    st.session_state["ws_inference_res"] = None
                    st.rerun()

        # Current Input Summary Box
        st.markdown(
            f"""
            <div class="pv-card" style="margin-top: 14px; font-size: 0.78rem; font-family: var(--pv-font-mono);">
                <div style="color: var(--pv-text-muted); margin-bottom: 4px;">ACTIVE SEQUENCE:</div>
                <div style="color: var(--pv-text-primary);">PATIENT: {st.session_state['ws_patient_id']}</div>
                <div style="color: var(--pv-text-primary);">SERIES: {st.session_state['ws_series_id']}</div>
                <div style="color: var(--pv-accent-primary);">TOTAL SLICES: {len(st.session_state['ws_slices'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state["ws_warning"]:
            st.warning(f"{st.session_state['ws_warning']}")

    # -------------------------------------------------------------
    # CENTER PANEL: CT VIEWPORT (DARK CANVAS)
    # -------------------------------------------------------------
    with col_viewer:
        st.markdown(
            """
            <div style="font-weight: 600; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pv-text-secondary); margin-bottom: 8px;">
                2. Axial CT Cross-Section Viewport
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_ct_viewport(
            slices=st.session_state["ws_slices"],
            slice_indices=st.session_state["ws_slice_indices"],
            patient_id=st.session_state["ws_patient_id"],
            series_id=st.session_state["ws_series_id"],
            key_prefix="ws_viewport",
        )

    # -------------------------------------------------------------
    # RIGHT PANEL: AI ANALYSIS & PREDICTION PRESENTATION
    # -------------------------------------------------------------
    with col_analysis:
        st.markdown(
            """
            <div style="font-weight: 600; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pv-text-secondary); margin-bottom: 8px;">
                3. AI Sequence Inference
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Inference Action Button
        if len(st.session_state["ws_slices"]) > 0:
            if st.button(":material/play_arrow: Run Sequence Inference", type="primary", use_container_width=True):
                with st.spinner("Extracting slice features & evaluating BiLSTM..."):
                    raw_inputs = st.session_state.get("ws_raw_inputs", st.session_state["ws_slices"])
                    res = predictor.predict_sequence(raw_inputs)
                    st.session_state["ws_inference_res"] = res

        # Render Inference Results
        res = st.session_state["ws_inference_res"]
        if res:
            is_covid = res["predicted_index"] == 1
            badge_color = "#00b4d8" if is_covid else "#10b981"
            prob_val = res["covid_19_probability"] if is_covid else res["normal_probability"]

            st.markdown(
                f"""
                <div class="pv-card" style="border-left: 4px solid {badge_color}; margin-top: 10px;">
                    <div style="font-size: 0.74rem; text-transform: uppercase; color: var(--pv-text-muted); font-weight: 600;">Model Classification</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {badge_color}; margin: 2px 0;">
                        {res['predicted_class']}
                    </div>
                    <div style="font-size: 0.82rem; font-family: var(--pv-font-mono); color: var(--pv-text-secondary);">
                        Predicted Score P({res['predicted_class']}): <strong>{prob_val * 100:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Technical Probability Metrics
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("P(COVID-19)", f"{res['covid_19_probability']:.2%}")
            with col_p2:
                st.metric("P(Normal)", f"{res['normal_probability']:.2%}")

            st.markdown(
                f"""
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 10px; font-size: 0.76rem; font-family: var(--pv-font-mono); margin-top: 10px;">
                    <div>OPERATING THRESHOLD: <strong>&tau; = {res['threshold_applied']:.2f}</strong></div>
                    <div>SEQUENCE LENGTH: <strong>{res['num_slices']} slices</strong></div>
                    <div>TOTAL LATENCY: <strong>{res['inference_latency_ms']:.1f} ms</strong></div>
                    <div>PER-SLICE LATENCY: <strong>{res['per_slice_latency_ms']:.1f} ms/slice</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Click 'Run Sequence Inference' to evaluate the loaded sequence with the BiLSTM deployment checkpoint.")

        # Provenance & Non-Diagnostic Advisory Box
        st.markdown(
            """
            <div style="margin-top: 16px; font-size: 0.75rem; color: var(--pv-text-muted); line-height: 1.45; border-top: 1px solid var(--pv-border); padding-top: 10px;">
                <strong>Research Prototype Advisory:</strong> Predictions represent experimental deep learning scores 
                from an uncalibrated softmax objective. This tool is strictly investigational and must NOT be used for clinical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )
