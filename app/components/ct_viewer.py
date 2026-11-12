"""app/components/ct_viewer.py - Clinical CT Viewport Component.

Renders axial CT slice cross-sections on a dedicated pitch-black canvas in both Dark and Light themes.
Preserves isotropic 1:1 aspect ratio, radiodensity windowing display, and interactive slice navigation.
Zero decorative emojis; uses Material Symbols for navigation controls.
"""

from typing import List, Any, Optional
import streamlit as st
import numpy as np


def render_ct_viewport(
    slices: List[np.ndarray],
    slice_indices: List[int],
    patient_id: str = "Unknown",
    series_id: str = "Unknown",
    key_prefix: str = "ct_view"
) -> int:
    """Renders an interactive medical imaging viewport.

    Returns:
        The currently selected slice index (0 to len(slices)-1).
    """
    num_slices = len(slices)
    if num_slices == 0:
        st.markdown(
            """
            <div class="pv-workstation-panel" style="text-align: center; padding: 40px 20px;">
                <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="color: #64748b; margin-bottom: 8px;">
                    <rect width="18" height="18" x="3" y="3" rx="2"/>
                    <path d="M9 3v18"/>
                    <path d="M15 3v18"/>
                    <path d="M3 9h18"/>
                    <path d="M3 15h18"/>
                </svg>
                <div style="font-weight: 600; color: #f1f5f9; margin-bottom: 4px;">No CT Slices Loaded</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Upload ordered 16-bit TIFF cross-sections or select a sample series to initialize the viewport.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return 0

    # Initialize current slice index in session state if missing
    state_key = f"{key_prefix}_current_slice_idx"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    # Ensure index is in bounds
    if st.session_state[state_key] >= num_slices:
        st.session_state[state_key] = num_slices - 1

    current_idx = st.session_state[state_key]
    anatomical_idx = slice_indices[current_idx] if current_idx < len(slice_indices) else current_idx + 1

    # Viewport Top Bar (Patient & Acquisition Metadata)
    st.markdown(
        f"""
        <div class="pv-workstation-panel" style="margin-bottom: 10px; padding: 12px 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-weight: 600; color: #f1f5f9; font-size: 0.88rem;">PATIENT: {patient_id}</span>
                    <span style="color: #64748b; margin: 0 6px;">&bull;</span>
                    <span style="font-family: var(--pv-font-mono); color: #38bdf8; font-size: 0.82rem;">SERIES: {series_id}</span>
                </div>
                <div style="font-family: var(--pv-font-mono); font-size: 0.80rem; color: #00b4d8;">
                    SLICE: <strong>{current_idx + 1}</strong> / {num_slices} (ANATOMICAL INDEX: {anatomical_idx})
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Central Dark Canvas CT Display Container
    st.markdown(
        """
        <div style="background-color: #000000; border: 1px solid #1e293b; border-radius: 6px; padding: 12px; display: flex; justify-content: center; align-items: center; min-height: 460px;">
        """,
        unsafe_allow_html=True
    )

    active_slice = slices[current_idx]
    # Ensure values in [0.0, 1.0] for display
    if active_slice.max() > 1.0:
        clipped = np.clip(active_slice, 200, 1800)
        norm_slice = (clipped - 200.0) / 1600.0
    else:
        norm_slice = active_slice

    st.image(
        norm_slice,
        caption=f"Axial Cross-Section {anatomical_idx} &bull; Display Window: [200, 1800] Radiodensity Counts",
        width=480
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Slice Navigation Controls (Previous / Next Buttons & Slider)
    if num_slices > 1:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        col_prev, col_slider, col_next = st.columns([1.2, 3.6, 1.2])

        with col_prev:
            if st.button(":material/arrow_back: Previous", key=f"{key_prefix}_prev_btn", use_container_width=True):
                if st.session_state[state_key] > 0:
                    st.session_state[state_key] -= 1
                    st.rerun()

        with col_slider:
            slider_val = st.slider(
                "Axial Slice Navigation (z-axis)",
                min_value=0,
                max_value=num_slices - 1,
                value=st.session_state[state_key],
                label_visibility="collapsed",
                key=f"{key_prefix}_slider"
            )
            if slider_val != st.session_state[state_key]:
                st.session_state[state_key] = slider_val
                st.rerun()

        with col_next:
            if st.button("Next :material/arrow_forward:", key=f"{key_prefix}_next_btn", use_container_width=True):
                if st.session_state[state_key] < num_slices - 1:
                    st.session_state[state_key] += 1
                    st.rerun()
    else:
        st.caption("Single-slice CT scan: axial z-axis slider disabled.")

    return st.session_state[state_key]
