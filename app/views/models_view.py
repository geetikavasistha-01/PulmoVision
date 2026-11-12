"""app/views/models_view.py - Model Laboratory View (Page 3).

Interactive architectural exploration of Stage 1 2.5D CNNs and Stage 2 Recurrent Sequence Models.
Emphasizes the mathematical and computational distinction between 2.5D sequence models and true 3D volumetric CNNs.
Zero decorative emojis.
"""

import streamlit as st
import pandas as pd
from app.components.architecture_diagrams import (
    render_stage2_architecture_diagram,
    render_component_inspector,
)
from app.components.header import render_page_header


def render_models_view() -> None:
    """Renders the comprehensive model laboratory page."""
    render_page_header(
        title="Model Architecture Laboratory",
        subtitle="Controlled ablation matrix: comparing non-recurrent temporal pooling, bidirectional recurrence, and additive attention",
        section="Scientific Research Suite",
    )

    # 1. Paradigm Comparison: Slice-Level vs. Patient-Level Sequence Modeling vs True 3D
    st.markdown("### Modeling Paradigms in Medical AI")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%;">
                <div style="font-weight: 600; font-size: 0.88rem; color: var(--pv-text-primary); margin-bottom: 6px;">2D Single-Slice CNN</div>
                <div style="font-size: 0.78rem; font-family: var(--pv-font-mono); color: var(--pv-text-muted); margin-bottom: 8px;">Input: [1, 512, 512]</div>
                <p style="font-size: 0.84rem; color: var(--pv-text-secondary); line-height: 1.45; margin: 0;">
                    Evaluates single cross-sections independently. Completely discards through-plane anatomical continuity along the z-axis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%;">
                <div style="font-weight: 600; font-size: 0.88rem; color: var(--pv-text-primary); margin-bottom: 6px;">2.5D Contiguous Stacking</div>
                <div style="font-size: 0.78rem; font-family: var(--pv-font-mono); color: var(--pv-text-muted); margin-bottom: 8px;">Input: [3, 224, 224]</div>
                <p style="font-size: 0.84rem; color: var(--pv-text-secondary); line-height: 1.45; margin: 0;">
                    Stacks slice triplets (t-1, t, t+1) as pseudo-channels. Provides localized through-plane context but aggregates patients via heuristic voting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p3:
        st.markdown(
            """
            <div class="pv-card" style="border-color: var(--pv-accent-primary); height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 600; font-size: 0.88rem; color: var(--pv-accent-primary);">Stage 2 Sequence Modeling</span>
                    <span class="pv-badge pv-badge-verified" style="font-size: 0.68rem; margin: 0;">LEADER</span>
                </div>
                <div style="font-size: 0.78rem; font-family: var(--pv-font-mono); color: var(--pv-accent-secondary); margin-bottom: 8px;">Input: [T, 1280] ordered</div>
                <p style="font-size: 0.84rem; color: var(--pv-text-secondary); line-height: 1.45; margin: 0;">
                    Extracts spatial features via frozen EfficientNet-B0 and routes ordered embeddings through a Bidirectional LSTM with masked pooling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Critical Clarification: 2.5D Sequence Model vs True 3D Volumetric CNN
    st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pv-card" style="border-left: 4px solid var(--pv-accent-secondary);">
            <div style="font-weight: 600; font-size: 0.90rem; color: var(--pv-text-primary); margin-bottom: 6px;">
                Architectural Distinction: 2.5D Sequence Modeling vs. True 3D Volumetric Convolutions
            </div>
            <p style="font-size: 0.85rem; color: var(--pv-text-secondary); line-height: 1.5; margin: 0;">
                It is essential to distinguish between a <strong>true 3D volumetric CNN</strong> (such as 3D U-Net or 3D ResNet) and
                the <strong>2.5D sequence architecture</strong> implemented in PulmoVision:
            </p>
            <ul style="font-size: 0.85rem; color: var(--pv-text-secondary); line-height: 1.5; margin-top: 8px; margin-bottom: 0;">
                <li><strong>True 3D Volumetric CNNs</strong> apply isotropic 3D kernel convolutions (e.g. <code>3&times;3&times;3</code>) across voxels. They require dense isotropic resampling along the z-axis and impose massive GPU VRAM footprints that scale cubically with volume resolution.</li>
                <li><strong>PulmoVision 2.5D Sequence Architecture</strong> decouples spatial representation from volumetric aggregation. High-resolution in-plane spatial patterns are extracted into compact 1,280-dimensional vectors by a frozen 2D CNN backbone, while the craniocaudal anatomical continuity is captured by a 1D recurrent sequence aggregator (BiLSTM). This achieves end-to-end patient classification with only 1.46M trainable parameters on standard hardware.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # 3. Stage 2 Controlled Ablation Selector
    st.markdown("### Stage 2 Architecture Explorer")
    st.caption("Select an architecture to inspect its structural pipeline, tensor dimensions, and trainable parameter counts.")

    tab_b, tab_c, tab_a, tab_s1 = st.tabs([
        "Exp B: BiLSTM + Mean Pooling [Leader]",
        "Exp C: BiLSTM + Additive Attention",
        "Exp A: Atemporal Mean Pooling",
        "Stage 1 Baseline Architectures",
    ])

    with tab_b:
        render_stage2_architecture_diagram("B")

    with tab_c:
        render_stage2_architecture_diagram("C")

    with tab_a:
        render_stage2_architecture_diagram("A")

    with tab_s1:
        st.markdown(
            """
            <div class="pv-card">
                <h4 style="margin: 0 0 8px 0; color: var(--pv-text-primary);">Stage 1 Baseline Architectures (Pretrained on 2.5D Triplets)</h4>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 10px;">
                    <div style="background: var(--pv-bg-elevated); padding: 12px; border-radius: 6px;">
                        <div style="font-weight: 600; color: var(--pv-text-primary); font-size: 0.88rem;">Custom CNN 2D</div>
                        <div style="font-family: var(--pv-font-mono); font-size: 0.78rem; color: var(--pv-text-muted); margin: 4px 0;">93,122 params</div>
                        <div style="font-size: 0.76rem; color: var(--pv-text-secondary);">Baseline 4-block convolutional network without pretraining.</div>
                    </div>
                    <div style="background: var(--pv-bg-elevated); padding: 12px; border-radius: 6px;">
                        <div style="font-weight: 600; color: var(--pv-text-primary); font-size: 0.88rem;">ResNet50 2.5D</div>
                        <div style="font-family: var(--pv-font-mono); font-size: 0.78rem; color: var(--pv-text-muted); margin: 4px 0;">23,512,130 params</div>
                        <div style="font-size: 0.76rem; color: var(--pv-text-secondary);">ImageNet-1K pretrained residual network adapted for 3-slice input.</div>
                    </div>
                    <div style="background: var(--pv-bg-elevated); padding: 12px; border-radius: 6px;">
                        <div style="font-weight: 600; color: var(--pv-text-primary); font-size: 0.88rem;">EfficientNet-B0 2.5D</div>
                        <div style="font-family: var(--pv-font-mono); font-size: 0.78rem; color: var(--pv-text-muted); margin: 4px 0;">4,010,110 params</div>
                        <div style="font-size: 0.76rem; color: var(--pv-text-secondary);">Compound-scaled convolutional network achieving equal test accuracy with 83% fewer params.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Interactive Component Deep-Dive Inspector
    st.markdown("### Interactive Layer & Component Inspector")
    inspector_choice = st.selectbox(
        "Select architectural module to inspect:",
        options=["BiLSTM Encoder", "Masked Global Mean Pooling", "Bahdanau Additive Attention", "MLP Classifier"],
    )
    render_component_inspector(inspector_choice)

    # 5. Summary Table of Exact Trainable Parameters
    st.markdown("### Verified Parameter Summary")
    param_summary = [
        {
            "Architecture": "Exp A: EfficientNet-B0 + Global Mean Pooling",
            "Sequence Memory": "None (Atemporal)",
            "Pooling Strategy": "Uniform Masked Mean",
            "Trainable Params": 82114,
            "Param String": "82,114 (~82K)",
        },
        {
            "Architecture": "Exp B: EfficientNet-B0 + BiLSTM + Mean Pooling [Leader]",
            "Sequence Memory": "1-Layer Bidirectional LSTM",
            "Pooling Strategy": "Recurrent Masked Mean",
            "Trainable Params": 1460418,
            "Param String": "1,460,418 (~1.46M)",
        },
        {
            "Architecture": "Exp C: EfficientNet-B0 + BiLSTM + Additive Attention",
            "Sequence Memory": "1-Layer Bidirectional LSTM",
            "Pooling Strategy": "Bahdanau Additive Attention",
            "Trainable Params": 1476930,
            "Param String": "1,476,930 (~1.48M)",
        },
    ]
    df_params = pd.DataFrame(param_summary)
    st.dataframe(
        df_params[["Architecture", "Sequence Memory", "Pooling Strategy", "Param String"]],
        hide_index=True,
        width="stretch",
    )
