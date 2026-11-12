"""
PulmoVision — Research Overview View (Page 1)
Presents the research question, architectural paradigm, development cohort, leading results,
and end-to-end scientific methodology.
"""

import streamlit as st
from app.components.metric_cards import render_cohort_kpis, render_leader_benchmark_cards


def render_overview_view() -> None:
    """Renders the comprehensive research overview page."""
    
    # 1. Hero Section
    st.markdown(
        """
        <div class="pv-card" style="border-left: 4px solid var(--pv-accent-primary); padding: 22px 24px; margin-bottom: 24px;">
            <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--pv-accent-secondary); margin-bottom: 6px;">
                AI Medical Imaging Research Platform &bull; Stage 2 Frozen Benchmark
            </div>
            <h1 style="font-size: 2.2rem; font-weight: 700; margin: 0 0 10px 0; color: var(--pv-text-primary); letter-spacing: -0.02em;">
                Patient-Level Craniocaudal Sequence Modelling for Thoracic CT
            </h1>
            <p style="font-size: 1.05rem; line-height: 1.55; color: var(--pv-text-secondary); max-width: 950px; margin: 0 0 16px 0;">
                Investigating how bidirectional recurrent sequence architectures capture volumetric anatomical context 
                across contiguous axial thoracic CT slices, eliminating the spatial discontinuity and signal attenuation 
                inherent in independent 2D / 2.5D slice-level classification.
            </p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="pv-badge pv-badge-prototype">Educational Research Prototype</span>
                <span class="pv-badge pv-badge-verified">Stage 2 CV Complete</span>
                <span class="pv-badge pv-badge-quarantine">Held-Out Test Sealed</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Research Summary KPIs
    st.markdown("### Cohort Volume & Provenance")
    render_cohort_kpis()

    # 3. Leading Experiment Highlight
    st.markdown("### Leading Architecture Benchmark (Experiment B: BiLSTM + Mean Pooling 🏆)")
    st.caption("Pooled out-of-fold performance across all 122 development patients (5 outer folds).")
    render_leader_benchmark_cards()

    # 4. Central Research Question & Rationale
    col_q, col_why = st.columns([1, 1])

    with col_q:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--pv-accent-primary); margin-bottom: 8px;">
                    Core Research Question
                </div>
                <h3 style="font-size: 1.25rem; font-weight: 600; margin: 0 0 12px 0; color: var(--pv-text-primary);">
                    Can Bidirectional Recurrence Preserve Anatomical Continuity Along the Axial Axis?
                </h3>
                <p style="font-size: 0.92rem; line-height: 1.55; color: var(--pv-text-secondary); margin: 0;">
                    In conventional medical CNNs, CT slices are evaluated as isolated 2D images or 3-slice stacks (2.5D). 
                    However, real chest CT acquisitions are <strong>continuous volumetric series</strong> from apex to base. 
                    When viral pneumonia features (such as peripheral ground-glass opacities) are concentrated in basal lobes, 
                    unweighted uniform pooling across an entire volume can attenuate localized focal patterns. 
                    This study investigates whether <strong>Bidirectional LSTMs</strong> can retain sequential contextual memory 
                    to improve patient-level discrimination.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_why:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--pv-accent-secondary); margin-bottom: 8px;">
                    Methodological Guarantees
                </div>
                <h3 style="font-size: 1.25rem; font-weight: 600; margin: 0 0 12px 0; color: var(--pv-text-primary);">
                    Zero-Leakage Experimental Integrity
                </h3>
                <ul style="font-size: 0.90rem; line-height: 1.6; color: var(--pv-text-secondary); padding-left: 20px; margin: 0;">
                    <li><strong>Patient-Balanced Loss:</strong> Multi-series patients are assigned equal 1/N_p gradient weight per optimizer update to eliminate sampling bias.</li>
                    <li><strong>Frozen Feature Extractor:</strong> External ImageNet-1K EfficientNet-B0 backbone prevents in-domain feature leakage.</li>
                    <li><strong>Strict Outer CV:</strong> 5 outer stratified folds with inner 80/20 validation holdouts for threshold selection; zero outer refitting.</li>
                    <li><strong>Sealed Test Cohort:</strong> 255 patients (8,322 slices) remain completely unopened.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin: 24px 0 16px 0;'></div>", unsafe_allow_html=True)

    # 5. Scientific Research Workflow Diagram
    st.markdown("### End-to-End Scientific Research Workflow")
    st.caption("From raw 16-bit CT acquisition to patient-balanced sequence optimization and out-of-fold evaluation.")

    st.markdown(
        """
        <div class="pv-card" style="overflow-x: auto; padding: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; min-width: 880px; gap: 8px;">
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; text-align: center; flex: 1;">
                    <div style="font-size: 0.75rem; color: var(--pv-text-muted); font-family: var(--pv-font-mono);">PHASE 1</div>
                    <div style="font-weight: 600; font-size: 0.85rem; color: var(--pv-text-primary); margin: 4px 0;">Raw CT Dataset</div>
                    <div style="font-size: 0.72rem; color: var(--pv-text-secondary);">16-bit TIFFs<br>3,736 entries</div>
                </div>
                <div style="color: var(--pv-accent-primary); font-size: 1.2rem;">&rarr;</div>
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; text-align: center; flex: 1;">
                    <div style="font-size: 0.75rem; color: var(--pv-text-muted); font-family: var(--pv-font-mono);">PHASE 2</div>
                    <div style="font-weight: 600; font-size: 0.85rem; color: var(--pv-text-primary); margin: 4px 0;">Deduplication</div>
                    <div style="font-size: 0.72rem; color: var(--pv-text-secondary);">SHA256 forensic audit<br>3,132 canonical slices</div>
                </div>
                <div style="color: var(--pv-accent-primary); font-size: 1.2rem;">&rarr;</div>
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; text-align: center; flex: 1;">
                    <div style="font-size: 0.75rem; color: var(--pv-text-muted); font-family: var(--pv-font-mono);">PHASE 3</div>
                    <div style="font-weight: 600; font-size: 0.85rem; color: var(--pv-text-primary); margin: 4px 0;">CT Preprocessing</div>
                    <div style="font-size: 0.72rem; color: var(--pv-text-secondary);">Clip [200, 1800]<br>Scale [0, 1] &bull; 224&times;224</div>
                </div>
                <div style="color: var(--pv-accent-primary); font-size: 1.2rem;">&rarr;</div>
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; text-align: center; flex: 1;">
                    <div style="font-size: 0.75rem; color: var(--pv-text-muted); font-family: var(--pv-font-mono);">PHASE 4</div>
                    <div style="font-weight: 600; font-size: 0.85rem; color: var(--pv-text-primary); margin: 4px 0;">Feature Extraction</div>
                    <div style="font-size: 0.72rem; color: var(--pv-text-secondary);">Frozen EfficientNet-B0<br>1280-d embedding / slice</div>
                </div>
                <div style="color: var(--pv-accent-primary); font-size: 1.2rem;">&rarr;</div>
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; text-align: center; flex: 1;">
                    <div style="font-size: 0.75rem; color: var(--pv-text-muted); font-family: var(--pv-font-mono);">PHASE 5</div>
                    <div style="font-weight: 600; font-size: 0.85rem; color: var(--pv-text-primary); margin: 4px 0;">Sequence Modeling</div>
                    <div style="font-size: 0.72rem; color: var(--pv-text-secondary);">Ablation Matrix<br>Exp A, B, and C</div>
                </div>
                <div style="color: var(--pv-accent-primary); font-size: 1.2rem;">&rarr;</div>
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-border); border-radius: 6px; padding: 12px; text-align: center; flex: 1;">
                    <div style="font-size: 0.75rem; color: var(--pv-text-muted); font-family: var(--pv-font-mono);">PHASE 6</div>
                    <div style="font-weight: 600; font-size: 0.85rem; color: var(--pv-text-primary); margin: 4px 0;">Cross-Validation</div>
                    <div style="font-size: 0.72rem; color: var(--pv-text-secondary);">5 outer folds<br>Patient-balanced loss</div>
                </div>
                <div style="color: var(--pv-accent-primary); font-size: 1.2rem;">&rarr;</div>
                <div style="background-color: var(--pv-bg-elevated); border: 1px solid var(--pv-accent-primary); border-radius: 6px; padding: 12px; text-align: center; flex: 1;">
                    <div style="font-size: 0.75rem; color: var(--pv-accent-primary); font-family: var(--pv-font-mono);">DEPLOYMENT</div>
                    <div style="font-weight: 600; font-size: 0.85rem; color: var(--pv-text-primary); margin: 4px 0;">Research Inference</div>
                    <div style="font-size: 0.72rem; color: var(--pv-text-secondary);">Frozen Checkpoint<br>122 dev fit &bull; tau=0.50</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin: 24px 0 16px 0;'></div>", unsafe_allow_html=True)

    # 6. Historical Architecture Comparison (Stage 1 vs Stage 2)
    st.markdown("### Architectural Evolution: Stage 1 Baselines vs. Stage 2 Sequence Modeling")
    st.caption("Documenting distinct research stages and differing evaluation protocols (not a directly controlled head-to-head improvement).")

    stage_col1, stage_col2 = st.columns(2)

    with stage_col1:
        st.markdown(
            """
            <div class="pv-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 600; color: var(--pv-text-primary);">Stage 1: 2.5D Slice Classification Baseline</span>
                    <span class="pv-badge" style="background-color: var(--pv-bg-elevated); color: var(--pv-text-muted);">Completed</span>
                </div>
                <p style="font-size: 0.86rem; color: var(--pv-text-secondary); line-height: 1.5; margin-bottom: 10px;">
                    Trained 2D and 2.5D CNN backbones (Custom CNN, ResNet50, EfficientNet-B0) to classify individual CT slices. 
                    Patient-level status was derived by heuristic thresholding on the proportion of positive slices (&tau; = 0.20).
                </p>
                <div style="font-size: 0.80rem; font-family: var(--pv-font-mono); color: var(--pv-text-muted); background: var(--pv-bg-elevated); padding: 8px; border-radius: 4px;">
                    <div>&bull; Unit of optimization: Individual slice triplets</div>
                    <div>&bull; Patient decision: Post-hoc positive slice ratio &ge; 20%</div>
                    <div>&bull; Test generalizability: 0.8307 Macro F1 on held-out test</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with stage_col2:
        st.markdown(
            """
            <div class="pv-card" style="border-color: var(--pv-accent-primary);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 600; color: var(--pv-accent-primary);">Stage 2: Patient-Level Sequence Modeling</span>
                    <span class="pv-badge pv-badge-verified">Leading Milestone</span>
                </div>
                <p style="font-size: 0.86rem; color: var(--pv-text-secondary); line-height: 1.5; margin-bottom: 10px;">
                    Processes ordered craniocaudal slice sequences end-to-end using recurrent architectures (BiLSTM) 
                    and equal patient-balanced loss aggregation, directly optimizing the patient-level objective.
                </p>
                <div style="font-size: 0.80rem; font-family: var(--pv-font-mono); color: var(--pv-text-muted); background: var(--pv-bg-elevated); padding: 8px; border-radius: 4px;">
                    <div>&bull; Unit of optimization: Full patient scan sequence</div>
                    <div>&bull; Patient decision: End-to-end sequence classifier at &tau; = 0.50</div>
                    <div>&bull; Dev-cohort performance: 0.9461 Macro F1, 98.70% Sens</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

