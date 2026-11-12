"""app/views/explainability_view.py - Explainability & Technical Boundaries.

Educational overview of interpretability in deep thoracic CT sequence models:
- 2D Spatial Saliency (Grad-CAM) vs. 1D Temporal Slice Attention
- Critical clinical warning: Mathematical Attention != Clinical Radiologist Attention
- Explainability failure modes, spurious correlations, and domain shifts
- Research prototype safety boundaries
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from app.components.header import render_page_header
from app.components.status_badges import render_quarantine_badge


def render_explainability_view():
    """Render Page 6: Model Interpretability, Attention Mechanisms & Clinical Limits."""
    render_page_header(
        title="Model Interpretability & Boundaries",
        subtitle="Deconstructing 2D spatial saliency vs. 1D sequence attention, safety boundaries, and validation limits",
        section="Scientific Research Suite",
    )

    st.markdown("""
    Explainability in 2.5D deep learning architectures operates across two distinct mathematical axes:
    **within-slice spatial representation** (what features the CNN backbone detects across 3-slice slabs) and
    **across-slice temporal integration** (which axial slice positions the sequence aggregator weights most heavily).
    """)

    # 1. Attention vs Saliency Comparison
    st.markdown("### 1. Spatial Saliency vs. Temporal Sequence Attention")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="pv-card">
            <h4 style="color: var(--accent-primary); margin-top: 0;">2D Spatial Saliency (Backbone Level)</h4>
            <p><strong>Mechanism:</strong> Gradient-weighted Class Activation Mapping (Grad-CAM) on the final convolutional stage of EfficientNet-B0 (<code>_conv_head</code>, 1280 feature channels).</p>
            <ul>
                <li>Highlights localized anatomical regions that maximized positive gradient response toward the class score.</li>
                <li>Operates at the spatial resolution of the feature map (typically 7&times;7 or 8&times;8 for 224&times;224 input), interpolated back to image coordinates.</li>
                <li><strong>Limitation:</strong> Susceptible to high-frequency background noise, contrast variation, and border artifacts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pv-card">
            <h4 style="color: var(--accent-secondary); margin-top: 0;">1D Temporal Slice Attention (Aggregator Level)</h4>
            <p><strong>Mechanism:</strong> Additive Bahdanau-style attention over BiLSTM hidden states (Experiment C) or uniform temporal pooling (Experiment B).</p>
            <ul>
                <li>Assigns a scalar weight $\\alpha_t \\in (0, 1)$ with $\\sum_{t=1}^T \\alpha_t = 1$ to each axial slice position $t$.</li>
                <li>Scores which slice indices contribute most strongly to the patient-level latent vector $\\mathbf{h}_{\\text{patient}}$.</li>
                <li><strong>Limitation:</strong> Identifies which slices were mathematically salient, but cannot localize pathology within those slices.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Critical Warning: Attention != Clinical Attention
    st.markdown("### 2. Critical Boundary: Mathematical Attention &ne; Radiologist Attention")

    st.error("""
    **CLINICAL MISINTERPRETATION RISK: ATTENTION WEIGHTS ARE NOT RADIOLOGICAL READINGS**
    
    A common cognitive error among clinicians and machine learning researchers is treating deep neural attention weights
    $\\alpha_t$ as an authoritative surrogate for radiological focus. Peer-reviewed literature (Jain & Wallace 2019,
    Wiegreffe & Pinter 2019) has repeatedly demonstrated that:
    
    1. **Attention is an internal routing mechanism:** In recurrent networks, hidden states $\\mathbf{h}_t$ accumulate contextual information from prior and subsequent slices. High attention on slice $t$ may reflect the aggregation of findings from elsewhere in the volume rather than an acute finding on slice $t$ itself.
    2. **Multiple Attention Distributions Yield Equivalent Outputs:** Adversarial perturbations can produce radically altered attention weights without shifting the output classification probability.
    3. **Zero Anatomical Alignment:** The model has no explicit inductive prior for pulmonary lobes, bronchopulmonary segments, fissures, or lymph nodes. Attention can concentrate on scan margins, couch foam, or diaphragm motion artifacts if correlated with the training cohort label distribution.
    """)

    # 3. Interactive Synthetic Attention Demonstration
    st.markdown("### 3. Anatomical Slice Weighting Simulation")
    st.markdown("""
    Explore how temporal attention distributions alter slice selection across a simulated 30-slice sequence
    exhibiting peripheral consolidations in the lower lobes (slices 20–28).
    """)

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.markdown("#### Simulation Controls")
        dispersion = st.slider("Attention Sharpness (Softmax Temperature)", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
        peak_slice = st.slider("Primary Pathology Epicenter (Slice Index)", min_value=5, max_value=28, value=22, step=1)
        noise_level = st.slider("Artifact Interference Weight", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

    with sim_col2:
        # Generate synthetic attention curve
        slices = np.arange(1, 31)
        raw_scores = np.exp(-((slices - peak_slice) ** 2) / (2 * (3.5 * dispersion) ** 2))
        raw_scores += np.random.RandomState(42).uniform(0, noise_level, size=len(slices))
        # Softmax normalization
        exp_scores = np.exp(raw_scores / dispersion)
        attn_weights = exp_scores / np.sum(exp_scores)

        df_sim = pd.DataFrame({
            "Slice Index": slices,
            "Attention Weight": attn_weights,
            "Anatomical Zone": [
                "Apices (1-8)" if s <= 8 else
                "Mid-Zone / Carina (9-18)" if s <= 18 else
                "Bases / Lower Lobes (19-30)"
                for s in slices
            ],
        })

        sim_chart = alt.Chart(df_sim).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Slice Index:O", title="Axial Slice Index (Superior -> Inferior)"),
            y=alt.Y("Attention Weight:Q", title="Normalized Attention Weight α_t", axis=alt.Axis(format=".3f")),
            color=alt.Color("Anatomical Zone:N", scale=alt.Scale(
                domain=["Apices (1-8)", "Mid-Zone / Carina (9-18)", "Bases / Lower Lobes (19-30)"],
                range=["#64748b", "#38bdf8", "#0284c7"]
            )),
            tooltip=["Slice Index:O", alt.Tooltip("Attention Weight:Q", format=".4f"), "Anatomical Zone:N"]
        ).properties(height=260, title="Simulated 1D Temporal Attention Weights across Thoracic Volume")

        st.altair_chart(sim_chart, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Technical Failure Modes & Boundary Matrix
    st.markdown("### 4. Technical Failure Modes & Mitigation Strategies")

    st.markdown("""
    <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
        <thead>
            <tr style="border-bottom: 2px solid var(--border-color); text-align: left;">
                <th style="padding: 10px;">Failure Mode</th>
                <th style="padding: 10px;">Root Phenomenon</th>
                <th style="padding: 10px;">Observed Clinical Risk</th>
                <th style="padding: 10px;">PulmoVision Engineering Safeguard</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 10px; font-weight: 600;">Spurious Edge Activation</td>
                <td style="padding: 10px;">CT scanner bed couch or chest-wall boundaries correlate with label</td>
                <td style="padding: 10px;">Model makes high-confidence prediction without evaluating lung parenchyma</td>
                <td style="padding: 10px;">HU radiodensity clipping <code>[200, 1800]</code> suppresses couch foam artifacts; strict patient split guarantees no couch identity leakage</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 10px; font-weight: 600;">Sequence Truncation Drift</td>
                <td style="padding: 10px;">Varied slice count across patients (min 19, max 43)</td>
                <td style="padding: 10px;">Truncation can drop basal segments where dependent consolidations reside</td>
                <td style="padding: 10px;">Variable-length masking (<code>PyTorch pad_packed_sequence</code>) avoids forced arbitrary downsampling or cropping</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 10px; font-weight: 600;">Overconfidence on Non-COVID Pneumonia</td>
                <td style="padding: 10px;">Binary framing (COVID vs. Non-COVID) groups viral, bacterial, and normal together</td>
                <td style="padding: 10px;">Bacterial lobar pneumonia may trigger high COVID probability due to consolidation overlap</td>
                <td style="padding: 10px;">Mandatory warning banner stating model does not differentiate non-COVID viral or atypical bacterial pathogens</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: 600;">Contrast Agent Distribution Shift</td>
                <td style="padding: 10px;">Training cohort is non-contrast high-resolution CT</td>
                <td style="padding: 10px;">Intravenous iodinated contrast produces vascular hyperdensities mimicking ground-glass opacity</td>
                <td style="padding: 10px;">Input validator checks HU distribution statistics and warns if mean density exceeds non-contrast thresholds</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Held-out Evaluation Quarantine Integrity
    st.markdown("### 5. Research Integrity & Evaluation Quarantine Status")
    st.markdown("""
    To eliminate data dredging, post-hoc optimization, and optimistic bias, the PulmoVision project
    strictly isolates the final evaluation cohort from all exploratory analyses:
    """)

    q_col1, q_col2, q_col3 = st.columns([1, 1, 1])
    with q_col1:
        st.markdown("""
        <div class="pv-card" style="text-align: center;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--pv-error); margin-bottom: 6px;">
                <rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
            <br>
            <strong>Held-Out Cohort Status</strong>
            <p style="color: var(--pv-text-muted); font-size: 0.85rem; margin-top: 6px;"><code>data/metadata/.../test_partition</code></p>
            <div style="margin-top: 10px;"><span class="pv-badge pv-badge-quarantine">SEALED & QUARANTINED</span></div>
        </div>
        """, unsafe_allow_html=True)

    with q_col2:
        st.markdown("""
        <div class="pv-card" style="text-align: center;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--pv-success); margin-bottom: 6px;">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            <br>
            <strong>Leakage Prevention</strong>
            <p style="color: var(--pv-text-muted); font-size: 0.85rem; margin-top: 6px;">Zero test access during hyperparameter selection or threshold tuning.</p>
            <div style="margin-top: 10px;"><span class="pv-badge pv-badge-verified">STRICT PATIENT BOUNDARY</span></div>
        </div>
        """, unsafe_allow_html=True)

    with q_col3:
        st.markdown("""
        <div class="pv-card" style="text-align: center;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--pv-accent-primary); margin-bottom: 6px;">
                <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
                <rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
            </svg>
            <br>
            <strong>Audit Compliance</strong>
            <p style="color: var(--pv-text-muted); font-size: 0.85rem; margin-top: 6px;">Validated by automated SHA-256 integrity checks in test suite.</p>
            <div style="margin-top: 10px;"><span class="pv-badge pv-badge-prototype">RESEARCH COMPLIANT</span></div>
        </div>
        """, unsafe_allow_html=True)

