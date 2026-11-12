"""
PulmoVision — Research Overview View (Page 1)
Landing page explaining what PulmoVision is, the problem it explores, how it works,
dataset provenance, research performance, model architecture, and scientific boundaries.
Designed for accessible clarity for non-technical visitors while preserving scientific rigor.
"""

from typing import Dict, Any
import streamlit as st

from app.utils.data_loader import load_dev_manifest, load_stage2_experiment_results
from app.components.navigation import DESTINATIONS


def render_overview_view() -> None:
    """Renders the accessible, clear research overview page."""

    # -------------------------------------------------------------------------
    # 0. Load Authoritative Data
    # -------------------------------------------------------------------------
    manifest = load_dev_manifest()
    dev_patients = int(manifest["patient_id"].nunique()) if not manifest.empty else 122
    acq_series = int(manifest.groupby(["patient_id", "series_id"]).ngroups) if not manifest.empty else 239
    unique_slices = int(len(manifest)) if not manifest.empty else 3132
    holdout_patients = 255

    exp_results = load_stage2_experiment_results()
    exp_b = exp_results.get("Exp B (BiLSTM Mean)", {}).get("pooled_oof_metrics", {})
    macro_f1 = float(exp_b.get("macro_f1", 0.9461))
    roc_auc = float(exp_b.get("roc_auc", 0.9688))
    sensitivity = float(exp_b.get("sensitivity", 0.9870))
    specificity = float(exp_b.get("specificity", 0.8889))
    tp = int(exp_b.get("tp", 76))
    total_covid = tp + int(exp_b.get("fn", 1))
    tn = int(exp_b.get("tn", 40))
    total_normal = tn + int(exp_b.get("fp", 5))

    # -------------------------------------------------------------------------
    # 1. Hero Section — Explain the Project
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="pv-card" style="border-left: 4px solid var(--pv-accent-primary); padding: 26px 28px; margin-bottom: 28px;">
            <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 14px;">
                <span style="display: inline-flex; align-items: center; width: fit-content; padding: 5px 11px; border-radius: 4px; font-size: 0.78rem; font-weight: 500; text-transform: none; letter-spacing: normal; background-color: rgba(16, 185, 129, 0.12); color: var(--pv-success); border: 1px solid rgba(16, 185, 129, 0.35);">Stage 2 research completed</span>
                <span style="display: inline-flex; align-items: center; width: fit-content; padding: 5px 11px; border-radius: 4px; font-size: 0.78rem; font-weight: 500; text-transform: none; letter-spacing: normal; background-color: rgba(239, 68, 68, 0.10); color: var(--pv-error); border: 1px solid rgba(239, 68, 68, 0.30);">Independent test set sealed</span>
            </div>
            <h1 style="font-size: 2.25rem; font-weight: 700; margin: 0 0 12px 0; color: var(--pv-text-primary); letter-spacing: -0.025em; line-height: 1.25;">
                Understanding Lung CT Scans with Artificial Intelligence
            </h1>
            <p style="font-size: 1.12rem; line-height: 1.6; color: var(--pv-text-primary); max-width: 960px; margin: 0 0 10px 0; font-weight: 500;">
                PulmoVision is an educational research project exploring how artificial intelligence can analyze multiple chest CT scan images together to identify patterns associated with COVID-19-labelled cases.
            </p>
            <p style="font-size: 0.98rem; line-height: 1.55; color: var(--pv-text-secondary); max-width: 920px; margin: 0 0 18px 0;">
                Instead of examining each CT image independently, the research model studies the order and context of multiple neighboring scan slices.
            </p>
            <div class="pv-disclaimer-banner" style="margin-bottom: 0; padding: 10px 16px; font-size: 0.85rem;">
                <strong>Research prototype only:</strong> Not validated or approved for clinical diagnosis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 2. How It Works — Three-Step Visual Explanation
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 1.45rem; font-weight: 600; margin: 0 0 4px 0; color: var(--pv-text-primary);">
                How It Works
            </h2>
            <p style="font-size: 0.92rem; color: var(--pv-text-secondary); margin: 0;">
                A simple three-step process combining individual image features with anatomical sequence context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 20px; display: flex; flex-direction: column;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--pv-accent-primary); background: var(--pv-bg-elevated); padding: 3px 8px; border-radius: 4px;">
                        Step 1
                    </span>
                    <span style="color: var(--pv-accent-secondary); font-size: 1.1rem;">&bull;</span>
                </div>
                <h3 style="font-size: 1.1rem; font-weight: 600; margin: 0 0 8px 0; color: var(--pv-text-primary);">
                    CT Scan Images
                </h3>
                <p style="font-size: 0.90rem; line-height: 1.55; color: var(--pv-text-secondary); margin: 0;">
                    Multiple grayscale images are collected from chest CT scans, capturing consecutive views of the lungs from top to bottom.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col2:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 20px; display: flex; flex-direction: column; border-top: 3px solid var(--pv-accent-primary);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--pv-accent-primary); background: var(--pv-bg-elevated); padding: 3px 8px; border-radius: 4px;">
                        Step 2
                    </span>
                    <span style="color: var(--pv-accent-primary); font-size: 1.1rem;">&bull;</span>
                </div>
                <h3 style="font-size: 1.1rem; font-weight: 600; margin: 0 0 8px 0; color: var(--pv-text-primary);">
                    AI Model Analysis
                </h3>
                <p style="font-size: 0.90rem; line-height: 1.55; color: var(--pv-text-secondary); margin: 0;">
                    The model extracts image features and analyzes the sequence of neighboring scan slices to capture anatomical patterns across the entire volume.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_col3:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 20px; display: flex; flex-direction: column;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--pv-accent-primary); background: var(--pv-bg-elevated); padding: 3px 8px; border-radius: 4px;">
                        Step 3
                    </span>
                    <span style="color: var(--pv-accent-secondary); font-size: 1.1rem;">&bull;</span>
                </div>
                <h3 style="font-size: 1.1rem; font-weight: 600; margin: 0 0 8px 0; color: var(--pv-text-primary);">
                    Research Classification
                </h3>
                <p style="font-size: 0.90rem; line-height: 1.55; color: var(--pv-text-secondary); margin: 0;">
                    The model produces a classification score for COVID-labelled or Normal categories. This score represents an academic pattern comparison, not a medical confirmation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Dataset Overview — The Data Behind the Research
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 1.45rem; font-weight: 600; margin: 0 0 4px 0; color: var(--pv-text-primary);">
                The Data Behind the Research
            </h2>
            <p style="font-size: 0.92rem; color: var(--pv-text-secondary); margin: 0;">
                All model training was conducted on a carefully partitioned development cohort, leaving the holdout dataset completely untouched.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_col1, data_col2, data_col3, data_col4 = st.columns(4)

    with data_col1:
        st.markdown(
            f"""
            <div class="pv-card" style="border-left: 4px solid var(--pv-accent-primary); padding: 18px 16px;">
                <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-secondary); margin-bottom: 6px;">
                    Development Patients
                </div>
                <div style="font-size: 2.1rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-accent-primary); line-height: 1.1; margin-bottom: 6px;">
                    {dev_patients}
                </div>
                <div style="font-size: 0.82rem; color: var(--pv-text-secondary); line-height: 1.4;">
                    Used for model training and 5-fold cross-validation (77 COVID-labelled, 45 Normal).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with data_col2:
        st.markdown(
            f"""
            <div class="pv-card" style="border-left: 4px solid var(--pv-accent-secondary); padding: 18px 16px;">
                <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-secondary); margin-bottom: 6px;">
                    Acquisition Series
                </div>
                <div style="font-size: 2.1rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-accent-secondary); line-height: 1.1; margin-bottom: 6px;">
                    {acq_series}
                </div>
                <div style="font-size: 0.82rem; color: var(--pv-text-secondary); line-height: 1.4;">
                    Distinct scan series capturing anatomical sequences (1–3 series per patient).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with data_col3:
        st.markdown(
            f"""
            <div class="pv-card" style="border-left: 4px solid var(--pv-success); padding: 18px 16px;">
                <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-secondary); margin-bottom: 6px;">
                    Unique CT Images
                </div>
                <div style="font-size: 2.1rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-success); line-height: 1.1; margin-bottom: 6px;">
                    {unique_slices:,}
                </div>
                <div style="font-size: 0.82rem; color: var(--pv-text-secondary); line-height: 1.4;">
                    Clean unique slices after cryptographic deduplication resolved 604 duplicate entries.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with data_col4:
        st.markdown(
            f"""
            <div class="pv-card" style="border-left: 4px solid var(--pv-warning); padding: 18px 16px;">
                <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-secondary); margin-bottom: 6px;">
                    Independent Holdout
                </div>
                <div style="font-size: 2.1rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-warning); line-height: 1.1; margin-bottom: 6px;">
                    {holdout_patients}
                </div>
                <div style="font-size: 0.82rem; color: var(--pv-text-secondary); line-height: 1.4;">
                    Independent test patients, kept strictly sealed and not evaluated during development.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size: 0.82rem; color: var(--pv-text-muted); margin-top: 4px; line-height: 1.5;">
            <em>Note on data discipline:</em> Duplicate image entries across folders were audited and resolved before modeling. Not all 377 patients in the raw archive were used for training; 255 patients remain isolated in the sealed holdout set.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Research Results — How Did the Model Perform?
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 1.45rem; font-weight: 600; margin: 0 0 4px 0; color: var(--pv-text-primary);">
                How Did the Model Perform?
            </h2>
            <p style="font-size: 0.92rem; color: var(--pv-text-secondary); margin: 0;">
                Performance evaluated via five-fold patient-level cross-validation across the 122 development patients.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        st.markdown(
            f"""
            <div class="pv-card" style="border-left: 4px solid var(--pv-accent-primary); padding: 18px 16px;">
                <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-secondary); margin-bottom: 6px;">
                    Macro F1 Score
                </div>
                <div style="font-size: 2.1rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-accent-primary); line-height: 1.1; margin-bottom: 6px;">
                    {macro_f1 * 100:.2f}%
                </div>
                <div style="font-size: 0.82rem; color: var(--pv-text-secondary); line-height: 1.4;">
                    A combined measure of performance across both classification categories.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown(
            f"""
            <div class="pv-card" style="border-left: 4px solid var(--pv-accent-secondary); padding: 18px 16px;">
                <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-secondary); margin-bottom: 6px;">
                    ROC-AUC Score
                </div>
                <div style="font-size: 2.1rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-accent-secondary); line-height: 1.1; margin-bottom: 6px;">
                    {roc_auc * 100:.2f}%
                </div>
                <div style="font-size: 0.82rem; color: var(--pv-text-secondary); line-height: 1.4;">
                    How well the model distinguishes between categories across different thresholds.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col3:
        st.markdown(
            f"""
            <div class="pv-card" style="border-left: 4px solid var(--pv-success); padding: 18px 16px;">
                <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-secondary); margin-bottom: 6px;">
                    Sensitivity
                </div>
                <div style="font-size: 2.1rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-success); line-height: 1.1; margin-bottom: 6px;">
                    {sensitivity * 100:.2f}%
                </div>
                <div style="font-size: 0.82rem; color: var(--pv-text-secondary); line-height: 1.4;">
                    {tp} of {total_covid} COVID-labelled development patients were identified (1 false negative).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col4:
        st.markdown(
            f"""
            <div class="pv-card" style="border-left: 4px solid var(--pv-info); padding: 18px 16px;">
                <div style="font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--pv-text-secondary); margin-bottom: 6px;">
                    Specificity
                </div>
                <div style="font-size: 2.1rem; font-weight: 700; font-family: var(--pv-font-mono); color: var(--pv-info); line-height: 1.1; margin-bottom: 6px;">
                    {specificity * 100:.2f}%
                </div>
                <div style="font-size: 0.82rem; color: var(--pv-text-secondary); line-height: 1.4;">
                    {tn} of {total_normal} Normal development patients were identified (5 false positives).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="pv-card" style="padding: 14px 18px; margin-top: 4px; margin-bottom: 8px; background-color: var(--pv-bg-elevated);">
            <div style="font-size: 0.88rem; line-height: 1.5; color: var(--pv-text-secondary);">
                <strong>Accessible Explanation:</strong> These results describe predictions on development data during cross-validation.
                They are not independent test results and do not represent clinical diagnostic accuracy. The 94.61% Macro F1 reflects balanced mathematical classification across development folds, not a 94.61% real-world diagnostic accuracy.
            </div>
        </div>
        <div style="font-size: 0.80rem; color: var(--pv-text-muted); margin-bottom: 24px;">
            Detailed fold-level metrics, confidence intervals, confusion matrices, and architecture comparisons are available on the <em>Experiment Analytics</em> page.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 5. Explain the Selected Model — What AI Model Did We Build?
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <h2 style="font-size: 1.45rem; font-weight: 600; margin: 0 0 6px 0; color: var(--pv-text-primary);">
                What AI Model Did We Build?
            </h2>
            <p style="font-size: 0.96rem; line-height: 1.6; color: var(--pv-text-secondary); margin: 0;">
                The selected model combines an image feature extractor with a bidirectional LSTM.
                The image extractor learns visual features from each CT slice.
                The LSTM analyzes the ordered sequence of those features to capture context across neighboring slices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Technical Architecture Specifications"):
        st.markdown(
            """
            <div style="font-size: 0.85rem; line-height: 1.65; color: var(--pv-text-secondary); padding: 4px 0;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Feature Extractor:</strong> EfficientNet-B0 convolutional neural network.</li>
                    <li><strong>Pretraining:</strong> Frozen ImageNet-1K pretrained weights (1,280-dimensional feature embedding per slice).</li>
                    <li><strong>Sequence Architecture:</strong> Bidirectional Long Short-Term Memory (BiLSTM) capturing top-to-bottom and bottom-to-top continuity.</li>
                    <li><strong>Hidden Size:</strong> 128 units per direction (256 concatenated dimensions).</li>
                    <li><strong>Temporal Aggregation:</strong> Masked global mean pooling across the sequence dimension.</li>
                    <li><strong>Trainable Parameters:</strong> 1,460,418 parameters (lightweight, rapid inference on standard hardware).</li>
                    <li><strong>Operating Threshold:</strong> Standardized decision threshold at &tau; = 0.50.</li>
                    <li><strong>Checkpoint Hash:</strong> <span style="font-family: var(--pv-font-mono); color: var(--pv-accent-secondary);">0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed</span></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. What Makes This Research Interesting?
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 1.45rem; font-weight: 600; margin: 0 0 4px 0; color: var(--pv-text-primary);">
                What Makes This Research Interesting?
            </h2>
            <p style="font-size: 0.92rem; color: var(--pv-text-secondary); margin: 0;">
                Six research practices that distinguish PulmoVision's methodology.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    int_col1, int_col2, int_col3 = st.columns(3)

    with int_col1:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 18px;">
                <h4 style="font-size: 0.98rem; font-weight: 600; margin: 0 0 6px 0; color: var(--pv-text-primary);">
                    Patient-Level Data Splitting
                </h4>
                <p style="font-size: 0.84rem; line-height: 1.5; color: var(--pv-text-secondary); margin: 0;">
                    CT slices from the same person never appear in both training and evaluation, preventing the model from recognizing individual patient anatomy rather than disease patterns.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 18px; margin-top: 14px;">
                <h4 style="font-size: 0.98rem; font-weight: 600; margin: 0 0 6px 0; color: var(--pv-text-primary);">
                    Direct Model Comparison
                </h4>
                <p style="font-size: 0.84rem; line-height: 1.5; color: var(--pv-text-secondary); margin: 0;">
                    We tested three distinct architectures to isolate whether adding recurrent sequence memory actually improves detection over simple feature averaging.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with int_col2:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 18px;">
                <h4 style="font-size: 0.98rem; font-weight: 600; margin: 0 0 6px 0; color: var(--pv-text-primary);">
                    Duplicate Image Detection
                </h4>
                <p style="font-size: 0.84rem; line-height: 1.5; color: var(--pv-text-secondary); margin: 0;">
                    Forensic SHA-256 hashing found and resolved 604 identical slices stored under different filenames, ensuring evaluation metrics are genuine.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 18px; margin-top: 14px;">
                <h4 style="font-size: 0.98rem; font-weight: 600; margin: 0 0 6px 0; color: var(--pv-text-primary);">
                    Separate Sealed Cohort
                </h4>
                <p style="font-size: 0.84rem; line-height: 1.5; color: var(--pv-text-secondary); margin: 0;">
                    A separate group of 255 patients has been kept completely untouched in a sealed quarantine to prevent accidental data contamination.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with int_col3:
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 18px;">
                <h4 style="font-size: 0.98rem; font-weight: 600; margin: 0 0 6px 0; color: var(--pv-text-primary);">
                    Ordered Sequence Analysis
                </h4>
                <p style="font-size: 0.84rem; line-height: 1.5; color: var(--pv-text-secondary); margin: 0;">
                    Rather than treating slices as detached photos, the model reads them in anatomical order from the top of the lungs to the bottom.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="pv-card" style="height: 100%; padding: 18px; margin-top: 14px;">
                <h4 style="font-size: 0.98rem; font-weight: 600; margin: 0 0 6px 0; color: var(--pv-text-primary);">
                    Transparent Documentation
                </h4>
                <p style="font-size: 0.84rem; line-height: 1.5; color: var(--pv-text-secondary); margin: 0;">
                    Every experiment, audit check, and limitation is openly documented and reproducible directly in the platform.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. Important Research Limitations
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="pv-card" style="border-left: 4px solid var(--pv-warning); padding: 20px 22px; background-color: var(--pv-bg-elevated); margin-bottom: 28px;">
            <h3 style="font-size: 1.15rem; font-weight: 600; margin: 0 0 10px 0; color: var(--pv-text-primary);">
                Important Research Limitations
            </h3>
            <ul style="font-size: 0.88rem; line-height: 1.6; color: var(--pv-text-secondary); margin: 0; padding-left: 20px;">
                <li><strong>Educational Prototype:</strong> This is an educational research prototype, not a medical diagnostic system approved by regulatory authorities.</li>
                <li><strong>Single-Source Dataset:</strong> The model was developed using a limited dataset from one hospital source; performance on external scanners is unverified.</li>
                <li><strong>Unopened Holdout:</strong> The independent holdout dataset of 255 patients has not been evaluated.</li>
                <li><strong>No Clinical Validation:</strong> External clinical validation and prospective clinical trials have not been performed.</li>
                <li><strong>Uncalibrated Scores:</strong> Model scores are raw neural network outputs, not calibrated medical probabilities.</li>
                <li><strong>Correlation, Not Causality:</strong> Visual explanations (Grad-CAM) highlight correlated image areas but do not establish medical reasoning or biological cause.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 8. Explore the Project
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h2 style="font-size: 1.45rem; font-weight: 600; margin: 0 0 4px 0; color: var(--pv-text-primary);">
                Explore the Project
            </h2>
            <p style="font-size: 0.92rem; color: var(--pv-text-secondary); margin: 0;">
                Navigate through the platform's research modules using the cards below or the sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dest_cards = [
        ("Dataset Explorer", "Explore the dataset, search patients and series, and inspect duplicate resolution."),
        ("Model Laboratory", "Understand the AI architectures and compare 2.5D sequences with 3D models."),
        ("Experiment Analytics", "Examine research results, ROC curves, and cross-validation comparisons."),
        ("CT Workstation", "Run the research inference workflow on real multi-slice thoracic CT scans."),
        ("Explainability", "Explore model visualizations (Grad-CAM) and their clinical boundaries."),
        ("Research Documentation", "Read the methodology, audit reports, and technical manuscripts."),
    ]

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    cols = [exp_col1, exp_col2, exp_col3]

    def _route_to_page(target_name: str) -> None:
        st.session_state["current_page"] = target_name
        st.query_params["page"] = target_name
        for name, icon in DESTINATIONS:
            if name == target_name:
                st.session_state["main_nav_radio"] = f"{icon} {name}"
                break

    for idx, (dest_name, dest_desc) in enumerate(dest_cards):
        col = cols[idx % 3]
        with col:
            st.markdown(
                f"""
                <div class="pv-card" style="padding: 16px; margin-bottom: 10px; height: 130px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-weight: 600; font-size: 0.95rem; color: var(--pv-text-primary); margin-bottom: 4px;">
                            {dest_name}
                        </div>
                        <div style="font-size: 0.80rem; line-height: 1.4; color: var(--pv-text-secondary);">
                            {dest_desc}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                f"Open {dest_name}",
                key=f"nav_btn_{idx}",
                use_container_width=True,
                on_click=_route_to_page,
                args=(dest_name,),
            )
