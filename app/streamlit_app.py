"""
PulmoVision — AI-Powered Medical Imaging Research Platform
Interactive Streamlit Dashboard for Thoracic CT Sequence Analysis & Explainable AI.
Stage 2: Patient-Level Craniocaudal Sequence Modelling
"""

import os
import glob
import json
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="PulmoVision — CT Sequence Research Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #0066cc;
        margin-bottom: 12px;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 20px;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .highlight-card {
        background-color: #e8f4fd;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #b6d4fe;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_sequence_predictor():
    """
    Cached predictor loader: loads deployment checkpoint and frozen backbone once.
    """
    from src.inference.sequence_predictor import SequencePredictor
    ckpt_path = "models/stage2_bilstm_deployment.pth"
    return SequencePredictor(checkpoint_path=ckpt_path)


def load_cached_results():
    """Loads verified nested CV result JSON records."""
    paths = {
        "Exp A (Global Mean)": "reports/experiments/stage2_expA_effnet_mean_pooling/nested_cv_results.json",
        "Exp B (BiLSTM + Mean)": "reports/experiments/stage2_expB_effnet_bilstm_mean/nested_cv_results.json",
        "Exp C (BiLSTM + Attn)": "reports/experiments/stage2_expC_effnet_bilstm_attention/nested_cv_results.json",
    }
    data = {}
    for k, p in paths.items():
        if os.path.exists(p):
            with open(p) as f:
                data[k] = json.load(f)
    return data


def main():
    st.sidebar.title("🫁 PulmoVision")
    st.sidebar.caption("Thoracic CT Deep Learning Research Prototype")

    pages = [
        "🏠 Overview",
        "📂 Dataset Explorer",
        "🧠 Model Architecture",
        "📊 Experiment Results",
        "🔬 CT Sequence Analysis",
        "💡 Explainability",
        "📖 Research Documentation"
    ]

    nav = st.sidebar.radio("Navigation", pages)

    # Global Research Disclaimer Banner
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <b>RESEARCH PROTOTYPE ONLY:</b> PulmoVision is an educational and academic research prototype 
        investigating deep learning sequence architectures. It is <b>NOT</b> a certified medical diagnostic device, 
        clinical decision support tool, or clinically validated system. Dataset labels reflect categorical archive annotations.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 1. OVERVIEW
    # -------------------------------------------------------------
    if nav == "🏠 Overview":
        st.title("🫁 PulmoVision: Patient-Level CT Sequence Modelling")
        st.caption("Investigating Craniocaudal Sequential Memory vs. Atemporal Pooling in Thoracic CT")

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("The Research Challenge")
            st.write("""
            In conventional deep learning for computed tomography, models typically evaluate 2D cross-sections 
            independently or stack 3 contiguous slices (2.5D), applying heuristic thresholds to positive slice counts.
            
            However, real chest CT scans are **anatomically continuous volumetric sequences** along the axial 
            z-axis (from lung apices to diaphragmatic bases). When viral pneumonia features are localized to basal segments, 
            simple atemporal averaging often attenuates focal visual signals across dozens of uninvolved slices.
            """)

            st.markdown("### Stage 2 Core Hypotheses")
            st.markdown("""
            * **Bidirectional Recurrence:** Can a Bidirectional LSTM maintain sequential context across ordered CT slices to improve patient-level diagnosis?
            * **Attention vs. Mean Pooling:** Does learned Additive Attention provide an incremental benefit over uniform recurrent pooling?
            * **Leakage-Free Optimization:** Evaluating models with frozen external feature extractors and exact patient-balanced loss aggregation.
            """)

        with col2:
            st.markdown("""
            <div class="highlight-card">
                <h4>🏆 Development Cohort Leader</h4>
                <p><b>Experiment B: EfficientNet-B0 + BiLSTM + Global Mean Pooling</b></p>
                <ul>
                    <li><b>Macro F1:</b> 0.9461 (95% CI: [0.8984, 0.9820])</li>
                    <li><b>ROC-AUC:</b> 0.9688 | <b>PR-AUC:</b> 0.9760</li>
                    <li><b>Sensitivity:</b> 98.70% (76/77 COVID-19 detected)</li>
                    <li><b>Specificity:</b> 88.89% (40/45 Normal detected)</li>
                </ul>
                <small>*Evaluated under leak-free 5-fold outer cross-validation on 122 development patients.</small>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Architecture Evolution: Stage 1 vs. Stage 2")
        evo_df = pd.DataFrame({
            "Dimension": ["Input Representation", "Backbone Status", "Patient Decision Method", "Optimization Level", "Dev Macro F1", "COVID-19 Sensitivity"],
            "Stage 1 (2D / 2.5D Baselines)": ["Single slice or 3-slice triplet", "Fine-tuned on dev slices", "Post-hoc positive slice ratio threshold (tau=0.20)", "Slice-level CrossEntropy", "0.8307 (EffNet 2.5D)", "94.4%"],
            "Stage 2 (Craniocaudal Sequence)": ["Full ordered volume sequence (T slices)", "Frozen ImageNet-1K EfficientNet-B0", "End-to-end patient probability P(COVID)", "Patient-Balanced Loss L_batch", "0.9461 (BiLSTM Mean)", "98.70%"]
        })
        st.dataframe(evo_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 2. DATASET EXPLORER
    # -------------------------------------------------------------
    elif nav == "📂 Dataset Explorer":
        st.title("📂 Dataset & Anatomical Cohort Discovery")
        st.caption("COVID-CTset Iranian Cohort: 16-Bit Grayscale Radiodensity & Anatomical De-duplication")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Development Patients", "122", "77 COVID, 45 Normal")
        col2.metric("Scan Series", "239", "Up to 3 series / patient")
        col3.metric("Unique CT Slices", "3,132", "Anatomically de-duplicated")
        col4.metric("Quarantined Test Set", "255 pts", "8,322 slices (Sealed)")

        st.divider()
        st.subheader("1. Anatomical De-Duplication Protocol")
        st.write("""
        Forensic SHA256 hashing of the original development archive revealed 604 byte-identical duplicate files 
        across redundant folders (`normal1` vs `normal4` / `normal5`). To preserve anatomical validity, we applied 
        **Option 1: Anatomical De-duplication**, retaining exactly 1 canonical file per `(patient, series, slice)` 
        and mapping duplicates into a verified provenance record.
        """)

        st.subheader("2. 16-Bit Radiodensity Preprocessing")
        st.write("""
        The images in COVID-CTset are stored as uncompressed 16-bit TIFF files with pixel values up to $\\approx 2204$.
        Because accompanying DICOM rescale parameters are not provided, these values are designated as **radiodensity counts** (not certified Hounsfield Units).
        """)

        prep_col1, prep_col2 = st.columns(2)
        with prep_col1:
            st.markdown("#### Standardization Pipeline")
            st.code("""
1. Intensity Windowing: Clip [200, 1800]
2. Dynamic Range Scaling: (I_clip - 200) / 1600 -> [0.0, 1.0]
3. Channel Replication: Grayscale -> 3 identical channels (R=G=B)
4. ImageNet Normalization: Standardize with mu=[0.485, 0.456, 0.406]
5. Bilinear Resize: 224 x 224 pixels (preserving 1:1 square aspect ratio)
            """, language="text")

        with prep_col2:
            st.markdown("#### Sequence Length Distribution")
            st.write("""
            - **Minimum slices per patient:** 1 (Edge case handled gracefully)
            - **Maximum slices per patient:** 84
            - **Median slices per patient:** 26
            - **Variable Sequence Handling:** Zero-padding to batch maximum length $T_{\\max}$ with boolean attention masking.
            """)

    # -------------------------------------------------------------
    # 3. MODEL ARCHITECTURE
    # -------------------------------------------------------------
    elif nav == "🧠 Model Architecture":
        st.title("🧠 Stage 2 Sequence Architectures")
        st.caption("Controlled Ablation Matrix: Atemporal Pooling vs. Recurrent Memory vs. Additive Attention")

        tab_b, tab_c, tab_a = st.tabs(["⭐ Experiment B (BiLSTM Mean)", "Experiment C (BiLSTM Attention)", "Experiment A (Atemporal Baseline)"])

        with tab_b:
            st.subheader("Experiment B: EfficientNet-B0 + BiLSTM + Global Mean Pooling (Development Leader)")
            st.write("""
            Ingests ordered slice embeddings into a 1-layer Bidirectional LSTM. Recurrent forward and backward 
            states are concatenated ($2H = 256$), followed by masked global mean pooling across the axial sequence.
            """)
            st.code("""
Input: [B, T_max, 1280] (Slice embeddings from frozen EfficientNet-B0)
  ↓
BiLSTM Encoder: 1 layer, hidden_dim=128 per direction -> Output: [B, T_max, 256]
  ↓
Masked Global Mean Pooling over T: [B, 256]
  ↓
MLP Classifier: Linear(256 -> 64) -> SiLU -> Dropout(0.3) -> Linear(64 -> 2)
  ↓
Output: Patient Logits [B, 2] -> Predicted Probability P(COVID-19)
Total Trainable Parameters: 1,460,418 (~1.46M)
            """, language="text")

        with tab_c:
            st.subheader("Experiment C: EfficientNet-B0 + BiLSTM + Additive Attention Pooling")
            st.write("""
            Replaces uniform mean pooling with single-head Bahdanau additive attention, computing dynamic 
            saliency weights $\\alpha_t$ across the axial slices.
            """)
            st.code("""
BiLSTM States: h_t in R^256
  ↓
Attention Energy: u_t = tanh(W_a * h_t + b_a) in R^64
Attention Scores: e_t = v_a^T * u_t in R^1
Attention Weights: alpha_t = softmax(e_t) in [0, 1] (masked over padding)
  ↓
Context Vector: c = sum_t (alpha_t * h_t) in R^256
  ↓
MLP Classifier: Linear(256 -> 64) -> SiLU -> Dropout(0.3) -> Linear(64 -> 2)
Total Trainable Parameters: 1,476,930 (~1.48M)
            """, language="text")

        with tab_a:
            st.subheader("Experiment A: EfficientNet-B0 + Global Mean Pooling (Atemporal Baseline)")
            st.write("""
            Computes uniform mean pooling directly over slice embeddings without temporal recurrent modeling.
            """)
            st.code("""
Input: [B, T_max, 1280]
  ↓
Masked Global Mean Pooling over T: [B, 1280]
  ↓
MLP Classifier: Linear(1280 -> 64) -> SiLU -> Dropout(0.3) -> Linear(64 -> 2)
Total Trainable Parameters: 82,114 (~82K)
            """, language="text")

    # -------------------------------------------------------------
    # 4. EXPERIMENT RESULTS
    # -------------------------------------------------------------
    elif nav == "📊 Experiment Results":
        st.title("📊 Verified Cross-Validation Results")
        st.caption("5-Fold Outer Cross-Validation with Inner Holdout (15 Total Runs, 100% Verified)")

        results = load_cached_results()

        st.subheader("1. Pooled Out-of-Fold Performance Summary (N = 122 Patients)")
        summary_rows = [
            {"Architecture": "Experiment A (Global Mean)", "Macro F1 (95% CI)": "0.8236 [0.7518, 0.8862]", "ROC-AUC": "0.8652", "PR-AUC": "0.9046", "Sensitivity": "77.92%", "Specificity": "91.11%", "TP": 60, "FP": 4, "TN": 41, "FN": 17},
            {"Architecture": "Experiment B (BiLSTM Mean) 🏆", "Macro F1 (95% CI)": "0.9461 [0.8984, 0.9820]", "ROC-AUC": "0.9688", "PR-AUC": "0.9760", "Sensitivity": "98.70%", "Specificity": "88.89%", "TP": 76, "FP": 5, "TN": 40, "FN": 1},
            {"Architecture": "Experiment C (BiLSTM Attn)", "Macro F1 (95% CI)": "0.9211 [0.8644, 0.9645]", "ROC-AUC": "0.9466", "PR-AUC": "0.9466", "Sensitivity": "93.51%", "Specificity": "91.11%", "TP": 72, "FP": 4, "TN": 41, "FN": 5},
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Cross-Fold Descriptive Variability (Mean ± Std)")
            st.table(pd.DataFrame({
                "Metric": ["Macro F1", "Accuracy", "Sensitivity", "Specificity", "ROC-AUC"],
                "Exp A (Global Mean)": ["0.8218 ± 0.0715", "0.8277 ± 0.0670", "0.7775 ± 0.0798", "0.9111 ± 0.1449", "0.8726 ± 0.0886"],
                "Exp B (BiLSTM Mean)": ["0.9454 ± 0.0588", "0.9510 ± 0.0526", "0.9875 ± 0.0280", "0.8889 ± 0.1111", "0.9804 ± 0.0358"],
                "Exp C (BiLSTM Attn)": ["0.9200 ± 0.0740", "0.9263 ± 0.0663", "0.9342 ± 0.0817", "0.9111 ± 0.1449", "0.9664 ± 0.0635"],
            }))

        with col2:
            st.markdown("### Threshold Stability Analysis")
            st.write("""
            Tuned across $\\tau^* \\in [0.20, 0.80]$ (step 0.05) exclusively on inner-validation partitions:
            - **Exp A:** $\\tau^* = [0.50, 0.50, 0.50, 0.50, 0.50]$ (100% stable at 0.50)
            - **Exp B:** $\\tau^* = [0.50, 0.50, 0.50, 0.50, 0.50]$ (100% stable at 0.50)
            - **Exp C:** $\\tau^* = [0.50, 0.50, 0.50, 0.50, 0.30]$ (80% at 0.50, 20% at 0.30)
            - **Boundary check:** Zero folds hit the extreme search boundaries ($0.20$ or $0.80$).
            """)

    # -------------------------------------------------------------
    # 5. CT SEQUENCE ANALYSIS
    # -------------------------------------------------------------
    elif nav == "🔬 CT Sequence Analysis":
        st.title("🔬 Interactive CT Sequence Analysis")
        st.caption("Upload ordered CT slices from a single patient series for patient-level recurrent sequence inference.")

        predictor = get_sequence_predictor()

        sample_dir = "data/interim/sample_audit/TrainValidation"
        sample_files = sorted(glob.glob("data/raw/TrainValidation/*patient100_SR_2*.tif"))[:5]
        if not sample_files and os.path.exists(sample_dir):
            sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.tif")))

        tab1, tab2 = st.tabs(["⚡ One-Click Verified Sample Scan", "📤 Upload Custom CT Slices"])

        with tab1:
            if sample_files:
                st.write(f"Found **{len(sample_files)}** sample slice(s) in development scan (`patient100_SR_2`).")
                if st.button("🚀 Analyze Sample Sequence with BiLSTM", type="primary"):
                    with st.spinner("Executing sequence validation, feature extraction & BiLSTM inference..."):
                        res = predictor.predict_sequence(sample_files)

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Predicted Category", res["predicted_class"])
                    col_m2.metric("COVID-19 Probability", f"{res['covid_19_probability']:.1%}")
                    col_m3.metric("Sequence Length", f"{res['num_slices']} slices")
                    col_m4.metric("Inference Latency", f"{res['inference_latency_ms']:.1f} ms", f"{res['per_slice_latency_ms']:.1f} ms/slice")

                    if res["warning"]:
                        st.warning(f"⚠️ {res['warning']}")

                    st.divider()
                    st.subheader("Craniocaudal Slice Visualizer")
                    if res["num_slices"] > 1:
                        slice_idx = st.slider("Axial Slice Navigation (z-axis)", 0, res["num_slices"] - 1, 0)
                    else:
                        slice_idx = 0
                    st.image(res["display_slices"][slice_idx], caption=f"Axial Slice {res['slice_indices'][slice_idx]} (Lung Window)", width=450)
            else:
                st.info("No sample files found in interim cache. Please use the Upload tab.")

        with tab2:
            st.subheader("Upload Ordered Slices from a CT Scan Series")
            st.caption("Supported formats: 16-bit TIFF (.tif, .tiff), PNG, JPG, NumPy (.npy). Slices must belong to the same patient and series.")

            uploaded_files = st.file_uploader(
                "Select CT slices",
                type=["tif", "tiff", "png", "jpg", "jpeg", "npy"],
                accept_multiple_files=True
            )

            if uploaded_files:
                st.write(f"Selected **{len(uploaded_files)}** file(s).")
                from src.inference.sequence_validator import CTSequenceValidator, SequenceValidationError

                try:
                    fnames = [f.name for f in uploaded_files]
                    ordered_f, meta = CTSequenceValidator.validate_and_order_sequence(uploaded_files, filenames=fnames)
                    st.success(f"✅ Sequence Provenance Verified: Patient `{meta['patient_id']}`, Series `{meta['series_id']}` ({meta['num_slices']} slices)")

                    if meta["warning"]:
                        st.warning(f"⚠️ {meta['warning']}")

                    if st.button("Run Volumetric Sequence Analysis", type="primary"):
                        with st.spinner("Extracting slice embeddings & evaluating BiLSTM..."):
                            res = predictor.predict_sequence(uploaded_files, filenames=fnames)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Patient Prediction", res["predicted_class"])
                        c2.metric("COVID-19 Probability", f"{res['covid_19_probability']:.2%}")
                        c3.metric("Total Latency", f"{res['inference_latency_ms']:.1f} ms")

                        st.subheader("Axial CT Cross-Section Viewer")
                        if res["num_slices"] > 1:
                            nav_idx = st.slider("Select Slice", 0, res["num_slices"] - 1, 0, key="custom_slider")
                        else:
                            nav_idx = 0
                        st.image(res["display_slices"][nav_idx], caption=f"Slice {res['slice_indices'][nav_idx]}", width=450)

                except SequenceValidationError as e:
                    st.error(f"❌ **Sequence Validation Error:** {str(e)}")

    # -------------------------------------------------------------
    # 6. EXPLAINABILITY
    # -------------------------------------------------------------
    elif nav == "💡 Explainability":
        st.title("💡 Explainability & Technical Boundaries")
        st.caption("Visual Spatial Saliency vs. Temporal Sequence Attribution")

        st.markdown("""
        ### Understanding Explainable AI in PulmoVision
        In deep learning for medical imaging, explainability methods must be technically rigorous and medically honest:
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. Spatial Explainability (Grad-CAM)")
            st.write("""
            - **Scope:** Evaluates 2D slice feature maps from convolutional layers (`features[-1]`).
            - **Mechanism:** Gradients of the target class are pooled across channels to generate spatial activation heatmaps.
            - **Application:** Identifies localized lung parenchyma regions driving slice-level convolutional features.
            """)
            if os.path.exists("reports/figures/gradcam_sample.png"):
                st.image("reports/figures/gradcam_sample.png", caption="Grad-CAM Sample Overlay", width=400)

        with col2:
            st.markdown("#### 2. Temporal Explainability (Attention Weights)")
            st.write("""
            - **Scope:** Evaluates axial slice sequence aggregation along the craniocaudal z-axis.
            - **Mechanism:** In Experiment C, additive self-attention outputs a scalar weight $\\alpha_t \\in [0, 1]$ per slice.
            - **Application:** Quantifies relative mathematical contribution of individual slices to the sequence representation.
            """)

        st.warning("""
        ⚠️ **CRITICAL TECHNICAL CAVEAT:** Attention weights and Grad-CAM saliency heatmaps represent mathematical 
        gradient flows and feature sensitivity. They do **NOT** represent certified radiologist attention, 
        clinical diagnostic reasoning, or definitive anatomical localization of pathology.
        """)

    # -------------------------------------------------------------
    # 7. RESEARCH DOCUMENTATION
    # -------------------------------------------------------------
    elif nav == "📖 Research Documentation":
        st.title("📖 Stage 2 Comprehensive Research Report")
        st.caption("PULMOVISION_STAGE_2_RESEARCH_REPORT.md (Authoritative Verified Record)")

        report_md_path = "reports/research/PULMOVISION_STAGE_2_RESEARCH_REPORT.md"
        if os.path.exists(report_md_path):
            with open(report_md_path) as f:
                content = f.read()
            st.markdown(content)
        else:
            st.error("Research report file not found on disk.")


if __name__ == "__main__":
    main()
