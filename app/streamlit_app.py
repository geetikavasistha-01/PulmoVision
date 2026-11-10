"""
PulmoVision — AI-Powered 2.5D Medical Imaging Research Application
Interactive Streamlit Dashboard for Thoracic CT Analysis & Explainable AI.
"""

import os
import streamlit as st
import numpy as np
from PIL import Image
from src.inference.predictor import PulmoPredictor

# Page Configuration
st.set_page_config(
    page_title="PulmoVision — 2.5D CT Research App",
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
        padding: 15px;
        border-left: 5px solid #0066cc;
        margin-bottom: 10px;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """
    Cached model loader: Loads model weights once per application process.
    """
    ckpt = "models/best_model.pth"
    return PulmoPredictor(checkpoint_path=ckpt)


def main():
    st.sidebar.title("🫁 PulmoVision")
    st.sidebar.caption("2.5D Deep Learning Research Prototype")

    nav = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔬 CT Analysis", "📊 Model Information", "📖 Methodology", "ℹ️ About"]
    )

    # -------------------------------------------------------------
    # PAGE 1: HOME
    # -------------------------------------------------------------
    if nav == "🏠 Home":
        st.title("🫁 PulmoVision — Medical AI Research Prototype")
        st.markdown("""
        <div class="disclaimer-box">
            ⚠️ <b>RESEARCH ONLY DISCLAIMER:</b> This application is an educational and research prototype. 
            It is <b>NOT</b> a diagnostic medical device, does not provide clinical advice, and dataset labels 
            must not be treated as confirmed medical diagnoses.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("What is PulmoVision?")
            st.write("""
            PulmoVision analyzes chest Computed Tomography (CT) scans using **2.5D deep learning**.
            Instead of analyzing individual 2D slices in isolation or requiring heavy 3D volumetric convolutions,
            it stacks adjacent anatomical slices $[Z-1, Z, Z+1]$ as separate input channels.
            
            This provides the convolutional neural network with **through-plane volumetric depth** while running
            fast enough for real-time interactive inference on consumer hardware.
            """)

            st.markdown("### Core Capabilities")
            st.markdown("""
            - 🔹 **16-bit CT Dynamic Range:** Handles raw high-contrast medical scans without quality degradation.
            - 🔹 **2.5D Slice Stacking:** Captures continuous lesion extent across adjacent CT cross-sections.
            - 🔹 **Explainable AI (Grad-CAM):** Pinpoints the precise lung parenchyma regions influencing predictions.
            - 🔹 **Zero-Leakage Integrity:** Patient-level split isolation prevents clinical data leakage.
            """)

        with col2:
            st.image("docs/data_and_pipeline_overview/pipeline_architecture_flow.png", caption="PulmoVision 2.5D Pipeline Flow")

    # -------------------------------------------------------------
    # PAGE 2: CT ANALYSIS
    # -------------------------------------------------------------
    elif nav == "🔬 CT Analysis":
        st.title("🔬 2.5D CT Scan Analysis & Grad-CAM")
        st.markdown("""
        <div class="disclaimer-box">
            Upload CT scan slices or use the pre-loaded audit sample below to inspect predictions and Grad-CAM attention maps.
        </div>
        """, unsafe_allow_html=True)

        predictor = load_predictor()

        sample_tiff = "data/interim/sample_audit/TrainValidation/137covid_patient100_SR_2_IM00028.tif"
        has_sample = os.path.exists(sample_tiff)

        tab1, tab2 = st.tabs(["⚡ One-Click Sample Test", "📤 Upload Custom CT Slices"])

        with tab1:
            if has_sample:
                st.write(f"**Loaded Sample:** `137covid_patient100_SR_2_IM00028.tif` (16-bit TIFF, $512 \\times 512$)")
                if st.button("🚀 Run 2.5D Analysis on Sample Scan", type="primary"):
                    with st.spinner("Processing 2.5D slice stack and generating Grad-CAM..."):
                        img = Image.open(sample_tiff)
                        arr = np.array(img, dtype=np.float32)
                        result = predictor.predict(slice_z=arr)

                    # Display Metrics
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric("Predicted Category", result["prediction"])
                    with res_col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                    with res_col3:
                        st.metric("Latency", f"{result['latency_ms']:.1f} ms", f"Device: {result['device']}")

                    st.divider()

                    # Display Images
                    img_col1, img_col2, img_col3 = st.columns(3)
                    with img_col1:
                        st.image(result["display_slice"], caption="Windowed Input Slice (Lung Window)", use_container_width=True)
                    with img_col2:
                        st.image(result["heatmap"], caption="Grad-CAM Activation Map", use_container_width=True, clamp=True)
                    with img_col3:
                        st.image(result["overlay"], caption="Grad-CAM Colormap Overlay", use_container_width=True)
            else:
                st.info("No pre-cached sample found on disk. Please upload files in the next tab.")

        with tab2:
            st.subheader("Upload 3 Adjacent CT Slices (2.5D Triplet)")
            st.caption("For genuine 2.5D spatial context, upload Slice Z-1, Slice Z, and Slice Z+1 from the same scan series.")

            from src.inference.upload_validator import CTUploadValidator, UploadValidationError, InferenceMode

            up_prev = st.file_uploader("1. Previous Slice (Z - 1) [Optional fallback]", type=["tif", "tiff", "png", "jpg"])
            up_curr = st.file_uploader("2. Target Slice (Z) [Required]", type=["tif", "tiff", "png", "jpg"])
            up_next = st.file_uploader("3. Next Slice (Z + 1) [Optional fallback]", type=["tif", "tiff", "png", "jpg"])

            if up_curr is not None:
                # Validate Provenance and Filenames
                try:
                    mode, meta = CTUploadValidator.validate_triplet_provenance(
                        fname_curr=up_curr.name,
                        fname_prev=up_prev.name if up_prev else None,
                        fname_next=up_next.name if up_next else None,
                    )
                    
                    if mode == InferenceMode.FULL_25D:
                        st.info(f"✅ **Provenance Verified:** Patient `{meta['patient_id']}`, Series `{meta['series_id']}` (Indices: {meta['indices']}) — **{mode.value}**")
                    else:
                        st.warning(f"⚠️ **Single-Slice Fallback Active:** Clamping target slice boundaries to simulate 3 channels.")

                    if st.button("Analyze Uploaded Scans"):
                        with st.spinner("Processing scans..."):
                            curr_arr = np.array(Image.open(up_curr), dtype=np.float32)
                            prev_arr = np.array(Image.open(up_prev), dtype=np.float32) if up_prev else curr_arr
                            next_arr = np.array(Image.open(up_next), dtype=np.float32) if up_next else curr_arr

                            res = predictor.predict(slice_z=curr_arr, slice_prev=prev_arr, slice_next=next_arr)

                        st.success(f"Analysis Complete: **{res['prediction']}** ({res['confidence']:.1%} confidence)")
                        u_col1, u_col2, u_col3 = st.columns(3)
                        with u_col1:
                            st.image(res["display_slice"], caption="Preprocessed Slice", use_container_width=True)
                        with u_col2:
                            st.image(res["heatmap"], caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
                        with u_col3:
                            st.image(res["overlay"], caption="Attention Overlay", use_container_width=True)

                except UploadValidationError as e:
                    st.error(f"❌ **Validation Rejected:** {str(e)}")

    # -------------------------------------------------------------
    # PAGE 3: MODEL INFORMATION
    # -------------------------------------------------------------
    elif nav == "📊 Model Information":
        st.title("📊 Model Specifications & Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Architecture Specifications")
            st.table({
                "Parameter": ["Model Backbone", "Input Dimensions", "Input Channels", "Number of Classes", "Total Parameters", "Device Acceleration"],
                "Value": ["EfficientNet-B0", "224 x 224 pixels", "3 (Neighbouring slices)", "2 (Normal, COVID-19)", "4,010,110 (~4.0M)", "Apple Silicon Metal (MPS) / CPU"]
            })

        with col2:
            st.markdown("### Validation Performance")
            st.table({
                "Evaluation Metric": ["Macro F1 Score", "Balanced Accuracy", "Average Inference Latency", "Class-Weighted Loss"],
                "Score": ["0.5211 (Baseline)", "59.4%", "~12 ms / sample", "CrossEntropyLoss(weight=[0.67, 2.0])"]
            })

        st.markdown("### Confusion Matrix")
        if os.path.exists("reports/figures/confusion_matrix.png"):
            st.image("reports/figures/confusion_matrix.png", width=500)

    # -------------------------------------------------------------
    # PAGE 4: METHODOLOGY
    # -------------------------------------------------------------
    elif nav == "📖 Methodology":
        st.title("📖 Research Methodology & Data Integrity")
        
        st.markdown("### 1. Cohort & Dataset Provenance")
        st.write("""
        Scans are derived from the **COVID-CTset** cohort collected at Negin Medical Center (Sari, Iran) 
        using a Siemens SOMATOM Scope scanner. Ground truth labels were verified through institutional RT-PCR 
        laboratory tests paired with expert radiologist consensus.
        """)

        if os.path.exists("docs/data_and_pipeline_overview/patient_distribution.png"):
            st.image("docs/data_and_pipeline_overview/patient_distribution.png", width=600)

        st.markdown("### 2. 16-Bit Radiodensity & Windowing")
        st.write("""
        Raw CT scans are captured in 16-bit TIFF format ($[0, \approx 2200]$ intensities). 
        To highlight pulmonary parenchyma and eliminate scanner bed artifacts, we apply a domain-specific
        lung window $[200, 1800]$ followed by dynamic range normalization to $[0.0, 1.0]$.
        """)

        if os.path.exists("docs/data_and_pipeline_overview/ct_intensity_and_slice_analysis.png"):
            st.image("docs/data_and_pipeline_overview/ct_intensity_and_slice_analysis.png", use_container_width=True)

    # -------------------------------------------------------------
    # PAGE 5: ABOUT
    # -------------------------------------------------------------
    elif nav == "ℹ️ About":
        st.title("ℹ️ About PulmoVision")
        st.write("""
        **PulmoVision** is an open-source educational deep learning research project built to demonstrate
        how to apply computer vision and transfer learning to multi-slice medical imaging.
        """)

        st.markdown("### Technology Stack")
        st.markdown("""
        - **Deep Learning Framework:** PyTorch & Torchvision
        - **Model Backbones:** EfficientNet-B0, ResNet50, DenseNet121, Custom CNN
        - **Medical Imaging:** Albumentations, OpenCV, Pillow
        - **Explainability:** Grad-CAM Hook Architecture
        - **User Interface:** Streamlit (Cached inference service)
        - **Source Control:** GitHub ([PulmoVision Repository](https://github.com/geetikavasistha-01/PulmoVision))
        """)

        st.info("Developed as part of the PulmoVision Advanced Agentic Medical AI Course.")


if __name__ == "__main__":
    main()

