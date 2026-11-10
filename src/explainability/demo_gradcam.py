"""
PulmoVision — Grad-CAM Demonstration Script
Loads best model checkpoint, runs Grad-CAM on physical sample CT slice,
and outputs 3-panel visualization to reports/figures/gradcam_sample.png.
"""

import os
import torch
import numpy as np
from PIL import Image
from src.models.model_factory import get_model
from src.training.checkpoint import load_checkpoint
from src.data.preprocessing import CTPreprocessor
from src.explainability.gradcam import GradCAM, render_gradcam_figure


def run_gradcam_demo(
    sample_path: str = "data/interim/sample_audit/TrainValidation/137covid_patient100_SR_2_IM00028.tif",
    checkpoint_path: str = "models/best_model.pth"
):
    print(f"\n[Grad-CAM] Loading trained model from: {checkpoint_path}")
    model = get_model("efficientnet_b0", in_channels=3, num_classes=2, pretrained=False)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # Preprocess CT slice
    preprocessor = CTPreprocessor(target_size=(224, 224), is_training=False)
    if os.path.exists(sample_path):
        slice_t = preprocessor(sample_path)  # [1, 224, 224]
        raw_img = Image.open(sample_path)
        raw_arr = np.array(raw_img, dtype=np.float32)
        windowed_2d = preprocessor.apply_windowing(raw_arr)
        # Resize original for visualization
        import cv2
        display_slice = cv2.resize(windowed_2d, (224, 224))
    else:
        slice_t = torch.randn(1, 224, 224)
        display_slice = slice_t[0].numpy()

    # Create 3-channel 2.5D input by stacking (simulated with slice replication for single sample demo)
    input_tensor = torch.cat([slice_t, slice_t, slice_t], dim=0).unsqueeze(0)  # [1, 3, 224, 224]

    # Initialize Grad-CAM
    cam = GradCAM(model=model)
    heatmap, pred_idx, confidence = cam.generate_heatmap(input_tensor)

    label_names = ["Normal", "COVID-19"]
    pred_label = label_names[pred_idx]
    print(f"[Grad-CAM] Predicted: {pred_label} (Confidence: {confidence:.2%})")

    # Generate overlay
    overlay = GradCAM.overlay_heatmap(heatmap, display_slice)

    # Render figure
    out_path = "reports/figures/gradcam_sample.png"
    render_gradcam_figure(
        original_slice=display_slice,
        heatmap=heatmap,
        overlay=overlay,
        pred_label=pred_label,
        confidence=confidence,
        output_path=out_path
    )


if __name__ == "__main__":
    run_gradcam_demo()

