"""
PulmoVision — Standalone Inference Service
Encapsulates model loading, 2.5D input preprocessing, prediction,
latency benchmarking, and Grad-CAM explainability.
"""

import os
import time
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.models.model_factory import get_model
from src.training.checkpoint import load_checkpoint
from src.data.preprocessing import CTPreprocessor
from src.explainability.gradcam import GradCAM


class PulmoPredictor:
    """
    Inference engine optimized for Streamlit and production serving.
    """
    CLASS_NAMES = ["Normal", "COVID-19"]

    def __init__(self, checkpoint_path: str = "models/best_model.pth", device: Optional[str] = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = get_model("efficientnet_b0", in_channels=3, num_classes=2, pretrained=False).to(self.device)
        
        # Load weights if available
        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, self.model, device=str(self.device))
        else:
            print(f"[Predictor Warning] Checkpoint {checkpoint_path} not found. Running with initialized weights.")
            
        self.model.eval()
        self.preprocessor = CTPreprocessor(target_size=(224, 224), is_training=False)
        self.gradcam = GradCAM(model=self.model)

    def prepare_25d_tensor(
        self,
        slice_z: np.ndarray,
        slice_prev: Optional[np.ndarray] = None,
        slice_next: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Processes 3 neighbouring slices or clamps for single slice fallback.
        Returns:
            stacked_tensor: [1, 3, 224, 224] on compute device
            display_slice: [224, 224] windowed 2D slice for visual rendering
        """
        # If neighbours are missing, fallback to edge replication (clamp rule)
        if slice_prev is None:
            slice_prev = slice_z
        if slice_next is None:
            slice_next = slice_z

        t_prev = self.preprocessor(slice_prev)
        t_curr = self.preprocessor(slice_z)
        t_next = self.preprocessor(slice_next)

        # Stack into [1, 3, 224, 224]
        stacked = torch.cat([t_prev, t_curr, t_next], dim=0).unsqueeze(0).to(self.device)

        # Preprocess target slice for display
        w_curr = self.preprocessor.apply_windowing(slice_z.astype(np.float32))
        import cv2
        display_slice = cv2.resize(w_curr, (224, 224))

        return stacked, display_slice

    def predict(
        self,
        slice_z: np.ndarray,
        slice_prev: Optional[np.ndarray] = None,
        slice_next: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference and returns predictions + Grad-CAM overlay.
        """
        start_time = time.time()
        tensor, display_slice = self.prepare_25d_tensor(slice_z, slice_prev, slice_next)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.CLASS_NAMES[pred_idx]

        # Generate Grad-CAM (requires grad)
        with torch.enable_grad():
            heatmap, _, _ = self.gradcam.generate_heatmap(tensor, target_class=pred_idx)
            overlay = GradCAM.overlay_heatmap(heatmap, display_slice)

        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "prediction": pred_label,
            "pred_idx": pred_idx,
            "confidence": confidence,
            "probabilities": {
                "Normal": float(probs[0]),
                "COVID-19": float(probs[1]),
            },
            "display_slice": display_slice,
            "heatmap": heatmap,
            "overlay": overlay,
            "latency_ms": latency_ms,
            "device": str(self.device),
        }
