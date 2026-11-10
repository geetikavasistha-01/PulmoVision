"""
PulmoVision — Explainable AI: Grad-CAM Implementation
Generates Gradient-Weighted Class Activation Maps to visualize the anatomical
regions driving model classification decisions.
"""

import os
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    """
    Hook-based Grad-CAM generator for convolutional neural networks.
    
    Args:
        model: Trained PyTorch classification model.
        target_layer: The convolutional layer to inspect (defaults to model.features[-1] in EfficientNet).
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Auto-detect target layer for EfficientNet or ResNet
            if hasattr(model, "features"):
                self.target_layer = model.features[-1]
            elif hasattr(model, "model") and hasattr(model.model, "features"):
                self.target_layer = model.model.features[-1]
            elif hasattr(model, "model") and hasattr(model.model, "layer4"):
                self.target_layer = model.model.layer4[-1]
            else:
                # Fallback to last conv block in custom cnn
                self.target_layer = list(model.modules())[-5]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates 2D normalized Grad-CAM heatmap [0.0, 1.0].
        
        Args:
            input_tensor: Shape [1, C, H, W]
            target_class: Integer target class index (defaults to argmax prediction).
        
        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0.0, 1.0]
            predicted_class: int
            confidence: float (softmax probability)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        pred_class = int(output.argmax(dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Backward pass for the target class logit
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Global average pooling of gradients: [1, Channels, 1, 1]
        pooled_grads = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weight activations by pooled gradients
        cam = torch.sum(pooled_grads * self.activations, dim=1).squeeze()
        cam = F.relu(cam)  # Apply ReLU: only positive contributions matter

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        heatmap = cam.cpu().numpy()

        # Resize heatmap to match input spatial dimensions (H, W)
        _, _, H, W = input_tensor.shape
        heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

        return heatmap_resized, pred_class, confidence

    @staticmethod
    def overlay_heatmap(
        heatmap: np.ndarray,
        original_slice: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Superimposes the jet colormap heatmap onto the grayscale CT slice.
        
        Returns:
            RGB overlay image (uint8, [H, W, 3])
        """
        # Ensure slice is in uint8 [0, 255]
        if original_slice.max() <= 1.0:
            slice_uint8 = (original_slice * 255).astype(np.uint8)
        else:
            slice_uint8 = np.clip(original_slice, 0, 255).astype(np.uint8)

        # Convert grayscale to 3-channel BGR
        if len(slice_uint8.shape) == 2:
            slice_bgr = cv2.cvtColor(slice_uint8, cv2.COLOR_GRAY2BGR)
        else:
            slice_bgr = slice_uint8

        # Colorize heatmap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend
        overlay = cv2.addWeighted(heatmap_color, alpha, slice_bgr, 1 - alpha, 0)
        # Convert BGR to RGB for matplotlib/Streamlit
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return overlay_rgb


def render_gradcam_figure(
    original_slice: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    pred_label: str,
    confidence: float,
    output_path: str = "reports/figures/gradcam_sample.png"
):
    """
    Saves a 3-panel figure: Original CT Slice | Grad-CAM Heatmap | Overlay.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    ax1.imshow(original_slice, cmap="bone")
    ax1.set_title("Input CT Slice (Windowed)", fontsize=11, fontweight="bold")
    ax1.axis("off")

    im2 = ax2.imshow(heatmap, cmap="jet")
    ax2.set_title("Grad-CAM Activation Map", fontsize=11, fontweight="bold")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3.imshow(overlay)
    ax3.set_title(f"Overlay ({pred_label}, Conf: {confidence:.1%})", fontsize=11, fontweight="bold")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Grad-CAM] Saved explainability visualization to: {output_path}")
