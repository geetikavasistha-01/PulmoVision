"""
PulmoVision — Stage 2 CT Sequence Inference Engine
Encapsulates end-to-end inference for patient-level craniocaudal sequence models:
Image Sequence -> Validation -> Preprocessing -> EfficientNet-B0 Features -> BiLSTM -> Prediction.
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.cnn_bilstm import PulmoBiLSTMMeanPooling
from src.features.feature_extractor import PretrainedFeatureExtractor
from src.data.preprocessing import CTPreprocessor
from src.inference.sequence_validator import CTSequenceValidator, SequenceValidationError


class SequencePredictor:
    """
    End-to-end patient-level inference service using the Stage 2 BiLSTM model.
    """
    CLASS_NAMES = ["Normal", "COVID-19"]

    def __init__(
        self,
        checkpoint_path: str = "models/stage2_bilstm_deployment.pth",
        device: Optional[str] = None,
        threshold: float = 0.50,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.feature_extractor = PretrainedFeatureExtractor(device=self.device)

        # Initialize BiLSTM architecture
        self.model = PulmoBiLSTMMeanPooling(
            input_dim=1280,
            hidden_dim=128,
            mlp_dim=64,
            num_classes=2,
            dropout=0.3,
        ).to(self.device)

        self.checkpoint_metadata = {}
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
                self.checkpoint_metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
            else:
                self.model.load_state_dict(ckpt)
            print(f"[SequencePredictor] Loaded deployment weights from {checkpoint_path}")
        else:
            print(f"[SequencePredictor Warning] Checkpoint {checkpoint_path} not found. Running with initialized weights.")

        self.model.eval()

    def load_image_as_array(self, image_input: Any) -> np.ndarray:
        """Loads uploaded file or path into 2D numpy array."""
        if isinstance(image_input, np.ndarray):
            arr = image_input
        elif isinstance(image_input, str):
            if image_input.endswith(".npy"):
                arr = np.load(image_input)
            else:
                img = Image.open(image_input)
                arr = np.array(img)
        elif hasattr(image_input, "read"):
            # UploadedFile in Streamlit or file-like object
            image_input.seek(0)
            if hasattr(image_input, "name") and image_input.name.endswith(".npy"):
                arr = np.load(image_input)
            else:
                img = Image.open(image_input)
                arr = np.array(img)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if arr.ndim == 3:
            # Grayscale channel extraction if multi-channel
            arr = arr[:, :, 0]

        return arr.astype(np.float32)

    def predict_sequence(
        self,
        slice_inputs: List[Any],
        filenames: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on an ordered CT slice collection.

        Args:
            slice_inputs: List of file paths, UploadedFile objects, or numpy arrays.
            filenames: Optional list of filenames for provenance validation.

        Returns:
            Dictionary containing prediction, probability, latency, and clinical disclaimers.
        """
        t_start = time.time()

        # 1. Validate sequence provenance and ordering
        ordered_inputs, val_meta = CTSequenceValidator.validate_and_order_sequence(
            slice_inputs, filenames=filenames
        )
        num_slices = val_meta["num_slices"]

        # 2. Preprocess slices and extract embeddings
        slice_embeddings = []
        display_slices = []
        for inp in ordered_inputs:
            raw_arr = self.load_image_as_array(inp)
            t_3ch = self.feature_extractor.preprocess_slice(raw_arr) # [3, 224, 224]
            emb = self.feature_extractor.extract_slice_embedding(t_3ch) # [1, 1280]
            slice_embeddings.append(emb.squeeze(0)) # [1280]

            # Create [0, 1] normalized slice for UI rendering
            clipped = np.clip(raw_arr, 200, 1800)
            disp = (clipped - 200.0) / 1600.0
            display_slices.append(disp)

        # 3. Stack into sequence embedding tensor: [1, T, 1280]
        seq_embeddings = torch.stack(slice_embeddings, dim=0).unsqueeze(0).to(self.device) # [1, T, 1280]
        lengths = torch.tensor([num_slices], dtype=torch.long).to(self.device)
        mask = torch.ones(1, num_slices, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(seq_embeddings, mask=mask, lengths=lengths) # [1, 2]
            probs = F.softmax(logits, dim=1).cpu().numpy()[0] # [2]

        covid_prob = float(probs[1])
        normal_prob = float(probs[0])
        pred_class_idx = 1 if covid_prob >= self.threshold else 0
        pred_label = self.CLASS_NAMES[pred_class_idx]
        confidence = float(probs[pred_class_idx])

        elapsed_ms = (time.time() - t_start) * 1000.0

        return {
            "predicted_class": pred_label,
            "predicted_index": pred_class_idx,
            "covid_19_probability": covid_prob,
            "normal_probability": normal_prob,
            "confidence": confidence,
            "threshold_applied": self.threshold,
            "num_slices": num_slices,
            "slice_indices": val_meta["slice_indices"],
            "patient_id": val_meta["patient_id"],
            "series_id": val_meta["series_id"],
            "is_single_slice": val_meta["is_single_slice"],
            "warning": val_meta["warning"],
            "inference_latency_ms": elapsed_ms,
            "per_slice_latency_ms": elapsed_ms / max(num_slices, 1),
            "display_slices": display_slices,
            "disclaimer": (
                "RESEARCH PROTOTYPE ONLY: This model prediction is generated by an experimental deep learning "
                "prototype for educational and investigational purposes. It is NOT a clinical diagnosis, "
                "medical advice, or a certified medical device."
            ),
        }
