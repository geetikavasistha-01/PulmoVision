"""
PulmoVision — Medical Image Preprocessing Pipeline
Handles 16-bit TIFF CT loading, windowing, safe data augmentations,
resizing, and PyTorch tensor conversion.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
from PIL import Image
import albumentations as A


class CTPreprocessor:
    """
    Standard preprocessor for thoracic CT slices in COVID-CTset.
    
    Attributes:
        target_size: (H, W) tuple for resizing (default: 224x224).
        window_range: (min_intensity, max_intensity) clipping interval.
        is_training: If True, applies safe anatomical augmentations.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        window_range: Tuple[float, float] = (200.0, 1800.0),
        is_training: bool = False,
    ):
        self.target_size = target_size
        self.window_range = window_range
        self.is_training = is_training
        
        # Safe CT Augmentation Pipeline (Albumentations)
        if self.is_training:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=7, border_mode=0, p=0.5), # 0 = constant background fill
                A.RandomScale(scale_limit=0.05, p=0.3),
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
            ])

    def load_slice(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Loads a 16-bit TIFF image from file path or validates existing numpy array.
        Returns float32 numpy array.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input)
            arr = np.array(img, dtype=np.float32)
        elif isinstance(image_input, np.ndarray):
            arr = image_input.astype(np.float32)
        else:
            raise TypeError(f"Expected str path or np.ndarray, got {type(image_input)}")
            
        return arr

    def apply_windowing(self, arr: np.ndarray) -> np.ndarray:
        """
        Clips radiodensity to lung parenchyma contrast window and normalizes to [0.0, 1.0].
        """
        w_min, w_max = self.window_range
        clipped = np.clip(arr, w_min, w_max)
        normalized = (clipped - w_min) / (w_max - w_min + 1e-7)
        return normalized.astype(np.float32)

    def __call__(self, image_input: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Full pipeline: Load -> Window -> Augment/Resize -> PyTorch Tensor [1, H, W].
        """
        # 1. Load raw 16-bit data
        raw_arr = self.load_slice(image_input)
        
        # 2. Window and normalize to [0.0, 1.0]
        windowed = self.apply_windowing(raw_arr)
        
        # 3. Apply spatial augmentations and resize
        transformed = self.transform(image=windowed)["image"]
        
        # 4. Convert to PyTorch Tensor: [H, W] -> [1, H, W]
        tensor = torch.from_numpy(transformed).unsqueeze(0).float()
        
        return tensor


def compare_normalizations(raw_arr: np.ndarray) -> dict:
    """
    Compares Min-Max, Percentile Clipping, and Lung Windowing.
    Useful for educational comparison and verification.
    """
    # 1. Linear Min-Max (Naive)
    linear = (raw_arr - raw_arr.min()) / (raw_arr.max() - raw_arr.min() + 1e-7)
    
    # 2. Percentile Clipping (Robust scaling: 1st to 99th percentile)
    p1, p99 = np.percentile(raw_arr, (1, 99))
    p_clipped = np.clip(raw_arr, p1, p99)
    percentile = (p_clipped - p1) / (p99 - p1 + 1e-7)
    
    # 3. Domain-Specific Lung Window [200, 1800]
    w_min, w_max = 200.0, 1800.0
    w_clipped = np.clip(raw_arr, w_min, w_max)
    windowed = (w_clipped - w_min) / (w_max - w_min + 1e-7)
    
    return {
        "linear_min_max": linear,
        "percentile_1_99": percentile,
        "lung_window": windowed,
    }
