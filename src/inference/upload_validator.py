"""
PulmoVision — Upload Validation & Provenance Verification Engine
Validates uploaded CT slice files, verifies patient & series consistency,
and prevents invalid multi-patient slice combinations.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pathlib import Path
from src.data.slice_stacking import CTSliceStacker


class InferenceMode(str, Enum):
    FULL_25D = "Full 2.5D (3 Valid Adjacent Slices)"
    ISOLATED_FALLBACK = "Isolated Slice Fallback (Single Slice Clamped)"


class UploadValidationError(ValueError):
    """Raised when an uploaded file or group of slices fails medical/provenance validation."""
    pass


class CTUploadValidator:
    ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    @classmethod
    def validate_single_file(cls, filename: str) -> Dict[str, Any]:
        """
        Validates extension and parses filename metadata.
        """
        ext = Path(filename).suffix.lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise UploadValidationError(
                f"Unsupported image format: '{ext}'. Allowed formats: {sorted(list(cls.ALLOWED_EXTENSIONS))}"
            )

        info = CTSliceStacker.parse_filename(filename)
        return info

    @classmethod
    def validate_triplet_provenance(
        cls,
        fname_curr: str,
        fname_prev: Optional[str] = None,
        fname_next: Optional[str] = None,
    ) -> Tuple[InferenceMode, Dict[str, Any]]:
        """
        Validates that Slice Z, Slice Z-1, and Slice Z+1 share the exact same
        patient ID and series ID, and possess valid sequential order.
        """
        info_curr = cls.validate_single_file(fname_curr)

        # Case 1: Isolated single slice upload
        if fname_prev is None and fname_next is None:
            return InferenceMode.ISOLATED_FALLBACK, {
                "patient_id": info_curr["patient_id"],
                "series_id": info_curr["series_id"],
                "slice_idx": info_curr["slice_idx"],
                "status": "Single slice fallback; clamped boundaries applied.",
            }

        # Case 2: Multi-slice upload
        info_prev = cls.validate_single_file(fname_prev) if fname_prev else info_curr
        info_next = cls.validate_single_file(fname_next) if fname_next else info_curr

        p_curr = info_curr["patient_id"]
        p_prev = info_prev["patient_id"]
        p_next = info_next["patient_id"]

        # 1. Patient ID Consistency
        if not (p_curr == p_prev == p_next):
            raise UploadValidationError(
                f"Cross-Patient Mismatch Violation! Target slice belongs to '{p_curr}', "
                f"previous slice to '{p_prev}', and next slice to '{p_next}'. "
                f"Slices from different individuals must NEVER be combined in 2.5D models."
            )

        # 2. Scan Series Consistency
        s_curr = info_curr["series_id"]
        s_prev = info_prev["series_id"]
        s_next = info_next["series_id"]
        if not (s_curr == s_prev == s_next):
            raise UploadValidationError(
                f"Series Mismatch Violation! Target slice belongs to '{s_curr}', "
                f"previous slice to '{s_prev}', and next slice to '{s_next}'. "
                f"Slices from different scan protocols cannot be combined."
            )

        # 3. Slice Index Ordering Check
        idx_curr = info_curr["slice_idx"]
        idx_prev = info_prev["slice_idx"]
        idx_next = info_next["slice_idx"]

        if idx_prev > idx_curr or idx_next < idx_curr:
            raise UploadValidationError(
                f"Slice Ordering Inversion! Slices must be ordered sequentially along Z-axis. "
                f"Got indices: prev={idx_prev}, target={idx_curr}, next={idx_next}."
            )

        mode = InferenceMode.FULL_25D if (fname_prev and fname_next) else InferenceMode.ISOLATED_FALLBACK

        return mode, {
            "patient_id": p_curr,
            "series_id": s_curr,
            "indices": (idx_prev, idx_curr, idx_next),
            "status": "Provenance verified successfully.",
        }
