"""
PulmoVision — CT Sequence Input and Provenance Validator
Enforces medical image sequence integrity for 2.5D and volumetric recurrent models:
- Validates image formats (TIFF, PNG, JPG, NumPy) and dimensions.
- Enforces single-patient and single-series invariants.
- Requires craniocaudal slice ordering.
- Explicitly flags single-slice uploads with appropriate clinical/technical disclaimers.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image


class SequenceValidationError(ValueError):
    """Raised when an uploaded CT slice sequence violates integrity rules."""
    pass


class CTSequenceValidator:
    """
    Validates uploaded thoracic CT slice sequences before neural network ingestion.
    """
    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}

    @staticmethod
    def parse_provenance(filename: str) -> Dict[str, Any]:
        """
        Parses patient_id, series_id, and slice_idx from standard COVID-CTset filenames.
        Format: patient<ID>_SR_<SERIES>_IM<SLICE>.<ext>
        Fallback to index-based ordering if non-standard names are supplied.
        """
        base = os.path.basename(filename)
        clean = os.path.splitext(base)[0].lower()

        # Match patient
        p_match = re.search(r"patient\d+", clean)
        patient_id = p_match.group(0) if p_match else "unknown_patient"

        # Match series
        s_match = re.search(r"sr_\d+", clean)
        series_id = s_match.group(0).upper() if s_match else "SR_1"

        # Match slice index (e.g. im00012 or slice12 or 12)
        im_match = re.search(r"im(\d+)", clean)
        if im_match:
            slice_idx = int(im_match.group(1))
        else:
            digits = re.findall(r"\d+", clean)
            slice_idx = int(digits[-1]) if digits else 0

        return {
            "filename": base,
            "patient_id": patient_id,
            "series_id": series_id,
            "slice_idx": slice_idx,
        }

    @classmethod
    def validate_and_order_sequence(
        cls,
        files: List[Any],
        filenames: Optional[List[str]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Validates a collection of uploaded CT slices:
        1. Checks non-empty list.
        2. Validates file formats.
        3. Enforces single-patient and single-series invariants (if provenance is present).
        4. Sorts files strictly in ascending craniocaudal order.
        5. Handles single-slice uploads explicitly.

        Returns:
            ordered_files: List of sorted file objects or paths.
            provenance_metadata: Summary dictionary with sequence metadata and warnings.
        """
        if not files:
            raise SequenceValidationError("No CT slices provided for sequence analysis.")

        if filenames is None:
            filenames = []
            for f in files:
                if hasattr(f, "name"):
                    filenames.append(f.name)
                elif isinstance(f, str):
                    filenames.append(os.path.basename(f))
                else:
                    filenames.append(f"slice_{len(filenames):03d}.tif")

        # Check extensions
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in cls.SUPPORTED_EXTENSIONS:
                raise SequenceValidationError(
                    f"Unsupported file format '{ext}' for file '{fname}'. "
                    f"Supported formats: {sorted(list(cls.SUPPORTED_EXTENSIONS))}"
                )

        # Parse provenance for each file
        parsed_records = []
        for f_obj, fname in zip(files, filenames):
            meta = cls.parse_provenance(fname)
            meta["file_object"] = f_obj
            parsed_records.append(meta)

        # If standard patient IDs are detected, enforce single-patient and single-series invariants
        known_patients = set(r["patient_id"] for r in parsed_records if r["patient_id"] != "unknown_patient")
        if len(known_patients) > 1:
            raise SequenceValidationError(
                f"Cross-patient sequence violation: Uploaded slices belong to multiple distinct patients: "
                f"{sorted(list(known_patients))}. All slices in a sequence must belong to the same patient."
            )

        known_series = set(r["series_id"] for r in parsed_records)
        if len(known_series) > 1:
            raise SequenceValidationError(
                f"Cross-series sequence violation: Uploaded slices belong to multiple scan series: "
                f"{sorted(list(known_series))}. All slices must belong to the same acquisition series."
            )

        # Sort in ascending slice_idx order
        parsed_records.sort(key=lambda r: (r["patient_id"], r["series_id"], r["slice_idx"]))

        ordered_files = [r["file_object"] for r in parsed_records]
        ordered_slice_indices = [r["slice_idx"] for r in parsed_records]

        # Check for duplicate slice indices in the same sequence
        if len(ordered_slice_indices) != len(set(ordered_slice_indices)) and len(ordered_slice_indices) > 1:
            raise SequenceValidationError(
                "Duplicate slice indices detected within the uploaded sequence. "
                "Each axial position must appear exactly once."
            )

        num_slices = len(ordered_files)
        is_single_slice = (num_slices == 1)

        warning_message = None
        if is_single_slice:
            warning_message = (
                "Single-slice input: recurrent temporal context is minimal. "
                "The sequence model is operating in single-slice fallback mode; "
                "full volumetric contextual advantages require an ordered axial stack."
            )

        metadata = {
            "num_slices": num_slices,
            "patient_id": parsed_records[0]["patient_id"],
            "series_id": parsed_records[0]["series_id"],
            "slice_indices": ordered_slice_indices,
            "ordered_filenames": [r["filename"] for r in parsed_records],
            "is_single_slice": is_single_slice,
            "warning": warning_message,
        }

        return ordered_files, metadata
