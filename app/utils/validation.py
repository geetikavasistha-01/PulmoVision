"""
PulmoVision — UI Validation Utilities
Inspects uploaded CT image files, validates extensions, bit-depth expectations,
and enforces patient/series sequence integrity.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
from src.inference.sequence_validator import CTSequenceValidator, SequenceValidationError


def validate_ct_uploads(uploaded_files: List[Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates uploaded file collection for patient/series consistency, ascending slice indices,
    and supported file formats.

    Returns:
        (is_valid, error_message, validation_metadata)
    """
    if not uploaded_files:
        return False, "No files selected. Please upload CT slices.", None

    filenames = [f.name if hasattr(f, "name") else str(f) for f in uploaded_files]

    try:
        ordered_files, meta = CTSequenceValidator.validate_and_order_sequence(
            uploaded_files, filenames=filenames
        )
        return True, None, meta
    except SequenceValidationError as e:
        return False, str(e), None
    except Exception as e:
        return False, f"Unexpected validation failure: {str(e)}", None

