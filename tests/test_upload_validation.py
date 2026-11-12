"""
Automated unit tests for CT upload validation and provenance verification.
"""

import pytest
from src.inference.upload_validator import (
    CTUploadValidator,
    UploadValidationError,
    InferenceMode,
)


def test_allowed_and_disallowed_formats():
    # Valid formats
    res = CTUploadValidator.validate_single_file("137covid_patient100_SR_2_IM00028.tif")
    assert res["patient_id"] == "patient100"

    CTUploadValidator.validate_single_file("slice.png")
    CTUploadValidator.validate_single_file("slice.jpg")

    # Invalid formats
    with pytest.raises(UploadValidationError, match="Unsupported image format"):
        CTUploadValidator.validate_single_file("scan.pdf")

    with pytest.raises(UploadValidationError, match="Unsupported image format"):
        CTUploadValidator.validate_single_file("malicious.exe")


def test_cross_patient_mismatch_rejection():
    f_curr = "137covid_patient100_SR_2_IM00028.tif"
    f_prev = "137covid_patient100_SR_2_IM00027.tif"
    f_next = "normal_patient102_SR_1_IM00029.tif"  # Different patient!

    with pytest.raises(UploadValidationError, match="Cross-Patient Mismatch Violation"):
        CTUploadValidator.validate_triplet_provenance(f_curr, f_prev, f_next)


def test_series_mismatch_rejection():
    f_curr = "137covid_patient100_SR_2_IM00028.tif"
    f_prev = "137covid_patient100_SR_2_IM00027.tif"
    f_next = "137covid_patient100_SR_3_IM00029.tif"  # SR_3 instead of SR_2!

    with pytest.raises(UploadValidationError, match="Series Mismatch Violation"):
        CTUploadValidator.validate_triplet_provenance(f_curr, f_prev, f_next)


def test_slice_order_inversion_rejection():
    f_curr = "137covid_patient100_SR_2_IM00028.tif"
    f_prev = "137covid_patient100_SR_2_IM00029.tif"  # 29 is AFTER 28!
    f_next = "137covid_patient100_SR_2_IM00030.tif"

    with pytest.raises(UploadValidationError, match="Slice Ordering Inversion"):
        CTUploadValidator.validate_triplet_provenance(f_curr, f_prev, f_next)


def test_valid_full_25d_mode():
    f_curr = "137covid_patient100_SR_2_IM00028.tif"
    f_prev = "137covid_patient100_SR_2_IM00027.tif"
    f_next = "137covid_patient100_SR_2_IM00029.tif"

    mode, meta = CTUploadValidator.validate_triplet_provenance(f_curr, f_prev, f_next)
    assert mode == InferenceMode.FULL_25D
    assert meta["patient_id"] == "patient100"
    assert meta["indices"] == (27, 28, 29)


def test_isolated_slice_fallback_mode():
    f_curr = "137covid_patient100_SR_2_IM00028.tif"
    mode, meta = CTUploadValidator.validate_triplet_provenance(f_curr)
    assert mode == InferenceMode.ISOLATED_FALLBACK
    assert meta["patient_id"] == "patient100"

