"""
Unit and Regression Tests for CT Sequence & Provenance Validator
"""

import pytest
import numpy as np
from src.inference.sequence_validator import CTSequenceValidator, SequenceValidationError


def test_filename_parsing():
    res = CTSequenceValidator.parse_provenance("patient14_SR_2_IM00035.tif")
    assert res["patient_id"] == "patient14"
    assert res["series_id"] == "SR_2"
    assert res["slice_idx"] == 35


def test_sorting_craniocaudal_order():
    files = ["slice_3.tif", "slice_1.tif", "slice_2.tif"]
    ordered, meta = CTSequenceValidator.validate_and_order_sequence(files)
    assert meta["slice_indices"] == [1, 2, 3]
    assert ordered == ["slice_1.tif", "slice_2.tif", "slice_3.tif"]
    assert meta["num_slices"] == 3
    assert meta["is_single_slice"] is False
    assert meta["warning"] is None


def test_cross_patient_rejection():
    files = ["patient1_SR_1_IM00001.tif", "patient2_SR_1_IM00002.tif"]
    with pytest.raises(SequenceValidationError, match="Cross-patient sequence violation"):
        CTSequenceValidator.validate_and_order_sequence(files)


def test_cross_series_rejection():
    files = ["patient1_SR_1_IM00001.tif", "patient1_SR_2_IM00002.tif"]
    with pytest.raises(SequenceValidationError, match="Cross-series sequence violation"):
        CTSequenceValidator.validate_and_order_sequence(files)


def test_single_slice_warning():
    files = ["patient5_SR_1_IM00010.tif"]
    ordered, meta = CTSequenceValidator.validate_and_order_sequence(files)
    assert meta["is_single_slice"] is True
    assert "Single-slice input" in meta["warning"]


def test_unsupported_extension_rejection():
    files = ["patient1_scan.pdf"]
    with pytest.raises(SequenceValidationError, match="Unsupported file format"):
        CTSequenceValidator.validate_and_order_sequence(files)


def test_empty_sequence_rejection():
    with pytest.raises(SequenceValidationError, match="No CT slices provided"):
        CTSequenceValidator.validate_and_order_sequence([])

