"""
Unit and Regression Tests for Stage 2 Sequence Predictor
"""

import os
import pytest
import numpy as np
import torch
from src.inference.sequence_predictor import SequencePredictor


@pytest.fixture
def dummy_slices():
    # 4 synthetic CT slices of size 512x512
    np.random.seed(42)
    return [np.random.randint(200, 1800, size=(512, 512), dtype=np.int16) for _ in range(4)]


def test_sequence_predictor_initialization():
    pred = SequencePredictor()
    assert pred.model is not None
    assert pred.threshold == 0.50
    assert pred.CLASS_NAMES == ["Normal", "COVID-19"]


def test_sequence_predictor_multi_slice_inference(dummy_slices):
    pred = SequencePredictor()
    filenames = ["patient1_SR_1_IM00001.tif", "patient1_SR_1_IM00002.tif", "patient1_SR_1_IM00003.tif", "patient1_SR_1_IM00004.tif"]
    res = pred.predict_sequence(dummy_slices, filenames=filenames)

    assert res["predicted_class"] in ["Normal", "COVID-19"]
    assert 0.0 <= res["covid_19_probability"] <= 1.0
    assert 0.0 <= res["normal_probability"] <= 1.0
    assert abs(res["covid_19_probability"] + res["normal_probability"] - 1.0) < 1e-4
    assert res["num_slices"] == 4
    assert res["slice_indices"] == [1, 2, 3, 4]
    assert res["is_single_slice"] is False
    assert len(res["display_slices"]) == 4
    assert "RESEARCH PROTOTYPE ONLY" in res["disclaimer"]


def test_sequence_predictor_single_slice_fallback():
    pred = SequencePredictor()
    single_slice = [np.random.randint(200, 1800, size=(512, 512), dtype=np.int16)]
    filenames = ["patient1_SR_1_IM00005.tif"]
    res = pred.predict_sequence(single_slice, filenames=filenames)

    assert res["num_slices"] == 1
    assert res["is_single_slice"] is True
    assert res["warning"] is not None
    assert "Single-slice input" in res["warning"]

