"""
Automated unit tests for PulmoVision 2.5D slice stacking and DataLoader pipeline.
"""

import pytest
import torch
import numpy as np
from src.data.slice_stacking import CTSliceStacker
from src.data.dataloader import CT25DDataset, create_dataloader
from src.data.preprocessing import CTPreprocessor


def test_filename_parsing():
    fname1 = "137covid_patient100_SR_2_IM00028.tif"
    info1 = CTSliceStacker.parse_filename(fname1)
    assert info1["patient_id"] == "patient100"
    assert info1["series_id"] == "SR_2"
    assert info1["slice_idx"] == 28
    assert info1["label"] == "covid"

    fname2 = "normal_patient5_SR_1_IM00003.tif"
    info2 = CTSliceStacker.parse_filename(fname2)
    assert info2["patient_id"] == "patient5"
    assert info2["series_id"] == "SR_1"
    assert info2["slice_idx"] == 3
    assert info2["label"] == "normal"


def test_triplet_generation_boundary_and_grouping():
    # Simulate unordered slices from 2 different patients
    mock_files = [
        "covid_patient1_SR_1_IM00003.tif",
        "covid_patient1_SR_1_IM00001.tif",
        "covid_patient1_SR_1_IM00002.tif",
        "normal_patient2_SR_1_IM00010.tif",
        "normal_patient2_SR_1_IM00011.tif",
    ]

    triplets = CTSliceStacker.generate_triplets(mock_files)
    # Patient 1 has 3 slices -> 3 triplets
    # Patient 2 has 2 slices -> 2 triplets
    assert len(triplets) == 5

    # Check Patient 1 boundary: First slice (IM00001) should have clamped prev
    t0_prev, t0_curr, t0_next, lbl0 = triplets[0]
    assert "IM00001" in t0_curr
    assert "IM00001" in t0_prev  # Clamped to itself at boundary
    assert "IM00002" in t0_next
    assert lbl0 == "covid"

    # Check that Patient 1 and Patient 2 never mix
    for p_prev, p_curr, p_next, _ in triplets:
        p1 = CTSliceStacker.parse_filename(p_prev)["patient_id"]
        p2 = CTSliceStacker.parse_filename(p_curr)["patient_id"]
        p3 = CTSliceStacker.parse_filename(p_next)["patient_id"]
        assert p1 == p2 == p3, f"Cross-patient leak detected! {p1}, {p2}, {p3}"


def test_dataset_and_dataloader_batch_shape():
    # Create mock dummy array slices
    dummy1 = np.random.randint(0, 2000, (64, 64), dtype=np.uint16)
    dummy2 = np.random.randint(0, 2000, (64, 64), dtype=np.uint16)
    dummy3 = np.random.randint(0, 2000, (64, 64), dtype=np.uint16)

    # 4 sample triplets using in-memory arrays
    mock_triplets = [
        (dummy1, dummy2, dummy3, "covid"),
        (dummy2, dummy3, dummy1, "normal"),
        (dummy3, dummy1, dummy2, "covid"),
        (dummy1, dummy2, dummy3, "normal"),
    ]

    preprocessor = CTPreprocessor(target_size=(224, 224), is_training=False)
    dataset = CT25DDataset(triplets=mock_triplets, preprocessor=preprocessor)

    # Test single sample
    tensor, label = dataset[0]
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32
    assert label == 1  # covid

    # Test DataLoader batching (batch_size=2)
    loader = create_dataloader(mock_triplets, preprocessor=preprocessor, batch_size=2)
    batch_tensors, batch_labels = next(iter(loader))

    assert batch_tensors.shape == (2, 3, 224, 224), f"Expected (2, 3, 224, 224), got {batch_tensors.shape}"
    assert batch_labels.shape == (2,)
    assert not torch.isnan(batch_tensors).any()
