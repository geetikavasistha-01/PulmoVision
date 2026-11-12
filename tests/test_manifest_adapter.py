"""tests/test_manifest_adapter.py - Unit and Regression Tests for ManifestAdapter.

Verifies:
1. Normalization from official development CSV manifest
2. Normalization from official development JSON manifest
3. Detection and rejection of missing required schema columns
4. Robust handling and defaulting of missing optional columns
5. Class filtering logic (COVID-19 vs Normal)
6. Case-insensitive multi-column search
7. Cohort metadata and summary aggregation
8. Empty manifest safe degradation
"""

from pathlib import Path
import pytest
import pandas as pd

from app.utils.paths import DEV_DEDUP_MANIFEST_CSV, DEV_DEDUP_MANIFEST_JSON
from app.utils.manifest_adapter import (
    ManifestAdapter,
    ManifestValidationError,
    REQUIRED_COLUMNS,
)


def test_adapt_from_real_csv():
    """Verify ManifestAdapter successfully loads and normalizes the real development CSV."""
    df = ManifestAdapter.adapt_manifest(DEV_DEDUP_MANIFEST_CSV)
    assert len(df) == 3132
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"
    assert "class_name" in df.columns
    assert "duplicate_count" in df.columns
    assert "duplicate_aliases" in df.columns
    assert df["patient_id"].nunique() == 122


def test_adapt_from_real_json():
    """Verify ManifestAdapter successfully loads and normalizes the real development JSON."""
    df = ManifestAdapter.adapt_manifest(DEV_DEDUP_MANIFEST_JSON)
    assert len(df) == 3132
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"
    assert df["patient_id"].nunique() == 122
    assert "canonical_filename" in df.columns


def test_missing_required_column_raises():
    """Verify missing required columns raise ManifestValidationError with clear message."""
    bad_df = pd.DataFrame({
        "series_id": ["SR_1"],
        "slice_idx": [1],
        "label": [1],
        "filename": ["img.tif"]
        # Missing 'patient_id'
    })
    with pytest.raises(ManifestValidationError, match="patient_id"):
        ManifestAdapter.adapt_manifest(bad_df)


def test_missing_filename_column_raises():
    """Verify missing both canonical_filename and filename raises ManifestValidationError."""
    bad_df = pd.DataFrame({
        "patient_id": ["patient1"],
        "series_id": ["SR_1"],
        "slice_idx": [1],
        "label": [1],
    })
    with pytest.raises(ManifestValidationError, match="filename"):
        ManifestAdapter.adapt_manifest(bad_df)


def test_missing_optional_columns_defaults():
    """Verify missing optional columns are populated with valid defaults."""
    minimal_df = pd.DataFrame({
        "patient_id": ["pt1", "pt2"],
        "series_id": ["SR_1", "SR_2"],
        "slice_idx": [10, 11],
        "label": [1, 0],
        "filename": ["pt1_im10.tif", "pt2_im11.tif"],
    })
    adapted = ManifestAdapter.adapt_manifest(minimal_df)
    assert "duplicate_count" in adapted.columns
    assert (adapted["duplicate_count"] == 0).all()
    assert "duplicate_aliases" in adapted.columns
    assert (adapted["duplicate_aliases"] == "").all()
    assert list(adapted["class_name"]) == ["COVID-19", "Normal"]


def test_validate_manifest_contract():
    """Verify validate_manifest returns tuple of (is_valid, missing_list)."""
    valid_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    is_valid, missing = ManifestAdapter.validate_manifest(valid_df)
    assert is_valid is True
    assert missing == []

    invalid_df = pd.DataFrame(columns=["patient_id", "series_id"])
    is_valid, missing = ManifestAdapter.validate_manifest(invalid_df)
    assert is_valid is False
    assert "canonical_filename" in missing


def test_filter_manifest_by_class():
    """Verify filtering by class labels accurately isolates patient cohorts."""
    df = ManifestAdapter.adapt_manifest(DEV_DEDUP_MANIFEST_CSV)
    
    covid_df = ManifestAdapter.filter_manifest(df, label_filter="COVID-19 (Label 1)")
    assert len(covid_df) == 1820
    assert (covid_df["label"] == 1).all()

    normal_df = ManifestAdapter.filter_manifest(df, label_filter="Normal (Label 0)")
    assert len(normal_df) == 1312
    assert (normal_df["label"] == 0).all()

    all_df = ManifestAdapter.filter_manifest(df, label_filter="All Classes")
    assert len(all_df) == 3132


def test_filter_manifest_search():
    """Verify search filter matches patient_id, series_id, or filename."""
    df = ManifestAdapter.adapt_manifest(DEV_DEDUP_MANIFEST_CSV)

    res_patient = ManifestAdapter.filter_manifest(df, search_query="patient1")
    assert len(res_patient) > 0
    assert "patient1" in res_patient["patient_id"].values

    res_series = ManifestAdapter.filter_manifest(df, search_query="SR_2")
    assert len(res_series) > 0
    assert (res_series["series_id"] == "SR_2").all()

    res_nonexistent = ManifestAdapter.filter_manifest(df, search_query="nonexistent_patient_xyz")
    assert len(res_nonexistent) == 0


def test_get_manifest_summary_counts():
    """Verify manifest summary aggregates exact authoritative development cohort numbers."""
    df = ManifestAdapter.adapt_manifest(DEV_DEDUP_MANIFEST_CSV)
    summary = ManifestAdapter.get_manifest_summary(df)

    assert summary["total_slices"] == 3132
    assert summary["unique_patients"] == 122
    assert summary["unique_series"] == 239
    assert summary["covid_patients"] == 77
    assert summary["normal_patients"] == 45
    assert summary["covid_slices"] == 1820
    assert summary["normal_slices"] == 1312
    assert summary["total_duplicates_resolved"] == 604


def test_empty_manifest_safe_handling():
    """Verify empty DataFrame safely returns empty normalized frame and empty summary."""
    empty_df = pd.DataFrame()
    adapted = ManifestAdapter.adapt_manifest(empty_df)
    assert adapted.empty
    for col in REQUIRED_COLUMNS:
        assert col in adapted.columns

    summary = ManifestAdapter.get_manifest_summary(empty_df)
    assert summary["total_slices"] == 0
    assert summary["unique_patients"] == 0
