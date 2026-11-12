"""
PulmoVision — Regression Tests for Stage 2 Anatomical De-duplication
Validates:
- Resolution of identical duplicate files into single canonical records.
- Exception raised when files with different hashes share the same composite key.
- Exception raised when files with conflicting labels share the same composite key.
- Deterministic canonical file selection invariant.
- Preservation of duplicate provenance (aliases and canonical mapping).
- Strict monotonic craniocaudal slice ordering with zero index repetition.
- Exactly one embedding per retained anatomical slice.
- Strict patient and series isolation (zero cross-series mixing).
- Quarantine of test1.csv.
"""
import os
import tempfile
import pytest
import pandas as pd
import torch

from src.data.deduplication import audit_and_deduplicate_manifest
from src.features.feature_extractor import PretrainedFeatureExtractor


class TestDuplicateResolutionInvariants:

    @pytest.fixture
    def synthetic_data_env(self):
        """Creates temporary files and synthetic manifest with duplicates and edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File 1 & 2: Byte-identical duplicates for patient1_SR_2_IM00010
            content_a = b"DICOM_HEADER_DUMMY_CONTENT_12345"
            p1_path = os.path.join(tmpdir, "normal1_patient1_SR_2_IM00010.tif")
            p2_path = os.path.join(tmpdir, "normal5_patient1_SR_2_IM00010.tif")
            with open(p1_path, "wb") as f:
                f.write(content_a)
            with open(p2_path, "wb") as f:
                f.write(content_a)

            # File 3: Unique slice for patient1_SR_2_IM00011
            content_b = b"DICOM_HEADER_DUMMY_CONTENT_67890"
            p3_path = os.path.join(tmpdir, "normal1_patient1_SR_2_IM00011.tif")
            with open(p3_path, "wb") as f:
                f.write(content_b)

            manifest_df = pd.DataFrame([
                {"filename": "normal1_patient1_SR_2_IM00010.tif", "class": "normal"},
                {"filename": "normal5_patient1_SR_2_IM00010.tif", "class": "normal"},
                {"filename": "normal1_patient1_SR_2_IM00011.tif", "class": "normal"},
            ])
            manifest_csv = os.path.join(tmpdir, "synthetic_manifest.csv")
            manifest_df.to_csv(manifest_csv, index=False)

            yield {
                "tmpdir": tmpdir,
                "manifest_csv": manifest_csv,
                "p1_path": p1_path,
                "p2_path": p2_path,
                "p3_path": p3_path,
            }

    def test_identical_duplicate_files_resolved(self, synthetic_data_env):
        """Tests that byte-identical files sharing (pid, sid, sidx) are resolved to 1 canonical record."""
        env = synthetic_data_env
        result = audit_and_deduplicate_manifest(
            manifest_csv=env["manifest_csv"],
            data_dir=env["tmpdir"],
            output_json_path=None,
            output_csv_path=None,
        )
        assert result["metadata"]["total_manifest_rows"] == 3
        assert result["metadata"]["total_unique_slices"] == 2
        assert result["metadata"]["total_duplicates_resolved"] == 1
        assert result["metadata"]["hash_verified_byte_identical"] is True

        # Check canonical record contains aliases
        canonical_slice = [s for s in result["canonical_slices"] if s["slice_idx"] == 10][0]
        assert canonical_slice["filename"] == "normal1_patient1_SR_2_IM00010.tif"
        assert canonical_slice["duplicate_aliases"] == ["normal5_patient1_SR_2_IM00010.tif"]

    def test_different_hashes_conflict_rejection(self, synthetic_data_env):
        """Tests that conflicting file contents for the same composite key raise ValueError."""
        env = synthetic_data_env
        # Modify file 2 content so hashes differ
        with open(env["p2_path"], "wb") as f:
            f.write(b"CORRUPTED_OR_DIFFERENT_CONTENT_99999")

        with pytest.raises(ValueError, match="conflicting file hashes"):
            audit_and_deduplicate_manifest(
                manifest_csv=env["manifest_csv"],
                data_dir=env["tmpdir"],
                output_json_path=None,
                output_csv_path=None,
                stop_on_conflict=True,
            )

    def test_conflicting_labels_rejection(self, synthetic_data_env):
        """Tests that conflicting labels for duplicate files raise ValueError."""
        env = synthetic_data_env
        # Make one duplicate label 'covid' and other 'normal'
        conflict_df = pd.DataFrame([
            {"filename": "normal1_patient1_SR_2_IM00010.tif", "class": "normal"},
            {"filename": "normal5_patient1_SR_2_IM00010.tif", "class": "covid"},
        ])
        conflict_csv = os.path.join(env["tmpdir"], "conflict_labels.csv")
        conflict_df.to_csv(conflict_csv, index=False)

        with pytest.raises(ValueError, match="conflicting labels"):
            audit_and_deduplicate_manifest(
                manifest_csv=conflict_csv,
                data_dir=env["tmpdir"],
                output_json_path=None,
                output_csv_path=None,
                stop_on_conflict=True,
            )

    def test_deterministic_canonical_selection(self, synthetic_data_env):
        """Tests that canonical selection is invariant to the order of rows in the manifest."""
        env = synthetic_data_env
        # Reverse order in manifest
        reversed_df = pd.DataFrame([
            {"filename": "normal5_patient1_SR_2_IM00010.tif", "class": "normal"},
            {"filename": "normal1_patient1_SR_2_IM00010.tif", "class": "normal"},
        ])
        rev_csv = os.path.join(env["tmpdir"], "reversed.csv")
        reversed_df.to_csv(rev_csv, index=False)

        res = audit_and_deduplicate_manifest(
            manifest_csv=rev_csv,
            data_dir=env["tmpdir"],
            output_json_path=None,
            output_csv_path=None,
        )
        canonical_name = res["canonical_slices"][0]["filename"]
        # Must still choose normal1 lexicographically first
        assert canonical_name == "normal1_patient1_SR_2_IM00010.tif"

    def test_safety_test1_csv_isolation(self):
        """Verifies that passing test1.csv raises immediate safety exception."""
        with pytest.raises(ValueError, match="Safety violation"):
            audit_and_deduplicate_manifest(manifest_csv="data/metadata/Annotation/CSV/test1.csv")


@pytest.fixture(scope="module")
def loaded_cache():
    cache_path = "data/processed/dev_patient_features_effnet_b0.pt"
    if not os.path.exists(cache_path):
        pytest.skip(f"Feature cache {cache_path} not found on disk")
    return torch.load(cache_path, map_location="cpu", weights_only=False)


class TestFeatureCacheIntegrityAndSequences:

    def test_actual_counts_and_metadata_protocol(self, loaded_cache):
        """Verifies actual patient, series, unique slice counts, and explicit metadata."""
        meta = loaded_cache["metadata"]
        assert meta["total_patients"] == 122
        assert meta["total_series"] == 239
        assert meta["total_slices"] == 3132
        assert meta["total_manifest_rows"] == 3736
        assert meta["duplicate_slices_resolved"] == 604
        assert "Option 1" in meta["deduplication_protocol"]

    def test_strict_slice_ordering_and_no_duplicates(self, loaded_cache):
        """Verifies strictly increasing slice indices and zero duplicate indices within every sequence."""
        series_dict = loaded_cache["series"]
        for sid_key, s_data in series_dict.items():
            indices = s_data["slice_indices"]
            # Strictly increasing: index[i] < index[i+1]
            for i in range(len(indices) - 1):
                assert indices[i] < indices[i + 1], f"Slice ordering violation in {sid_key}: {indices[i]} >= {indices[i+1]}"
            # Zero duplicates
            assert len(indices) == len(set(indices))

    def test_exactly_one_embedding_per_retained_slice(self, loaded_cache):
        """Verifies every retained slice has exactly one 1280D embedding."""
        series_dict = loaded_cache["series"]
        for sid_key, s_data in series_dict.items():
            embs = s_data["embeddings"]
            num_slices = s_data["num_slices"]
            assert embs.shape == (num_slices, 1280)
            assert not torch.isnan(embs).any()
            assert not torch.isinf(embs).any()

    def test_duplicate_aliases_provenance_preserved(self, loaded_cache):
        """Verifies duplicate provenance records exist and map to canonical files."""
        prov = loaded_cache.get("duplicate_provenance", {})
        assert len(prov) == 604
        # Spot check patient168
        p168_keys = [k for k in prov if "patient168" in k]
        assert len(p168_keys) > 0
        sample_key = p168_keys[0]
        assert prov[sample_key]["patient_id"] == "patient168"
        assert len(prov[sample_key]["duplicate_aliases"]) >= 1

    def test_patient_and_series_isolation(self, loaded_cache):
        """Verifies that no cross-patient or cross-series contamination occurs."""
        patients = loaded_cache["patients"]
        series_dict = loaded_cache["series"]

        # Check that all series in patients match series_dict
        for pid, p_data in patients.items():
            for sid, s_data in p_data["series"].items():
                assert s_data["patient_id"] == pid
                assert s_data["series_id"] == sid
                assert f"{pid}_{sid}" in series_dict
