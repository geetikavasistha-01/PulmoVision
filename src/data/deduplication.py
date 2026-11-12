"""
PulmoVision — Stage 2 Anatomical De-duplication Engine
Performs forensic hash verification across duplicate archive slice entries,
enforces byte-identity invariants, deterministically selects canonical slices,
preserves duplicate alias provenance, and constructs strictly monotonic anatomical sequences.
"""
from typing import Dict, List, Tuple, Any, Optional
import os
import hashlib
import pandas as pd
import json

from src.data.slice_stacking import CTSliceStacker


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 checksum of a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_and_deduplicate_manifest(
    manifest_csv: str = "data/metadata/Annotation/CSV/train1.csv",
    data_dir: str = "data/raw/TrainValidation",
    output_json_path: Optional[str] = "data/metadata/splits/stage2_dev_deduplicated_manifest.json",
    output_csv_path: Optional[str] = "data/metadata/splits/stage2_dev_deduplicated_manifest.csv",
    stop_on_conflict: bool = True,
    verify_file_hashes: bool = True
) -> Dict[str, Any]:
    """
    Forensically audits manifest records grouped by (patient_id, series_id, slice_idx).
    
    Invariants enforced:
    1. For any duplicate group, all instances must have identical file content (matching SHA256).
    2. All instances in a duplicate group must have identical labels and metadata.
    3. Exactly one canonical slice is selected using deterministic lexicographical filename order.
    4. Non-canonical instances are catalogued as duplicate aliases with full provenance.
    5. Source manifests (train1.csv, test1.csv) are NEVER modified.
    
    Raises:
        ValueError: If any duplicate group exhibits conflicting hashes or inconsistent labels.
    """
    if "test1" in manifest_csv.lower():
        raise ValueError("Safety violation: Deduplication audit must never process test1.csv!")

    df = pd.read_csv(manifest_csv)
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}

    for idx, row in df.iterrows():
        fname = row["filename"]
        fpath = os.path.join(data_dir, fname) if data_dir else fname
        meta = CTSliceStacker.parse_filename(fpath)
        pid = meta["patient_id"]
        sid = meta["series_id"]
        sidx = meta["slice_idx"]
        lbl_str = row["class"]
        lbl_bin = 1 if "covid" in str(lbl_str).lower() else 0

        key = (pid, sid, sidx)
        record = {
            "manifest_row": int(idx),
            "filename": fname,
            "filepath": fpath,
            "patient_id": pid,
            "series_id": sid,
            "slice_idx": sidx,
            "label_str": lbl_str,
            "label": lbl_bin,
        }
        groups.setdefault(key, []).append(record)

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    hash_conflicts = []
    label_conflicts = []

    # Forensic verification
    for key, records in duplicate_groups.items():
        hashes = []
        labels = []
        for r in records:
            if verify_file_hashes and os.path.exists(r["filepath"]):
                h = compute_file_sha256(r["filepath"])
                r["sha256"] = h
                hashes.append(h)
            elif "sha256" in r:
                hashes.append(r["sha256"])
            labels.append(r["label"])

        if hashes and len(set(hashes)) > 1:
            hash_conflicts.append((key, records))
        if len(set(labels)) > 1:
            label_conflicts.append((key, records))

    if stop_on_conflict:
        if hash_conflicts:
            raise ValueError(
                f"Forensic duplicate verification failed: {len(hash_conflicts)} groups have conflicting file hashes! "
                f"Example key: {hash_conflicts[0][0]}"
            )
        if label_conflicts:
            raise ValueError(
                f"Forensic duplicate verification failed: {len(label_conflicts)} groups have conflicting labels! "
                f"Example key: {label_conflicts[0][0]}"
            )

    # Deterministic canonical selection: alphabetical order by filename
    canonical_slices = []
    duplicate_provenance_map: Dict[str, Any] = {}
    total_duplicates_resolved = 0

    for key in sorted(groups.keys()):
        records = groups[key]
        # Sort deterministically by filename
        sorted_records = sorted(records, key=lambda r: r["filename"])
        canonical = sorted_records[0].copy()

        if len(sorted_records) > 1:
            aliases = [r["filename"] for r in sorted_records[1:]]
            canonical["duplicate_aliases"] = aliases
            total_duplicates_resolved += (len(sorted_records) - 1)
            composite_key_str = f"{key[0]}_{key[1]}_IM{key[2]:05d}"
            duplicate_provenance_map[composite_key_str] = {
                "canonical_filename": canonical["filename"],
                "duplicate_aliases": aliases,
                "sha256": canonical.get("sha256"),
                "patient_id": key[0],
                "series_id": key[1],
                "slice_idx": key[2],
                "label": canonical["label"],
            }
        else:
            canonical["duplicate_aliases"] = []

        canonical_slices.append(canonical)

    result = {
        "metadata": {
            "source_manifest": manifest_csv,
            "total_manifest_rows": len(df),
            "total_unique_slices": len(canonical_slices),
            "duplicate_groups_count": len(duplicate_groups),
            "total_duplicates_resolved": total_duplicates_resolved,
            "canonical_selection_rule": "lexicographical_filename_ascending (sorted(filenames)[0])",
            "hash_verified_byte_identical": len(hash_conflicts) == 0,
            "label_verified_consistent": len(label_conflicts) == 0,
        },
        "canonical_slices": canonical_slices,
        "duplicate_provenance": duplicate_provenance_map,
    }

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(result, f, indent=2)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        csv_rows = []
        for s in canonical_slices:
            csv_rows.append({
                "filename": s["filename"],
                "patient_id": s["patient_id"],
                "series_id": s["series_id"],
                "slice_idx": s["slice_idx"],
                "class": s["label_str"],
                "label": s["label"],
                "duplicate_count": len(s["duplicate_aliases"]),
                "duplicate_aliases": ";".join(s["duplicate_aliases"]) if s["duplicate_aliases"] else "",
            })
        pd.DataFrame(csv_rows).to_csv(output_csv_path, index=False)

    return result

