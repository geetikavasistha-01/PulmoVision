"""app/utils/manifest_adapter.py - Manifest Adapter and Normalization Engine.

Translates between raw dataset manifest formats (CSV / JSON) and a clean,
UI-facing normalized schema with strong typing, schema validation, and safe query filters.
Guarantees zero modifications to underlying source files and absolute quarantine protection.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import pandas as pd


# Canonical UI Schema Definition
REQUIRED_COLUMNS = [
    "patient_id",
    "series_id",
    "slice_idx",
    "label",
    "canonical_filename",
]

OPTIONAL_PROVENANCE_COLUMNS = [
    "class_name",
    "duplicate_count",
    "duplicate_aliases",
    "filepath",
    "manifest_row",
]


class ManifestValidationError(ValueError):
    """Raised when a manifest does not conform to required structural expectations."""
    pass


class ManifestAdapter:
    """Robust adapter normalizing raw CSV/JSON manifest files into a consistent UI schema."""

    @staticmethod
    def adapt_manifest(
        source: Union[pd.DataFrame, Dict[str, Any], str, Path]
    ) -> pd.DataFrame:
        """Adapts a CSV path, JSON path, raw dict, or raw DataFrame into normalized UI schema.

        Returns:
            Normalized pandas DataFrame containing all required UI columns and provenance fields.
        Raises:
            ManifestValidationError: If essential required patient/slice coordinates are missing.
        """
        raw_df: pd.DataFrame

        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Manifest file not found: {path}")
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ManifestAdapter._adapt_from_json_dict(data)
            else:
                raw_df = pd.read_csv(path)
        elif isinstance(source, dict):
            return ManifestAdapter._adapt_from_json_dict(source)
        elif isinstance(source, pd.DataFrame):
            raw_df = source.copy()
        else:
            raise TypeError(f"Unsupported manifest source type: {type(source)}")

        return ManifestAdapter._normalize_dataframe(raw_df)

    @staticmethod
    def _adapt_from_json_dict(data: Dict[str, Any]) -> pd.DataFrame:
        """Extracts canonical slice records from Stage 2 JSON manifest."""
        if "canonical_slices" in data:
            slices = data["canonical_slices"]
        elif isinstance(data, list):
            slices = data
        else:
            raise ManifestValidationError("JSON manifest missing 'canonical_slices' collection.")

        rows = []
        for s in slices:
            aliases = s.get("duplicate_aliases", [])
            if isinstance(aliases, list):
                alias_str = ", ".join(str(a) for a in aliases)
                dup_count = len(aliases)
            else:
                alias_str = str(aliases) if aliases else ""
                dup_count = 1 if alias_str else 0

            rows.append({
                "canonical_filename": s.get("filename") or s.get("canonical_filename", ""),
                "filename": s.get("filename") or s.get("canonical_filename", ""),
                "patient_id": str(s.get("patient_id", "")),
                "series_id": str(s.get("series_id", "")),
                "slice_idx": int(s.get("slice_idx", 0)),
                "label": int(s.get("label", 0)),
                "class_name": "COVID-19" if int(s.get("label", 0)) == 1 else "Normal",
                "class": s.get("label_str") or ("covid" if int(s.get("label", 0)) == 1 else "normal"),
                "duplicate_count": dup_count,
                "duplicate_aliases": alias_str,
                "filepath": s.get("filepath", ""),
                "manifest_row": s.get("manifest_row", None),
            })

        df = pd.DataFrame(rows)
        return ManifestAdapter._normalize_dataframe(df)

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all expected UI columns exist with valid types and default fallbacks."""
        if df.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_PROVENANCE_COLUMNS)

        res = df.copy()

        # Map canonical filename
        if "canonical_filename" not in res.columns:
            if "filename" in res.columns:
                res["canonical_filename"] = res["filename"].astype(str)
            elif "slice_filename" in res.columns:
                res["canonical_filename"] = res["slice_filename"].astype(str)
            else:
                raise ManifestValidationError(
                    "Manifest is missing a filename column ('canonical_filename' or 'filename')."
                )

        # Ensure 'filename' column also exists for backwards compatibility
        if "filename" not in res.columns:
            res["filename"] = res["canonical_filename"]

        # Validate required identifiers
        for req in ["patient_id", "series_id", "slice_idx", "label"]:
            if req not in res.columns:
                raise ManifestValidationError(f"Manifest missing required column: '{req}'")

        res["patient_id"] = res["patient_id"].astype(str)
        res["series_id"] = res["series_id"].astype(str)
        res["slice_idx"] = pd.to_numeric(res["slice_idx"], errors="coerce").fillna(0).astype(int)
        res["label"] = pd.to_numeric(res["label"], errors="coerce").fillna(0).astype(int)

        # Standardize class name
        if "class_name" not in res.columns:
            if "class" in res.columns:
                res["class_name"] = res["class"].apply(
                    lambda c: "COVID-19" if "covid" in str(c).lower() else "Normal"
                )
            else:
                res["class_name"] = res["label"].apply(lambda l: "COVID-19" if l == 1 else "Normal")

        # Standardize backward-compatible 'class' column
        if "class" not in res.columns:
            res["class"] = res["label"].apply(lambda l: "covid" if l == 1 else "normal")

        # Duplicate provenance handling
        if "duplicate_count" not in res.columns:
            if "duplicate_aliases" in res.columns:
                res["duplicate_count"] = res["duplicate_aliases"].apply(
                    lambda a: len(str(a).split(";")) if pd.notna(a) and str(a).strip() else 0
                )
            else:
                res["duplicate_count"] = 0
        else:
            res["duplicate_count"] = pd.to_numeric(res["duplicate_count"], errors="coerce").fillna(0).astype(int)

        if "duplicate_aliases" not in res.columns:
            res["duplicate_aliases"] = ""
        else:
            res["duplicate_aliases"] = res["duplicate_aliases"].fillna("").astype(str)

        return res

    @staticmethod
    def validate_manifest(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validates that a DataFrame contains all required UI schema columns.

        Returns:
            (is_valid, list_of_missing_columns)
        """
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        return (len(missing) == 0, missing)

    @staticmethod
    def filter_manifest(
        df: pd.DataFrame,
        label_filter: str = "All",
        search_query: str = "",
    ) -> pd.DataFrame:
        """Safely filters manifest by class label and text search across identifiers."""
        if df.empty:
            return df

        filtered = df.copy()

        # Class filter
        if label_filter in ["COVID-19", "COVID-19 (Label 1)", "1", 1]:
            filtered = filtered[filtered["label"] == 1]
        elif label_filter in ["Normal", "Normal (Label 0)", "0", 0]:
            filtered = filtered[filtered["label"] == 0]

        # Search filter
        if search_query and search_query.strip():
            q = search_query.strip().lower()
            mask = (
                filtered["patient_id"].astype(str).str.lower().str.contains(q, na=False) |
                filtered["series_id"].astype(str).str.lower().str.contains(q, na=False) |
                filtered["canonical_filename"].astype(str).str.lower().str.contains(q, na=False) |
                filtered["duplicate_aliases"].astype(str).str.lower().str.contains(q, na=False)
            )
            filtered = filtered[mask]

        return filtered

    @staticmethod
    def get_manifest_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates cohort-level metadata and verification metrics."""
        if df.empty:
            return {
                "total_slices": 0,
                "unique_patients": 0,
                "unique_series": 0,
                "covid_patients": 0,
                "normal_patients": 0,
                "covid_slices": 0,
                "normal_slices": 0,
                "slices_with_duplicates": 0,
                "total_duplicates_resolved": 0,
            }

        patient_df = df.drop_duplicates(subset=["patient_id"])
        covid_patients = int((patient_df["label"] == 1).sum())
        normal_patients = int((patient_df["label"] == 0).sum())

        covid_slices = int((df["label"] == 1).sum())
        normal_slices = int((df["label"] == 0).sum())

        # Count total acquisition series across patients
        if "patient_id" in df.columns and "series_id" in df.columns:
            unique_series_count = int(df.groupby(["patient_id", "series_id"]).ngroups)
        else:
            unique_series_count = int(df["series_id"].nunique()) if "series_id" in df.columns else 0

        dup_counts = int(df["duplicate_count"].sum()) if "duplicate_count" in df.columns else 0
        dup_slices = int((df["duplicate_count"] > 0).sum()) if "duplicate_count" in df.columns else 0

        return {
            "total_slices": len(df),
            "unique_patients": int(df["patient_id"].nunique()),
            "unique_series": unique_series_count,
            "covid_patients": covid_patients,
            "normal_patients": normal_patients,
            "covid_slices": covid_slices,
            "normal_slices": normal_slices,
            "slices_with_duplicates": dup_slices,
            "total_duplicates_resolved": dup_counts,
        }
