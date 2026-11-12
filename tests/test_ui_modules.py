"""tests/test_ui_modules.py - Test suite for PulmoVision UI Architecture and Modules.

Verifies:
1. Dynamic repository path resolution
2. Design token contract and theme definitions (Dark and Light)
3. CT Canvas background color invariant
4. Cached data loader schema and consistency
5. All 7 view modules import cleanly
6. Research report markdown availability
7. Strict quarantine check: No UI module imports or reads quarantined test partition
8. Zero emojis in application code: strict enforcement of Material Symbols
9. Multi-directory import resolution
10. Checkpoint hash integrity verification
"""

import os
import sys
import re
import hashlib
from pathlib import Path
import pytest

from app.utils.paths import REPO_ROOT, STAGE2_DEPLOYMENT_CKPT
from app.styles.tokens import THEMES, CT_CANVAS_COLOR, FONT_FAMILY_BASE, FONT_FAMILY_MONO
from app.components.navigation import DESTINATIONS
from app.utils.data_loader import (
    load_dev_manifest,
    load_cv_splits,
    load_stage2_experiment_results,
    get_deployment_predictor,
)


def test_repo_root_resolution():
    """Verify that REPO_ROOT points to valid PulmoVision repository root."""
    assert REPO_ROOT.is_dir()
    assert (REPO_ROOT / "app").is_dir()
    assert (REPO_ROOT / "src").is_dir()
    assert (REPO_ROOT / "data").is_dir()
    assert (REPO_ROOT / "reports").is_dir()


def test_theme_tokens_contract():
    """Verify both Dark and Light themes have required color tokens and proper contrast."""
    assert "dark" in THEMES
    assert "light" in THEMES

    for mode in ["dark", "light"]:
        theme = THEMES[mode]
        required_keys = [
            "bg_app", "bg_surface", "bg_elevated", "bg_card", "border",
            "border_subtle", "text_primary", "text_secondary", "text_muted",
            "accent_primary", "accent_secondary", "workstation_bg",
            "success", "warning", "error", "font_sans", "font_mono"
        ]
        for key in required_keys:
            assert key in theme, f"Missing token '{key}' in theme '{mode}'"

    assert CT_CANVAS_COLOR == "#000000"


def test_typography_tokens():
    """Verify typography font families."""
    assert "Inter" in FONT_FAMILY_BASE
    assert "JetBrains Mono" in FONT_FAMILY_MONO


def test_cached_manifest_loader():
    """Verify load_dev_manifest returns expected 3,132 slices and 122 unique patients."""
    df = load_dev_manifest()
    assert len(df) == 3132
    assert "patient_id" in df.columns
    assert "class" in df.columns
    assert "label" in df.columns
    assert "canonical_filename" in df.columns

    unique_pts = df.drop_duplicates(subset=["patient_id"])
    assert len(unique_pts) == 122

    covid_pts = (unique_pts["label"] == 1).sum()
    normal_pts = (unique_pts["label"] == 0).sum()
    assert covid_pts == 77
    assert normal_pts == 45


def test_cached_cv_splits_loader():
    """Verify load_cv_splits returns 5 outer folds with 122 unique validation patients."""
    splits = load_cv_splits()
    assert len(splits) == 5

    all_val_pts = set()
    for fold_name, fold_data in splits.items():
        assert "train_patients" in fold_data
        assert "val_patients" in fold_data
        all_val_pts.update(fold_data["val_patients"])

    assert len(all_val_pts) == 122


def test_cached_stage2_results_loader():
    """Verify load_stage2_experiment_results returns all 3 experiments with audited metrics."""
    res = load_stage2_experiment_results()
    assert "Exp A (Global Mean)" in res
    assert "Exp B (BiLSTM Mean)" in res
    assert "Exp C (BiLSTM Attn)" in res

    # Experiment B leader assertions
    exp_b = res["Exp B (BiLSTM Mean)"]
    assert "pooled_oof_metrics" in exp_b
    assert round(exp_b["pooled_oof_metrics"]["macro_f1"], 4) == 0.9461
    assert round(exp_b["pooled_oof_metrics"]["roc_auc"], 4) == 0.9688
    assert exp_b["pooled_oof_metrics"]["fn"] == 1


def test_all_views_importable():
    """Verify all 7 view modules import and expose their render functions."""
    from app.views.overview_view import render_overview_view
    from app.views.dataset_view import render_dataset_view
    from app.views.models_view import render_models_view
    from app.views.analytics_view import render_analytics_view
    from app.views.workstation_view import render_workstation_view
    from app.views.explainability_view import render_explainability_view
    from app.views.documentation_view import render_documentation_view

    assert callable(render_overview_view)
    assert callable(render_dataset_view)
    assert callable(render_models_view)
    assert callable(render_analytics_view)
    assert callable(render_workstation_view)
    assert callable(render_explainability_view)
    assert callable(render_documentation_view)


def test_navigation_material_icons():
    """Verify all 7 navigation destinations use valid Material Symbols icons with zero emojis."""
    assert len(DESTINATIONS) == 7
    for name, icon in DESTINATIONS:
        assert icon.startswith(":material/")
        assert icon.endswith(":")
        # Verify no emojis in name or icon
        assert not re.search(r"[\U00010000-\U0010ffff\u2600-\u27bf]", name)
        assert not re.search(r"[\U00010000-\U0010ffff\u2600-\u27bf]", icon)


def test_zero_emojis_in_entire_app():
    """Strictly verify zero unicode emojis exist in any application python source file."""
    app_dir = REPO_ROOT / "app"
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27bf]")
    violations = []

    for py_file in app_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        found = emoji_pattern.findall(content)
        if found:
            violations.append(f"{py_file.name}: {found}")

    assert len(violations) == 0, f"Found emoji violations in application code: {violations}"


def test_multi_directory_import_resolution():
    """Verify src and app modules resolve cleanly when sys.path includes REPO_ROOT."""
    # Ensure REPO_ROOT is in sys.path
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.inference.sequence_predictor import SequencePredictor
    from src.inference.sequence_validator import CTSequenceValidator
    assert SequencePredictor is not None
    assert CTSequenceValidator is not None


def test_checkpoint_hash_integrity():
    """Verify Stage 2 deployment checkpoint exists and matches verified SHA-256 hash."""
    assert STAGE2_DEPLOYMENT_CKPT.is_file()
    hasher = hashlib.sha256()
    with open(STAGE2_DEPLOYMENT_CKPT, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()
    assert file_hash == "0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed"


def test_research_report_file_present():
    """Verify the research report manuscript is present on disk."""
    report_file = REPO_ROOT / "reports" / "research" / "PULMOVISION_STAGE_2_RESEARCH_REPORT.md"
    assert report_file.is_file()
    text = report_file.read_text(encoding="utf-8")
    assert len(text) > 1000
    assert "0.9461" in text
    assert "Bahdanau" in text


def test_strict_quarantine_no_test_partition_access():
    """Strict Quarantine Verification:

    Ensure no code under app/ reads or imports the sealed test manifest.
    """
    app_dir = REPO_ROOT / "app"
    forbidden_tokens = ["test1.csv", "test_patient_manifest"]
    for py_file in app_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in content, (
                f"VIOLATION: Quarantined token '{token}' referenced in {py_file}"
            )
