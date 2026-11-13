"""
PulmoVision — Repository Path Resolution Utilities
Centralizes path resolution to guarantee robust imports regardless of invocation directory.
"""

import sys
import os
from pathlib import Path

# Resolve repository root: .../PulmoVision
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Guarantee repository root is at the front of sys.path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Centralized project asset paths
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DEV_DATA_DIR = DATA_DIR / "raw" / "TrainValidation"
DEMO_SAMPLES_DIR = DATA_DIR / "demo_samples"
SPLITS_DIR = METADATA_DIR / "splits"

# Canonical Model Checkpoints
STAGE2_DEPLOYMENT_CKPT = MODELS_DIR / "stage2_bilstm_deployment.pth"
STAGE1_BEST_CKPT = MODELS_DIR / "best_model.pth"

# Manifests and Results
DEV_DEDUP_MANIFEST_CSV = SPLITS_DIR / "stage2_dev_deduplicated_manifest.csv"
DEV_DEDUP_MANIFEST_JSON = SPLITS_DIR / "stage2_dev_deduplicated_manifest.json"
CV_5FOLD_MANIFEST_JSON = SPLITS_DIR / "cv_5fold_manifest.json"

STAGE2_RESULTS_DIR = REPORTS_DIR / "experiments"
STAGE2_EXP_A_JSON = STAGE2_RESULTS_DIR / "stage2_expA_effnet_mean_pooling" / "nested_cv_results.json"
STAGE2_EXP_B_JSON = STAGE2_RESULTS_DIR / "stage2_expB_effnet_bilstm_mean" / "nested_cv_results.json"
STAGE2_EXP_C_JSON = STAGE2_RESULTS_DIR / "stage2_expC_effnet_bilstm_attention" / "nested_cv_results.json"

STAGE2_RESEARCH_REPORT_MD = REPORTS_DIR / "research" / "PULMOVISION_STAGE_2_RESEARCH_REPORT.md"
STAGE2_AUDIT_REPORT_MD = REPORTS_DIR / "experiments" / "STAGE_2_AUDIT_REPORT.md"
FINAL_AUDIT_REPORT_MD = REPORTS_DIR / "experiments" / "FINAL_IMPLEMENTATION_AUDIT.md"

