"""
PulmoVision — Data & Experiment Result Loaders
Cached utilities for retrieving verified manifests, CV results, and model instances.
Guarantees absolute test-set quarantine (never accesses held-out test cohort).
"""

import json
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from app.utils.paths import (
    DEV_DEDUP_MANIFEST_CSV,
    DEV_DEDUP_MANIFEST_JSON,
    CV_5FOLD_MANIFEST_JSON,
    STAGE2_EXP_A_JSON,
    STAGE2_EXP_B_JSON,
    STAGE2_EXP_C_JSON,
    STAGE2_DEPLOYMENT_CKPT,
)


@st.cache_data
def load_dev_manifest() -> pd.DataFrame:
    """Loads approved Stage 2 deduplicated development manifest (122 patients, 3,132 slices)."""
    if DEV_DEDUP_MANIFEST_CSV.exists():
        df = pd.read_csv(DEV_DEDUP_MANIFEST_CSV)
        return df
    return pd.DataFrame()


@st.cache_data
def load_cv_splits() -> Dict[str, Any]:
    """Loads 5-fold cross-validation split manifest."""
    if CV_5FOLD_MANIFEST_JSON.exists():
        with open(CV_5FOLD_MANIFEST_JSON) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_stage2_experiment_results() -> Dict[str, Any]:
    """Loads authoritative nested CV results for Experiments A, B, and C."""
    results = {}
    mapping = {
        "Exp A (Global Mean)": STAGE2_EXP_A_JSON,
        "Exp B (BiLSTM Mean)": STAGE2_EXP_B_JSON,
        "Exp C (BiLSTM Attn)": STAGE2_EXP_C_JSON,
    }
    for name, path in mapping.items():
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


@st.cache_resource
def get_deployment_predictor():
    """
    Cached predictor instance loading the frozen EfficientNet-B0 backbone
    and the verified deployment checkpoint weights.
    """
    from src.inference.sequence_predictor import SequencePredictor
    return SequencePredictor(checkpoint_path=str(STAGE2_DEPLOYMENT_CKPT))

