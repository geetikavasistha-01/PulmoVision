"""
PulmoVision — Standardized Scientific Tables Component
Renders authoritative research tables, confusion matrices, and fold-level matrices.
"""

from typing import Dict, Any, List
import streamlit as st
import pandas as pd


def render_confusion_matrix_card(
    title: str,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    macro_f1: float,
    accent_color: str = "var(--pv-accent-primary)"
) -> None:
    """
    Renders a formatted 2x2 confusion matrix with clinical sensitivity and specificity.
    """
    total = tp + fp + tn + fn
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    html = f"""
    <div class="pv-card" style="border-top: 3px solid {accent_color}; margin-bottom: 14px;">
        <div style="font-weight: 600; font-size: 0.92rem; color: var(--pv-text-primary); margin-bottom: 4px;">
            {title}
        </div>
        <div style="font-size: 0.78rem; font-family: var(--pv-font-mono); color: var(--pv-text-secondary); margin-bottom: 12px;">
            Macro F1: <strong style="color: {accent_color};">{macro_f1:.4f}</strong> &bull; Sens: <strong>{sens:.1%}</strong> &bull; Spec: <strong>{spec:.1%}</strong>
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; text-align: center; font-family: var(--pv-font-mono);">
            <thead>
                <tr style="border-bottom: 1px solid var(--pv-border); color: var(--pv-text-muted);">
                    <th style="padding: 6px; text-align: left;">Actual \\ Pred</th>
                    <th style="padding: 6px; color: var(--pv-text-secondary);">Pred Normal</th>
                    <th style="padding: 6px; color: var(--pv-accent-primary);">Pred COVID</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid var(--pv-border_subtle);">
                    <td style="padding: 8px 6px; text-align: left; font-weight: 600; color: var(--pv-text-secondary);">Actual Normal</td>
                    <td style="padding: 8px 6px; background-color: rgba(16, 185, 129, 0.1); color: var(--pv-success); font-weight: 700;">{tn} (TN)</td>
                    <td style="padding: 8px 6px; background-color: rgba(239, 68, 68, 0.1); color: var(--pv-error);">{fp} (FP)</td>
                </tr>
                <tr>
                    <td style="padding: 8px 6px; text-align: left; font-weight: 600; color: var(--pv-text-secondary);">Actual COVID</td>
                    <td style="padding: 8px 6px; background-color: rgba(239, 68, 68, 0.15); color: var(--pv-error); font-weight: 700;">{fn} (FN)</td>
                    <td style="padding: 8px 6px; background-color: rgba(0, 180, 216, 0.15); color: var(--pv-accent-primary); font-weight: 700;">{tp} (TP)</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_cross_fold_summary_table() -> None:
    """Renders cross-fold variability (mean ± std dev across 5 outer folds)."""
    data = [
        {"Metric": "Macro F1 Score", "Exp A (Global Mean)": "0.8218 ± 0.0715", "Exp B (BiLSTM Mean) [Leader]": "0.9454 ± 0.0588", "Exp C (BiLSTM Attn)": "0.9200 ± 0.0740"},
        {"Metric": "Accuracy", "Exp A (Global Mean)": "0.8277 ± 0.0670", "Exp B (BiLSTM Mean) [Leader]": "0.9510 ± 0.0526", "Exp C (BiLSTM Attn)": "0.9263 ± 0.0663"},
        {"Metric": "Sensitivity (Recall)", "Exp A (Global Mean)": "0.7775 ± 0.0798", "Exp B (BiLSTM Mean) [Leader]": "0.9875 ± 0.0280", "Exp C (BiLSTM Attn)": "0.9342 ± 0.0817"},
        {"Metric": "Specificity", "Exp A (Global Mean)": "0.9111 ± 0.1449", "Exp B (BiLSTM Mean) [Leader]": "0.8889 ± 0.1111", "Exp C (BiLSTM Attn)": "0.9111 ± 0.1449"},
        {"Metric": "Precision", "Exp A (Global Mean)": "0.9458 ± 0.0828", "Exp B (BiLSTM Mean) [Leader]": "0.9404 ± 0.0588", "Exp C (BiLSTM Attn)": "0.9542 ± 0.0728"},
        {"Metric": "ROC-AUC", "Exp A (Global Mean)": "0.8726 ± 0.0886", "Exp B (BiLSTM Mean) [Leader]": "0.9804 ± 0.0358", "Exp C (BiLSTM Attn)": "0.9664 ± 0.0635"},
        {"Metric": "PR-AUC", "Exp A (Global Mean)": "0.9292 ± 0.0666", "Exp B (BiLSTM Mean) [Leader]": "0.9883 ± 0.0216", "Exp C (BiLSTM Attn)": "0.9767 ± 0.0449"},
    ]
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, width="stretch")


def render_threshold_stability_table() -> None:
    """Renders fold-by-fold optimal threshold selection matrix."""
    data = [
        {"Outer Fold": "Fold 1", "Exp A Threshold": "0.50", "Exp B Threshold": "0.50", "Exp C Threshold": "0.50"},
        {"Outer Fold": "Fold 2", "Exp A Threshold": "0.50", "Exp B Threshold": "0.50", "Exp C Threshold": "0.50"},
        {"Outer Fold": "Fold 3", "Exp A Threshold": "0.50", "Exp B Threshold": "0.50", "Exp C Threshold": "0.50"},
        {"Outer Fold": "Fold 4", "Exp A Threshold": "0.50", "Exp B Threshold": "0.50", "Exp C Threshold": "0.50"},
        {"Outer Fold": "Fold 5", "Exp A Threshold": "0.50", "Exp B Threshold": "0.50", "Exp C Threshold": "0.30"},
        {"Summary": "Stability", "Exp A Threshold": "100% stable at 0.50", "Exp B Threshold": "100% stable at 0.50", "Exp C Threshold": "80% at 0.50, 20% at 0.30"},
    ]
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, width="stretch")

