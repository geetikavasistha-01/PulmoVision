"""
PulmoVision — Interactive Scientific Data Visualizations
Generates charts using Altair and Streamlit-native components with authentic verified data.
Zero fabricated curves; uses stored out-of-fold continuous predictions for ROC and PR curves.
"""

from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import roc_curve, precision_recall_curve


def render_class_distribution_chart(df_manifest: pd.DataFrame) -> None:
    """Renders patient-level categorical archive label distribution."""
    if df_manifest.empty:
        st.info("Manifest data unavailable.")
        return

    # Count unique patients by label
    patient_classes = (
        df_manifest.groupby("patient_id")["label"]
        .first()
        .value_counts()
        .reset_index()
    )
    patient_classes.columns = ["Archive Label", "Patient Count"]
    patient_classes["Label Name"] = patient_classes["Archive Label"].map({1: "COVID-19", 0: "Normal"})
    patient_classes["Percentage"] = patient_classes["Patient Count"] / patient_classes["Patient Count"].sum()

    chart = (
        alt.Chart(patient_classes)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Label Name:N", title="Categorical Archive Label", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Patient Count:Q", title="Unique Patients"),
            color=alt.Color(
                "Label Name:N",
                scale=alt.Scale(domain=["COVID-19", "Normal"], range=["#00b4d8", "#64748b"]),
                legend=None
            ),
            tooltip=["Label Name", "Patient Count", alt.Tooltip("Percentage:Q", format=".1%")]
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_sequence_length_histogram(df_manifest: pd.DataFrame) -> None:
    """Renders histogram of unique slice counts per scan series."""
    if df_manifest.empty:
        st.info("Manifest data unavailable.")
        return

    # Slices per series
    slices_per_series = (
        df_manifest.groupby(["patient_id", "series_id"])
        .size()
        .reset_index(name="Slice Count")
    )

    chart = (
        alt.Chart(slices_per_series)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#38bdf8")
        .encode(
            x=alt.X("Slice Count:Q", bin=alt.Bin(maxbins=25), title="Slices per Series (Sequence Length T)"),
            y=alt.Y("count():Q", title="Number of Scan Series"),
            tooltip=[alt.Tooltip("count()", title="Series Count"), alt.Tooltip("Slice Count", bin=True)]
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_metric_comparison_barchart() -> None:
    """Renders authoritative Macro F1, ROC-AUC, Sensitivity, and Specificity across Experiments A, B, and C."""
    data = [
        {"Experiment": "Exp A (Global Mean)", "Metric": "Macro F1", "Score": 0.8236},
        {"Experiment": "Exp A (Global Mean)", "Metric": "ROC-AUC", "Score": 0.8652},
        {"Experiment": "Exp A (Global Mean)", "Metric": "Sensitivity", "Score": 0.7792},
        {"Experiment": "Exp A (Global Mean)", "Metric": "Specificity", "Score": 0.9111},
        
        {"Experiment": "Exp B (BiLSTM Mean) 🏆", "Metric": "Macro F1", "Score": 0.9461},
        {"Experiment": "Exp B (BiLSTM Mean) 🏆", "Metric": "ROC-AUC", "Score": 0.9688},
        {"Experiment": "Exp B (BiLSTM Mean) 🏆", "Metric": "Sensitivity", "Score": 0.9870},
        {"Experiment": "Exp B (BiLSTM Mean) 🏆", "Metric": "Specificity", "Score": 0.8889},
        
        {"Experiment": "Exp C (BiLSTM Attn)", "Metric": "Macro F1", "Score": 0.9211},
        {"Experiment": "Exp C (BiLSTM Attn)", "Metric": "ROC-AUC", "Score": 0.9466},
        {"Experiment": "Exp C (BiLSTM Attn)", "Metric": "Sensitivity", "Score": 0.9351},
        {"Experiment": "Exp C (BiLSTM Attn)", "Metric": "Specificity", "Score": 0.9111},
    ]
    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0.6, 1.0])),
            color=alt.Color(
                "Experiment:N",
                scale=alt.Scale(
                    domain=["Exp A (Global Mean)", "Exp B (BiLSTM Mean) 🏆", "Exp C (BiLSTM Attn)"],
                    range=["#64748b", "#00b4d8", "#f59e0b"]
                ),
                legend=alt.Legend(orient="bottom", title=None)
            ),
            xOffset="Experiment:N",
            tooltip=["Experiment", "Metric", alt.Tooltip("Score:Q", format=".4f")]
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def render_roc_curves_from_oof(exp_results: Dict[str, Any]) -> None:
    """
    Renders actual ROC Curves computed directly from stored continuous out-of-fold predictions.
    Guarantees scientific integrity: zero fabricated curves.
    """
    roc_records = []
    
    for exp_name, res in exp_results.items():
        if "oof_predictions" in res:
            oof = res["oof_predictions"]
            y_true = [item["true_label"] for item in oof.values()]
            y_scores = [item["pred_prob"] for item in oof.values()]

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            for f, t in zip(fpr, tpr):
                roc_records.append({
                    "Architecture": exp_name,
                    "False Positive Rate": float(f),
                    "True Positive Rate": float(t)
                })

    if not roc_records:
        st.info("Out-of-fold score arrays unavailable for ROC curve rendering.")
        return

    df_roc = pd.DataFrame(roc_records)

    # Diagonal reference line
    diag = pd.DataFrame({"False Positive Rate": [0.0, 1.0], "True Positive Rate": [0.0, 1.0]})
    chart_diag = (
        alt.Chart(diag)
        .mark_line(strokeDash=[4, 4], color="#475569", strokeWidth=1)
        .encode(x="False Positive Rate:Q", y="True Positive Rate:Q")
    )

    chart_roc = (
        alt.Chart(df_roc)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("False Positive Rate:Q", title="False Positive Rate (1 - Specificity)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("True Positive Rate:Q", title="True Positive Rate (Sensitivity)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Architecture:N",
                scale=alt.Scale(
                    domain=["Exp A (Global Mean)", "Exp B (BiLSTM Mean)", "Exp C (BiLSTM Attn)"],
                    range=["#64748b", "#00b4d8", "#f59e0b"]
                ),
                legend=alt.Legend(orient="bottom", title=None)
            ),
            tooltip=["Architecture", alt.Tooltip("False Positive Rate:Q", format=".3f"), alt.Tooltip("True Positive Rate:Q", format=".3f")]
        )
    )

    final_chart = (chart_diag + chart_roc).properties(height=340)
    st.altair_chart(final_chart, use_container_width=True)


def render_pr_curves_from_oof(exp_results: Dict[str, Any]) -> None:
    """
    Renders actual Precision-Recall Curves computed directly from stored continuous out-of-fold predictions.
    """
    pr_records = []
    
    for exp_name, res in exp_results.items():
        if "oof_predictions" in res:
            oof = res["oof_predictions"]
            y_true = [item["true_label"] for item in oof.values()]
            y_scores = [item["pred_prob"] for item in oof.values()]

            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            for p, r in zip(prec, rec):
                pr_records.append({
                    "Architecture": exp_name,
                    "Recall": float(r),
                    "Precision": float(p)
                })

    if not pr_records:
        st.info("Out-of-fold score arrays unavailable for PR curve rendering.")
        return

    df_pr = pd.DataFrame(pr_records)

    chart_pr = (
        alt.Chart(df_pr)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("Recall:Q", title="Recall (Sensitivity)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0.6, 1.0])),
            color=alt.Color(
                "Architecture:N",
                scale=alt.Scale(
                    domain=["Exp A (Global Mean)", "Exp B (BiLSTM Mean)", "Exp C (BiLSTM Attn)"],
                    range=["#64748b", "#00b4d8", "#f59e0b"]
                ),
                legend=alt.Legend(orient="bottom", title=None)
            ),
            tooltip=["Architecture", alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")]
        )
        .properties(height=340)
    )
    st.altair_chart(chart_pr, use_container_width=True)

