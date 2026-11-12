"""app/views/analytics_view.py - Experiment Analytics View (Page 4).

Deep-dive comparative evaluation across Stage 2 architectures using authoritative pooled out-of-fold metrics.
Includes real ROC/PR curves from stored continuous prediction arrays, confusion matrices, and fold stability.
Zero decorative emojis.
"""

import streamlit as st
import pandas as pd
from app.utils.data_loader import load_stage2_experiment_results
from app.components.charts import (
    render_metric_comparison_barchart,
    render_roc_curves_from_oof,
    render_pr_curves_from_oof,
)
from app.components.tables import (
    render_confusion_matrix_card,
    render_cross_fold_summary_table,
    render_threshold_stability_table,
)
from app.components.header import render_page_header


def render_analytics_view() -> None:
    """Renders the comprehensive experimental analytics page."""
    render_page_header(
        title="Stage 2 Cross-Validation Analytics",
        subtitle="Forensic evaluation of 5-fold outer cross-validation across 122 development patients (15 total runs)",
        section="Scientific Research Suite",
    )

    exp_results = load_stage2_experiment_results()

    # 1. Authoritative Benchmark Table
    st.markdown("### Authoritative Pooled Out-of-Fold Performance Summary (N = 122 Patients)")
    st.caption("Aggregates out-of-fold predictions across all 5 outer folds. Brackets indicate 95% empirical bootstrap confidence intervals.")

    summary_rows = [
        {
            "Experiment": "Exp A: EfficientNet-B0 + Global Mean Pooling",
            "Macro F1": "0.8236 [0.7518, 0.8862]",
            "ROC-AUC": "0.8652",
            "PR-AUC": "0.9046",
            "Sensitivity": "77.92% [68.35%, 86.85%]",
            "Specificity": "91.11% [82.00%, 97.78%]",
            "Accuracy": "82.79%",
            "False Negatives": "17 patients",
            "False Positives": "4 patients",
        },
        {
            "Experiment": "Exp B: EfficientNet-B0 + BiLSTM + Mean Pooling [Leader]",
            "Macro F1": "0.9461 [0.8984, 0.9820]",
            "ROC-AUC": "0.9688",
            "PR-AUC": "0.9760",
            "Sensitivity": "98.70% [95.89%, 100.00%]",
            "Specificity": "88.89% [78.72%, 97.62%]",
            "Accuracy": "95.08%",
            "False Negatives": "1 patient",
            "False Positives": "5 patients",
        },
        {
            "Experiment": "Exp C: EfficientNet-B0 + BiLSTM + Additive Attention",
            "Macro F1": "0.9211 [0.8644, 0.9645]",
            "ROC-AUC": "0.9466",
            "PR-AUC": "0.9466",
            "Sensitivity": "93.51% [87.49%, 98.65%]",
            "Specificity": "91.11% [80.85%, 97.87%]",
            "Accuracy": "92.62%",
            "False Negatives": "5 patients",
            "False Positives": "4 patients",
        },
    ]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # 2. Comparative Metric Bar Chart
    st.markdown("### Cross-Architecture Metric Comparison")
    render_metric_comparison_barchart()

    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # 3. Authentic ROC and Precision-Recall Curves
    st.markdown("### Discrimination Curves (Computed from Stored Out-of-Fold Continuous Scores)")
    st.caption("Plotted using genuine continuous out-of-fold patient predictions across all 122 patients. Zero fabricated curves.")

    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.markdown(
            "<p style='font-size: 0.88rem; font-weight: 600; color: var(--pv-text-primary); margin-bottom: 6px;'>Receiver Operating Characteristic (ROC)</p>",
            unsafe_allow_html=True,
        )
        render_roc_curves_from_oof(exp_results)

    with col_pr:
        st.markdown(
            "<p style='font-size: 0.88rem; font-weight: 600; color: var(--pv-text-primary); margin-bottom: 6px;'>Precision-Recall (PR) Curve</p>",
            unsafe_allow_html=True,
        )
        render_pr_curves_from_oof(exp_results)

    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # 4. Separate Confusion Matrices for Experiments A, B, and C
    st.markdown("### Confusion Matrices (Patient-Level Categorical Outcomes)")
    col_cm_a, col_cm_b, col_cm_c = st.columns(3)

    with col_cm_a:
        render_confusion_matrix_card(
            title="Experiment A (Global Mean)",
            tp=60,
            fp=4,
            tn=41,
            fn=17,
            macro_f1=0.8236,
            accent_color="#64748b",
        )
        st.caption("Atemporal pooling missed **17 COVID-19 patients** due to uniform averaging across normal parenchyma slices.")

    with col_cm_b:
        render_confusion_matrix_card(
            title="Experiment B (BiLSTM Mean) [Leader]",
            tp=76,
            fp=5,
            tn=40,
            fn=1,
            macro_f1=0.9461,
            accent_color="var(--pv-accent-primary)",
        )
        st.caption("BiLSTM sequence model detected **76 of 77 COVID-19 patients** (98.70% sensitivity), missing only patient89.")

    with col_cm_c:
        render_confusion_matrix_card(
            title="Experiment C (BiLSTM Attn)",
            tp=72,
            fp=4,
            tn=41,
            fn=5,
            macro_f1=0.9211,
            accent_color="#f59e0b",
        )
        st.caption("Additive attention outperformed atemporal pooling but missed 5 COVID-19 patients, trailing Experiment B.")

    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # 5. Descriptive Cross-Fold Variability
    st.markdown("### Cross-Fold Variability (Mean ± Standard Deviation across 5 Outer Folds)")
    render_cross_fold_summary_table()

    # 6. Threshold Stability Matrix
    st.markdown("### Threshold Selection Stability Matrix")
    st.caption("Optimal decision thresholds selected on inner validation holdouts over candidate grid [0.20, 0.80] in steps of 0.05.")
    render_threshold_stability_table()

    # 7. Methodology & Scientific Limitations Panel
    st.markdown("### Scientific Protocol & Limitations")
    st.markdown(
        """
        <div class="pv-card" style="background-color: var(--pv-bg-elevated);">
            <div style="font-weight: 600; font-size: 0.92rem; color: var(--pv-text-primary); margin-bottom: 6px;">
                Experimental Design & Scope Boundaries
            </div>
            <ul style="font-size: 0.86rem; color: var(--pv-text-secondary); line-height: 1.55; margin: 0; padding-left: 18px;">
                <li><strong>Evaluation Protocol:</strong> 5-fold outer cross-validation with an inner 80/20 validation holdout per outer fold (not full 5&times;5 nested CV). Models were never refitted on outer-training partitions.</li>
                <li><strong>Patient-Balanced Loss:</strong> Multi-series patients were assigned equal weight (1/N_p) per optimizer update, preventing patients with multiple scan series from dominating gradient updates.</li>
                <li><strong>Cohort Scale Limitations:</strong> Development cohort contains 122 patients (77 COVID-19, 45 Normal). Outer validation folds contain only 9 Normal patients each, resulting in discrete specificity steps of ~11.1%.</li>
                <li><strong>No In-Domain Fine-Tuning:</strong> Features were extracted using a frozen ImageNet-1K EfficientNet-B0 backbone to eliminate feature-extractor leakage, but in-domain CT fine-tuning was not explored.</li>
                <li><strong>Quarantine Guarantee:</strong> All 255 patients (8,322 slices) in the held-out test cohort remained strictly unopened.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
