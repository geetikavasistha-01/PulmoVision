# PulmoVision — Stage 2 Research Protocol & Ablation Plan: CNN-BiLSTM

**Document:** `docs/research/cnn_bilstm_experiment_protocol.md`  
**Status:** Protocol Proposal / Pre-Implementation Review  
**Author:** PulmoVision Research Team  
**Dataset Base:** COVID-CTset Development Partition ($122$ patients: $77$ COVID-19, $45$ Normal)  

---

## 1. Research Protocol Objectives

1. Determine whether recurrent bidirectional sequence modeling across ordered axial CT slices provides statistically superior patient-level classification over static 2D and 2.5D slice-aggregation heuristics.
2. Evaluate the specific contribution of **Additive Self-Attention Pooling** versus conventional temporal pooling (mean/max).
3. Prevent data leakage and test-set overfitting by strictly quarantining the official held-out test partition (`test1.csv`).

---

## 2. Experimental Partitions & Validation Methodology

### A. The Validation Cohort Limitation
In Stage 1, our development partition (`train1.csv`, 122 patients) was split into:
- `train_split.csv`: $98$ patients ($62$ COVID, $36$ Normal)
- `val_split.csv`: $24$ patients ($15$ COVID, $9$ Normal)

**Statistical Vulnerability:**
In a $24$-patient validation cohort, a single misclassified COVID patient causes a **$6.7\%$ drop in sensitivity** ($1/15$), while a single misclassified normal patient causes an **$11.1\%$ drop in specificity** ($1/9$). Relying solely on this single split to compare subtle architectural variations risks model selection on random split artifacts.

### B. Proposed Solution: 5-Fold Stratified Patient-Level Cross-Validation
To generate rigorous, publication-grade empirical evidence without touching the test set, we introduce **Stratified 5-Fold Patient-Level Cross-Validation exclusively within the 122 development patients**:

| Fold | Training Patients | Validation Patients | Val COVID Patients | Val Normal Patients |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 97–98 | 24–25 | ~15 | ~9 |
| **Fold 2** | 97–98 | 24–25 | ~15 | ~9 |
| **Fold 3** | 97–98 | 24–25 | ~15 | ~9 |
| **Fold 4** | 97–98 | 24–25 | ~15 | ~9 |
| **Fold 5** | 97–98 | 24–25 | ~15 | ~9 |
| **Total Out-of-Fold** | — | **122 (100% of Dev Cohort)** | **77** | **45** |

- **Out-of-Fold (OOF) Evaluation:** Aggregating validation predictions across all 5 folds provides out-of-fold metrics across **all 122 development patients**, dramatically reducing variance and delivering robust statistical comparisons.
- **Fixed Deterministic Partitioning:** Generated with a fixed seed (`seed=42`) and saved to `data/metadata/splits/cv_5fold_manifests.json`.

### C. Quarantine of Official Held-Out Test Set (`test1.csv`)
- `test1.csv` ($255$ patients, $8,322$ slices) **must not be used** for architecture tuning, hyperparameter search, or early stopping.
- Baseline results on `test1.csv` from Stage 1 (ResNet50 2.5D: Macro F1 $0.8307$, EfficientNet-B0 2.5D: Macro F1 $0.8307$) serve strictly as **frozen historical benchmarks**.
- Once a single champion Stage 2 model is selected based on out-of-fold cross-validation performance, it may be evaluated on `test1.csv` exactly once for final benchmark comparison.

---

## 3. Experimental Matrix

| Experiment ID | Architecture | Input Mode | Sequence Aggregation | Trainable Parameters | Primary Research Question |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Exp-A (Baseline Seq)** | EfficientNet-B0 + BiLSTM | 2D Single Slice | Global Mean Pooling | $\approx 5.48\text{M}$ | Does temporal recurrence alone outperform slice-aggregation heuristics? |
| **Exp-B (Champion Candidate)**| EfficientNet-B0 + BiLSTM + Attention | 2D Single Slice | Additive Self-Attention | $\approx 5.49\text{M}$ | Does attention-weighted sequence pooling improve lesion localization and specificity? |
| **Exp-C (Optional Later)** | EfficientNet-B0 + BiLSTM + Attention | 2.5D Triplet | Additive Self-Attention | $\approx 5.49\text{M}$ | Does combining local through-plane 2.5D context with global recurrence yield further gains? |

---

## 4. Controlled Ablation Study Design

To isolate the source of any empirical performance delta, the ablation matrix decomposes the system into four isolated steps:

```
[Ablation 1] CNN Spatial Features Only
             (Slice-level evaluation + post-hoc positive slice threshold tau)
                    │
                    ▼
[Ablation 2] CNN Spatial Features + Global Temporal Pooling (Mean / Max)
             (Atemporal volume baseline: tests if patient-level training alone helps without recurrence)
                    │
                    ▼
[Ablation 3] CNN Spatial Features + BiLSTM + Global Mean Pooling
             (Sequence baseline: isolates recurrent craniocaudal contextual modeling)
                    │
                    ▼
[Ablation 4] CNN Spatial Features + BiLSTM + Additive Self-Attention Pooling
             (Full proposed system: evaluates dynamic lesion-level saliency weighting)
```

### Ablation Invariant Controls:
1. **Identical Backbone:** Pretrained EfficientNet-B0 feature extractor (embedding dimension $D = 1280$).
2. **Identical Preprocessing:** Uniform 16-bit CT windowing ($[200, 1800]$), scaling to $[0, 1]$, and bilinear interpolation to $224 \times 224$.
3. **Identical Optimizer & Scheduler:** AdamW ($\text{lr}=1\times 10^{-4}$, $\text{weight decay}=1\times 10^{-4}$), Cosine Annealing, early stopping on validation Macro F1.
4. **Identical Patient Stratification:** Tested across the identical 5-fold development splits.

---

## 5. Evaluation Protocol & Statistical Metrics

### Primary Metric: Patient-Level Macro F1
Due to the class imbalance ($77$ COVID vs $45$ Normal in the development set), Macro F1 is the definitive selection metric:
$$\text{Macro F1} = \frac{\text{F1}_{\text{COVID}} + \text{F1}_{\text{Normal}}}{2}$$

### Diagnostic Reporting Suite (Per Experiment):
1. **Patient-Level Metrics:**
   - Balanced Accuracy ($\frac{\text{Sensitivity} + \text{Specificity}}{2}$)
   - Sensitivity (Recall) with **Wilson 95% Score Confidence Intervals**
   - Specificity with **Wilson 95% Score Confidence Intervals**
   - Precision and Negative Predictive Value (NPV)
   - Area Under the ROC Curve (ROC-AUC) computed from continuous output probabilities
   - Full Confusion Matrix ($TP, FP, TN, FN$)
2. **Computational & Operational Metrics:**
   - Parameter Count (CNN backbone vs. BiLSTM/Attention head)
   - Training Duration & Throughput (volumes/second on Apple MPS)
   - Inference Latency per full patient scan (milliseconds per volume)
   - Attention Dispersion (entropy of attention weights $\boldsymbol{\alpha}$ across axial slices)

---

## 6. Pre-Implementation Execution Gates

| Gate | Criterion | Status | Verification Protocol |
| :---: | :--- | :---: | :--- |
| **Gate 1** | Architecture Proposal Review | **PENDING** | Formal review of `cnn_bilstm_architecture.md` by developer. |
| **Gate 2** | Protocol & Split Design Approval | **PENDING** | Formal review of 5-fold development strategy and isolation of `test1.csv`. |
| **Gate 3** | Feature Caching Implementation | Awaiting Gates 1 & 2 | Build `src/experiments/extract_features.py` to extract 1280D embeddings without training. |
| **Gate 4** | Model Code Implementation | Awaiting Gates 1 & 2 | Build `src/models/cnn_bilstm.py` with unit tests for variable-length batches. |
| **Gate 5** | Execution of Experiments A & B | Awaiting Gate 4 | Run 5-fold CV and record metrics in `reports/experiments/stage2_cnn_bilstm/`. |

---
*Submitted for developer review. Implementation will proceed only upon explicit approval.*
