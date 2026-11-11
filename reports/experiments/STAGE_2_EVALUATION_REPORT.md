# PulmoVision — Stage 2 Research Evaluation Report: Patient-Level Craniocaudal Sequence Modelling

**Date:** November 11, 2026  
**Status:** Completed & Forensically Audited  
**Branch:** `feature/cnn-bilstm`  
**Hardware Accelerator:** Apple Silicon Metal Performance Shaders (MPS) / M1 Unified Memory  
**Dataset Base:** COVID-CTset Deduplicated Development Cohort ($122$ patients: $77$ COVID-19, $45$ Normal; $239$ series, $3,132$ unique slices)  
**Evaluation Protocol:** 5-Fold Outer Cross-Validation with Inner Validation Holdout (15 total runs) under strict Patient-Balanced Optimization  
**Test Set Status:** Strictly Quarantined (`test1.csv` unopened and unaccessed; SHA256: `31dc5260...`, timestamp March 7, 2022)  

> [!IMPORTANT]
> **Educational Research Prototype Boundary:** PulmoVision is strictly an academic AI research prototype designed for educational exploration of sequence architectures on medical imaging datasets. It is **NOT** a certified medical diagnostic device, clinical decision support system, or clinically validated software. Dataset labels represent categorical archive annotations, not confirmed clinical diagnoses.

---

## 1. Executive Summary

This report documents the completed experimental evaluation of Stage 2 research in PulmoVision, investigating **patient-level craniocaudal sequence modelling using ordered CT slice embeddings** compared with atemporal sequence pooling.

### Key Empirical Findings on the Development Cohort:
1. **Recurrent Sequence Modelling (Experiment B):** Introducing bidirectional temporal modelling (BiLSTM) over slice embeddings yielded the strongest performance observed across the development cohort:
   * **Macro F1:** Rose from $0.8236$ (Atemporal Baseline, Exp A) to **$0.9461$** (BiLSTM + Mean Pooling, Exp B), a **$+12.25\%$** absolute improvement ($95\%\text{ CI: } [0.8984, 0.9820]$).
   * **ROC-AUC:** Rose from $0.8652$ to **$0.9688$** ($+0.1036$).
   * **Sensitivity (COVID-19 Recall):** Rose from $77.92\%$ to **$98.70\%$** ($+20.78\%$). Out of 77 COVID-19 patients across the development cohort, Experiment B correctly classified **76 patients**, with exactly 1 false negative.
2. **Performance of Atemporal Mean Pooling (Experiment A):** The atemporal baseline missed **17 COVID-19 patients** (Sensitivity = $77.92\%$). A plausible hypothesis requiring further radiological investigation is that uniform averaging over 30–40 slice feature vectors dilutes localized visual abnormalities into background slice embeddings.
3. **Additive Attention Outcome (Experiment C):**
   * Experiment C (BiLSTM + Additive Attention) achieved a Macro F1 of $0.9211$ ($95\%\text{ CI: } [0.8644, 0.9645]$), outperforming the atemporal baseline ($0.8236$) but performing below BiLSTM with uniform mean pooling ($0.9461$).
   * In this experimental setting, **additive attention did not demonstrate an incremental performance benefit over BiLSTM mean pooling**. While attention weights provide slice-level saliency distributions, they introduced additional parameterization without improving classification accuracy on this cohort size.
4. **Conclusion on Architecture Leader:** Based strictly on the development cohort ($N=122$), **Experiment B (`effnet_b0_bilstm_mean`) is the current performance leader**. However, this finding reflects empirical results on a single partitioned development cohort and cannot be claimed as universally superior across diverse scanner protocols or uncalibrated populations.

---

## 2. Frozen Experimental Configuration

| Component | Experiment A | Experiment B | Experiment C |
| :--- | :--- | :--- | :--- |
| **Architecture Name** | `effnet_b0_mean_pooling` | `effnet_b0_bilstm_mean` | `effnet_b0_bilstm_attention` |
| **Model Class** | `PulmoGlobalMeanPooling` | `PulmoBiLSTMMeanPooling` | `PulmoBiLSTMAttention` |
| **Feature Extractor** | ImageNet-1K EfficientNet-B0 (Frozen) | ImageNet-1K EfficientNet-B0 (Frozen) | ImageNet-1K EfficientNet-B0 (Frozen) |
| **Feature Dimension ($D$)** | 1280 | 1280 | 1280 |
| **Recurrent Encoder** | None | 1-Layer BiLSTM ($H=128$, bidirectional) | 1-Layer BiLSTM ($H=128$, bidirectional) |
| **Recurrent Output Dim** | N/A | $2H = 256$ | $2H = 256$ |
| **Sequence Aggregation** | Masked Global Mean Pooling over $T$ slices | Masked Global Mean Pooling over $T$ hidden states | Masked Additive Attention ($\mathbf{W}_a \in \mathbb{R}^{64 \times 256}, \mathbf{v}_a \in \mathbb{R}^{1 \times 64}$) |
| **Classifier Head** | $\text{Linear}(1280 \to 64) \to \text{SiLU} \to \text{Drop}(0.3) \to \text{Linear}(64 \to 2)$ | $\text{Linear}(256 \to 64) \to \text{SiLU} \to \text{Drop}(0.3) \to \text{Linear}(64 \to 2)$ | $\text{Linear}(256 \to 64) \to \text{SiLU} \to \text{Drop}(0.3) \to \text{Linear}(64 \to 2)$ |
| **Trainable Parameters** | $82,114$ | $1,492,034$ | $1,508,483$ |
| **Optimizer** | AdamW ($\eta = 10^{-3}, \lambda = 10^{-4}$) | AdamW ($\eta = 10^{-3}, \lambda = 10^{-4}$) | AdamW ($\eta = 10^{-3}, \lambda = 10^{-4}$) |
| **Batch Size ($B$)** | 16 patients (all series bundled per patient) | 16 patients (all series bundled per patient) | 16 patients (all series bundled per patient) |
| **Max Epochs / Patience** | 30 epochs / 5 epochs early stopping | 30 epochs / 5 epochs early stopping | 30 epochs / 5 epochs early stopping |
| **Checkpoint Metric** | Inner-Val Macro F1 at $\tau=0.50$ | Inner-Val Macro F1 at $\tau=0.50$ | Inner-Val Macro F1 at $\tau=0.50$ |
| **Threshold Grid** | $\tau \in [0.20, 0.80]$, step $0.05$ (13 steps) | $\tau \in [0.20, 0.80]$, step $0.05$ (13 steps) | $\tau \in [0.20, 0.80]$, step $0.05$ (13 steps) |
| **Patient Loss Objective** | $\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{p=1}^B \left( \frac{1}{N_p} \sum_{s=1}^{N_p} \ell_{p, s} \right)$ | $\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{p=1}^B \left( \frac{1}{N_p} \sum_{s=1}^{N_p} \ell_{p, s} \right)$ | $\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{p=1}^B \left( \frac{1}{N_p} \sum_{s=1}^{N_p} \ell_{p, s} \right)$ |
| **Class Weighting Source**| Inner-train patient counts ($w_c = \frac{N_{\text{pts}}}{2 N_c}$) | Inner-train patient counts ($w_c = \frac{N_{\text{pts}}}{2 N_c}$) | Inner-train patient counts ($w_c = \frac{N_{\text{pts}}}{2 N_c}$) |

---

## 3. Architecture-Wise Performance Comparison

Each development patient received **exactly one out-of-fold (OOF) prediction** generated by an outer fold model trained without that patient.

### Pooled Out-of-Fold (OOF) Metrics ($N = 122$ Patients: $77$ COVID-19, $45$ Normal)

| Metric | Experiment A (`effnet_mean_pooling`) | Experiment B (`effnet_bilstm_mean`) | Experiment C (`effnet_bilstm_attention`) |
| :--- | :---: | :---: | :---: |
| **Macro F1** | $0.8236$ [$0.7518, 0.8862$] | **$0.9461$** [$0.8984, 0.9820$] | $0.9211$ [$0.8644, 0.9645$] |
| **ROC-AUC** | $0.8652$ | **$0.9688$** | $0.9466$ |
| **PR-AUC (Avg Precision)** | $0.9046$ | **$0.9760$** | $0.9466$ |
| **Sensitivity (Recall)** | $77.92\%$ [$68.35\%, 86.85\%$] | **$98.70\%$** [$95.89\%, 100.00\%$] | $93.51\%$ [$87.49\%, 98.65\%$] |
| **Specificity** | **$91.11\%$** [$82.00\%, 97.78\%$] | $88.89\%$ [$78.72\%, 97.62\%$] | **$91.11\%$** [$80.85\%, 97.87\%$] |
| **Precision** | $93.75\%$ | $93.83\%$ | **$94.74\%$** |
| **Accuracy** | $82.79\%$ | **$95.08\%$** | $92.62\%$ |
| **Balanced Accuracy** | $84.52\%$ | **$93.80\%$** | $92.31\%$ |
| **True Positives ($TP$)** | $60$ | **$76$** | $72$ |
| **False Positives ($FP$)** | **$4$** | $5$ | **$4$** |
| **True Negatives ($TN$)** | **$41$** | $40$ | **$41$** |
| **False Negatives ($FN$)** | $17$ | **$1$** | $5$ |

> [!NOTE]
> **Statistical Interpretation of Pooled OOF Metrics & Confidence Intervals:**
> 1. **Nature of Pooled OOF AUROC:** Pooled AUROC aggregates predictions across the 5 outer folds, where each fold model was trained on a slightly different inner-training partition ($\approx 77\text{--}78$ patients). It reflects the rank-order performance of the cross-validation system as a whole, rather than a single static parameterization.
> 2. **Conditional Interpretation of Bootstrap CIs:** The reported $95\%$ empirical confidence intervals ($1,000$ stratified bootstrap resamples) are **conditional on the observed cohort predictions**. They quantify sample-size sampling uncertainty within this specific dataset of 122 patients; they do **not** capture all sources of training-procedure variance (such as alternative feature extractor backbones, scan acquisition variation, or different split random seeds).

---

## 4. Descriptive Cross-Fold Variability ($\mu \pm \sigma$ across 5 outer folds)

Reporting fold mean and standard deviation characterizes the sensitivity of each architecture to different patient partitionings:

| Metric | Experiment A | Experiment B | Experiment C |
| :--- | :---: | :---: | :---: |
| **Macro F1** | $0.8218 \pm 0.0715$ | **$0.9454 \pm 0.0588$** | $0.9200 \pm 0.0740$ |
| **Accuracy** | $0.8277 \pm 0.0670$ | **$0.9510 \pm 0.0526$** | $0.9263 \pm 0.0663$ |
| **Sensitivity** | $0.7775 \pm 0.0798$ | **$0.9875 \pm 0.0280$** | $0.9342 \pm 0.0817$ |
| **Specificity** | $0.9111 \pm 0.1449$ | $0.8889 \pm 0.1111$ | **$0.9111 \pm 0.1449$** |
| **Precision** | $0.9458 \pm 0.0828$ | $0.9404 \pm 0.0588$ | **$0.9542 \pm 0.0728$** |
| **ROC-AUC** | $0.8726 \pm 0.0886$ | **$0.9804 \pm 0.0358$** | $0.9664 \pm 0.0635$ |
| **PR-AUC** | $0.9292 \pm 0.0666$ | **$0.9883 \pm 0.0216$** | $0.9767 \pm 0.0449$ |

---

## 5. Per-Fold Training Outcomes & Threshold Stability Analysis

### Fold-by-Fold Performance Matrix

```
Fold 1 (Outer Val: 16 COVID, 9 Normal = 25 pts; Inner Train: 77 pts, Inner Val: 20 pts)
  Exp A: Macro F1 = 0.9167 | Sens = 87.5% | Spec = 100.0% | AUC = 0.9444 | Tau* = 0.50 (TP=14, FP=0, TN=9, FN=2)
  Exp B: Macro F1 = 1.0000 | Sens = 100.0%| Spec = 100.0% | AUC = 1.0000 | Tau* = 0.50 (TP=16, FP=0, TN=9, FN=0)
  Exp C: Macro F1 = 1.0000 | Sens = 100.0%| Spec = 100.0% | AUC = 1.0000 | Tau* = 0.50 (TP=16, FP=0, TN=9, FN=0)

Fold 2 (Outer Val: 16 COVID, 9 Normal = 25 pts; Inner Train: 77 pts, Inner Val: 20 pts)
  Exp A: Macro F1 = 0.7396 | Sens = 81.3% | Spec = 66.7%  | AUC = 0.7222 | Tau* = 0.50 (TP=13, FP=3, TN=6, FN=3)
  Exp B: Macro F1 = 0.8663 | Sens = 93.8% | Spec = 77.8%  | AUC = 0.9167 | Tau* = 0.50 (TP=15, FP=2, TN=7, FN=1)
  Exp C: Macro F1 = 0.8162 | Sens = 93.8% | Spec = 66.7%  | AUC = 0.8542 | Tau* = 0.50 (TP=15, FP=3, TN=6, FN=1)

Fold 3 (Outer Val: 15 COVID, 9 Normal = 24 pts; Inner Train: 78 pts, Inner Val: 20 pts)
  Exp A: Macro F1 = 0.7913 | Sens = 66.7% | Spec = 100.0% | AUC = 0.8667 | Tau* = 0.50 (TP=10, FP=0, TN=9, FN=5)
  Exp B: Macro F1 = 1.0000 | Sens = 100.0%| Spec = 100.0% | AUC = 1.0000 | Tau* = 0.50 (TP=15, FP=0, TN=9, FN=0)
  Exp C: Macro F1 = 0.9564 | Sens = 93.3% | Spec = 100.0% | AUC = 1.0000 | Tau* = 0.50 (TP=14, FP=0, TN=9, FN=1)

Fold 4 (Outer Val: 15 COVID, 9 Normal = 24 pts; Inner Train: 78 pts, Inner Val: 20 pts)
  Exp A: Macro F1 = 0.8730 | Sens = 80.0% | Spec = 100.0% | AUC = 0.9185 | Tau* = 0.50 (TP=12, FP=0, TN=9, FN=3)
  Exp B: Macro F1 = 0.9545 | Sens = 100.0%| Spec = 88.9%  | AUC = 0.9926 | Tau* = 0.50 (TP=15, FP=1, TN=8, FN=0)
  Exp C: Macro F1 = 0.9545 | Sens = 100.0%| Spec = 88.9%  | AUC = 0.9778 | Tau* = 0.50 (TP=15, FP=1, TN=8, FN=0)

Fold 5 (Outer Val: 15 COVID, 9 Normal = 24 pts; Inner Train: 78 pts, Inner Val: 20 pts)
  Exp A: Macro F1 = 0.7884 | Sens = 73.3% | Spec = 88.9%  | AUC = 0.9111 | Tau* = 0.50 (TP=11, FP=1, TN=8, FN=4)
  Exp B: Macro F1 = 0.9062 | Sens = 100.0%| Spec = 77.8%  | AUC = 0.9926 | Tau* = 0.50 (TP=15, FP=2, TN=7, FN=0)
  Exp C: Macro F1 = 0.8730 | Sens = 80.0% | Spec = 100.0% | AUC = 1.0000 | Tau* = 0.30 (TP=12, FP=0, TN=9, FN=3)
```

### Threshold Stability Summary
* **Experiment A:** $\tau^* = [0.50, 0.50, 0.50, 0.50, 0.50]$ (100% stable at canonical prior 0.50).
* **Experiment B:** $\tau^* = [0.50, 0.50, 0.50, 0.50, 0.50]$ (100% stable at canonical prior 0.50).
* **Experiment C:** $\tau^* = [0.50, 0.50, 0.50, 0.50, 0.30]$ (80% at 0.50, 20% at 0.30).
* **Boundary Checks:** Zero folds selected boundary limits ($0.20$ or $0.80$), confirming that optimization produced well-calibrated probability distributions.

---

## 6. Confusion Matrices

### Experiment A: Global Mean Pooling (Atemporal Baseline)
```
                          PREDICTED
                     Normal      COVID-19       Total
ACTUAL  Normal       41 (TN)       4 (FP)        45
        COVID-19     17 (FN)      60 (TP)        77
        Total        58           64            122
```

### Experiment B: BiLSTM + Global Mean Pooling (Development Leader)
```
                          PREDICTED
                     Normal      COVID-19       Total
ACTUAL  Normal       40 (TN)       5 (FP)        45
        COVID-19      1 (FN)      76 (TP)        77
        Total        41           81            122
```

### Experiment C: BiLSTM + Additive Attention Pooling
```
                          PREDICTED
                     Normal      COVID-19       Total
ACTUAL  Normal       41 (TN)       4 (FP)        45
        COVID-19      5 (FN)      72 (TP)        77
        Total        46           76            122
```

---

## 7. Out-of-Fold Error Analysis & Limitations

Detailed inspection of misclassified cases across the architectures:

### Systematic Normal False Positives
Two normal patients were consistently assigned high COVID-19 probability across all three models:
* `patient146`: $P = 0.864$ (Exp A), $0.912$ (Exp B), $0.928$ (Exp C).
* `patient190`: $P = 0.918$ (Exp A), $0.973$ (Exp B), $0.996$ (Exp C).
* *Caveat on Radiological Interpretation:* Visual inspection of the underlying TIFF slices indicates high-density visual patterns and structural irregularities. However, **no radiological diagnosis of fibrosis or non-COVID consolidation can be made without independent review by a board-certified radiologist**. These cases demonstrate that visual features alone, derived from an ImageNet backbone without clinical priors, can confound general lung abnormalities with viral pneumonia patterns.

### The Single False Negative in Experiment B
* `patient89`: The single COVID-19 patient missed by BiLSTM ($P = 0.222$), also missed by Exp A ($P = 0.029$) and Exp C ($P = 0.217$).
* *Observation:* In `patient89`, visual abnormalities in slice embeddings appear confined to a very small proportion of the scan volume, indicating an edge case where sequence-level aggregation produces borderline scores.

### Observation on Sequential Representation
* Experiment A missed 17 COVID-19 patients, whereas Experiment B missed only 1.
* While it is plausible that sequential modeling along the craniocaudal axis mitigates the dilution of localized abnormalities across slices, **this remains an empirical hypothesis requiring formal multi-dataset validation and radiologist-annotated lesion localization before any definitive conclusion can be drawn**.

---

## 8. Comparison with Stage 1 Historical Baselines

> [!WARNING]
> **Methodological Boundary:** Stage 1 and Stage 2 evaluations operate under fundamentally different protocols and must not be directly equated.

| Dimension | Stage 1 Baselines (Phase 3) | Stage 2 Sequence Modeling (Phase A) |
| :--- | :--- | :--- |
| **Input Unit** | 2D single slices or 2.5D 3-slice triplets | Full patient sequence of ordered slices $T$ |
| **Backbone Adaptation** | End-to-end CNN fine-tuning on 80% dev slices | **Frozen ImageNet-1K EfficientNet-B0** (Zero in-domain fine-tuning) |
| **Patient Decision Method** | Post-hoc heuristic threshold ($\tau=0.20$) on slice count | End-to-end sequence model outputting calibrated $P(\text{COVID})$ |
| **Evaluation Framework** | Single split (98 train, 24 val) + held-out `test1.csv` | **5-Fold Outer Cross-Validation with Inner Holdout** (15 runs) |
| **Evaluated Cohort** | Held-out test set ($N=255$, only $18$ COVID positive) | Full development cohort ($N=122$, $77$ COVID, $45$ Normal) |
| **Patient Macro F1** | ResNet50 2.5D: $0.8307$; EffNet 2.5D: $0.8307$ | Exp A: $0.8236$; **Exp B: $0.9461$**; Exp C: $0.9211$ |
| **Patient Sensitivity** | $94.4\%$ (17/18 test COVID patients) | **$98.70\%$** (76/77 dev COVID patients on BiLSTM Mean) |

---

## 9. Failure and Limitation Log

1. **Omission of Image-Space Augmentation:** Embeddings were extracted once with a frozen backbone to prevent leakage and enable rapid 15-model cross-validation ($46.46\text{s}$ total run time). However, this precluded dynamic spatial augmentations during sequence optimization.
2. **Backbone Domain Gap:** EfficientNet-B0 was pretrained on natural objects (ImageNet-1K), not lung CT scans. Feature representations may lack sensitivity to subtle pulmonary textural shifts.
3. **Partition Size Constraints:** Each outer validation fold contains only 9 normal patients. Consequently, a single false positive shifts fold specificity by $\approx 11.1\%$. The pooled bootstrap interval ($[0.7872, 0.9762]$) provides the proper statistical bound on sampling variability.
4. **Intermediate Checkpoint Serialization:** Model state dicts for each fold were managed in memory for checkpoint selection and outer evaluation, but were not serialized to individual `.pth` files on disk (all metrics, parameters, and predictions are recorded in `nested_cv_results.json`).

---

## 10. Reproducibility & Artifact Inventory

| Artifact Description | File Path | File Size | SHA256 Checksum | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Deduplicated Development Cache** | `data/processed/dev_patient_features_effnet_b0.pt` | 15.67 MB | `74bb243d7484482940108d5051ab78a265a67752e796be32aac391d8618f3005` | Verified |
| **Outer 5-Fold Partition Manifest** | `data/metadata/splits/cv_5fold_manifest.json` | 18.5 KB | `a7d806cbf6e60b45d259fa89ee21ffb1a13b0c670b89aa6f95c0291fc22d645e` | Verified |
| **Deduplicated Dev Manifest (JSON)**| `data/metadata/splits/stage2_dev_deduplicated_manifest.json` | 913 KB | `da0e527d21c43b9e4a833d7b6862594a1dff5c9ba6602febe49fbc94473855ff` | Verified |
| **Deduplicated Dev Manifest (CSV)** | `data/metadata/splits/stage2_dev_deduplicated_manifest.csv` | 277 KB | `fc256a47b194f107f9ff90d1bf430fe359a1f28b49bcfdc11202868ceab44f6f` | Verified |
| **Experiment A Results JSON** | `reports/experiments/stage2_expA_effnet_mean_pooling/nested_cv_results.json` | 30.7 KB | `44a0106de340211853ca27e02c5e440a318838ec4df81466efdd15a1366bb441` | Verified |
| **Experiment B Results JSON** | `reports/experiments/stage2_expB_effnet_bilstm_mean/nested_cv_results.json` | 30.4 KB | `2ebb1acd12a99902df4532ac1e3e65717b81e3622871812e186bc85934340297` | Verified |
| **Experiment C Results JSON** | `reports/experiments/stage2_expC_effnet_bilstm_attention/nested_cv_results.json` | 30.9 KB | `376d86cf55450ff29d1376094d2e81693ed2e953349f38ab986cda8fcc61b393` | Verified |
| **Intermediate Model Checkpoints** | Individual fold `.pth` files | N/A | *Not serialized to disk by runner (held in memory during CV)* | **Noted** |
| **Sealed Test Set Manifest** | `data/metadata/Annotation/CSV/test1.csv` | 359 KB | `31dc526085d109580e0e7e9c87dcd3deda878d3c37b05a10245105c616dd8282` | **Quarantined** |
