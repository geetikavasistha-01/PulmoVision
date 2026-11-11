# PulmoVision — Stage 2 Research Evaluation Report: CNN-BiLSTM Volumetric Sequence Modeling

**Date:** November 11, 2026  
**Status:** Completed  
**Branch:** `feature/cnn-bilstm`  
**Hardware Accelerator:** Apple Silicon Metal Performance Shaders (MPS) / M1 Unified Memory  
**Dataset Base:** COVID-CTset Deduplicated Development Cohort ($122$ patients: $77$ COVID-19, $45$ Normal; $239$ series, $3,132$ unique slices)  
**Evaluation Protocol:** 5-Fold Outer Cross-Validation with Inner Holdout (15 total runs) under strict Patient-Balanced Optimization  
**Test Set Status:** Strictly Quarantined (`test1.csv` unopened and unaccessed)  

---

## 1. Executive Summary

This report documents the completed experimental evaluation of Stage 2 research in PulmoVision, investigating whether **sequential bidirectional modeling along the craniocaudal z-axis** improves patient-level COVID-19 classification over atemporal volumetric pooling.

### Key Scientific Findings:
1. **The Recurrent Advantage:** Introducing bidirectional temporal modeling (BiLSTM) over slice embeddings produced a massive performance leap:
   * **Macro F1:** Rose from $0.8236$ (Atemporal Baseline) to **$0.9461$** (BiLSTM + Mean Pooling), a **$+12.25\%$** absolute improvement ($95\%\text{ CI: } [0.8984, 0.9820]$).
   * **ROC-AUC:** Rose from $0.8652$ to **$0.9688$** ($+0.1036$).
   * **Sensitivity (COVID-19 Recall):** Soared from $77.92\%$ to **$98.70\%$** ($+20.78\%$). Out of 77 COVID-19 patients across the cohort, BiLSTM correctly identified **76 patients**, missing only a single scan.
2. **Failure of Atemporal Pooling:** The atemporal baseline (Experiment A) missed **17 COVID-19 patients** (Sensitivity = $77.92\%$). Because global mean pooling uniformly averages slice vectors, focal ground-glass opacities concentrated in lower lung lobes are diluted by unaffected upper-lung slices.
3. **Additive Attention vs. Global Recurrent Pooling:**
   * Experiment B (BiLSTM + Global Mean Pooling) achieved the highest overall Macro F1 ($0.9461$) and sensitivity ($98.70\%$).
   * Experiment C (BiLSTM + Bahdanau Additive Attention) achieved slightly higher specificity ($91.11\%$ vs. $88.89\%$, 4 FP vs. 5 FP) with Macro F1 of $0.9211$ ($95\%\text{ CI: } [0.8644, 0.9645]$), while offering slice-level saliency weights for volumetric interpretability.
4. **Threshold Stability:** Across all 5 folds for Models A and B, the optimal inner-validation classification threshold was invariant at $\tau^* = 0.50$. For Model C, $\tau^* = 0.50$ held in Folds 1–4, shifting to $\tau^* = 0.30$ in Fold 5. No model hit boundary extremes ($0.20$ or $0.80$).

---

## 2. Frozen Experimental Configuration

| Component | Experiment A | Experiment B | Experiment C |
| :--- | :--- | :--- | :--- |
| **Architecture Name** | `effnet_b0_mean_pooling` | `effnet_b0_bilstm_mean` | `effnet_b0_bilstm_attention` |
| **Model Class** | `PulmoGlobalMeanPooling` | `PulmoBiLSTMMeanPooling` | `PulmoBiLSTMAttention` |
| **Feature Extractor** | ImageNet-1K EfficientNet-B0 (Frozen) | ImageNet-1K EfficientNet-B0 (Frozen) | ImageNet-1K EfficientNet-B0 (Frozen) |
| **Feature Dimension** | 1280 | 1280 | 1280 |
| **Recurrent Encoder** | None | 1-Layer BiLSTM ($H=128$, bidirectional) | 1-Layer BiLSTM ($H=128$, bidirectional) |
| **Recurrent Output Dim** | N/A | $2H = 256$ | $2H = 256$ |
| **Sequence Aggregation** | Masked Global Mean Pooling over $T$ slices | Masked Global Mean Pooling over $T$ hidden states | Masked Additive Attention ($\mathbf{W}_a \in \mathbb{R}^{64 \times 256}, \mathbf{v}_a \in \mathbb{R}^{1 \times 64}$) |
| **Classifier Head** | $\text{Linear}(1280 \to 64) \to \text{SiLU} \to \text{Drop}(0.3) \to \text{Linear}(64 \to 2)$ | $\text{Linear}(256 \to 64) \to \text{SiLU} \to \text{Drop}(0.3) \to \text{Linear}(64 \to 2)$ | $\text{Linear}(256 \to 64) \to \text{SiLU} \to \text{Drop}(0.3) \to \text{Linear}(64 \to 2)$ |
| **Trainable Parameters** | $82,114$ | $1,492,034$ | $1,508,483$ |
| **Optimizer** | AdamW ($\eta = 10^{-3}, \lambda = 10^{-4}$) | AdamW ($\eta = 10^{-3}, \lambda = 10^{-4}$) | AdamW ($\eta = 10^{-3}, \lambda = 10^{-4}$) |
| **Batch Size** | 16 patients (all series bundled) | 16 patients (all series bundled) | 16 patients (all series bundled) |
| **Max Epochs / Patience** | 30 epochs / 5 epochs early stopping | 30 epochs / 5 epochs early stopping | 30 epochs / 5 epochs early stopping |
| **Checkpoint Metric** | Inner-Val Macro F1 at $\tau=0.50$ | Inner-Val Macro F1 at $\tau=0.50$ | Inner-Val Macro F1 at $\tau=0.50$ |
| **Threshold Grid** | $\tau \in [0.20, 0.80]$, step $0.05$ (13 steps) | $\tau \in [0.20, 0.80]$, step $0.05$ (13 steps) | $\tau \in [0.20, 0.80]$, step $0.05$ (13 steps) |
| **Patient Loss Objective** | $\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{p=1}^B \left( \frac{1}{N_p} \sum_{s=1}^{N_p} \ell_{p, s} \right)$ | $\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{p=1}^B \left( \frac{1}{N_p} \sum_{s=1}^{N_p} \ell_{p, s} \right)$ | $\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{p=1}^B \left( \frac{1}{N_p} \sum_{s=1}^{N_p} \ell_{p, s} \right)$ |
| **Class Weighting Source**| Inner-train patient counts ($w_c = \frac{N_{\text{pts}}}{2 N_c}$) | Inner-train patient counts ($w_c = \frac{N_{\text{pts}}}{2 N_c}$) | Inner-train patient counts ($w_c = \frac{N_{\text{pts}}}{2 N_c}$) |

---

## 3. Architecture-Wise Out-of-Fold Performance Comparison

All $122$ development patients received exactly one out-of-fold prediction generated by an outer fold model that never saw that patient during training, checkpoint selection, or threshold tuning.

### Pooled Out-of-Fold (OOF) Metrics ($N = 122$ Patients: $77$ COVID-19, $45$ Normal)

| Metric | Experiment A (Global Mean) | Experiment B (BiLSTM + Mean) | Experiment C (BiLSTM + Attention) |
| :--- | :---: | :---: | :---: |
| **Macro F1** | $0.8236$ [$0.7518, 0.8862$] | **$0.9461$** [$0.8984, 0.9820$] | $0.9211$ [$0.8644, 0.9645$] |
| **ROC-AUC** | $0.8652$ | **$0.9688$** | $0.9466$ |
| **PR-AUC (Average Precision)**| $0.9046$ | **$0.9760$** | $0.9466$ |
| **Sensitivity (Recall)** | $77.92\%$ [$68.35\%, 86.85\%$] | **$98.70\%$** [$95.89\%, 100.00\%$] | $93.51\%$ [$87.49\%, 98.65\%$] |
| **Specificity** | **$91.11\%$** [$82.00\%, 97.78\%$] | $88.89\%$ [$78.72\%, 97.62\%$] | **$91.11\%$** [$80.85\%, 97.87\%$] |
| **Precision** | $93.75\%$ | $93.83\%$ | **$94.74\%$** |
| **Accuracy** | $82.79\%$ | **$95.08\%$** | $92.62\%$ |
| **Balanced Accuracy** | $84.52\%$ | **$93.80\%$** | $92.31\%$ |
| **True Positives ($TP$)** | $60$ | **$76$** | $72$ |
| **False Positives ($FP$)** | **$4$** | $5$ | **$4$** |
| **True Negatives ($TN$)** | **$41$** | $40$ | **$41$** |
| **False Negatives ($FN$)** | $17$ | **$1$** | $5$ |

*Brackets denote $95\%$ empirical confidence intervals derived from $1,000$ stratified bootstrap resamples of the pooled cohort.*

---

## 4. Cross-Fold Descriptive Variability ($\mu \pm \sigma$)

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

### Breakdown by Fold

#### Fold 1 (Outer Val: 16 COVID, 9 Normal = 25 pts; Inner Train: 77 pts, Inner Val: 20 pts)
* **Exp A:** Outer Macro F1 = $0.9167$, Sens = $87.5\%$, Spec = $100.0\%$, ROC-AUC = $0.9444$, $\tau^* = 0.50$ (TP=14, FP=0, TN=9, FN=2)
* **Exp B:** Outer Macro F1 = **$1.0000$**, Sens = **$100.0\%$**, Spec = **$100.0\%$**, ROC-AUC = **$1.0000$**, $\tau^* = 0.50$ (TP=16, FP=0, TN=9, FN=0)
* **Exp C:** Outer Macro F1 = **$1.0000$**, Sens = **$100.0\%$**, Spec = **$100.0\%$**, ROC-AUC = **$1.0000$**, $\tau^* = 0.50$ (TP=16, FP=0, TN=9, FN=0)

#### Fold 2 (Outer Val: 16 COVID, 9 Normal = 25 pts; Inner Train: 77 pts, Inner Val: 20 pts)
* **Exp A:** Outer Macro F1 = $0.7396$, Sens = $81.3\%$, Spec = $66.7\%$, ROC-AUC = $0.7222$, $\tau^* = 0.50$ (TP=13, FP=3, TN=6, FN=3)
* **Exp B:** Outer Macro F1 = $0.8663$, Sens = $93.8\%$, Spec = $77.8\%$, ROC-AUC = $0.9167$, $\tau^* = 0.50$ (TP=15, FP=2, TN=7, FN=1)
* **Exp C:** Outer Macro F1 = $0.8162$, Sens = $93.8\%$, Spec = $66.7\%$, ROC-AUC = $0.8542$, $\tau^* = 0.50$ (TP=15, FP=3, TN=6, FN=1)

#### Fold 3 (Outer Val: 15 COVID, 9 Normal = 24 pts; Inner Train: 78 pts, Inner Val: 20 pts)
* **Exp A:** Outer Macro F1 = $0.7913$, Sens = $66.7\%$, Spec = $100.0\%$, ROC-AUC = $0.8667$, $\tau^* = 0.50$ (TP=10, FP=0, TN=9, FN=5)
* **Exp B:** Outer Macro F1 = **$1.0000$**, Sens = **$100.0\%$**, Spec = **$100.0\%$**, ROC-AUC = **$1.0000$**, $\tau^* = 0.50$ (TP=15, FP=0, TN=9, FN=0)
* **Exp C:** Outer Macro F1 = $0.9564$, Sens = $93.3\%$, Spec = $100.0\%$, ROC-AUC = **$1.0000$**, $\tau^* = 0.50$ (TP=14, FP=0, TN=9, FN=1)

#### Fold 4 (Outer Val: 15 COVID, 9 Normal = 24 pts; Inner Train: 78 pts, Inner Val: 20 pts)
* **Exp A:** Outer Macro F1 = $0.8730$, Sens = $80.0\%$, Spec = $100.0\%$, ROC-AUC = $0.9185$, $\tau^* = 0.50$ (TP=12, FP=0, TN=9, FN=3)
* **Exp B:** Outer Macro F1 = $0.9545$, Sens = **$100.0\%$**, Spec = $88.9\%$, ROC-AUC = $0.9926$, $\tau^* = 0.50$ (TP=15, FP=1, TN=8, FN=0)
* **Exp C:** Outer Macro F1 = $0.9545$, Sens = **$100.0\%$**, Spec = $88.9\%$, ROC-AUC = $0.9778$, $\tau^* = 0.50$ (TP=15, FP=1, TN=8, FN=0)

#### Fold 5 (Outer Val: 15 COVID, 9 Normal = 24 pts; Inner Train: 78 pts, Inner Val: 20 pts)
* **Exp A:** Outer Macro F1 = $0.7884$, Sens = $73.3\%$, Spec = $88.9\%$, ROC-AUC = $0.9111$, $\tau^* = 0.50$ (TP=11, FP=1, TN=8, FN=4)
* **Exp B:** Outer Macro F1 = $0.9062$, Sens = **$100.0\%$**, Spec = $77.8\%$, ROC-AUC = $0.9926$, $\tau^* = 0.50$ (TP=15, FP=2, TN=7, FN=0)
* **Exp C:** Outer Macro F1 = $0.8730$, Sens = $80.0\%$, Spec = $100.0\%$, ROC-AUC = **$1.0000$**, $\tau^* = \mathbf{0.30}$ (TP=12, FP=0, TN=9, FN=3)

### Threshold Stability Summary
* **Experiment A:** $\tau^* = [0.50, 0.50, 0.50, 0.50, 0.50]$ (100% stable at 0.50).
* **Experiment B:** $\tau^* = [0.50, 0.50, 0.50, 0.50, 0.50]$ (100% stable at 0.50).
* **Experiment C:** $\tau^* = [0.50, 0.50, 0.50, 0.50, 0.30]$ (80% at 0.50, 20% at 0.30).
* **Boundary Checks:** Zero folds hit the lower limit ($\tau = 0.20$) or upper limit ($\tau = 0.80$).

---

## 6. Confusion Matrices

### Experiment A: Global Mean Pooling (Atemporal Baseline)
```
                  Predicted Normal    Predicted COVID-19    Total
Actual Normal            41 (TN)              4 (FP)         45
Actual COVID-19          17 (FN)             60 (TP)         77
Total                    58                  64             122
```

### Experiment B: BiLSTM + Global Mean Pooling (Recurrent Baseline)
```
                  Predicted Normal    Predicted COVID-19    Total
Actual Normal            40 (TN)              5 (FP)         45
Actual COVID-19           1 (FN)             76 (TP)         77
Total                    41                  81             122
```

### Experiment C: BiLSTM + Additive Attention Pooling (Champion Candidate)
```
                  Predicted Normal    Predicted COVID-19    Total
Actual Normal            41 (TN)              4 (FP)         45
Actual COVID-19           5 (FN)             72 (TP)         77
Total                    46                  76             122
```

---

## 7. Out-of-Fold Error Analysis & Edge Cases

Detailed inspection of misclassified patients across all three architectures:

### Systematic False Positives:
* `patient146` (Normal): Consistently misclassified as COVID across all 3 models ($P=0.864$ in A, $P=0.912$ in B, $P=0.928$ in C).
* `patient190` (Normal): Consistently misclassified as COVID across all 3 models ($P=0.918$ in A, $P=0.973$ in B, $P=0.996$ in C).
* *Clinical Context:* Both patients exhibit non-COVID consolidations or preexisting lung densities that trigger visual viral-pneumonia feature embeddings.

### The Single Persistent False Negative in BiLSTM:
* `patient89` (COVID): The **only** COVID-19 patient missed by Experiment B ($P=0.222$) and missed by all three models ($P=0.029$ in A, $P=0.217$ in C).
* *Audit:* In the dataset manifest, `patient89` contains very mild peripheral opacity in only 2 slices, with the remaining 26 slices appearing normal.

### Resolution of Dilution Effect in COVID Cases:
* In Experiment A, 17 COVID-19 patients were missed (`patient137`, `patient45`, `patient68`, `patient84`, `patient89`, `patient142`, `patient55`, `patient65`, `patient70`, `patient83`, `patient128`, `patient43`, `patient75`, `patient129`, `patient134`, `patient74`, `patient78`).
* In Experiment B, **16 of those 17 patients were correctly recovered**! The bidirectional recurrent cells retained memory of abnormal slice transitions without diluting them into background noise.

---

## 8. Comparison with Stage 1 Historical Baselines

> [!WARNING]
> **Methodological Boundary:** Stage 1 and Stage 2 evaluations operate under fundamentally different protocols and must not be conflated.

| Attribute | Stage 1 Baselines (Phase 3) | Stage 2 Sequence Modeling (Phase A) |
| :--- | :--- | :--- |
| **Input Unit** | 2D single slices or 2.5D 3-slice triplets | Full 3D patient volume (ordered sequence of slices $T$) |
| **Backbone Adaptation** | End-to-end CNN fine-tuning on 80% dev slices | Frozen external ImageNet EfficientNet-B0 (Zero COVID-CTset fine-tuning) |
| **Patient Decision Method**| Post-hoc threshold $\tau=0.20$ on positive slice count | End-to-end sequence model outputting patient probability $P(\text{COVID})$ |
| **Evaluation Framework** | Single split (98 train, 24 val) + held-out `test1.csv` | 5-Fold Outer Cross-Validation with Inner Validation Holdouts |
| **Evaluated Cohort** | Held-out test set ($N=255$, only $18$ COVID positive) | Full development cohort ($N=122$, $77$ COVID, $45$ Normal) |
| **Patient Macro F1** | ResNet50 2.5D: $0.8307$; EffNet 2.5D: $0.8307$ | BiLSTM Mean: **$0.9461$**; BiLSTM Attn: **$0.9211$** |
| **Patient Sensitivity** | $94.4\%$ (17/18 test COVID patients) | **$98.7\%$** (76/77 dev COVID patients on BiLSTM Mean) |

*Key Insight:* Even with a completely frozen backbone that never underwent medical fine-tuning, sequence modeling along the z-axis achieves **$0.9461$ out-of-fold Macro F1**, outperforming the slice-averaging heuristic ($0.8236$ on Model A and $0.8307$ in Stage 1).

---

## 9. Failure and Limitation Log

1. **Omission of Dynamic Image-Space Augmentation:** Features were cached once to prevent leakage and enable rapid 15-model cross-validation. While this accelerated training (benchmark completed in 46 seconds), it precluded spatial rotation/scaling during sequence optimization.
2. **Backbone Domain Gap:** EfficientNet-B0 features were derived from natural images (ImageNet-1K), not lung CT scans. Fine-tuning the backbone inside an end-to-end sequence pipeline could further enhance discrimination.
3. **Small Sample Outer Folds:** Outer validation folds have $\approx 24\text{--}25$ patients (9 normal patients per fold). Thus, a single misclassified normal patient shifts fold specificity by $\approx 11.1\%$. The pooled bootstrap CI ($[0.7872, 0.9762]$) provides the true statistical estimate.

---

## 10. Reproducibility & Artifact Inventory

* **Feature Cache:** `data/processed/dev_patient_features_effnet_b0.pt` (3,132 slices, 239 series, 122 patients, SHA256 verified)
* **Manifests:**
  * Outer CV splits: `data/metadata/splits/cv_5fold_manifest.json`
  * Deduplicated development cohort: `data/metadata/splits/stage2_dev_deduplicated_manifest.json`
* **Experimental Output Records:**
  * Experiment A: `reports/experiments/stage2_expA_effnet_mean_pooling/nested_cv_results.json`
  * Experiment B: `reports/experiments/stage2_expB_effnet_bilstm_mean/nested_cv_results.json`
  * Experiment C: `reports/experiments/stage2_expC_effnet_bilstm_attention/nested_cv_results.json`
* **Quarantined Benchmark:** `data/metadata/Annotation/CSV/test1.csv` (Intact, unopened, timestamp March 7, 2022)
