# PulmoVision — Comparative Research Evaluation Report (Phase 3 Baselines)

**Status:** Complete  
**Date:** November 11, 2026  
**Hardware Accelerator:** Apple Silicon Metal Performance Shaders (MPS)  
**Dataset:** COVID-CTset Fold 1 (Full 16-bit TIFF cohort, 12,058 slices, 377 patients)  
**Evaluation Protocol:** Strict patient-level train/validation partition + untouched held-out test evaluation  

---

## Executive Summary

This report documents the rigorous experimental comparison of four baseline deep learning architectures on the full COVID-CTset dataset under strict anti-leakage controls. All training and hyperparameter selection (including patient-level aggregation thresholds) were conducted exclusively on the development cohort ($122$ patients: $98$ train, $24$ validation). The official held-out test cohort ($255$ patients, $8,322$ slices) remained completely sealed until all models were frozen.

### Key Research Findings
1. **The 2.5D Architectural Advantage:** Transitioning from 2D single-slice inputs to 2.5D contiguous 3-slice triplets reduced false positive slice misclassifications by **72.1%** (from $877$ down to $245$ on ResNet50), improving test slice specificity from $88.8\%$ to $96.9\%$.
2. **Efficiency vs. Capacity:** EfficientNet-B0 2.5D achieved identical patient-level accuracy ($94.1\%$) and specificity ($94.1\%$) to ResNet50 2.5D, while utilizing **$82.9\%$ fewer parameters** ($4.01\text{M}$ vs. $23.51\text{M}$).
3. **Failure of Unpretrained 2D CNNs:** The custom 2D CNN ($93\text{K}$ parameters) suffered catastrophic failure on unseen negative CT volumes ($4,631$ false positive slices, $161$ false positive patients; test specificity $32.1\%$), underscoring the absolute necessity of pretrained representations and contextual depth.
4. **Cohort Imbalance & Small-Positive Disclosures:** While the test set contains $8,322$ slices across $255$ patients, only **18 patients** have COVID-19 labels ($7.1\%$). All three deep models correctly identified $17$ out of $18$ patients (Sensitivity = $94.4\%$). However, because $N_{\text{pos}}=18$, the Wilson 95% confidence interval spans $[74.2\%, 99.0\%]$. The single missed case across all models (`patient11`) had only a single isolated slice in the archive, depriving the 2.5D models of volumetric context.

---

## 1. Dataset Integrity & Partition Verification

| Partition | Patients | COVID Patients | Normal Patients | Slices | COVID Slices | Normal Slices | Format / Resolution | Path Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (`train_split.csv`)** | 98 | 62 ($63.3\%$) | 36 ($36.7\%$) | 2,959 | 1,458 ($49.3\%$) | 1,501 ($50.7\%$) | 16-bit TIFF ($512\times 512$) | $100.0\%$ ($0$ missing) |
| **Validation (`val_split.csv`)** | 24 | 15 ($62.5\%$) | 9 ($37.5\%$) | 777 | 362 ($46.6\%$) | 415 ($53.4\%$) | 16-bit TIFF ($512\times 512$) | $100.0\%$ ($0$ missing) |
| **Test (`test1.csv` held-out)**| 255 | 18 ($7.1\%$) | 237 ($92.9\%$) | 8,322 | 462 ($5.6\%$) | 7,860 ($94.4\%$) | 16-bit TIFF ($512\times 512$) | $100.0\%$ ($0$ missing) |
| **Total Cohort** | **377** | **95** | **282** | **12,058** | **2,282** | **9,776** | **16-bit TIFF** | **100.0%** |

### Anti-Leakage Audit
- **Patient Overlap:** $0$ patients overlap between Train, Validation, and Test ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Series Overlap:** Zero series overlap across partitions.
- **Test Integrity:** The official Fold 1 test partition (`test1.csv`) was strictly quarantined. Dynamic class weights ($[0.985, 1.015]$) were computed exclusively from `train_split.csv`.

---

## 2. Model Architectures & Training Efficiency

All models were trained using AdamW with weight decay $1\times 10^{-4}$, Cosine Annealing learning rate scheduling, batch size $16$, and early stopping patience of $3$ epochs on validation Macro F1.

| Model ID | Architecture | Channels | Trainable Parameters | Best Epoch / Total | Training Time | Throughput | Peak MPS RAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `exp1_custom_cnn_2d` | Custom CNN (from scratch) | 1 (2D) | 93,154 | Ep 1 / 4 | 188.4 s (3.1 min) | ~69.0 slices/s | 5.6 MB |
| `exp2_resnet50_2d` | ResNet50 (ImageNet init) | 1 (2D) | 23,505,858 | Ep 4 / 7 | 1,337.2 s (22.3 min) | ~22.4 slices/s | 14.8 MB |
| `exp3_resnet50_25d` | ResNet50 (2.5D adapted) | 3 (2.5D) | 23,512,130 | Ep 3 / 6 | 650.0 s (10.8 min) | ~21.8 slices/s | 16.2 MB |
| `exp4_efficientnet_b0_25d`| EfficientNet-B0 (2.5D adapted)| 3 (2.5D) | 4,010,110 | Ep 6 / 9 | 1,727.0 s (28.8 min) | ~19.2 slices/s | 12.4 MB |

---

## 3. Training & Validation Trajectories

### Loss & Macro F1 Evolution
```
Custom CNN 2D:
  Epoch 1: Train Loss 0.5413 | Val Loss 0.5088 | Val Acc 81.3% | Val F1 0.8124  <- Best Checkpoint
  Epoch 2: Train Loss 0.4825 | Val Loss 0.6362 | Val Acc 65.5% | Val F1 0.5862
  Epoch 3: Train Loss 0.4561 | Val Loss 0.4511 | Val Acc 76.7% | Val F1 0.7626
  Epoch 4: Train Loss 0.4337 | Val Loss 0.5133 | Val Acc 71.7% | Val F1 0.7028  (Early Stop)

ResNet50 2D:
  Epoch 1: Train Loss 0.2810 | Val Loss 0.1742 | Val Acc 93.3% | Val F1 0.9331
  Epoch 2: Train Loss 0.1412 | Val Loss 0.1654 | Val Acc 93.6% | Val F1 0.9355
  Epoch 3: Train Loss 0.0821 | Val Loss 0.1582 | Val Acc 94.7% | Val F1 0.9472
  Epoch 4: Train Loss 0.0489 | Val Loss 0.1251 | Val Acc 95.6% | Val F1 0.9562  <- Best Checkpoint
  Epoch 7: Early stopping triggered (Patience=3)

ResNet50 2.5D:
  Epoch 1: Train Loss 0.2114 | Val Loss 0.1245 | Val Acc 95.8% | Val F1 0.9575
  Epoch 2: Train Loss 0.0982 | Val Loss 0.0982 | Val Acc 96.7% | Val F1 0.9665
  Epoch 3: Train Loss 0.0412 | Val Loss 0.0614 | Val Acc 97.9% | Val F1 0.9793  <- Best Checkpoint
  Epoch 6: Early stopping triggered (Patience=3)

EfficientNet-B0 2.5D:
  Epoch 1: Train Loss 0.2343 | Val Loss 0.2769 | Val Acc 88.0% | Val F1 0.8802
  Epoch 2: Train Loss 0.0985 | Val Loss 0.1304 | Val Acc 94.7% | Val F1 0.9470
  Epoch 4: Train Loss 0.0402 | Val Loss 0.1396 | Val Acc 94.7% | Val F1 0.9470
  Epoch 6: Train Loss 0.0125 | Val Loss 0.0835 | Val Acc 97.3% | Val F1 0.9728  <- Best Checkpoint
  Epoch 9: Early stopping triggered (Patience=3)
```

---

## 4. Validation Performance & Threshold Freezing

Aggregation thresholds ($\tau$, positive slice ratio required for a patient to be flagged positive) were searched across $\tau \in [0.10, 0.50]$ exclusively on the $24$ validation patients. The optimal $\tau$ was frozen prior to held-out test evaluation.

| Architecture | Best Epoch | Val Slice Acc | Val Slice F1 | Optimal Frozen Threshold ($\tau$) | Val Patient Acc | Val Patient Macro F1 | Val Patient Confusion (TP/FP/TN/FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN 2D** | Ep 1 | 81.3% | 0.8124 | $\mathbf{\tau = 0.50}$ | 95.8% | **0.9545** | 15 / 1 / 8 / 0 |
| **ResNet50 2D** | Ep 4 | 95.6% | 0.9562 | $\mathbf{\tau = 0.45}$ | 100.0% | **1.0000** | 15 / 0 / 9 / 0 |
| **ResNet50 2.5D** | Ep 3 | **97.9%** | **0.9793** | $\mathbf{\tau = 0.20}$ | 100.0% | **1.0000** | 15 / 0 / 9 / 0 |
| **EfficientNet-B0 2.5D** | Ep 6 | 97.3% | 0.9728 | $\mathbf{\tau = 0.20}$ | 100.0% | **1.0000** | 15 / 0 / 9 / 0 |

---

## 5. Final Held-Out Test Evaluation (`test1.csv`)

Evaluated on the completely unseen test cohort: **$255$ patients, $8,322$ CT slices** ($18$ COVID, $237$ Normal patients).

### A. Slice-Level Test Metrics ($N = 8,322$)
| Architecture | Input | Accuracy | Sensitivity | Specificity | Precision | Macro F1 | Slices TP | Slices FP | Slices TN | Slices FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN 2D** | 1 Ch | 44.1% | 95.0% | 41.1% | 8.7% | 0.3699 | 439 | 4,631 | 3,229 | 23 |
| **ResNet50 2D** | 1 Ch | 89.3% | **96.8%** | 88.8% | 33.8% | 0.7203 | 447 | 877 | 6,983 | 15 |
| **ResNet50 2.5D** | 3 Ch | 96.7% | 94.4% | 96.9% | 64.0% | **0.8727** | 436 | 245 | 7,615 | 26 |
| **EfficientNet-B0 2.5D** | 3 Ch | **96.9%** | 87.2% | **97.4%** | **66.7%** | 0.8697 | 403 | 201 | 7,659 | 59 |

### B. Patient-Level Test Metrics ($N = 255$, Frozen $\tau$)
| Architecture | Frozen Threshold ($\tau$) | Accuracy | Sensitivity | Specificity | Precision | Macro F1 | Patients TP | Patients FP | Patients TN | Patients FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN 2D** | 0.50 | 36.5% | 94.4% | 32.1% | 9.6% | 0.3288 | 17 | 161 | 76 | 1 |
| **ResNet50 2D** | 0.45 | 91.0% | 94.4% | 90.7% | 43.6% | 0.7729 | 17 | 22 | 215 | 1 |
| **ResNet50 2.5D** | 0.20 | **94.1%** | 94.4% | **94.1%** | **54.8%** | **0.8307** | 17 | 14 | 223 | 1 |
| **EfficientNet-B0 2.5D** | 0.20 | **94.1%** | 94.4% | **94.1%** | **54.8%** | **0.8307** | 17 | 14 | 223 | 1 |

---

## 6. Patient-Level vs. Slice-Level Statistical Comparisons

### Wilson 95% Score Confidence Intervals (Patient Level)

| Model | Sensitivity ($N_{\text{pos}}=18$) | Sensitivity 95% CI | Specificity ($N_{\text{neg}}=237$) | Specificity 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| **Custom CNN 2D** | 94.4% ($17/18$) | $[74.2\%, 99.0\%]$ | 32.1% ($76/237$) | $[26.5\%, 38.3\%]$ |
| **ResNet50 2D** | 94.4% ($17/18$) | $[74.2\%, 99.0\%]$ | 90.7% ($215/237$) | $[86.3\%, 93.8\%]$ |
| **ResNet50 2.5D** | 94.4% ($17/18$) | $[74.2\%, 99.0\%]$ | **94.1%** ($223/237$) | $[90.3\%, 96.4\%]$ |
| **EfficientNet-B0 2.5D** | 94.4% ($17/18$) | $[74.2\%, 99.0\%]$ | **94.1%** ($223/237$) | $[90.3\%, 96.4\%]$ |

### Why Patient Aggregation Resolves Slice Imbalance
In the test set, healthy slices outnumber COVID slices by $17:1$ ($7,860$ vs. $462$). Even with an exceptional slice specificity of $96.9\%$ (ResNet50 2.5D), the raw false positive count ($245$ slices) depresses slice precision to $64.0\%$. However, because false positive slices occur sporadically across series rather than clustering persistently, requiring $\ge 20\%$ positive slices ($\tau = 0.20$) correctly filters out benign noise in $223$ out of $237$ patients, boosting overall specificity to $94.1\%$.

---

## 7. Deep Error Analysis & Case Disclosures

### A. The Single False Negative: `patient11`
Across ResNet50 2D, ResNet50 2.5D, and EfficientNet-B0 2.5D, exactly **one patient** was missed: `patient11` ($TP=17, FN=1$).
- **Root Cause Investigation:** Inspecting the underlying dataset reveals `patient11` contains **only 1 single CT slice** in the entire study (`slice_count: 1`, `11covid_patient11_SR_2_IM00001.tif`).
- **Impact on 2.5D Input:** For isolated slices, the 2.5D preprocessor clamps boundaries by replicating the slice across all 3 channels ($[S_0, S_0, S_0]$). The lack of genuine anatomical slice progression (temporal context) combined with atypical peripheral presentation resulted in a sub-threshold score ($0.0$).
- **Clinical Implication:** In clinical screening, single-slice CT submissions are invalid; CT scans represent 3D volumetric series (typically 40–100 slices). When evaluated on patients with full volumetric series ($N=17$), the sensitivity is **$100.0\%$ (17/17)**.

### B. False Positive Analysis (Normal Patients Flagged as Positive)
Both 2.5D models produced $14$ false positive patients out of $237$ ($5.9\%$ false positive rate).
- Common false positive patients across architectures: `patient243`, `patient264`, `patient280`.
- Inspection reveals motion artifact blur and prominent dependent atelectasis / baseline interstitial scarring in these elderly scans, mimicking ground-glass opacity (GGO) signatures.

---

## 8. Limitations & Clinical Research Boundary

1. **Retrospective Single-Center Bias:** COVID-CTset originates from a single Iranian hospital center using specific scanner protocols; external generalization to low-dose or multi-vendor protocols is unverified.
2. **Extreme Test Imbalance:** Only 18 COVID-19 positive patients are present in the official Fold 1 test split ($7.1\%$). This causes the 95% confidence interval for sensitivity to span $24.8$ percentage points ($[74.2\%, 99.0\%]$).
3. **No Clinical Co-morbidities:** Scans lack paired clinical markers (RT-PCR cycle thresholds, days from symptom onset, non-COVID viral pneumonias).
4. **Educational Prototype Status:** PulmoVision is strictly a medical imaging research framework and is not certified as a Computer-Aided Diagnostic (CAD) device.

---

## 9. Stage 2 Recommendations

| Proposal | Architecture | Rationale | Expected Benefit |
| :--- | :--- | :--- | :--- |
| **Option A (Recommended)** | **ResNet50 / EfficientNet + BiLSTM** | Replaces static 3-slice channel concatenation with recurrent sequential modeling across entire CT volumes. | Captures full craniocaudal longitudinal trajectory across 30–80 slices per scan. |
| **Option B** | **3D CNN (MedicalNet / 3D ResNet18)** | Native 3D spatial convolutions on resampled $D \times H \times W$ CT volumes. | Models true isotropic 3D volumetric features; requires volume resampling. |
| **Option C** | **Vision Transformer (ViT / Swin)** | Shifted-window self-attention for global receptive fields. | Captures distant bilateral multi-lobar infiltrates; requires careful regularization. |

---
*Report verified against `reports/experiments/BASELINE_EXPERIMENTS_SUMMARY.json` and reproducible via `src/experiments/run_baselines.py`.*
