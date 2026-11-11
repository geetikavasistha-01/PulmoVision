# PulmoVision — Stage 2 Results & Artifact Audit Report

**Audit Date:** November 11, 2026  
**Auditor:** PulmoVision Independent Audit Suite  
**Branch:** `feature/cnn-bilstm`  
**Latest Baseline Commit:** `021145a`  
**Hardware Accelerator:** Apple Silicon Metal Performance Shaders (MPS)  
**Quarantine Boundary:** `data/metadata/Annotation/CSV/test1.csv` (100% Intact, Unopened)  

---

## 1. Audit Scope & Verification Gates

This audit independently inspects, validates, and re-computes all artifacts and metrics resulting from Stage 2 sequence modeling cross-validation across 15 total runs (3 architectures $\times$ 5 outer folds).

| Audit Item | Verification Method | Outcome | Status |
| :--- | :--- | :--- | :---: |
| **Run Completion** | Verified 5 outer fold evaluations per architecture across Experiments A, B, and C. | Exactly 15 runs completed without error. | **PASS** |
| **OOF Cohort Size** | Counted unique patient predictions in `oof_predictions` dictionary for each model. | Exactly 122 patients per architecture ($100\%$ cohort). | **PASS** |
| **Zero Patient Overlap** | Verified set intersection of inner-train, inner-val, and outer-val patient sets per fold. | $0$ patient overlap across partitions in all 5 folds. | **PASS** |
| **Held-Out Test Quarantine** | Inspected `test1.csv` modification timestamp, size, and SHA256 checksum. | Untouched (March 7, 2022; SHA256: `31dc5260...`). Zero test reads. | **PASS** |
| **Stage 1 Baselines** | Checked `git diff main..HEAD` for `reports/experiments/exp1-4` and checkpoints. | $0$ lines of diff. Stage 1 baselines 100% identical to `main`. | **PASS** |
| **Threshold Range** | Inspected all selected thresholds across the 15 runs. | No fold hit boundary extremes ($0.20$ or $0.80$). Folds 1–5 in Exp A & B selected $0.50$; Exp C selected $0.50$ (Folds 1–4) and $0.30$ (Fold 5). | **PASS** |
| **Metric Re-computation** | Re-computed confusion matrices, F1, accuracy, sensitivity, specificity, AUROC, PR-AUC from raw patient predictions. | $100\%$ match against reported JSON records to machine precision. | **PASS** |
| **Intermediate Checkpoints** | Checked filesystem for intermediate `.pth` checkpoint files for individual folds. | *Explicitly marked*: Fold checkpoints were managed in memory during CV and not serialized to individual `.pth` files. | **NOTED** |

---

## 2. Independent Metric Reconciliation

Metrics were recalculated from the raw patient predictions stored in each experiment's `nested_cv_results.json`:

```
================================================================================
EXPERIMENT A (Atemporal Global Mean Pooling Baseline)
================================================================================
Metric                      Independent Calc      JSON Record       Discrepancy
--------------------------------------------------------------------------------
True Positives (TP)              60                   60                 0
False Positives (FP)              4                    4                 0
True Negatives (TN)              41                   41                 0
False Negatives (FN)             17                   17                 0
Accuracy                     0.827869             0.827869           0.00e+00
Sensitivity (Recall)         0.779221             0.779221           0.00e+00
Specificity                  0.911111             0.911111           0.00e+00
Precision                    0.937500             0.937500           0.00e+00
Balanced Accuracy            0.845166             0.845166           0.00e+00
Macro F1                     0.823590             0.823590           0.00e+00
ROC-AUC                      0.865224             0.865224           0.00e+00
PR-AUC                       0.904628             0.904628           0.00e+00
Status: PERFECT MATCH (0 discrepancy)

================================================================================
EXPERIMENT B (BiLSTM + Global Mean Pooling — Development Leader)
================================================================================
Metric                      Independent Calc      JSON Record       Discrepancy
--------------------------------------------------------------------------------
True Positives (TP)              76                   76                 0
False Positives (FP)              5                    5                 0
True Negatives (TN)              40                   40                 0
False Negatives (FN)              1                    1                 0
Accuracy                     0.950820             0.950820           0.00e+00
Sensitivity (Recall)         0.987013             0.987013           0.00e+00
Specificity                  0.888889             0.888889           0.00e+00
Precision                    0.938272             0.938272           0.00e+00
Balanced Accuracy            0.937951             0.937951           0.00e+00
Macro F1                     0.946129             0.946129           1.11e-16
ROC-AUC                      0.968831             0.968831           0.00e+00
PR-AUC                       0.976040             0.976040           0.00e+00
Status: PERFECT MATCH (0 discrepancy)

================================================================================
EXPERIMENT C (BiLSTM + Additive Attention Pooling)
================================================================================
Metric                      Independent Calc      JSON Record       Discrepancy
--------------------------------------------------------------------------------
True Positives (TP)              72                   72                 0
False Positives (FP)              4                    4                 0
True Negatives (TN)              41                   41                 0
False Negatives (FN)              5                    5                 0
Accuracy                     0.926230             0.926230           0.00e+00
Sensitivity (Recall)         0.935065             0.935065           0.00e+00
Specificity                  0.911111             0.911111           0.00e+00
Precision                    0.947368             0.947368           0.00e+00
Balanced Accuracy            0.923088             0.923088           0.00e+00
Macro F1                     0.921138             0.921138           1.11e-16
ROC-AUC                      0.946609             0.946609           0.00e+00
PR-AUC                       0.946614             0.946614           0.00e+00
Status: PERFECT MATCH (0 discrepancy)
```

---

## 3. Methodological and Statistical Clarifications

1. **Pooled Out-of-Fold AUROC:**
   The pooled AUROC metric ($0.8652$ in A, $0.9688$ in B, $0.9466$ in C) aggregates out-of-fold continuous predictions across the 5 outer folds. Because each fold model was optimized on a distinct inner training partition ($\approx 77\text{--}78$ patients), the pooled AUROC characterizes the discrimination of the cross-validation system across the full development cohort, rather than a single fixed model checkpoint.
2. **Conditional Interpretation of Bootstrap CIs:**
   The empirical $95\%$ bootstrap confidence intervals ($1,000$ stratified resamples) are strictly conditional on the observed predictions for these 122 development patients. They reflect sampling error within this cohort; they do not represent universal confidence intervals encompassing scanner heterogeneity, demographic variance, or alternative backbone pretrainings.
3. **Scientific Terminology:**
   * Replaced all references to "full 3D model" or "3D CNN" with:  
     **"Patient-level craniocaudal sequence modelling using ordered CT slice embeddings."**
   * Mitigating the dilution of localized abnormalities is designated as a **plausible empirical hypothesis requiring independent multi-dataset validation**, not a proven clinical fact.
   * Visual irregularities in outlier normal cases (`patient146`, `patient190`) are documented as visual feature anomalies, not clinical diagnoses of consolidation or fibrosis.

---

## 4. Architectural Findings Summary

* **Experiment A (Atemporal Mean Pooling Baseline):** Macro F1 = $0.8236$, missed 17 COVID-19 patients.
* **Experiment B (BiLSTM + Mean Pooling):** Macro F1 = **$0.9461$**, Sensitivity = **$98.70\%$** (missed 1 COVID-19 patient). This architecture demonstrated the strongest performance observed on the development cohort.
* **Experiment C (BiLSTM + Additive Attention):** Macro F1 = $0.9211$, Sensitivity = $93.51\%$. **Additive attention did not demonstrate an incremental benefit over uniform BiLSTM mean pooling in this experiment**, as the additional projection parameters did not translate to superior classification on this sample size.

---

## 5. Artifact Inventory & Cryptographic Checksums

| Description | Absolute Path | SHA256 Checksum |
| :--- | :--- | :--- |
| **Deduplicated Features** | `data/processed/dev_patient_features_effnet_b0.pt` | `74bb243d7484482940108d5051ab78a265a67752e796be32aac391d8618f3005` |
| **5-Fold Splits Manifest** | `data/metadata/splits/cv_5fold_manifest.json` | `a7d806cbf6e60b45d259fa89ee21ffb1a13b0c670b89aa6f95c0291fc22d645e` |
| **Experiment A Results** | `reports/experiments/stage2_expA_effnet_mean_pooling/nested_cv_results.json` | `44a0106de340211853ca27e02c5e440a318838ec4df81466efdd15a1366bb441` |
| **Experiment B Results** | `reports/experiments/stage2_expB_effnet_bilstm_mean/nested_cv_results.json` | `2ebb1acd12a99902df4532ac1e3e65717b81e3622871812e186bc85934340297` |
| **Experiment C Results** | `reports/experiments/stage2_expC_effnet_bilstm_attention/nested_cv_results.json` | `376d86cf55450ff29d1376094d2e81693ed2e953349f38ab986cda8fcc61b393` |
| **Quarantined Test Set** | `data/metadata/Annotation/CSV/test1.csv` | `31dc526085d109580e0e7e9c87dcd3deda878d3c37b05a10245105c616dd8282` |

**Conclusion:** All 15 cross-validation runs, metrics, and partition invariants are 100% verified and mathematically validated under strict quarantine controls.
