# PulmoVision — Stage 2 Research Progress & Implementation Audit

**Branch:** `feature/cnn-bilstm`  
**Date:** November 11, 2026  
**Status:** Infrastructure & Deduplication Complete — Ready for Nested CV Authorization  

---

## 1. Repository & Environment Audit

| Component | Verified Specification | Status |
| :--- | :--- | :---: |
| **Active Git Branch** | `feature/cnn-bilstm` (created from clean `main`) | ✅ Verified |
| **Total Branch Count** | 5 local branches (`feature/cnn-bilstm`, `feature/model-development`, `feature/evaluation-xai`, `feature/streamlit-deployment`, `main`) | ✅ Verified (5-branch max preserved) |
| **Python Environment** | Python 3.13.2 (macOS ARM64) | ✅ Verified |
| **PyTorch & Torchvision** | PyTorch 2.13.0, torchvision 0.28.0 | ✅ Verified |
| **Hardware Accelerator** | Apple Silicon Metal Performance Shaders (`mps` available & built) | ✅ Verified |
| **Pretrained Weights ID** | `EfficientNet_B0_Weights.IMAGENET1K_V1` (`efficientnet_b0_rwightman-7f5810bc.pth`) | ✅ Verified (External ImageNet-1K) |
| **Pretrained Weights SHA256** | `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934` | ✅ Verified |
| **Development Manifest** | `train1.csv` — exactly 3,736 slice records across 122 patients | ✅ Verified (Unmodified) |
| **Test Manifest** | `test1.csv` — exactly 8,322 slice records across 255 patients | ✅ Verified (Quarantined) |
| **Stage 1 Checkpoints** | Exp 1 (1.1 MB), Exp 2 (282.5 MB), Exp 3 (282.6 MB), Exp 4 (48.6 MB) | ✅ Verified (Unchanged) |

---

## 2. Invariants & Execution Gates

1. **Feature Extraction:** External ImageNet-1K EfficientNet-B0 only. Zero in-domain exposure.
2. **Preprocessing:** 16-bit radiodensity clipping $[200, 1800]$, scaling to $[0, 1]$, channel replication $R=G=B$, ImageNet standardization, bilinear resize to $224 \times 224$.
3. **Anatomical De-duplication (Option 1):** Verified byte-identical redundant files across archive subfolders (`normal1` vs `normal5` / `normal4`) resolved to 1 canonical record with full duplicate alias provenance.
4. **Cross-Validation:** 5-Fold Outer Cross-Validation with Inner Validation Holdout; outer validation evaluated once for out-of-fold predictions generated without outer-validation leakage.
5. **Equal Patient Weighting:** Patient-balanced loss aggregation $\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{p=1}^B \left( \frac{1}{N_p} \sum_{s=1}^{N_p} \ell_{p, s} \right)$ guarantees equal patient influence across optimizer updates regardless of series count.
6. **Test Quarantine:** `test1.csv` remains strictly isolated as historical benchmark comparator.

---

## 3. Reproduction Instructions

### A. Anatomical De-duplication of Manifest
```bash
source venv/bin/activate
python3 -c '
from src.data.deduplication import audit_and_deduplicate_manifest
res = audit_and_deduplicate_manifest(
    manifest_csv="data/metadata/Annotation/CSV/train1.csv",
    data_dir="data/raw/TrainValidation",
    output_json_path="data/metadata/splits/stage2_dev_deduplicated_manifest.json",
    output_csv_path="data/metadata/splits/stage2_dev_deduplicated_manifest.csv"
)
print("Deduplicated slices:", res["metadata"]["total_unique_slices"])
'
```
* **Output Manifests:**
  * `data/metadata/splits/stage2_dev_deduplicated_manifest.json`
  * `data/metadata/splits/stage2_dev_deduplicated_manifest.csv`
* **Input Manifests:** `train1.csv` remains completely unmodified.

### B. Feature Cache Extraction & Generation
```bash
source venv/bin/activate
PYTHONPATH=. python3 src/features/run_extraction.py
```
* **Backbone:** torchvision `efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)`
* **Device:** Apple Silicon `mps` (batch size 32)
* **Output Cache:** `data/processed/dev_patient_features_effnet_b0.pt` (15.67 MB, 3,132 slices across 239 series in 122 patients)
* **Archived Historical Cache:** `data/processed/dev_patient_features_effnet_b0_raw_3736_with_duplicates.pt` (19.4 MB)

### C. 13-Point Cache Integrity Audit
```bash
source venv/bin/activate
PYTHONPATH=. python3 src/features/audit_feature_cache.py
```
* **Audit Result:** 13/13 Invariants passed, 0 errors, 0 warnings.

### D. Full Unit & Regression Test Suite
```bash
source venv/bin/activate
pytest tests/ -v
```
* **Current Result:** 55 passed in 8.74s (32 Stage 1 + 13 Stage 2 infrastructure + 10 Stage 2 de-duplication tests).
