# PulmoVision — Stage 2 Research Progress & Implementation Audit

**Branch:** `feature/cnn-bilstm`  
**Date:** November 11, 2026  
**Status:** Phase 1 Audit & Pre-Implementation Verification  

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
| **Development Manifest** | `train1.csv` — exactly 3,736 slice records across 122 patients | ✅ Verified |
| **Test Manifest** | `test1.csv` — exactly 8,322 slice records across 255 patients (quarantined) | ✅ Verified |
| **Stage 1 Checkpoints** | Exp 1 (1.1 MB), Exp 2 (282.5 MB), Exp 3 (282.6 MB), Exp 4 (48.6 MB) | ✅ Verified (Unchanged) |

---

## 2. Invariants & Execution Gates

1. **Feature Extraction:** External ImageNet-1K EfficientNet-B0 only. Zero in-domain exposure.
2. **Preprocessing:** 16-bit radiodensity clipping $[200, 1800]$, scaling to $[0, 1]$, channel replication $R=G=B$, ImageNet standardization, bilinear resize to $224 \times 224$.
3. **Cross-Validation:** Nested 5-fold CV with inner-validation for model selection and threshold tuning; outer validation evaluated once for unbiased OOF predictions.
4. **Test Quarantine:** `test1.csv` remains strictly isolated as historical benchmark comparator.

