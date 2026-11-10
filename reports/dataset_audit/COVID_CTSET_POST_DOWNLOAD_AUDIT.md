# PulmoVision — Post-Download Dataset Integrity Audit (Fold 1)

**Date:** November 2026  
**Auditor:** PulmoVision ML Engineering Team  
**Dataset Reference:** `mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset` (Fold 1)  
**Audit Status:** ✅ **PASSED — 100% Verified**

---

## 1. Archive & Extraction Verification

| Parameter | Specification | Result | Status |
|---|---|---|:---:|
| **Downloaded Archive** | `covidctset-a-large-covid19-ct-scans-dataset.zip` | $4.03\text{ GB}$ ($4,331,604,705\text{ bytes}$) | ✅ Verified |
| **CRC32 Checksum Test** | Zip integrity scan (`zf.testzip()`) | $0\text{ CRC errors}$ | ✅ Verified |
| **Extracted Slices** | `data/raw/TrainValidation/` | $12,058\text{ TIFF images}$ | ✅ Verified |
| **Available Disk Space** | Post-extraction disk check | $15\text{ GB available}$ | ✅ Sufficient |

---

## 2. Manifest Path Resolution

| Manifest | Manifest Slices | Matched on Disk | Missing Files | Resolution Rate |
|---|:---:|:---:|:---:|:---:|
| **`train_split.csv`** (Train) | 2,959 | 2,959 | 0 | **100.0%** |
| **`val_split.csv`** (Validation) | 777 | 777 | 0 | **100.0%** |
| **`test1.csv`** (Held-Out Test) | 8,322 | 8,322 | 0 | **100.0%** |
| **Total** | **12,058** | **12,058** | **0** | **100.0%** |

---

## 3. Patient & Series Boundary Isolation

Automated set-intersection audits confirm strict patient-level and series-level isolation:

$$\text{Train Patients} \cap \text{Val Patients} = \emptyset \quad (0 \text{ patients})$$
$$\text{Train Patients} \cap \text{Test Patients} = \emptyset \quad (0 \text{ patients})$$
$$\text{Val Patients} \cap \text{Test Patients} = \emptyset \quad (0 \text{ patients})$$

$$\text{Train Series} \cap \text{Val Series} = \emptyset$$
$$\text{Train Series} \cap \text{Test Series} = \emptyset$$
$$\text{Val Series} \cap \text{Test Series} = \emptyset$$

---

## 4. Physical Image Integrity

- **Sample Inspection:** 100 random physical CT slices sampled across the dataset.
- **Color Depth:** All confirmed 16-bit unsigned integers (`uint16`), single-channel grayscale (mode `I;16`).
- **Spatial Resolution:** Exactly $512 \times 512$ pixels.
- **Corrupt Files:** $0$ unreadable or truncated files found.

---

## 5. Audit Conclusion
The extracted dataset matches the approved manifests with zero discrepancies, zero path resolution failures, and zero cross-partition leakage. The dataset is fully approved for Phase 3B real-data smoke testing and Phase 3C baseline training.

