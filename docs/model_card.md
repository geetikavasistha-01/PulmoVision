# PulmoVision — Model Card: 2.5D EfficientNet-B0

Following the Model Cards for Model Reporting framework (Mitchell et al., 2019).

---

## 1. Model Details

- **Model Name:** PulmoVision 2.5D EfficientNet-B0
- **Version:** `1.0.0-stage1`
- **Model Type:** 2.5D Convolutional Neural Network with Compound Scaling
- **Input Dimensions:** `[B, 3, 224, 224]` (3 adjacent axial CT slices: $[Z-1, Z, Z+1]$)
- **Output:** Binary classification probabilities (`[Normal, COVID-19]`) + Grad-CAM heatmap
- **Pretrained Weights:** ImageNet-1K (`IMAGENET1K_V1`)
- **Framework:** PyTorch 2.15.0 / Torchvision
- **Developer:** PulmoVision Research Team

---

## 2. Intended Use

| Intended Use Cases | Out-of-Scope / Prohibited Uses |
|---|---|
| Educational demonstration of 2.5D medical computer vision | Primary clinical diagnosis of COVID-19 or any respiratory condition |
| Benchmarking patient-level vs slice-level data leakage | Replacing radiologist assessments in healthcare workflows |
| Exploring explainable AI attention maps (Grad-CAM) on CT scans | Making clinical treatment or isolation recommendations |
| Comparative research between 2D, 2.5D, and 3D backbones | Commercial or clinical diagnostic billing |

---

## 3. Training & Validation Data

- **Primary Dataset:** COVID-CTset (Rahimzadeh et al., 2021)
- **Source Center:** Negin Medical Center, Sari, Iran (Siemens SOMATOM Scope HRCT)
- **Cohort Size:** 377 unique patients (95 Positive, 282 Negative)
- **Data Partitioning Protocol (Zero Leakage):**
  - **Train Partition (`train_split.csv`):** 98 patients (62 Pos, 36 Neg) | 2,959 slices
  - **Internal Validation (`val_split.csv`):** 24 patients (15 Pos, 9 Neg) | 777 slices
  - **Held-Out Test (`test1.csv`):** 255 patients | 8,322 slices (**Untouched**)
  - **Patient Overlaps:** Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0.
- **Important Checkpoint Note:** The baseline weights in `models/best_model.pth` were generated via **synthetic verification batches** to validate Apple Metal MPS acceleration, loss convergence, and Grad-CAM hooks. Full-scale model training across all 377 patients will follow in Phase 3.
- **Preprocessing:** 16-bit TIFF radiodensity windowing $[200, 1800]$, dynamic range scaling to $[0.0, 1.0]$, bilinear resize to $224 \times 224$.

---

## 4. Evaluation Performance

Evaluated on held-out patient-level test split:

| Metric | Score |
|---|---|
| **Macro F1 Score** | $0.5211$ |
| **Sensitivity (Recall)** | $65.6\%$ |
| **Specificity** | $53.1\%$ |
| **Accuracy** | $56.2\%$ |
| **Inference Latency (MPS)** | $140.1\text{ ms}$ ($7.1\text{ scans/sec}$) |
| **Inference Latency (CPU)** | $899.9\text{ ms}$ ($1.1\text{ scans/sec}$) |

---

## 5. Ethical Considerations & Limitations

1. **Single-Institution Bias:** Data originated from a single center in Iran; models trained on this cohort may degrade when presented with scans from different scanner manufacturers (GE, Philips, Toshiba) or differing slice thickness protocols.
2. **Imbalanced Classes:** The $\approx 1:3$ ratio necessitates class-weighted cross-entropy loss.
3. **Interpretability Caution:** Grad-CAM displays areas of high activation, but these correlate with statistical patterns, not verified clinical pathology.

