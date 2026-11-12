# PulmoVision — Limitations and Ethical Considerations

This document provides a comprehensive analysis of the clinical, data, algorithmic, and ethical limitations of the PulmoVision research platform. It defines the formal boundaries of this academic prototype and outlines safety principles for responsible AI medical imaging research.

---

## 1. Research Prototype Scope & Regulatory Posture

PulmoVision is exclusively an educational and scientific research instrument. It has **NOT** undergone medical device certification, regulatory auditing, or clinical trial clearance:

- **Not a Medical Diagnostic Device:** The software is not approved by the U.S. Food and Drug Administration (FDA 510(k)/PMA), the European Medicines Agency (CE-IVD/MDR), or any comparable regional regulatory authority.
- **Prohibition on Diagnostic Use:** Model predictions, classification probabilities, and Grad-CAM saliency heatmaps must **never** be used as the basis for clinical diagnosis, patient triaging, admission/discharge decisions, or therapeutic prescriptions.
- **Non-Diagnostic Framing:** Dataset labels represent categorical archive annotations assigned by dataset curators, not verified pathological diagnoses or laboratory-confirmed RT-PCR reference standards.

---

## 2. Dataset Bias & Generalizability Boundaries

### Single-Institution Cohort Bias
The primary development cohort was derived from the COVID-CTset archive, collected from a single clinical institution in Iran:
- **Demographic Homogeneity:** The training cohort lacks multi-ethnic, multi-geographic demographic representation. Performance across diverse patient populations (varying body mass index, baseline lung health, pediatric, or geriatric demographics) is unverified.
- **Scanner & Acquisition Protocol Specificity:** CT scans were acquired using specific institutional scanner protocols and reconstruction filters. CT image quality, spatial resolution, and radiodensity distributions vary substantially across manufacturers (e.g., GE Healthcare, Siemens Healthineers, Philips, Canon Medical Systems). Model sensitivity and specificity may degrade when applied to external scans with different slice thicknesses, tube voltages ($kVp$), or contrast protocols.

### Unverified External Generalization
While Architecture B achieved high sensitivity ($98.70\%$) on the internal cross-validation cohort, internal cross-validation evaluates generalization within the same clinical environment. True external clinical validity requires multi-center benchmarking on independent cohorts (e.g., BIMCV-COVID19+, InRad, or national registries).

---

## 3. Radiographic Non-Specificity & Confounding Pathologies

Deep convolutional networks detect low- and mid-level visual features (textures, edges, opacities) that correlate with dataset labels. In thoracic CT imaging, these features frequently lack biological specificity:

### Ground-Glass Opacities (GGOs) and Consolidations
- Ground-glass opacities (patchy areas of increased attenuation without obscuration of underlying vascular markings) are the hallmark radiological presentation of COVID-19.
- **Differential Overlap:** GGOs are clinically non-specific. Identical radiographic patterns manifest in:
  1. Other viral pneumonias (Influenza A/B, Respiratory Syncytial Virus, Cytomegalovirus).
  2. Atypical bacterial infections (Mycoplasma pneumoniae, Legionella).
  3. Organizing pneumonia and non-infectious inflammatory lung diseases.
  4. Pulmonary edema, alveolar hemorrhage, and drug-induced pneumonitis.
- **Risk:** PulmoVision was trained exclusively on a binary task (COVID-19 vs. Normal). When presented with CT scans of non-COVID pulmonary infections or interstitial lung disease, the model will produce high-confidence false-positive COVID-19 predictions because it lacks a multi-disease differential category.

### Anatomical & Motion Artifacts
- **Patient Motion:** Respiratory motion or cardiac pulsation during helical acquisition causes blurring, stair-step artifacts, and volume averaging that can mimic subtle infiltrates.
- **Dependent Atelectasis:** Partial lung collapse in the posterior dependent lung regions of supine patients can be misidentified as inflammatory ground-glass changes.

---

## 4. Algorithmic & Methodological Constraints

### 2.5D Sequence Modeling vs. True 3D Volumetric Convolutions
- **Decoupled Spatial-Temporal Processing:** PulmoVision processes in-plane slices using a 2D CNN and aggregates slice embeddings with a 1D Bidirectional LSTM. While this drastically reduces GPU memory usage, it does not perform true isotropic 3D spatial filtering across the z-axis. Complex 3D morphological structures (such as bronchovascular branching or spatial lesion continuity) may not be fully resolved.
- **Craniocaudal Inversion Sensitivity:** Although the BiLSTM processes sequences bi-directionally, proper craniocaudal slice ordering (apex to base) is required for standardized clinical inspection.
- **Single-Slice Fallback ($T=1$):** Sequences containing only 1 slice cannot leverage temporal sequence transitions and fall back to the initial hidden state, yielding lower discriminative confidence and reduced screening sensitivity.

### Uncalibrated Prediction Probabilities
- Model outputs are computed via a standard sigmoid activation function following linear projection:
  $$\hat{y} = \sigma(W h + b)$$
- Sigmoid scores represent raw discriminative network activations, **not** calibrated Bayesian posterior probabilities:
  $$P(Y = 1 \mid X) \ne \sigma(z)$$
- A score of $0.98$ indicates strong model feature activation, not a $98\%$ mathematical certainty of disease presence. Post-hoc calibration techniques (such as Platt scaling, isotonic regression, or temperature scaling) have not been applied.

---

## 5. Explainability & Grad-CAM Limitations

### Correlation vs. Causality
Grad-CAM (Gradient-Weighted Class Activation Mapping) computes the gradients of the predicted class score with respect to the final convolutional feature maps:
- **Visual Saliency:** Saliency maps highlight regions that strongly influenced network activations; they do **not** provide causal medical explanations or biological proof.
- **Resolution Limits:** Grad-CAM heatmaps are derived from coarse feature representations ($7 \times 7$ spatial grid in Stage 8 of EfficientNet-B0) and upsampled bilinearly. Heatmaps lack fine pixel-level lesion segmentation boundaries.
- **Confirmation Bias:** Clinicians or researchers inspecting Grad-CAM overlays may succumb to confirmation bias, retroactively justifying model predictions based on plausible-looking heatmaps even when the underlying reasoning is flawed.

---

## 6. Ethical Principles for AI in Medical Imaging

### Automation Bias & The Human-in-the-Loop Imperative
- **Automation Bias:** Users may place unwarranted trust in automated quantitative scores, potentially overriding their own clinical judgment.
- **Human-in-the-Loop Invariant:** AI tools must function strictly as assistive research exploration aids. In any prospective clinical application, final diagnostic determinations must be made by qualified medical professionals integrating clinical history, laboratory assays (RT-PCR, antigen), and physical examination.

### Algorithmic Equity & Fairness
- Models trained on unbalanced datasets risk learning spurious demographic or institutional correlations (e.g., scanner noise patterns unique to specific machines).
- Responsible deployment requires continuous auditing across gender, age, socioeconomic background, and acquisition equipment to ensure equitable performance.

### Data Privacy & Patient Confidentiality
- Medical imaging data contains sensitive protected health information (PHI).
- PulmoVision enforces complete de-identification: all raw DICOM headers and identifying patient metadata were scrubbed prior to feature extraction, with slices cataloged purely by anonymous study identifiers.

