# PulmoVision — Technical Interview Preparation Guide

This guide prepares you to discuss the architecture, engineering decisions, and trade-offs of PulmoVision in Senior Machine Learning Engineer and Medical AI interviews.

---

### Q1: Why did you choose a 2.5D architecture over 2D or full 3D CNNs?
> **Answer:** Medical CT scans represent continuous 3D volumes where pulmonary lesions traverse multiple cross-sections. While a 2D CNN discards through-plane continuity, a full 3D CNN requires volumetric convolutions that demand cubic memory and compute ($O(D \cdot H \cdot W)$), limiting batch sizes and eliminating access to high-quality ImageNet pretrained weights.  
> 2.5D stacks adjacent slices $[Z-1, Z, Z+1]$ as input channels into a 2D backbone. The first convolutional layer naturally learns cross-slice spatial gradients across adjacent slices, giving volumetric context at a fraction of 3D compute while allowing direct transfer learning from ImageNet models.

---

### Q2: What is data leakage in medical imaging, and how did you prevent it?
> **Answer:** In medical imaging, CT scans contain dozens to hundreds of slices per patient. If you perform naive random slice-level splitting, slices from the same patient appear in both training and test sets. The model simply memorizes patient-specific morphological features (e.g. ribcage dimensions, scanner artifacts) and achieves $>98\%$ test accuracy that collapses on external validation.  
> In PulmoVision, we implemented **strict patient-level splitting**: all 377 patients were partitioned first (122 train, 255 test). Triplets are generated *only* within patient boundaries, guaranteeing **0 patient overlap** between splits.

---

### Q3: How did you handle 16-bit CT radiodensities? Why not just divide by 255?
> **Answer:** Standard computer vision libraries assume 8-bit images ($[0, 255]$). CT scans record tissue radiodensity in 16-bit integers ($[0, \approx 2200]$). Dividing by 255 truncates or distorts values. Furthermore, normal monitors cannot display $>2000$ values, making raw slices look pitch black.  
> We applied **domain-specific anatomical windowing**: clipping intensities to the lung parenchyma window ($[200, 1800]$) to eliminate scanner bed and air artifacts while maximizing contrast for ground-glass opacities, followed by min-max scaling to $[0.0, 1.0]$.

---

### Q4: Why are vertical flips unsafe augmentations for chest CT scans?
> **Answer:** In computer vision, vertical flipping is standard for object detection. However, in thoracic radiology, human anatomy has fixed vertical polarity (cephalocaudal orientation): the lung apices are superior, and the diaphragm is inferior. Flipping vertically inverts gravity-dependent lung pathologies and creates anatomically impossible representations.  
> We restricted augmentations to bilateral horizontal flips ($p=0.5$), subtle rotations ($\pm 7^\circ$), and minor scaling ($\pm 5\%$).

---

### Q5: How did you address the 1:3 class imbalance in COVID-CTset?
> **Answer:** Normal scans outnumber COVID scans 3 to 1 ($282$ vs $95$ patients). A naive classifier predicting the majority class achieves $74.8\%$ accuracy with zero clinical utility.  
> We implemented two mitigations:
> 1. **Class-Weighted Loss:** Scaled `CrossEntropyLoss` with inverse class frequencies ($[0.67, 2.0]$).
> 2. **Evaluation Metrics:** Monitored **Macro F1** and **Balanced Accuracy** rather than raw unweighted accuracy.

---

### Q6: How does Grad-CAM work under the hood?
> **Answer:** Grad-CAM calculates the gradient of the target class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer (`features[-1]` in EfficientNet). We spatially global-average-pool these gradients to compute importance weights $\alpha_k^c$:
> $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
> We then compute a weighted combination of feature maps followed by a ReLU activation to retain only features that positively correlate with the predicted class. The resulting 2D heatmap is resized and superimposed onto the original scan.

---

### Q7: If Grad-CAM highlights a region, does it prove clinical pathology?
> **Answer:** Absolutely not. Grad-CAM identifies **statistical feature correlation**, not causal clinical pathology. Models can latch onto non-clinical confounders (e.g., patient posture, chest tubes, or scanner calibration differences). In PulmoVision, Grad-CAM is strictly an interpretability inspection tool for researchers, never a diagnostic proof.

---

### Q8: How did you optimize inference latency for the Streamlit dashboard?
> **Answer:** We implemented three optimizations:
> 1. **Resource Caching:** `@st.cache_resource` loads model weights into memory once per process, eliminating repetitive file I/O.
> 2. **Efficient Backbone:** Selected EfficientNet-B0 ($4.0\text{M}$ parameters) over ResNet50 ($23.5\text{M}$), reducing warm inference latency to **$140\text{ ms}$** on Apple MPS and **$< 900\text{ ms}$** on standard CPU.
> 3. **Stateless Serving:** Kept the Streamlit process completely decoupled from the training and audit scripts.

---

### Q9: How did you handle the boundary slices in a 2.5D stack?
> **Answer:** For slice $Z=0$, slice $Z-1$ does not exist. For slice $Z=N-1$, slice $Z+1$ does not exist. Instead of dropping boundary slices (which loses data at the lung apex and base), we implemented **edge replication clamping**:
> - Slice $0 \to [0, 0, 1]$
> - Slice $N-1 \to [N-2, N-1, N-1]$  
> This ensures that every slice in the scan can be evaluated.

---

### Q10: What is your Stage 2 research plan?
> **Answer:** Stage 1 established a reproducible, fully-tested CNN baseline. In Stage 2, we explore:
> 1. **CNN + BiLSTM:** Passing slice embeddings through bidirectional LSTMs to capture long-range inter-slice sequence dependencies.
> 2. **Full 3D CNNs:** Benchmarking volumetric 3D kernels against our 2.5D baseline.
> 3. **Vision Transformers (ViT / Swin):** Evaluating self-attention across CT patches to capture global contextual relationships.  
> All experiments will strictly reuse the identical patient-level split to ensure fair, scientific comparisons.
