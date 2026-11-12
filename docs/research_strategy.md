# PulmoVision — Two-Stage Research Strategy

## Stage 1: Complete CNN-Based Prototype (Active)

We strictly follow the sequential 13-module roadmap to build, validate, explain, and deploy the core 2.5D CNN system:
- **Baseline Custom CNN:** [`src/models/custom_cnn.py`](file:///Users/geetikavasistha/PulmoVision/src/models/custom_cnn.py) (Module 4)
- **2.5D Multi-Slice Stacking Pipeline:** [`src/data/slice_stacking.py`](file:///Users/geetikavasistha/PulmoVision/src/data/slice_stacking.py) & [`src/data/dataloader.py`](file:///Users/geetikavasistha/PulmoVision/src/data/dataloader.py) (Module 5)
- **Transfer Learning Backbones:** ResNet50 & EfficientNet-B0 (Module 6)
- **Controlled Training & Tracking:** Class-weighted cross entropy, MLflow checkpoints (Module 7)
- **Evaluation:** Patient-level Macro F1, Confusion Matrix, Latency (Module 8)
- **Explainability:** Grad-CAM on target convolutional feature maps (Module 9)
- **Application & Serving:** Streamlit cached UI & deployment (Modules 10–12)

*Scope boundary:* No sequential (LSTM) or attention/transformer architectures are introduced during Stage 1.

---

## Stage 2: Independent Architecture Research (Post-Stabilization)

After Stage 1 is fully completed, tested, and deployable on `main`, dedicated independent research branches will explore advanced volumetric & sequence models:

```
main (Stage 1 Stable Baseline)
 ├── research/cnn-lstm       [CNN + BiLSTM for inter-slice sequence modelling]
 ├── research/3d-cnn         [Full 3D volumetric convolutions]
 └── research/transformers   [CNN + Vision Transformer / Swin Transformer]
```

### Invariants Maintained Across All Experiments:
1. **Identical Patient-Level Splits:** Strict reuse of patient IDs (Fold 1: 122 train, 255 test, zero overlap).
2. **Consistent Preprocessing:** Uniform 16-bit windowing $[200, 1800]$ and dynamic range scaling.
3. **Controlled Evaluation:** Identical metrics (Macro F1, Balanced Accuracy, latency) evaluated on the held-out test set only once.
4. **Reproducible Experiment Logs:** MLflow run records and standalone evaluation reports.

