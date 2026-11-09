# PulmoVision — Known Limitations

| Category | Limitation |
|---|---|
| **Clinical validity** | Not validated for clinical use; not FDA/CE approved |
| **Dataset bias** | Trained on a single institution's data; may not generalise |
| **Labels** | Dataset categories ≠ confirmed diagnoses |
| **2.5D scope** | Only local slice context; misses distant volumetric patterns |
| **Patient diversity** | Limited demographic representation in training data |
| **Scanner variation** | Performance may degrade on scans from different manufacturers/protocols |
| **Class imbalance** | Dataset may have unequal class distribution |
| **Grad-CAM** | Shows correlated regions, not causal explanations |
| **Deployment** | CPU-only inference may be slow for large uploads |
| **Single dataset** | No multi-source external validation unless explicitly added |

