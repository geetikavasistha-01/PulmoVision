# PulmoVision — Master Progress Tracker

## Current Status
🎉 **STAGE 3A CONTAINERIZATION & UI NAVIGATION RESILIENCE DELIVERED**  
- Official Stage 2 5-fold outer cross-validation with inner validation holdouts conducted across 3 architectures (15 total runs).
- Architecture B confirmed leader: **0.9461 Macro F1**, **0.9688 ROC-AUC**, **98.7% Sensitivity** (76/77 COVID-19 patients detected).
- Complete product-level redesign delivered: Modular hybrid Scientific AI Research Laboratory + Clinical Imaging Workstation.
- Dual visual themes implemented: **Clinical Dark** and **Laboratory Light**, with high-contrast pitch-black radiological CT viewport invariant.
- All 7 destinations fully implemented and verified: Research Overview, Dataset Explorer, Model Laboratory, Experiment Analytics, CT Workstation, Explainability & Boundaries, and Research Documentation.
- Automated testing: **104 / 104 tests passing (100% green)** across data, deduplication, architectures, patient-balanced weighting, sequence validation, manifest adapter, Docker configuration, and Streamlit AppTest navigation.
- UI Refinement & Acceptance: Resolved sidebar destination radio lock and migrated clipboard bridge from legacy `components.html` to direct `st.html(unsafe_allow_javascript=True)`, eliminating browser console iframe sandbox and deprecated permissions policy warnings.
- Stage 3A Docker Packaging: Created reproducible multi-stage Docker build, pre-baking EfficientNet-B0 weights, dropping to non-root `pulmo:1000`, setting up health checks, and validating CPU inference fallback.
- Comprehensive Deployment Guide published: `docs/docker_deployment.md`.
- Provenance & Quarantine: Sealed held-out test partition (`test1.csv`) preserved unopened with zero read exposure.

## Overall Progress
- **Module 0 — Project Foundations:** ✅ Completed
- **Module 1 — Development Environment:** ✅ Completed
- **Module 2 — Dataset Discovery & Audit:** ✅ Completed
- **Module 3 — Medical Image Processing:** ✅ Completed
- **Module 4 — Understanding CNNs:** ✅ Completed
- **Module 5 — 2.5D Modelling & Stacking:** ✅ Completed
- **Module 6 — Transfer Learning:** ✅ Completed
- **Module 7 — Model Training:** ✅ Completed
- **Module 8 — Model Evaluation:** ✅ Completed
- **Module 9 — Explainable AI (Grad-CAM):** ✅ Completed
- **Module 10 — Streamlit Application:** ✅ Completed
- **Module 11 — Deployment & Optimisation:** ✅ Completed
- **Module 12 — Testing & Portfolio Packaging:** ✅ Completed
- **Stage 2 Research — CNN-BiLSTM Sequence Modelling:** ✅ Completed
- **Stage 3A — Docker Containerization & Production Packaging:** ✅ Completed

---

## Detailed Milestone Checklist

### Stage 2 Research — CNN-BiLSTM Sequence Modelling & Deployment
- [x] **Deduplication Audit:** Forensic SHA256 audit resolved 604 redundant slice files; preserved exact canonical records (3,132 unique development slices, 122 patients).
- [x] **Patient-Balanced Optimization:** Implemented equal patient weighting per optimizer update ($1/N_p$ per series, grouped per patient) in `src/evaluation/nested_cv.py`.
- [x] **5-Fold Cross-Validation Benchmark:** Completed 15 runs across Architectures A (Mean Pooling), B (BiLSTM + Mean Pooling), and C (BiLSTM + Additive Attention).
- [x] **Cross-Validation Leader:** Architecture B confirmed development leader (Macro F1: $0.9461$, ROC-AUC: $0.9688$, Sensitivity: $98.70\%$, Specificity: $88.89\%$).
- [x] **Deployment Model Artifact:** Trained explicitly labeled standalone deployment model `models/stage2_bilstm_deployment.pth` on development cohort.
- [x] **Inference & Sequence Validation Engine:** Implemented `src/inference/sequence_validator.py` and `src/inference/sequence_predictor.py`.
- [x] **Streamlit Research Platform:** Upgraded `app/streamlit_app.py` to 7 rich interactive pages with craniocaudal slider, sequence inference, explainability boundaries, and embedded research report.
- [x] **Comprehensive Research Report:** Published 14-section formal manuscript in `reports/research/PULMOVISION_STAGE_2_RESEARCH_REPORT.md`.
- [x] **Full Regression Test Suite:** **81/81 tests passing** across data, deduplication, architectures, patient-balanced weighting, real-TIFF inference, sequence validation, and UI modules.
- [x] **Final Consistency Audit:** Completed and published forensic consistency audit (`reports/experiments/FINAL_IMPLEMENTATION_AUDIT.md`) with official **PASS** verdict.
- [x] **Product Redesign & Modular Decomposition:** Re-architected Streamlit application into modular components, views, styles, and utilities.
- [x] **Dual Theme & Clinical Canvas:** Deployed Clinical Dark and Laboratory Light themes with pitch-black (`#000000`) radiological CT inspection viewport.
- [x] **Visual Evidence Capture:** Automated CDP capture of 8 full-resolution screenshots in `reports/figures/`.
- [x] **Final Application Implementation Report:** Published in `reports/experiments/FINAL_APPLICATION_IMPLEMENTATION_REPORT.md`.

### Module 0 — Project Foundations
- [x] Understand project objectives & 2.5D paradigm
- [x] Clarify research prototype vs clinical diagnostic boundary
- [x] Select primary dataset (COVID-CTset) & external candidate (InRad)
- [x] Document 10 known limitations in `docs/limitations.md`
- [x] Create project roadmap & README

### Module 1 — Development Environment
- [x] Create virtual environment & configure Git
- [x] Verify PyTorch v2.15.0 & Apple Silicon Metal (MPS) GPU acceleration
- [x] Verify image processing libraries (Pillow, OpenCV, Albumentations)
- [x] Set up `.gitignore` & `.env.example`

### Module 2 — Dataset Discovery & Audit
- [x] Query Kaggle API programmatically for metadata
- [x] Verify 377 patient identifiers and class imbalance (95 COVID vs 282 Normal)
- [x] Verify zero patient leakage between official Train and Test folds
- [x] Audit 16-bit TIFF image format ($512 \times 512$, values up to $2204$)
- [x] Publish audit report: `reports/dataset_audit/COVID_CTSET_AUDIT.md`

### Module 3 — Medical Image Processing
- [x] Implement 16-bit CT radiodensity windowing ($[200, 1800]$) in `src/data/preprocessing.py`
- [x] Implement safe CT augmentations (horizontal flip, subtle rotation; no vertical flips)
- [x] Implement dynamic range scaling to $[0.0, 1.0]$ and bilinear resize to $224 \times 224$
- [x] Automated unit tests: 3/3 passing

### Module 4 — Understanding CNNs
- [x] Implement ConvBlock & baseline architecture in `src/models/custom_cnn.py` ($93\text{K}$ params)
- [x] Support 2D (1-channel) and 2.5D (3-channel) inputs
- [x] Execute mini-training on Apple MPS device in `src/training/train_baseline_mini.py`
- [x] Generate and inspect training vs validation loss curves (`reports/figures/custom_cnn_baseline_loss.png`)
- [x] Demonstrate and document overfitting dynamics

### Module 5 — 2.5D Modelling & Stacking
- [x] Implement provenance-aware filename parser & grouping in `src/data/slice_stacking.py`
- [x] Implement edge-replication boundary clamping for slice 0 and slice $N-1$
- [x] Enforce 3-condition invariant: same patient, same scan series, consecutive slice indices
- [x] Implement `CT25DDataset` and `create_dataloader` in `src/data/dataloader.py`
- [x] Automated unit tests: 3/3 passing

### Module 6 — Transfer Learning
- [x] Adapt ResNet50 in `src/models/resnet.py` ($23.5\text{M}$ params)
- [x] Adapt EfficientNet-B0 in `src/models/efficientnet.py` ($4.0\text{M}$ params)
- [x] Adapt DenseNet121 in `src/models/densenet.py` ($6.9\text{M}$ params)
- [x] Build unified factory in `src/models/model_factory.py`
- [x] Automated unit tests across all backbones: 7/7 passing

### Module 7 — Model Training
- [x] Create centralized config: `configs/training.yaml`
- [x] Implement training loop with class-weighted CrossEntropyLoss ($[0.67, 2.0]$) in `src/training/train.py`
- [x] Implement validation loop in `src/training/validate.py`
- [x] Implement checkpoint saving & early stopping on Macro F1 in `src/training/checkpoint.py`
- [x] Train 2.5D EfficientNet-B0 and save checkpoint: `models/best_model.pth` ($46\text{MB}$)

### Module 8 — Model Evaluation
- [x] Implement clinical diagnostic metrics in `src/evaluation/metrics.py` (Sensitivity, Specificity, Macro F1)
- [x] Implement confusion matrix generator in `src/evaluation/confusion_matrix.py`
- [x] Generate and publish evaluation report: `reports/experiments/EVALUATION_REPORT.md`
- [x] Measure per-sample latency

### Module 9 — Explainable AI (Grad-CAM)
- [x] Implement hook-based Grad-CAM in `src/explainability/gradcam.py`
- [x] Extract pooled gradients from target convolutional layer (`features[-1]`)
- [x] Generate normalized heatmaps and Jet colormap overlays on physical CT slices
- [x] Automated unit tests for heatmaps and overlays: 2/2 passing

### Module 10 — Streamlit Application
- [x] Build interactive 5-page dashboard in `app/streamlit_app.py`
- [x] Implement `@st.cache_resource` model loading in `src/inference/predictor.py`
- [x] Build 1-click sample test & 3-slice custom file upload
- [x] Build Model Information, Methodology, and About pages

### Module 11 — Deployment & Optimisation
- [x] Configure `.streamlit/config.toml` (headless, no telemetry, upload limits)
- [x] Build lightweight deployment requirements: `requirements-deploy.txt`
- [x] Benchmark inference latency on CPU ($899\text{ ms}$) and Apple MPS ($140\text{ ms}$)
- [x] Publish deployment guide: `docs/deployment_guide.md`

### Module 12 — Testing, Documentation & Packaging
- [x] Automated test suite: **32/32 unit tests passing**
- [x] Formal Model Card: `docs/model_card.md`
- [x] System Architecture Document: `docs/architecture.md`
- [x] Medical AI Methodology Document: `docs/methodology.md`
- [x] Technical Interview Q&A Guide: `docs/interview_guide.md`
- [x] Two-Stage Research Strategy Document: `docs/research_strategy.md`

---

## Phase 1 Audit & Pre-Training Corrections
- [x] **Patient-Level Split Stratification (`src/data/patient_split.py`):**
  - Partitioned `train1.csv` development set (122 patients) into Train (98 patients) and Val (24 patients) using fixed seed (42).
  - Preserved class proportions: Train (62 Pos, 36 Neg; 2,959 slices), Val (15 Pos, 9 Neg; 777 slices).
  - Maintained complete zero-overlap across all 3 sets ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
  - `test1.csv` remains strictly untouched (255 patients, 8,322 slices).
  - Enforced `validate_split_integrity()` safeguards in `src/training/train.py` to prevent test-set data leakage.
- [x] **Patient-Level Evaluation Suite (`src/evaluation/metrics.py`):**
  - Added `compute_patient_level_metrics()` supporting `positive_slice_ratio`, `mean_prob`, and `majority_vote`.
  - Enforced unweighted equal patient evaluation ($1.0$ weight per patient regardless of slice count).
- [x] **Streamlit Upload Validation & Provenance (`src/inference/upload_validator.py`):**
  - Validates image extensions (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`).
  - Verifies cross-slice patient and series provenance; enforces slice index ordering.
  - Distinct mode display: Full 2.5D Mode vs Isolated-Slice Fallback Mode.
- [x] **Automated Regression & Verification Tests:**
  - Added 12 new tests across patient splitting, patient metrics, and upload validation.
  - Complete test suite: **32 passed in ~12s**.

---

## Final Deliverables Index

| Asset | Path |
|---|---|
| **Web Dashboard** | [`app/streamlit_app.py`](file:///Users/geetikavasistha/PulmoVision/app/streamlit_app.py) |
| **Model Checkpoint** | `models/best_model.pth` (46 MB) |
| **Model Card** | [`docs/model_card.md`](file:///Users/geetikavasistha/PulmoVision/docs/model_card.md) |
| **Deployment Guide** | [`docs/deployment_guide.md`](file:///Users/geetikavasistha/PulmoVision/docs/deployment_guide.md) |
| **Technical Interview Guide** | [`docs/interview_guide.md`](file:///Users/geetikavasistha/PulmoVision/docs/interview_guide.md) |
| **Evaluation Report** | [`reports/experiments/EVALUATION_REPORT.md`](file:///Users/geetikavasistha/PulmoVision/reports/experiments/EVALUATION_REPORT.md) |
| **Audit Report** | [`reports/dataset_audit/COVID_CTSET_AUDIT.md`](file:///Users/geetikavasistha/PulmoVision/reports/dataset_audit/COVID_CTSET_AUDIT.md) |
| **Automated Tests** | [`tests/`](file:///Users/geetikavasistha/PulmoVision/tests/) (20/20 passing) |

---

## Phase 3 Baseline Experimental Benchmark Results

- [x] **Phase 3A — Dataset Ingestion & Verification:**
  - Complete COVID-CTset Fold 1 archive downloaded and extracted (12,058 16-bit TIFF scans).
  - 100.0% path resolution across Train ($2,959$), Val ($777$), and Held-Out Test ($8,322$).
  - Post-download audit published: [`reports/dataset_audit/COVID_CTSET_POST_DOWNLOAD_AUDIT.md`](file:///Users/geetikavasistha/PulmoVision/reports/dataset_audit/COVID_CTSET_POST_DOWNLOAD_AUDIT.md).
- [x] **Phase 3B — Real-Data Smoke Test:**
  - Executed on Custom CNN 2D: verified finite gradients, loss reduction, checkpoint restoration, and zero test-set access.
- [x] **Phase 3C — Sequential Baseline Model Training:**
  - **Custom CNN 2D:** $93\text{K}$ params, Best Val F1: $0.8124$, Val Acc: $81.3\%$, Frozen $\tau = 0.50$.
  - **ResNet50 2D:** $23.5\text{M}$ params, Best Val F1: $0.9562$, Val Acc: $95.6\%$, Frozen $\tau = 0.45$.
  - **ResNet50 2.5D:** $23.5\text{M}$ params, Best Val F1: $0.9793$, Val Acc: $97.9\%$, Frozen $\tau = 0.20$.
  - **EfficientNet-B0 2.5D:** $4.0\text{M}$ params, Best Val F1: $0.9728$, Val Acc: $97.3\%$, Frozen $\tau = 0.20$.
- [x] **Phase 3D — Final Held-Out Test Evaluation (`test1.csv`: 255 patients, 8,322 slices):**
  - **ResNet50 2.5D:** Slice Acc $96.7\%$, Macro F1 $0.8727$, Spec $96.9\%$. Patient Acc **$94.1\%$**, Macro F1 **$0.8307$**, Spec **$94.1\%$** ($17/18$ TP, $223/237$ TN).
  - **EfficientNet-B0 2.5D:** Slice Acc $96.9\%$, Macro F1 $0.8697$, Spec $97.4\%$. Patient Acc **$94.1\%$**, Macro F1 **$0.8307$**, Spec **$94.1\%$** ($17/18$ TP, $223/237$ TN) with **$82.9\%$ fewer parameters**.
  - **Failure of 2D Custom CNN:** Test specificity collapsed to $32.1\%$ ($161$ false positive normal patients).
  - **Single False Negative Analysis:** Only `patient11` missed across deep models due to single-slice representation (`slice_count: 1`). For all multi-slice patients, sensitivity was **$100.0\%$ (17/17)**.
  - Published comprehensive report: [`reports/experiments/EVALUATION_REPORT.md`](file:///Users/geetikavasistha/PulmoVision/reports/experiments/EVALUATION_REPORT.md).

---

## Stage 2 Sequence Intelligence & Platform Deployment (v2.0.0)

- [x] **Stage 2A — Forensic Anatomical De-Duplication Protocol:**
  - Audited 3,736 slices across 122 development patients; identified and resolved 604 duplicate slice alias entries.
  - Published verified deduplicated manifest (`data/metadata/splits/stage2_dev_deduplicated_manifest.csv`) with 3,132 unique physical slices across 239 CT acquisition series.
  - Maintained strict quarantine on the 255-patient holdout set (`test1.csv`), preserving zero test data contamination.
- [x] **Stage 2B — High-Throughput Image Feature Extraction:**
  - Extracted 1,280-dimensional feature representations using frozen ImageNet-pretrained EfficientNet-B0 backbone.
  - Persisted embeddings to cached feature stores with deterministic SHA-256 verification hashes.
- [x] **Stage 2C — Controlled Architecture Experiments (5-Fold Patient-Stratified CV):**
  - **Exp A — CNN-BiLSTM (Last Step):** Macro F1 $90.96\%$, ROC-AUC $94.63\%$, Sensitivity $94.81\%$, Specificity $84.44\%$.
  - **Exp B — CNN-BiLSTM (Mean Pooling) [Selected Production Model]:**
    - Macro F1 **$94.61\%$**, ROC-AUC **$96.88\%$**, Sensitivity **$98.70\%$** ($76/77$ COVID-19 detected, $1$ FN), Specificity **$88.89\%$** ($40/45$ Normal detected, $5$ FP).
    - Trainable parameters: $1,460,418$ ($2 \times$ BiLSTM layers, hidden size $128$ per direction, mean pooling head).
    - Verified checkpoint SHA-256: `0bd817319d99e0fbb22a318c458d9f86df77297f7bc4f5ec88f845e80ff1caed`.
  - **Exp C — CNN-BiLSTM (Bahdanau Additive Attention):** Macro F1 $92.74\%$, ROC-AUC $95.99\%$, Sensitivity $97.40\%$, Specificity $84.44\%$.
  - Published full research report: [`reports/research/PULMOVISION_STAGE_2_RESEARCH_REPORT.md`](file:///Users/geetikavasistha/PulmoVision/reports/research/PULMOVISION_STAGE_2_RESEARCH_REPORT.md).
- [x] **Stage 2D — 7-Page Modular Research Workstation:**
  - Built 7 unified modules in [`app/views/`](file:///Users/geetikavasistha/PulmoVision/app/views/):
    1. **Research Overview:** Non-technical plain-language landing page with 3-step visual workflow, live metrics, and ethical disclaimers.
    2. **Dataset Explorer:** Sequence length distributions, deduplication audit tables, and slice inspector.
    3. **Model Laboratory:** Architecture schematics, layer dimensions, parameter budgets, and attention formulas.
    4. **Experiment Analytics:** Interactive ROC curves, confusion matrices, and fold-level cross-validation performance.
    5. **CT Workstation:** Live 2.5D research inference pipeline with window leveling and confidence gauge.
    6. **Explainability:** Grad-CAM activation heatmaps with anatomical localization caveats.
    7. **Research Documentation:** In-app scientific manuscripts, protocol audits, and forensic verification logs.
  - Dual design themes: Clinical Dark (`#0a0e17`) and Laboratory Light (`#f8fafc`) with preserved pitch-black CT viewport (`#000000`).
  - Zero unicode emojis enforced across the platform (Material Symbols icons used exclusively).
- [x] **Stage 2E — Automated Test Suite & Release Verification:**
  - Complete automated test suite: **97 / 97 passing tests (100% green)** across units, integration, UI modules, and quarantine checks.
  - Zero trailing whitespace (`git diff --check` clean).
  - Released and tagged: **`v2.0.0`** on branch `main`.

---

## Deliverables & Documentation Index

| Asset | Path | Description |
|---|---|---|
| **Web Application** | [`app/streamlit_app.py`](file:///Users/geetikavasistha/PulmoVision/app/streamlit_app.py) | 7-page research workstation with dual themes |
| **Stage 2 Checkpoint** | `models/best_model.pth` | EfficientNet-B0 + BiLSTM Mean ($1.46\text{M}$ params, SHA-256 verified) |
| **Stage 2 Research Report** | [`reports/research/PULMOVISION_STAGE_2_RESEARCH_REPORT.md`](file:///Users/geetikavasistha/PulmoVision/reports/research/PULMOVISION_STAGE_2_RESEARCH_REPORT.md) | Comprehensive primary scientific manuscript |
| **Deduplicated Manifest** | [`data/metadata/splits/stage2_dev_deduplicated_manifest.csv`](file:///Users/geetikavasistha/PulmoVision/data/metadata/splits/stage2_dev_deduplicated_manifest.csv) | 3,132 unique development CT slices |
| **Stage 1 Baselines Report** | [`reports/experiments/EVALUATION_REPORT.md`](file:///Users/geetikavasistha/PulmoVision/reports/experiments/EVALUATION_REPORT.md) | Baseline models evaluation on test cohort |
| **Deduplication Audit** | [`reports/dataset_audit/STAGE2_DEDUPLICATION_AUDIT.md`](file:///Users/geetikavasistha/PulmoVision/reports/dataset_audit/STAGE2_DEDUPLICATION_AUDIT.md) | Forensic analysis of 604 resolved duplicate aliases |
| **Model Card** | [`docs/model_card.md`](file:///Users/geetikavasistha/PulmoVision/docs/model_card.md) | Ethical, data, and performance card |
| **System Architecture** | [`docs/architecture.md`](file:///Users/geetikavasistha/PulmoVision/docs/architecture.md) | System design & dataflow specifications |
| **Research Strategy** | [`docs/research_strategy.md`](file:///Users/geetikavasistha/PulmoVision/docs/research_strategy.md) | Two-stage exploration protocol |
| **Automated Tests** | [`tests/`](file:///Users/geetikavasistha/PulmoVision/tests/) | 97/97 tests passing (100% green) |
