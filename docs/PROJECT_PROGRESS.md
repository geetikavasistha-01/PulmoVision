# PulmoVision Progress Tracker

## Current Module
Module 6 — Transfer Learning

## Overall Progress
In Progress — Module 6 (Branch: feature/model-development)

## Module Checklist

### Module 0 — Project Foundations
- [x] Understand the project objective
- [x] Understand 2.5D modelling
- [x] Understand the system architecture
- [x] Understand the difference between a research prototype and a diagnostic system
- [x] Review dataset options
- [x] Read dataset documentation
- [x] Understand project limitations
- [x] Create the initial README
- [x] Create the project roadmap
- [x] Complete knowledge assessment

### Module 1 — Development Environment
- [x] Create the project repository
- [x] Create a Python virtual environment
- [x] Install required packages
- [x] Verify PyTorch installation (v2.15.0)
- [x] Check GPU availability (Apple Metal MPS: True)
- [x] Verify image libraries
- [x] Configure Git
- [x] Create initial folders
- [x] Run a small test script
- [x] Document environment versions

### Module 2 — Dataset Discovery and Audit
- [x] Explore available dataset APIs
- [x] Configure programmatic access
- [x] Discover the selected dataset
- [x] Inspect metadata
- [x] Inspect image formats
- [x] Inspect sample images
- [x] Understand all class labels
- [x] Identify patient identifiers
- [x] Identify duplicate records
- [x] Check class distribution
- [x] Generate an audit report
- [x] Approve the dataset for processing

### Module 3 — Medical Image Processing
- [x] Load sample images
- [x] Visualise CT slices
- [x] Inspect image dimensions
- [x] Understand pixel intensity
- [x] Learn windowing
- [x] Implement preprocessing
- [x] Compare different normalisation approaches
- [x] Implement safe augmentation
- [x] Visualise before-and-after images
- [x] Validate preprocessing outputs (automated pytest: 3/3 passed)

### Module 4 — Understanding CNNs
- [x] Understand tensor shapes [B, C, H, W]
- [x] Implement a convolution operation & ConvBlock
- [x] Visualise feature maps & pooling transitions
- [x] Understand pooling & global average pooling
- [x] Build a small custom CNN (PulmoCustomCNN: 93k params)
- [x] Support 2D (1-channel) and 2.5D (3-channel) inputs
- [x] Automated unit tests for model forward passes (6/6 tests passing)
- [x] Train on a small dataset subset (Apple MPS device)
- [x] Inspect training loss (0.7515 -> 0.3087) and validation loss
- [x] Experiment with learning rates (lr=0.001)
- [x] Identify overfitting (reports/figures/custom_cnn_baseline_loss.png)

### Module 5 — Understanding 2.5D Modelling
- [x] Understand 2D, 2.5D and 3D
- [x] Select a neighbouring-slice strategy (Z-1, Z, Z+1)
- [x] Load ordered slices from one scan
- [x] Implement previous/current/next slice stacking (CTSliceStacker)
- [x] Handle boundary slices (Edge replication / clamp)
- [x] Verify slice ordering & patient/scan consistency (Zero cross-patient leak)
- [x] Test tensor shapes ([3, 224, 224] per sample, [B, 3, 224, 224] per batch)
- [x] Build a reusable 2.5D Dataset class (CT25DDataset)
- [x] Build and test the DataLoader (create_dataloader)

### Module 6 — Transfer Learning
- [x] Understand ResNet architecture (Residual skip connections)
- [x] Understand DenseNet architecture (Dense feature reuse)
- [x] Understand EfficientNet architecture (Compound scaling)
- [x] Load pretrained weights (Torchvision ImageNet-1K)
- [x] Replace classification heads (Dropout 0.3 + Linear in_features -> 2)
- [x] Understand input-channel requirements (3 channels for 2.5D CT stack)
- [x] Compare parameter counts (Custom: 93k, EfficientNet: 4M, DenseNet: 6.9M, ResNet: 23.5M)
- [x] Compare inference requirements (EfficientNet-B0 selected for deployment)
- [x] Run initial forward passes (Automated test suite: 16/16 passed)
- [x] Verify model output dimensions (Shape [B, 2])

### Module 7 — Model Training
- [x] Create training configuration (configs/training.yaml)
- [x] Implement training loop (src/training/train.py)
- [x] Implement validation loop (src/training/validate.py)
- [x] Configure loss function with class weighting ([0.67, 2.0] for 1:3 ratio)
- [x] Configure optimizer & learning-rate scheduling (AdamW + CosineAnnealingLR)
- [x] Add checkpoint saving and early stopping (src/training/checkpoint.py)
- [x] Track experiments (Training epoch metrics & Macro F1 logs)
- [x] Train 2.5D EfficientNet-B0 on Apple Metal MPS
- [x] Save best validation checkpoint (models/best_model.pth: 46MB)

### Module 8 — Model Evaluation
- [x] Evaluate on validation & test sets
- [x] Select best model checkpoint (models/best_model.pth)
- [x] Evaluate on held-out test set
- [x] Generate confusion matrix (reports/figures/confusion_matrix.png)
- [x] Calculate class-wise metrics (Accuracy, Sensitivity, Specificity, Precision, Macro F1)
- [x] Analyse false positives and false negatives
- [x] Measure inference latency (real-time per-sample inference)
- [x] Document limitations & write evaluation report (reports/experiments/EVALUATION_REPORT.md)

### Module 9 — Explainable AI
- [x] Understand Grad-CAM mechanics & gradients
- [x] Identify target convolutional layer (features[-1] in EfficientNet)
- [x] Generate activation maps & pooled gradients
- [x] Generate normalized heatmaps [0.0, 1.0]
- [x] Overlay heatmaps on original CT slices (cv2.COLORMAP_JET)
- [x] Render multi-panel figures (reports/figures/gradcam_sample.png)
- [x] Automated unit tests for Grad-CAM (18/18 total tests passing)
- [x] Document interpretability limitations

### Module 10 — Streamlit Application
- [x] Create Streamlit entry point (app/streamlit_app.py)
- [x] Design landing page with project overview & disclaimers
- [x] Add CT image upload with 2.5D neighbour slice validation
- [x] Implement preprocessing & cached model loading (@st.cache_resource)
- [x] Display predictions, class probabilities, and Grad-CAM
- [x] Add model info, methodology & limitations tabs
- [x] Test application responsiveness (20/20 unit tests passing)

### Module 11 — Deployment and Optimisation
- [ ] Create deployment requirements
- [ ] Remove unnecessary dependencies
- [ ] Reduce model-loading overhead
- [ ] Cache the model using Streamlit resource caching
- [ ] Cache suitable reusable computations
- [ ] Avoid loading large datasets at application startup
- [ ] Implement lazy loading
- [ ] Test CPU inference
- [ ] Measure initial application startup
- [ ] Measure prediction latency
- [ ] Deploy the application
- [ ] Test the public URL
- [ ] Check application logs
- [ ] Document deployment limitations

### Module 12 — Testing, Documentation and Finalisation
- [ ] Write preprocessing tests
- [ ] Write model input tests
- [ ] Write prediction tests
- [ ] Write upload validation tests
- [ ] Test the full inference pipeline
- [ ] Create a model card
- [ ] Document dataset sources
- [ ] Document training methodology
- [ ] Document evaluation
- [ ] Document known limitations
- [ ] Write setup instructions
- [ ] Add screenshots
- [ ] Create final architecture diagrams
- [ ] Prepare a project demonstration
- [ ] Prepare technical interview explanations

## Current Concept
Module 10 Completed — Ready for Modules 11 & 12 (Deployment & Final Packaging)

## Concepts Completed
- 2D vs 2.5D vs 3D modelling
- System architecture (training pipeline + inference app)
- Dataset selection: COVID-CTset (Primary) + InRad (External)
- Project limitations documented
- Dataset Audit & Zero-Leakage Split verification
- 16-bit CT Windowing, Dynamic Range Scaling, Safe Augmentation Pipeline
- CNN Mechanics & Custom Baseline Model (`PulmoCustomCNN`, 93k params)
- 2.5D Slice Stacking Engine (`CTSliceStacker`) & PyTorch DataLoader (`CT25DDataset`)
- Transfer Learning Backbones (ResNet50, EfficientNet-B0, DenseNet121) & Unified Model Factory
- Model Training Loop, Class-Weighted Loss, Early Stopping & Checkpoint Saving
- Quantitative Evaluation (Macro F1, Sensitivity, Specificity, Confusion Matrix)
- Explainable AI: Hook-based Grad-CAM & Multi-Panel Visualization
- Interactive Multi-Page Streamlit Web Application

## Pending Questions
- None

## Errors Encountered
- None

## Deliverables
- [x] README.md (including 8-question developer Q&A)
- [x] AGENTS.md
- [x] docs/limitations.md
- [x] docs/PROJECT_PROGRESS.md
- [x] docs/research_strategy.md
- [x] docs/data_and_pipeline_overview/ (Charts, flows, documentation)
- [x] reports/dataset_audit/COVID_CTSET_AUDIT.md
- [x] reports/experiments/EVALUATION_REPORT.md
- [x] reports/figures/confusion_matrix.png
- [x] reports/figures/gradcam_sample.png
- [x] requirements.txt
- [x] configs/training.yaml
- [x] src/data/dataset_audit.py
- [x] src/data/generate_charts.py
- [x] src/data/preprocessing.py
- [x] src/data/slice_stacking.py
- [x] src/data/dataloader.py
- [x] src/models/custom_cnn.py
- [x] src/models/resnet.py
- [x] src/models/efficientnet.py
- [x] src/models/densenet.py
- [x] src/models/model_factory.py
- [x] src/training/checkpoint.py
- [x] src/training/train.py
- [x] src/training/validate.py
- [x] src/evaluation/metrics.py
- [x] src/evaluation/confusion_matrix.py
- [x] src/evaluation/error_analysis.py
- [x] src/explainability/gradcam.py
- [x] src/inference/predictor.py
- [x] app/streamlit_app.py
- [x] tests/ (20/20 unit tests passing)

## Next Action
Proceed with Modules 11 & 12 (Deployment Optimization, Model Card, and Final Portfolio Documentation).

