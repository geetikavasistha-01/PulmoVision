# PulmoVision Progress Tracker

## Current Module
Module 0 — Project Foundations

## Overall Progress
In Progress — Module 0

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
- [ ] Complete knowledge assessment

### Module 1 — Development Environment
- [ ] Create the project repository
- [ ] Create a Python virtual environment
- [ ] Install required packages
- [ ] Verify PyTorch installation
- [ ] Check GPU availability
- [ ] Verify image libraries
- [ ] Configure Git
- [ ] Create initial folders
- [ ] Run a small test script
- [ ] Document environment versions

### Module 2 — Dataset Discovery and Audit
- [ ] Explore available dataset APIs
- [ ] Configure programmatic access
- [ ] Discover the selected dataset
- [ ] Inspect metadata
- [ ] Inspect image formats
- [ ] Inspect sample images
- [ ] Understand all class labels
- [ ] Identify patient identifiers
- [ ] Identify duplicate records
- [ ] Check class distribution
- [ ] Generate an audit report
- [ ] Approve the dataset for processing

### Module 3 — Medical Image Processing
- [ ] Load sample images
- [ ] Visualise CT slices
- [ ] Inspect image dimensions
- [ ] Understand pixel intensity
- [ ] Learn windowing
- [ ] Implement preprocessing
- [ ] Compare different normalisation approaches
- [ ] Implement safe augmentation
- [ ] Visualise before-and-after images
- [ ] Validate preprocessing outputs

### Module 4 — Understanding CNNs
- [ ] Understand tensor shapes
- [ ] Implement a convolution operation
- [ ] Visualise feature maps
- [ ] Understand pooling
- [ ] Build a small CNN
- [ ] Train on a small dataset subset
- [ ] Inspect training loss
- [ ] Inspect validation loss
- [ ] Experiment with learning rates
- [ ] Identify overfitting

### Module 5 — Understanding 2.5D Modelling
- [ ] Understand 2D, 2.5D and 3D
- [ ] Select a neighbouring-slice strategy
- [ ] Load ordered slices from one scan
- [ ] Implement previous/current/next slice stacking
- [ ] Handle boundary slices
- [ ] Verify slice ordering
- [ ] Verify patient and scan consistency
- [ ] Visualise 2.5D input channels
- [ ] Test tensor shapes
- [ ] Build a reusable 2.5D Dataset class
- [ ] Build and test the DataLoader

### Module 6 — Transfer Learning
- [ ] Understand ResNet architecture
- [ ] Understand DenseNet architecture
- [ ] Understand EfficientNet architecture
- [ ] Load pretrained weights
- [ ] Replace classification heads
- [ ] Understand input-channel requirements
- [ ] Compare parameter counts
- [ ] Compare inference requirements
- [ ] Run initial forward passes
- [ ] Verify model output dimensions

### Module 7 — Model Training
- [ ] Create training configuration
- [ ] Implement training loop
- [ ] Implement validation loop
- [ ] Configure loss function
- [ ] Configure optimizer
- [ ] Add learning-rate scheduling
- [ ] Add checkpoint saving
- [ ] Add early stopping
- [ ] Track experiments with MLflow
- [ ] Train the baseline
- [ ] Train 2.5D ResNet
- [ ] Train 2.5D EfficientNet
- [ ] Compare experiments
- [ ] Save the best validation checkpoint

### Module 8 — Model Evaluation
- [ ] Evaluate on the validation set
- [ ] Select the model without using the test set for tuning
- [ ] Evaluate once on the held-out test set
- [ ] Generate a confusion matrix
- [ ] Calculate class-wise metrics
- [ ] Analyse false positives
- [ ] Analyse false negatives
- [ ] Compare model performance
- [ ] Measure inference latency
- [ ] Document limitations
- [ ] Perform external validation if compatible data is available

### Module 9 — Explainable AI
- [ ] Understand Grad-CAM
- [ ] Identify the target convolutional layer
- [ ] Generate activation maps
- [ ] Generate heatmaps
- [ ] Overlay heatmaps on original CT slices
- [ ] Compare correct predictions
- [ ] Inspect incorrect predictions
- [ ] Document limitations

### Module 10 — Streamlit Application
- [ ] Create Streamlit entry point
- [ ] Design landing page
- [ ] Add project overview
- [ ] Add CT image upload
- [ ] Validate uploaded files
- [ ] Implement preprocessing
- [ ] Load trained model
- [ ] Implement prediction
- [ ] Display class scores
- [ ] Display Grad-CAM
- [ ] Add model information
- [ ] Add methodology and limitations page
- [ ] Test invalid uploads
- [ ] Test multiple uploads
- [ ] Test application responsiveness

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
Module 0 — Knowledge assessment (final item)

## Concepts Completed
- 2D vs 2.5D vs 3D modelling
- System architecture (training pipeline + inference app)
- Dataset selection: COVID-CTset approved
- Project limitations documented

## Pending Questions
- None

## Errors Encountered
- None

## Deliverables
- [x] README.md
- [x] AGENTS.md
- [x] docs/limitations.md
- [x] docs/PROJECT_PROGRESS.md
- [x] .gitignore
- [x] .env.example
- [x] Project folder structure

## Next Action
Complete Module 0 knowledge assessment, then begin Module 1.

