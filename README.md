# 🫁 PulmoVision

> AI-powered chest CT scan analysis using 2.5D deep learning — an educational research prototype.

⚠️ **Research only.** This is NOT a diagnostic tool. Dataset labels are categories, not clinical diagnoses.

---

## What It Does

| Feature | Description |
|---|---|
| 2.5D Classification | Stacks neighbouring CT slices as channels for spatial context |
| Model Comparison | Custom CNN → ResNet50 → EfficientNet-B0 (2D and 2.5D) |
| Explainability | Grad-CAM heatmaps showing model attention regions |
| Web Interface | Streamlit app for upload, prediction, and visualisation |

## Architecture

```
CT Slices → 2.5D Stacking → Pretrained CNN → Classification + Grad-CAM → Streamlit Dashboard
```

| Component | Stack |
|---|---|
| Deep Learning | PyTorch, Torchvision |
| Data | NumPy, Pandas, OpenCV, Pillow |
| Medical Imaging | pydicom, SimpleITK (optional) |
| Experiment Tracking | MLflow |
| Explainability | Grad-CAM |
| Application | Streamlit |
| Testing | Pytest |

## Dataset

**COVID-CTset** (Kaggle, CC BY 4.0) — ~15,000 CT images, 377 patients, patient-level IDs, ordered slices.

## Project Structure

```
PulmoVision/
├── app/                  # Streamlit application
├── configs/              # YAML configuration files
├── data/                 # Raw/processed data (not committed)
├── notebooks/            # Exploration notebooks
├── src/
│   ├── data/             # Dataset access, audit, preprocessing
│   ├── models/           # CNN, ResNet, EfficientNet definitions
│   ├── training/         # Training loop, checkpoints, MLflow
│   ├── evaluation/       # Metrics, confusion matrix, error analysis
│   ├── explainability/   # Grad-CAM
│   └── inference/        # Prediction pipeline
├── models/               # Saved checkpoints (not committed)
├── reports/              # Audit reports, experiment results
├── tests/                # Automated tests
└── docs/                 # Documentation
```

## Quick Start

```bash
# Clone
git clone https://github.com/<your-username>/PulmoVision.git
cd PulmoVision

# Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run app
streamlit run app/streamlit_app.py
```

## Git Branching

| Branch | Modules | Purpose |
|---|---|---|
| `main` | — | Stable, deployable |
| `feature/data-pipeline` | 0–3 | Data access, audit, preprocessing |
| `feature/model-development` | 4–7 | CNNs, 2.5D, training |
| `feature/evaluation-xai` | 8–9 | Evaluation, Grad-CAM |
| `feature/streamlit-deployment` | 10–12 | App, deploy, tests, docs |

## License

Educational research project. Dataset used under CC BY 4.0.

## Disclaimer

This application is a research prototype for educational purposes only. It does not provide medical advice, diagnosis, or treatment recommendations.

