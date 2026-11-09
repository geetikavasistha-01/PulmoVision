# PulmoVision — AI IDE Instructions

## Project Description
PulmoVision is an AI-powered medical imaging research application that analyses
chest CT scans using 2.5D deep learning models. It is an educational research
prototype — NOT a diagnostic tool.

## Development Style
- Course-based, milestone-driven development.
- One concept at a time. Explain before generating code.
- Follow the module order defined in the PRD.
- Maintain `docs/PROJECT_PROGRESS.md` at every step.

## Key Rules
1. Never skip teaching — explain what, why, and how for every concept.
2. Never invent metrics or dataset characteristics.
3. Never treat dataset labels as confirmed medical diagnoses.
4. Never advance to the next module without developer confirmation.
5. Maintain the five-branch Git strategy (main + 4 feature branches).
6. Test each module before moving forward.

## Architecture
- **2.5D Medical Image Classification** using neighbouring CT slices as channels.
- **PyTorch** for deep learning. **Streamlit** for the application.
- Training pipeline and inference app are logically separate.

## Current Status
Refer to `docs/PROJECT_PROGRESS.md` for progress tracking.

