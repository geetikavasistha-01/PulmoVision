# PulmoVision — 2–3 Minute Demonstration Script

**Title:** Thoracic CT Sequence Intelligence: Academic Research Demonstration  
**Target Duration:** 2 minutes 45 seconds  
**Presenter Tone:** Objective, rigorous, technically precise, and scientifically honest.  
**Strict Rule:** No diagnostic claims; emphasize educational research prototype boundaries throughout.

---

### Segment 1: Introduction & The Sequence Modeling Problem [0:00 – 0:30]

**Visual:** Launch on **Research Overview** in *Clinical Dark* theme (`http://localhost:8501`). Point to the top metric cards (0.9461 Macro F1, 98.7% Sensitivity, 1,460,418 parameters).

**Narration:**
> "Welcome to PulmoVision, an AI medical imaging research platform exploring deep sequence modeling on thoracic CT scans.
> 
> In thoracic CT imaging, clinicians face a classic trade-off: traditional 2D slice classifiers treat adjacent anatomical slices as independent images, missing axial continuity and producing high false-positive rates. Conversely, full 3D isotropic CNNs impose severe GPU memory constraints and require dense z-axis resampling.
> 
> PulmoVision resolves this through a 2.5D sequence architecture that pairs frozen 2D convolutional feature extraction with bidirectional recurrent sequence modeling.
> 
> Before we begin, an essential disclaimer: PulmoVision is strictly an academic research prototype, not a medical diagnostic tool. Model predictions represent uncalibrated empirical research outputs."

---

### Segment 2: Dataset Explorer & Cryptographic Deduplication [0:30 – 1:00]

**Visual:** Navigate via the sidebar to **Dataset Explorer** (`:material/dataset:`). Demonstrate the search bar by typing `patient124`. Toggle the "Show duplicate provenance" checkbox to show canonical slices vs. aliases.

**Narration:**
> "Let's inspect the data foundation. PulmoVision is developed on the COVID-CTset archive, encompassing 122 development patients across 239 distinct acquisition series.
> 
> A critical milestone of our research was identifying and resolving 604 redundant duplicate slice aliases using cryptographic SHA-256 hashing. This normalized the archive to exactly 3,132 canonical anatomical slices.
> 
> Our Dataset Explorer enables researchers to query by patient, series, or label, and inspect full duplicate provenance. Throughout all development, the official test partition, `test1.csv`, remained strictly sealed and quarantined to prevent data leakage."

---

### Segment 3: Model Laboratory & Experiment Analytics [1:00 – 1:40]

**Visual:** Navigate to **Model Laboratory** (`:material/model_training:`) and show the parameter comparison, then click **Experiment Analytics** (`:material/analytics:`) to display the ROC curve and confusion matrix.

**Narration:**
> "In the Model Laboratory, we systematically benchmarked three sequence paradigms using 5-fold outer cross-validation with nested validation holdouts:
> 
> Experiment A applied uniform atemporal mean pooling, yielding an 82.4% Macro F1 with 17 false negatives.
> 
> Experiment B, our development leader, integrated a 2-layer Bidirectional LSTM with global mean pooling. It achieved a 94.6% Macro F1, a 0.9688 ROC-AUC, and reduced false negatives from 17 down to a single patient, achieving 98.7% sensitivity.
> 
> Experiment C evaluated additive temporal attention. While achieving 92.1% Macro F1, attention parameters exhibited mild overfitting on our 122-patient cohort compared to the regularizing effect of global pooling across contextualized recurrent states."

---

### Segment 4: CT Workstation & Live Sequence Inference [1:40 – 2:20]

**Visual:** Navigate to **CT Workstation** (`:material/medical_services:`). Select the pre-loaded series `patient1_SR_2 (COVID-19, 6 slices)`. Drag the craniocaudal axial slider. Point to the pitch-black CT viewport (`#000000`). Click `Run Sequence Inference`.

**Narration:**
> "Now let's examine the CT Workstation. The interface incorporates a clinical viewing canvas, maintaining a pitch-black radiological background in both Dark and Light themes.
> 
> When raw 16-bit TIFF series are loaded, our Sequence Validator checks for single-patient and single-series consistency and enforces craniocaudal slice ordering from apex to lung base.
> 
> Clicking 'Run Sequence Inference' executes our frozen deployment model against the uncompressed scans. In under 850 milliseconds—roughly 133 milliseconds per slice—the pipeline windows the radiodensities to parenchymal ranges, extracts embeddings, and runs bidirectional sequence aggregation. The model outputs a continuous score with latency profiling and explicit warnings regarding single-slice limitations."

---

### Segment 5: Explainability, Theme Switcher & Safety Posture [2:20 – 2:45]

**Visual:** Navigate to **Explainability & Boundaries** (`:material/visibility:`). Scroll to the Grad-CAM parenchymal overlay. Use the sidebar appearance switcher to toggle from Dark to Light mode.

**Narration:**
> "In Explainability and Boundaries, we inspect Grad-CAM activation maps overlaid on pulmonary parenchyma. Crucially, the platform emphasizes clinical limitations: ground-glass opacities captured by the network are non-specific and overlap with other viral pneumonias, organizing pneumonia, and atelectasis.
> 
> The platform seamlessly switches between Clinical Dark and Laboratory Light modes, maintaining consistent typography and semantic indicators.
> 
> With 95 passing automated unit and regression tests, zero emojis, and complete research documentation embedded directly in the application, PulmoVision demonstrates how rigorous ML engineering and medical imaging software can unite in an open research platform. Thank you."
