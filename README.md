# PulmoScan AI — Pneumonia Chest X-Ray Detection & 5-Model Benchmark Studio

End-to-end deep learning research pipeline and **Streamlit** clinical decision-support UI for automated **Pneumonia detection from frontal chest radiographs (CXR)**. Five convolutional architectures (**VGG16**, **EfficientNetB0**, **MobileNetV2**, **ResNet50**, and **Custom_CNN**) are trained, calibrated, stress-tested under clinical acquisition artifacts using **PyTorch CUDA**, and compared side by side with **Grad-CAM** visual explainability.

> **Clinical & Regulatory Disclaimer:** This software is an academic and research prototype for automated chest X-ray screening analysis. It is **not** FDA-cleared / CE-marked and must **not** be used as a standalone clinical diagnosis. Always correlate with clinical symptoms, vital signs (e.g., $\text{SpO}_2$), and formal physician/radiologist review.

---

## Key Features

- **5-Model End-to-End PyTorch CUDA Pipeline** ([`Pneumonia_Detection_Deep_Learning.ipynb`](Pneumonia_Detection_Deep_Learning.ipynb)): Patient-level stratified splitting (zero leakage), clinically valid medical image augmentation, two-stage transfer learning fine-tuning, validation-tuned threshold optimization, probability calibration (Brier Score & ECE), Grad-CAM explainability, and artifact perturbation stress testing.
- **Interactive Streamlit Screening Studio** ([`app.py`](app.py) + [`src/inference.py`](src/inference.py)):
  - **Tab 1 — Radiograph Screening & Clinical Actions:** Upload a frontal chest X-ray (`PNG`, `JPG`, `JPEG`, `BMP`, `TIFF`) or run untouched holdout/test samples; inspect predicted probability vs active threshold ($\tau$), immediate actions to take & protective precautions when positive (plus borderline/negative correlation notes), 3-panel **Grad-CAM** localization overlay, and a **Live 5-Model Cross-Architecture Consensus** check on the same image.
  - **Tab 2 — 5-Model Performance Comparison:** Sortable holdout benchmark table, multi-metric grouped bar charts, parameter-vs-latency efficiency frontier, and interactive Stage 1 & Stage 2 training/validation trajectories (`loss`, `accuracy`, `auc`).
  - **Tab 3 — Champion Model Deep Dive (`VGG16`):** Holdout confusion matrix ($N=600$), Balanced ($\tau=0.50$) vs High-Sensitivity Screening ($\tau=0.35$) threshold comparison, calibration diagnostics, and acquisition artifact resilience benchmarks.

---

## Holdout Test Benchmark (5 Architectures)

Evaluated on an untouched patient-level **Holdout Test cohort ($N = 600$ patients: $135$ Pneumonia, $465$ Normal)** from a stratified **$4,000$-patient RSNA Pneumonia Stage-2 subset** (`2,800` Train / `600` Val / `600` Test, `SEED = 42`, input resolution `224 x 224 x 3`, hardware: `NVIDIA GeForce RTX 5050 Laptop GPU`) at default decision threshold $\tau = 0.50$:

| Model | Composite Score | ROC-AUC | PR-AUC | Sensitivity | Specificity | Precision | F1 Score | Accuracy | Params (M) | GPU Latency (ms/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VGG16 (Champion)** | **0.6709** | **0.8328** | **0.6398** | 0.7407 | 0.7204 | 0.4348 | 0.5479 | 0.7250 | 14.86M | 3.95 |
| **EfficientNetB0** | 0.6503 | 0.7993 | 0.5368 | **0.7556** | **0.7312** | **0.4493** | **0.5635** | **0.7367** | 4.17M | 7.35 |
| **MobileNetV2** | 0.6496 | 0.8098 | 0.6280 | 0.7111 | 0.7140 | 0.4192 | 0.5275 | 0.7133 | 2.39M | 5.12 |
| **ResNet50** | 0.6483 | 0.8057 | 0.6079 | 0.7259 | 0.6946 | 0.4083 | 0.5227 | 0.7017 | 24.03M | 6.06 |
| **Custom_CNN** | 0.5424 | 0.7187 | 0.4080 | 0.6444 | 0.6925 | 0.3783 | 0.4767 | 0.6817 | **0.62M** | **1.51** |

### Champion (`VGG16`) Triage Thresholds & Robustness Summary

- **Balanced / Optimal F1 ($\tau = 0.50$):** Sensitivity `74.07%`, Specificity `72.04%`, Precision `43.48%`, F1 `0.5479` (`TP=100`, `FN=35`, `TN=335`, `FP=130`).
- **High-Sensitivity Clinical Triage ($\tau = 0.35$):** Sensitivity rises to **`82.96%`** (Specificity `60.00%`, Precision `37.58%`, F1 `0.5173`), reducing missed positive cases during initial screening.
- **Calibration:** Brier Score `0.1731` | Expected Calibration Error (ECE) `18.11%` (reflects intentional class-weighted training shift using `pos_weight = 3.4374`).
- **Acquisition Stress Test (`VGG16` ROC-AUC / Sensitivity):**
  - Baseline: `0.8328` / `74.07%`
  - Gaussian Sensor Noise ($\sigma = 15$): `0.7959` (`-0.0370`) / `81.48%`
  - Low Contrast ($\times 0.65$): `0.8348` (`+0.0019`) / `75.56%`
  - Overexposure ($+35$): `0.8441` (`+0.0112`) / `77.78%`
  - Respiratory Motion Blur: `0.8175` (`-0.0154`) / `77.04%`

---

## Repository Structure

```text
Pneumonia_Detection_Project/
├── .streamlit/
│   └── config.toml                         # Clinical light theme & headless server settings
├── results/
│   ├── evaluation_report.json              # Split sizes, thresholds, calibration & stress-test audit
│   ├── model_comparison.csv                # Holdout benchmark table across all 5 architectures
│   └── training_history.json               # Epoch-by-epoch Stage 1 & Stage 2 loss/accuracy/AUC
├── src/
│   ├── __init__.py
│   └── inference.py                        # Model builders, checkpoint loader, preprocessing & Grad-CAM
├── app.py                                  # PulmoScan AI Streamlit dashboard (3 interactive tabs)
├── Pneumonia_Detection_Deep_Learning.ipynb # Full 57-cell PyTorch CUDA training & evaluation notebook
├── requirements.txt                        # Python dependencies (CPU PyTorch index + headless OpenCV for cloud)
├── .gitignore                              # Excludes 11.36 GB dataset/, 233 MB models/*.pth & local IDE files
└── README.md
```

> **Excluded from Git via `.gitignore`:**
> - `dataset/` (~`11.36 GB`: `26,684` training CXRs + `3,000` test CXRs + stage-2 metadata CSVs) and `pneumonia_staged_dataset/` (`6.75 MB`).
> - `models/*.pth` (~`233.33 MB` across 6 PyTorch checkpoints, including `checkpoint_resnet50.pth` at `91.98 MB` and `best_pneumonia_model.pth` / `checkpoint_vgg16.pth` at `56.71 MB` each, excluded to stay under GitHub file-size limits).
> - Local machine paths (`.vscode/`, `pyrefly.toml`) and Python/Jupyter caches (`__pycache__/`, `.ipynb_checkpoints/`).

---

## Setup & Quick Start

### 1. Create Environment & Install Dependencies

Recommended Python version: **Python 3.11** (tested locally with `torch 2.15.0+cu134` and `streamlit 1.67.0`).

```bash
conda create -n pneumonia_gpu python=3.11 -y
conda activate pneumonia_gpu

# Optional (local NVIDIA GPU users): install your CUDA wheel first from https://pytorch.org/get-started/locally/
# Then install all project dependencies (Streamlit Cloud uses the CPU PyTorch index in requirements.txt):
pip install -r requirements.txt
```

### 2. Obtain Dataset & Generate Trained Weights (`models/*.pth`)

1. Download the **RSNA Pneumonia Detection Challenge (Stage-2)** PNG/metadata dataset and extract it into `dataset/` with this structure:
   ```text
   dataset/
   ├── stage2_train_metadata.csv
   ├── stage2_test_metadata.csv
   ├── Training/Images/*.png
   └── Test/*.png
   ```
2. Open and run [`Pneumonia_Detection_Deep_Learning.ipynb`](Pneumonia_Detection_Deep_Learning.ipynb) top to bottom (set `MAX_COHORT_SAMPLES = 4000` for the benchmarked subset or `None` to train on all `26,684` patients). Running Section 25 exports the `.pth` checkpoints into `models/` and refreshes `results/`.

*(Note: If you already possess trained checkpoints, place `best_pneumonia_model.pth` and `checkpoint_<arch>.pth` inside `models/`.)*

### 3. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The app starts in light clinical mode at `http://localhost:8501`.

---

## Uploading to GitHub (Important Notes)

This repository is pre-configured with [`.gitignore`](.gitignore) so standard `git add .` stays around **`~6 MB`** and avoids GitHub push rejections:

```bash
git init
git add .
git status   # Confirm dataset/, models/*.pth, .vscode/, and pyrefly.toml are NOT staged
git commit -m "Add Pneumonia detection notebook, evaluation results, and Streamlit UI"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

**Optional — Tracking Checkpoints via Git LFS:** If you also want trained `.pth` weights hosted in your GitHub repository, install [Git LFS](https://git-lfs.com/), remove `models/*.pth` and `*.pth` from `.gitignore`, and run:

```bash
git lfs install
git lfs track "models/*.pth"
git add .gitattributes models/*.pth
```

