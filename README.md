# CerebroAI: Deep Learning Brain MRI Diagnostic Suite & Generative Augmentation

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ADNI](https://img.shields.io/badge/Dataset-ADNI-blueviolet.svg)](https://adni.loni.usc.edu/)

**CerebroAI** is an end-to-end deep learning neuroimaging platform for automated **Alzheimer's Disease (AD)** vs. **Cognitively Normal (CN)** diagnosis from 3D T1-weighted Brain MRI scans. Incorporates generative data augmentation using **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to resolve severe class imbalance, a **Custom ResNet-18** classifier, **Grad-CAM Explainable AI (XAI)**, and an **interactive Streamlit Web Suite**.

---

## 🌟 Key Features

- **Leakage-Free Preprocessing**: Subject-level stratification across train/val/test splits to eliminate patient data leakage.
- **Volumetric Slice Extraction**: Automated conversion of 3D NIfTI (`.nii`) volumes into standardized 2D axial slices ($192 \times 160$) targeting hippocampal and ventricular regions.
- **Generative Data Augmentation (WGAN-GP)**: Improved Wasserstein GAN enforcing 1-Lipschitz continuity to synthesize realistic minority-class brain slices.
- **Custom ResNet-18 Classifier**: Optimized residual architecture tailored for single-channel grayscale neuroimaging.
- **Explainable AI (Grad-CAM)**: Visual saliency heatmaps localizing anatomical biomarkers (e.g. ventricular enlargement, temporal lobe atrophy).
- **Patient-Level Aggregation**: Majority-voting and multi-slice pooling engine translating 2D slice inference into a comprehensive 3D patient diagnosis.
- **Interactive Web Interface**: Streamlit application for real-time drag-and-drop MRI diagnostic evaluation.

---

## 🖼️ Architecture & Workflow

```text
               ADNI 3D T1-Weighted MRI Volumes (.nii)
                                 │
                                 ▼
       ┌───────────────────────────────────────────────────┐
       │ 1. Subject-Level Preprocessing & Filtering        │
       │    • Patient-stratified split (Train / Val / Test)│
       │    • Informative middle-brain slice selection     │
       │    • Intensity normalization & resizing (192x160) │
       └─────────────────────────┬─────────────────────────┘
                                 │
        ┌────────────────────────┴─────────────────────────┐
        ▼                                                  ▼
┌───────────────────────────────┐          ┌───────────────────────────────┐
│ 2. Class Imbalance Management │          │ 3. Generative Augmentation    │
│    • Stratified Undersampling │          │    • WGAN-GP Synthesis        │
│    • Random Subject Balancing │          │    • Realistic Minority Slices│
└───────────────┬───────────────┘          └───────────────┬───────────────┘
                │                                          │
                └────────────────────────┬─────────────────┘
                                         │
                                         ▼
       ┌───────────────────────────────────────────────────┐
       │ 4. Deep Residual Classification (Custom ResNet-18)│
       │    • Binary (AD vs. CN) & Multi-class (MCI)       │
       │    • Grad-CAM Visual Saliency & Interpretability  │
       │    • Multi-Slice Patient-Level Aggregator         │
       └───────────────────────────────────────────────────┘
```

![WGAN Generation Example](reports/Generation_Example.png)

---

## 📁 Repository Structure

```text
CerebroAI/
├── app.py                     # Interactive Streamlit Web Diagnostic App
├── main.py                    # Unified CLI entrypoint (preprocess, train, infer)
├── pyproject.toml             # Package metadata & build configuration
├── requirements.txt           # Python dependencies
├── data/
│   ├── adni-metadata/         # Clinical, demographic (PTDEMOG), APOE, TOMM40 CSVs
│   └── adni-images/           # Preprocessed axial PNG slices
├── models/                    # Model weights and checkpoints (.h5)
├── notebooks/                 # Exploratory data analysis & experiments
│   ├── biospecimen_analysis.ipynb
│   └── subjects_analysis.ipynb
├── output/                    # Grad-CAM outputs, confusion matrices, logs
├── reports/                   # Figures and project documentation
│   └── Generation_Example.png
└── src/                       # Production modular package
    ├── data/
    │   └── preprocessing.py   # Subject splitting, NIfTI loading, normalization
    ├── models/
    │   ├── resnet18.py        # Custom ResNet-18 architecture
    │   └── wgan_gp.py         # WGAN-GP generator, critic, and training loop
    ├── explainability/
    │   └── gradcam.py         # Grad-CAM heatmap & superposition generator
    └── utils/
        └── metrics.py         # Clinical metrics & patient-level aggregation
```

---

## ⚡ Quick Start & Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/CerebroAI.git
cd CerebroAI

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### 1. Preprocessing (Subject Split & Slice Extraction)
```bash
python main.py preprocess --raw_dir ./data/adni-mri --split_dir ./data/adni-split --png_dir ./data/adni-images
```

### 2. Train Custom ResNet-18 Classifier
```bash
python main.py train_classifier --data_dir ./data/adni-images --epochs 50 --batch_size 32
```

### 3. Train WGAN-GP Generative Augmentor
```bash
python main.py train_wgan --data_dir ./data/adni-images/train/ad --epochs 200 --latent_dim 128
```

### 4. Launch Interactive Web Diagnostic Suite
```bash
streamlit run app.py
```

---

## 📊 Evaluation & Clinical Metrics

The pipeline calculates key clinical performance metrics:
- **Sensitivity / Recall**: Detection rate of Alzheimer's Disease patients ($TP / (TP + FN)$).
- **Specificity**: True negative rate for healthy normal controls ($TN / (TN + FP)$).
- **Balanced Accuracy**: Arithmetic mean of sensitivity and specificity to account for class imbalance.
- **ROC-AUC**: Area under the receiver operating characteristic curve.
- **Patient-Level Voting**: Robust aggregation preventing slice-level classification bias.

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
