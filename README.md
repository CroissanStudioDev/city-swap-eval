# Resume Screening

This repository provides code and Jupyter notebooks for automated resume screening using BERT-based sequence classification. The aim is to classify resumes into predefined categories to support recruitment workflows and candidate analysis.

---

## Important Note: Missing Files Due to `.gitignore`

Several folders, large files, and checkpoints are intentionally **not included** in the repository and are specified in `.gitignore`, so you will not find the following:

- **Any actual datasets** (`data/`)
- **Trained or fine-tuned model checkpoints** (`models/`, `bert_9classes/`, `bert_9classes_sqrt_rw_2ep/`, `bert_9classes_gdro/`, etc.)
- **Experiment logs/outputs** (`runs/`)
- **Intermediate files** (`*.pkl`, `*.pt`, `*.joblib`, etc.)
- **Notebooks in certain draft/working folders** or repeated directories
- **EDA notebook** (`notebooks/00_eda.ipynb`)
- Other files and folders as listed in the repository's `.gitignore`

You are welcome to use the code and templates here, but you must supply your own data and generate your own models and results.

---

## Contents Overview

- **Scripts and configuration templates** for BERT sequence classification (not including trained models)
- **Jupyter notebooks** for experimentation, analysis, and visualization (some notebooks may not be present, see above)
- **Code for preprocessing, data exploration, and analysis**

---

## Notebook Guide

Within the `notebooks/` folder you’ll find:

- **10_baseline**: Example baseline experiment (requires user-provided data)
- **20–29, 60–63**: Model training pipelines and fairness methods (requires your own data/models)
- **30–31**: Notebooks for data-centric processing steps
- **40–42**: Fairness/interpretability analysis code and workflow
- **50**: Plot/figure generation notebooks

Some notebook files (most notably `00_eda.ipynb`) are not included in the repository.

---

## Directory Structure

- `notebooks/` – Notebooks for experimentation (some may be excluded as above)
- `figures/` - Plots, graphics, and other data generated from the notebooks

---

## Getting Started

1. **Clone the repository:**
    ```
    git clone https://github.com/yourusername/resume-screening.git
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Add your data:**  
   Place your datasets inside a `data/` folder (not included in this repository).

4. **Explore notebooks, scripts, and configuration templates.**  
   _Note: Most components require your own data; model checkpoints and some notebooks are not provided due to `.gitignore`._

---

## Model & Config

- **Model:** BERT for Sequence Classification (9 classes)
- **No trained model files/checkpoints are included in the repository**

---

## Results

Reference metrics from the BERT + sqrt_rw baseline on private data (for context only):

- **Accuracy:** 0.609
- **Macro F1:** 0.621
- **TPR Gap (Worst Robust):** 0.329
- **TPR Gap (Macro Robust):** 0.116

To reproduce these results, you will need to train your own models on your data.

---

## Notes

- Many referenced files (datasets, model checkpoints, experimental logs, some notebook files) are **not** available due to `.gitignore`.
- Intended for research and experimentation.