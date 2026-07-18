<div align="center">

# 🔷 Machine Learning-Based Anomaly Detection in Distributed Control Systems

### Using the Tennessee Eastman Process Dataset

![Python](https://img.shields.io/badge/Python-3.11-1E3A8A?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-2563EB?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-1D4ED8&style=for-the-badge&logo=tensorflow&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-3B82F6?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-0EA5E9?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Progress-38BDF8?style=for-the-badge)

**MSc Artificial Intelligence · Bahrain Polytechnic**
**Maryam Ali · Student ID 202509001**

</div>

---

## 🔹 Overview

This repository contains the implementation for an MSc thesis investigating early, accurate anomaly detection in Distributed Control System (DCS) environments. The project benchmarks three machine learning paradigms — **unsupervised**, **supervised**, and **semi-supervised** — against the **Tennessee Eastman Process (TEP)** dataset, and applies **SHAP explainability** to translate model outputs into actionable insight for control-room engineers.

The work is grounded in real DCS operational practice, drawing on professional experience with the Yokogawa CENTUM VP platform.

---

## 🔹 Table of Contents

- [Research Questions](#-research-questions)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Setup](#-setup)
- [Reproducing Results](#-reproducing-results)
- [Models](#-models)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Author](#-author)
- [Citation](#-citation)

---

## 🔹 Research Questions

| # | Question |
|---|---|
| **RQ1** | Can machine learning models detect process anomalies earlier and more accurately than traditional threshold-based methods? |
| **RQ2** | Which of Isolation Forest, Random Forest, and LSTM Autoencoder achieves the best trade-off between detection accuracy and false alarm rate? |
| **RQ3** | How can SHAP explainability identify which process variables drive detected anomalies, supporting DCS alarm rationalisation? |

---

## 🔹 Repository Structure

```
├── TEP_Anomaly_Detection.ipynb   # Full pipeline: data → EDA → models → SHAP
├── figures/                      # Generated plots (confusion matrices, ROC, SHAP)
├── models/                       # Saved LSTM Autoencoder weights
├── csv/                          # Cached converted CSVs (generated on first run)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

> **Note:** The raw `.RData` dataset files are **not** stored in this repository. See [Dataset](#-dataset) below.

---

## 🔹 Dataset

This project uses the **Tennessee Eastman Process (TEP)** simulation dataset.

| | |
|---|---|
| **Source** | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) |
| **Citation** | Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). *Tennessee Eastman Reference Simulation Dataset for Fault Detection*. Harvard Dataverse. doi:10.7910/DVN/6C3JR1 |
| **License** | CC0 (public domain) |
| **Size** | 52 process variables · 21 fault types + normal operation |

The dataset is **not redistributed in this repository** to keep it lightweight. Download the four `.RData` files directly from the Harvard Dataverse link above and place them in the project root before running the notebook. A working copy is also kept on Google Drive for personal backup.

---

## 🔹 Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/msc-thesis-tep-anomaly-detection.git
cd msc-thesis-tep-anomaly-detection
```

**2. Create a virtual environment** *(recommended)*
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**
Download the four `.RData` files from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) and place them in the project root.

**5. Launch the notebook**
```bash
jupyter notebook TEP_Anomaly_Detection.ipynb
```

---

## 🔹 Reproducing Results

Run all cells top to bottom. The notebook is fully self-contained:

1. **Data loading** — converts `.RData` → cached CSV (first run only; subsequent runs load from cache)
2. **EDA** — missing values, fault distribution, sensor time-series, correlation heatmap
3. **Preprocessing** — binary framing (normal/fault), stratified 70/30 split, standardisation
4. **Model training** — Isolation Forest → Random Forest → LSTM Autoencoder
5. **Evaluation** — Precision, Recall, F1, ROC-AUC for all models on a shared test set
6. **SHAP explainability** — TreeExplainer on Random Forest, summary + waterfall plots

All figures save automatically to `figures/`. Random seed fixed at `42` throughout for full reproducibility.

---

## 🔹 Models

| Model | Paradigm | Role |
|---|---|---|
| 🔷 **Isolation Forest** | Unsupervised | Trains on normal data only; no labels required |
| 🔷 **Random Forest** | Supervised | Full fault labels; basis for SHAP explainability |
| 🔷 **LSTM Autoencoder** | Semi-supervised | Reconstruction error on 10-sample windows of normal operation |

---

## 🔹 Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|
| Isolation Forest | `TBD` | `TBD` | `TBD` | `TBD` |
| Random Forest | `TBD` | `TBD` | `TBD` | `TBD` |
| LSTM Autoencoder | `TBD` | `TBD` | `TBD` | `TBD` |

*Full results, ROC curves, and SHAP visualisations available in `figures/` after running the notebook.*

---

## 🔹 Tech Stack

![Python](https://img.shields.io/badge/-Python-1E3A8A?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-2563EB?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-1D4ED8?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-3B82F6?style=flat-square&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-0EA5E9?style=flat-square&logo=tensorflow&logoColor=white)
![SHAP](https://img.shields.io/badge/-SHAP-38BDF8?style=flat-square)
![Jupyter](https://img.shields.io/badge/-Jupyter-1E40AF?style=flat-square&logo=jupyter&logoColor=white)

---

## 🔹 Author

**Maryam Ali**
DCS Engineer, Yokogawa · MSc Artificial Intelligence, Bahrain Polytechnic
📧 202509001@polytechnic.bh

---

## 🔹 Citation

If referencing this work:

```
M. Ali, "Machine Learning-Based Anomaly Detection in Distributed Control Systems
Using the Tennessee Eastman Process Dataset," MSc Thesis, Bahrain Polytechnic, 2026.
```

---

<div align="center">

*Submitted as part of the MSc AI Thesis, Bahrain Polytechnic — June–September 2026*

</div>
