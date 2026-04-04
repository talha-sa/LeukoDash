<div align="center">

# 🩸 LeukoDash

### AI-Powered Leukemia Gene Expression Analysis Dashboard

[![Streamlit App](https://img.shields.io/badge/🚀%20Live%20Demo-LeukoDash-FF4B4B?style=for-the-badge)](https://leukodash.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![FYP](https://img.shields.io/badge/Final%20Year%20Project-UAF-8B5CF6?style=for-the-badge)](https://uaf.edu.pk)

> 🎓 *Final Year Project — University of Agriculture, Faisalabad (UAF)*

---

**LeukoDash** is a no-code bioinformatics web dashboard that empowers researchers and clinicians to perform end-to-end leukemia gene expression analysis — from raw data to survival curves and ML-based cancer classification — all through an intuitive visual interface.

</div>

---

## 🌟 Key Highlights

| Feature | Details |
|--------|---------|
| 🧬 Dataset | Golub et al. (1999) — 72 patients, 7,129 genes |
| 🎯 ML Accuracy | Up to **100%** on ALL vs AML classification |
| 🧪 Analysis Modules | 7 scientifically rigorous modules |
| 💻 No Coding Required | Fully interactive web interface |
| ☁️ Cloud Deployed | Live on Streamlit Cloud |

---

## 🔬 Analysis Modules

<details>
<summary>📌 <b>Click to explore all modules</b></summary>

<br>

### 🧫 1. Biomarker Discovery
Identifies the top most variable genes across patient samples — these serve as potential diagnostic biomarkers for leukemia subtype distinction.

### 📊 2. Gene Expression Visualization
Explore expression patterns through interactive **heatmaps** and **distribution histograms** to understand gene activity across ALL and AML samples.

### 📉 3. Differential Expression Analysis *(New!)*
Statistically compares gene expression between ALL and AML groups, highlighting significantly up- and down-regulated genes using volcano plots and fold-change analysis.

### 🔵 4. Clustering Analysis *(New!)*
Unsupervised clustering (K-Means / Hierarchical) to discover natural groupings in the data — revealing hidden biological structure without labels.

### 🌀 5. Dimensionality Reduction — PCA & UMAP *(New!)*
Reduces 7,129 gene dimensions down to 2D/3D space for visual exploration of sample separation, variance, and cluster structure.

### 📈 6. Kaplan-Meier Survival Curves *(New!)*
Generates survival probability plots to compare patient outcomes between leukemia subtypes — a clinically critical visualization used in oncology research.

### 🤖 7. ML-Based Leukemia Classification *(Upgraded!)*
Train and compare **three machine learning models** head-to-head:
- Logistic Regression
- Random Forest 🌲
- Support Vector Machine (SVM)

Each model outputs accuracy, confusion matrix, and classification report for full transparency.

</details>

---

## 🧬 Dataset

> **Golub et al. (1999)** — *Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring* — Science

| Property | Value |
|----------|-------|
| 👥 Samples | 72 patient samples |
| 🧬 Features | 7,129 gene expression values |
| 🏷️ Classes | ALL (Acute Lymphoblastic Leukemia) & AML (Acute Myeloid Leukemia) |
| 📦 Source | [Kaggle Dataset](https://kaggle.com/datasets/crawford/gene-expression) |

---

## 🛠️ Built With

![Python](https://skillicons.dev/icons?i=python,github,vscode)

| Library | Purpose |
|---------|---------|
| `Streamlit` | Web app framework |
| `Scikit-learn` | ML models & evaluation |
| `Pandas / NumPy` | Data processing |
| `Matplotlib / Seaborn` | Visualization |
| `SciPy` | Statistical testing |
| `Lifelines` | Kaplan-Meier survival analysis |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/talha-sa/LeukoDash.git
cd LeukoDash

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

> 💡 Python 3.9+ recommended

---

## 👨‍💻 Developer

<div align="center">

**Talha Saleem**

BSc Bioinformatics — University of Agriculture, Faisalabad (UAF) · 2021–2025

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/talha-sa)
[![GitHub](https://img.shields.io/badge/GitHub-talha--sa-181717?style=for-the-badge&logo=github)](https://github.com/talha-sa)
[![Live App](https://img.shields.io/badge/🌐%20Live%20App-leukodash.streamlit.app-FF4B4B?style=for-the-badge)](https://leukodash.streamlit.app)

</div>

---

<div align="center">

*Made with ❤️ for bioinformatics research | UAF Final Year Project 2025*

</div>