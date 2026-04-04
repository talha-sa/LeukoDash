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

**LeukoDash** is a no-code bioinformatics web dashboard for end-to-end leukemia gene expression analysis — from raw data to survival curves, pathway enrichment, and ML-based cancer classification — all through an intuitive visual interface.

</div>

---

## 🌟 Key Highlights

| Feature | Details |
|--------|---------|
| 🧬 Dataset | Golub et al. (1999) — 72 patients, 7,129 genes |
| 🎯 ML Accuracy | Up to **100%** on ALL vs AML classification |
| 🧪 Analysis Modules | 3 fully integrated scientific modules |
| 🌐 GEO Integration | Fetch datasets directly from NCBI GEO database |
| 💻 No Coding Required | Fully interactive web interface |
| ☁️ Cloud Deployed | Live on Streamlit Cloud |

---

## 🔬 Modules

### 🧫 Module 1 — Biomarker Discovery

> Identify statistically significant leukemia biomarkers using differential expression analysis.

<details>
<summary>📌 <b>See full feature list</b></summary>

<br>

**Data Input:**
- 🌐 Fetch directly from **NCBI GEO Database** by entering an accession number (e.g. `GSE1432`)
- 📁 Upload your own gene expression CSV file

**Analysis:**
- Define two sample groups (e.g. ALL vs AML) manually
- Runs **independent t-test** per gene across both groups
- Calculates **Log2 Fold Change**, **p-value**, and significance thresholds (`|log2FC| > 1` and `p < 0.05`)
- Classifies genes as **Upregulated**, **Downregulated**, or **Not Significant**

**Visualizations:**
- 🌋 **Volcano Plot** — interactive scatter with color-coded gene directions and top-10 gene labels
- 🧬 **Top 20 Significant Genes Table** with fold change and p-values

**Pathway Enrichment:**
- 🔗 Maps significant genes to **KEGG biological pathways** via **Enrichr** (gseapy)
- Displays top 10 enriched pathways as an interactive bar chart

**Downloads:**
- ⬇️ Full DE results (CSV)
- ⬇️ Significant genes only (CSV)

</details>

---

### 📊 Module 2 — Gene Expression Visualization

> Explore expression patterns, cluster patients, and compare leukemia subtypes visually.

<details>
<summary>📌 <b>See full feature list</b></summary>

<br>

**Data Input:**
- 📁 Upload a gene expression CSV
- ♻️ Or reuse data already loaded in the Biomarker module (shared session state)

**Visualizations:**
- 🔥 **Interactive Heatmap** — adjustable top-N variable genes × samples (RdBu color scale)
- 🔍 **Gene Search** — search any gene and view its expression bar chart across all samples
- Per-gene stats: Mean, Std Dev, Min, Max

**Dimensionality Reduction + Clustering:**
- 🔵 **PCA** — with explained variance % on axes
- 🌀 **t-SNE** — with automatic PCA pre-reduction for speed
- 🤖 **K-Means clustering** — adjustable number of clusters (2–6)
- Interactive scatter plot — each dot = one patient

**Group Comparison:**
- Define ALL vs AML sample groups
- 📊 Grouped bar chart of **top 20 variable genes** comparing mean expression across both groups

**Downloads:**
- ⬇️ Cleaned expression data (CSV)

</details>

---

### 📈 Module 3 — Survival Prediction

> Predict leukemia subtype (ALL vs AML) using ML and visualize patient survival.

<details>
<summary>📌 <b>See full feature list</b></summary>

<br>

**Data Input:**
- Upload gene expression CSV (train data)
- Upload labels CSV (`actual.csv` with patient IDs and cancer type)
- Auto-merges both datasets by patient ID

**Kaplan-Meier Survival Analysis:**
- 📉 Plots survival probability over time for ALL vs AML
- Uses real `time` + `event` columns if available in labels file
- Falls back to **simulated KM curves** for demonstration if survival columns are absent

**ML Model Training:**
- Choose from 3 models:
  - 🌲 **Random Forest**
  - 📐 **Logistic Regression**
  - ⚡ **Support Vector Machine (SVM)**
- Adjustable test/train split (10–40%)
- Optional **5-Fold Cross Validation** for more reliable accuracy estimates

**Results & Evaluation:**
- 🔢 Interactive **Confusion Matrix** (heatmap)
- 📈 **Cross-Validation accuracy bar chart** with mean line
- 🧬 **Top 15 Feature Importances** (Random Forest only)
- 📋 Full **Classification Report** (precision, recall, F1-score)

**Individual Patient Prediction:**
- 🩺 Upload a single patient's gene expression file
- Instantly predicts ALL or AML with **model confidence %**

**Downloads:**
- ⬇️ Classification report (CSV)
- ⬇️ Trained model as `.pkl` file

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
| `Scikit-learn` | ML models, PCA, t-SNE, K-Means, evaluation |
| `Lifelines` | Kaplan-Meier survival analysis |
| `GEOparse` | Fetch datasets from NCBI GEO |
| `gseapy` | Pathway enrichment via Enrichr (KEGG) |
| `SciPy` | t-test for differential expression |
| `Plotly` | Interactive charts & visualizations |
| `Pandas / NumPy` | Data processing |
| `Matplotlib / Seaborn` | Static plots |

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