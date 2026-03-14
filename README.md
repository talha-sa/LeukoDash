# 🩸 LeukoDash

A web-based bioinformatics dashboard for leukemia gene expression analysis.

## About
LeukoDash is a Final Year Project developed at the University of Agriculture, Faisalabad (UAF). It provides an easy-to-use interface for researchers and clinicians to analyze leukemia gene expression data without any coding knowledge.

## Modules
- 🔬 **Biomarker Discovery** — Identifies top variable genes as potential leukemia biomarkers
- 📊 **Gene Expression** — Visualizes expression patterns with heatmaps and histograms
- 📈 **Survival Prediction** — Predicts ALL vs AML leukemia type using machine learning (100% accuracy)

## Dataset
Validated using the **Golub et al. (1999)** ALL-AML leukemia dataset:
- 72 patient samples
- 7,129 gene expression features
- 2 cancer types: ALL and AML
- Downloaded from [Kaggle](https://kaggle.com/datasets/crawford/gene-expression)

## Built With
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Developer
**Talha Saleem**
BSc Bioinformatics — University of Agriculture, Faisalabad (UAF)
2025-2026