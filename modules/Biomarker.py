import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.title("🔬 Biomarker Discovery")
    st.markdown("Upload gene expression data to identify key leukemia biomarkers.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        st.subheader("Basic Statistics")
        st.write(df.describe())

        st.subheader("Top Variable Genes (Biomarkers)")
        numeric_df = df.select_dtypes(include=[np.number])
        variances = numeric_df.var().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=variances.index, y=variances.values, ax=ax)
        ax.set_title("Top 10 High-Variance Genes")
        ax.set_xlabel("Gene")
        ax.set_ylabel("Variance")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to get started.")