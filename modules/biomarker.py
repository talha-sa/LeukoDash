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

        # Get numeric columns only, remove call columns
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df[[col for col in numeric_df.columns if "call" not in str(col)]]

        # Top variable genes
        variances = numeric_df.var().sort_values(ascending=False).head(10)
        gene_names = df["Gene Description"].iloc[variances.index.astype(int)].values if "Gene Description" in df.columns else variances.index

        st.subheader("🧬 Top 10 Biomarker Genes (Highest Variance)")

        # Color palette
        colors = sns.color_palette("Reds_r", 10)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(10), variances.values[::-1], color=colors[::-1], edgecolor="white")
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"Gene {i+1}" for i in range(10)][::-1], fontsize=10)
        ax.set_xlabel("Variance Score", fontsize=12)
        ax.set_title("Top 10 High-Variance Genes (Potential Biomarkers)", fontsize=14, fontweight="bold")

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, variances.values[::-1])):
            ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                   f'{val:,.0f}', va='center', fontsize=9)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # Pie chart of top 5 genes
        st.subheader("📊 Top 5 Biomarkers — Variance Distribution")
        top5 = variances.head(5)
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        colors2 = ["#e74c3c", "#c0392b", "#e67e22", "#f39c12", "#2980b9"]
        wedges, texts, autotexts = ax2.pie(
            top5.values,
            labels=[f"Gene {i+1}" for i in range(5)],
            autopct="%1.1f%%",
            colors=colors2,
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2)
        )
        for text in autotexts:
            text.set_fontsize(11)
            text.set_fontweight("bold")
        ax2.set_title("Variance Share of Top 5 Biomarkers", fontsize=13, fontweight="bold")
        st.pyplot(fig2)

        # Download biomarkers
        st.subheader("📥 Download Biomarker Results")
        biomarker_df = pd.DataFrame({
            "Rank": range(1, 11),
            "Gene Index": variances.index,
            "Variance Score": variances.values
        })
        csv = biomarker_df.to_csv(index=False)
        st.download_button(
            label="Download Top 10 Biomarkers as CSV",
            data=csv,
            file_name="leukodash_biomarkers.csv",
            mime="text/csv"
        )

    else:
        st.info("Please upload a CSV file to get started.")
