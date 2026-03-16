import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.title("📊 Gene Expression Visualization")
    st.markdown("Visualize gene expression patterns across leukemia samples.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Save to session state
        st.session_state.total_patients = df.select_dtypes(include=[np.number]).shape[1]
        st.session_state.gene_features = f"{df.select_dtypes(include=[np.number]).shape[0]:,}"
        st.session_state.data_loaded = True

        st.subheader("Preview of Data")
        st.dataframe(df.head())

        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df[[col for col in numeric_df.columns if "call" not in str(col)]]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Genes", numeric_df.shape[0])
        with col2:
            st.metric("Total Samples", numeric_df.shape[1])
        with col3:
            st.metric("Missing Values", numeric_df.isnull().sum().sum())

        st.subheader("🔥 Gene Expression Heatmap")
        st.markdown("Showing first 20 genes × 20 samples")
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(
            numeric_df.iloc[:20, :20],
            cmap="coolwarm",
            ax=ax,
            linewidths=0.3,
            linecolor="white"
        )
        ax.set_title("Gene Expression Heatmap (Top 20 Genes × 20 Samples)",
                    fontsize=13, fontweight="bold")
        ax.set_xlabel("Patient Samples")
        ax.set_ylabel("Gene Index")
        st.pyplot(fig)

        st.subheader("📈 Gene Expression Distribution")
        gene_index = st.slider("Select Gene Index", 0, min(99, numeric_df.shape[0]-1), 0)
        gene_data = numeric_df.iloc[gene_index]

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(gene_data, bins=20, color="#e74c3c", edgecolor="white", alpha=0.85)
        ax2.set_title(f"Expression Distribution — Gene {gene_index}",
                     fontsize=13, fontweight="bold")
        ax2.set_xlabel("Expression Level")
        ax2.set_ylabel("Frequency")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)

        st.subheader("📦 Sample Expression Boxplot")
        st.markdown("Showing first 20 samples")
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        numeric_df.iloc[:, :20].T.boxplot(ax=ax3, patch_artist=True)
        ax3.set_title("Expression Distribution Across Samples",
                     fontsize=13, fontweight="bold")
        ax3.set_xlabel("Gene Index")
        ax3.set_ylabel("Expression Level")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        st.subheader("📥 Download Processed Data")
        csv = numeric_df.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Data as CSV",
            data=csv,
            file_name="leukodash_expression_data.csv",
            mime="text/csv"
        )

    else:
        st.info("Please upload a CSV file to get started.")