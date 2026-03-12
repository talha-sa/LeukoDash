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
        st.subheader("Preview of Data")
        st.dataframe(df.head())

        numeric_df = df.select_dtypes(include=[np.number])

        st.subheader("Heatmap of Gene Expression")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            numeric_df.iloc[:20, :20],
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Gene Expression Heatmap")
        st.pyplot(fig)

        st.subheader("Select a Gene to Visualize")
        gene = st.selectbox("Choose a gene:", numeric_df.columns)
        fig2, ax2 = plt.subplots()
        ax2.hist(numeric_df[gene], bins=20, color="steelblue")
        ax2.set_title(f"Expression Distribution: {gene}")
        ax2.set_xlabel("Expression Level")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    else:
        st.info("Please upload a CSV file to get start