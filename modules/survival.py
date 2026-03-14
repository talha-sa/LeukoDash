import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def show():
    st.title("📈 Survival Prediction")
    st.markdown("Predict leukemia type (ALL vs AML) using gene expression data.")

    st.subheader("Step 1: Upload Gene Expression Data")
    train_file = st.file_uploader("Upload train CSV", type=["csv"], key="train")

    st.subheader("Step 2: Upload Labels File")
    label_file = st.file_uploader("Upload actual.csv", type=["csv"], key="labels")

    if train_file and label_file:
        train_df = pd.read_csv(train_file)
        actual_df = pd.read_csv(label_file)

        # Transpose train data
        train_df = train_df.drop(columns=["Gene Description", "Gene Accession Number"], errors="ignore")
        train_df = train_df[[col for col in train_df.columns if "call" not in str(col)]]
        train_T = train_df.T
        train_T.columns = [f"gene_{i}" for i in range(train_T.shape[1])]
        train_T.index = train_T.index.astype(str)

        # Prepare labels
        actual_df.columns = actual_df.columns.str.strip()
        actual_df["patient"] = actual_df["patient"].astype(str)
        actual_df = actual_df.set_index("patient")

        # Merge
        merged = train_T.join(actual_df, how="inner")

        if merged.empty:
            st.error("Could not merge datasets. Check patient IDs match.")
            return

        st.success(f"Merged dataset: {merged.shape[0]} patients, {merged.shape[1]-1} genes")

        # Show label distribution
        st.subheader("Cancer Type Distribution")
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        counts = merged["cancer"].value_counts()
        colors = ["#e74c3c", "#2980b9"]
        ax1.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
        ax1.set_title("ALL vs AML Patient Count", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Cancer Type")
        ax1.set_ylabel("Number of Patients")
        st.pyplot(fig1)

        target_col = "cancer"
        X = merged.drop(columns=[target_col])
        y = merged[target_col]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        if st.button("Run Prediction Model"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            # Accuracy metric
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"{acc * 100:.2f}%")
            with col2:
                st.metric("Patients Tested", len(y_test))

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                       xticklabels=le.classes_,
                       yticklabels=le.classes_, ax=ax2)
            ax2.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred,
                        target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

    else:
        st.info("Please upload both files to get started.")