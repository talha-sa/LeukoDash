import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

        st.subheader("Labels Preview")
        st.dataframe(actual_df.head())

        # Transpose train data so rows=patients, columns=genes
        train_df = train_df.drop(columns=["Gene Description", "Gene Accession Number"], errors="ignore")
        # Remove "call" columns
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
            st.success(f"Model Accuracy: {acc * 100:.2f}%")

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred,
                        target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
    else:
        st.info("Please upload both files to get started.")