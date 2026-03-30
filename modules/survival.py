import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lifelines import KaplanMeierFitter
import time
import pickle
import io

# ─────────────────────────────────────────────
# MAIN SHOW FUNCTION
# ─────────────────────────────────────────────
def show():
    st.title("📈 Survival Prediction")
    st.markdown("Predict leukemia type (ALL vs AML) and analyze patient survival using machine learning.")

    st.markdown("---")

    # ════════════════════════════════════════
    # SECTION 1: Data Upload
    # ════════════════════════════════════════
    st.subheader("📂 Step 1: Upload Data")

    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("Upload Gene Expression CSV (train data)", type=["csv"], key="train")
    with col2:
        label_file = st.file_uploader("Upload Labels CSV (actual.csv)", type=["csv"], key="labels")

    if train_file and label_file:
        train_df = pd.read_csv(train_file)
        actual_df = pd.read_csv(label_file)

        # Clean train data
        train_df = train_df.drop(columns=["Gene Description", "Gene Accession Number"], errors="ignore")
        train_df = train_df[[col for col in train_df.columns if "call" not in str(col)]]
        train_T = train_df.T
        train_T.columns = [f"gene_{i}" for i in range(train_T.shape[1])]
        train_T.index = train_T.index.astype(str)

        # Clean labels
        actual_df.columns = actual_df.columns.str.strip().str.lower()

        if "patient" not in actual_df.columns:
            st.error(f"Could not find 'patient' column. Found: {list(actual_df.columns)}")
            return
        if "cancer" not in actual_df.columns:
            st.error(f"Could not find 'cancer' column. Found: {list(actual_df.columns)}")
            return

        actual_df["patient"] = actual_df["patient"].astype(str)
        actual_df = actual_df.set_index("patient")

        # Merge
        merged = train_T.join(actual_df, how="inner")

        if merged.empty:
            st.error("Could not merge datasets. Check patient IDs match between files.")
            return

        # Update session state
        st.session_state.total_patients = merged.shape[0]
        st.session_state.gene_features = f"{merged.shape[1]-1:,}"
        st.session_state.data_loaded = True

        st.success(f"✅ Datasets merged! {merged.shape[0]} patients × {merged.shape[1]-1} genes")

        st.markdown("---")

        # ════════════════════════════════════════
        # SECTION 2: Data Overview
        # ════════════════════════════════════════
        st.subheader("📊 Step 2: Dataset Overview")

        counts = merged["cancer"].value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", merged.shape[0])
        col2.metric("ALL Patients", int(counts.get("ALL", 0)))
        col3.metric("AML Patients", int(counts.get("AML", 0)))

        # Interactive cancer type bar chart
        fig_dist = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map={"ALL": "#e74c3c", "AML": "#2980b9"},
            title="ALL vs AML Patient Distribution",
            labels={"x": "Cancer Type", "y": "Number of Patients"},
            template="plotly_white",
            height=350
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")

        # ════════════════════════════════════════
        # SECTION 3: Kaplan-Meier Survival Curves
        # ════════════════════════════════════════
        st.subheader("📉 Step 3: Kaplan-Meier Survival Analysis")
        st.markdown("Shows survival probability over time for ALL vs AML patients.")

        has_survival = "time" in actual_df.columns and "event" in actual_df.columns

        if has_survival:
            kmf = KaplanMeierFitter()
            fig_km = go.Figure()

            for cancer_type, color in zip(["ALL", "AML"], ["#e74c3c", "#2980b9"]):
                mask = merged["cancer"] == cancer_type
                subset = merged[mask]
                if len(subset) > 0:
                    kmf.fit(
                        durations=subset["time"],
                        event_observed=subset["event"],
                        label=cancer_type
                    )
                    fig_km.add_trace(go.Scatter(
                        x=kmf.timeline,
                        y=kmf.survival_function_[cancer_type],
                        mode="lines",
                        name=cancer_type,
                        line=dict(color=color, width=2.5)
                    ))

            fig_km.update_layout(
                title="Kaplan-Meier Survival Curves: ALL vs AML",
                xaxis_title="Time (Days)",
                yaxis_title="Survival Probability",
                template="plotly_white",
                height=400,
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_km, use_container_width=True)

        else:
            st.info("💡 Your labels file doesn't have 'time' and 'event' columns for survival analysis. Showing simulated Kaplan-Meier curves based on cancer type for demonstration.")

            # Simulate KM curves for demonstration
            np.random.seed(42)
            time_all = np.random.exponential(scale=800, size=int(counts.get("ALL", 20)))
            time_aml = np.random.exponential(scale=400, size=int(counts.get("AML", 20)))

            kmf = KaplanMeierFitter()
            fig_km = go.Figure()

            for times, cancer_type, color in zip(
                [time_all, time_aml], ["ALL", "AML"], ["#e74c3c", "#2980b9"]
            ):
                events = np.ones(len(times))
                kmf.fit(durations=times, event_observed=events, label=cancer_type)
                fig_km.add_trace(go.Scatter(
                    x=kmf.timeline,
                    y=kmf.survival_function_[cancer_type],
                    mode="lines",
                    name=cancer_type,
                    line=dict(color=color, width=2.5)
                ))

            fig_km.update_layout(
                title="Kaplan-Meier Survival Curves (Simulated): ALL vs AML",
                xaxis_title="Time (Days)",
                yaxis_title="Survival Probability",
                template="plotly_white",
                height=400,
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_km, use_container_width=True)
            st.caption("Note: Simulated curves shown for visualization. Add 'time' and 'event' columns to your labels file for real survival analysis.")

        st.markdown("---")

        # ════════════════════════════════════════
        # SECTION 4: ML Model Training
        # ════════════════════════════════════════
        st.subheader("🤖 Step 4: Train Prediction Model")

        model_choice = st.selectbox("Choose ML Model:", [
            "Random Forest",
            "Logistic Regression",
            "Support Vector Machine (SVM)"
        ])

        test_size = st.slider("Test set size (%)", 10, 40, 20)
        run_cv = st.checkbox("Run Cross-Validation (more reliable accuracy)", value=True)

        target_col = "cancer"
        X = merged.drop(columns=[target_col])

        # Remove any non-numeric columns
        X = X.select_dtypes(include=[np.number])
        y = merged[target_col]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        if st.button("🚀 Train & Evaluate Model"):
            with st.spinner(f"Training {model_choice}... ⏳"):

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size/100, random_state=42
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, random_state=42)
                else:
                    model = SVC(kernel="linear", random_state=42, probability=True)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
st.session_state["model_accuracy"] = f"{acc*100:.1f}%"

                # Cross validation
                cv_scores = None
                if run_cv:
                    cv_scores = cross_val_score(model, scaler.fit_transform(X), y_encoded, cv=5)

                # Save model to session state
                st.session_state["trained_model"] = model
                st.session_state["trained_scaler"] = scaler
                st.session_state["label_encoder"] = le
                st.session_state["feature_cols"] = list(X.columns)
                st.session_state["model_choice"] = model_choice
                st.session_state["X_test"] = X_test_scaled
                st.session_state["y_test"] = y_test
                st.session_state["y_pred"] = y_pred
                st.session_state["acc"] = acc
                st.session_state["cv_scores"] = cv_scores

            st.success("✅ Model trained successfully!")
            st.balloons()

        # ── Show Results ──
        if "trained_model" in st.session_state:
            acc = st.session_state["acc"]
            y_test = st.session_state["y_test"]
            y_pred = st.session_state["y_pred"]
            cv_scores = st.session_state["cv_scores"]
            model = st.session_state["trained_model"]
            le = st.session_state["label_encoder"]
            model_choice = st.session_state["model_choice"]

            st.markdown("---")
            st.subheader("📊 Model Performance")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test Accuracy", f"{acc*100:.1f}%")
            m2.metric("Patients Tested", len(y_test))
            m3.metric("Features Used", X.shape[1])
            if cv_scores is not None:
                m4.metric("CV Accuracy (5-fold)", f"{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

            # Interactive Confusion Matrix
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Reds",
                x=list(le.classes_),
                y=list(le.classes_),
                labels={"x": "Predicted", "y": "Actual"},
                title="Confusion Matrix",
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Cross-validation chart
            if cv_scores is not None:
                st.subheader("📈 Cross-Validation Accuracy (5 Folds)")
                fig_cv = px.bar(
                    x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                    y=cv_scores * 100,
                    color=cv_scores * 100,
                    color_continuous_scale="Reds",
                    title="5-Fold Cross Validation Accuracy",
                    labels={"x": "Fold", "y": "Accuracy (%)"},
                    template="plotly_white",
                    height=350
                )
                fig_cv.add_hline(
                    y=cv_scores.mean()*100,
                    line_dash="dash",
                    annotation_text=f"Mean: {cv_scores.mean()*100:.1f}%"
                )
                fig_cv.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_cv, use_container_width=True)

            # Feature Importance (Random Forest)
            if model_choice == "Random Forest":
                st.subheader("🧬 Top 15 Most Important Genes")
                st.markdown("These genes had the most influence on the model's predictions.")
                importances = pd.Series(model.feature_importances_, index=X.columns)
                top15 = importances.nlargest(15)

                fig_imp = px.bar(
                    x=top15.values,
                    y=top15.index,
                    orientation="h",
                    color=top15.values,
                    color_continuous_scale="Reds",
                    title="Top 15 Gene Feature Importances",
                    labels={"x": "Importance Score", "y": "Gene"},
                    template="plotly_white",
                    height=450
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                st.plotly_chart(fig_imp, use_container_width=True)

            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)

            st.markdown("---")

            # ════════════════════════════════════════
            # SECTION 5: Individual Patient Prediction
            # ════════════════════════════════════════
            st.subheader("🩺 Step 5: Predict Individual Patient")
            st.markdown("Upload a single patient's gene expression data to get a prediction.")

            patient_file = st.file_uploader(
                "Upload single patient CSV (same format as training data)",
                type=["csv"],
                key="patient_pred"
            )

            if patient_file:
                patient_df = pd.read_csv(patient_file)
                patient_df = patient_df.drop(columns=["Gene Description", "Gene Accession Number"], errors="ignore")
                patient_df = patient_df[[col for col in patient_df.columns if "call" not in str(col)]]

                # Transpose to get 1 row
                patient_T = patient_df.T
                patient_T.columns = [f"gene_{i}" for i in range(patient_T.shape[1])]

                # Align features
                feature_cols = st.session_state["feature_cols"]
                patient_aligned = patient_T.reindex(columns=feature_cols, fill_value=0)

                scaler = st.session_state["trained_scaler"]
                patient_scaled = scaler.transform(patient_aligned)

                pred = model.predict(patient_scaled)
                pred_label = le.inverse_transform(pred)[0]

                # Get probability if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(patient_scaled)[0]
                    confidence = max(proba) * 100
                else:
                    confidence = None

                # Show result
                if pred_label == "ALL":
                    st.error(f"🔴 Predicted Type: **{pred_label}** (Acute Lymphoblastic Leukemia)")
                else:
                    st.warning(f"🟡 Predicted Type: **{pred_label}** (Acute Myeloid Leukemia)")

                if confidence:
                    st.metric("Model Confidence", f"{confidence:.1f}%")

            st.markdown("---")

            # ════════════════════════════════════════
            # SECTION 6: Download Results
            # ════════════════════════════════════════
            st.subheader("📥 Download Results")

            col1, col2 = st.columns(2)
            with col1:
                csv = report_df.to_csv()
                st.download_button(
                    "⬇️ Download Classification Report (CSV)",
                    data=csv,
                    file_name="leukodash_classification_report.csv",
                    mime="text/csv"
                )
            with col2:
                # Save model as pickle
                model_bytes = pickle.dumps(model)
                st.download_button(
                    "⬇️ Download Trained Model (.pkl)",
                    data=model_bytes,
                    file_name="leukodash_model.pkl",
                    mime="application/octet-stream"
                )

    else:
        st.info("👆 Please upload both CSV files above to get started.")