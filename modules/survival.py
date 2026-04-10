import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from lifelines import KaplanMeierFitter
import gc
import io
import requests

# ─────────────────────────────────────────────
# REQ 1+2: Cached data loader
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_golub_survival():
    local_paths = [
        "data/golub_data.csv",
        "./data/golub_data.csv",
        "golub_data.csv"
    ]
    for path in local_paths:
        try:
            df = pd.read_csv(path, index_col=0)
            df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
            return df
        except FileNotFoundError:
            continue
        except Exception:
            continue

    urls = [
        "https://raw.githubusercontent.com/talha-sa/LeukoDash/main/data/golub_data.csv",
        "https://raw.githubusercontent.com/talha-sa/LeukoDash/master/data/golub_data.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), index_col=0)
                df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
                return df
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────
# REQ 1: Cached feature preparation
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def prepare_features(df_json, n_top_genes=500):
    df = pd.read_json(io.StringIO(df_json)).astype("float32")
    df = df.fillna(df.mean())
    top_genes = df.var(axis=1).nlargest(n_top_genes).index
    df = df.loc[top_genes]
    X = df.T.values.astype("float32")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gc.collect()
    return X_scaled, list(df.index)


# ─────────────────────────────────────────────
# REQ 1 (cache_resource): ML Models
# ─────────────────────────────────────────────
@st.cache_resource
def get_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }


# ─────────────────────────────────────────────
# REQ 1+5: Cached cross-validation
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_cross_validation(X_json, y_list, model_name, cv_folds=5):
    X = np.array(pd.read_json(io.StringIO(X_json)))
    y = np.array(y_list)
    models = get_models()
    model = models[model_name]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    gc.collect()
    return scores


# ─────────────────────────────────────────────
# MAIN SHOW FUNCTION
# ─────────────────────────────────────────────
def show():
    st.title("📈 Survival Prediction")
    st.markdown("Predict leukemia type (ALL vs AML) using machine learning models.")

    with st.expander("📖 How to use this module"):
        st.markdown("""
        1. **Load Data** — Use Golub dataset, GEO loaded data, or upload CSV
        2. **Assign Labels** — Tell the app which samples are ALL and which are AML
        3. **Configure Model** — Choose ML algorithm and settings
        4. **Train & Evaluate** — View accuracy, confusion matrix, and gene importances
        5. **Predict** — Run prediction on individual samples

        **Supported models:** Random Forest, Logistic Regression, SVM
        All models use stratified k-fold cross-validation to prevent overfitting.
        """)

    st.markdown("---")

    # ── Load Data ──
    st.subheader("📂 Step 1: Load Data")

    data_source = st.radio(
        "Data source:",
        ["📦 Golub Dataset (default)", "🔄 Use GEO Loaded Data", "📁 Upload CSV"],
        horizontal=True
    )

    df = None

    if "Golub" in data_source:
        with st.spinner("Loading..."):
            df = load_golub_survival()
        if df is not None:
            st.success(f"✅ Golub dataset: {df.shape[0]} genes × {df.shape[1]} samples")
        else:
            st.error("❌ Could not load Golub dataset. Please upload CSV manually.")

    elif "GEO" in data_source:
        if "geo_df" in st.session_state and st.session_state["geo_df"] is not None:
            df = st.session_state["geo_df"]
            st.success(f"✅ GEO dataset: {df.shape[0]} genes × {df.shape[1]} samples")
        else:
            st.warning("⚠️ No GEO dataset found. Go to Biomarker Discovery first.")

    else:
        uploaded = st.file_uploader("Upload CSV (rows=genes, columns=samples)", type=["csv"])
        if uploaded:
            with st.spinner("Reading..."):
                progress = st.progress(0)
                try:
                    chunks = []
                    for i, chunk in enumerate(pd.read_csv(uploaded, index_col=0, chunksize=2000)):
                        chunk = chunk.apply(pd.to_numeric, errors="coerce").astype("float32")
                        chunks.append(chunk)
                        progress.progress(min((i + 1) * 20, 90))
                    df = pd.concat(chunks)
                    del chunks
                    gc.collect()
                    progress.progress(100)
                    progress.empty()
                    st.success(f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples")
                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Could not read file: {str(e)}")

    if df is None or df.empty:
        st.info("👆 Load a dataset to continue.")
        return

    st.markdown("---")

    # ── Labels ──
    st.subheader("🏷️ Step 2: Assign Sample Labels")

    n_samples = df.shape[1]
    half = n_samples // 2
    default_labels = ["ALL"] * half + ["AML"] * (n_samples - half)

    with st.expander("⚙️ Label Configuration"):
        st.markdown("Enter one label per line (ALL or AML), matching the order of your samples.")
        labels_input = st.text_area(
            "Labels:",
            value="\n".join(default_labels),
            height=120
        )

    labels = [l.strip() for l in labels_input.strip().split("\n") if l.strip()]

    # FIX 3: Missing label handler
    label_series = pd.Series(labels)
    missing_mask = label_series.isin(["", "nan", "NaN", "None", "N/A"]) | label_series.isna()
    missing_count = missing_mask.sum()
    if missing_count > 0:
        st.warning(f"⚠️ {missing_count} sample(s) removed due to missing labels.")
        labels = label_series[~missing_mask].tolist()

    if len(labels) != n_samples:
        st.warning(f"⚠️ Labels ({len(labels)}) must match samples ({n_samples}). Using defaults.")
        labels = default_labels

    label_counts = pd.Series(labels).value_counts()
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.metric("Group 1 (ALL)", label_counts.get("ALL", 0))
    with col_l2:
        st.metric("Group 2 (AML)", label_counts.get("AML", 0))

    st.markdown("---")

    # ── Model Settings ──
    st.subheader("⚙️ Step 3: Model Settings")

    with st.expander("ℹ️ Model Guide"):
        st.markdown("""
        - **Random Forest** — Best for small datasets, gives gene importances. Recommended.
        - **Logistic Regression** — Fast and interpretable. Good baseline model.
        - **SVM** — Powerful for high-dimensional data like gene expression.

        **Cross-validation folds:** Higher = more reliable accuracy but slower training.
        **Top genes:** Lower = faster training. 500 is a good balance.
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox("ML Model", ["Random Forest", "Logistic Regression", "SVM"])
    with col2:
        cv_folds = st.slider("CV Folds", 3, 10, 5)
    with col3:
        n_top_genes = st.slider("Top genes", 100, 1000, 500, 100,
                                help="Uses top N most variable genes for training")

    # ── Train ──
    if st.button("🚀 Train & Evaluate Model"):
        if len(set(labels)) < 2:
            st.warning("⚠️ Need at least 2 different labels to train.")
        else:
            progress = st.progress(0)
            status = st.empty()
            status.text("📊 Preparing features...")
            progress.progress(10)

            with st.spinner("Training... ⏳"):
                try:
                    X_scaled, gene_names = prepare_features(df.to_json(), n_top_genes)
                    progress.progress(30)

                    le = LabelEncoder()
                    y = le.fit_transform(labels[:n_samples])

                    status.text(f"🔁 Running {cv_folds}-fold cross-validation...")
                    progress.progress(40)

                    X_df = pd.DataFrame(X_scaled)
                    scores = run_cross_validation(
                        X_df.to_json(),
                        y.tolist(),
                        model_name,
                        cv_folds
                    )
                    progress.progress(70)

                    status.text("🎯 Fitting final model...")
                    models = get_models()
                    model = models[model_name]
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)

                    progress.progress(90)
                    gc.collect()

                    acc = f"{scores.mean()*100:.1f}%"
                    st.session_state["model_accuracy"] = acc
                    st.session_state["trained_model"] = model
                    st.session_state["label_encoder"] = le
                    st.session_state["gene_names"] = gene_names
                    st.session_state["X_scaled"] = X_scaled
                    st.session_state["y"] = y
                    st.session_state["y_pred"] = y_pred
                    st.session_state["cv_scores"] = scores
                    st.session_state["survival_done"] = True
                    st.session_state["model_name_used"] = model_name
                    st.session_state["cv_folds_used"] = cv_folds
                    st.session_state["n_top_genes_used"] = n_top_genes
                    st.session_state["le_used"] = le

                    progress.progress(100)
                    status.empty()
                    progress.empty()
                    st.success(f"✅ Done! CV Accuracy: **{acc}** ± {scores.std()*100:.1f}%")

                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"❌ Training failed: {str(e)}")

    # ── Show Results ──
    if "cv_scores" in st.session_state:
        scores = st.session_state["cv_scores"]
        y = st.session_state["y"]
        y_pred = st.session_state["y_pred"]
        le = st.session_state["label_encoder"]
        X_scaled = st.session_state["X_scaled"]
        model_name = st.session_state.get("model_name_used", "Model")
        cv_folds = st.session_state.get("cv_folds_used", 5)
        n_top_genes = st.session_state.get("n_top_genes_used", 500)

        st.markdown("---")
        st.subheader("📊 Step 4: Model Performance")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean CV Accuracy", f"{scores.mean()*100:.1f}%")
        m2.metric("Std Deviation", f"±{scores.std()*100:.1f}%")
        m3.metric("Best Fold", f"{scores.max()*100:.1f}%")
        m4.metric("Worst Fold", f"{scores.min()*100:.1f}%")

        # CV Scores Chart
        fig_cv = px.bar(
            x=[f"Fold {i+1}" for i in range(len(scores))],
            y=scores * 100,
            title="Cross-Validation Accuracy per Fold",
            labels={"x": "Fold", "y": "Accuracy (%)"},
            color=scores * 100,
            color_continuous_scale="RdYlGn",
            template="plotly_white"
        )
        fig_cv.add_hline(
            y=scores.mean() * 100,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Mean: {scores.mean()*100:.1f}%"
        )
        fig_cv.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cv, use_container_width=True)

        # Confusion Matrix
        st.subheader("🔲 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        class_names = le.classes_

        fig_cm = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            color_continuous_scale="Reds",
            labels=dict(x="Predicted", y="Actual"),
            title="Confusion Matrix — Predicted vs Actual",
            text_auto=True
        )
        fig_cm.update_layout(template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification Report
        with st.expander("📋 Full Classification Report"):
            report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)

        # Feature Importance (Random Forest only)
        if model_name == "Random Forest" and "trained_model" in st.session_state:
            st.subheader("🧬 Top 20 Important Genes")

            with st.expander("ℹ️ What is Gene Importance?"):
                st.markdown("""
                Random Forest measures how much each gene contributes to correct classification.
                Higher importance = that gene is more useful for distinguishing ALL from AML.
                These are your most powerful biomarkers for this dataset.
                """)

            model = st.session_state["trained_model"]
            gene_names = st.session_state["gene_names"]
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            top_genes = [gene_names[i] for i in top_idx]
            top_scores = importances[top_idx]

            fig_imp = px.bar(
                x=top_scores,
                y=top_genes,
                orientation="h",
                title="Top 20 Gene Importances (Random Forest)",
                labels={"x": "Importance Score", "y": "Gene"},
                color=top_scores,
                color_continuous_scale="Reds",
                template="plotly_white"
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            gc.collect()

        # ── Individual Prediction ──
        st.markdown("---")
        st.subheader("🔍 Step 5: Individual Sample Prediction")

        with st.expander("ℹ️ How this works"):
            st.markdown("""
            Select any sample index from your dataset.
            The trained model will predict whether it's ALL or AML
            and show a confidence score for each class.
            """)

        if X_scaled is not None:
            sample_idx = st.slider("Select sample index", 0, len(X_scaled) - 1, 0)

            if st.button("🔮 Predict This Sample"):
                try:
                    model = st.session_state["trained_model"]
                    sample = X_scaled[sample_idx].reshape(1, -1)
                    pred = model.predict(sample)[0]
                    prob = model.predict_proba(sample)[0] if hasattr(model, "predict_proba") else None

                    predicted_label = le.inverse_transform([pred])[0]
                    actual_label = le.inverse_transform([y[sample_idx]])[0]
                    is_correct = predicted_label == actual_label

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted", predicted_label)
                    with col2:
                        st.metric("Actual", actual_label)
                    with col3:
                        st.metric("Result", "✅ Correct" if is_correct else "❌ Incorrect")

                    if prob is not None:
                        prob_df = pd.DataFrame({
                            "Class": le.classes_,
                            "Probability": prob
                        })
                        fig_prob = px.bar(
                            prob_df,
                            x="Class",
                            y="Probability",
                            color="Probability",
                            color_continuous_scale="RdYlGn",
                            title=f"Prediction Confidence — Sample {sample_idx}",
                            template="plotly_white"
                        )
                        fig_prob.update_layout(
                            coloraxis_showscale=False,
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

        # ─────────────────────────────────────────────
        # ✅ NEW: Biological Insight Box
        # ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🧠 Model Interpretation")

        mean_acc = scores.mean() * 100

        if mean_acc >= 80:
            interpretation_color = "#27ae60"
            performance_label = "Excellent"
            interpretation_text = "The model reliably distinguishes ALL from AML. Results are suitable for research reporting."
        elif mean_acc >= 65:
            interpretation_color = "#e67e22"
            performance_label = "Moderate"
            interpretation_text = "Model shows promise but needs more data or feature tuning for clinical confidence."
        else:
            interpretation_color = "#e74c3c"
            performance_label = "Low"
            interpretation_text = "Consider adjusting sample labels, increasing top genes, or trying a different model."

        st.markdown(f"""
        <div style='background:white; border-left:5px solid {interpretation_color};
                    padding:20px; border-radius:10px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-top:10px;'>
            <h4 style='color:{interpretation_color}; margin:0 0 12px 0;'>
                📌 Performance Rating: {performance_label} — {mean_acc:.1f}% Mean Accuracy
            </h4>
            <p style='color:#444; font-size:15px; margin:0 0 16px 0;'>
                {interpretation_text}
            </p>
            <table style='width:100%; font-size:14px; border-collapse:collapse;'>
                <tr style='border-bottom:1px solid #eee;'>
                    <td style='padding:6px 0;'><b>Model Used</b></td>
                    <td style='padding:6px 0;'>{model_name}</td>
                    <td style='padding:6px 0;'><b>CV Folds</b></td>
                    <td style='padding:6px 0;'>{cv_folds}</td>
                </tr>
                <tr style='border-bottom:1px solid #eee;'>
                    <td style='padding:6px 0;'><b>Mean Accuracy</b></td>
                    <td style='padding:6px 0;'>{mean_acc:.1f}%</td>
                    <td style='padding:6px 0;'><b>Std Deviation</b></td>
                    <td style='padding:6px 0;'>±{scores.std()*100:.1f}%</td>
                </tr>
                <tr style='border-bottom:1px solid #eee;'>
                    <td style='padding:6px 0;'><b>Best Fold</b></td>
                    <td style='padding:6px 0;'>{scores.max()*100:.1f}%</td>
                    <td style='padding:6px 0;'><b>Worst Fold</b></td>
                    <td style='padding:6px 0;'>{scores.min()*100:.1f}%</td>
                </tr>
                <tr>
                    <td style='padding:6px 0;'><b>Genes Used</b></td>
                    <td style='padding:6px 0;'>Top {n_top_genes} variable genes</td>
                    <td style='padding:6px 0;'><b>Classes</b></td>
                    <td style='padding:6px 0;'>{", ".join(le.classes_)}</td>
                </tr>
            </table>
            <br>
            <p style='margin:0; font-size:12px; color:#888;'>
                ℹ️ Cross-validation splits data into {cv_folds} folds — each fold is tested once
                while the rest train the model. This prevents overfitting and gives a reliable
                accuracy estimate even on small leukemia datasets.
                A low standard deviation (±{scores.std()*100:.1f}%) indicates the model is consistent across folds.
            </p>
        </div>
        """, unsafe_allow_html=True)

        gc.collect()

        # ── Download ──
        st.markdown("---")
        st.subheader("📥 Download Results")

        col1, col2 = st.columns(2)
        with col1:
            report_out = classification_report(y, y_pred, target_names=class_names, output_dict=True)
            report_csv = pd.DataFrame(report_out).transpose().round(3).to_csv()
            st.download_button(
                "⬇️ Classification Report (CSV)",
                data=report_csv,
                file_name="leukodash_classification_report.csv",
                mime="text/csv"
            )
        with col2:
            if model_name == "Random Forest" and "trained_model" in st.session_state:
                imp_df = pd.DataFrame({
                    "Gene": top_genes,
                    "Importance": top_scores
                })
                st.download_button(
                    "⬇️ Gene Importances (CSV)",
                    data=imp_df.to_csv(index=False),
                    file_name="leukodash_gene_importances.csv",
                    mime="text/csv"
                )

    elif not st.session_state.get("survival_done"):
        st.info("👆 Configure your model above and click Train to see results.")