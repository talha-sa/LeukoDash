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
    url = "https://raw.githubusercontent.com/talha-sa/LeukoDash/main/data/golub_data.csv"
    try:
        r = requests.get(url, timeout=30)
        df = pd.read_csv(io.StringIO(r.text), index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce").astype("float32")  # REQ 2
        return df
    except Exception:
        return None


# ─────────────────────────────────────────────
# REQ 1: Cached feature preparation
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def prepare_features(df_json, n_top_genes=500):
    """REQ 3+5: Select top variable genes early, vectorized"""
    df = pd.read_json(io.StringIO(df_json)).astype("float32")  # REQ 2
    df = df.fillna(df.mean())

    # REQ 5: Vectorized top gene selection
    top_genes = df.var(axis=1).nlargest(n_top_genes).index
    df = df.loc[top_genes]

    X = df.T.values.astype("float32")  # samples x genes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gc.collect()  # REQ 4
    return X_scaled, list(df.index)


# ─────────────────────────────────────────────
# REQ 1 (cache_resource): ML Model — loaded once, reused
# ─────────────────────────────────────────────
@st.cache_resource
def get_models():
    """REQ 1: cache_resource keeps models in memory without reloading"""
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
    """REQ 5: Vectorized sklearn cross-validation (no manual loops)"""
    X = np.array(pd.read_json(io.StringIO(X_json)))
    y = np.array(y_list)

    models = get_models()
    model = models[model_name]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    # REQ 5: cross_val_score is fully vectorized internally
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    gc.collect()  # REQ 4
    return scores


def show():
    st.title("📈 Survival Prediction")
    st.markdown("Predict leukemia type (ALL vs AML) using machine learning models.")
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
            st.error("Could not load Golub dataset.")

    elif "GEO" in data_source:
        if "geo_df" in st.session_state:
            df = st.session_state["geo_df"]
            st.success(f"✅ GEO dataset: {df.shape[0]} genes × {df.shape[1]} samples")
        else:
            st.warning("No GEO dataset found. Go to Biomarker Discovery first.")

    else:
        uploaded = st.file_uploader("Upload CSV (rows=genes, columns=samples)", type=["csv"])
        if uploaded:
            with st.spinner("Reading..."):
                progress = st.progress(0)
                chunks = []
                for i, chunk in enumerate(pd.read_csv(uploaded, index_col=0, chunksize=2000)):
                    chunk = chunk.apply(pd.to_numeric, errors="coerce").astype("float32")  # REQ 2+3
                    chunks.append(chunk)
                    progress.progress(min((i+1)*20, 90))
                df = pd.concat(chunks)
                del chunks
                gc.collect()  # REQ 4
                progress.progress(100)
                progress.empty()
            st.success(f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples")

    if df is None or df.empty:
        st.info("👆 Load a dataset to continue.")
        return

    st.markdown("---")

    # ── Labels ──
    st.subheader("🏷️ Step 2: Assign Sample Labels")
    n_samples = df.shape[1]
    half = n_samples // 2
    default_labels = ["ALL"] * half + ["AML"] * (n_samples - half)

    labels_input = st.text_area(
        "One label per line (ALL or AML), matching sample order:",
        value="\n".join(default_labels),
        height=120
    )
    labels = [l.strip() for l in labels_input.strip().split("\n") if l.strip()]

    if len(labels) != n_samples:
        st.warning(f"⚠️ Labels ({len(labels)}) must match samples ({n_samples}).")
        labels = default_labels

    label_counts = pd.Series(labels).value_counts()
    st.write("Label distribution:", label_counts.to_dict())

    st.markdown("---")

    # ── Model Settings ──
    st.subheader("⚙️ Step 3: Model Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox("Select ML Model", ["Random Forest", "Logistic Regression", "SVM"])
    with col2:
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
    with col3:
        n_top_genes = st.slider("Top genes to use", 100, 1000, 500, 100,
                                help="Lower = faster training")

    # ── Train Model ──
    if st.button("🚀 Train & Evaluate Model"):
        if len(set(labels)) < 2:
            st.warning("Need at least 2 different labels.")
        else:
            # REQ 6: Step-by-step progress
            progress = st.progress(0)
            status = st.empty()

            status.text("📊 Preparing features...")
            progress.progress(10)

            with st.spinner("Training... ⏳"):
                # REQ 1+3+5: Cached vectorized feature prep
                X_scaled, gene_names = prepare_features(df.to_json(), n_top_genes)
                progress.progress(30)

                le = LabelEncoder()
                y = le.fit_transform(labels[:n_samples])

                status.text(f"🔁 Running {cv_folds}-fold cross-validation...")
                progress.progress(40)

                # REQ 1+5: Cached CV - no manual loop
                X_df = pd.DataFrame(X_scaled)
                scores = run_cross_validation(
                    X_df.to_json(),
                    y.tolist(),
                    model_name,
                    cv_folds
                )
                progress.progress(70)

                status.text("🎯 Fitting final model...")
                models = get_models()  # REQ 1: cached resource
                model = models[model_name]
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)

                progress.progress(90)
                gc.collect()  # REQ 4

                # Save results
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

                progress.progress(100)
                status.empty()
                progress.empty()

            st.success(f"✅ Done! CV Accuracy: **{acc}** ± {scores.std()*100:.1f}%")

    # ── Show Results ──
    if "cv_scores" in st.session_state:
        scores = st.session_state["cv_scores"]
        y = st.session_state["y"]
        y_pred = st.session_state["y_pred"]
        le = st.session_state["label_encoder"]
        X_scaled = st.session_state["X_scaled"]

        st.markdown("---")
        st.subheader("📊 Model Performance")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean CV Accuracy", f"{scores.mean()*100:.1f}%")
        m2.metric("Std Deviation", f"±{scores.std()*100:.1f}%")
        m3.metric("Best Fold", f"{scores.max()*100:.1f}%")
        m4.metric("Worst Fold", f"{scores.min()*100:.1f}%")

        # CV scores chart
        fig_cv = px.bar(
            x=[f"Fold {i+1}" for i in range(len(scores))],
            y=scores * 100,
            title="Cross-Validation Accuracy per Fold",
            labels={"x": "Fold", "y": "Accuracy (%)"},
            color=scores * 100,
            color_continuous_scale="RdYlGn",
            template="plotly_white"
        )
        fig_cv.add_hline(y=scores.mean()*100, line_dash="dash",
                         line_color="blue", annotation_text="Mean")
        st.plotly_chart(fig_cv, use_container_width=True)

        # Confusion Matrix
        st.subheader("🔲 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        class_names = le.classes_

        fig_cm = px.imshow(
            cm,
            x=class_names, y=class_names,
            color_continuous_scale="Reds",
            labels=dict(x="Predicted", y="Actual"),
            title="Confusion Matrix",
            text_auto=True
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

        # Feature Importance (Random Forest only)
        if model_name == "Random Forest" and "trained_model" in st.session_state:
            st.subheader("🧬 Top 20 Important Genes")
            model = st.session_state["trained_model"]
            gene_names = st.session_state["gene_names"]

            # REQ 5: Vectorized importance extraction
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            top_genes = [gene_names[i] for i in top_idx]
            top_scores = importances[top_idx]

            fig_imp = px.bar(
                x=top_scores, y=top_genes,
                orientation="h",
                title="Top 20 Gene Importances",
                labels={"x": "Importance", "y": "Gene"},
                color=top_scores,
                color_continuous_scale="Reds",
                template="plotly_white"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)
            gc.collect()  # REQ 4

        # ── Individual Prediction ──
        st.markdown("---")
        st.subheader("🔍 Individual Sample Prediction")

        if X_scaled is not None:
            sample_idx = st.slider("Select sample index", 0, len(X_scaled)-1, 0)
            if st.button("🔮 Predict This Sample"):
                model = st.session_state["trained_model"]
                sample = X_scaled[sample_idx].reshape(1, -1)
                pred = model.predict(sample)[0]
                prob = model.predict_proba(sample)[0] if hasattr(model, "predict_proba") else None

                predicted_label = le.inverse_transform([pred])[0]
                actual_label = le.inverse_transform([y[sample_idx]])[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted", predicted_label)
                with col2:
                    st.metric("Actual", actual_label)

                if prob is not None:
                    prob_df = pd.DataFrame({
                        "Class": le.classes_,
                        "Probability": prob
                    })
                    fig_prob = px.bar(
                        prob_df, x="Class", y="Probability",
                        color="Probability",
                        color_continuous_scale="RdYlGn",
                        title="Prediction Confidence",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

        gc.collect()  # REQ 4