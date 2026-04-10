import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lifelines import KaplanMeierFitter
import GEOparse


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_golub() -> tuple[pd.DataFrame, pd.Series]:
    """Return (X_features, y_labels) from the Golub / GSE13159 dataset."""
    try:
        gse = GEOparse.get_GEO(geo="GSE13159", destdir="./data/", silent=True)
    except Exception:
        return None, None

    try:
        pivoted = gse.pivot_samples("VALUE").apply(pd.to_numeric, errors="coerce").dropna()
    except Exception:
        return None, None

    labels = {}
    for gsm_name, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0].upper()
        char  = " ".join(gsm.metadata.get("characteristics_ch1", [])).upper()
        combined = title + " " + char
        if "ALL" in combined:
            labels[gsm_name] = "ALL"
        elif "AML" in combined:
            labels[gsm_name] = "AML"

    common = [s for s in pivoted.columns if s in labels]
    if len(common) < 10:
        return None, None

    X = pivoted[common].T          # samples × genes
    y = pd.Series([labels[s] for s in common], index=common)
    return X, y


# ── Main ──────────────────────────────────────────────────────────────────────

def show():
    st.header("🩺 Survival Prediction")
    st.markdown(
        "Train machine learning models on gene expression data to classify "
        "leukemia subtypes and explore survival dynamics."
    )

    # ── Kaplan-Meier Section ──────────────────────────────────────────────────
    st.subheader("📈 Kaplan-Meier Survival Curves (Simulated)")
    st.caption(
        "Real survival data requires clinical metadata. "
        "The curves below are simulated to illustrate the concept."
    )

    np.random.seed(42)
    n = 80
    all_times  = np.random.exponential(scale=36, size=n)
    aml_times  = np.random.exponential(scale=18, size=n)
    all_event  = np.random.binomial(1, 0.65, size=n)
    aml_event  = np.random.binomial(1, 0.80, size=n)

    kmf_all = KaplanMeierFitter()
    kmf_aml = KaplanMeierFitter()
    kmf_all.fit(all_times, all_event, label="ALL")
    kmf_aml.fit(aml_times, aml_event, label="AML")

    fig_km = go.Figure()
    for kmf, color, name in [(kmf_all, "#1f77b4", "ALL"), (kmf_aml, "#d62728", "AML")]:
        fig_km.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_[name],
            mode="lines", name=name, line=dict(color=color, width=2.5)
        ))
    fig_km.update_layout(
        title="Kaplan-Meier — ALL vs AML (Simulated)",
        xaxis_title="Time (months)", yaxis_title="Survival Probability",
        template="plotly_white", height=380
    )
    st.plotly_chart(fig_km, use_container_width=True)

    # ── ML Classification Section ─────────────────────────────────────────────
    st.subheader("🤖 ML-Based Subtype Classification")

    model_choice = st.selectbox(
        "Choose a classifier",
        ["Random Forest", "Logistic Regression", "Support Vector Machine"]
    )
    n_top = st.slider("Number of top genes to use as features", 50, 500, 200, 50)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Loading dataset from NCBI GEO…"):
            X, y = _load_golub()

        if X is None:
            st.error(
                "Dataset not found or connection timed out. "
                "Please try again or check your internet connection."
            )
            return

        # Feature selection: top N genes by variance
        variances   = X.var()
        top_genes   = variances.nlargest(n_top).index
        X_sel       = X[top_genes]

        le          = LabelEncoder()
        y_enc       = le.fit_transform(y)
        scaler      = StandardScaler()
        X_scaled    = scaler.fit_transform(X_sel)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ── Model setup ───────────────────────────────────────────────────────
        if model_choice == "Random Forest":
            clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        elif model_choice == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            clf = SVC(kernel="rbf", probability=True, random_state=42)

        with st.spinner(f"Training {model_choice} with 5-fold cross-validation…"):
            scores = cross_val_score(clf, X_scaled, y_enc, cv=cv, scoring="accuracy")

        # ── Results ───────────────────────────────────────────────────────────
        st.success("✅ Training complete!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean CV Accuracy", f"{scores.mean():.3f}")
        col2.metric("Std Dev", f"± {scores.std():.3f}")
        col3.metric("Folds", "5")

        # CV accuracy bar
        fig_cv = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=scores,
            marker_color=["#2ecc71" if s >= 0.90 else "#e74c3c" for s in scores],
            text=[f"{s:.3f}" for s in scores],
            textposition="outside"
        ))
        fig_cv.update_layout(
            title="Cross-Validation Accuracy per Fold",
            yaxis=dict(range=[0, 1.05], title="Accuracy"),
            template="plotly_white", height=350
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        # ── Feature Importance (Random Forest only) ───────────────────────────
        if model_choice == "Random Forest":
            with st.spinner("Computing feature importances…"):
                clf.fit(X_scaled, y_enc)          # fit on full data for importances

            importances = clf.feature_importances_
            feat_df = (
                pd.DataFrame({"Gene": top_genes, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

            st.subheader("🧬 Top 10 Most Important Genes (Random Forest)")
            st.caption(
                "These are the genes the model weighted most heavily "
                "when distinguishing ALL from AML."
            )

            fig_fi = go.Figure(go.Bar(
                x=feat_df["Importance"][::-1],
                y=feat_df["Gene"][::-1],
                orientation="h",
                marker=dict(
                    color=feat_df["Importance"][::-1],
                    colorscale="Reds",
                    showscale=False
                ),
                text=[f"{v:.4f}" for v in feat_df["Importance"][::-1]],
                textposition="outside"
            ))
            fig_fi.update_layout(
                title="Feature Importance — Top 10 Genes",
                xaxis_title="Importance Score",
                yaxis_title="Gene",
                template="plotly_white",
                height=420,
                margin=dict(l=120)
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # NCBI links for top genes
            st.markdown("**🔗 Explore these genes on NCBI:**")
            links = " · ".join(
                f"[{g}](https://www.ncbi.nlm.nih.gov/gene/?term={g})"
                for g in feat_df["Gene"]
            )
            st.markdown(links)

        # ── Individual Patient Prediction ─────────────────────────────────────
        st.subheader("🔮 Individual Patient Prediction")
        st.caption(
            "Enter a comma-separated list of gene expression values "
            f"({n_top} values matching the top {n_top} most variable genes)."
        )

        sample_input = st.text_area(
            "Paste expression values (comma-separated)",
            placeholder=f"E.g., 5.2, 3.1, 8.4, … ({n_top} values total)",
            height=80
        )

        if st.button("Predict Subtype"):
            try:
                vals = [float(x.strip()) for x in sample_input.split(",")]
                if len(vals) != n_top:
                    st.warning(f"Please provide exactly {n_top} values (got {len(vals)}).")
                else:
                    clf.fit(X_scaled, y_enc)
                    sample_scaled = scaler.transform([vals])
                    pred          = clf.predict(sample_scaled)[0]
                    proba         = clf.predict_proba(sample_scaled)[0]
                    label         = le.inverse_transform([pred])[0]
                    confidence    = max(proba) * 100
                    st.success(
                        f"**Predicted Subtype: {label}** "
                        f"(Confidence: {confidence:.1f}%)"
                    )
            except ValueError:
                st.error("Invalid input. Please enter numeric values separated by commas.")