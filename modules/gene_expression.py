import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import gc
import io
import requests

# ─────────────────────────────────────────────
# HELPER: Safe dataframe to numpy (no JSON)
# Avoids FileNotFoundError from JSON serialization
# ─────────────────────────────────────────────
def df_to_array(df):
    """Converts dataframe to float32 numpy array safely."""
    arr = df.values.astype("float32")
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


# ─────────────────────────────────────────────
# HELPER: Load Golub dataset safely
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_golub():
    """Tries local file first, then GitHub URL."""
    
    # Try local file first
    local_paths = [
        "data/golub_data.csv",
        "./data/golub_data.csv",
        "golub_data.csv"
    ]
    
    for path in local_paths:
        try:
            df = pd.read_csv(path, index_col=0)
            df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
            return df, None
        except FileNotFoundError:
            continue
        except Exception as e:
            continue

    # Try GitHub URL
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
                return df, None
        except Exception:
            continue

    return None, "❌ Golub dataset not found locally or on GitHub. Please upload your CSV manually."


# ─────────────────────────────────────────────
# HELPER: PCA — works directly on numpy array
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_pca(_arr, sample_names, n_components=2):
    """
    Takes numpy array directly — no JSON serialization.
    Underscore prefix on _arr tells Streamlit not to hash it.
    """
    try:
        arr = np.nan_to_num(_arr, nan=0.0).astype("float32")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(arr.T)  # samples x genes
        pca = PCA(n_components=min(n_components, scaled.shape[1], scaled.shape[0]))
        components = pca.fit_transform(scaled)
        variance = pca.explained_variance_ratio_ * 100
        gc.collect()
        return components, variance, None
    except Exception as e:
        return None, None, f"PCA failed: {str(e)}"


# ─────────────────────────────────────────────
# HELPER: t-SNE — works directly on numpy array
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_tsne(_arr, n_samples):
    """Takes numpy array directly."""
    try:
        arr = np.nan_to_num(_arr, nan=0.0).astype("float32")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(arr.T)  # samples x genes

        # PCA pre-reduction for speed
        n_components = min(50, scaled.shape[1], scaled.shape[0] - 1)
        if n_components > 1:
            pca_pre = PCA(n_components=n_components)
            scaled = pca_pre.fit_transform(scaled)

        perplexity = min(30, n_samples - 1, 5)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        result = tsne.fit_transform(scaled)
        gc.collect()
        return result, None
    except Exception as e:
        return None, f"t-SNE failed: {str(e)}"


# ─────────────────────────────────────────────
# HELPER: KMeans — works directly on numpy array
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_kmeans(_arr, n_clusters):
    """Takes numpy array directly."""
    try:
        arr = np.nan_to_num(_arr, nan=0.0).astype("float32")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(arr.T)  # samples x genes
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        gc.collect()
        return labels, None
    except Exception as e:
        return None, f"Clustering failed: {str(e)}"


# ─────────────────────────────────────────────
# HELPER: Drop zero variance genes
# ─────────────────────────────────────────────
def drop_zero_variance(df):
    variances = df.var(axis=1)
    zero_count = (variances == 0).sum()
    if zero_count > 0:
        df = df[variances > 0]
        st.info(f"🧹 Auto-removed {zero_count} zero-variance genes before analysis.")
    return df


# ─────────────────────────────────────────────
# MAIN SHOW FUNCTION
# ─────────────────────────────────────────────
def show():
    st.title("📊 Gene Expression Visualization")
    st.markdown("Explore expression patterns using heatmaps, PCA, t-SNE and clustering.")

    with st.expander("📖 How to use this module"):
        st.markdown("""
        1. **Load Data** — Use Golub dataset, GEO loaded data, or upload your own CSV
        2. **Set Labels** — Assign sample types (ALL/AML) for color coding
        3. **Explore Tabs** — Heatmap, PCA, t-SNE, Clustering
        4. All heavy computations are **cached** — runs fast after first time
        """)

    st.markdown("---")

    # ── Load Data ──
    st.subheader("📂 Load Data")

    data_source = st.radio(
        "Select data source:",
        ["📦 Golub Dataset (default)", "🔄 Use GEO Loaded Data", "📁 Upload CSV"],
        horizontal=True
    )

    df = None
    source_name = "Unknown"

    # ── Golub ──
    if "Golub" in data_source:
        with st.spinner("Loading Golub dataset..."):
            progress = st.progress(0)
            df, err = load_golub()
            progress.progress(100)
            progress.empty()

        if err:
            st.error(err)
            st.info("💡 Please upload your golub_data.csv manually using the Upload CSV option.")
        else:
            source_name = "Golub 1999"
            st.success(f"✅ Golub dataset loaded: {df.shape[0]} genes × {df.shape[1]} samples")

    # ── GEO ──
    elif "GEO" in data_source:
        if "geo_df" in st.session_state and st.session_state["geo_df"] is not None:
            df = st.session_state["geo_df"]
            source_name = st.session_state.get("geo_accession", "GEO Dataset")
            st.success(f"✅ Using {source_name}: {df.shape[0]} genes × {df.shape[1]} samples")
        else:
            st.warning("⚠️ No GEO dataset found in session. Go to **Biomarker Discovery** first and fetch a dataset.")

    # ── Upload ──
    else:
        uploaded = st.file_uploader("Upload CSV (rows=genes, columns=samples)", type=["csv"])
        if uploaded:
            with st.spinner("Reading file..."):
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
                    source_name = uploaded.name
                    st.success(f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples")
                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Could not read file: {str(e)}")

    # ── Nothing loaded ──
    if df is None or df.empty:
        st.info("👆 Load a dataset above to continue.")
        return

    # ── Drop zero variance ──
    df = drop_zero_variance(df)

    if df.empty:
        st.error("❌ All genes have zero variance. Please upload a valid expression dataset.")
        return

    # ── Labels ──
    st.markdown("---")
    st.subheader("🏷️ Sample Labels")

    n_samples = df.shape[1]
    half = n_samples // 2
    default_labels = ["ALL"] * half + ["AML"] * (n_samples - half)
    labels = default_labels

    with st.expander("⚙️ Customize Sample Labels"):
        st.markdown("Enter one label per line, matching the order of samples in your dataset.")
        label_input = st.text_area(
            "Labels (one per line):",
            value="\n".join(default_labels),
            height=150
        )
        custom_labels = [l.strip() for l in label_input.strip().split("\n") if l.strip()]
        if len(custom_labels) == n_samples:
            labels = custom_labels
            st.success(f"✅ {len(labels)} labels set.")
        elif len(custom_labels) != n_samples:
            st.warning(f"⚠️ Label count ({len(custom_labels)}) doesn't match samples ({n_samples}). Using defaults.")

    # ── Pre-compute numpy array once ──
    arr = df_to_array(df)
    sample_names = list(df.columns)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "🔵 PCA", "🌀 t-SNE", "🔲 Clustering"])

    # ─────────────────────────────
    # TAB 1: HEATMAP
    # ─────────────────────────────
    with tab1:
        st.subheader("🔥 Gene Expression Heatmap")

        with st.expander("ℹ️ About this plot"):
            st.markdown("""
            Shows expression levels (Z-score normalized) for the most variable genes.
            - **Red** = high expression
            - **Blue** = low expression
            - Rows = genes, Columns = samples
            """)

        n_genes = st.slider("Number of top variable genes", 20, min(200, df.shape[0]), 50, 10)

        if st.button("🔥 Generate Heatmap"):
            try:
                with st.spinner("Building heatmap... ⏳"):
                    progress = st.progress(0)

                    # Vectorized top gene selection
                    progress.progress(20)
                    top_idx = np.argsort(arr.var(axis=1))[::-1][:n_genes]
                    heat_arr = arr[top_idx, :]
                    heat_genes = df.index[top_idx].tolist()
                    progress.progress(50)

                    # Vectorized Z-score normalization
                    row_mean = heat_arr.mean(axis=1, keepdims=True)
                    row_std = heat_arr.std(axis=1, keepdims=True) + 1e-9
                    z_scored = (heat_arr - row_mean) / row_std
                    progress.progress(75)

                    fig = go.Figure(data=go.Heatmap(
                        z=z_scored,
                        x=sample_names,
                        y=heat_genes,
                        colorscale="RdBu_r",
                        zmid=0,
                        colorbar=dict(title="Z-score"),
                    ))
                    fig.update_layout(
                        title=f"Top {n_genes} Most Variable Genes — {source_name}",
                        height=max(400, n_genes * 14),
                        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                        yaxis=dict(tickfont=dict(size=8)),
                        template="plotly_white"
                    )
                    progress.progress(100)
                    progress.empty()

                st.plotly_chart(fig, use_container_width=True)
                gc.collect()

            except Exception as e:
                st.error(f"❌ Heatmap failed: {str(e)}")

    # ─────────────────────────────
    # TAB 2: PCA
    # ─────────────────────────────
    with tab2:
        st.subheader("🔵 PCA — Principal Component Analysis")

        with st.expander("ℹ️ About PCA"):
            st.markdown("""
            Reduces thousands of gene dimensions to 2 principal components.
            Each dot = one patient sample. Colors = leukemia type.
            Clusters that separate well = dataset is biologically meaningful.
            """)

        if st.button("▶️ Run PCA"):
            try:
                progress = st.progress(0)
                status = st.empty()
                status.text("Scaling data...")
                progress.progress(25)

                with st.spinner("Computing PCA... ⏳"):
                    components, variance, err = compute_pca(arr, tuple(sample_names))

                if err:
                    progress.empty()
                    status.empty()
                    st.error(f"❌ {err}")
                else:
                    progress.progress(80)
                    status.text("Rendering...")

                    pca_df = pd.DataFrame({
                        "PC1": components[:, 0],
                        "PC2": components[:, 1],
                        "Sample": sample_names,
                        "Label": labels[:len(sample_names)]
                    })

                    fig = px.scatter(
                        pca_df, x="PC1", y="PC2",
                        color="Label",
                        hover_name="Sample",
                        title=f"PCA — PC1 ({variance[0]:.1f}%) vs PC2 ({variance[1]:.1f}%) | {source_name}",
                        template="plotly_white",
                        height=500,
                        render_mode="webgl"
                    )
                    fig.update_traces(marker=dict(size=10, opacity=0.8))

                    progress.progress(100)
                    status.empty()
                    progress.empty()

                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"📊 PC1: **{variance[0]:.1f}%** variance | PC2: **{variance[1]:.1f}%** variance")
                    gc.collect()

            except Exception as e:
                st.error(f"❌ PCA error: {str(e)}")

    # ─────────────────────────────
    # TAB 3: t-SNE
    # ─────────────────────────────
    with tab3:
        st.subheader("🌀 t-SNE Visualization")

        with st.expander("ℹ️ About t-SNE"):
            st.markdown("""
            Non-linear dimensionality reduction — better at showing clusters than PCA.
            Uses PCA pre-reduction for speed and stability.
            May take 20-40 seconds for large datasets.
            """)

        if n_samples < 4:
            st.warning("⚠️ Need at least 4 samples for t-SNE.")
        else:
            if st.button("▶️ Run t-SNE"):
                try:
                    progress = st.progress(0)
                    status = st.empty()
                    status.text("Running PCA pre-reduction...")
                    progress.progress(15)

                    with st.spinner("Running t-SNE... ⏳"):
                        tsne_result, err = compute_tsne(arr, n_samples)

                    if err:
                        progress.empty()
                        status.empty()
                        st.error(f"❌ {err}")
                    else:
                        progress.progress(85)
                        status.text("Rendering...")

                        tsne_df = pd.DataFrame({
                            "Dim1": tsne_result[:, 0],
                            "Dim2": tsne_result[:, 1],
                            "Sample": sample_names,
                            "Label": labels[:len(sample_names)]
                        })

                        fig = px.scatter(
                            tsne_df, x="Dim1", y="Dim2",
                            color="Label",
                            hover_name="Sample",
                            title=f"t-SNE Visualization | {source_name}",
                            template="plotly_white",
                            height=500,
                            render_mode="webgl"
                        )
                        fig.update_traces(marker=dict(size=10, opacity=0.8))

                        progress.progress(100)
                        status.empty()
                        progress.empty()

                        st.plotly_chart(fig, use_container_width=True)
                        gc.collect()

                except Exception as e:
                    st.error(f"❌ t-SNE error: {str(e)}")

    # ─────────────────────────────
    # TAB 4: Clustering
    # ─────────────────────────────
    with tab4:
        st.subheader("🔲 K-Means Clustering")

        with st.expander("ℹ️ About Clustering"):
            st.markdown("""
            Groups samples into clusters based on gene expression similarity.
            Visualized on PCA axes for easy interpretation.
            Compare cluster assignments vs true labels to evaluate separation.
            """)

        n_clusters = st.slider("Number of clusters", 2, min(6, n_samples - 1), 2)

        if st.button("▶️ Run Clustering"):
            try:
                progress = st.progress(0)
                status = st.empty()
                status.text("Running KMeans...")
                progress.progress(20)

                with st.spinner("Clustering... ⏳"):
                    cluster_labels, err1 = compute_kmeans(arr, n_clusters)
                    progress.progress(55)

                    status.text("Running PCA for visualization...")
                    components, variance, err2 = compute_pca(arr, tuple(sample_names))
                    progress.progress(85)

                if err1:
                    st.error(f"❌ Clustering: {err1}")
                elif err2:
                    st.error(f"❌ PCA for visualization: {err2}")
                else:
                    cluster_df = pd.DataFrame({
                        "PC1": components[:, 0],
                        "PC2": components[:, 1],
                        "Sample": sample_names,
                        "True Label": labels[:len(sample_names)],
                        "Cluster": [f"Cluster {c + 1}" for c in cluster_labels]
                    })

                    fig = px.scatter(
                        cluster_df, x="PC1", y="PC2",
                        color="Cluster",
                        symbol="True Label",
                        hover_name="Sample",
                        hover_data={"True Label": True, "Cluster": True},
                        title=f"K-Means (k={n_clusters}) on PCA | {source_name}",
                        template="plotly_white",
                        height=500,
                        render_mode="webgl"
                    )
                    fig.update_traces(marker=dict(size=10, opacity=0.8))

                    progress.progress(100)
                    status.empty()
                    progress.empty()

                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📋 Cluster Assignment Table"):
                        st.dataframe(
                            cluster_df[["Sample", "True Label", "Cluster"]].reset_index(drop=True),
                            use_container_width=True
                        )

                    gc.collect()

            except Exception as e:
                st.error(f"❌ Clustering error: {str(e)}")