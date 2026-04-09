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

# ─────────────────────────────────────────────
# REQ 1+2: Cached data loader
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_golub():
    import io, requests
    url = "https://raw.githubusercontent.com/talha-sa/LeukoDash/main/data/golub_data.csv"
    try:
        r = requests.get(url, timeout=30)
        df = pd.read_csv(io.StringIO(r.text), index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce").astype("float32")  # REQ 2
        return df
    except Exception:
        return None


# REQ 1+5: Cached + vectorized PCA
@st.cache_data(show_spinner=False)
def compute_pca(df_json, n_components=2):
    df = pd.read_json(df_json).astype("float32")
    df = df.fillna(df.mean())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.T)  # samples x genes
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
    variance = pca.explained_variance_ratio_ * 100
    gc.collect()  # REQ 4
    return components, variance


# REQ 1+5: Cached + vectorized t-SNE
@st.cache_data(show_spinner=False)
def compute_tsne(df_json):
    df = pd.read_json(df_json).astype("float32")
    df = df.fillna(df.mean())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.T)
    # REQ 5: Use PCA first to reduce dims before t-SNE (vectorized)
    pca_50 = PCA(n_components=min(50, scaled.shape[1]))
    reduced = pca_50.fit_transform(scaled)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df.columns)-1))
    result = tsne.fit_transform(reduced)
    gc.collect()  # REQ 4
    return result


# REQ 1+5: Cached + vectorized KMeans
@st.cache_data(show_spinner=False)
def compute_kmeans(df_json, n_clusters=2):
    df = pd.read_json(df_json).astype("float32")
    df = df.fillna(df.mean())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.T)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    gc.collect()  # REQ 4
    return labels


def show():
    st.title("📊 Gene Expression Visualization")
    st.markdown("Explore expression patterns using heatmaps, PCA, t-SNE and clustering.")
    st.markdown("---")

    # ── Load Data ──
    st.subheader("📂 Load Data")
    data_source = st.radio(
        "Select data source:",
        ["📦 Use Golub Dataset (default)", "🔄 Use GEO Loaded Data", "📁 Upload CSV"],
        horizontal=True
    )

    df = None

    if "Golub" in data_source:
        with st.spinner("Loading Golub dataset..."):
            df = load_golub()
        if df is not None:
            st.success(f"✅ Golub dataset loaded: {df.shape[0]} genes × {df.shape[1]} samples")
        else:
            st.error("Could not load Golub dataset. Check your GitHub repo URL.")

    elif "GEO" in data_source:
        if "geo_df" in st.session_state:
            df = st.session_state["geo_df"]
            st.success(f"✅ Using GEO dataset: {df.shape[0]} genes × {df.shape[1]} samples")
        else:
            st.warning("⚠️ No GEO dataset loaded. Go to Biomarker Discovery first.")

    else:
        uploaded = st.file_uploader("Upload CSV (rows=genes, columns=samples)", type=["csv"])
        if uploaded:
            with st.spinner("Reading file..."):
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

    # ── Labels ──
    st.markdown("---")
    st.subheader("🏷️ Sample Labels")
    n_samples = df.shape[1]
    half = n_samples // 2
    default_labels = ["ALL"] * half + ["AML"] * (n_samples - half)
    labels = default_labels

    use_custom = st.checkbox("Set custom sample labels")
    if use_custom:
        label_input = st.text_area(
            "Enter one label per line (same order as samples):",
            value="\n".join(default_labels)
        )
        labels = [l.strip() for l in label_input.strip().split("\n") if l.strip()]
        if len(labels) != n_samples:
            st.warning(f"⚠️ Labels count ({len(labels)}) must match samples ({n_samples}).")
            labels = default_labels

    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "🔵 PCA", "🌀 t-SNE", "🔲 Clustering"])

    # ────────────────────────────
    # TAB 1: HEATMAP
    # ────────────────────────────
    with tab1:
        st.subheader("🔥 Gene Expression Heatmap")

        n_genes = st.slider("Number of top variable genes to show", 20, 200, 50, 10)

        if st.button("Generate Heatmap"):
            with st.spinner("Building heatmap... ⏳"):
                progress = st.progress(0)

                # REQ 5: Vectorized top gene selection
                progress.progress(20)
                top_var_genes = df.var(axis=1).nlargest(n_genes).index
                heat_df = df.loc[top_var_genes]
                progress.progress(50)

                # REQ 5: Vectorized z-score normalization
                heat_array = heat_df.values.astype("float32")
                row_mean = heat_array.mean(axis=1, keepdims=True)
                row_std = heat_array.std(axis=1, keepdims=True) + 1e-9
                z_scored = (heat_array - row_mean) / row_std
                progress.progress(75)

                # REQ 7: WebGL heatmap for large data
                fig = go.Figure(data=go.Heatmap(
                    z=z_scored,
                    x=list(heat_df.columns),
                    y=list(heat_df.index),
                    colorscale="RdBu_r",
                    zmid=0,
                    colorbar=dict(title="Z-score"),
                ))

                fig.update_layout(
                    title=f"Top {n_genes} Most Variable Genes",
                    height=max(400, n_genes * 12),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                    yaxis=dict(tickfont=dict(size=8)),
                    template="plotly_white"
                )

                progress.progress(100)
                st.plotly_chart(fig, use_container_width=True)
                progress.empty()

                gc.collect()  # REQ 4 after heatmap

    # ────────────────────────────
    # TAB 2: PCA
    # ────────────────────────────
    with tab2:
        st.subheader("🔵 PCA — Principal Component Analysis")

        if st.button("Run PCA"):
            with st.spinner("Computing PCA... ⏳"):
                # REQ 6: Progress bar
                progress = st.progress(0)
                status = st.empty()

                status.text("Scaling data...")
                progress.progress(25)

                # REQ 1+5: Cached vectorized PCA
                components, variance = compute_pca(df.to_json())
                progress.progress(75)

                status.text("Rendering plot...")
                pca_df = pd.DataFrame({
                    "PC1": components[:, 0],
                    "PC2": components[:, 1],
                    "Sample": df.columns,
                    "Label": labels[:len(df.columns)]
                })

                # REQ 7: WebGL for scatter
                fig = px.scatter(
                    pca_df, x="PC1", y="PC2",
                    color="Label",
                    hover_name="Sample",
                    title=f"PCA — PC1 ({variance[0]:.1f}%) vs PC2 ({variance[1]:.1f}%)",
                    template="plotly_white",
                    height=500,
                    render_mode="webgl"  # REQ 7
                )

                progress.progress(100)
                status.empty()
                progress.empty()
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"📊 PC1 explains **{variance[0]:.1f}%** | PC2 explains **{variance[1]:.1f}%** of variance.")
                gc.collect()  # REQ 4

    # ────────────────────────────
    # TAB 3: t-SNE
    # ────────────────────────────
    with tab3:
        st.subheader("🌀 t-SNE Visualization")
        st.info("ℹ️ t-SNE uses PCA pre-reduction for speed and stability.")

        if df.shape[1] < 4:
            st.warning("Need at least 4 samples for t-SNE.")
        else:
            if st.button("Run t-SNE"):
                with st.spinner("Running t-SNE (may take ~30 seconds)... ⏳"):
                    progress = st.progress(0)
                    status = st.empty()

                    status.text("Running PCA pre-reduction...")
                    progress.progress(20)

                    # REQ 1+5: Cached vectorized t-SNE
                    tsne_result = compute_tsne(df.to_json())
                    progress.progress(80)

                    status.text("Rendering...")
                    tsne_df = pd.DataFrame({
                        "Dim1": tsne_result[:, 0],
                        "Dim2": tsne_result[:, 1],
                        "Sample": df.columns,
                        "Label": labels[:len(df.columns)]
                    })

                    # REQ 7: WebGL render mode
                    fig = px.scatter(
                        tsne_df, x="Dim1", y="Dim2",
                        color="Label",
                        hover_name="Sample",
                        title="t-SNE Visualization",
                        template="plotly_white",
                        height=500,
                        render_mode="webgl"  # REQ 7
                    )

                    progress.progress(100)
                    status.empty()
                    progress.empty()
                    st.plotly_chart(fig, use_container_width=True)
                    gc.collect()  # REQ 4

    # ────────────────────────────
    # TAB 4: K-Means Clustering
    # ────────────────────────────
    with tab4:
        st.subheader("🔲 K-Means Clustering")

        n_clusters = st.slider("Number of clusters", 2, 6, 2)

        if st.button("Run Clustering"):
            with st.spinner("Clustering samples... ⏳"):
                progress = st.progress(0)
                status = st.empty()

                status.text("Running KMeans...")
                progress.progress(20)

                # REQ 1+5: Cached vectorized clustering
                cluster_labels = compute_kmeans(df.to_json(), n_clusters)
                progress.progress(60)

                status.text("Running PCA for visualization...")
                components, variance = compute_pca(df.to_json())
                progress.progress(85)

                cluster_df = pd.DataFrame({
                    "PC1": components[:, 0],
                    "PC2": components[:, 1],
                    "Sample": df.columns,
                    "True Label": labels[:len(df.columns)],
                    "Cluster": [f"Cluster {c+1}" for c in cluster_labels]
                })

                # REQ 7: WebGL for scatter
                fig = px.scatter(
                    cluster_df, x="PC1", y="PC2",
                    color="Cluster",
                    symbol="True Label",
                    hover_name="Sample",
                    title=f"K-Means Clustering (k={n_clusters}) on PCA",
                    template="plotly_white",
                    height=500,
                    render_mode="webgl"  # REQ 7
                )

                progress.progress(100)
                status.empty()
                progress.empty()
                st.plotly_chart(fig, use_container_width=True)
                gc.collect()  # REQ 4