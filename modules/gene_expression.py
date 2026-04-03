import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def show():
    st.title("📊 Gene Expression Visualization")
    st.markdown("Explore gene expression patterns, cluster patients, and compare leukemia subtypes.")

    st.markdown("---")

    st.subheader("📂 Step 1: Load Data")

    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload CSV File", "♻️ Use Data from Biomarker Module"],
        horizontal=True
    )

    df = None

    if "Use Data from Biomarker" in input_method:
        if "geo_df" in st.session_state:
            df = st.session_state["geo_df"]
            st.success(f"✅ Using data from Biomarker module! Shape: {df.shape[0]} genes × {df.shape[1]} samples")
        else:
            st.warning("No data found from Biomarker module. Please upload a CSV below.")
            input_method = "📁 Upload CSV File"

    if "Upload" in input_method:
        uploaded_file = st.file_uploader(
            "Upload gene expression CSV (rows = genes, columns = samples)",
            type=["csv"]
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file, index_col=0)
            st.success(f"✅ File loaded! Shape: {df.shape[0]} genes × {df.shape[1]} samples")

    if df is not None and not df.empty:

        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.dropna(how="all")
        numeric_df.index = numeric_df.index.astype(str)

        st.session_state.total_patients = numeric_df.shape[1]
        st.session_state.gene_features = f"{numeric_df.shape[0]:,}"
        st.session_state.data_loaded = True

        st.subheader("👁️ Data Preview")
        st.dataframe(numeric_df.head(5), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Genes", numeric_df.shape[0])
        m2.metric("Total Samples", numeric_df.shape[1])
        m3.metric("Missing Values", int(numeric_df.isnull().sum().sum()))

        st.markdown("---")

        st.subheader("👥 Step 2: Define Sample Groups (Optional)")
        use_groups = st.checkbox("I want to define sample groups for comparison")

        group1_cols, group2_cols = [], []
        group1_name, group2_name = "Group 1", "Group 2"

        if use_groups:
            all_cols = list(numeric_df.columns)
            col1, col2 = st.columns(2)
            with col1:
                group1_name = st.text_input("Group 1 Name", value="ALL")
                group1_cols = st.multiselect("Group 1 Samples", all_cols, default=all_cols[:len(all_cols)//2])
            with col2:
                group2_name = st.text_input("Group 2 Name", value="AML")
                group2_cols = st.multiselect("Group 2 Samples", all_cols, default=all_cols[len(all_cols)//2:])

        st.markdown("---")

        st.subheader("🔥 Interactive Gene Expression Heatmap")
        n_genes = st.slider("Number of top variable genes to show", 10, 50, 20)
        n_samples = st.slider("Number of samples to show", 10, min(50, numeric_df.shape[1]), 20)

        top_var_genes = numeric_df.var(axis=1).nlargest(n_genes).index
        heatmap_data = numeric_df.loc[top_var_genes].iloc[:, :n_samples]

        fig_heatmap = px.imshow(
            heatmap_data,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title=f"Top {n_genes} Variable Genes × {n_samples} Samples",
            labels={"x": "Samples", "y": "Genes", "color": "Expression"}
        )
        fig_heatmap.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")

        st.subheader("🔍 Gene Search")
        gene_options = list(numeric_df.index.astype(str))
        selected_gene = st.selectbox("Search and select a gene:", gene_options)

        if selected_gene:
            try:
                gene_expr = numeric_df.loc[str(selected_gene)]
                if isinstance(gene_expr, pd.DataFrame):
                    gene_expr = gene_expr.iloc[0]
                gene_expr = pd.to_numeric(gene_expr, errors="coerce")

                if use_groups and group1_cols and group2_cols:
                    colors = []
                    for col in gene_expr.index:
                        if col in group1_cols:
                            colors.append(group1_name)
                        elif col in group2_cols:
                            colors.append(group2_name)
                        else:
                            colors.append("Other")
                    fig_gene = px.bar(
                        x=list(gene_expr.index), y=list(gene_expr.values),
                        color=colors,
                        color_discrete_map={group1_name: "#e74c3c", group2_name: "#2980b9", "Other": "#95a5a6"},
                        title=f"Expression of {selected_gene} Across Samples",
                        labels={"x": "Sample", "y": "Expression Level"},
                        template="plotly_white"
                    )
                else:
                    fig_gene = px.bar(
                        x=list(gene_expr.index), y=list(gene_expr.values),
                        color=list(gene_expr.values),
                        color_continuous_scale="Reds",
                        title=f"Expression of {selected_gene} Across Samples",
                        labels={"x": "Sample", "y": "Expression Level"},
                        template="plotly_white"
                    )
                fig_gene.update_layout(height=400)
                st.plotly_chart(fig_gene, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean Expression", f"{gene_expr.mean():.2f}")
                c2.metric("Std Dev", f"{gene_expr.std():.2f}")
                c3.metric("Min", f"{gene_expr.min():.2f}")
                c4.metric("Max", f"{gene_expr.max():.2f}")

            except Exception as e:
                st.warning(f"Could not find gene '{selected_gene}'. Please try another.")

        st.markdown("---")

        st.subheader("🔵 Dimensionality Reduction — PCA / t-SNE")
        st.markdown("Reduces thousands of genes into 2D so you can see how samples cluster.")

        dr_method = st.radio("Choose method:", ["PCA", "t-SNE"], horizontal=True)
        n_clusters = st.slider("Number of clusters (K-Means)", 2, 6, 2)

        if st.button(f"▶️ Run {dr_method} + Clustering"):
            with st.spinner(f"Running {dr_method}... ⏳"):
                try:
                    X = numeric_df.T.fillna(0)
                    X = X.loc[:, X.std() > 0]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    if dr_method == "PCA":
                        n_comp = min(2, X_scaled.shape[0], X_scaled.shape[1])
                        reducer = PCA(n_components=n_comp, random_state=42)
                        coords = reducer.fit_transform(X_scaled)
                        explained = reducer.explained_variance_ratio_ * 100
                        axis_labels = {
                            "x": f"PC1 ({explained[0]:.1f}% variance)",
                            "y": f"PC2 ({explained[1]:.1f}% variance)" if len(explained) > 1 else "PC2"
                        }
                    else:
                        n_pca = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
                        X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X_scaled)
                        perp = min(30, len(X) - 1)
                        coords = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X_pca)
                        axis_labels = {"x": "t-SNE 1", "y": "t-SNE 2"}

                    n_clust = min(n_clusters, len(X))
                    cluster_labels = KMeans(n_clusters=n_clust, random_state=42, n_init=10).fit_predict(coords)

                    plot_df = pd.DataFrame({
                        "x": coords[:, 0], "y": coords[:, 1],
                        "Sample": list(X.index),
                        "Cluster": [f"Cluster {c+1}" for c in cluster_labels]
                    })

                    if use_groups and group1_cols and group2_cols:
                        def get_group(sample):
                            if sample in group1_cols:
                                return group1_name
                            elif sample in group2_cols:
                                return group2_name
                            return "Other"
                        plot_df["Group"] = plot_df["Sample"].apply(get_group)
                        color_col = "Group"
                    else:
                        color_col = "Cluster"

                    st.session_state["dr_plot_df"] = plot_df
                    st.session_state["dr_axis_labels"] = axis_labels
                    st.session_state["dr_color_col"] = color_col
                    st.session_state["dr_method"] = dr_method
                    st.success("✅ Done!")

                except Exception as e:
                    st.error(f"Error during {dr_method}: {str(e)}")

        if "dr_plot_df" in st.session_state:
            plot_df = st.session_state["dr_plot_df"]
            axis_labels = st.session_state["dr_axis_labels"]
            color_col = st.session_state["dr_color_col"]
            dr_label = st.session_state["dr_method"]

            fig_dr = px.scatter(
                plot_df, x="x", y="y", color=color_col,
                hover_name="Sample",
                title=f"{dr_label} Plot — Sample Clustering",
                labels=axis_labels,
                template="plotly_white", height=500,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_dr.update_traces(marker=dict(size=10, opacity=0.8))
            st.plotly_chart(fig_dr, use_container_width=True)
            st.info("💡 Each dot = one patient sample. Samples close together have similar gene expression.")

        st.markdown("---")

        if use_groups and group1_cols and group2_cols:
            st.subheader(f"⚖️ {group1_name} vs {group2_name} Comparison")
            top_genes = numeric_df.var(axis=1).nlargest(20).index
            mean_g1 = numeric_df.loc[top_genes, group1_cols].mean(axis=1)
            mean_g2 = numeric_df.loc[top_genes, group2_cols].mean(axis=1)
            compare_df = pd.DataFrame({
                "Gene": list(top_genes) * 2,
                "Mean Expression": list(mean_g1) + list(mean_g2),
                "Group": [group1_name] * len(top_genes) + [group2_name] * len(top_genes)
            })
            fig_compare = px.bar(
                compare_df, x="Gene", y="Mean Expression", color="Group",
                barmode="group",
                color_discrete_map={group1_name: "#e74c3c", group2_name: "#2980b9"},
                title=f"Top 20 Variable Genes: {group1_name} vs {group2_name}",
                template="plotly_white", height=450
            )
            fig_compare.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_compare, use_container_width=True)
            st.markdown("---")

        st.subheader("📥 Download Results")
        csv = numeric_df.to_csv()
        st.download_button(
            "⬇️ Download Cleaned Expression Data (CSV)",
            data=csv,
            file_name="leukodash_expression_data.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Please load a dataset above to get started.")