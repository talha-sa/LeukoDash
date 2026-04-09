import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import gseapy as gp
import requests
import gzip
import gc
import io

# ─────────────────────────────────────────────
# REQ 1+2+3: Cached + float32 + chunked GEO loader
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_geo_data(accession, max_genes=1000):
    try:
        accession = accession.strip().upper()
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:-3]}nnn/{accession}/matrix/{accession}_series_matrix.txt.gz"

        response = requests.get(url, timeout=60, stream=True)
        if response.status_code != 200:
            return None, f"❌ Could not find {accession} on GEO. Check accession number."

        compressed = io.BytesIO(response.content)
        with gzip.open(compressed, "rt", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # REQ 3: Parse only data lines, skip metadata
        data_lines = [l for l in lines if not l.startswith("!") and not l.startswith("#")]
        del lines
        gc.collect()  # REQ 4

        if not data_lines:
            return None, "❌ Could not parse expression data."

        raw = "".join(data_lines)
        del data_lines

        # REQ 3: Read in chunks to avoid RAM spike
        chunk_iter = pd.read_csv(
            io.StringIO(raw),
            sep="\t",
            index_col=0,
            chunksize=2000  # process 2000 genes at a time
        )

        chunks = []
        for chunk in chunk_iter:
            chunk = chunk.apply(pd.to_numeric, errors="coerce")
            chunk = chunk.dropna(how="all")
            # REQ 2: Convert to float32 immediately per chunk
            chunk = chunk.astype("float32")
            chunks.append(chunk)

        df = pd.concat(chunks)
        del chunks
        gc.collect()  # REQ 4

        # REQ 3: Keep only top variable genes early
        if df.shape[0] > max_genes:
            top_genes = df.var(axis=1).nlargest(max_genes).index
            df = df.loc[top_genes]

        gc.collect()  # REQ 4
        return df, None

    except requests.exceptions.Timeout:
        return None, "❌ Request timed out. Try again."
    except MemoryError:
        return None, "❌ Still too large. Reduce max genes slider to 500."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────
# REQ 5: Vectorized Differential Expression (no for loops)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_differential_expression(df_json, group1_cols, group2_cols):
    """
    Fully vectorized DE analysis using numpy.
    No gene-by-gene for loop — processes all genes at once.
    """
    df = pd.read_json(io.StringIO(df_json))

    g1 = df[list(group1_cols)].values.astype("float32")
    g2 = df[list(group2_cols)].values.astype("float32")

    # Vectorized means
    mean1 = np.nanmean(g1, axis=1)
    mean2 = np.nanmean(g2, axis=1)

    # Vectorized fold change
    fold_change = (mean2 + 1e-9) / (mean1 + 1e-9)
    log2FC = np.log2(fold_change)

    # Vectorized t-test (scipy handles arrays)
    t_stat, pvalues = stats.ttest_ind(g1, g2, axis=1, nan_policy="omit")
    pvalues = np.clip(pvalues, 1e-300, 1.0)
    neg_log10_pval = -np.log10(pvalues)

    result_df = pd.DataFrame({
        "Gene": df.index,
        "Mean_Group1": np.round(mean1, 3),
        "Mean_Group2": np.round(mean2, 3),
        "Fold_Change": np.round(fold_change, 3),
        "Log2FC": np.round(log2FC, 3),
        "P_Value": pvalues,
        "Neg_Log10_Pvalue": np.round(neg_log10_pval, 3)
    })

    result_df["Significant"] = (
        (result_df["Log2FC"].abs() > 1) &
        (result_df["P_Value"] < 0.05)
    )
    result_df["Direction"] = np.select(
        [result_df["Log2FC"] > 1, result_df["Log2FC"] < -1],
        ["Upregulated", "Downregulated"],
        default="Not Significant"
    )

    gc.collect()  # REQ 4
    return result_df.sort_values("P_Value").reset_index(drop=True)


# ─────────────────────────────────────────────
# Pathway Enrichment
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pathway_enrichment(gene_tuple, database="KEGG_2021_Human"):
    try:
        enr = gp.enrichr(
            gene_list=list(gene_tuple),
            gene_sets=database,
            organism="human",
            outdir=None,
            verbose=False
        )
        results = enr.results[["Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]]
        results = results[results["P-value"] < 0.05].head(10)
        return results, None
    except Exception as e:
        return None, f"Pathway enrichment error: {str(e)}"


# ─────────────────────────────────────────────
# MAIN SHOW FUNCTION
# ─────────────────────────────────────────────
def show():
    st.title("🔬 Biomarker Discovery")
    st.markdown("Identify statistically significant leukemia biomarkers using differential expression analysis.")
    st.markdown("---")

    st.subheader("📂 Step 1: Load Data")

    input_method = st.radio(
        "Choose input method:",
        ["🌐 Fetch from GEO Database", "📁 Upload CSV File"],
        horizontal=True
    )

    df = None
    group1_cols = []
    group2_cols = []

    # ── GEO Fetch ──
    if "GEO" in input_method:
        st.info("💡 Examples: **GSE1432** (ALL/AML), **GSE13159** (Leukemia), **GSE2658** (Myeloma)")
        accession = st.text_input("Enter GEO Accession Number:")

        max_genes = st.slider(
            "⚙️ Max genes to load",
            min_value=500, max_value=5000, value=1000, step=500,
            help="Limits to most variable genes. Prevents memory crash."
        )

        if st.button("🔄 Fetch Dataset"):
            if accession:
                # REQ 6: Progress bar + spinner
                progress = st.progress(0)
                status = st.empty()

                status.text("📡 Connecting to NCBI GEO...")
                progress.progress(10)

                with st.spinner(f"Downloading {accession}... ⏳"):
                    progress.progress(30)
                    status.text("📥 Downloading series matrix...")
                    df, error = fetch_geo_data(accession, max_genes)
                    progress.progress(80)

                if error:
                    progress.empty()
                    status.empty()
                    st.error(error)
                    st.info("💡 Try reducing max genes or check the accession number.")
                else:
                    progress.progress(100)
                    status.text("✅ Done!")
                    st.session_state["geo_df"] = df
                    st.session_state["geo_accession"] = accession
                    st.session_state["current_dataset"] = accession
                    st.success(f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples")
                    if df.shape[0] == max_genes:
                        st.warning(f"⚠️ Auto-limited to top {max_genes} most variable genes.")
                    progress.empty()
                    status.empty()
            else:
                st.warning("Please enter an accession number.")

        if "geo_df" in st.session_state:
            df = st.session_state["geo_df"]
            st.info(f"📦 Active: **{st.session_state.get('geo_accession', 'GEO')}** — {df.shape[0]} genes × {df.shape[1]} samples")

    # ── CSV Upload ──
    else:
        uploaded_file = st.file_uploader("Upload gene expression CSV (rows=genes, columns=samples)", type=["csv"])
        if uploaded_file:
            # REQ 6: Progress feedback
            with st.spinner("Reading file..."):
                progress = st.progress(0)
                # REQ 3: Chunked CSV reading
                chunks = []
                chunk_iter = pd.read_csv(uploaded_file, index_col=0, chunksize=2000)
                for i, chunk in enumerate(chunk_iter):
                    chunk = chunk.apply(pd.to_numeric, errors="coerce")
                    chunk = chunk.astype("float32")  # REQ 2
                    chunks.append(chunk)
                    progress.progress(min((i + 1) * 20, 90))

                df = pd.concat(chunks)
                del chunks
                gc.collect()  # REQ 4
                progress.progress(100)
                progress.empty()

            st.success(f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples")

    # ── Data Loaded ──
    if df is not None and not df.empty:
        st.session_state.total_patients = df.shape[1]
        st.session_state.gene_features = f"{df.shape[0]:,}"
        st.session_state.data_loaded = True

        st.subheader("👁️ Data Preview")
        st.dataframe(df.head(5), use_container_width=True)
        st.markdown("---")

        st.subheader("👥 Step 2: Define Sample Groups")
        all_cols = list(df.columns)
        col1, col2 = st.columns(2)

        with col1:
            group1_name = st.text_input("Group 1 Name", value="ALL")
            group1_cols = st.multiselect("Group 1 Samples", all_cols, default=all_cols[:len(all_cols)//2])

        with col2:
            group2_name = st.text_input("Group 2 Name", value="AML")
            group2_cols = st.multiselect("Group 2 Samples", all_cols, default=all_cols[len(all_cols)//2:])

        if st.button("🚀 Run Differential Expression Analysis"):
            if not group1_cols or not group2_cols:
                st.warning("Please select samples for both groups.")
            else:
                # REQ 6: Progress bar for analysis
                progress = st.progress(0)
                status = st.empty()
                status.text("⚙️ Running vectorized t-tests...")
                progress.progress(20)

                with st.spinner("Analyzing... ⏳"):
                    # REQ 5: Pass JSON for caching compatibility
                    df_json = df.to_json()
                    progress.progress(50)
                    result_df = run_differential_expression(
                        df_json,
                        tuple(group1_cols),
                        tuple(group2_cols)
                    )
                    progress.progress(90)
                    st.session_state["de_results"] = result_df
                    st.session_state["group1_name"] = group1_name
                    st.session_state["group2_name"] = group2_name
                    st.session_state["analysis_done"] = True
                    gc.collect()  # REQ 4

                progress.progress(100)
                status.empty()
                progress.empty()
                st.success("✅ Analysis complete!")

    # ── Results ──
    if "de_results" in st.session_state:
        result_df = st.session_state["de_results"]
        group1_name = st.session_state.get("group1_name", "Group 1")
        group2_name = st.session_state.get("group2_name", "Group 2")

        sig_genes = result_df[result_df["Significant"]]
        up_genes = result_df[result_df["Direction"] == "Upregulated"]
        down_genes = result_df[result_df["Direction"] == "Downregulated"]

        st.markdown("---")
        st.subheader("📊 Step 3: Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Genes", len(result_df))
        m2.metric("Significant", len(sig_genes), delta="p<0.05 |log2FC|>1")
        m3.metric(f"Upregulated ({group2_name})", len(up_genes))
        m4.metric(f"Downregulated ({group2_name})", len(down_genes))

        # REQ 7: WebGL volcano plot for performance
        st.subheader("🌋 Volcano Plot")
        color_map = {
            "Upregulated": "#e74c3c",
            "Downregulated": "#2980b9",
            "Not Significant": "#bdc3c7"
        }

        # REQ 7: Use WebGL render mode when points > 10,000
        render_mode = "webgl" if len(result_df) > 10000 else "auto"

        fig = px.scatter(
            result_df,
            x="Log2FC",
            y="Neg_Log10_Pvalue",
            color="Direction",
            color_discrete_map=color_map,
            hover_name="Gene",
            hover_data={"Log2FC": ":.2f", "P_Value": ":.4f", "Direction": True},
            labels={"Log2FC": "Log2 Fold Change", "Neg_Log10_Pvalue": "-Log10(P-value)"},
            title=f"Volcano Plot: {group2_name} vs {group1_name}",
            template="plotly_white",
            height=550,
            render_mode=render_mode  # REQ 7
        )

        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray", annotation_text="p=0.05")
        fig.add_vline(x=1, line_dash="dash", line_color="gray")
        fig.add_vline(x=-1, line_dash="dash", line_color="gray")

        top_label = sig_genes.nsmallest(10, "P_Value")
        for _, row in top_label.iterrows():
            fig.add_annotation(
                x=row["Log2FC"], y=row["Neg_Log10_Pvalue"],
                text=row["Gene"], showarrow=True,
                arrowhead=2, font=dict(size=9)
            )

        st.plotly_chart(fig, use_container_width=True)
        gc.collect()  # REQ 4 after heavy plot

        st.subheader("🧬 Top 20 Significant Genes")
        st.dataframe(
            sig_genes[["Gene", "Log2FC", "P_Value", "Fold_Change", "Direction"]].head(20).reset_index(drop=True),
            use_container_width=True
        )

        # ── Pathway Enrichment ──
        st.markdown("---")
        st.subheader("🔗 Step 4: Pathway Enrichment")

        if len(sig_genes) > 0:
            gene_tuple = tuple(sig_genes["Gene"].head(100).tolist())

            if st.button("🔍 Run Pathway Enrichment"):
                with st.spinner("Querying Enrichr... ⏳"):
                    pathway_df, err = run_pathway_enrichment(gene_tuple)

                if err:
                    st.warning(err)
                elif pathway_df is not None and not pathway_df.empty:
                    st.session_state["pathway_results"] = pathway_df
                    st.success(f"✅ Found {len(pathway_df)} enriched pathways!")
                else:
                    st.info("No significant pathways found.")
        else:
            st.info("No significant genes found. Adjust group assignments.")

        if "pathway_results" in st.session_state:
            pathway_df = st.session_state["pathway_results"]
            st.dataframe(pathway_df[["Term", "Overlap", "P-value", "Genes"]].reset_index(drop=True), use_container_width=True)

            fig2 = px.bar(
                pathway_df.head(10),
                x=-np.log10(pathway_df["P-value"].head(10)),
                y="Term",
                orientation="h",
                color=-np.log10(pathway_df["P-value"].head(10)),
                color_continuous_scale="Reds",
                title="Top Enriched KEGG Pathways",
                template="plotly_white",
                height=450
            )
            fig2.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Downloads ──
        st.markdown("---")
        st.subheader("📥 Download Results")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ All DE Results (CSV)", result_df.to_csv(index=False),
                               "leukodash_DE_results.csv", "text/csv")
        with col2:
            if len(sig_genes) > 0:
                st.download_button("⬇️ Significant Genes (CSV)", sig_genes.to_csv(index=False),
                                   "leukodash_significant_genes.csv", "text/csv")

    elif df is None:
        st.info("👆 Load a dataset above to get started.")