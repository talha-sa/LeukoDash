import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.stats import false_discovery_control
import gseapy as gp
import requests
import gzip
import gc
import io
import mygene

# ─────────────────────────────────────────────
# GEO FETCHER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_geo_data(accession, max_genes=1000, max_samples=None):
    try:
        accession = accession.strip().upper()

        if not accession.startswith("GSE") or not accession[3:].isdigit():
            return None, "❌ Invalid accession format. Use GSE followed by numbers (e.g. GSE1432)"

        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:-3]}nnn/{accession}/matrix/{accession}_series_matrix.txt.gz"
        response = requests.get(url, timeout=60, stream=True)

        if response.status_code == 404:
            return None, "❌ Accession not found. Dataset may not exist or may be private."
        elif response.status_code == 403:
            return None, "❌ Access denied. This dataset is restricted."
        elif response.status_code != 200:
            return None, f"❌ Could not connect to GEO (HTTP {response.status_code}). Try again later."

        compressed = io.BytesIO(response.content)
        with gzip.open(compressed, "rt", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        data_lines = [l for l in lines if not l.startswith("!") and not l.startswith("#")]
        del lines
        gc.collect()

        if not data_lines:
            return None, "❌ Dataset found but contains no readable expression data."

        raw = "".join(data_lines)
        del data_lines

        chunks = []
        chunk_iter = pd.read_csv(io.StringIO(raw), sep="\t", index_col=0, chunksize=2000)
        for chunk in chunk_iter:
            chunk = chunk.apply(pd.to_numeric, errors="coerce")
            chunk = chunk.dropna(how="all")
            chunk = chunk.astype("float32")
            chunks.append(chunk)

        df = pd.concat(chunks)
        del chunks
        gc.collect()

        if max_samples is not None and df.shape[1] > max_samples:
            df = df.iloc[:, :max_samples]

        if df.shape[0] > max_genes:
            top_genes = df.var(axis=1).nlargest(max_genes).index
            df = df.loc[top_genes]

        df = drop_zero_variance_genes(df)
        df = df.astype("float32")
        gc.collect()
        return df, None

    except requests.exceptions.Timeout:
        return None, "❌ Request timed out. Try again in a moment."
    except requests.exceptions.ConnectionError:
        return None, "❌ No internet connection or GEO server unreachable."
    except MemoryError:
        return None, "❌ Dataset too large. Reduce max genes to 500 or enable sample subsetting."
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"


# ─────────────────────────────────────────────
# ZERO VARIANCE FILTER
# ─────────────────────────────────────────────
def drop_zero_variance_genes(df):
    variances = df.var(axis=1)
    zero_var_count = (variances == 0).sum()
    if zero_var_count > 0:
        df = df[variances > 0]
        st.info(f"🧹 Auto-removed {zero_var_count} zero-variance genes.")
    return df


# ─────────────────────────────────────────────
# UPGRADE 3: NORMALIZATION CHECK
# ─────────────────────────────────────────────
def check_normalization(df):
    """
    Checks if data looks like log2 normalized microarray data.
    Raw RNA-seq counts typically go up to 100,000+.
    Log2 microarray data typically ranges from 4 to 16.
    """
    try:
        max_val = float(df.max().max())
        median_val = float(df.median().median())

        if max_val > 1000:
            st.warning(
                f"⚠️ Normalization Warning: Your data has a maximum value of "
                f"{round(max_val):,} which suggests raw count data. "
                f"LeukoDash is optimized for log2-normalized microarray expression values. "
                f"Raw RNA-seq counts will produce distorted volcano plots and unreliable t-test results. "
                f"Please apply log2 normalization before uploading."
            )
        else:
            st.success(
                f"✅ Data check passed — values look compatible with log2 normalized "
                f"microarray data (median: {round(median_val, 2)}, max: {round(max_val, 2)})."
            )
    except Exception:
        pass


# ─────────────────────────────────────────────
# MISSING LABEL HANDLER
# ─────────────────────────────────────────────
def clean_labels(labels, df_columns):
    label_series = pd.Series(labels, index=df_columns)
    missing_mask = label_series.isin(["", "nan", "NaN", "None", "N/A"]) | label_series.isna()
    missing_count = missing_mask.sum()
    if missing_count > 0:
        st.warning(f"⚠️ {missing_count} sample(s) removed due to missing labels.")
        label_series = label_series[~missing_mask]
    return label_series.tolist(), label_series.index.tolist(), missing_count


# ─────────────────────────────────────────────
# UPGRADE 2: VECTORIZED DE WITH FDR CORRECTION
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_differential_expression(df_json, group1_cols, group2_cols):
    """
    Fully vectorized t-test with Benjamini-Hochberg FDR correction.
    FDR correction prevents false positives from multiple testing.
    """
    df = pd.read_json(io.StringIO(df_json))
    g1 = df[list(group1_cols)].values.astype("float32")
    g2 = df[list(group2_cols)].values.astype("float32")

    mean1 = np.nanmean(g1, axis=1)
    mean2 = np.nanmean(g2, axis=1)
    fold_change = (mean2 + 1e-9) / (mean1 + 1e-9)
    log2FC = np.log2(fold_change)

    t_stat, pvalues = stats.ttest_ind(g1, g2, axis=1, nan_policy="omit")
    pvalues = np.clip(pvalues, 1e-300, 1.0)
    neg_log10_pval = -np.log10(pvalues)

    # ✅ UPGRADE 2: Benjamini-Hochberg FDR correction
    fdr_pvalues = false_discovery_control(pvalues, method="bh")
    fdr_pvalues = np.clip(fdr_pvalues, 1e-300, 1.0)

    result_df = pd.DataFrame({
        "Gene": df.index,
        "Mean_Group1": np.round(mean1, 3),
        "Mean_Group2": np.round(mean2, 3),
        "Fold_Change": np.round(fold_change, 3),
        "Log2FC": np.round(log2FC, 3),
        "P_Value": pvalues,
        "FDR_P_Value": fdr_pvalues,
        "Neg_Log10_Pvalue": np.round(neg_log10_pval, 3)
    })

    # ✅ Significance now based on FDR corrected p-value
    result_df["Significant"] = (
        (result_df["Log2FC"].abs() > 1) &
        (result_df["FDR_P_Value"] < 0.05)
    )
    result_df["Direction"] = np.select(
        [result_df["Log2FC"] > 1, result_df["Log2FC"] < -1],
        ["Upregulated", "Downregulated"],
        default="Not Significant"
    )

    gc.collect()
    return result_df.sort_values("FDR_P_Value").reset_index(drop=True)


# ─────────────────────────────────────────────
# UPGRADE 1: MYGENE PROBE TO SYMBOL MAPPING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_gene_mapping(probe_tuple):
    """
    Converts Affymetrix Probe IDs to HGNC Gene Symbols using MyGene.
    Only maps significant genes to avoid API rate limits.
    Falls back gracefully if mapping fails.
    """
    try:
        mg = mygene.MyGeneInfo()
        probe_list = list(probe_tuple)

        # Query mygene for symbol mapping
        result = mg.querymany(
            probe_list,
            scopes="reporter",
            fields="symbol",
            species="human",
            returnall=False,
            verbose=False
        )

        # Build mapping dictionary
        mapping = {}
        for item in result:
            query = item.get("query", "")
            symbol = item.get("symbol", None)
            if symbol and not item.get("notfound", False):
                mapping[query] = symbol

        return mapping

    except Exception:
        # If mapping fails return empty dict — probe IDs used as fallback
        return {}


# ─────────────────────────────────────────────
# PATHWAY ENRICHMENT
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pathway_enrichment(gene_tuple, database="KEGG_2021_Human"):
    try:
        # ✅ Convert all to string — fixes int has no attribute strip
        gene_list = [str(g).strip() for g in gene_tuple if str(g).strip() != ""]

        if not gene_list:
            return None, "No valid gene names to run enrichment."

        enr = gp.enrichr(
            gene_list=gene_list,
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
# REPORT GENERATOR
# ─────────────────────────────────────────────
def generate_report(result_df, accession, group1_name, group2_name):
    sig_genes = result_df[result_df["Significant"]]
    up = result_df[result_df["Direction"] == "Upregulated"]
    down = result_df[result_df["Direction"] == "Downregulated"]

    summary = pd.DataFrame({
        "Metric": [
            "Dataset", "Comparison",
            "Total Genes Tested", "Significant Genes (FDR)",
            "Upregulated", "Downregulated",
            "Statistical Method", "FDR Method"
        ],
        "Value": [
            accession, f"{group2_name} vs {group1_name}",
            len(result_df), len(sig_genes),
            len(up), len(down),
            "Independent t-test (vectorized)",
            "Benjamini-Hochberg FDR correction"
        ]
    })

    output = io.StringIO()
    output.write("=== LeukoDash Analysis Report ===\n\n")
    output.write("SUMMARY\n")
    summary.to_csv(output, index=False)
    output.write("\n\nSIGNIFICANT GENES\n")
    sig_genes.to_csv(output, index=False)
    return output.getvalue()


# ─────────────────────────────────────────────
# MAIN SHOW FUNCTION
# ─────────────────────────────────────────────
def show():
    st.title("🔬 Biomarker Discovery")
    st.markdown("Identify statistically significant leukemia biomarkers using differential expression analysis.")

    with st.expander("📖 How to use this module"):
        st.markdown("""
        1. **Load Data** — Fetch from GEO by accession number or upload CSV
        2. **Define Groups** — Assign samples to Group 1 (ALL) and Group 2 (AML)
        3. **Run Analysis** — Vectorized t-tests with FDR correction
        4. **Adjust Thresholds** — Live p-value and log2FC sliders
        5. **Map Gene Symbols** — Convert probe IDs to real gene names
        6. **Pathway Enrichment** — KEGG biological pathway analysis
        7. **Download** — Export all results as CSV

        **Accession format:** GSE followed by numbers only (e.g. GSE1432, GSE13159)
        """)

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
    current_accession = "Custom"

    # ── GEO Fetch ──
    if "GEO" in input_method:
        with st.expander("💡 GEO Accession Help"):
            st.markdown("""
            - Format: **GSE** followed by numbers only
            - Examples: `GSE1432`, `GSE13159`, `GSE2658`
            - Must be a **public** dataset
            - For large datasets (>500 samples) enable subsetting below
            """)

        # ✅ UPGRADE 4: Try Example button next to input
        col_acc, col_genes, col_example = st.columns([3, 2, 1])
        with col_acc:
            accession = st.text_input(
                "Enter GEO Accession Number:",
                placeholder="e.g. GSE1432"
            )
        with col_genes:
            max_genes = st.slider(
                "⚙️ Max genes", 500, 5000, 1000, 500,
                help="Limits to most variable genes"
            )
        with col_example:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🧪 Example"):
                accession = "GSE1432"
                st.info("Loaded example: GSE1432")

        with st.expander("⚙️ Large Dataset Options"):
            enable_subset = st.checkbox(
                "Enable sample subsetting (recommended for >500 samples)"
            )
            max_samples = None
            if enable_subset:
                max_samples = st.slider(
                    "Max samples", 50, 500, 100, 50,
                    help="Load only first N samples"
                )
                st.info(f"ℹ️ Only first **{max_samples}** samples will be loaded.")

        if st.button("🔄 Fetch Dataset"):
            if not accession:
                st.warning("⚠️ Please enter an accession number.")
            elif not accession.strip().upper().startswith("GSE"):
                st.error("❌ Format must start with GSE (e.g. GSE1432)")
            else:
                progress = st.progress(0)
                status = st.empty()
                status.text("📡 Connecting to NCBI GEO...")
                progress.progress(10)

                with st.spinner(f"Fetching {accession.strip().upper()}... ⏳"):
                    progress.progress(30)
                    status.text("📥 Downloading series matrix...")
                    df, error = fetch_geo_data(
                        accession,
                        max_genes=max_genes,
                        max_samples=max_samples if enable_subset else None
                    )
                    progress.progress(85)

                progress.empty()
                status.empty()

                if error:
                    st.error(error)
                    with st.expander("🛠️ Troubleshooting"):
                        st.markdown("""
                        - Verify accession at [GEO](https://www.ncbi.nlm.nih.gov/geo/)
                        - Must be a public dataset
                        - Try example dataset: **GSE1432**
                        - For large datasets enable sample subsetting
                        """)
                else:
                    st.session_state["geo_df"] = df
                    st.session_state["geo_accession"] = accession.strip().upper()
                    st.session_state["current_dataset"] = accession.strip().upper()
                    current_accession = accession.strip().upper()

                    # ✅ UPGRADE 3: Normalization check on load
                    check_normalization(df)

                    st.success(
                        f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples"
                    )
                    if enable_subset and max_samples:
                        st.info(f"📊 Subset mode: first {max_samples} samples only")

        if "geo_df" in st.session_state:
            df = st.session_state["geo_df"]
            current_accession = st.session_state.get("geo_accession", "GEO")
            st.info(
                f"📦 Active: **{current_accession}** — "
                f"{df.shape[0]} genes × {df.shape[1]} samples"
            )

    # ── CSV Upload ──
    else:
        with st.expander("📋 CSV Format Requirements"):
            st.markdown("""
            - **Rows** = Genes (names or probe IDs)
            - **Columns** = Samples (patient IDs)
            - **Values** = Numeric expression values
            - Missing values handled automatically
            """)

        uploaded_file = st.file_uploader(
            "Upload gene expression CSV", type=["csv"]
        )
        if uploaded_file:
            with st.spinner("Reading file..."):
                progress = st.progress(0)
                chunks = []
                try:
                    for i, chunk in enumerate(
                        pd.read_csv(uploaded_file, index_col=0, chunksize=2000)
                    ):
                        chunk = chunk.apply(pd.to_numeric, errors="coerce")
                        chunk = chunk.astype("float32")
                        chunks.append(chunk)
                        progress.progress(min((i + 1) * 20, 90))

                    df = pd.concat(chunks)
                    del chunks
                    df = drop_zero_variance_genes(df)
                    gc.collect()
                    progress.progress(100)
                    progress.empty()

                    # ✅ UPGRADE 3: Normalization check on upload
                    check_normalization(df)
                    st.success(
                        f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples"
                    )

                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Could not read file: {str(e)}")

    # ── Data Loaded ──
    if df is not None and not df.empty:
        st.session_state.total_patients = df.shape[1]
        st.session_state.gene_features = f"{df.shape[0]:,}"
        st.session_state.data_loaded = True

        # Memory usage indicator
        data_size_mb = round(
            df.memory_usage(deep=True).sum() / 1024 / 1024, 2
        )
        st.caption(f"📊 Dataset memory usage: {data_size_mb} MB")

        with st.expander("👁️ Data Preview"):
            st.dataframe(df.head(5), use_container_width=True)
            st.caption(
                f"Showing 5 of {df.shape[0]} genes × {df.shape[1]} samples"
            )

        st.markdown("---")
        st.subheader("👥 Step 2: Define Sample Groups")
        st.markdown("Assign samples to Group 1 (ALL) and Group 2 (AML)")

        all_cols = list(df.columns)
        col1, col2 = st.columns(2)

        with col1:
            group1_name = st.text_input("Group 1 Name", value="ALL")
            group1_cols = st.multiselect(
                "Group 1 Samples", all_cols,
                default=all_cols[:len(all_cols) // 2]
            )
        with col2:
            group2_name = st.text_input("Group 2 Name", value="AML")
            group2_cols = st.multiselect(
                "Group 2 Samples", all_cols,
                default=all_cols[len(all_cols) // 2:]
            )

        overlap = set(group1_cols) & set(group2_cols)
        if overlap:
            st.error(
                f"❌ {len(overlap)} sample(s) in both groups. "
                f"Remove duplicates before proceeding."
            )

        if st.button("🚀 Run Differential Expression Analysis"):
            if not group1_cols or not group2_cols:
                st.warning("⚠️ Please select samples for both groups.")
            elif overlap:
                st.error("❌ Fix overlapping samples first.")
            else:
                progress = st.progress(0)
                status = st.empty()
                status.text("⚙️ Preparing data...")
                progress.progress(20)

                with st.spinner("Analyzing... ⏳"):
                    try:
                        df_json = df.to_json()
                        progress.progress(40)
                        status.text(
                            "🔬 Running vectorized t-tests + FDR correction..."
                        )
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
                        st.session_state["current_accession_report"] = current_accession
                        gc.collect()

                        progress.progress(100)
                        status.empty()
                        progress.empty()
                        st.success(
                            "✅ Analysis complete! "
                            "Significance based on FDR corrected p-values."
                        )

                    except Exception as e:
                        progress.empty()
                        status.empty()
                        st.error(f"❌ Analysis failed: {str(e)}")

    # ── Results ──
    if "de_results" in st.session_state:
        result_df = st.session_state["de_results"].copy()
        group1_name = st.session_state.get("group1_name", "Group 1")
        group2_name = st.session_state.get("group2_name", "Group 2")

        st.markdown("---")
        st.subheader("📊 Step 3: Results")

        # FDR info box
        st.info(
            "✅ **Statistical Rigor:** Significance is based on "
            "**Benjamini-Hochberg FDR corrected p-values** "
            "(FDR < 0.05) to control false discovery rate across "
            "multiple gene tests."
        )

        # Threshold sliders
        with st.expander("⚙️ Adjust Significance Thresholds", expanded=True):
            st.markdown(
                "Move sliders to update cutoffs — "
                "volcano plot and counts update instantly."
            )
            sl1, sl2 = st.columns(2)
            with sl1:
                pval_threshold = st.slider(
                    "FDR P-value threshold",
                    min_value=0.001, max_value=0.1,
                    value=0.05, step=0.001, format="%.3f",
                    help="Applied to FDR corrected p-values"
                )
            with sl2:
                log2fc_threshold = st.slider(
                    "Log2 Fold Change threshold",
                    min_value=0.5, max_value=3.0,
                    value=1.0, step=0.1, format="%.1f",
                    help="Standard cutoff is 1.0 (2x fold change)"
                )
            st.info(
                f"📌 Thresholds: FDR p < **{pval_threshold}** "
                f"| |log2FC| > **{log2fc_threshold}**"
            )

        # Recalculate with slider values
        result_df["Significant"] = (
            (result_df["Log2FC"].abs() > log2fc_threshold) &
            (result_df["FDR_P_Value"] < pval_threshold)
        )
        result_df["Direction"] = np.select(
            [result_df["Log2FC"] > log2fc_threshold,
             result_df["Log2FC"] < -log2fc_threshold],
            ["Upregulated", "Downregulated"],
            default="Not Significant"
        )

        sig_genes = result_df[result_df["Significant"]]
        up_genes = result_df[result_df["Direction"] == "Upregulated"]
        down_genes = result_df[result_df["Direction"] == "Downregulated"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Genes", len(result_df))
        m2.metric("Significant (FDR)", len(sig_genes),
                  delta=f"FDR<{pval_threshold}")
        m3.metric(f"↑ {group2_name}", len(up_genes))
        m4.metric(f"↓ {group2_name}", len(down_genes))

        # ── Volcano Plot ──
        st.subheader("🌋 Volcano Plot")

        with st.expander("ℹ️ How to read this plot"):
            st.markdown("""
            - **Red dots** = Upregulated in Group 2
            - **Blue dots** = Downregulated in Group 2
            - **Grey dots** = Not significant after FDR correction
            - **Vertical lines** = Log2 Fold Change threshold
            - **Horizontal line** = FDR p-value cutoff
            - **Labels** = Probe IDs (use Gene Symbol Mapper below)
            """)

        color_map = {
            "Upregulated": "#e74c3c",
            "Downregulated": "#2980b9",
            "Not Significant": "#bdc3c7"
        }
        render_mode = "webgl" if len(result_df) > 10000 else "auto"

        fig = px.scatter(
            result_df,
            x="Log2FC",
            y="Neg_Log10_Pvalue",
            color="Direction",
            color_discrete_map=color_map,
            hover_name="Gene",
            hover_data={
                "Log2FC": ":.2f",
                "P_Value": ":.4f",
                "FDR_P_Value": ":.4f",
                "Direction": True
            },
            labels={
                "Log2FC": "Log2 Fold Change",
                "Neg_Log10_Pvalue": "-Log10(P-value)"
            },
            title=f"Volcano Plot: {group2_name} vs {group1_name}",
            template="plotly_white",
            height=580,
            render_mode=render_mode
        )

        fig.add_hline(
            y=-np.log10(pval_threshold),
            line_dash="dash", line_color="gray", line_width=1.5,
            annotation_text=f"FDR p = {pval_threshold}",
            annotation_position="right"
        )
        fig.add_vline(
            x=log2fc_threshold,
            line_dash="dash", line_color="#e74c3c", line_width=1.2
        )
        fig.add_vline(
            x=-log2fc_threshold,
            line_dash="dash", line_color="#2980b9", line_width=1.2
        )

        y_max = result_df["Neg_Log10_Pvalue"].max()
        x_max = result_df["Log2FC"].abs().max()

        fig.add_annotation(
            x=min(x_max * 0.75, 4), y=y_max * 0.92,
            text=f"⬆ Upregulated<br>in {group2_name}",
            showarrow=False,
            font=dict(size=11, color="#e74c3c"),
            bgcolor="rgba(231,76,60,0.08)",
            bordercolor="#e74c3c", borderwidth=1, borderpad=4
        )
        fig.add_annotation(
            x=max(-x_max * 0.75, -4), y=y_max * 0.92,
            text=f"⬇ Downregulated<br>in {group2_name}",
            showarrow=False,
            font=dict(size=11, color="#2980b9"),
            bgcolor="rgba(41,128,185,0.08)",
            bordercolor="#2980b9", borderwidth=1, borderpad=4
        )

        top_label = sig_genes.nsmallest(10, "FDR_P_Value")
        for _, row in top_label.iterrows():
            gene_name = str(row["Gene"])
            if "_at" in gene_name or "_s_at" in gene_name:
                display_name = f"Probe:{gene_name.split('_')[0]}"
            else:
                display_name = gene_name

            fig.add_annotation(
                x=row["Log2FC"], y=row["Neg_Log10_Pvalue"],
                text=display_name, showarrow=True,
                arrowhead=2, arrowwidth=1.5, arrowcolor="#555",
                bgcolor="white", bordercolor="#ccc", borderwidth=1,
                font=dict(size=9, color="#222"), opacity=0.9
            )

        fig.update_layout(
            legend_title_text="Expression Direction",
            legend=dict(bgcolor="white", bordercolor="#ddd", borderwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)
        gc.collect()

        # Significant genes table
        with st.expander("🧬 View Top 20 Significant Genes (FDR)"):
            display_cols = [
                "Gene", "Log2FC", "P_Value",
                "FDR_P_Value", "Fold_Change", "Direction"
            ]
            st.dataframe(
                sig_genes[display_cols].head(20).reset_index(drop=True),
                use_container_width=True
            )
            st.caption(
                "FDR_P_Value = Benjamini-Hochberg corrected p-value. "
                "Significance based on FDR < " + str(pval_threshold)
            )

        # ── Biological Insight Box ──
        st.markdown("---")
        st.subheader("🧠 Biological Interpretation")

        total_sig = len(sig_genes)
        total_up = len(up_genes)
        total_down = len(down_genes)

        if total_sig > 0:
            dominant = "Downregulated" if total_down > total_up else "Upregulated"
            dominant_count = max(total_down, total_up)
            dominant_pct = int((dominant_count / total_sig) * 100)
            dominant_color = "#2980b9" if dominant == "Downregulated" else "#e74c3c"
            biological_meaning = (
                f"{group2_name} shows <b>massive suppression</b> of key "
                f"genetic pathways compared to {group1_name}."
                if dominant == "Downregulated"
                else f"{group2_name} shows <b>overactivation</b> of key "
                f"genetic pathways compared to {group1_name}."
            )

            st.markdown(f"""
            <div style='background:white; border-left:5px solid {dominant_color};
                        padding:20px; border-radius:10px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-top:10px;'>
                <h4 style='color:{dominant_color}; margin:0 0 12px 0;'>
                    📌 Key Finding: {dominant} genes dominate
                    ({dominant_pct}% of FDR-significant genes)
                </h4>
                <p style='color:#444; font-size:15px; margin:0 0 16px 0;'>
                    {biological_meaning}
                </p>
                <table style='width:100%; font-size:14px; border-collapse:collapse;'>
                    <tr style='border-bottom:1px solid #eee;'>
                        <td style='padding:6px 0;'><b>Dataset</b></td>
                        <td style='padding:6px 0;'>
                            {st.session_state.get("geo_accession", "Custom")}
                        </td>
                        <td style='padding:6px 0;'><b>Comparison</b></td>
                        <td style='padding:6px 0;'>{group2_name} vs {group1_name}</td>
                    </tr>
                    <tr style='border-bottom:1px solid #eee;'>
                        <td style='padding:6px 0;'><b>Total Genes</b></td>
                        <td style='padding:6px 0;'>{len(result_df)}</td>
                        <td style='padding:6px 0;'><b>Significant (FDR)</b></td>
                        <td style='padding:6px 0;'>{total_sig}</td>
                    </tr>
                    <tr style='border-bottom:1px solid #eee;'>
                        <td style='padding:6px 0;'><b>Upregulated</b></td>
                        <td style='padding:6px 0;'>{total_up}</td>
                        <td style='padding:6px 0;'><b>Downregulated</b></td>
                        <td style='padding:6px 0;'>{total_down}</td>
                    </tr>
                    <tr style='border-bottom:1px solid #eee;'>
                        <td style='padding:6px 0;'><b>FDR Threshold</b></td>
                        <td style='padding:6px 0;'>p &lt; {pval_threshold}</td>
                        <td style='padding:6px 0;'><b>Log2FC Threshold</b></td>
                        <td style='padding:6px 0;'>|log2FC| &gt; {log2fc_threshold}</td>
                    </tr>
                    <tr>
                        <td style='padding:6px 0;'><b>Test Used</b></td>
                        <td style='padding:6px 0;'>Independent t-test (vectorized)</td>
                        <td style='padding:6px 0;'><b>Correction</b></td>
                        <td style='padding:6px 0;'>Benjamini-Hochberg FDR</td>
                    </tr>
                </table>
                <br>
                <p style='margin:0; font-size:12px; color:#888;'>
                    ℹ️ Probe IDs (e.g. 204533_at) are Affymetrix microarray identifiers.
                    Use the Gene Symbol Mapper below to convert to HGNC gene names.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ✅ UPGRADE 4: Top Biomarker Highlight Box
            st.markdown("<br>", unsafe_allow_html=True)
            top_gene_row = sig_genes.iloc[0]
            top_gene = str(top_gene_row["Gene"])
            top_fc = round(float(top_gene_row["Fold_Change"]), 2)
            top_fdr = round(float(top_gene_row["FDR_P_Value"]), 5)
            top_direction = top_gene_row["Direction"]
            direction_color = "#e74c3c" if top_direction == "Upregulated" else "#2980b9"

            st.markdown(f"""
            <div style='background:#f0faf0; border-left:5px solid #27ae60;
                        padding:16px; border-radius:10px; margin-top:8px;'>
                <h4 style='color:#27ae60; margin:0 0 8px 0;'>
                    🔬 Top Biomarker Identified
                </h4>
                <p style='margin:0; font-size:15px; color:#333;'>
                    Gene <code>{top_gene}</code> shows a
                    <b style='color:{direction_color};'>{top_fc}x expression difference</b>
                    between groups (FDR p = {top_fdr}) —
                    the strongest candidate for further biological investigation.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info(
                "No significant genes detected with current thresholds. "
                "Try relaxing the sliders above."
            )

        # ── UPGRADE 1: Gene Symbol Mapper ──
        st.markdown("---")
        st.subheader("🧬 Gene Symbol Mapper")

        with st.expander("ℹ️ What is this?"):
            st.markdown("""
            GEO datasets from Affymetrix microarrays use **Probe IDs** like `204533_at`
            instead of real gene names. This tool converts them to **HGNC Gene Symbols**
            like `HOXA9` or `MYC` using the MyGene.info database.

            Mapped gene symbols are then used for more accurate **Pathway Enrichment**
            since KEGG pathways are indexed by gene symbol not probe ID.
            """)

        if len(sig_genes) > 0:
            if st.button("🔄 Map Probe IDs to Gene Symbols"):
                with st.spinner(
                    "Querying MyGene.info API for gene symbols... ⏳"
                ):
                    probe_tuple = tuple(
                        str(g) for g in sig_genes["Gene"].head(100).tolist()
                    )
                    mapping = get_gene_mapping(probe_tuple)

                    if mapping:
                        mapped_count = len(mapping)
                        st.session_state["gene_mapping"] = mapping
                        st.success(
                            f"✅ Mapped {mapped_count} probe IDs to gene symbols."
                        )

                        # Show mapping table
                        mapping_df = pd.DataFrame(
                            list(mapping.items()),
                            columns=["Probe ID", "Gene Symbol"]
                        )
                        with st.expander(
                            f"📋 View {mapped_count} Mapped Genes"
                        ):
                            st.dataframe(
                                mapping_df.reset_index(drop=True),
                                use_container_width=True
                            )
                    else:
                        st.warning(
                            "⚠️ Could not map probe IDs. "
                            "This may happen if genes are already symbols "
                            "or if the API is temporarily unavailable. "
                            "Pathway enrichment will proceed with probe IDs."
                        )
        else:
            st.info(
                "Run analysis first to enable gene symbol mapping."
            )

        # ── Pathway Enrichment ──
        st.markdown("---")
        st.subheader("🔗 Step 4: Pathway Enrichment Analysis")

        with st.expander("ℹ️ What is Pathway Enrichment?"):
            st.markdown("""
            Maps your significant genes to known biological pathways
            from the **KEGG database** via Enrichr.
            This reveals which biological processes are disrupted in leukemia.
            **Run Gene Symbol Mapper first** for best results.
            """)

        if len(sig_genes) > 0:
            # ✅ UPGRADE 5: Use mapped symbols if available
            if "gene_mapping" in st.session_state and st.session_state["gene_mapping"]:
                mapping = st.session_state["gene_mapping"]
                raw_genes = sig_genes["Gene"].head(100).tolist()
                gene_list_for_enrichment = [
                    mapping.get(str(g), str(g)) for g in raw_genes
                ]
                st.info(
                    f"✅ Using **mapped gene symbols** for pathway enrichment "
                    f"({len(mapping)} genes mapped)."
                )
            else:
                gene_list_for_enrichment = [
                    str(g) for g in sig_genes["Gene"].head(100).tolist()
                ]
                st.caption(
                    "💡 Tip: Run Gene Symbol Mapper above first "
                    "for more accurate pathway results."
                )

            gene_tuple = tuple(gene_list_for_enrichment)

            if st.button("🔍 Run Pathway Enrichment"):
                with st.spinner("Querying Enrichr database... ⏳"):
                    try:
                        pathway_df, err = run_pathway_enrichment(gene_tuple)
                        if err:
                            st.warning(f"⚠️ {err}")
                        elif pathway_df is not None and not pathway_df.empty:
                            st.session_state["pathway_results"] = pathway_df
                            st.success(
                                f"✅ Found {len(pathway_df)} enriched pathways!"
                            )
                        else:
                            st.info(
                                "No significant pathways found. "
                                "Try running Gene Symbol Mapper first."
                            )
                    except Exception as e:
                        st.error(f"❌ Pathway enrichment failed: {str(e)}")
        else:
            st.info(
                "No significant genes found. "
                "Try relaxing thresholds using the sliders above."
            )

        if "pathway_results" in st.session_state:
            pathway_df = st.session_state["pathway_results"]

            with st.expander("📋 View Pathway Table"):
                st.dataframe(
                    pathway_df[["Term", "Overlap", "P-value", "Genes"]
                               ].reset_index(drop=True),
                    use_container_width=True
                )

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
            fig2.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Download Results ──
        st.markdown("---")
        st.subheader("📥 Download Results")
        st.markdown(
            "Export findings — suitable for research papers and presentations."
        )

        col1, col2, col3 = st.columns(3)

        # ✅ UPGRADE 4: All downloads use index=False
        with col1:
            st.download_button(
                "⬇️ All DE Results (CSV)",
                data=result_df.to_csv(index=False),
                file_name="leukodash_DE_results.csv",
                mime="text/csv",
                help="Full table with raw and FDR p-values"
            )

        with col2:
            if len(sig_genes) > 0:
                st.download_button(
                    "⬇️ Significant Genes (FDR CSV)",
                    data=sig_genes.to_csv(index=False),
                    file_name="leukodash_significant_genes.csv",
                    mime="text/csv",
                    help="FDR-significant genes only"
                )

        with col3:
            report_data = generate_report(
                result_df,
                st.session_state.get("current_accession_report", "Custom"),
                group1_name,
                group2_name
            )
            st.download_button(
                "⬇️ Full Analysis Report (CSV)",
                data=report_data,
                file_name="leukodash_full_report.csv",
                mime="text/csv",
                help="Summary + significant genes + FDR stats"
            )

        # Gene symbol mapping download
        if "gene_mapping" in st.session_state and st.session_state["gene_mapping"]:
            mapping_df = pd.DataFrame(
                list(st.session_state["gene_mapping"].items()),
                columns=["Probe_ID", "Gene_Symbol"]
            )
            st.download_button(
                "⬇️ Gene Symbol Mapping (CSV)",
                data=mapping_df.to_csv(index=False),
                file_name="leukodash_gene_mapping.csv",
                mime="text/csv",
                help="Probe ID to HGNC Gene Symbol mapping table"
            )

    elif df is None:
        st.info("👆 Load a dataset above to get started.")