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
# FIX 1: Clean error messages for invalid accession
# FIX 2: Sample subsetting for large datasets
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_geo_data(accession, max_genes=1000, max_samples=None):
    try:
        accession = accession.strip().upper()

        if not accession.startswith("GSE") or not accession[3:].isdigit():
            return None, "❌ Invalid accession format. Use format: GSE followed by numbers (e.g. GSE1432)"

        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:-3]}nnn/{accession}/matrix/{accession}_series_matrix.txt.gz"

        response = requests.get(url, timeout=60, stream=True)

        if response.status_code == 404:
            return None, "❌ Accession not found. This dataset may not exist or may be restricted/private. Please check your ID."
        elif response.status_code == 403:
            return None, "❌ Access denied. This dataset is restricted. Please use a public accession number."
        elif response.status_code != 200:
            return None, f"❌ Could not connect to GEO (HTTP {response.status_code}). Try again later."

        compressed = io.BytesIO(response.content)
        with gzip.open(compressed, "rt", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        data_lines = [l for l in lines if not l.startswith("!") and not l.startswith("#")]
        del lines
        gc.collect()

        if not data_lines:
            return None, "❌ Dataset found but contains no readable expression data. Try a different accession."

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

        # FIX 2: Sample subsetting
        if max_samples is not None and df.shape[1] > max_samples:
            df = df.iloc[:, :max_samples]

        # Gene limit
        if df.shape[0] > max_genes:
            top_genes = df.var(axis=1).nlargest(max_genes).index
            df = df.loc[top_genes]

        # FIX 4: Drop zero variance genes
        df = drop_zero_variance_genes(df)

        df = df.astype("float32")
        gc.collect()
        return df, None

    except requests.exceptions.Timeout:
        return None, "❌ Request timed out. GEO server is slow right now. Please try again in a moment."
    except requests.exceptions.ConnectionError:
        return None, "❌ No internet connection or GEO server is unreachable."
    except MemoryError:
        return None, "❌ Dataset too large for server memory. Reduce max genes to 500 or enable sample subsetting."
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}\n\nPlease check your accession number and try again."


# ─────────────────────────────────────────────
# FIX 4: Zero variance gene filter
# ─────────────────────────────────────────────
def drop_zero_variance_genes(df):
    variances = df.var(axis=1)
    zero_var_count = (variances == 0).sum()
    if zero_var_count > 0:
        df = df[variances > 0]
        st.info(f"🧹 Auto-removed {zero_var_count} zero-variance (dead) genes to ensure analysis quality.")
    return df


# ─────────────────────────────────────────────
# FIX 3: Missing label handler
# ─────────────────────────────────────────────
def clean_labels(labels, df_columns):
    label_series = pd.Series(labels, index=df_columns)
    missing_mask = label_series.isin(["", "nan", "NaN", "None", "N/A"]) | label_series.isna()
    missing_count = missing_mask.sum()
    if missing_count > 0:
        st.warning(f"⚠️ {missing_count} sample(s) removed due to missing classification labels.")
        label_series = label_series[~missing_mask]
    valid_cols = label_series.index.tolist()
    valid_labels = label_series.tolist()
    return valid_labels, valid_cols, missing_count


# ─────────────────────────────────────────────
# Vectorized DE Analysis
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_differential_expression(df_json, group1_cols, group2_cols):
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

    gc.collect()
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
# Report Generator
# ─────────────────────────────────────────────
def generate_report(result_df, accession, group1_name, group2_name):
    sig_genes = result_df[result_df["Significant"]]
    up = result_df[result_df["Direction"] == "Upregulated"]
    down = result_df[result_df["Direction"] == "Downregulated"]

    summary = pd.DataFrame({
        "Metric": [
            "Dataset", "Comparison",
            "Total Genes Tested", "Significant Genes",
            "Upregulated", "Downregulated"
        ],
        "Value": [
            accession, f"{group2_name} vs {group1_name}",
            len(result_df), len(sig_genes),
            len(up), len(down)
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
        1. **Load Data** — Fetch from GEO using accession number (e.g. GSE1432) or upload your own CSV
        2. **Define Groups** — Assign samples to Group 1 (ALL) and Group 2 (AML)
        3. **Run Analysis** — Differential expression using vectorized t-tests
        4. **Explore Results** — Volcano plot, significant genes table, pathway enrichment
        5. **Download Report** — Export results as CSV for use in papers or presentations

        **Supported accession formats:** GSE followed by numbers only (e.g. GSE1432, GSE13159)

        **CSV format:** Rows = genes, Columns = samples. First column = gene names/IDs.
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
            - Format must be **GSE** followed by numbers only
            - Examples: `GSE1432`, `GSE13159`, `GSE2658`
            - Dataset must be **public** (private datasets will be rejected cleanly)
            - If dataset is large (>500 samples), use the subsetting option below
            """)

        col_acc, col_genes = st.columns(2)
        with col_acc:
            accession = st.text_input("Enter GEO Accession Number:", placeholder="e.g. GSE1432")
        with col_genes:
            max_genes = st.slider("⚙️ Max genes", 500, 5000, 1000, 500,
                                  help="Limits to most variable genes to prevent crash")

        with st.expander("⚙️ Large Dataset Options"):
            enable_subset = st.checkbox("Enable sample subsetting (recommended for datasets >500 samples)")
            max_samples = None
            if enable_subset:
                max_samples = st.slider(
                    "Max samples to load",
                    min_value=50, max_value=500, value=100, step=50,
                    help="Loads only the first N samples for faster analysis"
                )
                st.info(f"ℹ️ Only the first **{max_samples}** samples will be analyzed.")

        if st.button("🔄 Fetch Dataset"):
            if not accession:
                st.warning("⚠️ Please enter an accession number.")
            elif not accession.strip().upper().startswith("GSE"):
                st.error("❌ Invalid format. Accession must start with 'GSE' (e.g. GSE1432). SRP, GPL formats are not supported.")
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
                    with st.expander("🛠️ Troubleshooting Tips"):
                        st.markdown("""
                        - Double-check the accession number at [GEO](https://www.ncbi.nlm.nih.gov/geo/)
                        - Make sure the dataset is **public**, not restricted
                        - Try a well-known dataset first: **GSE1432**
                        - If the dataset is very large, enable sample subsetting above
                        """)
                else:
                    st.session_state["geo_df"] = df
                    st.session_state["geo_accession"] = accession.strip().upper()
                    st.session_state["current_dataset"] = accession.strip().upper()
                    current_accession = accession.strip().upper()
                    st.success(f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples")
                    if enable_subset and max_samples:
                        st.info(f"📊 Showing first {max_samples} samples (subset mode active)")

        if "geo_df" in st.session_state:
            df = st.session_state["geo_df"]
            current_accession = st.session_state.get("geo_accession", "GEO")
            st.info(f"📦 Active: **{current_accession}** — {df.shape[0]} genes × {df.shape[1]} samples")

    # ── CSV Upload ──
    else:
        with st.expander("📋 CSV Format Requirements"):
            st.markdown("""
            - **Rows** = Genes (gene names or probe IDs as first column)
            - **Columns** = Samples (patient IDs as headers)
            - **Values** = Numeric expression values only
            - Missing values are handled automatically
            """)

        uploaded_file = st.file_uploader("Upload gene expression CSV", type=["csv"])
        if uploaded_file:
            with st.spinner("Reading file..."):
                progress = st.progress(0)
                chunks = []
                try:
                    for i, chunk in enumerate(pd.read_csv(uploaded_file, index_col=0, chunksize=2000)):
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
                    st.success(f"✅ Loaded: {df.shape[0]} genes × {df.shape[1]} samples")

                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Could not read file: {str(e)}\n\nMake sure your CSV has genes as rows and samples as columns.")

    # ── Data Loaded ──
    if df is not None and not df.empty:
        st.session_state.total_patients = df.shape[1]
        st.session_state.gene_features = f"{df.shape[0]:,}"
        st.session_state.data_loaded = True

        with st.expander("👁️ Data Preview"):
            st.dataframe(df.head(5), use_container_width=True)
            st.caption(f"Showing 5 of {df.shape[0]} genes × {df.shape[1]} samples")

        st.markdown("---")
        st.subheader("👥 Step 2: Define Sample Groups")
        st.markdown("Assign samples to Group 1 (e.g. ALL) and Group 2 (e.g. AML)")

        all_cols = list(df.columns)
        col1, col2 = st.columns(2)

        with col1:
            group1_name = st.text_input("Group 1 Name", value="ALL")
            group1_cols = st.multiselect("Group 1 Samples", all_cols,
                                         default=all_cols[:len(all_cols)//2])
        with col2:
            group2_name = st.text_input("Group 2 Name", value="AML")
            group2_cols = st.multiselect("Group 2 Samples", all_cols,
                                         default=all_cols[len(all_cols)//2:])

        overlap = set(group1_cols) & set(group2_cols)
        if overlap:
            st.error(f"❌ {len(overlap)} sample(s) assigned to both groups. Remove duplicates before proceeding.")

        if st.button("🚀 Run Differential Expression Analysis"):
            if not group1_cols or not group2_cols:
                st.warning("⚠️ Please select samples for both groups.")
            elif overlap:
                st.error("❌ Fix overlapping samples before running analysis.")
            else:
                progress = st.progress(0)
                status = st.empty()
                status.text("⚙️ Preparing data...")
                progress.progress(20)

                with st.spinner("Analyzing... ⏳"):
                    try:
                        df_json = df.to_json()
                        progress.progress(40)
                        status.text("🔬 Running vectorized t-tests...")

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
                        st.success("✅ Analysis complete!")

                    except Exception as e:
                        progress.empty()
                        status.empty()
                        st.error(f"❌ Analysis failed: {str(e)}")

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
        m3.metric(f"↑ {group2_name}", len(up_genes))
        m4.metric(f"↓ {group2_name}", len(down_genes))

        # ─────────────────────────────────────────────
        # ✅ NEW: Improved Volcano Plot
        # ─────────────────────────────────────────────
        st.subheader("🌋 Volcano Plot")

        with st.expander("ℹ️ How to read this plot"):
            st.markdown("""
            - **Red dots** = Upregulated genes in Group 2
            - **Blue dots** = Downregulated genes in Group 2
            - **Grey dots** = Not statistically significant
            - **Vertical dashed lines** = Log2 Fold Change threshold (±1)
            - **Horizontal dashed line** = p-value cutoff (p = 0.05)
            - **Top-right corner** = Significantly upregulated biomarkers
            - **Top-left corner** = Significantly downregulated biomarkers
            - **Gene labels** = Probe IDs from Affymetrix microarray (cross-reference via NCBI Gene)
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
            hover_data={"Log2FC": ":.2f", "P_Value": ":.4f", "Direction": True},
            labels={"Log2FC": "Log2 Fold Change", "Neg_Log10_Pvalue": "-Log10(P-value)"},
            title=f"Volcano Plot: {group2_name} vs {group1_name}",
            template="plotly_white",
            height=580,
            render_mode=render_mode
        )

        # ✅ NEW: Colored threshold lines
        fig.add_hline(
            y=-np.log10(0.05),
            line_dash="dash",
            line_color="gray",
            line_width=1.5,
            annotation_text="p = 0.05",
            annotation_position="right"
        )
        fig.add_vline(x=1, line_dash="dash", line_color="#e74c3c", line_width=1.2)
        fig.add_vline(x=-1, line_dash="dash", line_color="#2980b9", line_width=1.2)

        # ✅ NEW: Quadrant labels
        y_max = result_df["Neg_Log10_Pvalue"].max()
        x_max = result_df["Log2FC"].abs().max()

        fig.add_annotation(
            x=min(x_max * 0.75, 4),
            y=y_max * 0.92,
            text=f"⬆ Upregulated<br>in {group2_name}",
            showarrow=False,
            font=dict(size=11, color="#e74c3c"),
            bgcolor="rgba(231,76,60,0.08)",
            bordercolor="#e74c3c",
            borderwidth=1,
            borderpad=4
        )
        fig.add_annotation(
            x=max(-x_max * 0.75, -4),
            y=y_max * 0.92,
            text=f"⬇ Downregulated<br>in {group2_name}",
            showarrow=False,
            font=dict(size=11, color="#2980b9"),
            bgcolor="rgba(41,128,185,0.08)",
            bordercolor="#2980b9",
            borderwidth=1,
            borderpad=4
        )

        # ✅ NEW: Cleaned probe ID labels
        top_label = sig_genes.nsmallest(10, "P_Value")
        for _, row in top_label.iterrows():
            gene_name = str(row["Gene"])
            if "_at" in gene_name or "_s_at" in gene_name or "_x_at" in gene_name:
                display_name = f"Probe:{gene_name.split('_')[0]}"
            else:
                display_name = gene_name

            fig.add_annotation(
                x=row["Log2FC"],
                y=row["Neg_Log10_Pvalue"],
                text=display_name,
                showarrow=True,
                arrowhead=2,
                arrowwidth=1.5,
                arrowcolor="#555",
                bgcolor="white",
                bordercolor="#ccc",
                borderwidth=1,
                font=dict(size=9, color="#222"),
                opacity=0.9
            )

        fig.update_layout(
            legend_title_text="Expression Direction",
            legend=dict(
                bgcolor="white",
                bordercolor="#ddd",
                borderwidth=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        gc.collect()

        # Significant genes table
        with st.expander("🧬 View Top 20 Significant Genes"):
            st.dataframe(
                sig_genes[["Gene", "Log2FC", "P_Value", "Fold_Change", "Direction"]].head(20).reset_index(drop=True),
                use_container_width=True
            )

        # ─────────────────────────────────────────────
        # ✅ NEW: Biological Insight Box
        # ─────────────────────────────────────────────
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
                f"{group2_name} shows <b>massive suppression</b> of key genetic pathways compared to {group1_name}."
                if dominant == "Downregulated"
                else f"{group2_name} shows <b>overactivation</b> of key genetic pathways compared to {group1_name}."
            )

            st.markdown(f"""
            <div style='background:white; border-left:5px solid {dominant_color};
                        padding:20px; border-radius:10px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-top:10px;'>
                <h4 style='color:{dominant_color}; margin:0 0 12px 0;'>
                    📌 Key Finding: {dominant} genes dominate ({dominant_pct}% of significant genes)
                </h4>
                <p style='color:#444; font-size:15px; margin:0 0 16px 0;'>
                    {biological_meaning}
                </p>
                <table style='width:100%; font-size:14px; border-collapse:collapse;'>
                    <tr style='border-bottom:1px solid #eee;'>
                        <td style='padding:6px 0;'><b>Dataset</b></td>
                        <td style='padding:6px 0;'>{st.session_state.get("geo_accession", "Custom Upload")}</td>
                        <td style='padding:6px 0;'><b>Comparison</b></td>
                        <td style='padding:6px 0;'>{group2_name} vs {group1_name}</td>
                    </tr>
                    <tr style='border-bottom:1px solid #eee;'>
                        <td style='padding:6px 0;'><b>Total Genes Tested</b></td>
                        <td style='padding:6px 0;'>{len(result_df)}</td>
                        <td style='padding:6px 0;'><b>Significant Genes</b></td>
                        <td style='padding:6px 0;'>{total_sig}</td>
                    </tr>
                    <tr style='border-bottom:1px solid #eee;'>
                        <td style='padding:6px 0;'><b>Upregulated</b></td>
                        <td style='padding:6px 0;'>{total_up}</td>
                        <td style='padding:6px 0;'><b>Downregulated</b></td>
                        <td style='padding:6px 0;'>{total_down}</td>
                    </tr>
                    <tr>
                        <td style='padding:6px 0;'><b>Threshold</b></td>
                        <td style='padding:6px 0;'>|log2FC| > 1, p &lt; 0.05</td>
                        <td style='padding:6px 0;'><b>Test Used</b></td>
                        <td style='padding:6px 0;'>Independent t-test (vectorized)</td>
                    </tr>
                </table>
                <br>
                <p style='margin:0; font-size:12px; color:#888;'>
                    ℹ️ Gene labels showing Probe IDs (e.g. 204533_at) are Affymetrix microarray identifiers.
                    These map to specific genes on the human genome and can be cross-referenced
                    using the NCBI Gene database or the dataset platform annotation file (GPL).
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("No significant genes detected. Try adjusting your group assignments or thresholds.")

        # ── Pathway Enrichment ──
        st.markdown("---")
        st.subheader("🔗 Step 4: Pathway Enrichment Analysis")

        with st.expander("ℹ️ What is Pathway Enrichment?"):
            st.markdown("""
            Maps your significant genes to known biological pathways from the **KEGG database**.
            This tells you *which biological processes* are disrupted in leukemia based on your data.
            Results show pathways sorted by statistical significance (-log10 p-value).
            """)

        if len(sig_genes) > 0:
            gene_tuple = tuple(sig_genes["Gene"].head(100).tolist())
            if st.button("🔍 Run Pathway Enrichment"):
                with st.spinner("Querying Enrichr database... ⏳"):
                    try:
                        pathway_df, err = run_pathway_enrichment(gene_tuple)
                        if err:
                            st.warning(f"⚠️ {err}")
                        elif pathway_df is not None and not pathway_df.empty:
                            st.session_state["pathway_results"] = pathway_df
                            st.success(f"✅ Found {len(pathway_df)} enriched pathways!")
                        else:
                            st.info("No significantly enriched pathways found.")
                    except Exception as e:
                        st.error(f"❌ Pathway enrichment failed: {str(e)}")
        else:
            st.info("No significant genes found. Try adjusting your group assignments.")

        if "pathway_results" in st.session_state:
            pathway_df = st.session_state["pathway_results"]

            with st.expander("📋 View Pathway Table"):
                st.dataframe(
                    pathway_df[["Term", "Overlap", "P-value", "Genes"]].reset_index(drop=True),
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
        st.markdown("Export your findings — suitable for research papers, clinical reports, or presentations.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "⬇️ All DE Results (CSV)",
                data=result_df.to_csv(index=False),
                file_name="leukodash_DE_results.csv",
                mime="text/csv",
                help="Full differential expression table"
            )

        with col2:
            if len(sig_genes) > 0:
                st.download_button(
                    "⬇️ Significant Genes Only (CSV)",
                    data=sig_genes.to_csv(index=False),
                    file_name="leukodash_significant_genes.csv",
                    mime="text/csv",
                    help="Only statistically significant genes (p<0.05, |log2FC|>1)"
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
                help="Summary + significant genes — suitable for sharing with supervisors"
            )

    elif df is None:
        st.info("👆 Load a dataset above to get started.")