import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import GEOparse
import gseapy as gp
import os
import tempfile

# ─────────────────────────────────────────────
# HELPER: Parse GEO dataset into expression df
# ─────────────────────────────────────────────
def fetch_geo_data(accession):
    """
    Downloads a GEO dataset by accession number (e.g. GSE1432)
    and returns a cleaned expression DataFrame.
    Rows = genes, Columns = samples
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            gse = GEOparse.get_GEO(geo=accession.strip(), destdir=tmpdir, silent=True)

            # Get the first platform's gene table
            gpl_name = list(gse.gpls.keys())[0]
            gpl = gse.gpls[gpl_name]

            # Build expression matrix from all GSM samples
            expr_data = {}
            for gsm_name, gsm in gse.gsms.items():
                if gsm.table is not None and not gsm.table.empty:
                    # Use ID_REF as index and VALUE as expression
                    sample_df = gsm.table.set_index("ID_REF")["VALUE"]
                    expr_data[gsm_name] = pd.to_numeric(sample_df, errors="coerce")

            if not expr_data:
                return None, "Could not extract expression data from this GEO dataset."

            expr_df = pd.DataFrame(expr_data)

            # Try to map probe IDs to gene symbols using platform table
            if "Gene Symbol" in gpl.table.columns:
                gene_map = gpl.table.set_index("ID")["Gene Symbol"]
                expr_df.index = expr_df.index.map(lambda x: gene_map.get(x, x))
            elif "GENE_ASSIGNMENT" in gpl.table.columns:
                gene_map = gpl.table.set_index("ID")["GENE_ASSIGNMENT"]
                expr_df.index = expr_df.index.map(lambda x: str(gene_map.get(x, x)).split("//")[1].strip() if "//" in str(gene_map.get(x, x)) else x)

            expr_df = expr_df.dropna(how="all")
            expr_df = expr_df[~expr_df.index.duplicated(keep="first")]

            return expr_df, None

    except Exception as e:
        return None, f"Error fetching GEO data: {str(e)}"


# ─────────────────────────────────────────────
# HELPER: Differential Expression Analysis
# ─────────────────────────────────────────────
def run_differential_expression(df, group1_cols, group2_cols):
    """
    Runs t-test between two groups.
    Returns DataFrame with: gene, fold_change, log2FC, pvalue, neg_log10_pval, significant
    """
    results = []

    for gene in df.index:
        g1 = pd.to_numeric(df.loc[gene, group1_cols], errors="coerce").dropna()
        g2 = pd.to_numeric(df.loc[gene, group2_cols], errors="coerce").dropna()

        if len(g1) < 2 or len(g2) < 2:
            continue

        mean1 = g1.mean()
        mean2 = g2.mean()

        # Fold change: Group2 vs Group1
        fold_change = (mean2 + 1e-9) / (mean1 + 1e-9)
        log2FC = np.log2(fold_change)

        # T-test
        t_stat, pvalue = stats.ttest_ind(g1, g2)
        pvalue = max(pvalue, 1e-300)  # avoid log(0)
        neg_log10_pval = -np.log10(pvalue)

        results.append({
            "Gene": str(gene),
            "Mean_Group1": round(mean1, 3),
            "Mean_Group2": round(mean2, 3),
            "Fold_Change": round(fold_change, 3),
            "Log2FC": round(log2FC, 3),
            "P_Value": pvalue,
            "Neg_Log10_Pvalue": round(neg_log10_pval, 3)
        })

    result_df = pd.DataFrame(results)

    # Mark significant genes: |log2FC| > 1 AND p-value < 0.05
    result_df["Significant"] = (
        (result_df["Log2FC"].abs() > 1) &
        (result_df["P_Value"] < 0.05)
    )
    result_df["Direction"] = result_df["Log2FC"].apply(
        lambda x: "Upregulated" if x > 1 else ("Downregulated" if x < -1 else "Not Significant")
    )

    return result_df.sort_values("P_Value")


# ─────────────────────────────────────────────
# HELPER: Pathway Enrichment via Enrichr
# ─────────────────────────────────────────────
def run_pathway_enrichment(gene_list, database="KEGG_2021_Human"):
    """
    Takes a list of significant gene names and runs Enrichr pathway analysis.
    Returns top enriched pathways.
    """
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=database,
            organism="Human",
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

    # ── Data Input Section ──
    st.subheader("📂 Step 1: Load Data")

    input_method = st.radio(
        "Choose input method:",
        ["🌐 Fetch from GEO Database (Enter Accession ID)", "📁 Upload CSV File Manually"],
        horizontal=True
    )

    df = None
    group1_cols = []
    group2_cols = []

    # ── GEO Fetch ──
    if "GEO" in input_method:
        st.info("💡 Example accession numbers: **GSE1432** (ALL/AML), **GSE2658** (Multiple Myeloma)")
        accession = st.text_input("Enter GEO Accession Number (e.g. GSE1432):")

        if st.button("🔄 Fetch Dataset"):
            if accession:
                with st.spinner(f"Downloading {accession} from GEO... this may take a minute ⏳"):
                    df, error = fetch_geo_data(accession)
                    if error:
                        st.error(error)
                    else:
                        st.session_state["geo_df"] = df
                        st.success(f"✅ Dataset loaded! Shape: {df.shape[0]} genes × {df.shape[1]} samples")
            else:
                st.warning("Please enter an accession number.")

        if "geo_df" in st.session_state:
            df = st.session_state["geo_df"]

    # ── CSV Upload ──
    else:
        uploaded_file = st.file_uploader("Upload gene expression CSV (rows = genes, columns = samples)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file, index_col=0)
            st.success(f"✅ File loaded! Shape: {df.shape[0]} genes × {df.shape[1]} samples")

    # ── If data is loaded ──
    if df is not None and not df.empty:

        # Update session state counters for home page
        st.session_state.total_patients = df.shape[1]
        st.session_state.gene_features = f"{df.shape[0]:,}"
        st.session_state.data_loaded = True

        st.subheader("👁️ Data Preview")
        st.dataframe(df.head(5), use_container_width=True)

        st.markdown("---")

        # ── Group Assignment ──
        st.subheader("👥 Step 2: Define Sample Groups")
        st.markdown("Select which samples belong to **Group 1** (e.g. ALL) and **Group 2** (e.g. AML)")

        all_cols = list(df.columns)
        col1, col2 = st.columns(2)

        with col1:
            group1_name = st.text_input("Group 1 Name", value="ALL")
            group1_cols = st.multiselect("Select Group 1 Samples", all_cols, default=all_cols[:len(all_cols)//2])

        with col2:
            group2_name = st.text_input("Group 2 Name", value="AML")
            group2_cols = st.multiselect("Select Group 2 Samples", all_cols, default=all_cols[len(all_cols)//2:])

        # ── Run Analysis ──
        if st.button("🚀 Run Differential Expression Analysis"):
            if not group1_cols or not group2_cols:
                st.warning("Please select samples for both groups.")
            else:
                with st.spinner("Running analysis... ⏳"):
                    result_df = run_differential_expression(df, group1_cols, group2_cols)
                    st.session_state["de_results"] = result_df
                    st.session_state["group1_name"] = group1_name
                    st.session_state["group2_name"] = group2_name
                st.success("✅ Analysis complete!")

    # ── Show Results ──
    if "de_results" in st.session_state:
        result_df = st.session_state["de_results"]
        group1_name = st.session_state.get("group1_name", "Group 1")
        group2_name = st.session_state.get("group2_name", "Group 2")

        sig_genes = result_df[result_df["Significant"]]
        up_genes = result_df[result_df["Direction"] == "Upregulated"]
        down_genes = result_df[result_df["Direction"] == "Downregulated"]

        st.markdown("---")
        st.subheader("📊 Step 3: Results Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Genes Tested", len(result_df))
        m2.metric("Significant Genes", len(sig_genes), delta=f"p<0.05, |log2FC|>1")
        m3.metric(f"Upregulated in {group2_name}", len(up_genes))
        m4.metric(f"Downregulated in {group2_name}", len(down_genes))

        # ── Volcano Plot ──
        st.subheader("🌋 Volcano Plot")
        st.markdown("Each dot = one gene. Red = upregulated, Blue = downregulated, Grey = not significant.")

        color_map = {
            "Upregulated": "#e74c3c",
            "Downregulated": "#2980b9",
            "Not Significant": "#bdc3c7"
        }

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
            height=550
        )

        # Add threshold lines
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray", annotation_text="p=0.05")
        fig.add_vline(x=1, line_dash="dash", line_color="gray")
        fig.add_vline(x=-1, line_dash="dash", line_color="gray")

        # Label top 10 significant genes
        top_genes = sig_genes.nsmallest(10, "P_Value")
        for _, row in top_genes.iterrows():
            fig.add_annotation(
                x=row["Log2FC"], y=row["Neg_Log10_Pvalue"],
                text=row["Gene"], showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1,
                font=dict(size=9)
            )

        st.plotly_chart(fig, use_container_width=True)

        # ── Top Significant Genes Table ──
        st.subheader("🧬 Top 20 Significant Genes")
        display_cols = ["Gene", "Log2FC", "P_Value", "Fold_Change", "Direction"]
        st.dataframe(
            sig_genes[display_cols].head(20).reset_index(drop=True),
            use_container_width=True
        )

        # ── Pathway Enrichment ──
        st.markdown("---")
        st.subheader("🔗 Step 4: Pathway Enrichment Analysis")
        st.markdown("Maps your significant genes to known biological pathways (KEGG database).")

        if len(sig_genes) > 0:
            gene_list = sig_genes["Gene"].head(100).tolist()

            if st.button("🔍 Run Pathway Enrichment"):
                with st.spinner("Querying Enrichr database... ⏳"):
                    pathway_df, err = run_pathway_enrichment(gene_list)

                if err:
                    st.warning(f"Pathway enrichment issue: {err}")
                elif pathway_df is not None and not pathway_df.empty:
                    st.session_state["pathway_results"] = pathway_df
                    st.success(f"✅ Found {len(pathway_df)} enriched pathways!")
                else:
                    st.info("No significantly enriched pathways found.")
        else:
            st.info("No significant genes found. Try adjusting your group assignments.")

        if "pathway_results" in st.session_state:
            pathway_df = st.session_state["pathway_results"]

            st.dataframe(pathway_df[["Term", "Overlap", "P-value", "Genes"]].reset_index(drop=True), use_container_width=True)

            # Bar chart of top pathways
            fig2 = px.bar(
                pathway_df.head(10),
                x=-np.log10(pathway_df["P-value"].head(10)),
                y="Term",
                orientation="h",
                color=-np.log10(pathway_df["P-value"].head(10)),
                color_continuous_scale="Reds",
                labels={"x": "-Log10(P-value)", "y": "Pathway"},
                title="Top Enriched KEGG Pathways",
                template="plotly_white",
                height=450
            )
            fig2.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Download Results ──
        st.markdown("---")
        st.subheader("📥 Download Results")

        col1, col2 = st.columns(2)
        with col1:
            csv1 = result_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download All DE Results (CSV)",
                data=csv1,
                file_name="leukodash_DE_results.csv",
                mime="text/csv"
            )
        with col2:
            if len(sig_genes) > 0:
                csv2 = sig_genes.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Significant Genes Only (CSV)",
                    data=csv2,
                    file_name="leukodash_significant_genes.csv",
                    mime="text/csv"
                )

    elif df is None:
        st.info("👆 Please load a dataset above to get started.")