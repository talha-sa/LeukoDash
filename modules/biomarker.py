import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import GEOparse
import requests

def run():
    st.header("🔬 Biomarker Discovery")
    st.markdown("Identify differentially expressed genes between ALL and AML leukemia subtypes.")

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Biomarker Thresholds")
    pval_threshold = st.sidebar.slider(
        "P-value Threshold", min_value=0.01, max_value=0.10,
        value=0.05, step=0.01,
        help="Genes with p-value below this are considered significant."
    )
    lfc_threshold = st.sidebar.slider(
        "Log₂ Fold Change Threshold", min_value=0.0, max_value=2.0,
        value=1.0, step=0.1,
        help="Minimum absolute log2 fold change to flag a gene as a biomarker."
    )

    # ── Dataset Input ─────────────────────────────────────────────────────────
    st.subheader("📥 Load GEO Dataset")
    geo_id = st.text_input("Enter GEO Accession ID", value="GSE13159",
                           help="Example: GSE13159 (Golub Leukemia Dataset)")

    if st.button("🔍 Fetch & Analyse", type="primary"):
        with st.spinner("Fetching dataset from NCBI GEO…"):
            try:
                gse = GEOparse.get_GEO(geo=geo_id, destdir="./data/", silent=True)
            except Exception:
                st.error(
                    "Dataset not found or connection timed out. "
                    "Please try a different ID."
                )
                return

        st.success(f"✅ Loaded **{geo_id}** — {gse.metadata.get('title', [''])[0]}")

        # ── Extract expression matrix ─────────────────────────────────────────
        try:
            pivoted = gse.pivot_samples("VALUE")
        except Exception:
            st.error("Could not parse expression values from this dataset.")
            return

        pivoted = pivoted.apply(pd.to_numeric, errors="coerce").dropna()

        # ── Sample classification (ALL vs AML) ───────────────────────────────
        all_samples, aml_samples = [], []
        for gsm_name, gsm in gse.gsms.items():
            title = gsm.metadata.get("title", [""])[0].upper()
            char = " ".join(gsm.metadata.get("characteristics_ch1", [])).upper()
            combined = title + " " + char
            if "ALL" in combined:
                all_samples.append(gsm_name)
            elif "AML" in combined:
                aml_samples.append(gsm_name)

        all_samples = [s for s in all_samples if s in pivoted.columns]
        aml_samples = [s for s in aml_samples if s in pivoted.columns]

        if len(all_samples) < 2 or len(aml_samples) < 2:
            st.warning(
                f"Found {len(all_samples)} ALL and {len(aml_samples)} AML samples. "
                "Need at least 2 of each for analysis."
            )
            return

        st.info(f"🧬 Samples: **{len(all_samples)} ALL** | **{len(aml_samples)} AML**")

        # ── Differential Expression ───────────────────────────────────────────
        all_data = pivoted[all_samples]
        aml_data = pivoted[aml_samples]

        results = []
        for gene in pivoted.index:
            a = all_data.loc[gene].values.astype(float)
            b = aml_data.loc[gene].values.astype(float)
            if np.std(a) == 0 and np.std(b) == 0:
                continue
            try:
                _, p = stats.ttest_ind(a, b)
                mean_a = np.mean(a)
                mean_b = np.mean(b)
                with np.errstate(divide='ignore', invalid='ignore'):
                    lfc = np.log2(mean_a + 1) - np.log2(mean_b + 1)
                results.append({"Gene": gene, "Log2FC": lfc, "P_value": p})
            except Exception:
                continue

        df = pd.DataFrame(results).dropna()
        df["-log10(p)"] = -np.log10(df["P_value"].clip(lower=1e-300))

        # ── Significant genes ─────────────────────────────────────────────────
        sig_mask = (df["P_value"] < pval_threshold) & (df["Log2FC"].abs() >= lfc_threshold)
        sig_df = df[sig_mask].copy()

        st.subheader("📊 Results Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Genes Tested", len(df))
        col2.metric("Significant Genes", len(sig_df))
        col3.metric("% Significant", f"{100*len(sig_df)/max(len(df),1):.1f}%")

        # ── Volcano Plot ──────────────────────────────────────────────────────
        st.subheader("🌋 Volcano Plot")

        colors = np.where(sig_mask.values, "red", "lightgray")
        hover_text = [
            f"<b>{g}</b><br>Log2FC: {l:.3f}<br>p-value: {p:.2e}"
            for g, l, p in zip(df["Gene"], df["Log2FC"], df["P_value"])
        ]

        fig = go.Figure()

        # Non-significant dots
        ns_mask = ~sig_mask
        fig.add_trace(go.Scatter(
            x=df.loc[ns_mask, "Log2FC"],
            y=df.loc[ns_mask, "-log10(p)"],
            mode="markers",
            marker=dict(color="lightgray", size=4, opacity=0.6),
            text=[hover_text[i] for i in df.index[ns_mask]],
            hoverinfo="text",
            name="Non-significant"
        ))

        # Significant dots
        if sig_mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[sig_mask, "Log2FC"],
                y=df.loc[sig_mask, "-log10(p)"],
                mode="markers",
                marker=dict(color="red", size=6, opacity=0.85),
                text=[hover_text[i] for i in df.index[sig_mask]],
                hoverinfo="text",
                name="Significant"
            ))

        # Threshold lines
        pval_line = -np.log10(pval_threshold)
        fig.add_hline(y=pval_line, line_dash="dash", line_color="navy",
                      annotation_text=f"p = {pval_threshold}", annotation_position="top right")
        fig.add_vline(x=lfc_threshold,  line_dash="dot", line_color="green", opacity=0.5)
        fig.add_vline(x=-lfc_threshold, line_dash="dot", line_color="green", opacity=0.5)

        fig.update_layout(
            title="Volcano Plot — ALL vs AML",
            xaxis_title="Log₂ Fold Change",
            yaxis_title="-log₁₀(P-value)",
            template="plotly_white",
            height=480,
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Top 20 Genes Table with Gene Info Links ───────────────────────────
        st.subheader(f"🏆 Top 20 Differentially Expressed Genes")
        st.caption(f"Filtered: p < {pval_threshold} and |Log2FC| ≥ {lfc_threshold}")

        if sig_df.empty:
            st.warning(
                "No genes met the current thresholds. "
                "Try loosening the p-value or Log2FC sliders on the left."
            )
        else:
            top20 = (
                sig_df.reindex(sig_df["P_value"].abs().sort_values().index)
                .head(20)
                .reset_index(drop=True)
            )

            # Add NCBI Gene Info hyperlink column
            top20["NCBI Gene Link"] = top20["Gene"].apply(
                lambda g: f"[🔗 Search NCBI](https://www.ncbi.nlm.nih.gov/gene/?term={g})"
            )
            top20["Log2FC"]   = top20["Log2FC"].round(4)
            top20["P_value"]  = top20["P_value"].apply(lambda x: f"{x:.2e}")

            display_cols = ["Gene", "Log2FC", "P_value", "NCBI Gene Link"]
            st.markdown(top20[display_cols].to_markdown(index=False),
                        unsafe_allow_html=True)

            st.info(
                "💡 **Gene Info:** Click any 🔗 link above to search that gene on NCBI — "
                "you'll find its official name, function, and clinical relevance."
            )

        # ── CSV Export ────────────────────────────────────────────────────────
        st.subheader("📤 Export Significant Genes")
        if not sig_df.empty:
            csv_data = sig_df[["Gene", "Log2FC", "P_value"]].to_csv(index=False)
            st.download_button(
                label="⬇️ Download Significant Genes as CSV",
                data=csv_data,
                file_name=f"{geo_id}_significant_genes.csv",
                mime="text/csv",
                help="Download the full list of statistically significant genes."
            )
        else:
            st.info("Adjust thresholds to get significant genes before exporting.")