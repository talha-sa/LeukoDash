import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="LeukoDash",
    page_icon="🩸",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    .stButton>button {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        border-radius: 25px;
        padding: 0.5em 2em;
        font-size: 16px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #c0392b, #e74c3c);
        transform: scale(1.05);
    }
    .card-red {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(231,76,60,0.4);
    }
    .card-blue {
        background: linear-gradient(135deg, #2980b9, #1a5276);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(41,128,185,0.4);
    }
    .card-green {
        background: linear-gradient(135deg, #27ae60, #1e8449);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(39,174,96,0.4);
    }
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-top: 4px solid #e74c3c;
    }
    .stat-card-active {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(231,76,60,0.25);
        border-top: 4px solid #27ae60;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 20px rgba(39,174,96,0.25); }
        50% { box-shadow: 0 4px 30px rgba(39,174,96,0.5); }
        100% { box-shadow: 0 4px 20px rgba(39,174,96,0.25); }
    }
    .stat-number {
        font-size: 36px;
        font-weight: bold;
        color: #e74c3c;
    }
    .stat-number-active {
        font-size: 36px;
        font-weight: bold;
        color: #27ae60;
    }
    .stat-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    .banner {
        background: linear-gradient(135deg, #1a1a2e, #e74c3c);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .footer {
        background: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-top: 40px;
    }
    .status-badge-active {
        background: #27ae60;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-inactive {
        background: #e74c3c;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        display: inline-block;
    }
    .activity-card {
        background: white;
        border-left: 4px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.06);
    }
    .activity-card-green {
        background: white;
        border-left: 4px solid #27ae60;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.06);
    }
    .progress-bar-container {
        background: #f0f2f6;
        border-radius: 10px;
        height: 10px;
        margin-top: 8px;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #e74c3c, #27ae60);
        border-radius: 10px;
        height: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Session State Init ──
if "total_patients" not in st.session_state:
    st.session_state.total_patients = 0
if "gene_features" not in st.session_state:
    st.session_state.gene_features = "0"
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "model_accuracy" not in st.session_state:
    st.session_state["model_accuracy"] = "N/A"
if "current_dataset" not in st.session_state:
    st.session_state["current_dataset"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "survival_done" not in st.session_state:
    st.session_state["survival_done"] = False
if "activity_log" not in st.session_state:
    st.session_state["activity_log"] = []

# Sidebar
st.sidebar.markdown("<h1 style='text-align:center; color:white;'>🩸</h1>", unsafe_allow_html=True)
st.sidebar.title("LeukoDash")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "🔬 Biomarker Discovery",
    "📊 Gene Expression",
    "📈 Survival Prediction"
])

# ── Live status in sidebar ──
st.sidebar.markdown("---")
if st.session_state.data_loaded:
    st.sidebar.markdown("""
        <div style='background:#27ae60; padding:10px; border-radius:8px; text-align:center;'>
            <b>✅ Dataset Loaded</b>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <div style='margin-top:8px; font-size:13px;'>
            📦 <b>Dataset:</b> {st.session_state.get('current_dataset', 'Custom')}<br>
            👥 <b>Samples:</b> {st.session_state.total_patients}<br>
            🧬 <b>Genes:</b> {st.session_state.gene_features}
        </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
        <div style='background:#e74c3c; padding:10px; border-radius:8px; text-align:center;'>
            <b>⚠️ No Dataset Loaded</b>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.info("Go to Biomarker Discovery to load a dataset.")

# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if page == "🏠 Home":

    # Banner
    st.markdown("""
    <div class='banner'>
        <h1>🩸 LeukoDash</h1>
        <h3>A Bioinformatics Dashboard for Leukemia Analysis</h3>
        <p>Empowering researchers with AI-driven gene expression analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Live Status Bar ──
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        if st.session_state.data_loaded:
            st.markdown("""
                <div style='background:#eafaf1; border:1px solid #27ae60; border-radius:10px; padding:12px; text-align:center;'>
                    <span class='status-badge-active'>🟢 Dataset Active</span>
                    <p style='margin:6px 0 0 0; font-size:13px; color:#555;'>Data ready for analysis</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background:#fdf2f2; border:1px solid #e74c3c; border-radius:10px; padding:12px; text-align:center;'>
                    <span class='status-badge-inactive'>🔴 No Data Loaded</span>
                    <p style='margin:6px 0 0 0; font-size:13px; color:#555;'>Load a dataset to begin</p>
                </div>
            """, unsafe_allow_html=True)

    with col_s2:
        if st.session_state.get("analysis_done"):
            st.markdown("""
                <div style='background:#eafaf1; border:1px solid #27ae60; border-radius:10px; padding:12px; text-align:center;'>
                    <span class='status-badge-active'>🟢 Analysis Complete</span>
                    <p style='margin:6px 0 0 0; font-size:13px; color:#555;'>DE results available</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background:#fdf2f2; border:1px solid #e74c3c; border-radius:10px; padding:12px; text-align:center;'>
                    <span class='status-badge-inactive'>🔴 Analysis Pending</span>
                    <p style='margin:6px 0 0 0; font-size:13px; color:#555;'>Run differential expression</p>
                </div>
            """, unsafe_allow_html=True)

    with col_s3:
        accuracy = st.session_state.get("model_accuracy", "N/A")
        if accuracy != "N/A":
            st.markdown(f"""
                <div style='background:#eafaf1; border:1px solid #27ae60; border-radius:10px; padding:12px; text-align:center;'>
                    <span class='status-badge-active'>🟢 Model Trained</span>
                    <p style='margin:6px 0 0 0; font-size:13px; color:#555;'>Accuracy: {accuracy}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background:#fdf2f2; border:1px solid #e74c3c; border-radius:10px; padding:12px; text-align:center;'>
                    <span class='status-badge-inactive'>🔴 Model Not Trained</span>
                    <p style='margin:6px 0 0 0; font-size:13px; color:#555;'>Run survival prediction</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dynamic Stats ──
    st.subheader("📊 Live Dataset Stats")

    col1, col2, col3, col4 = st.columns(4)

    card_class = "stat-card-active" if st.session_state.data_loaded else "stat-card"
    num_class = "stat-number-active" if st.session_state.data_loaded else "stat-number"

    with col1:
        st.markdown(f"""
        <div class='{card_class}'>
            <div class='{num_class}'>{st.session_state.total_patients if st.session_state.data_loaded else "—"}</div>
            <div class='stat-label'>👥 Samples Loaded</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='{card_class}'>
            <div class='{num_class}'>{st.session_state.gene_features if st.session_state.data_loaded else "—"}</div>
            <div class='stat-label'>🧬 Gene Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        dataset_name = st.session_state.get("current_dataset", "—") if st.session_state.data_loaded else "—"
        st.markdown(f"""
        <div class='{card_class}'>
            <div class='{num_class}' style='font-size:22px; padding-top:6px;'>{dataset_name}</div>
            <div class='stat-label'>📦 Active Dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        acc_display = st.session_state.get("model_accuracy", "—")
        st.markdown(f"""
        <div class='{card_class}'>
            <div class='{num_class}'>{acc_display}</div>
            <div class='stat-label'>🎯 Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Workflow Progress ──
    st.subheader("🔄 Analysis Workflow Progress")

    steps = {
        "Step 1 — Load Dataset": st.session_state.data_loaded,
        "Step 2 — Biomarker Discovery": st.session_state.get("analysis_done", False),
        "Step 3 — Gene Expression": st.session_state.get("expression_done", False),
        "Step 4 — Survival Prediction": st.session_state.get("model_accuracy", "N/A") != "N/A"
    }

    completed = sum(steps.values())
    total_steps = len(steps)
    progress_pct = int((completed / total_steps) * 100)

    st.markdown(f"""
        <div style='background:white; padding:16px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.07);'>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='font-weight:bold;'>Overall Progress</span>
                <span style='color:#27ae60; font-weight:bold;'>{progress_pct}% Complete ({completed}/{total_steps} steps)</span>
            </div>
            <div class='progress-bar-container'>
                <div class='progress-bar-fill' style='width:{progress_pct}%;'></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    step_cols = st.columns(4)
    for i, (step_name, done) in enumerate(steps.items()):
        with step_cols[i]:
            if done:
                st.markdown(f"""
                <div class='activity-card-green'>
                    <b>✅ {step_name}</b><br>
                    <span style='font-size:12px; color:#27ae60;'>Completed</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='activity-card'>
                    <b>⏳ {step_name}</b><br>
                    <span style='font-size:12px; color:#e74c3c;'>Pending</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Actions ──
    st.subheader("⚡ Quick Actions")
    q1, q2, q3 = st.columns(3)

    with q1:
        if st.button("🔬 Go to Biomarker Discovery"):
            st.session_state["_nav"] = "biomarker"
            st.info("👈 Click **Biomarker Discovery** in the sidebar to load a dataset.")

    with q2:
        if st.button("📊 Go to Gene Expression"):
            st.info("👈 Click **Gene Expression** in the sidebar.")

    with q3:
        if st.button("📈 Go to Survival Prediction"):
            st.info("👈 Click **Survival Prediction** in the sidebar.")

    st.markdown("---")

    # ── Module Cards ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card-red'>
            <h2>🔬</h2>
            <h4>Biomarker Discovery</h4>
            <p>Identify top variable genes as potential leukemia biomarkers via differential expression</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card-blue'>
            <h2>📊</h2>
            <h4>Gene Expression</h4>
            <p>Visualize expression patterns with heatmaps, PCA, t-SNE and clustering</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card-green'>
            <h2>📈</h2>
            <h4>Survival Prediction</h4>
            <p>Predict ALL vs AML leukemia type using Random Forest, SVM & Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### About LeukoDash
    LeukoDash is a web-based bioinformatics tool designed to assist researchers and clinicians
    in analyzing leukemia gene expression data. It provides an easy-to-use interface for
    biomarker discovery, gene expression visualization, and cancer type prediction.

    ### How to Use
    1. Go to **Biomarker Discovery** → load a GEO dataset or upload CSV
    2. Define sample groups (ALL vs AML)
    3. Run differential expression analysis
    4. Explore Gene Expression visualizations
    5. Train ML models in Survival Prediction
    """)

    st.markdown("""
    <div class='footer'>
        <p>🩸 <b>LeukoDash</b> — Final Year Project</p>
        <p>Developed by <b>Talha Saleem</b> | University of Agriculture, Faisalabad (UAF)</p>
        <p>Department of Bioinformatics | 2025-2026</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🔬 Biomarker Discovery":
    from modules.biomarker import show
    show()

elif page == "📊 Gene Expression":
    from modules.gene_expression import show
    show()

elif page == "📈 Survival Prediction":
    from modules.survival import show
    show()