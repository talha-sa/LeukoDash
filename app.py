import streamlit as st
import pandas as pd
import numpy as np

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
    .stat-number {
        font-size: 36px;
        font-weight: bold;
        color: #e74c3c;
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
    </style>
""", unsafe_allow_html=True)

# Initialize session state for counters
if "total_patients" not in st.session_state:
    st.session_state.total_patients = 72
if "gene_features" not in st.session_state:
    st.session_state.gene_features = "7,129"
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

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
st.sidebar.markdown("---")
st.sidebar.info("Upload CSV gene expression data to get started.")

if page == "🏠 Home":

    # Banner
    st.markdown("""
    <div class='banner'>
        <h1>🩸 LeukoDash</h1>
        <h3>A Bioinformatics Dashboard for Leukemia Analysis</h3>
        <p>Empowering researchers with AI-driven gene expression analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats counters — dynamic based on session state
    st.subheader("📊 Dataset Overview")
    if st.session_state.data_loaded:
        st.success("✅ Stats updated based on your uploaded dataset!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{st.session_state.total_patients}</div>
            <div class='stat-label'>Total Patients</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{st.session_state.gene_features}</div>
            <div class='stat-label'>Gene Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>2</div>
            <div class='stat-label'>Cancer Types</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
    model_acc = st.session_state.get("model_accuracy", "N/A")
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{model_acc}</div>
        <div class='stat-label'>Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Module cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card-red'>
            <h2>🔬</h2>
            <h4>Biomarker Discovery</h4>
            <p>Identify top variable genes as potential leukemia biomarkers</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card-blue'>
            <h2>📊</h2>
            <h4>Gene Expression</h4>
            <p>Visualize expression patterns with heatmaps and histograms</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card-green'>
            <h2>📈</h2>
            <h4>Survival Prediction</h4>
            <p>Predict ALL vs AML leukemia type using machine learning</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### About LeukoDash
    LeukoDash is a web-based bioinformatics tool designed to assist researchers and clinicians
    in analyzing leukemia gene expression data. It provides an easy-to-use interface for
    biomarker discovery, gene expression visualization, and cancer type prediction.

    ### Dataset
    Validated using the **Golub et al. (1999)** ALL-AML leukemia dataset —
    a gold standard benchmark in leukemia research with 7129 gene features across 72 patients.

    ### How to Use
    1. Select a module from the sidebar
    2. Upload your CSV gene expression file
    3. Explore results instantly
    """)

    # Footer
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