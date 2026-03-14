import streamlit as st

st.set_page_config(
    page_title="LeukoDash",
    page_icon="🩸",
    layout="wide"
)

# Custom CSS - Colorful Theme
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
    
    h1, h2, h3 { color: #1a1a2e; }
    
    .banner {
        background: linear-gradient(135deg, #1a1a2e, #e74c3c);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/blood-drop.png", width=80)
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

    st.markdown("""
    <div class='banner'>
        <h1>🩸 LeukoDash</h1>
        <h3>A Bioinformatics Dashboard for Leukemia Analysis</h3>
        <p>Empowering researchers with AI-driven gene expression analysis</p>
    </div>
    """, unsafe_allow_html=True)

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

elif page == "🔬 Biomarker Discovery":
    from modules.biomarker import show
    show()

elif page == "📊 Gene Expression":
    from modules.gene_expression import show
    show()

elif page == "📈 Survival Prediction":
    from modules.survival import show
    show()