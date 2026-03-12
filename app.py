import streamlit as st

st.set_page_config(
    page_title="LeukoDash",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 LeukoDash")
st.subheader("A Bioinformatics Dashboard for Leukemia Analysis")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Biomarker Discovery",
    "Gene Expression",
    "Survival Prediction"
])

if page == "Home":
    st.markdown("""
    Welcome to **LeukoDash** — a web-based bioinformatics dashboard for leukemia research.
    
    ### Modules:
    - 🔬 **Biomarker Discovery** — Identify key leukemia biomarkers
    - 📊 **Gene Expression** — Visualize gene expression patterns
    - 📈 **Survival Prediction** — Predict patient survival outcomes
    
    Use the sidebar to navigate between modules.
    """)

elif page == "Biomarker Discovery":
    from modules.biomarker import show
    show()

elif page == "Gene Expression":
    from modules.gene_expression import show
    show()

elif page == "Survival Prediction":
    from modules.survival import show
    show()