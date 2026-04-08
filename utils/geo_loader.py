import pandas as pd
import streamlit as st
import GEOparse
import gc  # garbage collector - frees memory

@st.cache_data(show_spinner=False)
def load_geo_dataset(accession_number, max_genes=1000):
    """
    Safely loads any GEO dataset without crashing.
    - Limits to top 1000 most variable genes
    - Uses float32 to save memory
    - Caches result so it loads only once
    """
    try:
        # Step 1 - Show progress to user
        progress = st.progress(0)
        status = st.empty()
        
        status.text(f"Fetching {accession_number} from GEO...")
        progress.progress(20)
        
        # Step 2 - Download series matrix (lighter than full dataset)
        gse = GEOparse.get_GEO(
            geo=accession_number,
            destdir="./data/cache/",  # saves locally so no re-download
            silent=True
        )
        progress.progress(50)
        status.text("Processing dataset...")
        
        # Step 3 - Extract expression data
        df = gse.pivot_samples("VALUE")
        df = df.T  # rows = samples, columns = genes
        
        progress.progress(70)
        
        # Step 4 - Check size and warn user
        n_genes = df.shape[1]
        n_samples = df.shape[0]
        
        if n_genes > max_genes:
            st.warning(
                f"⚠️ Dataset has {n_genes} genes and {n_samples} samples. "
                f"Auto-selecting top {max_genes} most variable genes to prevent crash."
            )
            # Keep only most variable genes
            top_genes = df.var().nlargest(max_genes).index
            df = df[top_genes]
        
        # Step 5 - Reduce memory usage
        df = df.astype("float32")
        
        # Step 6 - Free unused memory
        del gse
        gc.collect()
        
        progress.progress(100)
        status.text("✅ Dataset loaded successfully!")
        
        return df, None  # returns data, no error
        
    except MemoryError:
        return None, "❌ Dataset too large for server memory. Try a smaller dataset."
    
    except Exception as e:
        return None, f"❌ Error loading dataset: {str(e)}"