import streamlit as st
import pandas as pd
import sys
import os
import time

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import core logic
from main_etl import run_etl
from main_calibration import run_calibration
from utils.search import get_ticker_suggestions

st.set_page_config(page_title="Data Manager", layout="wide")

st.title("Data Pipeline Manager")
st.markdown("### Fetch new market data and recalibrate models")

# --- Inputs ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Search Ticker")
    # User types here
    search_query = st.text_input("Type to search (e.g. 'Nvidia')", value="", help="Type company name or ticker")

    selected_ticker = None
    
    # Logic: Show suggestions if user types
    if len(search_query) > 1:
        df_suggestions = get_ticker_suggestions(search_query)
        
        if not df_suggestions.empty:
            # Create a label for the selectbox: "SYMBOL - Name (Exchange)"
            df_suggestions['label'] = df_suggestions['Ticker'] + " - " + df_suggestions['Name']
            
            # Selectbox to pick the specific asset
            selection = st.selectbox("Select Asset:", df_suggestions['label'])
            
            # Extract the actual ticker symbol back from the selection
            if selection:
                selected_ticker = selection.split(" - ")[0]
        else:
            st.warning("No matches found.")
            # Fallback: Allow them to just use the raw text if search fails
            selected_ticker = search_query.upper()
    
    # If no search, default to SPY
    elif len(search_query) == 0:
         selected_ticker = "SPY"

    st.markdown("---")
    st.metric("Target Ticker", selected_ticker)
    
    # Options
    run_calib = st.checkbox("Run Calibration immediately?", value=True)
    
    # --- Execution Button ---
    if st.button("Run Pipeline", type="primary", disabled=(not selected_ticker)):
        status_container = st.empty()
        
        # 1. Run ETL
        status_container.info(f"Step 1/2: Fetching data for {selected_ticker}...")
        try:
            with st.spinner("Talking to Yahoo Finance..."):
                run_etl(selected_ticker)
            
            st.success(f"ETL Complete: Data saved for {selected_ticker}")
            
            # 2. Run Calibration
            if run_calib:
                status_container.info(f"Step 2/2: Calibrating volatility surfaces...")
                with st.spinner("Fitting SVI models..."):
                    run_calibration(selected_ticker)
                st.success(f"Calibration Complete: Models ready.")
                
            status_container.success("Pipeline Finished Successfully!")
            time.sleep(2)
            
        except Exception as e:
            st.error(f"Pipeline Failed: {e}")

# --- Existing Data Viewer ---
with col2:
    st.subheader("Local Repository Status")
    data_root = "data/processed"
    
    if os.path.exists(data_root):
        tickers = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        
        if tickers:
            status_data = []
            for t in tickers:
                t_path = os.path.join(data_root, t)
                files = sorted([f for f in os.listdir(t_path) if f.endswith('.parquet')])
                if files:
                    latest = files[-1]
                    mod_time = os.path.getmtime(os.path.join(t_path, latest))
                    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                    
                    model_path = os.path.join("data/models", t, latest.replace("chain", "svi_params").replace(".parquet", ".csv"))
                    has_model = "✅" if os.path.exists(model_path) else "❌"
                    
                    status_data.append({
                        "Ticker": t, 
                        "Last Updated": ts, 
                        "Model": has_model, 
                        "File Size (MB)": round(os.path.getsize(os.path.join(t_path, latest))/1024/1024, 2)
                    })
            
            st.dataframe(
                pd.DataFrame(status_data), 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No data folders found.")
    else:
        st.warning("Data directory initialized but empty.")