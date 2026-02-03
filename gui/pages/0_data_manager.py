import streamlit as st
import pandas as pd
import sys
import os
import time
import plotly.express as px

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

    # Create Tabs for cleaner UI
    tab_status, tab_clean, tab_qc = st.tabs(["📁 File Status", "✅ Clean Data Map", "🔍 Drop-off Analysis"])

    with tab_status:
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
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("No data folders found.")
        else:
            st.warning("Data directory initialized but empty.")

    # --- Tab 2: Clean Data Map (NEW) ---
    with tab_clean:
        st.markdown("### Valid Data Availability")
        if os.path.exists("data/clean"):
            avail_tickers = [d for d in os.listdir("data/clean") if os.path.isdir(os.path.join("data/clean", d))]
            # Default to selected ticker if available
            default_idx = avail_tickers.index(selected_ticker) if selected_ticker in avail_tickers else 0
            clean_ticker = st.selectbox("Select Ticker", avail_tickers, index=default_idx, key="clean_select")
            
            if clean_ticker:
                clean_dir = os.path.join("data/clean", clean_ticker)
                files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.parquet')])
                
                if files:
                    latest_clean = files[-1]
                    st.caption(f"Visualizing: {latest_clean}")
                    
                    df_clean = pd.read_parquet(os.path.join(clean_dir, latest_clean))
                    df_clean['expiry_date'] = pd.to_datetime(df_clean['expiry'])
                    
                    # Metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Valid Nodes", len(df_clean))
                    c2.metric("Strikes", df_clean['strike'].nunique())
                    c3.metric("Expiries", df_clean['expiry'].nunique())
                    
                    # Visualization Controls
                    viz_col1, viz_col2 = st.columns([2, 1])
                    color_mode = viz_col1.radio(
                        "Color Map By:", 
                        ["Implied Volatility (IV)", "Moneyness", "Volume", "Open Interest"], 
                        index=0, 
                        horizontal=True
                    )
                    
                    # Map friendly name to dataframe column
                    col_map = {
                        "Implied Volatility (IV)": "mid_iv",
                        "Moneyness": "moneyness",
                        "Volume": "volume",
                        "Open Interest": "openInterest"
                    }
                    target_col = col_map[color_mode]
                    
                    # Scatter Plot: Strike vs Expiry (Colored by Volume)
                    fig = px.scatter(
                        df_clean,
                        x='expiry_date',
                        y='strike',
                        color='volume',
                        hover_data=['bid', 'ask', 'mid_iv', 'volume', 'openInterest'],
                        color_continuous_scale='Spectral_r' if target_col == 'mid_iv' else 'Viridis',
                        title=f"{clean_ticker} - Option Surface Grid ({color_mode})",
                        opacity=0.9
                    )
                    
                    # Set fixed marker size and add a thin border for clarity
                    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
                    
                    # Add current spot price line if available
                    if 'underlying_price' in df_clean.columns:
                        spot = df_clean['underlying_price'].iloc[0]
                        fig.add_hline(y=spot, line_dash="dash", line_color="white", annotation_text="Spot")

                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No clean data files found.")
        else:
            st.info("Run ETL to generate clean data.")

    with tab_qc:
        st.markdown("### Data Quality Report")
        st.caption("Analyze why data points were excluded from calibration.")
        
        # 1. Select Ticker to Inspect
        if os.path.exists("data/dropped"):
            avail_tickers = [d for d in os.listdir("data/dropped") if os.path.isdir(os.path.join("data/dropped", d))]
            qc_ticker = st.selectbox("Select Ticker", avail_tickers)
            
            if qc_ticker:
                dropped_dir = os.path.join("data/dropped", qc_ticker)
                files = sorted([f for f in os.listdir(dropped_dir) if f.endswith('.parquet')])
                
                if files:
                    latest_drop_file = files[-1]
                    st.caption(f"Showing latest: {latest_drop_file}")
                    
                    # Load Data
                    df_drop = pd.read_parquet(os.path.join(dropped_dir, latest_drop_file))
                    
                    # --- METRICS ---
                    total_dropped = len(df_drop)
                    st.metric("Total Dropped Points", total_dropped)
                    
                    # --- CHART 1: Reasons Bar Chart ---
                    st.markdown("#### Drop Reasons")
                    reason_counts = df_drop['drop_reason'].value_counts().reset_index()
                    reason_counts.columns = ['Reason', 'Count']
                    
                    fig_bar = px.bar(reason_counts, x='Reason', y='Count', color='Reason', text='Count')
                    st.plotly_chart(fig_bar, width='stretch')
                    
                    # --- CHART 2: Scatter (Where were they?) ---
                    st.markdown("#### Location of Drops (Strike vs Expiry)")
                    # Convert expiry to datetime for better plotting
                    df_drop['expiry_date'] = pd.to_datetime(df_drop['expiry'])
                    
                    fig_scatter = px.scatter(
                        df_drop, 
                        x='expiry_date', 
                        y='strike', 
                        color='drop_reason',
                        hover_data=['bid', 'ask', 'volume', 'mid_iv'],
                        title="Dropped Points by Strike & Expiry",
                        opacity=0.6
                    )
                    st.plotly_chart(fig_scatter, width='stretch')

                    # --- RAW DATA VIEW ---
                    with st.expander("View Raw Dropped Data"):
                        st.dataframe(df_drop)
                else:
                    st.info("No dropped data files found.")
        else:
            st.info("Run the pipeline to generate quality reports.")