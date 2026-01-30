import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from analytics.svi import SVIModel

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Snapshot Inspector")
DATA_DIR = "data/processed"
MODEL_DIR = "data/models"

# --- Loaders ---
@st.cache_data
def load_available_tickers():
    if not os.path.exists(DATA_DIR): return []
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

def get_available_snapshots(ticker):
    """Returns list of snapshot filenames sorted new -> old"""
    p_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(p_dir): return []
    # Sort descending so latest is first
    files = sorted([f for f in os.listdir(p_dir) if f.endswith('.parquet')], reverse=True)
    return files

def load_data(ticker, snapshot_file):
    """Loads a specific snapshot file and its corresponding model params"""
    p_dir = os.path.join(DATA_DIR, ticker)
    path = os.path.join(p_dir, snapshot_file)
    
    if not os.path.exists(path): return None, None
    
    df_chain = pd.read_parquet(path)
    
    # Attempt to load matching parameters
    # Convention: chain_X.parquet -> svi_params_X.csv
    m_dir = os.path.join(MODEL_DIR, ticker)
    param_file = snapshot_file.replace("chain", "svi_params").replace(".parquet", ".csv")
    param_path = os.path.join(m_dir, param_file)
    
    if os.path.exists(param_path):
        df_params = pd.read_csv(param_path)
        # Ensure expiry column is string for strict matching
        df_params['expiry'] = df_params['expiry'].astype(str)
    else:
        df_params = pd.DataFrame()
        
    return df_chain, df_params

# --- UI Layout ---
st.sidebar.header("Data Selection")
tickers = load_available_tickers()

if not tickers:
    st.error("No data found. Please run the Data Pipeline first.")
    st.stop()

# 1. Select Ticker
selected_ticker = st.sidebar.selectbox("Ticker", tickers)

# 2. Select Snapshot (New Feature)
snapshots = get_available_snapshots(selected_ticker)
if not snapshots:
    st.warning(f"No snapshots found for {selected_ticker}")
    st.stop()

# Clean up filenames for display (optional)
# e.g. "chain_2023-12-01_1600.parquet" -> "2023-12-01 16:00"
snapshot_map = {f: f.replace("chain_", "").replace(".parquet", "") for f in snapshots}
selected_snapshot_file = st.sidebar.selectbox(
    "Snapshot Time", 
    options=snapshots, 
    format_func=lambda x: snapshot_map[x]
)

# 3. Load Data
df, df_params = load_data(selected_ticker, selected_snapshot_file)

if df is None or df.empty:
    st.error("Data loaded but appears empty.")
    st.stop()

# 4. Select Expiry
df['expiry'] = df['expiry'].astype(str)
expiries = sorted(df['expiry'].unique())

if not expiries:
    st.warning(f"No expirations found in this snapshot.")
    st.stop()

selected_expiry = st.sidebar.selectbox("Expiration", expiries)

# --- Processing for View ---
subset = df[df['expiry'] == selected_expiry].copy()

if subset.empty:
    st.warning(f"No data found for expiry: {selected_expiry}")
    st.stop()

subset = subset.sort_values('strike')
T = subset['T'].iloc[0]
F = subset['F'].iloc[0]
r = subset['r'].iloc[0]

# Check for Model
slice_params = pd.DataFrame()
if not df_params.empty:
    slice_params = df_params[df_params['expiry'] == selected_expiry]

has_model = not slice_params.empty

# --- Dashboard Header ---
st.header(f"{selected_ticker} Surface | {snapshot_map[selected_snapshot_file]}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expiration", selected_expiry)
col2.metric("Time to Expiry (T)", f"{T:.4f} yrs")
col3.metric("Forward Price (F)", f"{F:.2f}")
col4.metric("Risk-Free Rate (r)", f"{r:.2%}")

if has_model:
    rmse = slice_params['rmse'].iloc[0]
    st.sidebar.markdown("### Model Fit Stats")
    st.sidebar.success(f"Calibration RMSE: {rmse:.5f}")

# --- Plotting ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Filter visual noise for the plot (hide crazy IVs > 300%)
plot_data = subset[(subset['mid_iv'] > 0) & (subset['mid_iv'] < 3.0)]

calls = plot_data[plot_data['type'] == 'C']
puts = plot_data[plot_data['type'] == 'P']
calib_points = plot_data.dropna(subset=['vol_surface'])

# 1. Market Data (Puts)
fig.add_trace(go.Scatter(
    x=puts['strike'], y=puts['mid_iv'],
    mode='markers', name='Put Market',
    marker=dict(symbol='diamond-open', color='green', size=5, opacity=0.7)
), secondary_y=False)

# 2. Market Data (Calls)
fig.add_trace(go.Scatter(
    x=calls['strike'], y=calls['mid_iv'],
    mode='markers', name='Call Market',
    marker=dict(symbol='circle-open', color='red', size=5, opacity=0.7)
), secondary_y=False)

# 3. Calibration Targets (The points the model actually fit to)
fig.add_trace(go.Scatter(
    x=calib_points['strike'], y=calib_points['vol_surface'],
    mode='markers', name='Calibrated (OTM)',
    marker=dict(symbol='x', color='black', size=6)
), secondary_y=False)

# 4. SVI Curve & Density
if has_model:
    params = slice_params.iloc[0]
    svi = SVIModel([params.a, params.b, params.rho, params.m, params.sigma])
    
    # Create smooth domain
    k_min, k_max = plot_data['moneyness'].min() - 0.2, plot_data['moneyness'].max() + 0.2
    k_range = np.linspace(k_min, k_max, 200)
    strikes_range = F * np.exp(k_range)
    
    vols = [svi.get_vol(k, T) for k in k_range]
    pdfs = [svi.get_density(k) / K_val for k, K_val in zip(k_range, strikes_range)]
    
    # Volatility Curve
    fig.add_trace(go.Scatter(
        x=strikes_range, y=vols,
        mode='lines', name='SVI Model',
        line=dict(color='blue', width=3)
    ), secondary_y=False)
    
    # PDF Area
    fig.add_trace(go.Scatter(
        x=strikes_range, y=pdfs,
        mode='lines', name='Implied Probability',
        line=dict(color='purple', width=0),
        fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.15)'
    ), secondary_y=True)

    with st.sidebar:
        with st.expander("SVI Parameters", expanded=True):
            st.table(slice_params[['a','b','rho','m','sigma']].T)

# --- Graph Styling ---
fig.update_layout(
    title=f"Volatility Smile vs Market Data ({selected_expiry})",
    xaxis_title="Strike Price",
    height=600,
    hovermode="closest",
    legend=dict(orientation="h", y=1.1)
)

fig.update_yaxes(title_text="Implied Volatility", secondary_y=False)
fig.update_yaxes(title_text="Probability Density", secondary_y=True, showgrid=False)

st.plotly_chart(fig, use_container_width=True)