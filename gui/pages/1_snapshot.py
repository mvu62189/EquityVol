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

def load_data(ticker):
    p_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(p_dir): return None, None
    
    files = sorted([f for f in os.listdir(p_dir) if f.endswith('.parquet')])
    if not files: return None, None
    
    latest_chain = files[-1]
    df_chain = pd.read_parquet(os.path.join(p_dir, latest_chain))
    
    m_dir = os.path.join(MODEL_DIR, ticker)
    param_file = latest_chain.replace("chain", "svi_params").replace(".parquet", ".csv")
    param_path = os.path.join(m_dir, param_file)
    
    if os.path.exists(param_path):
        df_params = pd.read_csv(param_path)
        # Ensure expiry column is string for matching
        df_params['expiry'] = df_params['expiry'].astype(str)
    else:
        df_params = pd.DataFrame()
        
    return df_chain, df_params

# --- UI Layout ---
st.sidebar.header("Selection")
tickers = load_available_tickers()

if not tickers:
    st.error("No data found.")
    st.stop()

selected_ticker = st.sidebar.selectbox("Ticker", tickers)
df, df_params = load_data(selected_ticker)

if df is None:
    st.error("Could not load data.")
    st.stop()

# Ensure expiry is string for selection
df['expiry'] = df['expiry'].astype(str)
expiries = sorted(df['expiry'].unique())
selected_expiry = st.sidebar.selectbox("Expiration", expiries)

# --- Processing ---
subset = df[df['expiry'] == selected_expiry].copy()
subset = subset.sort_values('strike')
T = subset['T'].iloc[0] 
F = subset['F'].iloc[0]
r = subset['r'].iloc[0]

# Get SVI Params
slice_params = pd.DataFrame()
if not df_params.empty:
    slice_params = df_params[df_params['expiry'] == selected_expiry]

has_model = not slice_params.empty

st.header(f"{selected_ticker} Volatility Surface: {selected_expiry}")
col1, col2, col3 = st.columns(3)
col1.metric("Time to Expiry (T)", f"{T:.4f} years")
col2.metric("Forward Price (F)", f"{F:.2f}")
col3.metric("Risk-Free Rate (r)", f"{r:.2%}")

if has_model:
    rmse = slice_params['rmse'].iloc[0]
    st.caption(f"Model Fit RMSE: {rmse:.5f} (Lower is better)")

# --- Plotting ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Filter visual noise
plot_data = subset[(subset['mid_iv'] > 0) & (subset['mid_iv'] < 3.0)]

# Separation Logic
calls = plot_data[plot_data['type'] == 'C']
puts = plot_data[plot_data['type'] == 'P']
calibration_points = plot_data.dropna(subset=['vol_surface'])

# 1. Puts (Diamonds)
fig.add_trace(go.Scatter(
    x=puts['strike'], y=puts['bid_iv'],
    mode='markers', name='Put Bid',
    marker=dict(symbol='diamond-open', color='green', size=6)
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=puts['strike'], y=puts['ask_iv'],
    mode='markers', name='Put Ask',
    marker=dict(symbol='diamond-open', color='red', size=6)
), secondary_y=False)

# 2. Calls (Circles)
fig.add_trace(go.Scatter(
    x=calls['strike'], y=calls['bid_iv'],
    mode='markers', name='Call Bid',
    marker=dict(symbol='circle-open', color='green', size=6)
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=calls['strike'], y=calls['ask_iv'],
    mode='markers', name='Call Ask',
    marker=dict(symbol='circle-open', color='red', size=6)
), secondary_y=False)

# 3. Calibration Targets (What the model actually fitted)
fig.add_trace(go.Scatter(
    x=calibration_points['strike'], y=calibration_points['vol_surface'],
    mode='markers', name='Calibrated Data (OTM)',
    marker=dict(symbol='x', color='black', size=8, line=dict(width=1))
), secondary_y=False)

# 4. SVI Model Curve
if has_model:
    params = slice_params.iloc[0]
    svi = SVIModel([params.a, params.b, params.rho, params.m, params.sigma])
    
    k_range = np.linspace(plot_data['moneyness'].min() - 0.2, plot_data['moneyness'].max() + 0.2, 200)
    strikes_range = F * np.exp(k_range)
    
    vols = [svi.get_vol(k, T) for k in k_range]
    pdfs = [svi.get_density(k) / K_val for k, K_val in zip(k_range, strikes_range)]
    
    fig.add_trace(go.Scatter(
        x=strikes_range, y=vols,
        mode='lines', name='SVI Fit',
        line=dict(color='blue', width=3)
    ), secondary_y=False)
    
    # Scale PDF for visibility? usually ok on secondary axis
    fig.add_trace(go.Scatter(
        x=strikes_range, y=pdfs,
        mode='lines', name='Probability Density',
        line=dict(color='purple', width=0, shape='spline'),
        fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.1)'
    ), secondary_y=True)

    with st.sidebar:
        st.markdown("### SVI Parameters")
        st.table(slice_params[['a','b','rho','m','sigma']].T)

fig.update_layout(
    title=f"Smile Calibration vs Market Data",
    xaxis_title="Strike Price",
    height=700,
    hovermode="closest",
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)
fig.update_yaxes(title_text="Implied Volatility", secondary_y=False)
fig.update_yaxes(title_text="RND", secondary_y=True, showgrid=False)

st.plotly_chart(fig, use_container_width=True)