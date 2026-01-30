import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from analytics.maxent import MaxEntModel
from analytics.svi import SVIModel
from scipy.optimize import least_squares

st.set_page_config(layout="wide", page_title="MaxEnt Distribution")
DATA_DIR = "data/processed"

# --- Loaders ---
def load_data(ticker):
    p_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(p_dir): return None
    files = sorted([f for f in os.listdir(p_dir) if f.endswith('.parquet')])
    if not files: return None
    return pd.read_parquet(os.path.join(p_dir, files[-1]))

def clean_arbitrage(strikes, prices):
    """
    Removes quotes that violate basic monotonicity (Call price must decrease).
    """
    if len(strikes) < 2: return strikes, prices
    keep_indices = [0]
    current_price = prices[0]
    for i in range(1, len(strikes)):
        if prices[i] < current_price - 0.0001:
            keep_indices.append(i)
            current_price = prices[i]
    return strikes[keep_indices], prices[keep_indices]

def get_svi_density(F, T, x_grid, df_slice):
    """ Fits SVI quickly to get a density prior """
    is_otm = ((df_slice['type']=='P') & (df_slice['strike']<F)) | ((df_slice['type']=='C') & (df_slice['strike']>F))
    calib_data = df_slice[is_otm].copy()
    if calib_data.empty: return None, None
    
    k_obs = calib_data['moneyness'].values
    ivs = np.nan_to_num(calib_data['mid_iv'].values, nan=0.0)
    w_obs = (ivs**2) * T
    
    if len(w_obs) < 5: return None, None

    def residuals(params):
        model = SVIModel(params)
        w_model = np.array([model.get_variance(k) for k in k_obs])
        return w_model - w_obs
    
    x0 = [np.min(w_obs), 0.1, -0.5, 0.0, 0.1]
    bounds = ([0,0,-0.99,-1,0.001], [2,2,0.99,1,2])
    
    try:
        res = least_squares(residuals, x0, bounds=bounds, loss='soft_l1')
        svi = SVIModel(res.x)
        k_grid = np.log(x_grid / F)
        pdf_vals = []
        for k, x_val in zip(k_grid, x_grid):
            d = svi.get_density(k)
            pdf_vals.append(d / x_val)
        return np.array(pdf_vals), res.x
    except:
        return None, None

# --- Main App ---
st.title("Non-Parametric Density (MaxEnt)")
col1, col2 = st.columns(2)
with col1:
    tickers = [d for d in os.listdir(DATA_DIR)] if os.path.exists(DATA_DIR) else []
    ticker = st.selectbox("Ticker", tickers)

if not ticker: st.stop()
df = load_data(ticker)

# Expiry Selection
df['expiry'] = df['expiry'].astype(str)
expiries = sorted(df['expiry'].unique())
expiry = st.sidebar.selectbox("Expiration", expiries)

# --- 1. Data Selection Mode ---
data_mode = st.sidebar.selectbox(
    "Price Source", 
    [
        "OTM Splice (Mid)", 
        "All Calls (Ask) - Recommended", 
        "All Calls (Bid)", 
        "All Calls (Mid)",
        "All Puts (Ask)", 
        "All Puts (Bid)",
        "All Puts (Mid)"
    ],
    help="Select which prices to use. 'Ask' is usually cleanest. 'OTM Splice' mixes Puts and Calls."
)

# --- 2. Data Preparation ---
subset = df[df['expiry'] == expiry].copy()
T = subset['T'].iloc[0]
F = subset['F'].iloc[0]
r = subset['r'].iloc[0]

subset['target_price'] = np.nan

# Define Parity Conversion Helper
# Synthetic Call = Put + (F - K)
def put_to_call(put_price, K):
    return put_price + (F - K)

# Filter logic based on Mode
if "OTM Splice" in data_mode:
    # High Strikes (Calls)
    mask_c = (subset['type'] == 'C') & (subset['strike'] >= F)
    subset.loc[mask_c, 'target_price'] = subset.loc[mask_c, 'mid'] * np.exp(r*T)
    
    # Low Strikes (Puts)
    mask_p = (subset['type'] == 'P') & (subset['strike'] < F)
    p_fwd = subset.loc[mask_p, 'mid'] * np.exp(r*T)
    subset.loc[mask_p, 'target_price'] = put_to_call(p_fwd, subset.loc[mask_p, 'strike'])

elif "All Calls" in data_mode:
    # Select Price Type
    ptype = 'ask' if "(Ask)" in data_mode else 'bid' if "(Bid)" in data_mode else 'mid'
    
    mask = subset['type'] == 'C'
    subset.loc[mask, 'target_price'] = subset.loc[mask, ptype] * np.exp(r*T)

elif "All Puts" in data_mode:
    ptype = 'ask' if "(Ask)" in data_mode else 'bid' if "(Bid)" in data_mode else 'mid'
    
    mask = subset['type'] == 'P'
    p_raw = subset.loc[mask, ptype] * np.exp(r*T)
    subset.loc[mask, 'target_price'] = put_to_call(p_raw, subset.loc[mask, 'strike'])

# Clean
clean_data = subset.dropna(subset=['target_price']).sort_values('strike')
clean_data = clean_data.groupby('strike')[['target_price']].mean().reset_index()

# --- 3. Sidebar Settings ---
with st.sidebar:
    st.markdown("---")
    st.header("Solver Settings")
    n_nodes = st.slider("Grid Nodes (N)", 50, 500, 200, 10)
    use_svi = st.checkbox("Use SVI Prior", value=True)
    
    min_k, max_k = float(clean_data['strike'].min()), float(clean_data['strike'].max())
    strike_range = st.slider("Strike Range", min_k, max_k, (min_k, max_k))
    
    input_data = clean_data[
        (clean_data['strike'] >= strike_range[0]) & 
        (clean_data['strike'] <= strike_range[1])
    ].copy()
    
    st.info(f"Strikes: {len(input_data)}")
    run = st.button("Solve Optimization", type="primary")

# --- 4. Execution ---
if run:
    status = st.empty()
    
    # Init Model
    dummy = MaxEntModel(F, T, n_nodes=n_nodes)
    grid_x = dummy.x
    prior_pdf = None
    
    if use_svi:
        prior_pdf, _ = get_svi_density(F, T, grid_x, subset)

    # Sanitize
    K_raw = input_data['strike'].values
    C_raw = input_data['target_price'].values
    K_vec, C_vec = clean_arbitrage(K_raw, C_raw)
    
    dropped = len(K_raw) - len(K_vec)
    if dropped > 0: st.warning(f"Removed {dropped} points (Vertical Arbitrage).")
    
    # Solve
    model = MaxEntModel(F, T, n_nodes=n_nodes, custom_prior=prior_pdf)
    res = model.solve(K_vec, C_vec)
    
    # --- Visualization ---
    col_l, col_r = st.columns(2)
    
    # A. Density
    fig_dist = go.Figure()
    if prior_pdf is not None:
        fig_dist.add_trace(go.Scatter(x=model.x, y=model.prior/model.dx, mode='lines', name='SVI Prior', line=dict(color='gray', dash='dash')))
        
    fig_dist.add_trace(go.Scatter(x=model.x, y=model.pdf, fill='tozeroy', name='MaxEnt Posterior', line=dict(color='purple')))
    fig_dist.update_layout(title="Implied Density", height=450)
    col_l.plotly_chart(fig_dist, width='stretch')
    
    # B. Arbitrage Check (The Green Line)
    fig_arb = go.Figure()
    slopes = np.diff(C_vec) / np.diff(K_vec)
    mid_ks = (K_vec[1:] + K_vec[:-1]) / 2
    
    # Convexity (Density Proxy)
    convexity = np.diff(slopes) / np.diff(mid_ks)
    mid_mid_ks = (mid_ks[1:] + mid_ks[:-1]) / 2
    
    fig_arb.add_trace(go.Scatter(x=mid_mid_ks, y=convexity*100, mode='lines+markers', name='Convexity', line=dict(color='green')))
    fig_arb.add_hline(y=0, line_dash='dash', line_color='red')
    fig_arb.update_layout(title="Convexity Check (Should be > 0)", height=450)
    col_r.plotly_chart(fig_arb, width='stretch')