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

def get_svi_density(F, T, x_grid, df_slice):
    """ Helper: Fits SVI quickly to get a density prior on the x_grid """
    # Filter for OTM only
    is_otm = ((df_slice['type']=='P') & (df_slice['strike']<F)) | ((df_slice['type']=='C') & (df_slice['strike']>F))
    calib_data = df_slice[is_otm].copy()
    if calib_data.empty: return None, None
        
    k_obs = calib_data['moneyness'].values
    # Clean IVs
    ivs = calib_data['mid_iv'].values
    ivs = np.nan_to_num(ivs, nan=0.0)
    w_obs = (ivs**2) * T
    
    if len(w_obs) < 5: return None, None

    # Fit SVI
    def residuals(params):
        model = SVIModel(params)
        w_model = np.array([model.get_variance(k) for k in k_obs])
        return w_model - w_obs
    
    x0 = [np.min(w_obs), 0.1, -0.5, 0.0, 0.1]
    bounds = ([0,0,-0.99,-1,0.001], [2,2,0.99,1,2])
    
    try:
        res = least_squares(residuals, x0, bounds=bounds, loss='soft_l1')
        svi = SVIModel(res.x)
        
        # Generate Density on Grid
        k_grid = np.log(x_grid / F)
        pdf_vals = []
        for k, x_val in zip(k_grid, x_grid):
            # SVI returns density wrt k. Convert to density wrt price: PDF(S) = PDF(k) / S
            d = svi.get_density(k)
            pdf_vals.append(d / x_val)
            
        return np.array(pdf_vals), res.x
    except:
        return None, None

# --- Main App ---
st.title("Non-Parametric Density (MaxEnt)")
st.markdown(r"""
**Methodology:**
Recover the Risk Neutral Density by solving for the probability vector $p$ that maximizes entropy while satisfying market pricing constraints ($C = A \cdot p$).
Uses **Relative Entropy** with an **SVI Prior** to handle equity skew correctly.
""")

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

# --- 1. Data Preparation (The OTM Splice) ---
subset = df[df['expiry'] == expiry].copy()
T = subset['T'].iloc[0]
F = subset['F'].iloc[0]
r = subset['r'].iloc[0]

# Create 'synthetic_fwd_call' column
subset['synthetic_fwd_call'] = np.nan

# A. High Strikes (Call is OTM) -> Use Market Call * exp(rT)
mask_call_otm = (subset['type'] == 'C') & (subset['strike'] >= F)
subset.loc[mask_call_otm, 'synthetic_fwd_call'] = subset.loc[mask_call_otm, 'mid'] * np.exp(r*T)

# B. Low Strikes (Put is OTM) -> Use Market Put + Parity
# C_syn = P_mkt * exp(rT) + (F - K)
mask_put_otm = (subset['type'] == 'P') & (subset['strike'] < F)
P_fwd = subset.loc[mask_put_otm, 'mid'] * np.exp(r*T)
K_vals = subset.loc[mask_put_otm, 'strike']
subset.loc[mask_put_otm, 'synthetic_fwd_call'] = P_fwd + (F - K_vals)

# C. Clean and Aggregate
# We drop rows where we couldn't calculate a price (deep ITM options that we ignore)
clean_data = subset.dropna(subset=['synthetic_fwd_call']).sort_values('strike')

# Group by strike to handle duplicates (e.g. if dataset has multiple snapshots or overlaps)
# IMPORTANT: This creates the dataframe used for the Solver
clean_data = clean_data.groupby('strike')[['synthetic_fwd_call']].mean().reset_index()

# --- 2. Sidebar Settings ---
with st.sidebar:
    st.header("Matrix Settings")
    n_nodes = st.slider("Grid Nodes (N)", 50, 5000, 200, 10)
    use_svi = st.checkbox("Use SVI Prior", value=True)
    
    # Strike Range
    min_k, max_k = float(clean_data['strike'].min()), float(clean_data['strike'].max())
    strike_range = st.slider("Strike Range", min_k, max_k, (min_k, max_k))
    
    # Filter Inputs
    input_data = clean_data[
        (clean_data['strike'] >= strike_range[0]) & 
        (clean_data['strike'] <= strike_range[1])
    ].copy()
    
    st.info(f"Constraints: {len(input_data)} strikes")
    
    run = st.button("Solve Optimization", type="primary")

# --- 3. Execution ---
if run:
    status = st.empty()
    status.info("Initializing...")
    
    # Init Model to get Grid
    dummy = MaxEntModel(F, T, n_nodes=n_nodes)
    grid_x = dummy.x
    
    prior_pdf = None
    prior_name = "Lognormal"
    
    if use_svi:
        status.info("Fitting SVI Prior...")
        # Note: We pass 'subset' (raw data with IVs) to SVI, not 'clean_data' (just prices)
        prior_pdf, _ = get_svi_density(F, T, grid_x, subset)
        if prior_pdf is not None:
            prior_name = "SVI"
            
    # Solve
    status.info(f"Optimizing ({prior_name} Prior)...")
    model = MaxEntModel(F, T, n_nodes=n_nodes, custom_prior=prior_pdf)
    
    # Get Constraints
    K_vec = input_data['strike'].values
    C_vec = input_data['synthetic_fwd_call'].values
    
    res = model.solve(K_vec, C_vec)
    status.empty()
    
    # --- 4. Visualization ---
    if res['success']:
        st.success(f"Converged! Error: {res['error']:.5f}")
    else:
        st.warning(f"Solver stopped: {res['message']}")
        
    col_l, col_r = st.columns(2)
    
    # Plot A: Density
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(
        x=model.x, y=model.prior / model.dx,
        mode='lines', name=f'Prior ({prior_name})',
        line=dict(color='gray', dash='dash')
    ))
    fig_dist.add_trace(go.Scatter(
        x=model.x, y=model.pdf,
        fill='tozeroy', name='MaxEnt Posterior',
        line=dict(color='purple', width=3)
    ))
    fig_dist.update_layout(title="Probability Density", xaxis_title="Price", height=500)
    col_l.plotly_chart(fig_dist, width='stretch')
    
    # Plot B: Pricing Error
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=K_vec, y=C_vec, mode='markers', name='Market', marker=dict(color='red')
    ))
    fig_err.add_trace(go.Scatter(
        x=K_vec, y=model.model_prices, mode='lines', name='Model', line=dict(color='blue', width=1)
    ))
    
    # Secondary axis for error
    fig_err.update_layout(title="Pricing Fit", height=500)
    col_r.plotly_chart(fig_err, width='stretch')

    # ... inside the "Plots" section ...
    
    # 3. Arbitrage Check (Slope of Prices)
    # The slope dC/dK is roughly -Prob(S > K). It must be strictly increasing (less negative).
    # -d2C/dK2 is the Probability Density. It must be Positive.
    
    fig_arb = go.Figure()
    
    # Calculate slopes
    slopes = np.diff(C_vec) / np.diff(K_vec)
    mid_ks = (K_vec[1:] + K_vec[:-1]) / 2
    
    fig_arb.add_trace(go.Scatter(
        x=mid_ks, y=slopes, mode='lines+markers', name='Slope (Delta)',
        line=dict(color='orange')
    ))
    
    # Add Zero Line (If slope is positive, vertical arb exists)
    fig_arb.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Second Derivative (Convexity)
    convexity = np.diff(slopes) / np.diff(mid_ks)
    mid_mid_ks = (mid_ks[1:] + mid_ks[:-1]) / 2
    
    fig_arb.add_trace(go.Scatter(
        x=mid_mid_ks, y=convexity * 100, # Scale up for visibility
        mode='lines+markers', name='Convexity (Density Proxy)',
        line=dict(color='green'),
        yaxis="y2"
    ))
    
    fig_arb.update_layout(
        title="Arbitrage Check (Green must be > 0)", 
        yaxis_title="Slope",
        yaxis2=dict(title="Convexity", overlaying="y", side="right")
    )
    
    st.plotly_chart(fig_arb, width='stretch')