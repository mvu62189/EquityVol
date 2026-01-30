import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from scipy.signal import savgol_filter

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from analytics.maxent import MaxEntModel
from analytics.svi import SVIModel
from scipy.optimize import least_squares

st.set_page_config(layout="wide", page_title="MaxEnt Lab")
DATA_DIR = "data/processed"

# --- Loaders ---
def load_data(ticker):
    p_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(p_dir): return None
    files = sorted([f for f in os.listdir(p_dir) if f.endswith('.parquet')])
    if not files: return None
    return pd.read_parquet(os.path.join(p_dir, files[-1]))

def repair_convexity(strikes, prices):
    """
    Greatest Convex Minorant (GCM) Algorithm.
    Iteratively adjusts prices downwards until they form a perfectly convex curve.
    - Preserves stride (uses all points).
    - Removes arbitrage (negative density) by flattening bumps.
    """
    K = np.array(strikes)
    P = np.array(prices)
    
    max_passes = 200
    tol = 1e-6
    
    for _ in range(max_passes):
        changes = 0
        
        # 1. Enforce Monotonicity (Vertical Arb)
        # Price must decrease as strike increases: P[i] <= P[i-1]
        for i in range(1, len(P)):
            if P[i] > P[i-1] - tol:
                P[i] = P[i-1] - tol
                changes += 1
        
        # 2. Enforce Convexity (Butterfly Arb)
        # P[i] must be <= Chord(P[i-1], P[i+1])
        for i in range(1, len(P)-1):
            k_left, k_mid, k_right = K[i-1], K[i], K[i+1]
            p_left, p_curr, p_right = P[i-1], P[i], P[i+1]
            
            # Linear Interpolation (The Chord)
            p_max_valid = p_left + (k_mid - k_left) * (p_right - p_left) / (k_right - k_left)
            
            if p_curr > p_max_valid + tol:
                P[i] = p_max_valid
                changes += 1
                
        if changes == 0:
            break
            
    return K, P

def get_svi_density(F, T, x_grid, df_slice):
    """ Fits SVI quickly to get a density prior """
    is_otm = (
        ((df_slice['type']=='P') & (df_slice['strike']<F)) | 
        ((df_slice['type']=='C') & (df_slice['strike']>F))
    ) & (df_slice['volume'] > 0)   
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

def put_to_call(put_price, K, F):
    return put_price + (F - K)

# --- Main App ---
st.title("Non-Parametric Density (MaxEnt)")
col1, col2 = st.columns(2)
with col1:
    tickers = [d for d in os.listdir(DATA_DIR)] if os.path.exists(DATA_DIR) else []
    ticker = st.selectbox("Ticker", tickers)

if not ticker: st.stop()
df = load_data(ticker)
df['expiry'] = df['expiry'].astype(str)
expiry = st.sidebar.selectbox("Expiration", sorted(df['expiry'].unique()))

# --- Data Preparation (OTM Splice Default) ---
subset = df[df['expiry'] == expiry].copy()
T = subset['T'].iloc[0]
F = subset['F'].iloc[0]
r = subset['r'].iloc[0]
subset['target_price'] = np.nan

# Filter: Exclude 0 Volume (Dead Quotes)
vol_mask = subset['volume'] > 0
active = subset[vol_mask].copy()

# OTM Splice Logic
# 1. High Strikes -> Calls (Mid)
mask_c = (active['type'] == 'C') & (active['strike'] >= F)
active.loc[mask_c, 'target_price'] = active.loc[mask_c, 'mid'] * np.exp(r*T)

# 2. Low Strikes -> Puts (Mid) + Parity
mask_p = (active['type'] == 'P') & (active['strike'] < F)
p_raw = active.loc[mask_p, 'mid'] * np.exp(r*T)
active.loc[mask_p, 'target_price'] = put_to_call(p_raw, active.loc[mask_p, 'strike'], F)

# Clean
clean_data = active.dropna(subset=['target_price']).sort_values('strike')
# Deduplicate strikes (rare cases where volume exists for both exactly at ATM)
clean_data = clean_data.groupby('strike')[['target_price']].mean().reset_index()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("---")
    st.header("Preprocessing")
    st.caption("Mode: OTM Splice (Mid) + Vol>0")
    
    repair = st.checkbox("Auto-Repair Convexity", value=True, help="Adjusts prices downwards to enforce 'Greatest Convex Minorant'. Prevents solver crashes.")
    stride = st.slider("Strike Stride", 1, 10, 1, help="Use 1 for max detail. Increase if data is extremely noisy.")
    
    st.header("Solver Settings")
    n_nodes = st.slider("Grid Nodes", 50, 4000, 900)
    use_svi = st.checkbox("Use SVI Prior", value=True)
    
    min_k, max_k = float(clean_data['strike'].min()), float(clean_data['strike'].max())
    strike_range = st.slider("Range", min_k, max_k, (min_k, max_k))
    
    input_data = clean_data[
        (clean_data['strike'] >= strike_range[0]) & 
        (clean_data['strike'] <= strike_range[1])
    ].copy()
    
    st.info(f"Active Points: {len(input_data)}")
    run = st.button("Solve", type="primary")

if run:
    # 1. Subsample
    K_raw = input_data['strike'].values
    C_raw = input_data['target_price'].values
    
    if stride > 1:
        K_raw = K_raw[::stride]
        C_raw = C_raw[::stride]
    
    # 2. Repair
    if repair:
        K_vec, C_vec = repair_convexity(K_raw, C_raw)
        
        # Calculate Adjustments stats
        diffs = C_vec - C_raw
        adj_count = np.sum(diffs < -1e-5)
        if adj_count > 0:
            st.toast(f"🔧 Repaired {adj_count} prices to enforce convexity.")
    else:
        K_vec, C_vec = K_raw, C_raw
    
    if len(K_vec) < 5:
        st.error("Too few points remaining. Widen range.")
        st.stop()

    # 3. Solver
    model = MaxEntModel(F, T, n_nodes=n_nodes)
    prior_pdf = None
    if use_svi:
        prior_pdf, _ = get_svi_density(F, T, model.x, subset)
    
    model = MaxEntModel(F, T, n_nodes=n_nodes, custom_prior=prior_pdf)
    res = model.solve(K_vec, C_vec)
    
    # --- Visualization ---
    if res['success']:
        st.success(f"Converged! Error: {res['error']:.5f}")
    else:
        st.warning(f"Solver stopped: {res['message']}")

    col_l, col_r = st.columns(2)
    
    # Plot 1: Pricing Check
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=K_raw, y=C_raw, mode='markers', name='Market Input', marker=dict(color='gray', opacity=0.4, size=5)))
    if repair:
        fig_p.add_trace(go.Scatter(x=K_vec, y=C_vec, mode='markers', name='Repaired Input', marker=dict(color='red', symbol='x', size=4)))
    fig_p.add_trace(go.Scatter(x=K_vec, y=model.model_prices, mode='lines', name='MaxEnt Fit', line=dict(color='blue')))
    fig_p.update_layout(title="Pricing Check (Log Scale)",  height=450)
    col_l.plotly_chart(fig_p, width='stretch')
    
    # Plot 2: Arbitrage (Dual Delta)
    fig_arb = go.Figure()
    
    dK = np.diff(K_vec)
    dC = np.diff(C_vec)
    dual_delta = dC / dK
    mid_k_delta = (K_vec[1:] + K_vec[:-1]) / 2
    
    convexity = np.diff(dual_delta)
    mid_k_conv = (mid_k_delta[1:] + mid_k_delta[:-1]) / 2
    
    fig_arb.add_trace(go.Scatter(x=mid_k_delta, y=dual_delta, mode='lines+markers', name='Dual Delta', line=dict(color='orange')))
    fig_arb.add_trace(go.Scatter(x=mid_k_conv, y=convexity * 100, mode='lines', name='Convexity', line=dict(color='green'), yaxis="y2"))
    
    fig_arb.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_arb.update_layout(
        title="Arbitrage Check (Green > 0)",
        yaxis2=dict(title="Convexity", overlaying="y", side="right"),
        height=450
    )
    col_r.plotly_chart(fig_arb, width='stretch')
    
    # Plot 3: Implied Distribution
    st.subheader("Implied Risk Neutral Distribution")
    fig_d = go.Figure()
    
    # Calculate Density (p / dx)
    # This makes the scale invariant to the number of grid nodes.
    density = model.pdf 

    # Plot SVI Prior (if used)
    if prior_pdf is not None:
        # prior_pdf from get_svi_density is ALREADY a density
        # But MaxEntModel normalized it to a PMF internally.
        # So we use model.prior (PMF) and divide by dx to get Density back.
        prior_density = model.prior/model.dx
        fig_d.add_trace(go.Scatter(x=model.x, y=model.prior/model.dx, mode='lines', name='SVI Prior', line=dict(color='gray', dash='dash')))

    # Plot Posterior
    # Using pdf/dx to show Density (area under curve = 1)
    fig_d.add_trace(go.Scatter(x=model.x, y=model.pdf, fill='tozeroy', name='MaxEnt Density', line=dict(color='purple')))
    
    fig_d.update_layout(height=500, xaxis_title="Strike", yaxis_title="Probability Density")
    st.plotly_chart(fig_d, width='stretch')