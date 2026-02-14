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
from analytics.arbitrage import repair_convexity

st.set_page_config(layout="wide", page_title="MaxEnt 3D Surface")
DATA_DIR = "data/processed"
MODEL_DIR = "data/models"

# --- HELPER FUNCTIONS ---
def filter_wings(strikes, prices, F, T, r=0.0, quantile_tol=0.001):
    """
    Smart Wing Filter:
    Scans from ATM outwards to identify where the Dual Delta (Slope) 
    stabilizes into the 'irrelevant' wings (Deep ITM/OTM).
    
    - Left Tail: Cuts when slope saturates near -exp(-rT)
    - Right Tail: Cuts when slope saturates near 0
    - This removes noisy tails (e.g. -0.98 vs -1.0) that cause convexity spikes.
    """
    if len(strikes) < 5: return strikes, prices
    
    df_factor = np.exp(-r * T)
    # Numerical Slope (Dual Delta)
    slopes = np.diff(prices) / np.diff(strikes)
    
    # Find index closest to ATM
    atm_idx = np.abs(strikes[:-1] - F).argmin()
    
    # 1. Left Scan (ATM -> Left)
    # Stop if slope becomes too steep (<= -df_factor + tol)
    left_cut = 0
    for i in range(atm_idx, -1, -1):
        if slopes[i] <= -df_factor + quantile_tol:
            left_cut = i + 1
            break
            
    # 2. Right Scan (ATM -> Right)
    # Stop if slope becomes too flat (>= -tol)
    right_cut = len(strikes) - 1
    for i in range(atm_idx, len(slopes)):
        if slopes[i] >= -quantile_tol:
            right_cut = i
            break
    
    # Safety: Ensure we don't cut everything
    if right_cut - left_cut < 3:
        return strikes, prices
        
    return strikes[left_cut:right_cut+1], prices[left_cut:right_cut+1]

def filter_strikes_by_value(strikes, prices, min_diff):
    if len(strikes) == 0: return np.array([]), np.array([])
    if min_diff <= 0: return strikes, prices
    keep_k, keep_p = [strikes[0]], [prices[0]]
    for k, p in zip(strikes[1:], prices[1:]):
        if k >= keep_k[-1] + min_diff:
            keep_k.append(k)
            keep_p.append(p)
    if keep_k[-1] != strikes[-1]:
         keep_k.append(strikes[-1])
         keep_p.append(prices[-1])
    return np.array(keep_k), np.array(keep_p)

def load_clean_data(ticker):
    p_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(p_dir): return None
    files = sorted([f for f in os.listdir(p_dir) if f.endswith('.parquet')])
    if not files: return None
    return pd.read_parquet(os.path.join(p_dir, files[-1]))

def load_svi_params(ticker, expiry_str):
    m_dir = os.path.join(MODEL_DIR, ticker)
    if not os.path.exists(m_dir): return None
    files = sorted([f for f in os.listdir(m_dir) if f.endswith('.csv')])
    if not files: return None
    df_params = pd.read_csv(os.path.join(m_dir, files[-1]))
    row = df_params[df_params['expiry'].astype(str) == expiry_str]
    if row.empty: return None
    return row.iloc[0]

def put_to_call(put_price, K, F):
    return put_price + (F - K)

# --- MAIN APP ---
st.title("MaxEnt Density Surface (Batch Processing)")

# 1. Select Ticker
col_sel, _ = st.columns(2)
with col_sel:
    tickers = [d for d in os.listdir(DATA_DIR)] if os.path.exists(DATA_DIR) else []
    ticker = st.selectbox("Ticker", tickers)

if not ticker: st.stop()

# Load Data
df = load_clean_data(ticker)
df['expiry'] = df['expiry'].astype(str)
unique_expiries = sorted(df['expiry'].unique())

# --- BATCH SETTINGS ---
with st.sidebar:
    st.header("Optimization Settings")
    repair = st.checkbox("Auto-Repair Convexity", value=True)
    dollar_stride = st.number_input("Min Strike Gap ($)", min_value=0.5, value=1.0, step=0.5)
    
    st.markdown("### Filtering")
    filter_wings_on = st.checkbox("Filter Irrelevant Wings", value=True)
    wing_tol = st.slider("Wing Cut Tolerance (CDF)", 0.001, 0.05, 0.001, step=0.001, 
                         help="Stops scanning when Dual Delta is within this range of 0 or -1. Higher = Cuts more aggressively.")

    st.markdown("---")
    st.header("Grid Settings")
    grid_multiple = st.slider("Grid Multiplier", 0.1, 5.0, 1.0, step=0.1)
    
    use_svi = st.checkbox("Use SVI Prior", value=True)
    
    st.markdown("---")
    # Increased default resolution to capture narrow peaks
    surf_res = st.slider("Surface Plot Resolution (X-Axis)", 100, 1000, 300, 
                         help="Number of points in the shared strike grid. Higher = smoother but slower.")

# --- BATCH EXECUTION ---
if st.button("Generate 3D Surface", type="primary"):
    
    # Storage for Pass 2
    solved_models = [] # List of dicts: {'expiry': str, 'x': array, 'pdf': array}
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    # --- PASS 1: SOLVE INDIVIDUAL SLICES ---
    for i, exp in enumerate(unique_expiries):
        status.text(f"Optimizing slice {i+1}/{len(unique_expiries)}: {exp}...")
        
        # A. Data Prep
        subset = df[df['expiry'] == exp].copy()
        if subset.empty: continue
        
        T = subset['T'].iloc[0]
        F = subset['F'].iloc[0]
        r = subset['r'].iloc[0]
        
        subset['target_price'] = np.nan
        mask_c = (subset['type'] == 'C') & (subset['strike'] >= F)
        subset.loc[mask_c, 'target_price'] = subset.loc[mask_c, 'mid'] * np.exp(r*T)

        mask_p = (subset['type'] == 'P') & (subset['strike'] < F)
        p_raw = subset.loc[mask_p, 'mid'] * np.exp(r*T)
        subset.loc[mask_p, 'target_price'] = put_to_call(p_raw, subset.loc[mask_p, 'strike'], F)

        clean_data = subset.dropna(subset=['target_price']).sort_values('strike')
        clean_data = clean_data.groupby('strike')['target_price'].mean().reset_index()
        
        if clean_data.empty: continue
        
        K_raw = clean_data['strike'].values
        C_raw = clean_data['target_price'].values

        # B1. Smart Wing Filter
        if filter_wings_on:
            K_raw, C_raw = filter_wings(K_raw, C_raw, F, T, r, quantile_tol=wing_tol)
        
        if len(K_raw) < 3: continue
        
        # B2. Stride Filter & Repair
        K_vec, C_vec = filter_strikes_by_value(K_raw, C_raw, dollar_stride)
        
        if repair:
            try:
                K_vec, C_vec = repair_convexity(K_vec, C_vec, strict_convexity=True, epsilon=1e-4)
            except: pass
        
        if len(K_vec) < 3: continue

        # C. Dynamic Grid Calculation
        # "Choose lowest and highest hundreds"
        min_k_curr = K_vec[0]
        max_k_curr = K_vec[-1]
        
        grid_min = np.floor(min_k_curr / 100.0) * 100.0
        grid_max = np.ceil(max_k_curr / 100.0) * 100.0
        
        # Auto-calc nodes
        width = grid_max - grid_min
        target_step = dollar_stride * grid_multiple
        n_nodes = int(width / target_step) + 1
        n_nodes = min(max(n_nodes, 10), 10000) # Safety cap

        # D. SVI Prior
        prior_pdf = None
        temp_x = np.linspace(grid_min, grid_max, n_nodes)
        
        if use_svi:
            params = load_svi_params(ticker, exp)
            if params is not None:
                svi = SVIModel([params.a, params.b, params.rho, params.m, params.sigma])
                k_grid_log = np.log(temp_x / F)
                prior_vals = [svi.get_density(k) / x_val for k, x_val in zip(k_grid_log, temp_x)]
                prior_pdf = np.array(prior_vals)

        # E. Solve
        model = MaxEntModel(F, T, n_nodes=n_nodes, custom_prior=prior_pdf, grid_bounds=(grid_min, grid_max))
        res = model.solve(K_vec, C_vec)
        
        solved_models.append({
            'expiry': exp,
            'x': model.x,
            'pdf': model.pdf
        })
            
        progress_bar.progress((i + 1) / len(unique_expiries))

    # --- PASS 2: ALIGN & INTERPOLATE ---
    if not solved_models:
        st.error("No expiries converged. Try adjusting 'Min Strike Gap' or 'Grid Buffer'.")
    else:
        status.text("Building 3D Surface...")
        
        # 1. Determine Global Bounds from ACTUAL solved models (avoids outliers in raw data)
        all_x = np.concatenate([m['x'] for m in solved_models])
        #global_min = np.min(all_x)
        #global_max = np.max(all_x)
        global_min = 400
        global_max = 900

        
        # 2. Create Common Grid
        common_strikes = np.linspace(global_min, global_max, surf_res)
        
        Z_rows = []
        Y_labels = []
        
        for m in solved_models:
            # Interpolate onto global grid
            pdf_interp = np.interp(common_strikes, m['x'], m['pdf'], left=0.0, right=0.0)
            
            Z_rows.append(pdf_interp)
            Y_labels.append(m['expiry'])
            
        Z = np.array(Z_rows) # Shape: (N_Expiries, N_Strikes)
        X = common_strikes
        Y = Y_labels # Using strings often safer for Categorical Y-Axis in Surface
        
        # Check for blank plot condition
        if Z.max() == 0:
            st.warning("Surface is flat (all zeros). Try increasing 'Surface Plot Resolution'.")
        
        # 3. Plot
        fig = go.Figure(data=[go.Surface(
            z=Z, 
            x=X, 
            y=Y, 
            colorscale='temps',
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )])
        
        fig.update_layout(
            title=f"MaxEnt Implied Density Surface ({ticker})",
            scene=dict(
                xaxis_title='Strike ($)',
                yaxis_title='Expiry',
                zaxis_title='Density',
                camera=dict(eye=dict(x=1.8, y=1.8, z=0.5)),
                aspectratio=dict(x=1, y=1, z=0.6)
            ),
            width=1200, height=900,
            margin=dict(l=10, r=10, b=10, t=50)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Debug Data
        with st.expander("Show Matrix Data"):
            st.write(f"Grid Range: {global_min:.2f} to {global_max:.2f}")
            st.write(f"Max Density Observed: {Z.max():.6f}")
            df_surf = pd.DataFrame(Z, index=Y, columns=np.round(X, 1))
            st.dataframe(df_surf)
            
    status.empty()