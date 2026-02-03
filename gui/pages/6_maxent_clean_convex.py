import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import bisect

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from analytics.maxent import MaxEntModel
from analytics.svi import SVIModel
from analytics.arbitrage import repair_convexity

st.set_page_config(layout="wide", page_title="MaxEnt Lab")
DATA_DIR = "data/processed"
MODEL_DIR = "data/models"

# --- HELPER: Dollar Stride Filter ---
def filter_strikes_by_value(strikes, prices, min_diff):
    """
    Selects strikes such that K[i+1] >= K[i] + min_diff.
    Always includes the first and last strike to preserve range.
    """
    if len(strikes) == 0: return np.array([]), np.array([])
    if min_diff <= 0: return strikes, prices
    
    keep_k = [strikes[0]]
    keep_p = [prices[0]]
    
    for k, p in zip(strikes[1:], prices[1:]):
        if k >= keep_k[-1] + min_diff:
            keep_k.append(k)
            keep_p.append(p)
    
    if keep_k[-1] != strikes[-1]:
         keep_k.append(strikes[-1])
         keep_p.append(prices[-1])
            
    return np.array(keep_k), np.array(keep_p)

# --- HELPER: Plot Layering ---
def add_liquidity_layer(fig, df_liq, yaxis_name='y2'):
    """Overlays Volume and OI on the specified axis."""
    fig.add_trace(go.Bar(
        x=df_liq['strike'], 
        y=df_liq['volume'], 
        name='Volume', 
        marker_color='cyan', 
        opacity=0.15,
        yaxis=yaxis_name,
        hoverinfo='y+name'
    ))
    fig.add_trace(go.Scatter(
        x=df_liq['strike'], 
        y=df_liq['openInterest'], 
        name='Open Interest', 
        line=dict(color='lime', width=1, dash='dot'), 
        opacity=0.4,
        yaxis=yaxis_name,
        hoverinfo='y+name'
    ))
    return fig

# --- HELPER: Sync Logic ---
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# --- Loaders ---
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

# --- Main App ---
st.title("Non-Parametric Density (MaxEnt)")
col1, col2 = st.columns(2)
with col1:
    tickers = [d for d in os.listdir(DATA_DIR)] if os.path.exists(DATA_DIR) else []
    ticker = st.selectbox("Ticker", tickers, key='ticker_select')

if not ticker: st.stop()
df = load_clean_data(ticker)
df['expiry'] = df['expiry'].astype(str)
expiry = st.sidebar.selectbox("Expiration", sorted(df['expiry'].unique()), key='expiry_select')

# --- Data Preparation ---
subset = df[df['expiry'] == expiry].copy()
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
clean_data = clean_data.groupby('strike').agg({
    'target_price': 'mean',
    'volume': 'sum',
    'openInterest': 'sum'
}).reset_index()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("---")
    st.header("Preprocessing")
    
    repair = st.checkbox("Auto-Repair Convexity", value=True, key="chk_repair")
    dollar_stride = st.number_input(
        "Min Strike Gap ($)", 
        min_value=0.0, value=1.0, step=0.5, 
        help="Strike Gap (gap)", key="num_stride"
    )
    
    # --- STRIKE RANGE SELECTION (Synced) ---
    if not clean_data.empty:
        avail_strikes = clean_data['strike'].values
        min_full = avail_strikes[0]
        max_full = avail_strikes[-1]
        
        # We create a unique ID for the current dataset
        current_selection_id = f"{ticker}_{expiry}"

        # If the user switched ticker, the old session state values (e.g. 760.0) 
        # might be invalid for the new chain. We must reset them.
        if 'last_selection_id' not in st.session_state or st.session_state.last_selection_id != current_selection_id:
            st.session_state.range_min = min_full
            st.session_state.range_max = max_full
            st.session_state.last_selection_id = current_selection_id

        st.subheader("Strike Range")

        st.subheader("Strike Range")
        
        # Callback: Slider -> Inputs
        def update_from_slider():
            val = st.session_state.range_slider
            st.session_state.range_min = val[0]
            st.session_state.range_max = val[1]

        # Callback: Inputs -> Slider
        def update_from_inputs():
            # Get current input values (from the widgets directly or state)
            # We use the widget keys to access the NEW values
            raw_min = st.session_state.input_min_key
            raw_max = st.session_state.input_max_key
            
            # Snap to nearest valid strike
            snap_min = find_nearest(avail_strikes, raw_min)
            snap_max = find_nearest(avail_strikes, raw_max)
            
            # Ensure Max >= Min
            if snap_max < snap_min: snap_max = snap_min
            
            st.session_state.range_min = snap_min
            st.session_state.range_max = snap_max

        # Guard against float precision errors by finding nearest valid index
        # This ensures the 'value' passed to select_slider ALWAYS exists in 'options'
        safe_min = find_nearest(avail_strikes, st.session_state.range_min)
        safe_max = find_nearest(avail_strikes, st.session_state.range_max)

        # 1. Slider (Snaps to Strike)
        # We set 'value' to the current state. 
        # Note: If state values are not in 'options' due to some rounding, this might warn, 
        # but find_nearest logic in update_inputs ensures they stay valid.
        st.select_slider(
            "Quick Select (Snaps to Strike)",
            options=avail_strikes,
            value=(safe_min, safe_max),
            key="range_slider",
            on_change=update_from_slider
        )

        # 2. Inputs (Precise)
        c1, c2 = st.columns(2)
        c1.number_input("Min Strike", value=float(st.session_state.range_min), key="input_min_key", on_change=update_from_inputs)
        c2.number_input("Max Strike", value=float(st.session_state.range_max), key="input_max_key", on_change=update_from_inputs)
        
        # --- LIVE COUNTS ---
        # 1. Filter Range
        range_data = clean_data[
            (clean_data['strike'] >= st.session_state.range_min) & 
            (clean_data['strike'] <= st.session_state.range_max)
        ].copy()            
    else:
        st.stop()

    st.markdown("---")
    st.header("Grid Settings")
    
    # [NEW] Grid Logic
    grid_buffer_dollars = st.number_input(
        "Buffer ($)", min_value=0.0, value=0.0, step=5.0,
        help="Amount to extend grid beyond Min/Max strikes."
    )
    
    grid_multiple = st.slider(
        "Grid Multiplier", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="Gap Size / Grid Step. 1.0 = Grid nodes align with gap size. 0.5 = Twice as dense."
    )
    
    # Calculate Nodes
    # Formula: Total Width / (Gap * Multiple)
    width = (st.session_state.range_max - st.session_state.range_min) + (2 * grid_buffer_dollars)
    target_step = dollar_stride * grid_multiple
    
    calc_nodes = int(width / target_step) + 1
    
    st.info(f"Auto-Calc: {calc_nodes} nodes (Step ~${target_step:.2f})")
    
    n_nodes = st.number_input("Grid Nodes", min_value=10, max_value=10000, value=calc_nodes, step=10)
    
    use_svi = st.checkbox("Use SVI Prior", value=True, key="chk_svi")
        
    run = st.button("Solve", type="primary")

if run:
    # Use the calculated range_data from sidebar (it's already filtered by Min/Max)
    K_raw = range_data['strike'].values
    C_raw = range_data['target_price'].values
    
    # 1. Apply Dollar Stride Filter (Real Run)
    K_vec, C_vec = filter_strikes_by_value(K_raw, C_raw, dollar_stride)
    
    # 2. Repair Convexity
    if repair:
        C_filtered = C_vec.copy()
        # Use strict mode for better density stability
        K_vec, C_vec = repair_convexity(K_vec, C_vec, strict_convexity=True, epsilon=1e-4)
        
        diffs = C_vec - C_filtered
        adj_count = np.sum(diffs < -1e-5)
        if adj_count > 0:
            st.toast(f"🔧 Repaired {adj_count} prices.")

    if len(K_vec) < 3:
        st.error("Too few points remaining.")
        st.stop()

    # 3. Dynamic Grid
    grid_min = min(K_vec) - grid_buffer_dollars
    grid_max = max(K_vec) + grid_buffer_dollars
    
    # 4. Solve
    model = MaxEntModel(F, T, n_nodes=n_nodes, grid_bounds=(grid_min, grid_max))
    prior_pdf = None
    if use_svi:
        params = load_svi_params(ticker, expiry)
        if params is not None:
            svi = SVIModel([params.a, params.b, params.rho, params.m, params.sigma])
            k_grid = np.log(model.x / F)
            prior_vals = [svi.get_density(k) / x for k, x in zip(k_grid, model.x)]
            prior_pdf = np.array(prior_vals)
    
    model = MaxEntModel(F, T, n_nodes=n_nodes, custom_prior=prior_pdf, grid_bounds=(grid_min, grid_max))
    res = model.solve(K_vec, C_vec)
    
    if res['success']:
        st.success(f"Converged! Error: {res['error']:.5f}")
    else:
        st.warning(f"Solver stopped: {res['message']}")

    col_l, col_r = st.columns(2)
    
    # --- PLOT 1: PRICING (with Volume/OI) ---
    fig_p = go.Figure()
    fig_p = add_liquidity_layer(fig_p, range_data, yaxis_name='y2')
    
    fig_p.add_trace(go.Scatter(x=K_raw, y=C_raw, mode='markers', name='Raw Input', marker=dict(color='gray', opacity=0.3)))
    fig_p.add_trace(go.Scatter(x=K_vec, y=C_vec, mode='markers', name='Used Input', marker=dict(color='red', symbol='x')))
    fig_p.add_trace(go.Scatter(x=K_vec, y=model.model_prices, mode='lines', name='MaxEnt Fit', line=dict(color='blue')))
    
    fig_p.update_layout(
        title="Pricing Check", 
        height=450,
        yaxis=dict(title="Option Price"),
        yaxis2=dict(title="Liquidity", overlaying='y', side='right', showgrid=False)
    )
    col_l.plotly_chart(fig_p, width='stretch')
    
    # --- PLOT 2: ARBITRAGE (with Volume/OI) ---
    dC_dK = np.diff(C_vec) / np.diff(K_vec)
    mid_k = (K_vec[1:] + K_vec[:-1]) / 2
    conv = np.diff(dC_dK)
    mid_k_conv = (mid_k[1:] + mid_k[:-1]) / 2
    
    fig_arb = go.Figure()
    fig_arb = add_liquidity_layer(fig_arb, range_data, yaxis_name='y3')
    
    fig_arb.add_trace(go.Scatter(x=mid_k, y=dC_dK, name='Dual Delta', line=dict(color='orange')))
    fig_arb.add_trace(go.Scatter(x=mid_k_conv, y=conv*100, name='Convexity', line=dict(color='green'), yaxis='y2'))
    
    fig_arb.update_layout(
        title="Arbitrage Check", 
        height=450,
        yaxis=dict(title="Dual Delta"),
        yaxis2=dict(title="Convexity (x100)", overlaying='y', side='right'),
        yaxis3=dict(overlaying='y', side='right', position=0.95, showgrid=False, visible=False) 
    )
    col_r.plotly_chart(fig_arb, width='stretch')
    
    # --- PLOT 3: DENSITY (with Volume/OI) ---
    fig_d = go.Figure()
    fig_d = add_liquidity_layer(fig_d, range_data, yaxis_name='y2')
    
    if prior_pdf is not None:
        fig_d.add_trace(go.Scatter(x=model.x, y=model.prior/model.dx, name='Prior', line=dict(dash='dash', color='gray')))
    fig_d.add_trace(go.Scatter(x=model.x, y=model.pdf, fill='tozeroy', name='MaxEnt Density', line=dict(color='purple')))
    
    fig_d.update_layout(
        title="Implied Distribution", 
        height=500,
        yaxis=dict(title="Density"),
        yaxis2=dict(title="Liquidity", overlaying='y', side='right', showgrid=False)
    )
    st.plotly_chart(fig_d, width='stretch')