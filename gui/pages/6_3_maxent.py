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

st.set_page_config(layout="wide", page_title="MaxEnt Lab")
DATA_DIR = "data/processed"
MODEL_DIR = "data/models"

# --- HELPER: Dollar Stride Filter ---
def filter_strikes_by_value(strikes, prices, min_diff):
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

# --- HELPER: Tail Filter (Logic 1) ---
def filter_left_tail(K, C, window_size=20.0):
    """
    Scans the left tail for the first stable region (low noise, convexity ~0)
    and returns the strike where the valid range should start.
    """
    if len(K) < 10: return K[0]
    
    dK, dC = np.diff(K), np.diff(C)
    delta = dC / dK
    mid_k = (K[1:] + K[:-1]) / 2
    gamma = np.diff(delta) / np.diff(mid_k) # Gamma at K[1]..K[N-2]
    
    candidates = []

    # 2. Define Scan Limit based on Gamma Threshold (0.005)
    # We look for the first point where gamma spikes.
    # gamma[i] roughly corresponds to convexity around K[i+1]
    high_gamma_indices = np.where(gamma > 0.004)[0]
    
    if len(high_gamma_indices) > 0:
        # Stop scanning before we hit the high gamma region
        # We add +1 because gamma[i] is aligned with K[i+1]
        limit_idx = high_gamma_indices[0] + 1
        scan_limit = K[limit_idx]
    else:
        # If gamma never spikes (flat curve?), scan the whole thing
        scan_limit = K[-1]
        
    
    for i in range(len(K)):
        start_k = K[i]
        if start_k > scan_limit: break
        
        # Window logic
        mask = (K >= start_k) & (K < start_k + window_size)
        indices = np.where(mask)[0]
        
        # Need enough points for stats
        if len(indices) < 4: continue
        
        # Map to Gamma indices (shift -1)
        idx_g = indices - 1
        idx_g = idx_g[(idx_g >= 0) & (idx_g < len(gamma))]
        if len(idx_g) < 2: continue
        
        win_gamma = gamma[idx_g]
        win_delta = delta[indices[indices < len(delta)]]
        
        mag_gamma = np.mean(np.abs(win_gamma))
        # Score: Minimize Delta movement and Gamma noise
        score = np.std(win_delta) + (100 * np.std(win_gamma))
        
        candidates.append({'start_k': start_k, 'mag': mag_gamma, 'score': score})
        
    if not candidates: return K[0]
    
    df_c = pd.DataFrame(candidates)
    # Filter for low convexity (approx 0) then find lowest movement score
    # Use median gamma of the tail as a baseline for "low"
    thresh = max(df_c['mag'].quantile(0.5), 1e-5)
    valid = df_c[df_c['mag'] <= thresh]
    
    if valid.empty: 
        best = df_c.loc[df_c['score'].idxmin()]
    else: 
        best = valid.loc[valid['score'].idxmin()]
    
    return best['start_k']

# --- HELPER: Convexity Ratio (Logic 2) ---
def get_convexity_ratio(K, C):
    """
    Returns ratio of (Strike Range where Gamma > 0.05) / (Total Strike Range).
    """
    if len(K) < 5: return 1.0 # Default to keep structure
    
    dK, dC = np.diff(K), np.diff(C)
    delta = dC / dK
    mid_k = (K[1:] + K[:-1]) / 2
    gamma = np.diff(delta) / np.diff(mid_k)
    
    # Check where Gamma exceeds 0.05
    # Gamma[i] corresponds roughly to K[i+1]
    high_conv_indices = np.where(gamma > 0.005)[0]
    
    if len(high_conv_indices) == 0:
        return 0.0 # No significant convexity
        
    # First index from left, Last index from right (which is just last idx of the array)
    k_start = K[high_conv_indices[0] + 1]
    k_end = K[high_conv_indices[-1] + 1]
    
    core_range = k_end - k_start
    total_range = K[-1] - K[0]
    
    if total_range == 0: return 0.0
    return core_range / total_range

# --- HELPER: Range Adjustment (Logic 3) ---
def adjust_range_modulo(K, C, step_size):
    """
    Truncates K (and C) minimally such that (K_max - K_min) is divisible by step_size.
    Prioritizes keeping the widest possible range.
    """
    if len(K) < 2: return K, C
    
    N = len(K)
    best_start, best_end = 0, N-1
    max_width = -1.0
    
    # Search for largest subset. 
    # Since we likely only need to shave off a few points, we search near full length.
    # Limit search to trimming up to 10 points from either side to save time.
    limit_trim = min(15, N)
    
    for i in range(limit_trim): # Trim from left
        for j in range(N-1, N-1-limit_trim, -1): # Trim from right
            if j <= i: continue
            
            w = K[j] - K[i]
            # Check float divisibility
            rem = w % step_size
            if min(rem, step_size - rem) < 1e-4:
                if w > max_width:
                    max_width = w
                    best_start, best_end = i, j
    
    # If no subset found (rare), return original or severe fallback
    if max_width < 0:
        return K, C
        
    return K[best_start:best_end+1], C[best_start:best_end+1]

def get_safe_strike_range(K, C, delta_threshold=0.005):
             dK, dC = np.diff(K), np.diff(C)
             delta = dC / dK
             # Pad to match K size
             delta = np.concatenate([delta, [delta[-1]]])
             
             # Valid region: Delta strictly between -0.99 and -0.01
             mask = (delta > (-1.0 + delta_threshold)) & (delta < -delta_threshold)
             
             if np.sum(mask) < 3: return K[0], K[-1] # Fallback
             
             # Find contiguous range
             idx = np.where(mask)[0]
             return K[idx[0]], K[idx[-1]]

def filter_collinear_strikes(K, C, min_butterfly=1e-3):
    """
    Removes strikes where the price is effectively linear relative to neighbors.
    (i.e., Butterfly Spread value < min_butterfly).
    This prevents 'Zero Density' constraints that crash MaxEnt.
    """
    # Iterative removal to handle long chains of linear points
    while True:
        if len(K) < 3: break
        
        # Calculate Butterfly Spread (Curvature proxy)
        # Butterfly = P(i-1) - 2P(i) + P(i+1)
        # But for non-uniform grid, we use the "height below chord" metric:
        # P_chord = w1*P_left + w2*P_right
        
        k_left = K[:-2]
        k_center = K[1:-1]
        k_right = K[2:]
        
        p_left = C[:-2]
        p_center = C[1:-1]
        p_right = C[2:]
        
        # Calculate Linear Interpolation Price at Center
        # (p_right - p_left) / (k_right - k_left) * (k_center - k_left) + p_left
        p_chord = p_left + (p_right - p_left) * (k_center - k_left) / (k_right - k_left)
        
        # "Sag" is how far the price is below the linear chord (Convexity)
        # We want Sag > Threshold. If Sag ~ 0, it's linear.
        sag = p_chord - p_center
        
        # Find indices where convexity is too small (Linear or Concave)
        # We check for sag < threshold. 
        # Note: repair_convexity ensures sag >= 0, so we just check near-zero.
        bad_indices = np.where(sag < min_butterfly)[0]
        
        if len(bad_indices) == 0:
            break
            
        # Remove the "worst" point (most linear) in this pass
        # We remove only one per pass (or non-adjacent set) to preserve structure,
        # but for speed, removing the global minimum curvature is safest.
        
        # Find index with smallest sag (closest to linear)
        worst_idx_in_sag = np.argmin(sag)
        real_idx_to_drop = worst_idx_in_sag + 1 # +1 offset because sag array is shortened
        
        # Debug:
        # print(f"Dropping Strike {K[real_idx_to_drop]} (Sag: {sag[worst_idx_in_sag]:.6f})")
        
        K = np.delete(K, real_idx_to_drop)
        C = np.delete(C, real_idx_to_drop)
        
    return K, C

# --- HELPER: Sync Logic ---
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def add_liquidity_layer(fig, df_liq, yaxis_name='y2'):
    fig.add_trace(go.Bar(x=df_liq['strike'], y=df_liq['volume'], name='Volume', marker_color='cyan', opacity=0.15, yaxis=yaxis_name, hoverinfo='y+name'))
    fig.add_trace(go.Scatter(x=df_liq['strike'], y=df_liq['openInterest'], name='Open Interest', line=dict(color='lime', width=1, dash='dot'), opacity=0.4, yaxis=yaxis_name, hoverinfo='y+name'))
    return fig
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
def put_to_call(put_price, K, F): return put_price + (F - K)

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
clean_data = clean_data.groupby('strike').agg({'target_price': 'mean', 'volume': 'sum', 'openInterest': 'sum'}).reset_index()

with st.sidebar:
    st.markdown("---")
    st.header("Preprocessing")
    repair = st.checkbox("Auto-Repair Convexity", value=True, key="chk_repair")
    dollar_stride = st.number_input("Min Strike Gap ($)", min_value=0.0, value=1.0, step=0.5, key="num_stride")
    
    if not clean_data.empty:
        avail_strikes = clean_data['strike'].values
        min_full, max_full = avail_strikes[0], avail_strikes[-1]
        curr_id = f"{ticker}_{expiry}"
        if 'last_id' not in st.session_state or st.session_state.last_id != curr_id:
            st.session_state.range_min = min_full
            st.session_state.range_max = max_full
            st.session_state.last_id = curr_id

        st.subheader("Strike Range (Pre-Filter)")
        def update_slider():
            st.session_state.range_min = st.session_state.slider[0]
            st.session_state.range_max = st.session_state.slider[1]
        def update_inputs():
            st.session_state.range_min = find_nearest(avail_strikes, st.session_state.in_min)
            st.session_state.range_max = find_nearest(avail_strikes, st.session_state.in_max)
        
        s_min = find_nearest(avail_strikes, st.session_state.range_min)
        s_max = find_nearest(avail_strikes, st.session_state.range_max)
        st.select_slider("Select", options=avail_strikes, value=(s_min, s_max), key="slider", on_change=update_slider)
        c1, c2 = st.columns(2)
        c1.number_input("Min", value=float(st.session_state.range_min), key="in_min", on_change=update_inputs)
        c2.number_input("Max", value=float(st.session_state.range_max), key="in_max", on_change=update_inputs)
        
        range_data = clean_data[(clean_data['strike'] >= s_min) & (clean_data['strike'] <= s_max)].copy()
    else: st.stop()

    st.markdown("---")
    st.header("Grid Settings")
    grid_multiple = st.slider("Base Multiplier", 0.1, 10.0, 1.0, 0.1)
    
    # Placeholder for estimated nodes (updated at runtime)
    st.info("Nodes will be calculated dynamically based on filtered range.")
    
    use_svi = st.checkbox("Use SVI Prior", value=True)
    run = st.button("Solve", type="primary")

if run:
    K_raw = range_data['strike'].values
    C_raw = range_data['target_price'].values
    
    # 1. Stride Filter
    K_vec, C_vec = filter_strikes_by_value(K_raw, C_raw, dollar_stride)
    
    # 2. Repair Convexity
    if repair:
        C_orig = C_vec.copy()
        K_vec, C_vec = repair_convexity(K_vec, C_vec, strict_convexity=True, epsilon=1e-4)
        if np.sum(C_vec - C_orig < -1e-5) > 0: st.toast("Repaired convexity.")

    safe_min, safe_max = get_safe_strike_range(K_vec, C_vec, delta_threshold=0.005)
        
    mask = (K_vec >= safe_min) & (K_vec <= safe_max)
    if np.sum(mask) > 10:
            K_vec, C_vec = K_vec[mask], C_vec[mask]
            st.caption(f"✂️ Truncated Empty Tails: [{safe_min:.1f} ... {safe_max:.1f}]")

    # 4. [NEW] Body Thinning (Collinearity Filter)
    # Remove ATM linear segments that crash the Hessian
    # min_butterfly=1e-4 roughly corresponds to "Zero Density" for the solver.
    #initial_len = len(K_vec)
    #K_vec, C_vec = filter_collinear_strikes(K_vec, C_vec, min_butterfly=1e-4)
    
    #if len(K_vec) < initial_len:
    #    st.toast(f"🧹 Thinned {initial_len - len(K_vec)} linear strikes.")

    # 4. Determine Grid Multiplier (Convexity Logic)
    ratio = get_convexity_ratio(K_vec, C_vec)
    effective_mult = 1.0 if ratio < 0.25 else grid_multiple
    
    if effective_mult != grid_multiple:
        st.info(f"⚡ High Tail Concentration (Ratio {ratio:.2f}). Overriding Multiplier: **{effective_mult}**")
    
    # 5. Calculate Target Step & Adjust Range
    target_step = dollar_stride * effective_mult
    
        
    # Adjust K_vec so (Max - Min) is divisible by step
    K_vec, C_vec = adjust_range_modulo(K_vec, C_vec, target_step)
    
    if len(K_vec) < 3:
        st.error("Range too small after adjustment.")
        st.stop()
        
    # 6. Recalculate Grid Nodes
    current_width = K_vec[-1] - K_vec[0]
    
    
    # Since we enforced divisibility, this should be very close to integer
    n_nodes = int(round(current_width / target_step)) + 1
    n_nodes = max(10, n_nodes)
    
    st.write(f"**Grid Config**: Range [{K_vec[0]} - {K_vec[-1]}] | Width {current_width} | Step {target_step} | Nodes {n_nodes}")

    # 7. Solve
    grid_min = K_vec[0]
    grid_max = K_vec[-1]
    
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
    
    if res['success']: st.success(f"Converged! Error: {res['error']:.5f}")
    else: st.warning(f"Solver stopped: {res['message']}")

    # --- PLOTS ---
    c_l, c_r = st.columns(2)
    
    fig_p = go.Figure()
    fig_p = add_liquidity_layer(fig_p, range_data)
    fig_p.add_trace(go.Scatter(x=K_raw, y=C_raw, mode='markers', name='Raw', marker=dict(color='gray', opacity=0.3)))
    fig_p.add_trace(go.Scatter(x=K_vec, y=C_vec, mode='markers', name='Used', marker=dict(color='red', symbol='x')))
    fig_p.add_trace(go.Scatter(x=K_vec, y=model.model_prices, mode='lines', name='Fit', line=dict(color='blue')))
    fig_p.update_layout(title="Pricing", height=400)
    c_l.plotly_chart(fig_p, width='stretch')
    
    dC_dK = np.diff(C_vec) / np.diff(K_vec)
    mid_k = (K_vec[1:] + K_vec[:-1]) / 2
    conv = np.diff(dC_dK)
    
    fig_arb = go.Figure()
    fig_arb.add_trace(go.Scatter(x=mid_k, y=dC_dK, name='Delta'))
    fig_arb.add_trace(go.Scatter(x=mid_k[:-1], y=conv*100, name='Gamma x100', yaxis='y2'))
    fig_arb.update_layout(title="Arbitrage", height=400, yaxis2=dict(overlaying='y', side='right'))
    c_r.plotly_chart(fig_arb, width='stretch')
    
    fig_d = go.Figure()
    if prior_pdf is not None:
        fig_d.add_trace(go.Scatter(x=model.x, y=model.prior/model.dx, name='Prior', line=dict(dash='dash', color='gray')))
    fig_d.add_trace(go.Scatter(x=model.x, y=model.pdf, fill='tozeroy', name='MaxEnt', line=dict(color='purple')))
    fig_d.update_layout(title="Density", height=500)
    st.plotly_chart(fig_d, width='stretch')
    
    df_out = pd.DataFrame({"Strike": model.x, "PDF": model.pdf, "Prob": model.pdf*model.dx})
    st.dataframe(df_out, width='stretch')