import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from analytics.maxent import MaxEntModel
from analytics.svi import SVIModel
from analytics.arbitrage import repair_convexity

st.set_page_config(layout="wide", page_title="Surface Lab 3D")
DATA_DIR = "data/clean"
MODEL_DIR = "data/models"

# --- HELPER FUNCTIONS ---
def get_available_snapshots(ticker):
    p_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(p_dir): return []
    files = sorted([f for f in os.listdir(p_dir) if f.endswith('.parquet')], reverse=True)
    return files

def load_snapshot(ticker, filename):
    return pd.read_parquet(os.path.join(DATA_DIR, ticker, filename))

def load_svi_params(ticker, expiry_str):
    m_dir = os.path.join(MODEL_DIR, ticker)
    if not os.path.exists(m_dir): return None
    files = sorted([f for f in os.listdir(m_dir) if f.endswith('.csv')])
    if not files: return None
    df_params = pd.read_csv(os.path.join(m_dir, files[-1]))
    row = df_params[df_params['expiry'].astype(str) == expiry_str]
    if row.empty: return None
    return row.iloc[0]

def filter_strikes_by_value(strikes, prices, volumes, ois, min_diff):
    if len(strikes) == 0: return np.array([]), np.array([]), np.array([]), np.array([])
    if min_diff <= 0: return strikes, prices, volumes, ois
    keep_idx = [0]
    for i in range(1, len(strikes)):
        if strikes[i] >= strikes[keep_idx[-1]] + min_diff:
            keep_idx.append(i)
    if keep_idx[-1] != len(strikes) - 1:
        keep_idx.append(len(strikes) - 1)
    idx = np.array(keep_idx)
    return strikes[idx], prices[idx], volumes[idx], ois[idx]

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# --- SURFACE GENERATOR ---
class SurfaceGenerator:
    def __init__(self, ticker, df_data):
        self.ticker = ticker
        self.df = df_data
        self.all_expiries = sorted(self.df['expiry'].unique())
        
    def generate(self, config, name, exclude_expiries=[]):
        surface_data = []
        active_expiries = [e for e in self.all_expiries if e not in exclude_expiries]
        if not active_expiries: return None
        
        g_min, g_max = config['range_min'], config['range_max']
        mesh_size = 100 
        common_x = np.linspace(g_min, g_max, mesh_size)
        mesh_z, mesh_t, mesh_dates = [], [], []
        
        progress_bar = st.progress(0, text=f"Building Surface: {name}")
        
        for i, exp in enumerate(active_expiries):
            progress_bar.progress((i + 1) / len(active_expiries))
            
            slice_ = self.df[self.df['expiry'] == exp].copy()
            if slice_.empty: continue
            
            T = slice_['T'].iloc[0]
            F = slice_['F'].iloc[0]
            r = slice_['r'].iloc[0]
            
            # Data Prep
            slice_['target'] = np.nan
            mask_c = (slice_['type'] == 'C') & (slice_['strike'] >= F)
            slice_.loc[mask_c, 'target'] = slice_.loc[mask_c, 'mid'] * np.exp(r*T)
            mask_p = (slice_['type'] == 'P') & (slice_['strike'] < F)
            p_raw = slice_.loc[mask_p, 'mid'] * np.exp(r*T)
            slice_.loc[mask_p, 'target'] = p_raw + (F - slice_.loc[mask_p, 'strike'])
            
            clean = slice_.dropna(subset=['target']).sort_values('strike').groupby('strike').agg({
                'target': 'mean', 'volume': 'sum', 'openInterest': 'sum'
            }).reset_index()
            
            K_vec = clean['strike'].values; C_vec = clean['target'].values
            Vol_vec = clean['volume'].values; OI_vec = clean['openInterest'].values
            
            if config['gap'] > 0:
                K_vec, C_vec, Vol_vec, OI_vec = filter_strikes_by_value(K_vec, C_vec, Vol_vec, OI_vec, config['gap'])
            if config['repair']:
                K_vec, C_vec = repair_convexity(K_vec, C_vec, strict_convexity=True, epsilon=1e-4)
            if len(K_vec) < 3: continue

            # --- [FIX] Page 6 Grid Logic Ported Exact ---
            # 1. Additive Buffer ($)
            #buffer_val = config['buffer_dollars']
            
            # 2. Exact Boundaries with 1e-5 Jitter
            l_min = min(K_vec) #- buffer_val - 1e-5
            l_max = max(K_vec) #+ buffer_val + 1e-5
            
            # 3. Exact Node Calculation
            # Nodes = Width / (Gap * Mult)
            width = l_max - l_min
            step = config['gap'] * config['grid_mult']
            n_nodes = int(width / step) + 1
            
            # Clamp safety (Page 6 max was 10000, we keep it reasonable for 3D perf)
            n_nodes = max(50, min(5000, n_nodes))
            
            # SVI Prior 
            prior_pdf, svi_pdf = None, None
            params = load_svi_params(self.ticker, str(exp))
            temp_model = MaxEntModel(F, T, n_nodes=n_nodes, grid_bounds=(l_min, l_max))
            
            if params is not None:
                svi = SVIModel([params.a, params.b, params.rho, params.m, params.sigma])
                k_grid_svi = np.log(temp_model.x / F)
                svi_vals = [svi.get_density(k)/x if x>0 else 0 for k, x in zip(k_grid_svi, temp_model.x)]
                svi_pdf = np.array(svi_vals)
                if config['use_prior']: prior_pdf = svi_pdf

            x_local = temp_model.x
            y_local = None

            if config['mode'] == 'MaxEnt':
                model = MaxEntModel(F, T, n_nodes=n_nodes, custom_prior=prior_pdf, grid_bounds=(l_min, l_max))
                res = model.solve(K_vec, C_vec)
                
                # Best Effort Check (Matches Page 6 visual output)
                if res['success']:
                    st.success(f"Converged! Error: {res['error']:.5f}")
                    y_local = model.pdf
                else:
                    st.warning(f"Solver stopped: {res['message']}")
                    y_local = model.pdf
            else:
                if svi_pdf is not None: y_local = svi_pdf

            if y_local is None: continue 

            # Stitch & Interpolate
            x_left = np.linspace(g_min, l_min, num=20)[:-1]
            x_right = np.linspace(l_max, g_max, num=20)[1:]
            x_global = np.concatenate([x_left, x_local, x_right])
            y_global = np.concatenate([np.zeros_like(x_left), y_local, np.zeros_like(x_right)])
            
            y_interpolated = np.interp(common_x, x_global, y_global, left=0, right=0)
            mesh_z.append(y_interpolated)
            mesh_t.append(T)
            mesh_dates.append(str(exp))
            
            surface_data.append({
                'expiry': str(exp), 'T': T,
                'x_local': x_local, 'y_local': y_local,
                'x_global': x_global, 'y_global': y_global,
                'strikes_pos': K_vec, 'vol_pos': Vol_vec, 'oi_pos': OI_vec
            })
            
        progress_bar.empty()
        
        return {
            'slices': surface_data,
            'mesh': {
                'x': common_x,
                'y': np.array(mesh_t),
                'z': np.array(mesh_z),
                'dates': mesh_dates
            }
        }

# --- UI ---
if 'surfaces' not in st.session_state: st.session_state.surfaces = [] 

st.title("Surface Lab 3D")

with st.sidebar:
    st.header("1. Data Source")
    tickers = [d for d in os.listdir(DATA_DIR)] if os.path.exists(DATA_DIR) else []
    ticker = st.selectbox("Ticker", tickers)
    
    snapshots = get_available_snapshots(ticker)
    if not snapshots: st.stop()
    sel_snapshot = st.selectbox("Snapshot", snapshots)
    df = load_snapshot(ticker, sel_snapshot)
    
    all_strikes = sorted(df['strike'].unique())
    all_dates = sorted(df['expiry'].unique())
    min_k, max_k = all_strikes[0], all_strikes[-1]
    
    st.markdown("---")
    st.header("2. Surface Builder")
    
    exclude = st.multiselect("Exclude Expiries", all_dates)
    
    with st.form("builder"):
        surf_name = st.text_input("Name", value=f"Surf {len(st.session_state.surfaces)+1}")
        c1, c2 = st.columns(2)
        mode = c1.selectbox("Model", ["MaxEnt", "SVI Prior Only"])
        use_prior = c2.checkbox("Use Prior", True)
        
        st.markdown("##### Optimization")

        gap = st.number_input("Gap ($)", 0.1, 50.0, 1.0)
        grid_mult = st.number_input("Grid Mult", 0.1, 5.0, 1.0)
        repair = st.checkbox("Repair Convexity", True)
        
        safe_min = find_nearest(all_strikes, min_k)
        safe_max = find_nearest(all_strikes, max_k)

        st.markdown("##### View Range")
        sel_range = st.slider("Global Range", float(safe_min), float(safe_max), (float(safe_min), float(safe_max)))
        
        if st.form_submit_button("Generate"):
            cfg = {
                'mode': mode, 'use_prior': use_prior, 
                'gap': gap, 'grid_mult': grid_mult, 'repair': repair, 
                'range_min': sel_range[0], 'range_max': sel_range[1]
            }
            gen = SurfaceGenerator(ticker, df)
            result = gen.generate(cfg, surf_name, exclude)
            if result:
                st.session_state.surfaces.append({
                    'name': surf_name, 'config': cfg, 
                    'data': result, 
                    'visible': True, 
                    'color': np.random.choice(['cyan', 'magenta', 'lime', 'orange'])
                })
                st.success("Added")

    st.markdown("---")
    st.header("3. View Settings")
    show_extrap = st.checkbox("Show Extrapolated Wings", True)
    show_mesh = st.checkbox("Interpolate Surface (Mesh)", False)
    
    st.subheader("Positioning Overlay")
    pos_metric = st.selectbox("Metric", ["None", "Volume", "Open Interest"])
    pos_scale = 1.0
    if pos_metric != "None": pos_scale = st.slider("Scale", 0.1, 5.0, 1.0, 0.1)

    st.subheader("Active Layers")
    to_rem = []
    
    # Auto-clean legacy data
    for i, s in enumerate(st.session_state.surfaces):
        if isinstance(s['data'], list): to_rem.append(i)
    if to_rem:
        for i in sorted(to_rem, reverse=True): del st.session_state.surfaces[i]
        st.rerun()
        
    for i, s in enumerate(st.session_state.surfaces):
        c1, c2, c3 = st.columns([0.1, 0.7, 0.2])
        s['visible'] = c1.checkbox("", s['visible'], key=f"v{i}")
        c2.markdown(f"<span style='color:{s['color']}'>█</span> {s['name']}", unsafe_allow_html=True)
        if c3.button("🗑️", key=f"d{i}"): to_rem.append(i)
    for i in sorted(to_rem, reverse=True): del st.session_state.surfaces[i]; st.rerun()

# --- PLOT ---
if not st.session_state.surfaces: st.stop()

fig = go.Figure()

max_density = 0; max_pos_val = 0
all_T_vals = set(); T_to_Date = {}

for surf in st.session_state.surfaces:
    if not surf['visible']: continue
    mesh_data = surf['data']['mesh']
    for t, date in zip(mesh_data['y'], mesh_data['dates']):
        all_T_vals.add(t); T_to_Date[t] = date
    for slice_ in surf['data']['slices']:
        y_key = 'y_global' if show_extrap else 'y_local'
        max_density = max(max_density, slice_[y_key].max())
        if pos_metric == "Volume": max_pos_val = max(max_pos_val, slice_['vol_pos'].max())
        elif pos_metric == "Open Interest": max_pos_val = max(max_pos_val, slice_['oi_pos'].max())

if max_density == 0: max_density = 0.01
if max_pos_val == 0: max_pos_val = 1

for surf in st.session_state.surfaces:
    if not surf['visible']: continue
    
    if show_mesh:
        mesh = surf['data']['mesh']
        fig.add_trace(go.Surface(
            x=mesh['x'], y=mesh['y'], z=mesh['z'],
            colorscale=[[0, surf['color']], [1, surf['color']]], 
            opacity=0.3, name=f"{surf['name']} Mesh", showscale=False, hoverinfo='skip'
        ))
    
    for slice_ in surf['data']['slices']:
        x_plot = slice_['x_global'] if show_extrap else slice_['x_local']
        y_plot = slice_['y_global'] if show_extrap else slice_['y_local']
        
        fig.add_trace(go.Scatter3d(
            x=x_plot, y=[slice_['T']]*len(x_plot), z=y_plot,
            mode='lines', line=dict(color=surf['color'], width=4),
            showlegend=False, hovertemplate=f"T: %{{y:.3f}}<br>K: %{{x:.1f}}<br>Pdf: %{{z:.4f}}<extra></extra>"
        ))
        
        if pos_metric != "None":
            raw_vals = slice_['vol_pos'] if pos_metric == "Volume" else slice_['oi_pos']
            scaled_vals = (raw_vals / max_pos_val) * max_density * pos_scale
            metric_color = 'chartreuse' if pos_metric == "Volume" else 'gold'
            
            fig.add_trace(go.Scatter3d(
                x=slice_['strikes_pos'], y=[slice_['T']]*len(slice_['strikes_pos']), z=scaled_vals,
                mode='markers+lines', marker=dict(size=3, color=metric_color),
                line=dict(color=metric_color, width=2, dash='dot'), showlegend=False,
                hovertemplate=f"<b>{pos_metric}</b><br>Val: %{{customdata:,.0f}}<extra></extra>",
                customdata=raw_vals
            ))

sorted_Ts = sorted(list(all_T_vals))
date_labels = [T_to_Date[t] for t in sorted_Ts]

fig.update_layout(
    title=f"Surface Lab ({ticker} @ {sel_snapshot})",
    scene=dict(
        xaxis_title='Strike',
        yaxis=dict(title='Expiration', tickmode='array', tickvals=sorted_Ts, ticktext=date_labels, gridcolor='gray'),
        zaxis_title='Density', xaxis=dict(gridcolor='gray'), zaxis=dict(gridcolor='gray'),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.6))
    ),
    height=800, margin=dict(t=40, b=0, l=0, r=0)
)
st.plotly_chart(fig, width='stretch')