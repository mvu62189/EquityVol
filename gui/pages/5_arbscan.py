import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

st.set_page_config(layout="wide", page_title="Arbitrage Scanner")
DATA_DIR = "data/processed"

# --- Loaders ---
def load_data(ticker):
    p_dir = os.path.join(DATA_DIR, ticker)
    if not os.path.exists(p_dir): return None
    files = sorted([f for f in os.listdir(p_dir) if f.endswith('.parquet')])
    if not files: return None
    return pd.read_parquet(os.path.join(p_dir, files[-1]))

# --- Core Logic: The Scanners ---

def scan_vertical_arb(chain, opt_type='C'):
    """
    Scans for Monotonicity Violations.
    Call Arb: Buy Low Strike (Ask) < Sell High Strike (Bid).
    Put Arb: Buy High Strike (Ask) < Sell Low Strike (Bid).
    """
    subset = chain[chain['type'] == opt_type].sort_values('strike').reset_index(drop=True)
    opportunities = []

    # Iterate through chain
    for i in range(len(subset) - 1):
        leg1 = subset.iloc[i]   # Lower Strike
        leg2 = subset.iloc[i+1] # Higher Strike
        
        # Filter: Skip if spreads are zero or empty
        if leg1['ask'] == 0 or leg2['bid'] == 0: continue

        if opt_type == 'C':
            # BULL SPREAD ARB: Buy Low, Sell High
            # Cost = Ask(Low) - Bid(High)
            # If Cost < 0, it's an arb (Credit to enter Bull Spread)
            cost = leg1['ask'] - leg2['bid']
            
            if cost < 0:
                opportunities.append({
                    'Type': 'Call Vertical (Bull)',
                    'Buy_Strike': leg1['strike'],
                    'Sell_Strike': leg2['strike'],
                    'Buy_Ask': leg1['ask'],
                    'Sell_Bid': leg2['bid'],
                    'Net_Credit': -cost, # Positive profit
                    'Max_Profit': (leg2['strike'] - leg1['strike']) - cost,
                    'RoI': 'Infinite'
                })
                
        elif opt_type == 'P':
            # BEAR SPREAD ARB: Buy High, Sell Low
            # Cost = Ask(High) - Bid(Low)
            # If Cost < 0, it's an arb
            cost = leg2['ask'] - leg1['bid']
            
            if cost < 0:
                opportunities.append({
                    'Type': 'Put Vertical (Bear)',
                    'Buy_Strike': leg2['strike'],
                    'Sell_Strike': leg1['strike'],
                    'Buy_Ask': leg2['ask'],
                    'Sell_Bid': leg1['bid'],
                    'Net_Credit': -cost,
                    'Max_Profit': (leg2['strike'] - leg1['strike']) - cost,
                    'RoI': 'Infinite'
                })
                
    return pd.DataFrame(opportunities)

def scan_butterfly_arb(chain, opt_type='C'):
    """
    Scans for Convexity Violations.
    Strategy: Long Butterfly (Buy Wing, Sell 2x Body, Buy Wing)
    Cost = Ask(Low) - 2*Bid(Mid) + Ask(High)
    If Cost < 0, you are paid to own a structure that cannot lose money.
    """
    subset = chain[chain['type'] == opt_type].sort_values('strike').reset_index(drop=True)
    opportunities = []

    # Need 3 consecutive strikes
    for i in range(len(subset) - 2):
        leg1 = subset.iloc[i]   # Low
        leg2 = subset.iloc[i+1] # Mid
        leg3 = subset.iloc[i+2] # High
        
        # 1. Check Equidistance (approx)
        dist1 = leg2['strike'] - leg1['strike']
        dist2 = leg3['strike'] - leg2['strike']
        
        if abs(dist1 - dist2) > 0.05: continue # Skip irregular strikes
        
        # 2. Check Data Quality
        if leg1['ask']<=0 or leg2['bid']<=0 or leg3['ask']<=0: continue
        
        # 3. Calculate Trade Cost (Worst Case: Buy Ask, Sell Bid)
        # Long Fly = +1 Low(Ask) -2 Mid(Bid) +1 High(Ask)
        entry_cost = leg1['ask'] - (2 * leg2['bid']) + leg3['ask']
        
        if entry_cost < 0:
            opportunities.append({
                'Type': f'{opt_type} Butterfly',
                'Strikes': f"{leg1['strike']} / {leg2['strike']} / {leg3['strike']}",
                'Ask_Low': leg1['ask'],
                'Bid_Mid': leg2['bid'],
                'Ask_High': leg3['ask'],
                'Entry_Credit': -entry_cost,
                'Max_Profit': dist1 - entry_cost # Strike Width + Credit
            })
            
    return pd.DataFrame(opportunities)

# --- UI Interface ---
st.title("⚡ Arbitrage Scanner")
st.markdown("Scans for **Hard Arbitrage** (Negative Prices) using strict Bid/Ask logic.")

col1, col2 = st.columns([1, 3])

with col1:
    tickers = [d for d in os.listdir(DATA_DIR)] if os.path.exists(DATA_DIR) else []
    ticker = st.selectbox("Select Ticker", tickers)
    
    if ticker:
        df = load_data(ticker)
        df['expiry'] = df['expiry'].astype(str)
        expiry = st.selectbox("Select Expiry", sorted(df['expiry'].unique()))
        
        # Filters
        min_vol = st.number_input("Min Volume", value=0, help="Filter out dead quotes")
        st.info("Tip: Real arb is rare. If you see many results, check if quotes are stale (timestamp).")

with col2:
    if ticker and expiry:
        # Prep Data
        slice_ = df[df['expiry'] == expiry].copy()
        
        # Filter by Volume (optional, prevents ghost quotes)
        if min_vol > 0:
            slice_ = slice_[slice_['volume'] >= min_vol]
        
        st.subheader(f"Results for {ticker} @ {expiry}")
        
        # 1. Run Vertical Scan
        v_arbs_c = scan_vertical_arb(slice_, 'C')
        v_arbs_p = scan_vertical_arb(slice_, 'P')
        v_arbs = pd.concat([v_arbs_c, v_arbs_p], ignore_index=True)
        
        if not v_arbs.empty:
            st.error(f"FOUND {len(v_arbs)} VERTICAL ARBITRAGE OPPORTUNITIES")
            st.dataframe(v_arbs.style.format({'Net_Credit': '{:.2f}', 'Buy_Ask': '{:.2f}', 'Sell_Bid': '{:.2f}'}), use_container_width=True)
        else:
            st.success("No Vertical Arbitrage found (Market is Monotonic).")
            
        st.markdown("---")
        
        # 2. Run Butterfly Scan
        f_arbs_c = scan_butterfly_arb(slice_, 'C')
        f_arbs_p = scan_butterfly_arb(slice_, 'P')
        f_arbs = pd.concat([f_arbs_c, f_arbs_p], ignore_index=True)
        
        if not f_arbs.empty:
            st.error(f"FOUND {len(f_arbs)} BUTTERFLY ARBITRAGE OPPORTUNITIES")
            st.dataframe(f_arbs.style.format({'Entry_Credit': '{:.2f}', 'Ask_Low': '{:.2f}', 'Bid_Mid': '{:.2f}'}), use_container_width=True)
        else:
            st.success("No Butterfly Arbitrage found (Market is Convex).")