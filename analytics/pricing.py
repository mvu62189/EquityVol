import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Standard Normal CDF/PDF
N = norm.cdf
n = norm.pdf

def calculate_implied_forward(calls, puts, r, T, S_ref=None):
    """
    Calculates Implied Forward (F) using Put-Call Parity on ATM strikes.
    F = K + e^(rT) * (C - P)
    
    Expects 'calls' and 'puts' to be dataframes/slices for a single expiry.
    """
    # 0. range_filter check
    if S_ref is not None:
        # Define valid strike window (e.g., 10% around spot)
        # Deep ITM/OTM options introduce massive spread error into F calc
        min_k = S_ref * 0.90
        max_k = S_ref * 1.10
        
        calls = calls[(calls['strike'] >= min_k) & (calls['strike'] <= max_k)]
        puts = puts[(puts['strike'] >= min_k) & (puts['strike'] <= max_k)]
    
    if calls.empty or puts.empty:
        return np.nan

    # 1. Merge on Strike
    cols_to_keep = ['strike', 'mid']
    if 'bid' in calls.columns: cols_to_keep.append('bid')
    
    # Merge on Strike to find pairs 
    merged = calls[cols_to_keep].merge(puts[cols_to_keep], on='strike', suffixes=('_c', '_p'))
    
    # 2. Filter for active quotes (if bid info available)
    if 'bid_c' in merged.columns and 'bid_p' in merged.columns:
        merged = merged[
            (merged['bid_c'] > 0.05) & 
            (merged['bid_p'] > 0.05)
        ]
    
    if merged.empty:
        return np.nan

    # 3. Calculate F candidates
    # Parity: F = K + exp(rT) * (C - P)
    merged['F_implied'] = merged['strike'] + np.exp(r * T) * (merged['mid_c'] - merged['mid_p'])
    
    # 4. Robust Aggregation (Median of valid pairs)
    # We trust the median more than the single "best" fit which might be a lucky wide spread
    return merged['F_implied'].median()

def black_76_price(F, K, T, r, sigma, option_type='C'):
    """
    Black-76 Price (using Forward). 
    Used for Equities by treating Dividends implicitly via F.
    """
    if T <= 0:
        return max(0, F - K) if option_type == 'C' else max(0, K - F)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    disc = np.exp(-r * T)
    
    if option_type == 'C':
        return disc * (F * N(d1) - K * N(d2))
    else:
        return disc * (K * N(-d2) - F * N(-d1))

def implied_vol_solver(price, F, K, T, r, option_type='C'):
    """
    Backs out IV from price using Newton-Brent method.
    """
    # 1. Intrinsic Value Check (Deep ITM options often trade at intrinsic)
    disc = np.exp(-r * T)
    if option_type == 'C':
        intrinsic = max(0, (F - K) * disc)
    else:
        intrinsic = max(0, (K - F) * disc)
        
    if price <= intrinsic + 0.001:
        return 0.0 # Return 0 vol if price is basically intrinsic

    # 2. Solver Objective
    def obj(sigma):
        return black_76_price(F, K, T, r, sigma, option_type) - price
    
    try:
        # Search between 1% and 500% vol
        return brentq(obj, 0.001, 5.0, xtol=1e-4)
    except Exception:
        return np.nan # Failed to converge (likely arb data)

def calculate_greeks_bs(F, K, T, r, sigma, option_type='C'):
    """
    Analytic Greeks for Black-76
    """
    if T <= 1e-5 or sigma <= 0:
        return {'Delta': 0, 'Gamma': 0, 'Vega': 0}

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    
    if option_type == 'C':
        delta = disc * N(d1)
    else:
        delta = -disc * N(-d1)
        
    gamma = (disc * n(d1)) / (F * sigma * np.sqrt(T))
    vega = F * disc * n(d1) * np.sqrt(T) / 100.0 # Scaled for 1% change
    
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega}