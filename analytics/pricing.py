import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Standard Normal CDF/PDF
N = norm.cdf
n = norm.pdf

def calculate_implied_forward(calls, puts, r, T):
    """
    Calculates Implied Forward (F) using Put-Call Parity on ATM strikes.
    F = K + e^(rT) * (C - P)
    
    Expects 'calls' and 'puts' to be dataframes/slices for a single expiry.
    """
    # Merge on Strike to find pairs
    merged = calls[['strike', 'mid']].merge(puts[['strike', 'mid']], on='strike', suffixes=('_c', '_p'))
    
    # Filter for liquid ATM options (Smallest spread, closest to spot?)
    # Simple proxy: Find strike where abs(C - P) is smallest (closest to ATM)
    merged['diff'] = abs(merged['mid_c'] - merged['mid_p'])
    atm_row = merged.loc[merged['diff'].idxmin()]
    
    K = atm_row['strike']
    C = atm_row['mid_c']
    P = atm_row['mid_p']
    
    F = K + np.exp(r * T) * (C - P)
    return F

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