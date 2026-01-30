import numpy as np
import pandas as pd

def smart_strike_selection(df, min_vol=0, top_n=50):
    """
    Selects strikes based on Liquidity (Vol/OI).
    Returns a sorted DataFrame of selected rows.
    """
    # Safety: Handle empty or zero data
    if df.empty: return df
    
    max_vol = df['volume'].max() if df['volume'].max() > 0 else 1
    max_oi = df['openInterest'].max() if df['openInterest'].max() > 0 else 1
    
    # Score: 70% Vol + 30% OI
    scores = 0.7 * (df['volume'] / max_vol) + 0.3 * (df['openInterest'] / max_oi)
    df = df.copy()
    df['liquidity_score'] = scores
    
    # Filter 1: Min Volume
    valid = df[df['volume'] >= min_vol]
    
    if len(valid) > top_n:
        top_picks = valid.nlargest(top_n, 'liquidity_score')
        # Always include Min/Max to preserve range
        min_k = valid.loc[valid['strike'].idxmin()]
        max_k = valid.loc[valid['strike'].idxmax()]
        
        combined = pd.concat([top_picks, min_k.to_frame().T, max_k.to_frame().T])
        return combined.drop_duplicates(subset=['strike']).sort_values('strike')
    else:
        return valid.sort_values('strike')

def get_lower_convex_hull(strikes, prices):
    """
    Monotone Chain Algorithm.
    Returns (strikes, prices) that form the Lower Convex Hull (Arbitrage Free).
    """
    # Strict Sort
    sort_idx = np.argsort(strikes)
    K = np.array(strikes)[sort_idx]
    P = np.array(prices)[sort_idx]
    
    points = list(zip(K, P))
    lower = []
    
    for p in points:
        while len(lower) >= 2:
            A = lower[-2]
            B = lower[-1]
            C = p
            # Cross Product: If > 0, it's a Concave turn (Right turn). Remove B.
            val = (B[1] - A[1]) * (C[0] - B[0]) - (C[1] - B[1]) * (B[0] - A[0])
            if val > 1e-10: 
                lower.pop()
            else:
                break
        lower.append(p)
    
    hull_k = np.array([p[0] for p in lower])
    hull_p = np.array([p[1] for p in lower])
    
    # Interpolate back to original grid to preserve input shape
    clean_prices = np.interp(K, hull_k, hull_p)
    
    return K, clean_prices