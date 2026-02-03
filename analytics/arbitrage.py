import numpy as np

def repair_convexity(strikes, prices, strict_convexity=True, epsilon=1e-5):
    """
    Greatest Convex Minorant (GCM) Algorithm.
    Iteratively adjusts prices downwards until they form a perfectly convex curve.
    - Preserves stride (uses all points).
    - Removes arbitrage (negative density) by flattening bumps.
    """
    K = np.array(strikes)
    P = np.array(prices)
    
    max_passes = 200
    tol = 1e-5
    sag = abs(epsilon) if strict_convexity else 0.0

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
            
            if p_curr > p_max_valid:
                P[i] = p_max_valid - sag
                changes += 1
                
        if changes == 0:
            break
            
    return K, P