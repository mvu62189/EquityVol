import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from analytics.svi import SVIModel
from tqdm import tqdm

class SurfaceCalibrator:
    def __init__(self):
        pass

    def calibrate_surface(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        """
        Fits SVI using Dynamic filtering (cutting wings for short T) and Spread-weighted optimization.
        """
        # Ensure expiry grouping key exists
        if 'expiry_str' not in df_processed.columns:
            df_processed['expiry_str'] = df_processed['expiry'].astype(str)
            
        expiries = sorted(df_processed['expiry_str'].unique())
        results = []

        print(f"Calibrating surface ({len(expiries)} expiries) with Dynamic Wing Cutoff...")

        for exp in tqdm(expiries):
            # 1. Base Filter (OTM & Valid)
            slice_full = df_processed[
                (df_processed['expiry_str'] == exp) & 
                (df_processed['vol_surface'].notna()) & 
                (df_processed['vol_surface'] > 0)
            ].copy()

            if len(slice_full) < 5: 
                continue
            
            T = slice_full['T'].iloc[0]
            if T < 0.002: # Skip < 0.5 days
                continue

            # --- DYNAMIC WING CUTOFF ---
            # Goal: For short expiries, ignore deep OTM quotes as they are noisy/illiquid.
            
            # A. Calculate approximate ATM Vol for this expiry
            atm_subset = slice_full[np.abs(slice_full['moneyness']) < 0.05]
            if not atm_subset.empty:
                atm_vol = atm_subset['vol_surface'].mean()
            else:
                atm_vol = slice_full['vol_surface'].median()
            
            # B. Define Cutoff Thresholds based on Time
            # Logic: We keep data within range +/- k_limit
            # Short dates need tight limits. Long dates can use full chain.
            
            if T < 0.005: # < 1 Day
                min_k = 0.01  # Always keep at least +/- 2%
                max_k = 0.04  # Never go beyond +/- 5% (Extremely Aggressive cut)
                n_std = 1.5   # Keep within 1.5 standard deviations
            elif T < 0.01: # < 2 Days
                min_k = 0.02  # Always keep at least +/- 3%
                max_k = 0.05  # Never go beyond +/- 7% (Very Aggressive cut)
                n_std = 1.5   # Keep within 2 standard deviations
            elif T < 0.015: # < 3 Days
                min_k = 0.04  # Always keep at least +/- 4%
                max_k = 0.08  # Never go beyond +/- 8% (Aggressive cut)
                n_std = 2   # Keep within 2.5 standard deviations
            elif  T < 0.02: # < 5 Days
                min_k = 0.05  # Always keep at least +/- 5%
                max_k = 0.10  # Never go beyond +/- 10% (Very Aggressive cut)
                n_std = 2   # Keep within 2.5 standard deviations
            elif T < 0.08: # < 1 Month
                min_k = 0.08  # Always keep at least +/- 8%
                max_k = 0.20  # Never go beyond +/- 20% (Aggressive cut)
                n_std = 3   # Keep within 3.5 standard deviations
            elif T < 0.25: # 1-3 Months
                min_k = 0.15
                max_k = 0.50
                n_std = 5.0
            else: # > 3 Months (LEAPS etc)
                min_k = 0.20
                max_k = 2.50 # Keep almost everything
                n_std = 6.0

            # Calculate limit based on volatility envelope
            vol_derived_limit = atm_vol * np.sqrt(T) * n_std
            k_limit = np.clip(vol_derived_limit, min_k, max_k)
            
            # Apply the cut
            subset = slice_full[np.abs(slice_full['moneyness']) <= k_limit].copy()
            
            # Safety: If cut removed too many points, fallback to slightly wider or full
            if len(subset) < 5:
                if len(slice_full) >= 5:
                    subset = slice_full # Fallback to everything
                else:
                    continue

            # 2. Prepare Variables
            k_obs = subset['moneyness'].values
            w_obs = (subset['vol_surface'].values ** 2) * T
            
            # --- ADAPTIVE WEIGHTING ---
            bid_iv = subset['bid_iv'].values
            ask_iv = subset['ask_iv'].values
            spreads = ask_iv - bid_iv
            
            # Clean spreads
            spreads = np.where((spreads > 0) & np.isfinite(spreads), spreads, 0.05)
            spreads = np.maximum(spreads, 0.0001) # Floor at 1bp
            
            # Base weight: Inverse Spread
            liquidity_weights = 1.0 / spreads

            # ATM Boost: Increases based on how short the expiry is
            if T < 0.1:
                # Super aggressive pinning for short term (Gamma is high here)
                gaussian_width = 0.04
                magnifier = 200.0 
            else:
                gaussian_width = 0.10
                magnifier = 50.0

            moneyness_weights = np.exp(-0.5 * (k_obs / gaussian_width)**2)
            
            final_weights = liquidity_weights * (1.0 + magnifier * moneyness_weights)
            
            # Normalize to avoid numerical issues
            final_weights = final_weights / np.mean(final_weights)

            # 3. Optimization
            def residuals(params):
                model = SVIModel(params)
                w_model = np.array([model.get_variance(k) for k in k_obs])
                return (w_model - w_obs) * final_weights
            
            # Initial Guess [a, b, rho, m, sigma]
            min_w = np.min(w_obs)
            x0 = [min_w, 0.1, -0.6, 0.0, 0.1]
            
            # Constraints
            # Rho (skew) capped at 0.0 to prevent "Call Wing Lift" artifacts on equities
            bounds = (
                [0.0, 0.0, -0.999, -1.0, 0.001], # Lower
                [2.0, 5.0,  0.0,  1.0, 2.0]      # Upper
            )

            try:
                res = least_squares(residuals, x0, bounds=bounds, loss='soft_l1', f_scale=0.1)
                
                r_dict = {
                    'expiry': exp,
                    'T': T,
                    'F': subset['F'].iloc[0],
                    'a': res.x[0],
                    'b': res.x[1],
                    'rho': res.x[2],
                    'm': res.x[3],
                    'sigma': res.x[4],
                    'rmse': np.sqrt(np.mean(res.fun**2))
                }
                results.append(r_dict)
            except Exception as e:
                # print(f"Fit failed for {exp}: {e}")
                pass

        return pd.DataFrame(results)

    def augment_with_surface(self, df_processed, df_params):
        param_dict = df_params.set_index('expiry').to_dict('index')
        
        def get_model_vol(row):
            exp = str(row['expiry'])
            if exp not in param_dict: return np.nan
            p = param_dict[exp]
            svi = SVIModel([p['a'], p['b'], p['rho'], p['m'], p['sigma']])
            return svi.get_vol(row['moneyness'], row['T'])

        df_processed['iv_model'] = df_processed.apply(get_model_vol, axis=1)
        return df_processed