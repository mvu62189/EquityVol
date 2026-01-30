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
        Fits SVI using Gaussian-weighted Moneyness to pin ATM volatility.
        """
        # Ensure we have string expiries for grouping
        if 'expiry_str' not in df_processed.columns:
            df_processed['expiry_str'] = df_processed['expiry'].astype(str)
            
        expiries = df_processed['expiry_str'].unique()
        results = []

        print(f"Calibrating surface (Gaussian ATM Focus)...")

        for exp in tqdm(expiries):
            # 1. Filter for valid calibration data (OTM only)
            subset = df_processed[
                (df_processed['expiry_str'] == exp) & 
                (df_processed['vol_surface'].notna()) & 
                (df_processed['vol_surface'] > 0)
            ].copy()

            if len(subset) < 5: 
                continue
            
            T = subset['T'].iloc[0]
            if T < 0.00002: # Skip < 0.005 days ~ 7 minutes
                continue

            # 2. Prepare Variables
            k_obs = subset['moneyness'].values
            w_obs = (subset['vol_surface'].values ** 2) * T
            
            # --- ENHANCED WEIGHTING LOGIC ---
            
            # A. Liquidity Weighting (1 / Spread)
            bid_iv = subset['bid_iv'].values
            ask_iv = subset['ask_iv'].values
            spreads = ask_iv - bid_iv
            
            # Handle bad data: if spread <= 0 or NaN, assume a default loose spread (e.g. 5% vol)
            spreads = np.where((spreads > 0) & np.isfinite(spreads), spreads, 0.05)
            # Floor spread to prevent infinite weights on accidentally zero-spread rows
            spreads = np.maximum(spreads, 0.0005) 
            
            liquidity_weights = 1.0 / spreads

            # B. Moneyness Weighting (Gaussian Kernel)
            # This creates a "Bell Curve" of importance centered at k=0 (ATM)
            # sigma=0.1 means the weight decays significantly by +/- 10% OTM/ITM.
            # We want strict adherence to the center.
            gaussian_width = 0.10 
            moneyness_weights = np.exp(-0.5 * (k_obs / gaussian_width)**2)
            
            # C. Composite Weight
            # We take the liquidity weight and BOOST it by factor of 100 at the ATM money.
            # This forces the curve to pass through the ATM region, fixing the "rounded bottom" issue.
            atm_magnifier = 100.0
            
            final_weights = liquidity_weights * (1.0 + atm_magnifier * moneyness_weights)
            
            # Normalize weights for optimizer stability
            final_weights = final_weights / np.mean(final_weights)

            # 3. Objective Function
            def residuals(params):
                model = SVIModel(params)
                w_model = np.array([model.get_variance(k) for k in k_obs])
                return (w_model - w_obs) * final_weights
            
            # 4. Initial Guess & Bounds
            # To fix "falling toward call wing", we constrain rho (skew) slightly less aggressive initially
            min_w = np.min(w_obs)
            
            # [a, b, rho, m, sigma]
            x0 = [min_w, 0.1, -0.5, 0.0, 0.1]
            
            bounds = (
                [0.0, 0.0, -0.99, -1.0, 0.001], # Lower
                [max(min_w*2, 2.0), 5.0, 0.0, 1.0, 2.0]     # Upper (Restrict Rho <= 0 for equities generally, but 0.99 ok)
            )
            # Note: I set upper rho to 0.0 or 0.99? 
            # Equities usually have negative skew. Allowing positive rho might cause the "U" to flip.
            # Let's keep rho < 0.99 to be safe, but typically it converges to -0.6.

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
                print(f"Fit failed for {exp}: {e}")

        return pd.DataFrame(results)

    def augment_with_surface(self, df_processed, df_params):
        # (Same as before)
        param_dict = df_params.set_index('expiry').to_dict('index')
        
        def get_model_vol(row):
            exp = str(row['expiry'])
            if exp not in param_dict: return np.nan
            p = param_dict[exp]
            svi = SVIModel([p['a'], p['b'], p['rho'], p['m'], p['sigma']])
            return svi.get_vol(row['moneyness'], row['T'])

        df_processed['iv_model'] = df_processed.apply(get_model_vol, axis=1)
        return df_processed