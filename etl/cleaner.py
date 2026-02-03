import pandas as pd
import numpy as np
from etl.rates_loader import RatesLoader
from utils.time_utils import calculate_time_to_expiry
from analytics.pricing import calculate_implied_forward, implied_vol_solver, calculate_greeks_bs
from tqdm import tqdm

class ChainProcessor:
    def __init__(self):
        self.rates_loader = RatesLoader()

    def process_chain(self, df_raw: pd.DataFrame):
        """
        Takes raw dataframe, adds: T, r, Forward, Bid/Ask/Mid IVs, Greeks, and Surface Vol
        """
        if df_raw.empty: return df_raw
        
        print("Processing raw chain...")
        df = df_raw.copy()
        
        # 1. Calc Time to Expiry
        snapshot_time = df['snapshot_time'].iloc[0]
        if isinstance(snapshot_time, str):
            snapshot_time = pd.to_datetime(snapshot_time)
            
        df['T'] = calculate_time_to_expiry(df['expiry'], snapshot_time)
        
        # 2. Get Rates
        unique_Ts = df['T'].unique()
        r_map = {t: self.rates_loader.get_rate(t) for t in unique_Ts}
        df['r'] = df['T'].map(r_map)
        
        # 3. Calculate Forwards (Per Expiry)
        expiries = df['expiry'].unique()
        fwd_map = {}
        
        print("Calibrating Forwards...")
        for exp in tqdm(expiries, desc="Forwards"):
            slice_ = df[df['expiry'] == exp]
            calls = slice_[slice_['type'] == 'C'].copy()
            puts = slice_[slice_['type'] == 'P'].copy()
            
            if calls.empty or puts.empty:
                continue
            
            # Calculate mid for parity check
            calls['mid'] = (calls['bid'] + calls['ask']) / 2
            puts['mid'] = (puts['bid'] + puts['ask']) / 2
            
            T_val = slice_['T'].iloc[0]
            r_val = slice_['r'].iloc[0]
            S_current = slice_['underlying_price'].iloc[0]

            try:
                F = calculate_implied_forward(calls, puts, r_val, T_val, S_ref=S_current)
                
            except Exception as e:
                # Fallback to spot if parity fails
                F = S_current * np.exp(r_val * T_val)
                
            fwd_map[exp] = F
            
        df['F'] = df['expiry'].map(fwd_map)
        
        # 4. Solve Implied Volatilities (Bid, Ask, Mid)
        # 4. Solve Implied Volatilities (Optimized Loop)
        # We only solve if:
        #   a. Option is OTM (Call > F, Put < F)
        #   b. Option has a valid quote (Bid > 0, Ask > 0)
        df['mid'] = (df['bid'] + df['ask']) / 2
        
        bid_ivs = []
        ask_ivs = []
        mid_ivs = []
        surface_vols = []

        print("Solving Implied Vols (Bid/Ask/Mid)...")

        # Pre-calc columns for speed
        itr = zip(df['bid'], df['ask'], df['mid'], df['F'], df['strike'], df['T'], df['r'], df['type'])

        # Using iterrows for specific row processing
        for bid, ask, mid, F, K, T, r, type_ in tqdm(itr, total=df.shape[0], desc="IV Solver"):
            
            # --- FILTER LOGIC ---
            is_otm = (type_ == 'C' and K >= F) or (type_ == 'P' and K < F)
            has_quote = (bid > 0) and (ask > 0)
            
            if is_otm and has_quote:
                # Solve
                b_iv = implied_vol_solver(bid, F, K, T, r, type_)
                a_iv = implied_vol_solver(ask, F, K, T, r, type_)
                m_iv = implied_vol_solver(mid, F, K, T, r, type_)
                
                bid_ivs.append(b_iv)
                ask_ivs.append(a_iv)
                mid_ivs.append(m_iv)
                surface_vols.append(m_iv) # It's OTM, so it belongs on surface
            else:
                # Skip
                bid_ivs.append(np.nan)
                ask_ivs.append(np.nan)
                mid_ivs.append(np.nan)
                surface_vols.append(np.nan) # Filtered out
            
        df['bid_iv'] = bid_ivs
        df['ask_iv'] = ask_ivs
        df['mid_iv'] = mid_ivs
        df['vol_surface'] = surface_vols # Now implicitly filters ITM
        
        # 5. Greeks (Vectorized where possible, or skip NaN vols)
        # Only calc greeks if we have a vol
        greeks_list = []
        for idx, row in df.iterrows():
            vol = row['mid_iv']
            if pd.isna(vol) or vol <= 0:
                 g = {'Delta': 0.0, 'Gamma': 0.0, 'Vega': 0.0}
            else:
                 g = calculate_greeks_bs(row['F'], row['strike'], row['T'], row['r'], vol, row['type'])
            greeks_list.append(g)
            
        greeks_df = pd.DataFrame(greeks_list).astype(float)
        df = pd.concat([df, greeks_df], axis=1)

        # 6. Moneyness
        df['moneyness'] = np.log(df['strike'] / df['F'])

        return df

    def get_clean_surface(self, df: pd.DataFrame):
        """
        Returns TWO dataframes:
        1. df_clean: Valid OTM data with quotes (Includes 0 volume tails).
        2. df_dropped: Rejected data (ITM, Dead Quotes).
        """
        df = df.copy()
        df['drop_reason'] = None 

        # 1. Check Validity
        # Since process_chain puts NaN in 'vol_surface' for ITM/Dead quotes,
        # we can simply check that column.
        mask_invalid = (df['drop_reason'].isna()) & (df['vol_surface'].isna())
        df.loc[mask_invalid, 'drop_reason'] = 'ITM / Dead Quote / Solver Fail'

        # 3. Split
        df_dropped = df[df['drop_reason'].notna()].copy()
        df_clean = df[df['drop_reason'].isna()].drop(columns=['drop_reason'])

        return df_clean, df_dropped