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
            
            try:
                F = calculate_implied_forward(calls, puts, r_val, T_val)
            except:
                # Fallback to spot if parity fails
                F = slice_['underlying_price'].iloc[0] * np.exp(r_val * T_val)
                
            fwd_map[exp] = F
            
        df['F'] = df['expiry'].map(fwd_map)
        
        # 4. Solve Implied Volatilities (Bid, Ask, Mid)
        df['mid'] = (df['bid'] + df['ask']) / 2
        
        bid_ivs = []
        ask_ivs = []
        mid_ivs = []
        
        print("Solving Implied Vols (Bid/Ask/Mid)...")
        # Using iterrows for specific row processing
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="IV Solver"):
            F, K, T, r, type_ = row['F'], row['strike'], row['T'], row['r'], row['type']
            
            # Solve for all three
            bid_iv = implied_vol_solver(row['bid'], F, K, T, r, type_)
            ask_iv = implied_vol_solver(row['ask'], F, K, T, r, type_)
            mid_iv = implied_vol_solver(row['mid'], F, K, T, r, type_)
            
            bid_ivs.append(bid_iv)
            ask_ivs.append(ask_iv)
            mid_ivs.append(mid_iv)
            
        df['bid_iv'] = bid_ivs
        df['ask_iv'] = ask_ivs
        df['mid_iv'] = mid_ivs
        
        # 5. Calculate Greeks (Using Mid IV)
        greeks_list = []
        for idx, row in df.iterrows():
            vol = row['mid_iv']
            if pd.isna(vol) or vol == 0:
                 g = {'Delta': 0.0, 'Gamma': 0.0, 'Vega': 0.0}
            else:
                 g = calculate_greeks_bs(row['F'], row['strike'], row['T'], row['r'], vol, row['type'])
            greeks_list.append(g)
            
        greeks_df = pd.DataFrame(greeks_list)
        # Ensure float types
        greeks_df = greeks_df.astype(float)
        
        df = pd.concat([df, greeks_df], axis=1)

        # 6. Create 'vol_surface' Column (Select OTM options)
        # Call OTM if K > F
        # Put OTM if K < F
        surface_vols = []
        
        for idx, row in df.iterrows():
            # If F is NaN (failed parity), skip
            if pd.isna(row['F']):
                surface_vols.append(np.nan)
                continue

            is_call = (row['type'] == 'C')
            K = row['strike']
            F = row['F']
            
            is_otm = (is_call and K > F) or (not is_call and K < F)
            
            if is_otm:
                surface_vols.append(row['mid_iv'])
            else:
                surface_vols.append(np.nan) # ITM options not used for surface fitting
        
        df['vol_surface'] = surface_vols

        # 7. Moneyness (Log Strike / Forward)
        df['moneyness'] = np.log(df['strike'] / df['F'])

        return df