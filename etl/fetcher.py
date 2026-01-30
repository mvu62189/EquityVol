import yfinance as yf
import pandas as pd
import datetime as dt
import time
from tqdm import tqdm  # For progress bar

class EquityFetcher:
    def __init__(self, ticker_symbol: str):
        """
        Initializes the fetcher for a specific ticker.
        """
        self.symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(self.symbol)
        
    def get_spot_price(self) -> float:
        """
        Fetches the latest available price for the underlying.
        Uses 'fast_info' if available, falls back to 1-day history.
        """
        try:
            # fast_info is often faster/more real-time
            price = self.ticker.fast_info.last_price
            if price is None:
                raise ValueError("fast_info returned None")
        except Exception:
            # Fallback to history
            hist = self.ticker.history(period="1d", interval="1m")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
            else:
                # Fallback for illiquid/closed market
                price = self.ticker.info.get('previousClose', 0.0)
        
        return float(price)

    def fetch_option_chain(self) -> pd.DataFrame:
        """
        Iterates over all available expiration dates, pulls calls/puts,
        and returns a single unified DataFrame with snapshot timestamps.
        """
        expirations = self.ticker.options
        if not expirations:
            print(f"No options data found for {self.symbol}")
            return pd.DataFrame()

        print(f"Fetching {len(expirations)} expiries for {self.symbol}...")
        
        # 1. Get Spot Price for this Snapshot
        snapshot_spot = self.get_spot_price()
        snapshot_time = dt.datetime.now()
        
        all_options = []

        # 2. Loop through every expiration date
        for expiry_str in tqdm(expirations, desc="Pulling Chains"):
            try:
                # yfinance returns a named tuple: (calls, puts)
                chain = self.ticker.option_chain(expiry_str)
                
                # Process Calls
                calls = chain.calls.copy()
                calls['type'] = 'C'
                
                # Process Puts
                puts = chain.puts.copy()
                puts['type'] = 'P'
                
                # Merge current expiry slice
                df_slice = pd.concat([calls, puts], ignore_index=True)
                
                # Inject Metadata
                df_slice['expiry'] = pd.to_datetime(expiry_str).date()
                df_slice['snapshot_time'] = snapshot_time
                df_slice['underlying_symbol'] = self.symbol
                df_slice['underlying_price'] = snapshot_spot
                
                all_options.append(df_slice)
                
                # Polite delay to avoid rate limiting
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"Failed to fetch expiry {expiry_str}: {e}")

        if not all_options:
            return pd.DataFrame()

        # 3. Concatenate and Clean
        final_df = pd.concat(all_options, ignore_index=True)
        
        # Standardize Columns
        # Rename yfinance columns to our spec if needed (yfinance uses camelCase usually)
        # Expected from yf: [contractSymbol, lastTradeDate, strike, lastPrice, bid, ask, change, volume, openInterest, impliedVolatility]
        
        # Ensure numeric types for critical columns
        cols_to_numeric = ['strike', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        for col in cols_to_numeric:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)

        return final_df

if __name__ == "__main__":
    # Quick Test
    fetcher = EquityFetcher("SPY")
    df = fetcher.fetch_option_chain()
    print(f"\nSuccess! Fetched {len(df)} contracts.")
    print(f"Spot Price Used: {df['underlying_price'].iloc[0]}")
    print(df.head())