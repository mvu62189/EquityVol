import argparse
import os
from datetime import datetime
from etl.fetcher import EquityFetcher
from etl.cleaner import ChainProcessor

def run_etl(ticker_symbol: str):
    # 1. Init
    print(f"--- Starting ETL for {ticker_symbol} ---")
    fetcher = EquityFetcher(ticker_symbol)
    
    # 2. Fetch
    df = fetcher.fetch_option_chain()
    
    if df.empty:
        print("ETL Aborted: No data returned.")
        return

    # 3. Save Strategy
    # Structure: data/raw/{TICKER}/{YYYY-MM-DD_HHmm}.parquet
    now_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    save_dir = os.path.join("data", "raw", ticker_symbol.upper())
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"chain_{now_str}.parquet"
    full_path = os.path.join(save_dir, filename)
    
    # Save using PyArrow engine (fast and type-safe)
    df.to_parquet(full_path, index=False)
    
    processor = ChainProcessor()
    df_processed = processor.process_chain(df)
    
    processed_path = full_path.replace("raw", "processed")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_parquet(processed_path, index=False)
    
    df_clean, df_dropped = processor.get_clean_surface(df_processed)
    
    clean_path = full_path.replace("raw", "clean") # data/clean/{TICKER}/...
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df_clean.to_parquet(clean_path, index=False)
    
    # Save Dropped Data (for Visualization/QC)
    dropped_path = full_path.replace("raw", "dropped") # data/dropped/SPY/...
    os.makedirs(os.path.dirname(dropped_path), exist_ok=True)
    df_dropped.to_parquet(dropped_path, index=False)

    print(f"--- ETL Complete ---")
    print(f"Saved: {full_path}")
    print(f"Total Rows: {len(df)}")

    print(f"Full Processed: {len(df_processed)} rows")
    print(f"Processed saved: {processed_path}")

    print(f"Clean (Vol>0): {len(df_clean)} rows -> {clean_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Equity Option Chain ETL")
    parser.add_argument("--ticker", type=str, default="SPY", help="Underlying Ticker (e.g. SPY, NVDA)")
    
    args = parser.parse_args()
    run_etl(args.ticker)