import argparse
import pandas as pd
import os
from etl.cleaner import ChainProcessor # Ensure previous cleanup works
from analytics.calibration import SurfaceCalibrator

def run_calibration(ticker):
    # 1. Load Processed Data
    data_dir = os.path.join("data", "clean", ticker.upper())
    # Find latest file
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if not files:
        print("No processed data found. Run main_etl.py first.")
        return
        
    latest_file = sorted(files)[-1]
    path = os.path.join(data_dir, latest_file)
    print(f"Loading: {path}")
    
    df = pd.read_parquet(path)

    # 2. Run Calibration
    calibrator = SurfaceCalibrator()
    df_params = calibrator.calibrate_surface(df)
    
    if df_params.empty:
        print("Calibration failed to produce any valid slices.")
        return

    # 3. Save Model Parameters
    model_dir = os.path.join("data", "models", ticker.upper())
    os.makedirs(model_dir, exist_ok=True)
    
    # Save as CSV for human readability, or Parquet/JSON for speed
    param_path = os.path.join(model_dir, latest_file.replace("chain", "svi_params").replace(".parquet", ".csv"))
    df_params.to_csv(param_path, index=False)
    print(f"Parameters saved: {param_path}")
    print(df_params.head())
    
    # 4. (Optional) Check fit quality
    avg_rmse = df_params['rmse'].mean()
    print(f"Average Surface RMSE: {avg_rmse:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="SPY")
    args = parser.parse_args()
    run_calibration(args.ticker)