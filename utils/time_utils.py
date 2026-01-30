import pandas as pd
import datetime as dt
import numpy as np

def calculate_time_to_expiry(expiry_date_series, snapshot_time):
    """
    Calculates T (Time in Years) with 16:00 ET expiration precision.
    """
    # Ensure expiry is datetime
    if not isinstance(expiry_date_series, pd.Series):
        expiry_date_series = pd.Series(expiry_date_series)
        
    # Standard equity options expire at 16:00 ET
    # We construct the exact expiry timestamp
    expiry_timestamps = pd.to_datetime(expiry_date_series) + pd.Timedelta(hours=16)
    
    # Calculate difference in total seconds
    # Force snapshot_time to be naive or tz-aware matching expiry if needed
    # Usually easier to treat everything as 'Market Time' (ET) naive
    diff_seconds = (expiry_timestamps - snapshot_time).dt.total_seconds()
    
    # Floor at 0 to avoid negative time
    diff_seconds = diff_seconds.clip(lower=0)
    
    # Annualize (Seconds in a year = 365 * 24 * 60 * 60 = 31536000)
    T = diff_seconds / 31536000.0
    
    # Handle 0DTE specific logic (prevent div by zero in pricing)
    # Set a minimum T of 1 minute (1.9e-6 years)
    T = T.clip(lower=1.9e-6)
    
    return T