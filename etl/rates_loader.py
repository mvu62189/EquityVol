import yfinance as yf
from scipy.interpolate import CubicSpline
import numpy as np

class RatesLoader:
    def __init__(self):
        # ^IRX: 13 Week, ^FVX: 5 Year, ^TNX: 10 Year, ^TYX: 30 Year
        self.tickers = {
            '^IRX': 0.25,
            '^FVX': 5.0,
            '^TNX': 10.0,
            '^TYX': 30.0
        }
        self.model = None

    def fetch_yield_curve(self):
        print("Fetching Treasury Yields...")
        durations = []
        rates = []
        
        # Anchor short term rate
        durations.append(0.0)
        
        for ticker, dur in self.tickers.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="1d")
                if not hist.empty:
                    # yfinance yields are index values (4.25 = 4.25%)
                    r = hist['Close'].iloc[-1] / 100.0
                    rates.append(r)
                    durations.append(dur)
            except Exception as e:
                print(f"Warning: Could not fetch rate {ticker}: {e}")

        if len(rates) > 0:
            rates.insert(0, rates[0]) # Anchor t=0 to the shortest rate found
        else:
            print("CRITICAL: Rates fetch failed. Using default 4.5%.")
            self.model = lambda x: 0.045
            return self.model

        self.model = CubicSpline(durations, rates, bc_type='natural')
        return self.model

    def get_rate(self, t):
        """ Returns risk-free rate for time t (years) """
        if self.model is None:
            self.fetch_yield_curve()
        
        # FIX: CubicSpline returns a numpy array. We must cast to float.
        rate_val = self.model(t)
        return float(rate_val)