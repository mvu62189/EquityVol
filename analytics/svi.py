import numpy as np

class SVIModel:
    """
    Raw SVI parameterization: 
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    where:
    k = log(Strike / Forward)
    w = Total Variance (Vol^2 * Time)
    """
    def __init__(self, params):
        # params: [a, b, rho, m, sigma]
        self.a = float(params[0])
        self.b = float(params[1])
        self.rho = float(params[2])
        self.m = float(params[3])
        self.sigma = float(params[4])

    def get_variance(self, k):
        """ Returns total variance w(k) """
        # Ensure sigma > 0 for stability
        s = max(1e-6, self.sigma)
        
        # SVI Formula
        w = self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m)**2 + s**2))
        return np.maximum(0.0, w) # Variance cannot be negative

    def get_vol(self, k, T):
        """ Returns annualized Implied Volatility """
        w = self.get_variance(k)
        if T <= 1e-6: return 0.0
        return np.sqrt(w / T)

    def get_density(self, k):
        """
        Calculates Risk Neutral Density (PDF) using analytical derivatives.
        Ref: Gatheral 'The Volatility Surface'
        """
        w = self.get_variance(k)
        if w <= 1e-6: return 0.0
        
        s = max(1e-6, self.sigma)
        
        # 1. First Derivative w'(k)
        disc = np.sqrt((k - self.m)**2 + s**2)
        dw_dk = self.b * (self.rho + (k - self.m) / disc)
        
        # 2. Second Derivative w''(k)
        d2w_dk2 = self.b * (s**2) / (disc**3)
        
        # 3. Gatheral's g(k) function
        # Relates convexity of variance to probability density
        g_k = (1 - 0.5 * k * dw_dk / w)**2 - 0.25 * (dw_dk**2) * (1/w + 0.25) + 0.5 * d2w_dk2
        
        # 4. Black-Scholes d2 term
        d2 = -np.sqrt(w) / 2.0 - k / np.sqrt(w) # Log-moneyness k convention matters here!
        # Standard convention: k = ln(K/F). 
        # d2 in BS = (ln(F/K) - 0.5*v*v*t) / (v*sqrt(t)) = (-k - 0.5*w) / sqrt(w)
        d2 = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
        
        # 5. Final PDF
        # Note: This is density wrt log-strike k. 
        # To get density wrt Strike K, divide by K.
        pdf = (g_k / np.sqrt(2 * np.pi * w)) * np.exp(-0.5 * d2**2)
        
        return max(0.0, pdf)