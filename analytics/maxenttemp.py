import numpy as np
from scipy.optimize import minimize

class MaxEntModel:
    def __init__(self, F, T, n_nodes=200, custom_prior=None):
        self.F = F
        self.T = T
        
        # 1. Define Grid (x)
        # We need a wide grid to capture tails. 
        # range: 0.01F to 4F is usually safe for equities.
        self.x_min = 0.01 * F
        self.x_max = 4.0 * F
        self.x = np.linspace(self.x_min, self.x_max, n_nodes)
        self.dx = self.x[1] - self.x[0]
        
        # 2. Define Prior (q)
        if custom_prior is not None:
            # Interpolate custom prior to current grid if needed
            # Assuming custom_prior matches n_nodes for simplicity here
            self.prior = custom_prior
        else:
            # Default: Lognormal Prior (Black-Scholes with roughly 20% vol)
            # This is just a fallback; the solver will twist it anyway.
            sigma = 0.5 
            mu = np.log(F) - 0.5*sigma**2
            self.prior = (1/(self.x * sigma * np.sqrt(2*np.pi))) * np.exp(-(np.log(self.x) - mu)**2 / (2*sigma**2))
            
        # Normalize Prior
        self.prior = self.prior / np.sum(self.prior)

    def solve(self, strikes, market_prices):
        """
        Robust Solver: Tries Newton-CG (Fast/Precise) first.
        If that fails/warns, falls back to L-BFGS-B (Robust).
        """
        # 1. Setup Data
        self.strikes = np.array(strikes)
        self.market_prices = np.array(market_prices)
        
        # A matrix: Payoffs (N_strikes x N_nodes)
        # A[i, j] = max(S_j - K_i, 0)
        self.A = np.maximum(self.x[None, :] - self.strikes[:, None], 0)
        
        # Constraint Matrix G: [S, Payoffs]
        # Row 0: Underlying Price (Forward constraint)
        # Rows 1..N: Option Payoffs
        self.G = np.vstack([self.x, self.A])
        
        # Target vector c: [F, Market_Prices]
        self.c = np.concatenate(([self.F], self.market_prices))
        
        # Initial Guess (Zeros usually works for Lagrange multipliers)
        lambda_0 = np.zeros(len(self.c))

        # --- Solver Strategy ---
        method_used = "Newton-CG"
        
        # Attempt 1: Newton-CG (Best for clean data)
        try:
            res = minimize(
                self._dual_objective, 
                lambda_0, 
                method='Newton-CG', 
                jac=self._dual_gradient, 
                hess=self._dual_hessian,
                options={'xtol': 1e-6, 'maxiter': 1000}
            )
        except Exception as e:
            res = None

        # Attempt 2: L-BFGS-B (Fallback for noisy data)
        # If Newton failed or didn't converge successfully
        if res is None or (not res.success and res.message != 'Optimization terminated successfully.'):
            print(f"Warning: Newton-CG failed. Retrying with L-BFGS-B.")
            res = minimize(
                self._dual_objective, 
                lambda_0, 
                method='L-BFGS-B', 
                jac=self._dual_gradient,
                # No Hessian needed for L-BFGS-B
                options={'ftol': 1e-9, 'maxiter': 5000}
            )
            method_used = "L-BFGS-B"

        # --- Process Result ---
        self.lambdas = res.x
        
        # Reconstruct Probability Distribution from Lambdas
        # p(x) = prior(x) * exp( lambda * G(x) )
        # We subtract max_exponent for numerical stability (Log-Sum-Exp trick)
        exponents = np.dot(self.lambdas, self.G)
        max_exp = np.max(exponents)
        
        numerator = self.prior * np.exp(exponents - max_exp)
        norm_factor = np.sum(numerator)
        self.pdf = numerator / norm_factor # Discrete Probability Mass
        
        # Calculate Model Prices
        # Price = sum( p_i * Payoff_i )
        self.model_prices = self.A @ self.pdf
        
        # Convert Discrete Mass to Density for plotting
        # PDF_density = Probability / dx
        self.pdf_density = self.pdf / self.dx
        
        self.solver_status = {
            'success': res.success,
            'method': method_used,
            'message': res.message,
            'pricing_error': np.sqrt(np.mean((self.model_prices - self.market_prices)**2))
        }
        
        return self.solver_status

    # --- Internal Math Helpers ---

    def _dual_objective(self, lambdas):
        """
        Calculates the Dual Function J(lambda).
        J(lambda) = ln( Z(lambda) ) - lambda . c
        Minimizing this is equivalent to Maximizing Entropy.
        """
        exponents = np.dot(lambdas, self.G)
        max_exp = np.max(exponents)
        
        # Z = sum( q_i * exp(exponents) )
        Z_terms = self.prior * np.exp(exponents - max_exp)
        Z = np.sum(Z_terms)
        
        return np.log(Z) + max_exp - np.dot(lambdas, self.c)

    def _dual_gradient(self, lambdas):
        """
        Gradient of the Dual Objective.
        Grad = E_p[G] - c
        """
        exponents = np.dot(lambdas, self.G)
        max_exp = np.max(exponents)
        
        numerator = self.prior * np.exp(exponents - max_exp)
        p = numerator / np.sum(numerator)
        
        E_G = np.dot(self.G, p)
        return E_G - self.c

    def _dual_hessian(self, lambdas):
        """
        Hessian of the Dual Objective.
        H = Covariance(G) under distribution p.
        """
        exponents = np.dot(lambdas, self.G)
        max_exp = np.max(exponents)
        
        numerator = self.prior * np.exp(exponents - max_exp)
        p = numerator / np.sum(numerator)
        
        E_G = np.dot(self.G, p)
        
        # Weighted Correlation Matrix
        G_weighted = self.G * p[None, :]
        E_GG = np.dot(G_weighted, self.G.T)
        
        return E_GG - np.outer(E_G, E_G)