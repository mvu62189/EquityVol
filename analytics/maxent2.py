import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

class MaxEntModel:
    def __init__(self, F, T, n_nodes=200, sigma_prior=0.2):
        self.F = F
        self.T = T
        self.sigma_prior = sigma_prior
        
        # 1. Define Grid (State Space)
        # Use log-spacing for better coverage of upside/downside
        # Range: +/- 8 sigma to ensure tails are covered (critical for convergence)
        std_dev = F * sigma_prior * np.sqrt(T)
        self.x = np.linspace(max(0.01, F - 8*std_dev), F + 8*std_dev, n_nodes)
        self.dx = np.diff(self.x)
        # Pad dx to match shape (trapezoidal approx)
        self.dx = np.append(self.dx, self.dx[-1])

        # 2. Construct Prior (q)
        # Lognormal density centered at F with volatility = sigma_prior
        # This acts as the "gravitational pull" for the model
        s = sigma_prior * np.sqrt(T)
        scale = F * np.exp(-0.5 * s**2)
        
        self.prior = lognorm.pdf(self.x, s, scale=scale)
        # Normalize prior on this grid
        self.prior = self.prior / np.sum(self.prior)

    def solve(self, strikes, market_prices):
        """
        Solves Dual Problem relative to Prior q.
        """
        strikes = np.array(strikes)
        prices = np.array(market_prices)
        
        # Payoff Matrix A (M x N)
        # Call Payoffs: max(x - K, 0)
        A = np.maximum(self.x[None, :] - strikes[:, None], 0)
        
        # Combine Constraints: [Martingale (x), Calls (A)]
        # G shape: (M+1, N)
        G = np.vstack([self.x, A])
        
        # Targets: [Forward, CallPrices]
        c = np.concatenate(([self.F], prices))
        
        # --- Dual Formulation with Prior ---
        # p(x) = q(x) * exp(lambda * G) / Z
        
        def objective(lam):
            # lam is (M+1,)
            # exponent: (N,)
            exponent = np.dot(lam, G)
            
            # Stable exp calculation
            max_exp = np.max(exponent)
            
            # Z = Sum[ q_i * exp(lam * G_i) ]
            # The prior 'q' enters here
            terms = self.prior * np.exp(exponent - max_exp)
            Z = np.sum(terms)
            
            log_Z = np.log(Z) + max_exp
            
            # J = log(Z) - lam * c
            return log_Z - np.dot(lam, c)

        def gradient(lam):
            exponent = np.dot(lam, G)
            max_exp = np.max(exponent)
            
            # Unnormalized p
            p_star = self.prior * np.exp(exponent - max_exp)
            Z = np.sum(p_star)
            
            # p_normalized
            p = p_star / Z
            
            # E[payoff] - MarketPrice
            # G.dot(p) gives expectation of constraints
            return np.dot(G, p) - c
        
        def hessian(lam):
            # Hessian is covariance matrix of constraints under measure p
            exponent = np.dot(lam, G)
            max_exp = np.max(exponent)
            p_star = self.prior * np.exp(exponent - max_exp)
            p = p_star / np.sum(p_star)
            
            # Expected values of constraints
            E_G = np.dot(G, p) # (M+1,)
            
            # E[G * G.T] - E[G]*E[G].T
            # Calculation: (M+1, N) * (N,) * (N, M+1) is expensive?
            # Optimized: G_weighted = G * sqrt(p)
            # H = G_weighted @ G_weighted.T - outer(E_G, E_G)
            
            G_weighted = G * np.sqrt(p)
            H = np.dot(G_weighted, G_weighted.T) - np.outer(E_G, E_G)
            return H

        # Initial Guess: Small zeros
        lam0 = np.zeros(len(c))
        
        # Optimization: Newton-CG is robust for entropy problems
        res = minimize(
            objective, lam0, method='Newton-CG', 
            jac=gradient, hess=hessian,
            options={'xtol': 1e-6, 'maxiter': 1000}
        )
        
        # Recover Distribution
        final_exp = np.dot(res.x, G)
        max_e = np.max(final_exp)
        p_final = self.prior * np.exp(final_exp - max_e)
        self.p_optimized = p_final / np.sum(p_final)
        
        # PDF Density
        self.pdf = self.p_optimized / self.dx
        
        # Check Pricing Error
        self.model_prices = A @ self.p_optimized
        
        return {
            'success': res.success,
            'message': res.message,
            'error': np.sqrt(np.mean((self.model_prices - prices)**2)),
            'p': self.p_optimized
        }