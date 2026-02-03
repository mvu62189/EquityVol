import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm

class MaxEntModel:
    def __init__(self, F, T, n_nodes=200, sigma_prior=0.2, custom_prior=None, grid_bounds=None):
        self.F = F
        self.T = T
        
        # 1. Define Grid (State Space)
        # We need a dense grid to capture the shape

        if grid_bounds is not None:
            # [CHANGE] Use user-calculated bounds (e.g. based on strikes)
            x_min, x_max = grid_bounds
            # Ensure safety buffers
            x_min = max(0.01, x_min)
            self.x = np.linspace(x_min, x_max, n_nodes)
        else:
            # Fallback: Theoretical 6-sigma range
            std_dev = F * sigma_prior * np.sqrt(T)
            self.x = np.linspace(max(0.01, F - 6*std_dev), F + 6*std_dev, n_nodes)
        
        self.dx = np.diff(self.x)
        self.dx = np.append(self.dx, self.dx[-1])

        # 2. Construct Prior (q)
        if custom_prior is not None:
            # A. Use provided custom prior (e.g. SVI)
            if len(custom_prior) != n_nodes:
                # Interpolate if dimensions mismatch
                old_x = np.linspace(self.x[0], self.x[-1], len(custom_prior))
                self.prior = np.interp(self.x, old_x, custom_prior)
            else:
                self.prior = np.array(custom_prior)
        else:
            # B. Fallback: Lognormal density (Flat Vol)
            s = sigma_prior * np.sqrt(T)
            scale = F * np.exp(-0.5 * s**2)
            self.prior = lognorm.pdf(self.x, s, scale=scale)

        # Normalize prior ensuring sum(p) = 1
        self.prior = np.maximum(self.prior, 1e-12) # Safety floor
        self.prior = self.prior / np.sum(self.prior)

    def solve(self, strikes, market_prices):
        """
        Solves Dual Problem: Min Relative Entropy vs Prior q.
        """
        strikes = np.array(strikes)
        prices = np.array(market_prices)
        
        # Payoff Matrix A
        A = np.maximum(self.x[None, :] - strikes[:, None], 0)
        
        # Constraints Matrix G: [Martingale; Calls]
        G = np.vstack([self.x, A])
        c = np.concatenate(([self.F], prices))
        
        # --- Newton-CG Optimization ---
        
        def objective(lam):
            exponent = np.dot(lam, G)
            max_exp = np.max(exponent)
            # The prior 'q' is the base measure
            terms = self.prior * np.exp(exponent - max_exp)
            Z = np.sum(terms)
            return np.log(Z) + max_exp - np.dot(lam, c)

        def gradient(lam):
            exponent = np.dot(lam, G)
            max_exp = np.max(exponent)
            p_star = self.prior * np.exp(exponent - max_exp)
            p = p_star / np.sum(p_star)
            return np.dot(G, p) - c
        
        def hessian(lam):
            exponent = np.dot(lam, G)
            max_exp = np.max(exponent)
            p_star = self.prior * np.exp(exponent - max_exp)
            p = p_star / np.sum(p_star)
            
            G_weighted = G * np.sqrt(p)
            E_G = np.dot(G, p)
            return np.dot(G_weighted, G_weighted.T) - np.outer(E_G, E_G)

        lam0 = np.zeros(len(c))
        
        try:
            res = minimize(
                objective, lam0, method='Newton-CG', 
                jac=gradient, hess=hessian,
                options={'xtol': 1e-8, 'maxiter': 50000}
            )
            success = res.success
            msg = res.message
            lam_final = res.x
        except Exception as e:
            success = False
            msg = str(e)
            lam_final = lam0

        # Recover Distribution
        final_exp = np.dot(lam_final, G)
        max_e = np.max(final_exp)
        p_final = self.prior * np.exp(final_exp - max_e)
        self.p_optimized = p_final / np.sum(p_final)
        
        self.pdf = self.p_optimized / self.dx
        self.model_prices = A @ self.p_optimized
        
        return {
            'success': success,
            'message': msg,
            'error': np.sqrt(np.mean((self.model_prices - prices)**2)),
            'p': self.p_optimized
        }