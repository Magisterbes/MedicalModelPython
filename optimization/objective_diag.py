"""Objective function and fitting for diagnosis hazard parameters.

FIXED from C# original:
    - Uses CORRECT Poisson log-likelihood instead of incorrect binomial LL
      for proportion data: C(a) ~ Poisson(h(a) * P(a))
    - L = sum_a [ C(a)*log(h(a)*P(a)) - h(a)*P(a) ]
    
    The C# original used: sum_a [ I(a)*log(h) + (1-I(a))*log(1-h) ]
    which is invalid for proportion I(a) = C(a)/P(a).
    
Fitting uses L-BFGS-B (smooth, 6 parameters) with bounds.
Nelder-Mead kept as baseline for comparison.
"""

from typing import Callable, Tuple, Dict, Any, List
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import logging

logger = logging.getLogger(__name__)


def neg_log_likelihood_diag(
    betas: np.ndarray,
    train_incidence: np.ndarray,
    train_population: np.ndarray,
) -> float:
    """Negative Poisson log-likelihood for diagnosis hazard.
    
    Models: C(a) ~ Poisson(h(a) * P(a))
    where:
        h(a) = exp(beta0 + sum_k beta_k * H(a - t_k))
        C(a) = observed cases at age a
        P(a) = population at age a
    
    Parameters
    ----------
    betas : np.ndarray
        Log-hazard coefficients [beta0, beta1, ..., beta_K].
        beta0 is the base log-hazard (at age -1 in C# convention).
        beta_k for k >= 1 are step changes at breakpoints.
    train_incidence : np.ndarray
        Observed incidence proportions I(a) = C(a) / P(a).
    train_population : np.ndarray
        Population P(a) at each age.
    
    Returns
    -------
    float
        Negative Poisson log-likelihood (to be minimized).
    """
    n_ages = min(len(train_incidence), len(train_population))
    
    # Reconstruct hazard values from piecewise betas
    # Breakpoints matching C# definition: 
    # t_k = [-1, 0, 40, 50, 60, 70] for default 6-coefficient config
    # beta0 acts from age -1 (i.e., always active)
    breakpoints = np.array([-1, 0, 40, 50, 60, 70], dtype=np.float64)
    
    # Ensure betas length matches breakpoints
    K = min(len(betas), len(breakpoints))
    
    hazards = np.zeros(n_ages)
    for a in range(n_ages):
        log_h = betas[0]  # Always-active base
        for k in range(1, K):
            if a >= breakpoints[k]:
                log_h += betas[k]
        h_val = np.exp(log_h)
        hazards[a] = min(h_val, 1.0)  # Clamp to [0, 1] (matching C#)
    
    # Poisson log-likelihood
    # C(a) = I(a) * P(a)
    cases = train_incidence[:n_ages] * train_population[:n_ages]
    rates = hazards * train_population[:n_ages]
    
    # L = sum [ C*log(lambda) - lambda ]
    # Avoid log(0) issues
    ll = 0.0
    for a in range(n_ages):
        if rates[a] > 0 and cases[a] > 0:
            ll += cases[a] * np.log(rates[a]) - rates[a]
        elif cases[a] == 0:
            ll -= rates[a]  # Poisson: 0*log(lambda) - lambda = -lambda
    
    return -ll  # Negative for minimization


def fit_diagnose_hazard(
    initial_betas: np.ndarray,
    train_incidence: np.ndarray,
    train_population: np.ndarray,
    method: str = 'L-BFGS-B',
    bounds: Tuple[float, float] = (-5.0, 5.0),
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit diagnosis hazard parameters via maximum likelihood.
    
    Parameters
    ----------
    initial_betas : np.ndarray
        Initial parameter values.
    train_incidence : np.ndarray
        Training incidence proportions.
    train_population : np.ndarray
        Training population counts.
    method : str
        Optimization method: 'L-BFGS-B', 'Nelder-Mead', 'differential_evolution'.
    bounds : tuple
        Bounds for each parameter (min, max).
    verbose : bool
        If True, print convergence details.
    
    Returns
    -------
    dict with keys:
        'betas' : fitted parameters
        'neg_ll' : final negative log-likelihood
        'n_iter' : number of iterations
        'success' : convergence flag
        'method' : method used
    """
    n_params = len(initial_betas)
    
    if method == 'L-BFGS-B':
        bnds = [bounds for _ in range(n_params)]
        result = minimize(
            neg_log_likelihood_diag,
            x0=initial_betas,
            args=(train_incidence, train_population),
            method='L-BFGS-B',
            bounds=bnds,
            options={'maxiter': 2000, 'ftol': 1e-8, 'disp': verbose},
        )
    elif method == 'Nelder-Mead':
        result = minimize(
            neg_log_likelihood_diag,
            x0=initial_betas,
            args=(train_incidence, train_population),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': verbose},
        )
    elif method == 'differential_evolution':
        bnds = [bounds for _ in range(n_params)]
        result = differential_evolution(
            neg_log_likelihood_diag,
            args=(train_incidence, train_population),
            bounds=bnds,
            maxiter=1000,
            tol=1e-8,
            disp=verbose,
            seed=42,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'betas': result.x,
        'neg_ll': float(result.fun),
        'n_iter': getattr(result, 'nit', getattr(result, 'nfev', 0)),
        'success': bool(result.success),
        'method': method,
    }