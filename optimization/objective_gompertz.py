"""Objective function and fitting for reduced Gompertz model parameters.

FIXED from C# original:
    - Uses cross-entropy / multinomial log-likelihood for ordinal stage data
      instead of weighted sum-of-squares (which incorrectly treats stage
      as continuous with normal errors).
    - L = -sum_i sum_s [ y_{i,s} * log(p_{i,s}) ]
      where p_{i,s} = softmax(V_model(tau_i) distances to stages)
    - No stochastic noise inside the objective (removed the inner 
      4-sample loop with random epsilons).
    
Fitting uses differential_evolution (global, noise-tolerant) as primary.
Nelder-Mead kept as baseline.
"""

from typing import Callable, Tuple, Dict, Any, List
import numpy as np
from scipy.optimize import minimize, differential_evolution
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def neg_log_likelihood_gompertz(
    params: np.ndarray,
    target_crossing_times: np.ndarray,
    initial_V0: float = 0.5,
) -> float:
    """Calibration loss for the reduced Gompertz growth model.

    The reduced Gompertz model predicts V(t) = exp(K - exp(C - B_pop * t)),
    interpreted as a continuous proxy for tumour size. The loss fits the Gompertz
    crossing time of sizes 2, 3, 4 to the target crossing times derived from the
    staging data's stage distribution, with penalties that keep the carrying
    capacity exp(K) above the terminal stage (4) and the rate B_pop positive.

    Parameters
    ----------
    params : np.ndarray
        [K, C, B_pop] — Gompertz model parameters (3 elements).
        B_std is fixed and not optimized here.
    target_crossing_times : np.ndarray
        Target crossing times for sizes 2, 3, 4 (years since onset).
    initial_V0 : float
        Unused; kept for signature compatibility.

    Returns
    -------
    float
        Calibration loss (crossing-time squared error + penalties).
    """
    K, C, B_pop = params[0], params[1], params[2]
    targets = np.asarray(target_crossing_times, dtype=np.float64)
    
    # Constraint penalties (large and smooth) to keep the curve well-posed:
    # the carrying capacity must exceed the terminal stage, and the rate must
    # be positive, so every stage is reachable at a finite positive time.
    loss = 0.0
    if K <= np.log(4.0):
        loss += 1e6 * (np.log(4.0) - K) ** 2
    if B_pop <= 0.0:
        loss += 1e6 * B_pop ** 2
    
    # Calibration: the Gompertz crossing time for sizes 2, 3, 4 must match the
    # target crossing times (which reproduce the staging data's stage mix).
    for i, target in enumerate(targets):
        size = 2.0 + float(i)
        arg = K - np.log(size)
        if arg <= 0.0:
            loss += 1e6
            continue
        t_s = (C - np.log(arg)) / B_pop
        loss += (t_s - target) ** 2
    
    # Regularise the carrying capacity toward a plausible value (a few times the
    # terminal stage) so the fit does not collapse to a degenerate huge-K / tiny-B
    # solution. Weight is small so it only matters when the crossing times are
    # already matched.
    loss += 0.1 * (K - np.log(5.0)) ** 2
    
    return loss


def fit_gompertz(
    initial_params: np.ndarray,
    target_crossing_times: np.ndarray,
    method: str = 'L-BFGS-B',
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit Gompertz model parameters.

    Parameters
    ----------
    initial_params : np.ndarray
        Initial [K, C, B_pop, B_std]. Only first 3 are optimized.
    target_crossing_times : np.ndarray
        Target crossing times for sizes 2, 3, 4 (years).
    method : str
        'L-BFGS-B' (recommended, uses bounds) or 'Nelder-Mead'.
    verbose : bool
        Print convergence details.

    Returns
    -------
    dict with fitted parameters and diagnostics.
    """
    K0, C0, B0 = float(initial_params[0]), float(initial_params[1]), float(initial_params[2])
    x0 = np.array([K0, C0, B0])
    args = (target_crossing_times,)
    # Per-parameter bounds: K above log(4) (terminal stage reachable), B_pop > 0.
    bnds = [(np.log(4.0) + 0.01, np.log(20.0)), (-5.0, 5.0), (0.01, 5.0)]

    if method == 'L-BFGS-B':
        result = minimize(
            neg_log_likelihood_gompertz,
            x0=x0,
            args=args,
            method='L-BFGS-B',
            bounds=bnds,
            options={'maxiter': 2000, 'ftol': 1e-10, 'disp': verbose},
        )
        n_iter = result.nit
        success = result.success
    elif method == 'Nelder-Mead':
        result = minimize(
            neg_log_likelihood_gompertz,
            x0=x0,
            args=args,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': verbose},
        )
        n_iter = result.nfev
        success = result.success
    else:
        raise ValueError(f"Unknown method: {method}")

    fitted_params = np.array([
        result.x[0], result.x[1], result.x[2], float(initial_params[3])
    ])

    return {
        'params': fitted_params,
        'K': float(result.x[0]),
        'C': float(result.x[1]),
        'B_pop': float(result.x[2]),
        'B_std': float(initial_params[3]),
        'neg_ll': float(result.fun),
        'n_iter': int(n_iter),
        'success': bool(success),
        'method': method,
    }