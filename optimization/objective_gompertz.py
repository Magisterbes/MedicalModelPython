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
    lead_times: np.ndarray,
    stages: np.ndarray,
    initial_V0: float = 0.5,
) -> float:
    """Negative cross-entropy loss for Gompertz model fit.
    
    The reduced Gompertz model predicts V(t) = exp(K - exp(C - B_pop * t)).
    We interpret V(t) as a continuous proxy for stage, and use
    a Gaussian kernel to convert V(t) to soft class probabilities:
    
    p_s = exp(-(V(t) - s)^2 / (2*sigma^2)) / sum_{s'} exp(-(V(t) - s')^2/(2*sigma^2))
    
    Then minimize: -sum_i log p_{stage_i}
    
    This properly handles ordinal data without assuming continuity/normality.
    
    Parameters
    ----------
    params : np.ndarray
        [K, C, B_pop] — Gompertz model parameters (3 elements).
        B_std is fixed and not optimized here.
    lead_times : np.ndarray
        Lead time (time since incidence) for each observation.
    stages : np.ndarray
        Integer stages (1, 2, 3, or 4) for each observation.
    initial_V0 : float
        Penalty target for V(0): should be small (near 0). Default 0.5.
    
    Returns
    -------
    float
        Negative (cross-entropy loss + V0 penalty).
    """
    K, C, B_pop = params[0], params[1], params[2]
    sigma = 0.8  # Bandwidth for stage probability smoothing
    
    n = len(lead_times)
    
    # Compute V(t) for all observations
    # V(t) = exp(K - exp(C - B_pop * t))
    V = np.exp(K - np.exp(C - B_pop * lead_times))
    
    # Compute V(0) penalty
    V0 = np.exp(K - np.exp(C))
    penalty = (V0 - initial_V0) ** 2 * 1.0  # Weight can be adjusted
    
    # Cross-entropy: for each observation, compute softmax probabilities
    # over stages {1, 2, 3, 4} based on distance from V
    stage_targets = np.array([1.0, 2.0, 3.0, 4.0])
    
    ll = 0.0
    for i in range(n):
        dist_sq = (V[i] - stage_targets) ** 2
        logits = -dist_sq / (2.0 * sigma ** 2)
        # Softmax with max trick for numerical stability
        logits_max = np.max(logits)
        probs = np.exp(logits - logits_max)
        probs = probs / probs.sum()
        
        s_idx = int(stages[i]) - 1
        if 0 <= s_idx < 4 and probs[s_idx] > 1e-300:
            ll += np.log(probs[s_idx])
        else:
            ll += np.log(1e-300)  # Floor for numerical safety
    
    # Weighted sum: -LL + penalty
    # Normalize by n for scale
    loss = -ll / float(n) + penalty * 0.1
    
    return loss


def fit_gompertz(
    initial_params: np.ndarray,
    lead_times: np.ndarray,
    stages: np.ndarray,
    is_aggressive: bool = False,
    method: str = 'Nelder-Mead',
    bounds: Tuple[float, float] = (-3.0, 3.0),
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit Gompertz model parameters.
    
    Parameters
    ----------
    initial_params : np.ndarray
        Initial [K, C, B_pop, B_std]. Only first 3 are optimized.
    lead_times : np.ndarray
        Lead time data for each observation.
    stages : np.ndarray
        Integer stage data (1-4).
    is_aggressive : bool
        Whether fitting aggressive subtype.
    method : str
        'differential_evolution' (recommended), 'Nelder-Mead', 'L-BFGS-B'.
    bounds : tuple
        Bounds for K, C, B_pop.
    verbose : bool
        Print convergence details.
    
    Returns
    -------
    dict with fitted parameters and diagnostics.
    """
    K0, C0, B0 = float(initial_params[0]), float(initial_params[1]), float(initial_params[2])
    x0 = np.array([K0, C0, B0])
    
    if method == 'differential_evolution':
        bnds = [bounds, bounds, bounds]
        result = differential_evolution(
            neg_log_likelihood_gompertz,
            args=(lead_times, stages),
            bounds=bnds,
            maxiter=500,
            tol=1e-8,
            disp=verbose,
            seed=42,
            polish=True,  # Refine with L-BFGS-B at the end
        )
        n_iter = result.nfev
        success = result.success
    elif method == 'Nelder-Mead':
        result = minimize(
            neg_log_likelihood_gompertz,
            x0=x0,
            args=(lead_times, stages),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': verbose},
        )
        n_iter = result.nfev
        success = result.success
    elif method == 'L-BFGS-B':
        bnds = [bounds, bounds, bounds]
        result = minimize(
            neg_log_likelihood_gompertz,
            x0=x0,
            args=(lead_times, stages),
            method='L-BFGS-B',
            bounds=bnds,
            options={'maxiter': 2000, 'ftol': 1e-8, 'disp': verbose},
        )
        n_iter = result.nit
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
        'is_aggressive': is_aggressive,
    }


def expand_train_data(
    train_data: pd.DataFrame,
    lead_time_rates: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """Expand training data with sampled lead times.
    
    Equivalent to C# AdjustParamsGompertz.ExpandTrain().
    Each row is repeated 5 times with different sampled lead times
    to increase effective sample size.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        Individual staging data with 'Age', 'Stage', 'Aggressiveness'.
    lead_time_rates : np.ndarray
        Exponential rates (1/mean) for each stage.
    seed : int
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Expanded training data with 'LeadTime' column added.
    """
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    
    rows = []
    for i in range(len(train_data)):
        row = train_data.iloc[i]
        stage_idx = int(row['Stage']) - 1
        rate = lead_time_rates[stage_idx]
        
        for _ in range(5):  # 5 repeats per observation (matching C#)
            lead_time = np.ceil(rng.exponential(1.0 / rate))
            new_row = row.to_dict()
            new_row['LeadTime'] = float(lead_time)
            rows.append(new_row)
    
    result = pd.DataFrame(rows)
    
    # Compute stage weights (inverse frequency)
    stage_counts = result['Stage'].value_counts().sort_index()
    result['stage_weight'] = result['Stage'].map(
        lambda s: len(result) / (stage_counts.get(s, 1) * 4.0)
    )
    
    return result