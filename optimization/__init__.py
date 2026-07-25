"""Optimization module for parameter calibration.

Provides:
    - Fixed objective functions (corrected from C# originals)
    - Fitting routines using scipy.optimize and Optuna
    - Benchmarking utilities for comparing optimization methods
"""

from .objective_diag import neg_log_likelihood_diag, fit_diagnose_hazard
from .objective_gompertz import neg_log_likelihood_gompertz, fit_gompertz, expand_train_data
from .objective_mort import neg_log_likelihood_mort, fit_cancer_death_hazard

__all__ = [
    'neg_log_likelihood_diag',
    'neg_log_likelihood_gompertz',
    'neg_log_likelihood_mort',
    'fit_diagnose_hazard',
    'fit_gompertz',
    'fit_cancer_death_hazard',
]