"""Sensitivity analysis module for lead time parameter values.

Evaluates how key model outputs (incidence, mortality, survival, years saved)
vary when lead time assumptions are perturbed within a specified range.

This directly addresses the fundamental non-identifiability problem:
since IncidenceAge, GrowthRate, and LeadTime cannot be simultaneously
identified from clinical diagnosis data alone, the model's predictions
are conditional on the assumed lead time distribution.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import time
import logging

from .simulation import Simulation

logger = logging.getLogger(__name__)

# Default perturbation factors for lead time means
DEFAULT_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]

# Key output metrics to track
METRICS = [
    'n_cancer_cases',
    'n_cured_clinical',
    'n_cured_screening',
    'n_screening_detected',
    'n_alive_end',
    'incidence_rates',
    'mortality_rates',
    'screened_mortality_rates',
    'survival',
    'survival_screening',
    'years_saved',
]


def run_sensitivity_analysis(
    param_source: str = "config/parameters.toml",
    seed: Optional[int] = None,
    population: int = 10000,
    years: int = 15,  # Shorter for speed
    factors: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run simulation for multiple lead time perturbation factors.

    For each factor f, lead time means are multiplied by f,
    the simulation is re-run, and key output metrics are collected.

    Parameters
    ----------
    param_source : str
        Path to the parameter file.
    seed : int, optional
        Base random seed (fixed for reproducibility within the analysis).
    population : int
        Population size per run (smaller for speed).
    years : int
        Simulation horizon per run (shorter for speed).
    factors : list of float, optional
        Perturbation factors. Default: [0.5, 0.75, 1.0, 1.25, 1.5].

    Returns
    -------
    dict with keys:
        'factors' — list of factors used
        'metrics' — dict mapping metric name -> list of values per factor
        'baseline' — baseline lead time means
        'runtime_sec' — total wall time
    """
    if factors is None:
        factors = DEFAULT_FACTORS

    t0 = time.perf_counter()
    base_sim = Simulation(param_source=param_source, seed=seed)

    # Fit first, then vary lead times
    from run_simulation import _run_fitting
    logger.info("Fitting parameters before sensitivity...")
    _run_fitting(base_sim)

    base_means = list(base_sim.params.lead_time_by_stage_means)
    base_sim.params.init_population = int(population)
    base_sim.params.years_to_simulate = int(years)

    logger.info(f"Sensitivity analysis: {len(factors)} factors, "
                f"population={population}, years={years}")
    logger.info(f"Baseline lead time means: {base_means}")

    results: Dict[str, List[Any]] = {m: [] for m in METRICS}
    all_agg_stats: Dict[str, List[np.ndarray]] = {
        'incidence_rates': [],
        'mortality_rates': [],
        'screened_mortality_rates': [],
        'survival': [],
        'survival_screening': [],
    }

    for factor in factors:
        logger.info(f"  Factor {factor:.2f}...")
        new_means = [m * factor for m in base_means]
        base_sim.params.lead_time_by_stage_means = new_means
        base_sim.params._init_lead_time()

        # Re-run simulation
        base_sim.start()
        for _ in range(int(years) - 1):
            base_sim.iterate_year()
        base_sim.stats.gather_stats(base_sim.population, base_sim.params, base_sim.current_date)

        summary = base_sim.get_summary()
        for m in METRICS:
            if m in summary:
                results[m].append(summary[m])
            elif m in base_sim.stats.agg_stats:
                results[m].append(base_sim.stats.agg_stats[m].copy())
            else:
                results[m].append(None)

        for m in all_agg_stats:
            arr = base_sim.stats.agg_stats.get(m)
            all_agg_stats[m].append(arr.copy() if arr is not None else np.zeros(1))

    dt = time.perf_counter() - t0
    logger.info(f"Sensitivity analysis complete in {dt:.1f}s.")

    return {
        'factors': factors,
        'baseline_means': base_means,
        'metrics': results,
        'agg_stats': {k: [a.tolist() for a in v] for k, v in all_agg_stats.items()},
        'runtime_sec': dt,
    }