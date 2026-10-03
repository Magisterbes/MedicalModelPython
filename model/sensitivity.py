"""Sensitivity analysis module for lead time parameter values.

Evaluates how key model outputs (incidence, mortality, survival, years saved)
vary when lead time assumptions are perturbed within a specified range.

This directly addresses the fundamental non-identifiability problem:
since IncidenceAge, GrowthRate, and LeadTime cannot be simultaneously
identified from clinical diagnosis data alone, the model's predictions
are conditional on the assumed lead time distribution.
"""

from typing import Dict, List, Any, Optional
import os
import pickle
import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import savgol_filter

from .simulation import Simulation
from .random import get_random

logger = logging.getLogger(__name__)

# Env flag marking a spawned worker process. Workers inherit it and therefore
# never spawn again — a robust safeguard against recursive process creation on
# Windows (spawn) when the caller has no `if __name__ == '__main__'` guard.
_CHILD_FLAG = 'MEDICAL_SENS_MP_CHILD'

# Aggregate stat keys tracked across factors (order preserved for the API)
_AGG_KEYS = [
    'incidence_rates',
    'mortality_rates',
    'screened_mortality_rates',
    'survival',
    'survival_screening',
    'years_saved',
    'diagnose_stages_distribution',
    'screening_stages_distribution',
]

# Auto-parallelism threshold: a factor is only worth running in a separate
# process when its simulation is heavy enough to amortize process start-up
# (on Windows "spawn" each worker re-imports NumPy/Numba, ~2s per pool).
# Below this many "agent-years" per factor the sequential path is faster.
_MP_MIN_AGENT_YEARS = 10_000_000

# Curves smoothed with Savitzky-Golay for cleaner visualization
_SMOOTH_KEYS = [
    'incidence_rates',
    'mortality_rates',
    'screened_mortality_rates',
    'survival',
    'survival_screening',
    'years_saved',
]

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


def _mp_context():
    """Return a start method that is safe to spawn from a background thread.

    The Flask web UI runs this analysis in a daemon thread. On POSIX the default
    start method is ``fork``, which is unsafe when called from a non-main thread
    (the child may inherit locked mutexes and deadlock). ``forkserver`` uses a
    clean, single-threaded server process instead; on Windows ``spawn`` is used.
    """
    if os.name == 'posix':
        try:
            return mp.get_context('forkserver')
        except (ValueError, RuntimeError):
            pass
    try:
        return mp.get_context('spawn')
    except (ValueError, RuntimeError):
        return mp.get_context()


def _simulate(sim: "Simulation", years: int) -> None:
    """Run one full simulation pipeline for `years` years (shared code path)."""
    sim.start()
    for _ in range(int(years) - 1):
        sim.iterate_year()
    sim.stats.gather_stats(sim.population, sim.params, sim.current_date)


def _collect(sim: "Simulation"):
    """Collect METRICS values and aggregate arrays from a finished simulation."""
    summary = sim.get_summary()
    metrics: Dict[str, Any] = {}
    for m in METRICS:
        if m in summary:
            metrics[m] = summary[m]
        elif m in sim.stats.agg_stats:
            metrics[m] = sim.stats.agg_stats[m].copy()
        else:
            metrics[m] = None
    agg: Dict[str, np.ndarray] = {}
    for m in _AGG_KEYS:
        arr = sim.stats.agg_stats.get(m)
        agg[m] = arr.copy() if arr is not None else np.zeros(1)
    return metrics, agg


def _run_sensitivity_job(job):
    """Run a single sensitivity factor in isolation.

    Module-level (picklable) so it can be dispatched to worker processes on
    Windows (spawn). Each job re-creates the global RNG with the same base seed
    and reuses the already-fitted parameters, so a factor's result is a pure
    function of (seed, factor) — identical to the sequential implementation.
    """
    seed, population, years, factor, base_means, params_blob = job
    params = pickle.loads(params_blob)
    params.init_population = int(population)
    params.years_to_simulate = int(years)
    params.lead_time_by_stage_means = [m * factor for m in base_means]
    params._init_lead_time()
    # from_params skips re-reading the parameter file and retraining regressions
    sim = Simulation.from_params(params, seed=seed)
    _simulate(sim, years)
    return _collect(sim)


def run_sensitivity_analysis(
    param_source: str = "config/parameters.toml",
    seed: Optional[int] = None,
    population: int = 100000,
    years: int = 15,  # Shorter for speed
    factors: Optional[List[float]] = None,
    n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Run simulation for multiple lead time perturbation factors.

    For each factor f, lead time means are multiplied by f,
    the simulation is re-run, and key output metrics are collected.

    Factors are independent — each result is a pure function of
    (seed, factor) — so they are executed in parallel across CPU cores via a
    process pool. Because the base seed is fixed per factor, results are
    identical to a sequential run.

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
    n_jobs : int, optional
        Number of worker processes. ``None`` (default) auto-selects: factors
        are run in parallel only for heavy workloads (more than ~10M
        agent-years per factor), otherwise sequentially — this avoids paying
        process start-up cost for small jobs. Use ``1`` to force sequential and
        ``-1`` to use ``cpu_count() // 2`` workers.

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
    # Resolve the effective seed once so every factor (and worker) uses the same
    # stream, even when seed=None was requested.
    effective_seed = get_random().seed

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

    params_blob = pickle.dumps(base_sim.params)
    jobs = [(effective_seed, population, years, f, base_means, params_blob)
            for f in factors]

    in_child = os.environ.get(_CHILD_FLAG) == '1'

    if n_jobs is None:
        # Auto: parallelize only when each factor's simulation is heavy enough
        # to amortize process start-up; small jobs are faster sequentially.
        auto_ok = int(population) * int(years) >= _MP_MIN_AGENT_YEARS
        worker_cap = max(1, (os.cpu_count() or 1) // 2)
        n_jobs = min(len(factors), worker_cap) if auto_ok else 1
    elif n_jobs < 0:
        n_jobs = max(1, (os.cpu_count() or 1) // 2)

    use_mp = (not in_child) and n_jobs > 1 and len(factors) > 1

    outputs = None
    if use_mp:
        logger.info(f"Running {len(factors)} factors in parallel (n_jobs={n_jobs})...")
        os.environ[_CHILD_FLAG] = '1'  # inherited by workers → they run sequentially
        try:
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_mp_context()) as ex:
                outputs = list(ex.map(_run_sensitivity_job, jobs))
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"Multiprocessing failed ({exc}); falling back to sequential.")
            outputs = None
        finally:
            os.environ.pop(_CHILD_FLAG, None)

    if outputs is None:
        logger.info("Running factors sequentially...")
        outputs = []
        for job in jobs:
            logger.info(f"  Factor {job[3]:.2f}...")
            outputs.append(_run_sensitivity_job(job))

    results: Dict[str, List[Any]] = {m: [] for m in METRICS}
    all_agg_stats: Dict[str, List[np.ndarray]] = {m: [] for m in _AGG_KEYS}
    for metrics, agg in outputs:
        for m in METRICS:
            results[m].append(metrics[m])
        for m in _AGG_KEYS:
            all_agg_stats[m].append(agg[m])

    dt = time.perf_counter() - t0
    logger.info(f"Sensitivity analysis complete in {dt:.1f}s.")

    # Smooth rate curves with Savitzky-Golay filter for cleaner visualization
    for key in _SMOOTH_KEYS:
        if key in all_agg_stats and len(all_agg_stats[key]) > 0:
            win = min(11, len(all_agg_stats[key][0]) - 2)
            if win >= 5 and win % 2 == 1:
                for i in range(len(all_agg_stats[key])):
                    all_agg_stats[key][i] = savgol_filter(all_agg_stats[key][i], win, 3)
                    all_agg_stats[key][i] = np.maximum(all_agg_stats[key][i], 0)
    
    return {
        'factors': factors,
        'baseline_means': base_means,
        'metrics': results,
        'agg_stats': {k: [a.tolist() for a in v] for k, v in all_agg_stats.items()},
        'runtime_sec': dt,
    }
