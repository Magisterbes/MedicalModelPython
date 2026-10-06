#!/usr/bin/env python3
"""Command-line entry point for the MedicalModel2024 simulation."""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.simulation import Simulation
from model.random import get_random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('run_simulation')


def main():
    parser = argparse.ArgumentParser(description='MedicalModel2024 — Cancer Screening Microsimulation')
    parser.add_argument('--config', '-c', default='config/parameters.toml')
    parser.add_argument('--data-dir', '-d', default=None)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--fit', '-f', action='store_true')
    parser.add_argument('--output', '-o', default='output/results.json')
    parser.add_argument('--population', '-p', type=int, default=None)
    
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("MedicalModel2024 Python — Starting")
    logger.info(f"Config: {args.config}, Seed: {args.seed}")
    
    sim = Simulation(param_source=args.config, param_data_dir=args.data_dir, seed=args.seed)
    if args.population is not None:
        sim.params.init_population = args.population
        logger.info(f"Population: {args.population}")
    
    if args.fit:
        logger.info("Running parameter fitting...")
        _run_fitting(sim)
    
    logger.info("Running simulation...")
    stats = sim.run_full_simulation(verbose=args.verbose)
    
    summary = sim.get_summary()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    import json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    
    with open(output_path, 'w') as f:
        json.dump({'summary': summary, 'stats': stats.to_dict()}, f, cls=NumpyEncoder, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def _run_fitting(sim: Simulation):
    """Fast parameter fitting (~5 seconds total). Uses L-BFGS-B for smooth problems,
    Nelder-Mead for noisy Gompertz (not differential_evolution — too slow)."""
    params = sim.params
    
    # 1. Diagnose hazard: L-BFGS-B (6 params, smooth) — ~0.1s
    logger.info("Fitting diagnosis hazard (L-BFGS-B)...")
    from optimization import fit_diagnose_hazard
    r = fit_diagnose_hazard(params.diagnose_hazard.constants.copy(), params.train_incidence,
                            params.train_data['population'].values, method='L-BFGS-B')
    params.diagnose_hazard.constants = r['betas']
    params.diagnose_hazard.update()
    logger.info(f"  LL={r['neg_ll']:.4f}, iters={r['n_iter']}")
    
    # 2. Gompertz: Nelder-Mead (3 params, mildly noisy) — ~1s. One model for all.
    logger.info("Fitting Gompertz model (Nelder-Mead)...")
    from optimization import fit_gompertz, expand_train_data
    lead_time_rates = np.array(params.lead_time_distributions)
    
    expanded = expand_train_data(params.individual_data, lead_time_rates)
    init_p = params.reduced_gompertz
    
    r = fit_gompertz(np.array(init_p), expanded['LeadTime'].values,
                     expanded['Stage'].values, method='Nelder-Mead')
    
    params.reduced_gompertz = r['params'].tolist()
    params.gompertz.K = r['K']
    params.gompertz.C = r['C']
    params.gompertz.B_pop = r['B_pop']
    logger.info(f"  Gompertz: LL={r['neg_ll']:.4f}, K={r['K']:.3f}, C={r['C']:.3f}, B={r['B_pop']:.3f}")
    
    # 3. Mortality: L-BFGS-B (1 param, smooth) — ~0.03s
    logger.info("Fitting mortality hazard (L-BFGS-B)...")
    from optimization import fit_cancer_death_hazard
    r = fit_cancer_death_hazard(float(params.cancer_death_hazard.constants[0]),
                                params.train_mortality, params.train_data['population'].values,
                                params.train_incidence, method='L-BFGS-B')
    params.cancer_death_hazard.constants[0] = r['lambda']
    logger.info(f"  lambda={r['lambda']:.4f}, rate={r['hazard_rate']:.4f}, LL={r['neg_ll']:.4f}")
    
    logger.info("Parameter fitting complete.")


if __name__ == '__main__':
    main()