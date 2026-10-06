"""Benchmark fitting speed to find performance bottlenecks."""
import time
import numpy as np
from model.parameters import Parameters

params = Parameters.from_toml('config/parameters.toml')
params.data_dir = 'data'
t0 = time.time()
params.init_data()
print(f"init_data: {time.time() - t0:.2f}s")

from optimization import fit_diagnose_hazard, expand_train_data, fit_gompertz, fit_cancer_death_hazard

# Diagnose hazard (L-BFGS-B, 6 params, smooth)
t0 = time.time()
r = fit_diagnose_hazard(
    params.diagnose_hazard.constants.copy(),
    params.train_incidence,
    params.train_data['population'].values,
    method='L-BFGS-B',
)
print(f"Diag L-BFGS-B: {time.time() - t0:.2f}s, LL={r['neg_ll']:.4f}")

# Gompertz (Nelder-Mead, 3 params, fast) — one model for all
lead_time_rates = np.array(params.lead_time_distributions)
expanded = expand_train_data(params.individual_data, lead_time_rates)
init_p = params.reduced_gompertz
t0 = time.time()
r = fit_gompertz(
    np.array(init_p),
    expanded['LeadTime'].values,
    expanded['Stage'].values,
    method='Nelder-Mead',
)
dt = time.time() - t0
nll = r['neg_ll']
print(f"Gompertz: {dt:.2f}s, LL={nll:.4f}")

# Mortality (L-BFGS-B, 1 param, smooth)
t0 = time.time()
r = fit_cancer_death_hazard(
    float(params.cancer_death_hazard.constants[0]),
    params.train_mortality,
    params.train_data['population'].values,
    params.train_incidence,
    method='L-BFGS-B',
)
print(f"Mort L-BFGS-B: {time.time() - t0:.2f}s, LL={r['neg_ll']:.4f}")

print("\nDone. Fits: diagnose hazard (L-BFGS-B), single Gompertz (Nelder-Mead), mortality (L-BFGS-B).")