# MedicalModel2024 Python — Cancer Screening Microsimulation

Agent-based microsimulation model for evaluating cancer screening programmes.  
**Language:** Python 3.11+ | **UI:** Flask + Plotly.js | **Performance:** NumPy + Numba

## Quick Start

```bash
pip install -r requirements.txt

# CLI: run with 50K agents
python run_simulation.py --population 50000 --seed 42

# CLI: run with calibration (Fit + Simulate)
python run_simulation.py --population 100000 --fit --seed 12345

# Web UI (browser interface with interactive charts)
python web_app.py
# → Open http://127.0.0.1:5000
```

## What the Model Does

- Simulates a virtual population of up to 1 million people
- Each person can develop cancer, be diagnosed, receive treatment, and die (from cancer or natural causes)
- Screening can be simulated: regular testing detects cancers earlier, improving survival
- Compares outcomes with/without screening to estimate lives saved, years of life gained, and false positive rates

## Project Structure

```
MedicalModelPython/
├── run_simulation.py         # CLI entry point
├── web_app.py                # Flask web UI
├── model/                    # Core simulation engine
│   ├── simulation.py         # Main loop + orchestration
│   ├── population.py         # Population as Structure-of-Arrays + numba kernels
│   ├── parameters.py         # Parameter loading (TOML) + data loading
│   ├── hazard.py             # Hazard functions (Gompertz, Exp, LogLogistic, Piecewise)
│   ├── gompertz.py           # Reduced Gompertz tumour growth model
│   ├── logistic_regression.py # Multinomial stage-by-age regression
│   ├── distribution.py       # Empirical CDF/PDF distributions
│   ├── random.py             # Reproducible RNG with seed tracking
│   ├── stats.py              # Statistics collection and aggregation
│   └── sensitivity.py        # Lead-time sensitivity analysis
├── optimization/             # Parameter calibration
│   ├── objective_diag.py     # Diagnosis hazard (Poisson MLE, L-BFGS-B)
│   ├── objective_gompertz.py # Gompertz growth (cross-entropy, Nelder-Mead)
│   └── objective_mort.py     # Cancer death hazard (Poisson MLE, L-BFGS-B)
├── config/
│   └── parameters.toml       # Model configuration
├── data/
│   ├── data_agg_rus.csv      # Aggregate demographic data
│   └── data_ind.csv          # Individual staging data
├── templates/
│   └── index.html            # Web dashboard (Plotly charts)
├── python_port_analysis.tex  # Scientific analysis (English)
├── model_for_dummies.tex     # Plain-language description (English)
└── requirements.txt          # Dependencies
```

## Key Features

- **Structure-of-Arrays (SoA):** Population stored as NumPy arrays — ~10× less memory than object lists
- **Numba JIT:** Cancer history logic compiled to machine code — ~50× faster than pure Python
- **Vectorized operations:** Batch generation of ages, death ages, and diagnosis ages
- **Web UI:** Interactive parameter editor, save/load calibration, sensitivity analysis with overlaid curves
- **Reproducibility:** Deterministic seeds logged for every run

## Performance

| Operation | Python (vectorized) | Notes |
|-----------|---------------------|-------|
| Generate 1M population | ~2 s | NumPy batch + Numba |
| 1 simulation year | ~0.3 s | Vectorized masks |
| Calibration (Fit) | ~8 s | L-BFGS-B + Nelder-Mead |
| Full run (30 yr × 50K) | ~30 s | Including gather stats |

## Documentation

- `python_port_analysis.tex` — Full scientific analysis with equations, validity critique, and calibration details
- `model_for_dummies.tex` — Plain-language guide (no formulas, for non-specialists)

## License

Academic use. See original C# repository: [Magisterbes/BespalovPhd](https://github.com/Magisterbes/BespalovPhd)