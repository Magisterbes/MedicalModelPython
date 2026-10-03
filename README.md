# MedicalModel2024 Python — Cancer Screening Microsimulation

Agent-based microsimulation for evaluating cancer screening programmes.  
**Language:** Python 3.11+ | **UI:** Flask + Plotly.js | **Backend:** NumPy + Numba + scikit-learn

## Quick Start

```bash
pip install -r requirements.txt

# CLI: run with 50K agents
python run_simulation.py --population 50000 --seed 42

# CLI: run with calibration (Fit + Simulate)
python run_simulation.py --population 100000 --fit --seed 12345

# Web UI (interactive charts, parameter editor, sensitivity analysis)
python web_app.py
# → Open http://127.0.0.1:5000
```

## Docker

```bash
# Build the image
docker build -t med_flask_app:latest .

# Run the web UI → http://localhost:5000
docker run --rm -p 5000:5000 --name flask_tutorial med_flask_app:latest

# …or use Docker Compose (persists output/, config/ and data/ on the host)
docker compose up --build
```

The image runs the Flask UI on port 5000 under a non-root user, ships with a
health check at `/api/status`, pre-warms the Numba JIT cache at build time, and
installs a proper init (`tini`) for clean signal handling.

> **Note:** `requirements.txt` uses lower bounds only, so a rebuilt image may
> resolve newer library versions than your local environment. Results stay
> numerically equivalent, but re-running with a different NumPy/Numba version
> can shift the random stream very slightly (a handful of agents out of
> millions). Pin exact versions if you need bit-for-bit reproducibility.

## What the Model Does

- Simulates a virtual population of up to 1 million people
- Each person can develop cancer, be diagnosed, receive treatment, and die (cancer or natural causes)
- Screening programme simulation: regular testing detects cancers earlier, improving survival
- **Sensitivity analysis:** quantifies how lead-time assumptions affect outcomes
- Compares outcomes with/without screening: lives saved, years of life gained, stage distributions

## Project Structure

```
MedicalModelPython/
├── run_simulation.py           # CLI entry point
├── web_app.py                  # Flask web server with REST API
├── model/                      # Core simulation engine
│   ├── simulation.py           # Main loop, annual iteration, orchestration
│   ├── population.py           # Population as Structure-of-Arrays + Numba JIT kernels
│   ├── parameters.py           # TOML config parser, data loading, sklearn logistic regression
│   ├── hazard.py               # PiecewiseHazard (diagnosis), ExpHazard (cancer death)
│   ├── gompertz.py             # Reduced Gompertz tumour growth model
│   ├── distribution.py         # Empirical CDF/PDF distributions (vectorized)
│   ├── random.py               # Reproducible RNG with seed tracking
│   ├── stats.py                # Statistics collection, rates, survival curves
│   └── sensitivity.py          # Lead-time sensitivity + Savitzky-Golay smoothing
├── optimization/               # Parameter calibration
│   ├── objective_diag.py       # Diagnosis hazard (Poisson MLE, L-BFGS-B)
│   ├── objective_gompertz.py   # Gompertz growth (cross-entropy, Nelder-Mead)
│   └── objective_mort.py       # Cancer death hazard (Poisson MLE, L-BFGS-B)
├── config/
│   └── parameters.toml         # Model configuration
├── data/                       # CSV data files (can be replaced via UI)
│   ├── data_agg_rus.csv        # Aggregate demographic + incidence + mortality
│   └── data_ind.csv            # Individual staging records (age, stage, aggressiveness)
├── templates/
│   └── index.html              # Web dashboard with Plotly charts + sensitivity section
├── python_port_analysis.tex    # Scientific analysis with equations and validity critique
├── model_for_dummies.tex       # Plain-language guide (no formulas)
├── test_fit_speed.py           # Benchmark for calibration speed
└── requirements.txt            # Python dependencies
```

## Key Features

- **Structure-of-Arrays (SoA):** Population stored as NumPy arrays — ~10× less memory than object lists
- **Numba JIT:** Cancer history logic and screening compiled to machine code — ~50× faster
- **Vectorized operations:** Batch generation of ages, death ages, and diagnosis ages
- **Web UI:** Interactive parameter editor, CSV upload with validation, fit/simulate/save/load, sensitivity analysis
- **Sensitivity analysis:** Lead-time perturbation (×0.5…×1.5) with dedicated chart section showing:
  - Overlaid incidence, mortality, and survival curves (smoothed via Savitzky-Golay filter)
  - Filled area curves for years saved by lead-time factor
  - Stage distribution at baseline
  - Tornado bars and numeric table for aggregate metrics
- **Data file management:** Upload and validate new CSV datasets through the UI; switch data sources on-the-fly
- **Reproducibility:** Deterministic seeds logged for every run

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Generate 1M population | ~1.7 s | NumPy batch + Numba |
| 1 simulation year (1M agents) | ~0.05 s | Vectorized `np.bincount` statistics |
| Calibration (Fit) | ~0.2 s | Vectorized Gompertz + L-BFGS-B / Nelder-Mead |
| Sensitivity (5 factors × 30yr × 100K) | ~2 s | Factors run in parallel across cores for heavy runs |
| Full run (30 yr × 50K) | ~3 s | Including start-up and gather stats |

## Documentation

- `python_port_analysis.tex` — Full scientific analysis with equations, identifiability critique, and calibration details
- `model_for_dummies.tex` — Plain-language description for non-specialists

## License

Academic use. See original C# repository: [Magisterbes/BespalovPhd](https://github.com/Magisterbes/BespalovPhd)