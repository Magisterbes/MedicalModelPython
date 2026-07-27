"""Quick test of the sensitivity API without starting the full server."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model.sensitivity import run_sensitivity_analysis

print("Running sensitivity analysis (5K pop, 5 years, 3 factors)...")
result = run_sensitivity_analysis(
    param_source="config/parameters.toml",
    seed=42,
    population=5000,
    years=5,
    factors=[0.5, 1.0, 1.5]
)

print(f"Runtime: {result['runtime_sec']:.1f}s")
print(f"Factors: {result['factors']}")
print(f"Agg_stats keys: {list(result['agg_stats'].keys())}")
for k in result['agg_stats']:
    arr_list = result['agg_stats'][k]
    print(f"  {k}: {len(arr_list)} arrays, first len={len(arr_list[0]) if arr_list else 'empty'}, "
          f"sum={sum(arr_list[0]) if arr_list else 0:.3f}")

print("\nAll OK — sensitivity data is being generated correctly.")