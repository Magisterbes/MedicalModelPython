"""Flask web UI for MedicalModel2024 microsimulation."""
import json, os, sys, threading, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from flask import Flask, render_template, request, jsonify
from model.simulation import Simulation
from model.random import get_random

app = Flask(__name__)
_sim_instance = None
_sim_thread = None
_sim_result = None
_sim_running = False
_sim_progress = 0
_params_edit_cache = None  # Store edited params in-memory

FITTED_PARAMS_FILE = "config/params_fitted.json"
DATA_DIR = "data"
ALLOWED_EXTENSIONS = {'csv'}

import pandas as pd

def _validate_aggregate_csv(filepath: str) -> tuple[bool, str]:
    """Validate aggregate CSV has required columns and valid data."""
    try:
        df = pd.read_csv(filepath, sep=';')
    except Exception as e:
        return False, f"Failed to parse CSV: {e}"
    required = ['Age', 'population', 'cases', 'deaths cancer', 'deaths all']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}. Required: {', '.join(required)}"
    # Validate numeric, non-negative
    for col in required[1:]:  # skip Age
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric"
        if (df[col] < 0).any():
            return False, f"Column '{col}' contains negative values"
    # Age must be 0-110
    if not pd.api.types.is_numeric_dtype(df['Age']):
        return False, "Column 'Age' must be numeric"
    if (df['Age'] < 0).any() or (df['Age'] > 110).any():
        return False, "Column 'Age' values must be between 0 and 110"
    return True, f"OK ({len(df)} rows)"

def _validate_staging_csv(filepath: str) -> tuple[bool, str]:
    """Validate staging CSV has required columns and valid data."""
    try:
        df = pd.read_csv(filepath, sep=';')
    except Exception as e:
        return False, f"Failed to parse CSV: {e}"
    required = ['Age', 'Stage', 'Aggressiveness']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}. Required: {', '.join(required)}"
    # Stage must be 1-4
    if not pd.api.types.is_numeric_dtype(df['Stage']):
        return False, "Column 'Stage' must be numeric"
    invalid_stage = df[~df['Stage'].isin([1, 2, 3, 4])]
    if len(invalid_stage) > 0:
        return False, f"Column 'Stage' must be 1-4. Invalid rows: {len(invalid_stage)}"
    # Aggressiveness must be 0 or 1
    invalid_agg = df[~df['Aggressiveness'].isin([0, 1])]
    if len(invalid_agg) > 0:
        return False, f"Column 'Aggressiveness' must be 0 or 1. Invalid rows: {len(invalid_agg)}"
    if (df['Age'] < 0).any() or (df['Age'] > 110).any():
        return False, "Column 'Age' values must be between 0 and 110"
    return True, f"OK ({len(df)} rows)"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    global _sim_thread, _sim_running, _sim_result, _sim_instance, _sim_progress
    if _sim_running:
        return jsonify({'error': 'Simulation already running'}), 409
    data = request.get_json() or {}
    config = data.get('config', 'config/parameters.toml')
    seed = data.get('seed', None)
    population = data.get('population', None)
    do_fit = data.get('fit', False)
    override_params = data.get('override_params', None)
    
    _sim_running = True
    _sim_progress = 0
    _sim_result = None
    
    def run_sim():
        global _sim_instance, _sim_result, _sim_progress, _sim_running
        try:
            _sim_instance = Simulation(param_source=config, seed=seed)
            p = _sim_instance.params
            if population is not None:
                p.init_population = int(population)
            # Apply overrides from UI
            if override_params:
                _apply_overrides(p, override_params)
            if do_fit:
                from run_simulation import _run_fitting
                _run_fitting(_sim_instance)
                _sim_progress = 5
            _sim_instance.start()
            _sim_progress = 5
            years = p.years_to_simulate
            for year in range(years - 1):
                _sim_instance.iterate_year()
                _sim_progress = 5 + int(90 * (year + 1) / years)
            _sim_progress = 92
            _sim_instance.stats.gather_stats(_sim_instance.population, p, _sim_instance.current_date)
            _sim_progress = 98
            _sim_result = {'summary': _sim_instance.get_summary(),
                           'stats': _sim_instance.stats.to_dict()}
            _sim_progress = 100
        except Exception as e:
            import traceback
            _sim_result = {'error': f'{type(e).__name__}: {e}\n{traceback.format_exc()}'}
            _sim_progress = -1
        finally:
            _sim_running = False
    
    _sim_thread = threading.Thread(target=run_sim, daemon=True)
    _sim_thread.start()
    return jsonify({'status': 'started'})

@app.route('/api/status')
def api_status():
    return jsonify({'running': _sim_running, 'progress': _sim_progress,
                    'has_results': _sim_result is not None})

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

@app.route('/api/results')
def api_results():
    if _sim_result is None:
        return jsonify({'error': 'No results available'}), 404
    return app.response_class(response=json.dumps(dict(_sim_result), cls=NumpyEncoder),
                              status=200, mimetype='application/json')

@app.route('/api/parameters')
def api_parameters():
    """Get current model parameters."""
    p = _sim_instance.params if _sim_instance else None
    if p is None:
        return jsonify({'error': 'No simulation initialized'}), 404
    return jsonify({
        'diagnose_hazard_constants': (p.diagnose_hazard.constants.tolist() 
                                       if hasattr(p.diagnose_hazard, 'constants') else []),
        'cancer_death_hazard_lambda': float(p.cancer_death_hazard.constants[0]),
        'treatment_efficiency': p.treatment_efficiency,
        'age_cure_constants': p.age_cure_constants,
        'lead_time_means': p.lead_time_by_stage_means,
        'test_tp': p.test_tp,
        'test_fp': p.test_fp,
        'selected_test': p.selected_test,
        'participation_rate': p.participation_rate,
        'screening_start_age': p.start_age,
        'screening_finish_age': p.finish_age,
        'screening_frequency': p.frequency,
        'screening_date': p.screening_date,
        'reoccurrence_prob': p.reoccurrence_probability,
        'growth_rate_limits': p.growth_rate_limits,
        'aggressiveness_threshold': p.aggressiveness_rate_threshold,
        'gompertz_aggressive': p.reduced_gompertz_aggressive,
        'gompertz_non_aggressive': p.reduced_gompertz_non_aggressive,
    })

@app.route('/api/sensitivity', methods=['POST'])
def api_sensitivity():
    """Run lead time sensitivity analysis."""
    global _sim_running
    if _sim_running:
        return jsonify({'error': 'Simulation already running'}), 409
    _sim_running = True
    data = request.get_json() or {}
    population = int(data.get('population', 300000))
    years = int(data.get('years', 15))
    seed = data.get('seed', None)
    factors = data.get('factors', [0.5, 0.75, 1.0, 1.25, 1.5])
    n_jobs = data.get('n_jobs', None)  # None = auto (parallel only for heavy runs)
    if n_jobs is not None:
        n_jobs = int(n_jobs)

    def run_sens():
        global _sim_running
        try:
            from model.sensitivity import run_sensitivity_analysis
            result = run_sensitivity_analysis(
                param_source=data.get('config', 'config/parameters.toml'),
                seed=seed, population=population, years=years, factors=factors,
                n_jobs=n_jobs)
            global _sensitivity_result
            _sensitivity_result = result
        except Exception as e:
            import traceback
            _sensitivity_result = {'error': f'{type(e).__name__}: {e}'}
        finally:
            _sim_running = False
    
    _sensitivity_result = None
    import threading
    threading.Thread(target=run_sens, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/api/sensitivity/result')
def api_sensitivity_result():
    global _sensitivity_result
    if _sensitivity_result is None:
        return jsonify({'error': 'No sensitivity results yet. Check /api/status for completion.'}), 404
    result = dict(_sensitivity_result)
    # Convert numpy arrays in metrics to lists
    if 'metrics' in result:
        for k, v in result['metrics'].items():
            if isinstance(v, list):
                result['metrics'][k] = [
                    x.tolist() if isinstance(x, np.ndarray) else x for x in v
                ]
    return app.response_class(
        response=json.dumps(result, cls=NumpyEncoder),
        status=200, mimetype='application/json')

_sensitivity_result = None

@app.route('/api/data/list')
def api_data_list():
    """List available CSV data files."""
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        return jsonify({'files': []})
    csv_files = sorted([f.name for f in data_dir.glob('*.csv')])
    return jsonify({'files': csv_files})

@app.route('/api/data/upload', methods=['POST'])
def api_data_upload():
    """Upload and validate a CSV data file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext != 'csv':
        return jsonify({'error': f'Only .csv files allowed. Got: .{ext}'}), 400
    
    data_type = request.form.get('type', 'aggregate')
    filepath = os.path.join(DATA_DIR, file.filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    file.save(filepath)
    
    # Validate
    if data_type == 'aggregate':
        ok, msg = _validate_aggregate_csv(filepath)
    else:
        ok, msg = _validate_staging_csv(filepath)
    if not ok:
        os.remove(filepath)  # Delete invalid file
        return jsonify({'error': msg}), 400
    
    return jsonify({'status': 'ok', 'filename': file.filename, 'validation': msg})

@app.route('/api/parameters/save', methods=['POST'])
def api_save_params():
    """Save current (or fitted) parameters to a JSON file."""
    p = _sim_instance.params if _sim_instance else None
    if p is None:
        return jsonify({'error': 'No simulation initialized'}), 404
    save_path = request.get_json().get('path', FITTED_PARAMS_FILE) if request.get_json() else FITTED_PARAMS_FILE
    data = {
        'diagnose_hazard_constants': p.diagnose_hazard.constants.tolist(),
        'cancer_death_hazard_lambda': float(p.cancer_death_hazard.constants[0]),
        'treatment_efficiency': p.treatment_efficiency,
        'age_cure_constants': p.age_cure_constants,
        'lead_time_means': p.lead_time_by_stage_means,
        'test_tp_selected': p.test_tp_selected,
        'test_fp_selected': p.test_fp_selected,
        'participation_rate': p.participation_rate,
        'gompertz_aggressive': p.reduced_gompertz_aggressive,
        'gompertz_non_aggressive': p.reduced_gompertz_non_aggressive,
        'growth_rate_limits': p.growth_rate_limits,
        'aggressiveness_threshold': p.aggressiveness_rate_threshold,
        'reoccurrence_prob': p.reoccurrence_probability,
        'screening_start': p.start_age,
        'screening_finish': p.finish_age,
        'screening_frequency': p.frequency,
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    return jsonify({'status': 'ok', 'path': save_path})

@app.route('/api/parameters/load', methods=['POST'])
def api_load_params():
    """Load parameters from a saved JSON file and apply to current simulation."""
    p = _sim_instance.params if _sim_instance else None
    if p is None:
        return jsonify({'error': 'No simulation initialized — run simulation first'}), 404
    load_path = request.get_json().get('path', FITTED_PARAMS_FILE) if request.get_json() else FITTED_PARAMS_FILE
    if not os.path.exists(load_path):
        return jsonify({'error': f'File not found: {load_path}'}), 404
    with open(load_path) as f:
        data = json.load(f)
    _apply_overrides(p, data)
    return jsonify({'status': 'ok', 'path': load_path})

def _apply_overrides(p, data):
    """Apply parameter overrides from a dictionary."""
    if 'diagnose_hazard_constants' in data:
        arr = np.array(data['diagnose_hazard_constants'])
        p.diagnose_hazard.constants = arr
        p.diagnose_hazard.update()
    if 'cancer_death_hazard_lambda' in data:
        p.cancer_death_hazard.constants[0] = float(data['cancer_death_hazard_lambda'])
    if 'treatment_efficiency' in data:
        p.treatment_efficiency = list(data['treatment_efficiency'])
    if 'age_cure_constants' in data:
        p.age_cure_constants = list(data['age_cure_constants'])
    if 'lead_time_means' in data:
        p.lead_time_by_stage_means = list(data['lead_time_means'])
    if 'test_tp_selected' in data:
        p.test_tp_selected = float(data['test_tp_selected'])
    if 'test_fp_selected' in data:
        p.test_fp_selected = float(data['test_fp_selected'])
    if 'participation_rate' in data:
        p.participation_rate = float(data['participation_rate'])
    if 'gompertz_aggressive' in data:
        g = data['gompertz_aggressive']
        p.reduced_gompertz_aggressive = [float(g[0]), float(g[1]), float(g[2]), float(g[3]) if len(g)>3 else 0.1]
        p.gompertz_aggressive.K = p.reduced_gompertz_aggressive[0]
        p.gompertz_aggressive.C = p.reduced_gompertz_aggressive[1]
        p.gompertz_aggressive.B_pop = p.reduced_gompertz_aggressive[2]
    if 'gompertz_non_aggressive' in data:
        g = data['gompertz_non_aggressive']
        p.reduced_gompertz_non_aggressive = [float(g[0]), float(g[1]), float(g[2]), float(g[3]) if len(g)>3 else 0.1]
        p.gompertz_non_aggressive.K = p.reduced_gompertz_non_aggressive[0]
        p.gompertz_non_aggressive.C = p.reduced_gompertz_non_aggressive[1]
        p.gompertz_non_aggressive.B_pop = p.reduced_gompertz_non_aggressive[2]
    if 'growth_rate_limits' in data:
        p.growth_rate_limits = list(data['growth_rate_limits'])
    if 'aggressiveness_threshold' in data:
        p.aggressiveness_rate_threshold = float(data['aggressiveness_threshold'])
    if 'reoccurrence_prob' in data:
        p.reoccurrence_probability = float(data['reoccurrence_prob'])
    if 'screening_start' in data:
        p.start_age = int(data['screening_start'])
    if 'screening_finish' in data:
        p.finish_age = int(data['screening_finish'])
    if 'screening_frequency' in data:
        p.frequency = int(data['screening_frequency'])
    if 'train_data_filename' in data or 'train_staging_data_filename' in data:
        p.reinit_with_data_files(
            train_file=data.get('train_data_filename'),
            staging_file=data.get('train_staging_data_filename'),
        )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(f"MedicalModel2024 at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()