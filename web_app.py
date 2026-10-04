"""Flask web UI for MedicalModel2024 microsimulation."""
import json, logging, os, sys, threading, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from flask import Flask, render_template, request, jsonify, send_file
from model.simulation import Simulation
from model.random import get_random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('web_app')

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

# Reduced Gompertz model V(t) = exp(K - exp(C - B*t)) with B = exp(log(B_pop) + eps)
GOMPERTZ_KEYS = ('K', 'C', 'B_pop', 'B_std')

# Generated download bundle. Rewritten on every run so only the latest one is kept
# on disk; nothing large is held in memory (the files are streamed).
EXPORT_DIR = os.environ.get('EXPORT_DIR', 'output/exports')
_exports: list = []

import pandas as pd

# Human-readable descriptions shown next to each download link
_FILE_DESCRIPTIONS = {
    'agents.csv': 'Per-agent medical history — every agent, including healthy ones',
    'agents_cancer.csv': 'Per-agent medical history — cancer patients only',
    'summary.csv': 'Run summary (one metric per row)',
    'meta.json': 'Run metadata and the column legend',
    'chart_incidence.csv': 'Incidence rate by age',
    'chart_mortality.csv': 'Mortality rate by age, with and without screening',
    'chart_stages.csv': 'Stage distribution at diagnosis',
    'chart_survival.csv': 'Cause-specific survival',
    'chart_years_saved.csv': 'Years of life saved by screening',
    'chart_diagnosis_rates.csv': 'Diagnosis rate by age',
    'sens_metrics.csv': 'Aggregate metrics for every lead-time factor',
    'sens_chart_incidence.csv': 'Incidence curves for every factor',
    'sens_chart_mortality.csv': 'Mortality curves for every factor',
    'sens_chart_screened_mortality.csv': 'Screened-mortality curves for every factor',
    'sens_chart_survival.csv': 'Survival curves for every factor',
    'sens_chart_survival_screening.csv': 'Screened-survival curves for every factor',
    'sens_chart_years_saved.csv': 'Years-saved curves for every factor',
    'sens_chart_stages.csv': 'Clinical stage distribution per factor',
    'sens_chart_screening_stages.csv': 'Screening stage distribution per factor',
}


def _manifest(names):
    """Build the download list from files that actually exist on disk."""
    files = []
    for name in names:
        path = os.path.join(EXPORT_DIR, name)
        if not os.path.exists(path):
            continue
        description = _FILE_DESCRIPTIONS.get(name)
        if description is None and name.startswith('agents_factor_'):
            description = ('Per-agent history for lead-time factor '
                           + name[len('agents_factor_'):-len('.csv')] + ' (cancer patients only)')
        files.append({'name': name, 'size': os.path.getsize(path),
                      'description': description or ''})
    return files


def _export_single_run(sim):
    """Stream the single-run download bundle to disk and return its manifest."""
    from model import export
    export.prepare_export_dir(EXPORT_DIR)
    names = ['agents.csv', 'agents_cancer.csv']
    export.write_agents_csv(sim.population, sim.current_date,
                            os.path.join(EXPORT_DIR, 'agents.csv'))
    export.write_agents_csv(sim.population, sim.current_date,
                            os.path.join(EXPORT_DIR, 'agents_cancer.csv'), cancer_only=True)
    names += export.write_charts_csv(sim.stats.agg_stats, EXPORT_DIR)
    names.append(export.write_summary_csv(sim.get_summary(), EXPORT_DIR))
    export.write_meta(EXPORT_DIR, {
        'kind': 'single_run',
        'population': int(sim.params.init_population),
        'years': int(sim.params.years_to_simulate),
        'seed': get_random().seed,
    })
    names.append('meta.json')
    return _manifest(names)


def _export_sensitivity_result(result):
    """Collect the sensitivity bundle (agent files were written by the jobs)."""
    from model import export
    names = export.write_sensitivity_charts_csv(result, EXPORT_DIR)
    metrics_file = export.write_sensitivity_metrics_csv(result, EXPORT_DIR)
    if metrics_file:
        names.append(metrics_file)
    for factor in result.get('factors', []):
        names.append('agents_factor_%g.csv' % float(factor))
    export.write_meta(EXPORT_DIR, {
        'kind': 'sensitivity',
        'factors': result.get('factors'),
        'runtime_sec': result.get('runtime_sec'),
    })
    names.append('meta.json')
    return _manifest(names)


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
        global _sim_instance, _sim_result, _sim_progress, _sim_running, _exports
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
            # Stream the download bundle to disk while the agent arrays are still
            # available, then release them (they are the bulk of the memory — a
            # 1M-agent run holds ~1.2 GB that nothing reads afterwards).
            try:
                _exports = _export_single_run(_sim_instance)
            except Exception as exc:
                logger.warning("Export failed: %s", exc)
                _exports = []
            _sim_instance.population = None
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


@app.route('/api/downloads')
def api_downloads():
    """List the files generated by the latest run (agents history, chart CSVs)."""
    total = sum(f['size'] for f in _exports)
    return jsonify({'files': _exports, 'ready': bool(_exports), 'total_bytes': total})


@app.route('/api/download/<path:name>')
def api_download(name):
    """Stream a generated file from disk.

    Only files listed in the current manifest can be fetched, which also
    prevents any path traversal.
    """
    entry = next((f for f in _exports if f['name'] == name), None)
    if entry is None:
        return jsonify({'error': f'Unknown file: {name}'}), 404
    path = os.path.join(EXPORT_DIR, entry['name'])
    if not os.path.exists(path):
        return jsonify({'error': 'File is no longer available'}), 404
    return send_file(path, as_attachment=True, download_name=entry['name'])

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
    global _sim_running, _exports
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

    # Prepare the export directory before the jobs start, so the worker processes
    # write into a clean folder (they add one file per factor).
    try:
        from model import export as export_mod
        export_mod.prepare_export_dir(EXPORT_DIR)
    except Exception as exc:
        logger.warning("Could not prepare the export directory: %s", exc)
    _exports = []

    def run_sens():
        global _sim_running, _exports
        try:
            from model.sensitivity import run_sensitivity_analysis
            result = run_sensitivity_analysis(
                param_source=data.get('config', 'config/parameters.toml'),
                seed=seed, population=population, years=years, factors=factors,
                n_jobs=n_jobs, export_dir=EXPORT_DIR)
            global _sensitivity_result
            _sensitivity_result = result
            try:
                _exports = _export_sensitivity_result(result)
            except Exception as exc:
                logger.warning("Sensitivity export failed: %s", exc)
                _exports = []
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
    """List available CSV data files, classified by dataset kind."""
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        return jsonify({'aggregate': [], 'staging': [], 'files': []})
    aggregate, staging = [], []
    for f in sorted(data_dir.glob('*.csv')):
        ok_agg, _ = _validate_aggregate_csv(str(f))
        if ok_agg:
            aggregate.append(f.name)
            continue
        ok_stg, _ = _validate_staging_csv(str(f))
        if ok_stg:
            staging.append(f.name)
    # 'files' kept for backward compatibility (union of both kinds)
    return jsonify({'aggregate': aggregate, 'staging': staging,
                    'files': aggregate + staging})


DATASETS_MANIFEST = os.path.join(DATA_DIR, 'datasets.json')


def _read_meta(data_dir: Path, filename: str) -> dict:
    """Sidecar provenance written by the tools in tools/, if it is present."""
    path = data_dir / (Path(filename).stem + '.meta.json')
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def list_datasets() -> dict:
    """Named datasets from data/datasets.json, filtered to files present on disk.

    An entry is dropped unless *both* of its CSVs exist. That is deliberate: the
    selector sets the aggregate and staging files together, so this is what stops
    a run from silently pairing, say, a US aggregate with German staging.
    Each surviving entry is enriched with its .meta.json provenance.
    """
    try:
        with open(DATASETS_MANIFEST, encoding='utf-8') as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError('manifest is not an object')
    except (OSError, ValueError) as exc:
        logger.warning('Dataset manifest unusable (%s): %s', DATASETS_MANIFEST, exc)
        return {'datasets': [], 'default': None, 'note': ''}

    data_dir = Path(DATA_DIR)
    datasets, default_id = [], None
    for entry in manifest.get('datasets', []):
        aggregate, staging = entry.get('aggregate'), entry.get('staging')
        if not aggregate or not staging:
            logger.warning('Dataset %s: needs both aggregate and staging', entry.get('id'))
            continue
        missing = [f for f in (aggregate, staging) if not (data_dir / f).exists()]
        if missing:
            logger.warning('Dataset %s: skipping, missing %s',
                           entry.get('id'), ', '.join(missing))
            continue
        item = dict(entry)
        item['meta'] = _read_meta(data_dir, aggregate)
        item['staging_meta'] = _read_meta(data_dir, staging)
        item.pop('default', None)          # 'default' belongs to the manifest root
        datasets.append(item)
        if entry.get('default'):
            default_id = entry['id']

    if default_id is None and datasets:
        default_id = datasets[0]['id']
    return {'datasets': datasets, 'default': default_id,
            'note': manifest.get('note', '')}


@app.route('/api/datasets')
def api_datasets():
    """Named default datasets (data/datasets.json), validated against disk."""
    return jsonify(list_datasets())


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
    # Reduced Gompertz models V(t) = exp(K - exp(C - B*t)), B = exp(log(B_pop) + eps).
    # Accept both the 4-element list form and the UI's separate per-variable fields
    # (gompertz_aggressive_K / _C / _B_pop / _B_std).
    for name in ('gompertz_aggressive', 'gompertz_non_aggressive'):
        list_value = data.get(name)
        fields = {k: data.get(f'{name}_{k}') for k in GOMPERTZ_KEYS}
        if list_value is None and all(v is None for v in fields.values()):
            continue
        values = list(getattr(p, 'reduced_' + name))
        if list_value is not None:
            if not isinstance(list_value, (list, tuple)):
                list_value = [list_value]
            values = [float(v) for v in list_value]
        # Pad to exactly 4 (K, C, B_pop, B_std) *before* the per-field overrides:
        # a short list would otherwise drop B_std to a hard-coded fallback — which
        # is exactly the bug this replaced — and could raise IndexError.
        values = (values + [0.1] * 4)[:4]
        for i, key in enumerate(GOMPERTZ_KEYS):
            if fields[key] is not None:
                values[i] = float(fields[key])
        setattr(p, 'reduced_' + name, values)
        model = getattr(p, name)
        model.K, model.C, model.B_pop, model.B_std = values[0], values[1], values[2], values[3]
    if 'growth_rate_limits' in data:
        p.growth_rate_limits = list(data['growth_rate_limits'])
    if 'aggressiveness_threshold' in data:
        p.aggressiveness_rate_threshold = float(data['aggressiveness_threshold'])
    if 'reoccurrence_prob' in data:
        p.reoccurrence_probability = float(data['reoccurrence_prob'])
    # Screening settings — accept both the UI keys and the canonical ones
    start_age = data.get('screening_start', data.get('screening_start_age'))
    if start_age is not None:
        p.start_age = int(start_age)
    finish_age = data.get('screening_finish', data.get('screening_finish_age'))
    if finish_age is not None:
        p.finish_age = int(finish_age)
    if 'screening_frequency' in data:
        p.frequency = int(data['screening_frequency'])
    # Data files — accept both the UI keys and the canonical ones
    train_file = data.get('train_data_filename', data.get('data_aggregate_filename'))
    staging_file = data.get('train_staging_data_filename', data.get('data_staging_filename'))
    if train_file is not None or staging_file is not None:
        p.reinit_with_data_files(
            train_file=train_file,
            staging_file=staging_file,
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