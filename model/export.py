"""Streaming exports of simulation results to disk.

Everything here writes to files in fixed-size chunks, so a large population is
never materialised as a DataFrame (which would cost several times the size of
the agent arrays — fatal on a 512 MB instance). The files are generated once,
right after a run, and then served from disk on demand.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Dict, List, Optional

import numpy as np

from .population import compute_death_causes_vec

logger = logging.getLogger(__name__)

# Agent CSV column order: (name, format, description)
_INT = '%d'
_F = '%.6g'
AGENT_COLUMNS = [
    ('agent_id', _INT, '0-based agent index'),
    ('date_birth', _INT, 'birth date (negative = born before t0)'),
    ('age_at_end', _INT, 'age at the end of the simulation'),
    ('is_alive', _INT, '1 = alive at the end'),
    ('natural_death_age', _INT, 'age at natural (non-cancer) death'),
    ('death_cause', _INT, '0=none/alive, 1=natural, 2=cancer, 3=saved by screening, 4=natural after cure'),
    ('has_cancer', _INT, '1 = ever develops cancer'),
    ('cancer_incidence_age', _INT, 'age at cancer onset (-1 = no cancer)'),
    ('diagnosis_age', _INT, 'age at clinical diagnosis (110 = never diagnosed)'),
    ('cancer_diagnose_stage', _INT, 'clinical stage I-IV at diagnosis (-1 = none)'),
    ('cancer_is_aggressive', _INT, '1 = aggressive tumour'),
    ('cancer_growth_rate', _F, 'tumour growth rate gamma'),
    ('age_stage1', _INT, 'age when the tumour reached stage 1 (cancer onset)'),
    ('age_stage2', _INT, 'age when the tumour reached stage 2'),
    ('age_stage3', _INT, 'age when the tumour reached stage 3'),
    ('age_stage4', _INT, 'age when the tumour reached stage 4'),
    ('cancer_death_age_init', _F, 'cancer death age without treatment (1000 = n/a)'),
    ('cancer_death_age_cure', _F, 'death age when cured by treatment (1000 = n/a)'),
    ('cancer_death_age_screen', _F, 'death age with screening (1000 = n/a)'),
    ('cancer_is_cured', _INT, '1 = cured by clinical treatment'),
    ('cancer_is_screening_cured', _INT, '1 = cured via screening detection'),
    ('cancer_screening_found', _INT, '1 = detected by screening'),
    ('cancer_screening_age', _INT, 'age at screening detection (-1 = none)'),
    ('cancer_screening_stage', _INT, 'stage at screening detection (-1 = none)'),
    ('cancer_reoccurred', _INT, '1 = cancer reoccurred after cure'),
]


def prepare_export_dir(export_dir: str) -> str:
    """(Re)create the export directory — only the latest run is kept on disk."""
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir, ignore_errors=True)
    os.makedirs(export_dir, exist_ok=True)
    return export_dir


def write_agents_csv(pop, current_date: int, path: str,
                     cancer_only: bool = False, chunk: int = 50_000) -> int:
    """Write the per-agent medical history as a ';'-separated CSV.

    Each agent is one row. The arrays are streamed in chunks, so peak extra
    memory stays at ``chunk * n_columns * 8`` bytes (a few MB) regardless of the
    population size. ``cancer_only`` keeps just the agents that ever developed
    cancer (used for the sensitivity runs).
    """
    n = pop.n_agents
    ages = pop.ages(current_date).astype(np.int64)
    death_cause = compute_death_causes_vec(
        pop.is_alive, ages.astype(np.int32), pop.natural_death_age, pop.has_cancer,
        pop.cancer_death_age_init, pop.cancer_death_age_cure, pop.cancer_death_age_screen,
        pop.cancer_is_cured, pop.cancer_is_screening_cured, pop.cancer_screening_found,
        pop.diagnosis_age, pop.unreal_life_length,
    ).astype(np.int64)

    stages = pop.cancer_stages_ages
    columns = [
        np.arange(n, dtype=np.int64),
        pop.date_birth.astype(np.int64),
        ages,
        pop.is_alive.astype(np.int64),
        pop.natural_death_age.astype(np.int64),
        death_cause,
        pop.has_cancer.astype(np.int64),
        pop.cancer_incidence_age.astype(np.int64),
        pop.diagnosis_age.astype(np.int64),
        pop.cancer_diagnose_stage.astype(np.int64),
        pop.cancer_is_aggressive.astype(np.int64),
        pop.cancer_growth_rate,
        stages[:, 0].astype(np.int64),
        stages[:, 1].astype(np.int64),
        stages[:, 2].astype(np.int64),
        stages[:, 3].astype(np.int64),
        pop.cancer_death_age_init,
        pop.cancer_death_age_cure,
        pop.cancer_death_age_screen,
        pop.cancer_is_cured.astype(np.int64),
        pop.cancer_is_screening_cured.astype(np.int64),
        pop.cancer_screening_found.astype(np.int64),
        pop.cancer_screening_age.astype(np.int64),
        pop.cancer_screening_stage.astype(np.int64),
        pop.cancer_reoccurred.astype(np.int64),
    ]
    fmts = [fmt for _, fmt, _ in AGENT_COLUMNS]
    headers = [name for name, _, _ in AGENT_COLUMNS]

    selection = np.where(pop.has_cancer)[0] if cancer_only else None
    n_rows = int(selection.size) if selection is not None else n

    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write(';'.join(headers) + '\n')
        for start in range(0, n_rows, chunk):
            stop = min(start + chunk, n_rows)
            idx = selection[start:stop] if selection is not None else slice(start, stop)
            block = np.empty((stop - start, len(columns)), dtype=np.float64)
            for j, col in enumerate(columns):
                block[:, j] = col[idx]
            np.savetxt(f, block, delimiter=';', fmt=fmts)

    logger.info("Exported %d agent records to %s", n_rows, path)
    return n_rows


def write_meta(export_dir: str, meta: Dict) -> None:
    """Write run metadata and the column legend next to the CSV files."""
    payload = dict(meta)
    payload['delimiter'] = ';'
    payload['agent_columns'] = [
        {'name': name, 'description': desc} for name, _, desc in AGENT_COLUMNS
    ]
    with open(os.path.join(export_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _series(agg: Dict, key: str) -> np.ndarray:
    value = agg.get(key)
    return np.asarray(value, dtype=np.float64) if value is not None else np.zeros(0)


def _dump_series(export_dir: str, name: str, headers: List[str],
                 columns: List[np.ndarray], written: List[str]) -> None:
    arr = np.column_stack([np.asarray(c, dtype=np.float64) for c in columns])
    np.savetxt(os.path.join(export_dir, name), arr, delimiter=';',
               fmt='%.6g', header=';'.join(headers), comments='')
    written.append(name)


def write_charts_csv(agg_stats: Dict, export_dir: str,
                     survival_by_group: Dict = None,
                     cure_by_group: Dict = None) -> List[str]:
    """Write the single-run chart series as tidy CSVs (one file per chart)."""
    written: List[str] = []

    incidence = _series(agg_stats, 'incidence_rates')
    if incidence.size:
        _dump_series(export_dir, 'chart_incidence.csv', ['age', 'incidence_rate'],
                     [np.arange(incidence.size), incidence], written)

    mortality = _series(agg_stats, 'mortality_rates')
    if mortality.size:
        _dump_series(export_dir, 'chart_mortality.csv',
                     ['age', 'mortality_no_screening', 'mortality_with_screening'],
                     [np.arange(mortality.size), mortality,
                      _series(agg_stats, 'screened_mortality_rates')], written)

    diag = _series(agg_stats, 'diagnose_stages_distribution')
    if diag.size:
        _dump_series(export_dir, 'chart_stages.csv',
                     ['stage', 'clinical_count', 'screening_count'],
                     [np.arange(1, diag.size + 1), diag,
                      _series(agg_stats, 'screening_stages_distribution')], written)

    survival = _series(agg_stats, 'survival')
    if survival.size:
        _dump_series(export_dir, 'chart_survival.csv',
                     ['years_since_diagnosis', 'survival_no_screening', 'survival_with_screening'],
                     [np.arange(survival.size), survival,
                      _series(agg_stats, 'survival_screening')], written)

    years_saved = _series(agg_stats, 'years_saved')
    if years_saved.size:
        _dump_series(export_dir, 'chart_years_saved.csv', ['age', 'years_saved'],
                     [np.arange(years_saved.size), years_saved], written)

    diagnosis = _series(agg_stats, 'diagnosis_rates')
    if diagnosis.size:
        _dump_series(export_dir, 'chart_diagnosis_rates.csv', ['age', 'diagnosis_rate'],
                     [np.arange(diagnosis.size), diagnosis], written)

    if survival_by_group:
        keys = [k for k in ('all_agg', 'all_nonagg',
                            's1_agg', 's1_nonagg', 's2_agg', 's2_nonagg',
                            's3_agg', 's3_nonagg', 's4_agg', 's4_nonagg')
                if survival_by_group.get(k)]
        if keys:
            n = len(next(iter(survival_by_group.values())))
            _dump_series(export_dir, 'chart_survival_by_aggressiveness.csv',
                         ['years_since_diagnosis'] + keys,
                         [np.arange(n)] + [np.asarray(survival_by_group[k], dtype=np.float64)
                                           for k in keys], written)

    if cure_by_group:
        rows = []
        for group in ('all', 's1', 's2', 's3', 's4'):
            row = cure_by_group.get(group)
            if row:
                rows.append([group,
                             '%.6g' % row.get('agg', float('nan')),
                             '%.6g' % row.get('nonagg', float('nan'))])
        if rows:
            _write_table(os.path.join(export_dir, 'table_cure_by_group.csv'),
                         ['group', 'aggressive_cure_fraction', 'non_aggressive_cure_fraction'],
                         rows)
            written.append('table_cure_by_group.csv')

    return written


def write_sensitivity_charts_csv(result: Dict, export_dir: str) -> List[str]:
    """Write one CSV per sensitivity chart: first column = x, then one per factor."""
    written: List[str] = []
    factors = result.get('factors') or []
    agg = result.get('agg_stats') or {}
    if not factors:
        return written

    charts = [
        ('incidence_rates', 'sens_chart_incidence.csv', 'age'),
        ('mortality_rates', 'sens_chart_mortality.csv', 'age'),
        ('screened_mortality_rates', 'sens_chart_screened_mortality.csv', 'age'),
        ('survival', 'sens_chart_survival.csv', 'years_since_diagnosis'),
        ('survival_screening', 'sens_chart_survival_screening.csv', 'years_since_diagnosis'),
        ('years_saved', 'sens_chart_years_saved.csv', 'age'),
        ('diagnose_stages_distribution', 'sens_chart_stages.csv', 'stage'),
        ('screening_stages_distribution', 'sens_chart_screening_stages.csv', 'stage'),
    ]
    labels = ['factor_%g' % float(f) for f in factors]
    for key, name, xlabel in charts:
        series = agg.get(key) or []
        if not series:
            continue
        n = max(len(s) for s in series)
        if n == 0:
            continue
        columns = [np.arange(n)] + [np.asarray(s, dtype=np.float64) for s in series]
        _dump_series(export_dir, name, [xlabel] + labels[:len(series)], columns, written)

    return written


def _write_table(path: str, header: List[str], rows: List[List[str]]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(';'.join(header) + '\n')
        for row in rows:
            f.write(';'.join(row) + '\n')


def write_summary_csv(summary: Dict, export_dir: str) -> str:
    """One metric per row for the single run."""
    rows = [[str(k), str(v)] for k, v in summary.items()]
    _write_table(os.path.join(export_dir, 'summary.csv'), ['metric', 'value'], rows)
    return 'summary.csv'


def write_sensitivity_metrics_csv(result: Dict, export_dir: str) -> Optional[str]:
    """Aggregate metrics per sensitivity factor (one metric per row)."""
    metrics = result.get('metrics') or {}
    factors = result.get('factors') or []
    if not metrics or not factors:
        return None
    header = ['metric'] + ['factor_%g' % float(f) for f in factors]
    rows = []
    for name, values in metrics.items():
        row = [str(name)]
        for value in values:
            try:
                row.append('%.6g' % float(value))
            except (TypeError, ValueError):
                row.append('')
        rows.append(row)
    _write_table(os.path.join(export_dir, 'sens_metrics.csv'), header, rows)
    return 'sens_metrics.csv'

