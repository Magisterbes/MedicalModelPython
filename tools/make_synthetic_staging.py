"""Synthesise the staging dataset that accompanies the GLOBOCAN aggregate file.

GLOBOCAN publishes incidence and mortality but **no stage or aggressiveness**
information, so this file is synthetic. It exists only so the model can fit its
stage-by-age and tumour-growth components for the same geography.

Design
------
* Patient ages are drawn from the real GLOBOCAN incidence-by-age profile of the
  aggregate file, so the stage sample is consistent with it.
* The stage mix at diagnosis follows published colorectal figures (localised
  ~39 %, regional ~35 %, distant ~21 %), split into stages I-IV.
* Aggressiveness is drawn per stage — later stages are more often aggressive.

Everything is driven by a fixed seed, so re-running reproduces the file exactly.

Output: data/globocan_colorectum_usa_ind.csv  (Age;Stage;Aggressiveness)

Usage:  python tools/make_synthetic_staging.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
DEFAULT_PREFIX = 'globocan_colorectum_usa'
N_RECORDS = 2000
SEED = 2022

# Stage at diagnosis (I-IV). Localised ~39 % -> I+II, regional ~35 % -> III,
# distant ~21 % -> IV (remainder distributed over the localised stages).
STAGE_MIX = {1: 0.22, 2: 0.25, 3: 0.30, 4: 0.23}

# P(aggressive | stage) — later stages are more often aggressive tumours.
AGGRESSIVE_BY_STAGE = {1: 0.10, 2: 0.20, 3: 0.45, 4: 0.65}


def load_incidence_profile(path: str):
    """Read the aggregate file and return (ages, cases) arrays."""
    ages, cases = [], []
    with open(path, encoding='utf-8') as handle:
        for row in csv.DictReader(handle, delimiter=';'):
            ages.append(int(row['Age']))
            cases.append(float(row['cases']))
    return np.array(ages, dtype=np.int64), np.array(cases, dtype=np.float64)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Synthesise the staging dataset for a GLOBOCAN aggregate file.')
    parser.add_argument('--prefix', default=DEFAULT_PREFIX,
                        help='reads data/<prefix>_agg.csv, writes data/<prefix>_ind.csv')
    parser.add_argument('--records', type=int, default=N_RECORDS,
                        help='number of synthetic patient records')
    parser.add_argument('--seed', type=int, default=SEED, help='random seed')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    in_agg = os.path.join(DATA_DIR, f'{args.prefix}_agg.csv')
    out_ind = os.path.join(DATA_DIR, f'{args.prefix}_ind.csv')
    out_meta = os.path.join(DATA_DIR, f'{args.prefix}_ind.meta.json')

    if not os.path.exists(in_agg):
        print(f'Missing {in_agg} - run tools/build_globocan_dataset.py first.')
        return 1

    ages, cases = load_incidence_profile(in_agg)
    weights = cases / cases.sum()

    rng = np.random.default_rng(args.seed)
    sampled_ages = rng.choice(ages, size=args.records, p=weights)

    stage_ids = np.array(sorted(STAGE_MIX), dtype=np.int64)
    stage_probs = np.array([STAGE_MIX[s] for s in stage_ids], dtype=np.float64)
    stage_probs = stage_probs / stage_probs.sum()
    stages = rng.choice(stage_ids, size=args.records, p=stage_probs)

    aggressive_prob = np.array([AGGRESSIVE_BY_STAGE[s] for s in stages], dtype=np.float64)
    aggressive = (rng.random(args.records) < aggressive_prob).astype(np.int64)

    order = np.lexsort((aggressive, stages, sampled_ages))
    with open(out_ind, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('Age;Stage;Aggressiveness\n')
        for i in order:
            handle.write('%d;%d;%d\n' % (sampled_ages[i], stages[i], aggressive[i]))

    realized_stage = {int(s): float((stages == s).mean()) for s in stage_ids}
    realized_agg = {int(s): float(aggressive[stages == s].mean()) for s in stage_ids}
    meta = {
        'synthetic': True,
        'note': ('Stage and aggressiveness are NOT published by GLOBOCAN. This file is '
                 'synthesised for the accompanying aggregate dataset; do not interpret it '
                 'as registry data.'),
        'aggregate_source': os.path.basename(in_agg),
        'n_records': args.records,
        'seed': args.seed,
        'target_stage_mix': STAGE_MIX,
        'realized_stage_mix': realized_stage,
        'aggressive_probability_by_stage': AGGRESSIVE_BY_STAGE,
        'realized_aggressive_share': realized_agg,
        'age_range': [int(sampled_ages.min()), int(sampled_ages.max())],
        'delimiter': ';',
    }
    with open(out_meta, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    print(f'  written: {out_ind}  ({args.records} records, seed {args.seed})')
    print(f'  written: {out_meta}')
    print(f'  ages {int(sampled_ages.min())}-{int(sampled_ages.max())}, '
          f'median {int(np.median(sampled_ages))}')
    print('  realized stage mix :',
          {k: round(v, 3) for k, v in realized_stage.items()})
    print('  aggressive share   :',
          {k: round(v, 3) for k, v in realized_agg.items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
