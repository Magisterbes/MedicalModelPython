"""Build the default aggregate dataset from GLOBOCAN 2022 (IARC / Global Cancer Observatory).

Dataset: colorectal cancer (colon C18 + rectum C19-20), United States of
America, both sexes, GLOBOCAN 2022 (version 1.1).

Source: the Global Cancer Observatory gateway that powers
https://gco.iarc.who.int/today . The numbers are fetched ONCE and committed to
the repository, so the application never depends on the API at runtime; this
script only exists so the dataset can be reproduced.

    Ferlay J, Ervik M, Lam F, Laversanne M, Colombet M, Mery L, Piñeros M,
    Znaor A, Soerjomataram I, Bray F (2024). Global Cancer Observatory:
    Cancer Today (version 1.1). Lyon, France: International Agency for
    Research on Cancer. https://gco.iarc.who.int/today

Outputs (delimiter ';', one row per single year of age 0..110):

    Age;population;cases;deaths cancer;deaths all

'deaths all' is not published by GLOBOCAN; it is a Makeham-Gompertz all-cause
mortality pattern calibrated to a US life expectancy of ~77.5 years (the model
only uses its shape).

Usage:  python tools/build_globocan_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

API = 'https://gco-api.iarc.fr/api/globocan/v3/2022'
CANCERS = (8, 9)         # 8 = colon (C18), 9 = rectum (C19-20)
SEX = 0                  # both sexes
YEAR = 2022
N_GROUPS = 18            # 0-4, 5-9, ... 85+
MAX_AGE = 110
STAGE_GROUPS = 17        # 17 five-year bands cover ages 0..84
LIFE_EXPECTANCY = 77.5   # default target e0 for the modelled all-cause mortality

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

# Set from the command line in main(); keeping them module level avoids extra
# plumbing through every fetch helper.
COUNTRY = 840
COUNTRY_LABEL = 'United States of America'
OUT_AGG = os.path.join(DATA_DIR, 'globocan_colorectum_usa_agg.csv')
OUT_META = os.path.join(DATA_DIR, 'globocan_colorectum_usa_agg.meta.json')


def _get(url: str) -> dict:
    """GET a JSON document from the GCO gateway."""
    request = urllib.request.Request(url, headers={'User-Agent': 'MedicalModel2024-dataset-builder'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode('utf-8'))


def fetch_cumulative(kind: int, upto: int) -> int:
    """Cumulative total for age bands 0..upto (``ages_group`` is inclusive)."""
    cancer_path = '_'.join(str(c) for c in CANCERS)
    url = f'{API}/data/rate/{kind}/{SEX}/{COUNTRY}/{cancer_path}/?ages_group=0_{upto}'
    payload = _get(url)
    return int(sum(int(row['total']) for row in payload.get('dataset', [])))


def fetch_band_totals(kind: int) -> list[int]:
    """Per-band totals for the 18 five-year bands (0-4 ... 85+).

    The gateway only accepts ``ages_group`` ranges that start at band 0, so the
    individual bands are recovered as differences of cumulative totals
    (``0_13`` minus ``0_12`` and so on). Summed back they reproduce the published
    national totals exactly.
    """
    cumulative = [fetch_cumulative(kind, k) for k in range(N_GROUPS)]
    bands = [cumulative[0]]
    for k in range(1, N_GROUPS):
        bands.append(cumulative[k] - cumulative[k - 1])
    return bands


def fetch_all() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (population, cases, deaths) as arrays over the 18 five-year bands."""
    population = fetch_population()
    cases = fetch_band_totals(0)
    deaths = fetch_band_totals(1)
    return (np.array(population, dtype=np.float64),
            np.array(cases, dtype=np.float64),
            np.array(deaths, dtype=np.float64))


def fetch_population() -> list[int]:
    """Population of the country in YEAR, per five-year age band (index 0 = 0-4)."""
    url = (f'{API}/data/population/0/{SEX}/{COUNTRY}/{CANCERS[0]}/'
           f'?ages_group=0_{N_GROUPS - 1}')
    payload = _get(url)
    for row in payload.get('dataset', []):
        if int(row.get('year', 0)) == YEAR:
            return [int(row['age'][str(i + 1)]) for i in range(N_GROUPS)]
    raise RuntimeError(f'No population data for {YEAR} in the API response')


def makeham_gompertz(ages: np.ndarray, e0_target: float = LIFE_EXPECTANCY,
                     a_level: float = 0.0005, c_slope: float = 0.090):
    """All-cause mortality hazard mu(a) = A + B*exp(c*a) with e0 calibrated.

    B is solved by bisection so the implied life expectancy equals
    ``e0_target``. Returns (mu, survival) as arrays over ``ages``.
    """
    ages = np.asarray(ages, dtype=np.float64)

    def survival_for(b_scale: float) -> np.ndarray:
        mu_local = a_level + b_scale * np.exp(c_slope * ages)
        return np.concatenate(([1.0], np.cumprod(np.exp(-mu_local[:-1]))))

    def life_expectancy(b_scale: float) -> float:
        return float(survival_for(b_scale).sum() - 0.5)

    lo, hi = 1e-8, 1e-1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if life_expectancy(mid) > e0_target:
            lo = mid
        else:
            hi = mid
    b_scale = 0.5 * (lo + hi)
    mu = a_level + b_scale * np.exp(c_slope * ages)
    return mu, survival_for(b_scale)


def expand_band_rates(band_rates: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Turn per-band rates into per-single-year rates.

    Within a band the rate is constant. GLOBOCAN publishes 85+ as one open band,
    so that same rate is applied to every age from 85 to MAX_AGE — the data is
    never extrapolated beyond what the source actually reports.
    """
    per_year = np.zeros(MAX_AGE + 1, dtype=np.float64)
    for band in range(STAGE_GROUPS):
        start = band * 5
        per_year[start:start + 5] = band_rates[band]
    per_year[85:] = band_rates[N_GROUPS - 1]
    return per_year


def write_aggregate(path: str, ages: np.ndarray, population: np.ndarray,
                    cases: np.ndarray, deaths_cancer: np.ndarray,
                    deaths_all: np.ndarray) -> None:
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('Age;population;cases;deaths cancer;deaths all\n')
        for i, age in enumerate(ages):
            handle.write('%d;%.4f;%.4f;%.4f;%.4f\n'
                         % (age, population[i], cases[i], deaths_cancer[i], deaths_all[i]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Build an aggregate dataset (Age;population;cases;deaths cancer;deaths all) '
                    'from GLOBOCAN 2022 for colorectal cancer.')
    parser.add_argument('--country', type=int, default=840,
                        help='GLOBOCAN country code (ISO numeric): 840 USA, 276 Germany, '
                             '392 Japan, ...')
    parser.add_argument('--country-name', default='United States of America',
                        help='country label stored in the metadata')
    parser.add_argument('--out-prefix', default='globocan_colorectum_usa',
                        help='output file prefix inside data/')
    parser.add_argument('--life-expectancy', type=float, default=LIFE_EXPECTANCY,
                        help='target life expectancy for the modelled all-cause mortality')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    global COUNTRY, COUNTRY_LABEL, OUT_AGG, OUT_META
    args = parse_args(argv)
    COUNTRY = args.country
    COUNTRY_LABEL = args.country_name
    OUT_AGG = os.path.join(DATA_DIR, f'{args.out_prefix}_agg.csv')
    OUT_META = os.path.join(DATA_DIR, f'{args.out_prefix}_agg.meta.json')

    print(f'Fetching GLOBOCAN 2022 - colorectum, {COUNTRY_LABEL} ({COUNTRY}) ...')
    pop_bands, case_bands, death_bands = fetch_all()
    ages = np.arange(0, MAX_AGE + 1, dtype=np.float64)

    total_cases = int(case_bands.sum())
    total_deaths = int(death_bands.sum())
    total_pop = int(pop_bands.sum())
    print(f'  totals: {total_cases:,} cases | {total_deaths:,} deaths | '
          f'population {total_pop:,}')

    # Population as counts per single year; the open 85+ band is split with the
    # modelled survival curve so the very old are progressively rarer.
    mu, survival = makeham_gompertz(ages, e0_target=args.life_expectancy)
    population = expand_band_rates(pop_bands / 5.0, ages)
    tail = survival[85:] / survival[85:].sum()
    population[85:] = pop_bands[N_GROUPS - 1] * tail

    cases_rate = case_bands / np.maximum(pop_bands, 1.0)
    deaths_rate = death_bands / np.maximum(pop_bands, 1.0)
    cases = expand_band_rates(cases_rate, ages) * population
    deaths_cancer = expand_band_rates(deaths_rate, ages) * population
    deaths_all = mu * population

    os.makedirs(os.path.dirname(OUT_AGG), exist_ok=True)
    write_aggregate(OUT_AGG, ages, population, cases, deaths_cancer, deaths_all)

    meta = {
        'source': 'GLOBOCAN 2022 (version 1.1), IARC / Global Cancer Observatory',
        'url': 'https://gco.iarc.who.int/today',
        'citation': ('Ferlay J, Ervik M, Lam F, Laversanne M, Colombet M, Mery L, '
                     'Pineros M, Znaor A, Soerjomataram I, Bray F (2024). Global Cancer '
                     'Observatory: Cancer Today (version 1.1). Lyon, France: International '
                     'Agency for Research on Cancer. https://gco.iarc.who.int/today'),
        'site': 'Colorectum (colon C18 + rectum C19-20)',
        'country': f'{COUNTRY_LABEL} ({COUNTRY})',
        'sex': 'both',
        'year': YEAR,
        'totals': {'cases': total_cases, 'deaths': total_deaths, 'population': total_pop},
        'notes': ('Cases and deaths are real GLOBOCAN 2022 figures per five-year age band, '
                  'expanded to single years of age. GLOBOCAN reports 85+ as one open band, '
                  'so that rate is applied to ages 85-110. "deaths all" is not published by '
                  'GLOBOCAN: it is a Makeham-Gompertz all-cause mortality pattern '
                  f'calibrated to a life expectancy of {args.life_expectancy} years.'),
        'delimiter': ';',
    }
    with open(OUT_META, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    crude_cases = total_cases / total_pop * 1e5
    crude_deaths = total_deaths / total_pop * 1e5
    print(f'  written: {OUT_AGG}')
    print(f'  written: {OUT_META}')
    print(f'  implied crude rates (per 100k): cases {crude_cases:.2f}, '
          f'deaths {crude_deaths:.2f}')
    print('  GLOBOCAN reference: colon 31.43 + rectum 13.72 = 45.15 per 100k cases')
    return 0


if __name__ == '__main__':
    sys.exit(main())
