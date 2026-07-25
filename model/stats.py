"""Statistics collection and aggregation — fast version."""
from typing import Dict
import numpy as np


class StatsCollection:
    def __init__(self, years_to_simulate: int, unreal_life_length: int = 110, stage_distribution_length: int = 4):
        self.years_to_simulate = years_to_simulate
        self.unreal_life_length = unreal_life_length
        self.stage_distribution_length = stage_distribution_length
        sz = unreal_life_length + 1
        self.age_distributions = {y: np.zeros(sz, dtype=np.int32) for y in range(years_to_simulate)}
        self.cancer_mortality = {y: np.zeros(sz, dtype=np.int32) for y in range(years_to_simulate)}
        self.cancer_screening_mortality = {y: np.zeros(sz, dtype=np.int32) for y in range(years_to_simulate)}
        self.at_risk = {y: np.zeros(sz, dtype=np.int32) for y in range(years_to_simulate)}
        self.incidence = {y: np.zeros(sz, dtype=np.int32) for y in range(years_to_simulate)}
        self.diagnosis = {y: np.zeros(sz, dtype=np.int32) for y in range(years_to_simulate)}
        self.agg_stats = {
            'diagnose_stages_distribution': np.zeros(4, dtype=np.float64),
            'screening_stages_distribution': np.zeros(4, dtype=np.float64),
            'people_saved': np.zeros(sz, dtype=np.float64),
            'years_saved': np.zeros(sz, dtype=np.float64),
            'false_positives': np.zeros(years_to_simulate + 1, dtype=np.float64),
            'mortality_rates': np.zeros(100, dtype=np.float64),
            'screened_mortality_rates': np.zeros(100, dtype=np.float64),
            'incidence_rates': np.zeros(100, dtype=np.float64),
            'survival': np.zeros(100, dtype=np.float64),
            'survival_screening': np.zeros(100, dtype=np.float64),
            'cancer_mortality_age': np.zeros(100, dtype=np.float64),
            'cancer_mortality_age_screen': np.zeros(100, dtype=np.float64),
            'after_diagnosis': np.zeros(200, dtype=np.float64),
            'diagnosis_rates': np.zeros(100, dtype=np.float64),
        }

    def gather_stats(self, population, params, current_date):
        pop = population
        final_ages = pop.ages(current_date)
        n = pop.n_agents
        has_c = pop.has_cancer
        d_age = pop.diagnosis_age
        n_death = pop.natural_death_age
        db = pop.date_birth
        cur = current_date
        agg = self.agg_stats

        # Diagnose stage distribution
        mask = (d_age != -1) & (d_age < n_death) & (d_age + db <= cur) & (pop.cancer_diagnose_stage != -1)
        for idx in np.where(mask)[0]:
            st = pop.cancer_diagnose_stage[idx] - 1
            if 0 <= st < 4:
                agg['diagnose_stages_distribution'][st] += 1

        # After-diagnosis & screening stages
        for idx in np.where(has_c)[0]:
            diff = pop.cancer_stages_ages[idx, 3] - d_age[idx]
            diff = min(max(int(diff), 0), 199)
            agg['after_diagnosis'][diff] += 1
            if pop.cancer_screening_stage[idx] != -1:
                st = pop.cancer_screening_stage[idx] - 1
                if 0 <= st < 4:
                    agg['screening_stages_distribution'][st] += 1

        # Survival curves (vectorized)
        _calc_survival_batch(pop, final_ages, agg)

        # People saved / years saved
        for idx in range(n):
            if not has_c[idx]:
                continue
            # Death cause logic inline
            init_d = pop.cancer_death_age_init[idx]
            nat_d = n_death[idx]
            cured = pop.cancer_is_cured[idx]
            scr_cured = pop.cancer_is_screening_cured[idx]
            alive = pop.is_alive[idx]

            if nat_d <= init_d:
                if cured:
                    pass  # NaturalCured
                elif scr_cured and not cured:
                    # NaturalSavedByScreening
                    inc = int(pop.cancer_incidence_age[idx])
                    if 0 <= inc < len(agg['people_saved']):
                        agg['people_saved'][inc] += 1
                    for t in range(int(pop.cancer_death_age_screen[idx]), int(nat_d)):
                        if 0 <= t < len(agg['years_saved']):
                            agg['years_saved'][t] += 1
            else:
                if not cured and not scr_cured:
                    pass  # Cancer death
                elif cured:
                    pass  # NaturalCured
                elif scr_cured:
                    # Case 2: still alive, screening cured
                    if alive and not cured and scr_cured:
                        inc = int(pop.cancer_incidence_age[idx])
                        if pop.cancer_screening_age[idx] <= final_ages[idx] and init_d <= final_ages[idx]:
                            if 0 <= inc < len(agg['people_saved']):
                                agg['people_saved'][inc] += 1
                            yrs = pop.cancer_death_age_screen[idx] - init_d
                            if 0 <= inc < len(agg['years_saved']):
                                agg['years_saved'][inc] += float(max(yrs, 0))

        self._gather_calc()

    def _gather_calc(self):
        agg = self.agg_stats
        agg['mortality_rates'] = _avg_stats(self.cancer_mortality, self.at_risk)
        agg['screened_mortality_rates'] = _avg_stats(self.cancer_screening_mortality, self.at_risk)
        agg['incidence_rates'] = _avg_stats(self.incidence, self.age_distributions)
        agg['diagnosis_rates'] = _avg_stats(self.diagnosis, self.age_distributions)
        agg['survival_screening'] = _cause_surv(agg['survival_screening'], agg['cancer_mortality_age_screen'])
        agg['survival'] = _cause_surv(agg['survival'], agg['cancer_mortality_age'])

    def to_dict(self) -> dict:
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.agg_stats.items()}


def _calc_survival_batch(pop, final_ages, agg):
    has_c = pop.has_cancer
    d_age = pop.diagnosis_age
    n_death = pop.natural_death_age
    scr_found = pop.cancer_screening_found
    scr_age = pop.cancer_screening_age
    scr_cured = pop.cancer_is_screening_cured
    init_death = pop.cancer_death_age_init
    scr_death = pop.cancer_death_age_screen

    for idx in range(pop.n_agents):
        if not has_c[idx]:
            continue
        if d_age[idx] > final_ages[idx] or d_age[idx] == -1:
            continue

        # Effective death age with screening
        if scr_found[idx] and scr_cured[idx]:
            eff_death_screen = scr_death[idx]
            eff_start = scr_age[idx]
        else:
            eff_death_screen = init_death[idx]
            eff_start = d_age[idx]
        
        min_no_screen = min(float(n_death[idx]), float(final_ages[idx]),
                            init_death[idx] if init_death[idx] != -1 else 1e9)
        min_screen = min(float(n_death[idx]), float(final_ages[idx]),
                         eff_death_screen if eff_death_screen != -1 else 1e9)

        _itter_surv(eff_start, int(min_screen), agg, 'survival_screening')
        _itter_surv(d_age[idx], int(min_no_screen), agg, 'survival')

        # Cancer mortality age (time from diagnosis to death)
        if init_death[idx] != -1 and init_death[idx] <= n_death[idx]:
            diff = int(init_death[idx] - d_age[idx])
            if 0 <= diff < len(agg['cancer_mortality_age']):
                agg['cancer_mortality_age'][diff] += 1
        # Screening-affected mortality age
        if eff_death_screen != -1 and eff_death_screen <= n_death[idx]:
            diff = int(eff_death_screen - eff_start)
            if 0 <= diff < len(agg['cancer_mortality_age_screen']):
                agg['cancer_mortality_age_screen'][diff] += 1


def _itter_surv(beg, fin, agg, key):
    if beg < 0: beg = 0
    if fin >= len(agg[key]): fin = len(agg[key]) - 1
    if fin < beg: return
    agg[key][0] += 1
    for t in range(fin - beg):
        idx = t + 1
        if idx < len(agg[key]):
            agg[key][idx] += 1.0


def _avg_stats(vals, at_risk, max_age=100):
    cases = np.zeros(max_age, dtype=np.float64)
    pop_total = np.zeros(max_age, dtype=np.float64)
    for year in range(1, len(vals)):
        for age in range(max_age):
            if age < len(vals[year]):
                cases[age] += vals[year][age]
            if age < len(at_risk[year]):
                pop_total[age] += at_risk[year][age]
    rates = np.zeros(max_age, dtype=np.float64)
    for age in range(max_age):
        if pop_total[age] > 0:
            rates[age] = cases[age] / pop_total[age]
    return rates


def _cause_surv(alive, dead):
    n = len(alive)
    S = np.zeros(n)
    H = 0.0
    for i in range(min(n - 1, len(dead))):
        if alive[i] > 0:
            H += dead[i] / alive[i]
        S[i] = np.exp(-H)
    return S