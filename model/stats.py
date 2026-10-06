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
        # Cause-specific survival and cure fraction, stratified by stage and
        # aggressiveness (filled in gather_stats).
        self.survival_by_group: Dict = {}
        self.cure_by_group: Dict = {}

    def gather_stats(self, population, params, current_date):
        pop = population
        final_ages = pop.ages(current_date)
        has_c = pop.has_cancer
        d_age = pop.diagnosis_age
        n_death = pop.natural_death_age
        db = pop.date_birth
        cur = current_date
        agg = self.agg_stats

        # Diagnose stage distribution (vectorized via np.bincount)
        mask = (d_age != -1) & (d_age < n_death) & (d_age + db <= cur) & (pop.cancer_diagnose_stage != -1)
        st = pop.cancer_diagnose_stage[mask].astype(np.int64) - 1
        st = st[(st >= 0) & (st < 4)]
        if st.size:
            agg['diagnose_stages_distribution'][:4] += np.bincount(st, minlength=4)[:4]

        idxc = np.where(has_c)[0]

        # After-diagnosis & screening stages (vectorized)
        if idxc.size:
            diff = (pop.cancer_stages_ages[idxc, 3].astype(np.int64)
                    - d_age[idxc].astype(np.int64))
            diff = np.clip(diff, 0, 199)
            agg['after_diagnosis'][:200] += np.bincount(diff, minlength=200)[:200]

            scr_st = pop.cancer_screening_stage[idxc].astype(np.int64) - 1
            scr_st = scr_st[(scr_st >= 0) & (scr_st < 4)]
            if scr_st.size:
                agg['screening_stages_distribution'][:4] += np.bincount(scr_st, minlength=4)[:4]

        # Survival curves (vectorized)
        _calc_survival_batch(pop, final_ages, agg)

        # Cause-specific survival and cure fraction by stage and aggressiveness
        self.survival_by_group, self.cure_by_group = compute_survival_by_group(pop, current_date)

        # People saved / years saved (vectorized over cancer agents)
        if idxc.size:
            init_d = pop.cancer_death_age_init[idxc]
            nat_d = n_death[idxc].astype(np.float64)
            cured = pop.cancer_is_cured[idxc]
            scr_cured = pop.cancer_is_screening_cured[idxc]
            alive = pop.is_alive[idxc]
            inc_c = pop.cancer_incidence_age[idxc].astype(np.int64)
            scr_death = pop.cancer_death_age_screen[idxc]
            scr_age = pop.cancer_screening_age[idxc]
            fin = final_ages[idxc]
            ps = agg['people_saved']
            ys = agg['years_saved']

            # Branch 1: natural death no later than cancer death, rescued by screening
            m1 = (nat_d <= init_d) & scr_cured & (~cured)
            if m1.any():
                inc1 = inc_c[m1]
                v = (inc1 >= 0) & (inc1 < ps.shape[0])
                if v.any():
                    ps[:] += np.bincount(inc1[v], minlength=ps.shape[0])[:ps.shape[0]]
                lo = scr_death[m1].astype(np.int64)
                hi = nat_d[m1].astype(np.int64)
                for k in range(lo.shape[0]):
                    t0 = max(int(lo[k]), 0)
                    t1 = min(int(hi[k]), ys.shape[0])
                    if t1 > t0:
                        ys[t0:t1] += 1.0

            # Branch 2: cancer death later than natural death (still alive), rescued
            m2 = (nat_d > init_d) & scr_cured & (~cured) & alive
            if m2.any():
                within = (scr_age[m2] <= fin[m2]) & (init_d[m2] <= fin[m2])
                sel = np.where(m2)[0][within]
                if sel.size:
                    inc2 = inc_c[sel]
                    v = (inc2 >= 0) & (inc2 < ps.shape[0])
                    if v.any():
                        ps[:] += np.bincount(inc2[v], minlength=ps.shape[0])[:ps.shape[0]]
                    yrs = np.maximum(scr_death[sel] - init_d[sel], 0.0)
                    vy = (inc2 >= 0) & (inc2 < ys.shape[0])
                    if vy.any():
                        ys[:] += np.bincount(inc2[vy], weights=yrs[vy],
                                             minlength=ys.shape[0])[:ys.shape[0]]

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
        d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.agg_stats.items()}
        d['survival_by_group'] = self.survival_by_group
        d['cure_by_group'] = self.cure_by_group
        return d


def _calc_survival_batch(pop, final_ages, agg):
    """Vectorized survival-curve accumulation over diagnosed cancer agents."""
    has_c = pop.has_cancer
    idxc = np.where(has_c)[0]
    if idxc.size == 0:
        return

    d_age_all = pop.diagnosis_age[idxc]
    final_all = final_ages[idxc]
    # Only agents already diagnosed by the end of the run are considered
    keep = (d_age_all != -1) & (d_age_all <= final_all)
    idxc = idxc[keep]
    if idxc.size == 0:
        return

    d_age = pop.diagnosis_age[idxc]
    final = final_ages[idxc].astype(np.float64)
    n_death = pop.natural_death_age[idxc].astype(np.float64)
    scr_found = pop.cancer_screening_found[idxc]
    scr_age = pop.cancer_screening_age[idxc]
    scr_cured = pop.cancer_is_screening_cured[idxc]
    init_death = pop.cancer_death_age_init[idxc]
    scr_death = pop.cancer_death_age_screen[idxc]

    # Effective death age / start with screening
    use_screen = scr_found & scr_cured
    eff_death_screen = np.where(use_screen, scr_death, init_death)
    eff_start = np.where(use_screen, scr_age, d_age)

    init_for_no = np.where(init_death != -1, init_death, 1e9)
    eff_for_screen = np.where(eff_death_screen != -1, eff_death_screen, 1e9)
    base = np.minimum(n_death, final)
    min_no_screen = np.minimum(base, init_for_no).astype(np.int64)
    min_screen = np.minimum(base, eff_for_screen).astype(np.int64)

    _itter_surv_batch(eff_start, min_screen, agg, 'survival_screening')
    _itter_surv_batch(d_age, min_no_screen, agg, 'survival')

    # Cancer mortality age (time from diagnosis to death)
    m = (init_death != -1) & (init_death <= n_death)
    if m.any():
        diff = (init_death[m] - d_age[m]).astype(np.int64)
        L = agg['cancer_mortality_age'].shape[0]
        diff = diff[(diff >= 0) & (diff < L)]
        if diff.size:
            agg['cancer_mortality_age'][:] += np.bincount(diff, minlength=L)[:L]
    # Screening-affected mortality age
    m2 = (eff_death_screen != -1) & (eff_death_screen <= n_death)
    if m2.any():
        diff = (eff_death_screen[m2] - eff_start[m2]).astype(np.int64)
        L = agg['cancer_mortality_age_screen'].shape[0]
        diff = diff[(diff >= 0) & (diff < L)]
        if diff.size:
            agg['cancer_mortality_age_screen'][:] += np.bincount(diff, minlength=L)[:L]


def _itter_surv_batch(begs, fins, agg, key):
    """Vectorized equivalent of `_itter_surv` over arrays of (beg, fin).

    For each valid (beg, fin): increments agg[key][0] by 1 and agg[key][1..fin-beg]
    by 1.0 — reproducing the per-agent Python loop of the original.
    """
    arr = agg[key]
    L = arr.shape[0]
    begs = np.asarray(begs, dtype=np.int64)
    fins = np.asarray(fins, dtype=np.int64)
    begs = np.maximum(begs, 0)
    fins = np.minimum(fins, L - 1)
    ok = fins >= begs
    begs = begs[ok]
    fins = fins[ok]
    if begs.size == 0:
        return

    arr[0] += begs.shape[0]
    lengths = fins - begs
    diff = np.zeros(L + 1, dtype=np.float64)
    diff[1] += begs.shape[0]
    end = lengths + 1
    end = end[end <= L]
    if end.size:
        diff -= np.bincount(end, minlength=L + 1)[:L + 1]
    arr += np.cumsum(diff)[:L]


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


def compute_survival_by_group(pop, current_date, horizon=30):
    """Cause-specific survival from diagnosis, stratified by stage and aggressiveness.

    "Survive cancer" means cured (clinically or via screening). An uncured patient
    dies of cancer when the untreated cancer-death age is reached before natural
    death — the same rule used by ``compute_death_causes_vec``. Cured patients, and
    uncured patients who die of other causes first, are censored.

    Returns
    -------
    (survival, cure_frac)
      survival : dict key -> list of S(t) for t = 0..horizon-1. Keys are
                 'all_agg'/'all_nonagg' (pooled over stage) and
                 's1_agg'..'s4_agg' / 's1_nonagg'..'s4_nonagg'.
      cure_frac: dict group -> {'agg': fraction cured, 'nonagg': fraction cured},
                 group in {'all', 's1', 's2', 's3', 's4'}.
    """
    final = pop.ages(current_date).astype(np.float64)
    idx = np.where(pop.has_cancer)[0]
    if idx.size == 0:
        return {}, {}

    d_age = pop.diagnosis_age[idx].astype(np.float64)
    keep = (d_age != -1) & (d_age <= final[idx])
    idx = idx[keep]
    if idx.size == 0:
        return {}, {}

    d_age = pop.diagnosis_age[idx].astype(np.float64)
    n_death = pop.natural_death_age[idx].astype(np.float64)
    final_i = final[idx].astype(np.float64)
    init_death = pop.cancer_death_age_init[idx].astype(np.float64)
    cured = (pop.cancer_is_cured[idx] | pop.cancer_is_screening_cured[idx])
    stage = pop.cancer_diagnose_stage[idx].astype(np.int64)
    agg = pop.cancer_is_aggressive[idx].astype(np.int64)

    # A cancer death is an event only when the tumour is not cured and the
    # untreated cancer-death age is reached before natural death.
    event = (~cured) & (init_death <= n_death)
    event_time = np.where(event, init_death - d_age, horizon + 1.0)
    t_end = np.where(event, event_time, np.minimum(n_death, final_i) - d_age)
    t_end = np.clip(t_end, 0.0, horizon)

    def _curve(sub):
        if not sub.any():
            return [1.0] * horizon
        te = t_end[sub]
        ee = event_time[sub]
        at_risk = np.zeros(horizon, dtype=np.float64)
        dead = np.zeros(horizon, dtype=np.float64)
        for t in range(horizon):
            at_risk[t] = float(np.sum(te >= t))
            dead[t] = float(np.sum((ee >= t) & (ee < t + 1.0)))
        S = np.zeros(horizon, dtype=np.float64)
        H = 0.0
        for t in range(horizon):
            if at_risk[t] > 0:
                H += dead[t] / at_risk[t]
            S[t] = float(np.exp(-H))
        return S.tolist()

    survival: Dict = {}
    cure_frac: Dict = {}
    for s in (0, 1, 2, 3, 4):  # 0 = pooled over all stages
        sname = 'all' if s == 0 else f's{s}'
        smask = np.ones(stage.shape[0], dtype=bool) if s == 0 else (stage == s)
        for a, aname in ((1, 'agg'), (0, 'nonagg')):
            sub = smask & (agg == a)
            survival[f'{sname}_{aname}'] = _curve(sub)
            cure_frac.setdefault(sname, {})[aname] = (
                float(cured[sub].mean()) if sub.any() else float('nan'))
    return survival, cure_frac