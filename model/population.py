"""Population as Structure-of-Arrays (SoA) with numba-accelerated cancer logic."""
from typing import Tuple, Optional, Dict, Any
import numpy as np
from numba import njit, prange


class Population:
    def __init__(self, n_agents: int, unreal_life_length: int = 110):
        self.n_agents = n_agents
        self.unreal_life_length = unreal_life_length
        self.date_birth = np.zeros(n_agents, dtype=np.int32)
        self.is_alive = np.ones(n_agents, dtype=bool)
        self.natural_death_age = np.zeros(n_agents, dtype=np.int32)
        self.diagnosis_age = np.full(n_agents, unreal_life_length, dtype=np.int32)
        self.has_cancer = np.zeros(n_agents, dtype=bool)
        self.cancer_incidence_age = np.full(n_agents, -1, dtype=np.int32)
        self.cancer_diagnose_stage = np.full(n_agents, -1, dtype=np.int8)
        self.cancer_is_aggressive = np.zeros(n_agents, dtype=bool)
        self.cancer_growth_rate = np.zeros(n_agents, dtype=np.float64)
        self.cancer_is_cured = np.zeros(n_agents, dtype=bool)
        self.cancer_is_screening_cured = np.zeros(n_agents, dtype=bool)
        self.cancer_screening_found = np.zeros(n_agents, dtype=bool)
        self.cancer_screening_age = np.full(n_agents, -1, dtype=np.int32)
        self.cancer_screening_stage = np.full(n_agents, -1, dtype=np.int8)
        self.cancer_death_age_init = np.full(n_agents, 1000.0, dtype=np.float64)
        self.cancer_death_age_cure = np.full(n_agents, 1000.0, dtype=np.float64)
        self.cancer_death_age_screen = np.full(n_agents, 1000.0, dtype=np.float64)
        self.cancer_reoccurred = np.zeros(n_agents, dtype=bool)
        self.cancer_after_diag_years = np.zeros(n_agents, dtype=np.int32)
        self.cancer_stages_ages = np.full((n_agents, 5), -1, dtype=np.int32)
        self._death_cause_cache = np.full(n_agents, -1, dtype=np.int8)
        self._death_cause_valid = np.zeros(n_agents, dtype=bool)
    
    @property
    def n_alive(self) -> int:
        return int(self.is_alive.sum())
    
    def ages(self, current_date: int) -> np.ndarray:
        return (current_date - self.date_birth).astype(np.int32)
    
    def invalidate_death_cause_cache(self):
        self._death_cause_valid[:] = False


def generate_population(n_agents, init_age_dist_cdf, aging_dist_cdf,
    diagnose_hazard_values, lead_time_rates, stage_by_age_probs,
    proportion_aggressive, growth_rate_limits, aggressiveness_rate_threshold,
    treatment_efficiency, age_cure_constants, cancer_death_hazard_lambda,
    reoccurrence_prob, unreal_life_length, seed):
    
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    pop = Population(n_agents, unreal_life_length)
    
    # Generate ages
    uniforms = rng.uniform(0.0, 1.0, size=n_agents)
    ages = np.searchsorted(init_age_dist_cdf, uniforms).astype(np.int32)
    pop.date_birth = -ages
    
    # Natural death ages
    uniforms = rng.uniform(0.0, 1.0, size=n_agents)
    death_ages = np.searchsorted(aging_dist_cdf, uniforms).astype(np.int32)
    mask = death_ages <= ages
    n_resample = mask.sum()
    if n_resample > 0:
        uniforms = rng.uniform(0.0, 1.0, size=n_resample)
        death_ages[mask] = np.searchsorted(aging_dist_cdf, uniforms).astype(np.int32)
        still_bad = death_ages <= ages
        death_ages[still_bad] = ages[still_bad] + 1
    pop.natural_death_age = death_ages
    
    # Diagnosis ages (vectorized Bernoulli trials)
    max_age = min(unreal_life_length, len(diagnose_hazard_values))
    all_uniforms = rng.uniform(0.0, 1.0, size=(n_agents, max_age))
    hazards_broadcast = diagnose_hazard_values[:max_age].reshape(1, -1)
    successes = all_uniforms < hazards_broadcast
    first_success_idx = np.argmax(successes, axis=1)
    has_success = successes[np.arange(n_agents), first_success_idx]
    diag_ages = np.where(has_success, first_success_idx.astype(np.int32), unreal_life_length)
    pop.diagnosis_age = np.where(ages <= 98, diag_ages, np.int32(unreal_life_length))
    
    # Who gets cancer
    has_cancer = (pop.diagnosis_age < unreal_life_length) & (pop.diagnosis_age < pop.natural_death_age)
    pop.has_cancer = has_cancer
    
    # Cancer histories via numba
    cancer_seeds = rng.integers(0, 2_000_000_000, size=n_agents, dtype=np.int64)
    
    _compute_cancer_histories_numba(
        n_agents, has_cancer, pop.diagnosis_age, pop.natural_death_age,
        pop.date_birth, cancer_seeds, lead_time_rates,
        np.array(list(stage_by_age_probs.keys()), dtype=np.int32),
        np.array(list(stage_by_age_probs.values()), dtype=np.float64),
        proportion_aggressive,
        np.array(growth_rate_limits, dtype=np.float64),
        aggressiveness_rate_threshold, treatment_efficiency,
        age_cure_constants, cancer_death_hazard_lambda,
        reoccurrence_prob, unreal_life_length,
        pop.cancer_incidence_age, pop.cancer_diagnose_stage,
        pop.cancer_is_aggressive, pop.cancer_growth_rate,
        pop.cancer_is_cured, pop.cancer_death_age_init,
        pop.cancer_death_age_cure, pop.cancer_reoccurred,
        pop.cancer_stages_ages)
    
    return pop


@njit(cache=True)
def _compute_cancer_histories_numba(
    n_agents, has_cancer, diagnosis_age, natural_death_age,
    date_birth, cancer_seeds, lead_time_rates,
    stage_by_age_keys, stage_by_age_vals,
    proportion_aggressive, growth_rate_limits,
    aggressiveness_rate_threshold, treatment_efficiency,
    age_cure_constants, cancer_death_hazard_lambda,
    reoccurrence_prob, unreal_life_length,
    cancer_incidence_age, cancer_diagnose_stage,
    cancer_is_aggressive, cancer_growth_rate,
    cancer_is_cured, cancer_death_age_init,
    cancer_death_age_cure, cancer_reoccurred,
    cancer_stages_ages):
    
    for i in range(n_agents):
        if not has_cancer[i]:
            continue
        
        diag_age = diagnosis_age[i]
        nat_death = natural_death_age[i]
        seed = cancer_seeds[i]
        
        # Stage at diagnosis
        age_gr = 10 * (diag_age // 10)
        probs = _get_probs_for_age(age_gr, stage_by_age_keys, stage_by_age_vals)
        stage = _multinomial_sample(probs, seed) + 1
        cancer_diagnose_stage[i] = stage
        
        # Aggressiveness
        prop_agg = proportion_aggressive[stage - 1]
        is_agg = _bernoulli(prop_agg, seed + 1)
        cancer_is_aggressive[i] = is_agg
        
        # Lead time
        lt_rate = lead_time_rates[stage - 1]
        ttd = np.ceil(-np.log(max(_uniform(seed + 2), 1e-15)) / lt_rate)
        
        # Growth rate
        if ttd > 0:
            if stage == 1:
                gamma = np.exp(np.log(stage + _uniform(seed + 3)) / ttd)
            else:
                gamma = np.exp(np.log(stage) / ttd)
        else:
            gamma = _sample_growth_rate(
                growth_rate_limits[0], growth_rate_limits[1],
                aggressiveness_rate_threshold, is_agg, seed + 4)
        cancer_growth_rate[i] = gamma
        
        # Stage ages
        incidence_age = int(max(diag_age - ttd, 0))
        cancer_incidence_age[i] = incidence_age
        
        stages = np.zeros(5, dtype=np.int32)
        stages[0] = incidence_age
        if gamma > 1.0:
            log_gamma = np.log(gamma)
            stages[1] = incidence_age + int(np.log(2.0) / log_gamma)
            stages[2] = incidence_age + int(np.log(3.0) / log_gamma)
            stages[3] = incidence_age + int(np.log(4.0) / log_gamma)
        else:
            stages[1] = incidence_age + 5
            stages[2] = incidence_age + 10
            stages[3] = incidence_age + 15
        
        delta = -np.log(max(_uniform(seed + 5), 1e-15)) / cancer_death_hazard_lambda
        stages[4] = stages[3] + int(delta)
        cancer_stages_ages[i, :] = stages
        cancer_death_age_init[i] = float(stages[4])
        
        if diag_age > nat_death and stages[4] > nat_death:
            continue
        
        # Treatment
        age_cure_eff = _get_age_cure_eff(diag_age, age_cure_constants)
        cure_prob = age_cure_eff * treatment_efficiency[stage - 1]
        if cure_prob > 1.0:
            cure_prob = 1.0
        cured = _bernoulli(cure_prob, seed + 6)
        cancer_is_cured[i] = cured
        
        if cured:
            new_age = diag_age + 5.0 + _uniform(seed + 7) * 5.0
            if new_age < stages[4]:
                new_age = stages[4] + _uniform(seed + 8) * 5.0
            cancer_death_age_cure[i] = new_age
        
        cancer_reoccurred[i] = _bernoulli(reoccurrence_prob, seed + 9)


# Simple LCG random for numba
@njit(cache=True)
def _uniform(seed):
    s = np.uint64(seed) + np.uint64(1)
    s = s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    s = s ^ (s >> np.uint64(21))
    return float(s & np.uint64(0xFFFFFFFFFFFFF)) / float(np.uint64(0xFFFFFFFFFFFFF) + 1)


@njit(cache=True)
def _bernoulli(p, seed):
    return _uniform(seed) < p


@njit(cache=True)
def _multinomial_sample(probs, seed):
    u = _uniform(seed)
    cumsum = 0.0
    for j in range(len(probs)):
        cumsum += probs[j]
        if u < cumsum:
            return j
    return len(probs) - 1


@njit(cache=True)
def _get_probs_for_age(age_gr, keys, vals):
    best_key = keys[0]
    best_dist = abs(age_gr - keys[0])
    for k in range(1, len(keys)):
        d = abs(age_gr - keys[k])
        if d < best_dist:
            best_dist = d
            best_key = keys[k]
    for k in range(len(keys)):
        if keys[k] == best_key:
            return vals[k]
    return vals[0]


@njit(cache=True)
def _sample_growth_rate(low, up, threshold, is_agg, seed):
    u = _uniform(seed)
    if is_agg:
        return low + (1.0 - threshold) * (up - low) * u
    else:
        return low + (1.0 - threshold) * (up - low) + threshold * (up - low) * u


@njit(cache=True)
def _get_age_cure_eff(age, age_cure_constants):
    if age <= 40:
        return age_cure_constants[0]
    elif age <= 50:
        return age_cure_constants[1]
    elif age <= 60:
        return age_cure_constants[2]
    elif age <= 70:
        return age_cure_constants[3]
    else:
        return age_cure_constants[4]


# --- Screening ---
@njit(cache=True)
def apply_screening_batch(n_agents, is_alive, ages, has_cancer,
    cancer_incidence_age, cancer_screening_found, cancer_screening_age,
    cancer_diagnosis_age, cancer_stages_ages, cancer_is_cured,
    cancer_is_screening_cured, cancer_death_age_init, cancer_death_age_screen,
    cancer_screening_stage, screening_start_age, screening_finish_age,
    participation_rate, test_tp, treatment_efficiency, age_cure_constants, seed_param):
    
    screening_detected = np.zeros(n_agents, dtype=np.int8)
    fp_count = 0
    
    for i in range(n_agents):
        if not is_alive[i]:
            continue
        
        a = ages[i]
        eligible = (a >= screening_start_age and a <= screening_finish_age
            and (not cancer_screening_found[i] or cancer_screening_age[i] == -1)
            and a < cancer_diagnosis_age[i])
        if not eligible:
            continue
        
        # Safe seed per agent (use modulo to avoid overflow)
        bseed = (seed_param & 0x3FFFFFFF) + (np.int64(i) & 0x3FFFFFFF) * 1000
        
        if _uniform(bseed + 1) >= participation_rate:
            continue
        
        if has_cancer[i] and cancer_incidence_age[i] < a:
            if _bernoulli(test_tp, bseed + 2):
                cancer_screening_found[i] = True
                cancer_screening_age[i] = a
                screening_detected[i] = True
                
                stages = cancer_stages_ages[i]
                st = _get_stage_by_age(stages, a)
                cancer_screening_stage[i] = st
                
                age_eff = _get_age_cure_eff(a, age_cure_constants)
                cure_prob = age_eff * treatment_efficiency[st - 1]
                if cure_prob > 1.0:
                    cure_prob = 1.0
                
                if _bernoulli(cure_prob, bseed + 3):
                    cancer_is_screening_cured[i] = True
                    new_age = a + 5.0 + _uniform(bseed + 4) * 5.0
                    if new_age < cancer_death_age_init[i]:
                        new_age = cancer_death_age_init[i] + _uniform(bseed + 5) * 5.0
                    cancer_death_age_screen[i] = new_age
                else:
                    cancer_is_screening_cured[i] = False
                    cancer_is_cured[i] = False
                    cancer_death_age_init[i] = float(stages[4])
        else:
            if _bernoulli(test_tp * 0.1, bseed + 6):
                fp_count += 1
    
    return fp_count, screening_detected


@njit(cache=True)
def _get_stage_by_age(stages, age):
    count = 0
    for s in range(len(stages)):
        if stages[s] <= age:
            count += 1
    if count >= len(stages):
        return len(stages) - 1
    if count < 1:
        return 1
    return count


# --- Vectorized death cause ---
def compute_death_causes_vec(is_alive, ages, natural_death_age, has_cancer,
    cancer_death_age_init, cancer_death_age_cure, cancer_death_age_screen,
    cancer_is_cured, cancer_is_screening_cured, cancer_screening_found,
    diagnosis_age, unreal_life_length):
    
    result = np.zeros(len(is_alive), dtype=np.int8)
    alive = is_alive
    cancer = has_cancer
    nat_death_eq = natural_death_age <= ages
    cancer_death_init_le = cancer_death_age_init <= ages
    cancer_death_init_gt_nat = cancer_death_age_init > natural_death_age
    
    result[alive & ~cancer] = 0
    mask_natural = (alive & ~cancer & nat_death_eq) | (cancer & nat_death_eq & cancer_death_init_gt_nat)
    result[mask_natural] = 1
    mask_nat_cured = cancer & nat_death_eq & cancer_death_init_le & cancer_is_cured
    result[mask_nat_cured] = 4
    mask_cancer_death = cancer & (~nat_death_eq) & cancer_death_init_le & ~cancer_is_cured & ~cancer_is_screening_cured
    result[mask_cancer_death] = 2
    mask_saved = cancer & nat_death_eq & cancer_death_init_le & ~cancer_is_cured & cancer_is_screening_cured
    result[mask_saved] = 3
    return result