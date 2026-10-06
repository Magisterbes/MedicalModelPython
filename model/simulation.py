"""Main simulation engine: initialization, annual iteration, and orchestration.

Replaces C# Environment.cs + Demographics.cs + Screening.cs.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging
import time

import numpy as np

from .parameters import Parameters
from .population import (
    Population, generate_population, apply_screening_batch
)
from .stats import StatsCollection
from .random import set_random, get_random

logger = logging.getLogger(__name__)


def _add_by_age(target_row: np.ndarray, sel_ages: np.ndarray) -> None:
    """Accumulate counts of `sel_ages` into `target_row` (indexed by age).

    Vectorized (np.bincount) replacement for the per-agent Python loops in
    `iterate_year` — up to ~20x faster than a Python loop. Ages outside the
    target range are ignored, matching the safety guards of the original
    guarded loops.
    """
    if sel_ages.size == 0:
        return
    sz = target_row.shape[0]
    if sel_ages.min() < 0 or sel_ages.max() >= sz:
        sel_ages = sel_ages[(sel_ages >= 0) & (sel_ages < sz)]
        if sel_ages.size == 0:
            return
    counts = np.bincount(sel_ages, minlength=sz)
    target_row[:sz] += counts[:sz]


class Simulation:
    """Top-level simulation orchestrator.
    
    Parameters
    ----------
    param_source : str
        Path to parameters file (.toml or legacy .txt).
    param_data_dir : str, optional
        Override data directory.
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        param_source: str = "config/parameters.toml",
        param_data_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            set_random(seed)
        rng = get_random()
        logger.info(f"Random seed: {rng.seed}")
        
        param_path = Path(param_source)
        if param_path.suffix == '.toml':
            self.params = Parameters.from_toml(str(param_path))
        elif param_path.suffix == '.txt':
            self.params = Parameters.from_legacy_txt(str(param_path))
        else:
            raise ValueError(f"Unsupported parameter file format: {param_path.suffix}")
        
        if param_data_dir is not None:
            self.params.data_dir = param_data_dir
        elif not Path(self.params.data_dir).is_absolute():
            project_root = param_path.parent.parent
            self.params.data_dir = str(project_root / self.params.data_dir)
        
        self.params.init_data()
        
        self.population: Optional[Population] = None
        self.stats: Optional[StatsCollection] = None
        self.current_date: int = 0

    @classmethod
    def from_params(
        cls,
        params: Parameters,
        seed: Optional[int] = None,
    ) -> "Simulation":
        """Create a Simulation from an already-initialized Parameters object.

        Unlike the normal constructor, this skips reading the parameter file and
        re-running `Parameters.init_data()` (which retrains regressions). It is
        intended for worker processes that receive pre-fitted parameters, e.g.
        in the parallel sensitivity analysis.
        """
        obj = cls.__new__(cls)
        if seed is not None:
            set_random(seed)
        obj.params = params
        obj.population = None
        obj.stats = None
        obj.current_date = 0
        return obj

    def start(self):
        """Generate initial population.
        
        Equivalent to C# Environment.Start().
        """
        logger.info(f"Generating initial population of {self.params.init_population} agents...")
        t0 = time.perf_counter()
        
        self.current_date = 0
        
        self.stats = StatsCollection(
            years_to_simulate=self.params.years_to_simulate,
            unreal_life_length=self.params.unreal_life_length,
            stage_distribution_length=self.params.stage_distribution_length,
        )
        
        rng = get_random()
        cancer_death_lambda = np.exp(self.params.cancer_death_hazard.constants[0])
        
        self.population = generate_population(
            n_agents=self.params.init_population,
            init_age_dist_cdf=self.params.init_age_dist.cdf,
            aging_dist_cdf=self.params.aging_dist.cdf,
            diagnose_hazard_values=self.params.diagnose_hazard.value_by_age,
            lead_time_rates=np.array(self.params.lead_time_distributions),
            stage_by_age_probs=self.params.stage_by_age_reg_generator,
            proportion_aggressive=self.params.proportion_of_aggressive,
            growth_rate_limits=(
                self.params.growth_rate_limits[0],
                self.params.growth_rate_limits[1],
            ),
            aggressiveness_rate_threshold=self.params.aggressiveness_rate_threshold,
            treatment_efficiency=np.array(self.params.treatment_efficiency),
            age_cure_constants=np.array(self.params.age_cure_constants),
            aggressiveness_cure_odds_ratio=np.array(
                self.params.aggressiveness_cure_odds_ratio),
            cancer_death_hazard_lambda=cancer_death_lambda,
            reoccurrence_prob=self.params.reoccurrence_probability,
            unreal_life_length=self.params.unreal_life_length,
            seed=rng.seed + 1,
        )
        
        dt = time.perf_counter() - t0
        logger.info(f"Population generated in {dt:.2f}s. "
                     f"Cancer cases: {self.population.has_cancer.sum()}")
    
    def iterate_year(self):
        """Advance simulation by one year.
        
        Equivalent to C# Environment.ItteratePopulation().
        """
        pop = self.population
        stats = self.stats
        params = self.params
        current_date = self.current_date
        
        pop.invalidate_death_cause_cache()
        ages = pop.ages(current_date)
        
        # First pass: natural deaths + screening
        alive_mask = pop.is_alive
        natural_death_mask = alive_mask & (ages == pop.natural_death_age)
        pop.is_alive[natural_death_mask] = False
        
        screening_period = (
            current_date >= params.screening_date
            and ((current_date - params.screening_date) % params.frequency) == 0
        )
        
        if screening_period:
            # Truncate seed to int32 for numba compatibility
            safe_seed = np.int32(get_random().seed & 0x7FFFFFFF) + np.int32(current_date * 10000)
            fp_count, _ = apply_screening_batch(
                n_agents=pop.n_agents,
                is_alive=pop.is_alive,
                ages=ages,
                has_cancer=pop.has_cancer,
                cancer_incidence_age=pop.cancer_incidence_age,
                cancer_screening_found=pop.cancer_screening_found,
                cancer_screening_age=pop.cancer_screening_age,
                cancer_diagnosis_age=pop.diagnosis_age,
                cancer_stages_ages=pop.cancer_stages_ages,
                cancer_is_cured=pop.cancer_is_cured,
                cancer_is_screening_cured=pop.cancer_is_screening_cured,
                cancer_death_age_init=pop.cancer_death_age_init,
                cancer_death_age_screen=pop.cancer_death_age_screen,
                cancer_screening_stage=pop.cancer_screening_stage,
                screening_start_age=params.start_age,
                screening_finish_age=params.finish_age,
                participation_rate=params.participation_rate,
                test_tp=params.test_tp_selected,
                treatment_efficiency=np.array(params.treatment_efficiency),
                age_cure_constants=np.array(params.age_cure_constants),
                seed_param=safe_seed,
            )
            stats.agg_stats['false_positives'][current_date] += fp_count
        
        # Second pass: statistics for alive agents (vectorized via np.bincount)
        still_alive = pop.is_alive
        _add_by_age(stats.age_distributions[current_date], ages[still_alive])
        
        # At-risk: alive agents not yet clinically diagnosed before current age
        at_risk_mask = still_alive & (
            (pop.diagnosis_age == params.unreal_life_length) | (pop.diagnosis_age > ages)
        )
        _add_by_age(stats.at_risk[current_date], ages[at_risk_mask])
        
        # Effective death age with screening (matches C# Person.CancerDeathAgeScreening)
        eff_death_screen = np.where(
            pop.cancer_screening_found & pop.cancer_is_screening_cured,
            pop.cancer_death_age_screen,  # cured by screening → die later
            pop.cancer_death_age_init      # else → same as no screening
        )
        
        # Cancer mortality (overall, no-screening scenario)
        cancer_death = (
            still_alive
            & pop.has_cancer
            & (ages == pop.cancer_death_age_init)
            & (pop.cancer_death_age_init <= pop.natural_death_age)
        )
        _add_by_age(stats.cancer_mortality[current_date], ages[cancer_death])
        
        # Screening-affected mortality (uses effective death age)
        cancer_death_screen = (
            still_alive
            & pop.has_cancer
            & (ages == eff_death_screen)
            & (eff_death_screen <= pop.natural_death_age)
        )
        _add_by_age(stats.cancer_screening_mortality[current_date], ages[cancer_death_screen])
        
        # Incidence and diagnosis tracking (needed for incidence_rates & diagnosis_rates)
        # An agent "gets cancer" in the year when their age equals cancer_incidence_age
        new_incidence = still_alive & pop.has_cancer & (ages == pop.cancer_incidence_age)
        _add_by_age(stats.incidence[current_date], ages[new_incidence])
        
        # Diagnosis happens at diagnosis_age (year when age reaches diagnosis_age)
        new_diag = still_alive & pop.has_cancer & (ages == pop.diagnosis_age) & (pop.diagnosis_age < params.unreal_life_length)
        _add_by_age(stats.diagnosis[current_date], ages[new_diag])
        
        self.current_date += 1
    
    def run_full_simulation(self, verbose: bool = True) -> StatsCollection:
        """Run the complete simulation from start to end."""
        self.start()
        
        years = self.params.years_to_simulate
        for year in range(years - 1):
            if verbose and year % 5 == 0:
                logger.info(f"Simulating year {year}/{years}...")
            self.iterate_year()
        
        logger.info("Gathering final statistics...")
        self.stats.gather_stats(self.population, self.params, self.current_date)
        
        logger.info("Simulation complete.")
        return self.stats
    
    def get_summary(self) -> dict:
        if self.stats is None:
            return {}
        pop = self.population
        params = self.params
        return {
            "n_agents": params.init_population,
            "n_alive_end": pop.n_alive,
            "n_cancer_cases": int(pop.has_cancer.sum()),
            "n_cured_clinical": int(pop.cancer_is_cured.sum()),
            "n_cured_screening": int(pop.cancer_is_screening_cured.sum()),
            "n_screening_detected": int(pop.cancer_screening_found.sum()),
            "years_simulated": self.current_date,
            "random_seed": get_random().seed,
        }