"""Reduced Gompertz model for tumor progression staging.

Replaces C# GompertzModel in Tech/GompertzModel.cs.

The reduced Gompertz model is:
    V(t) = exp(K - exp(C - B_ind * t))

where:
    K, C : population-level shape parameters
    B_ind = exp(log(B_pop) + eps), eps ~ N(0, B_std^2)

V(t) is interpreted as a proxy for tumor size/stage. Individual variability
is modeled through a normal random effect on log(B).

Mathematical notes:
- At t=0: V(0) = exp(K - exp(C)), should be near 0 (small tumor)
- As t -> inf: V(t) -> exp(K) (carrying capacity)
- The inflection point is at t = C / B_ind

For aggregation, the function used depends on context:
- ComputeOutput(id, age): with individual random effect (simulation)
- ComputeOutputTrain(age): population-mean prediction (fitting)
- ComputeTrainAges(ages): vectorized population-mean predictions
"""

import numpy as np


class GompertzModel:
    """Reduced Gompertz growth model with individual random effects.
    
    Parameters
    ----------
    K : float
        Carrying capacity parameter. V(t) -> exp(K) as t -> inf.
    C : float
        Shape/offset parameter. Determines initial value V(0) = exp(K - exp(C)).
    B_pop : float
        Population-mean rate parameter (log-scale).
    B_std : float
        Standard deviation of individual random effects on log(B).
    """
    
    def __init__(self, K: float, C: float, B_pop: float, B_std: float):
        self.K = K
        self.C = C
        self.B_pop = B_pop
        self.B_std = B_std
    
    def to_array(self) -> np.ndarray:
        """Return parameters as array [K, C, B_pop, B_std]."""
        return np.array([self.K, self.C, self.B_pop, self.B_std], dtype=np.float64)
    
    @staticmethod
    def from_array(params: np.ndarray) -> 'GompertzModel':
        """Create from parameter array [K, C, B_pop, B_std]."""
        return GompertzModel(
            float(params[0]), float(params[1]),
            float(params[2]), float(params[3])
        )
    
    def compute_output(self, seed: int, age: float) -> float:
        """Compute V(age) with individual random effect.
        
        Equivalent to C# GompertzModel.ComputeOutput(id, age).
        Uses a deterministic random generator seeded by agent ID.
        
        Parameters
        ----------
        seed : int
            Agent ID used as random seed for individual variability.
        age : float
            Age (time since incidence) in years.
        
        Returns
        -------
        float
            V(age) — continuous proxy for tumor size/stage.
        """
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        eps = rng.normal(0.0, self.B_std)
        B_ind = np.exp(np.log(self.B_pop) + eps)
        V = np.exp(self.K - np.exp(self.C - B_ind * age))
        return V
    
    def compute_output_train(self, age: float) -> float:
        """Compute population-mean V(age) without individual variation.
        
        Equivalent to C# GompertzModel.ComputeOutputTrain(age).
        Used during model fitting.
        
        Parameters
        ----------
        age : float
            Age (time since incidence) in years.
        
        Returns
        -------
        float
            Population-mean V(age).
        """
        V = np.exp(self.K - np.exp(self.C - self.B_pop * age))
        return V
    
    def compute_train_ages(self, ages: np.ndarray) -> np.ndarray:
        """Vectorized population-mean prediction for multiple ages.
        
        Equivalent to C# GompertzModel.ComputeTrainAges(ages).
        
        Parameters
        ----------
        ages : np.ndarray
            Array of ages.
        
        Returns
        -------
        np.ndarray
            Array of V(age) values, same shape as ages.
        """
        ages = np.asarray(ages, dtype=np.float64)
        V = np.exp(self.K - np.exp(self.C - self.B_pop * ages))
        return V


def gompertz_v(t: np.ndarray, K: float, C: float, B: float) -> np.ndarray:
    """Vectorized reduced Gompertz function.
    
    V(t) = exp(K - exp(C - B * t))
    
    Parameters
    ----------
    t : np.ndarray
        Time values.
    K, C, B : float
        Model parameters.
    
    Returns
    -------
    np.ndarray
        V(t) values.
    """
    t = np.asarray(t, dtype=np.float64)
    return np.exp(K - np.exp(C - B * t))


def compute_stage_age(growth_rate: float, incidence_age: float, n_stages: int = 4) -> np.ndarray:
    """Compute ages at which each tumor stage is reached.
    
    Uses the formula: t_s = t_0 + log(s) / log(growth_rate)
    for s = 1, 2, 3, 4, and adds an extra terminal stage.
    
    Equivalent to C# Cancer.GetStagesAges().
    
    Parameters
    ----------
    growth_rate : float
        Tumor growth rate gamma.
    incidence_age : int
        Age at tumor incidence (first malignant cell).
    n_stages : int
        Number of clinical stages (default 4: I, II, III, IV).
    
    Returns
    -------
    np.ndarray
        Array of ages [t_0, t_1, t_2, t_3, t_4], where:
        t_0 = incidence_age
        t_s = incidence_age + floor(log(s)/log(gamma)) for s=1..n_stages
        t_{n_stages} = t_{n_stages-1} + Delta (terminal, set by caller or default)
    """
    stages = np.zeros(n_stages + 1, dtype=np.int32)
    stages[0] = int(incidence_age)
    
    if growth_rate > 1.0:
        log_gamma = np.log(growth_rate)
        for s in range(1, n_stages):
            stages[s] = int(incidence_age + np.log(s + 1) / log_gamma)
    else:
        # Degenerate case: very slow growth
        for s in range(1, n_stages):
            stages[s] = int(incidence_age + s * 10)  # arbitrary spacing
    
    return stages


def get_stage_by_age(stages_ages: np.ndarray, age: int) -> int:
    """Determine which stage a tumor is at, given age.
    
    Equivalent to C# Cancer.GetStageByAge(age).
    
    Parameters
    ----------
    stages_ages : np.ndarray
        Array [t_0, t_1, t_2, t_3, t_4] of stage ages.
    age : int
        Current age.
    
    Returns
    -------
    int
        Stage number (1-indexed: 1=I, 2=II, 3=III, 4=IV).
    """
    diff = stages_ages - age
    past_stages = diff[diff <= 0]
    if len(past_stages) >= len(stages_ages):
        return len(stages_ages) - 1  # Terminal
    return len(past_stages)