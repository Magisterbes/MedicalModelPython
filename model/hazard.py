"""Hazard functions: PiecewiseHazard (diagnosis) + ExpHazard (cancer death).

Replaces the larger C# Hazard.cs. GompHazard and LogLogisticHazard
were never used in the actual simulation and have been removed.
"""

from abc import ABC, abstractmethod
import numpy as np


class Hazard(ABC):
    """Abstract base for hazard functions."""
    
    def __init__(self, constants: np.ndarray):
        self.constants = np.asarray(constants, dtype=np.float64)
    
    @abstractmethod
    def get_value(self, time: float) -> float: ...
    
    def update(self): pass


class ExpHazard(Hazard):
    """Constant hazard: h(t) = exp(L). Used for CancerDeathHazard."""
    def __init__(self, constant: float):
        super().__init__(np.array([constant]))
    def get_value(self, time: float) -> float:
        return np.exp(self.constants[0])


class PiecewiseHazard(Hazard):
    """Piecewise constant log-hazard for diagnosis.
    
    log h(t) = beta_0 + sum_k H(t - t_k) * beta_k
    Values cached for ages 0..unreal_life_length.
    """
    
    def __init__(self, constants: np.ndarray, years: np.ndarray, unreal_life_length: int = 110):
        super().__init__(constants)
        self.years = np.asarray(years, dtype=np.float64)
        self.unreal_life_length = unreal_life_length
        self.value_by_age = np.zeros(unreal_life_length + 1, dtype=np.float64)
        self._build_cache()
    
    def _build_cache(self):
        for i in range(self.unreal_life_length + 1):
            result = self.constants[0]
            for k in range(1, len(self.constants)):
                if i >= self.years[k - 1]:
                    result += self.constants[k]
            self.value_by_age[i] = min(np.exp(result), 1.0)
    
    def get_value(self, time: float) -> float:
        t_int = int(time)
        return self.value_by_age[t_int] if t_int < len(self.value_by_age) else self.value_by_age[-1]
    
    def T_batch(self, n_agents: int) -> np.ndarray:
        """Vectorized Bernoulli trials: first age where Ber(h(a)) == 1."""
        max_a = min(self.unreal_life_length, len(self.value_by_age))
        u = np.random.uniform(0, 1, size=(n_agents, max_a))
        successes = u < self.value_by_age[:max_a].reshape(1, -1)
        idx = np.argmax(successes, axis=1)
        ok = successes[np.arange(n_agents), idx]
        return np.where(ok, idx.astype(np.int32), self.unreal_life_length).astype(np.int32)
    
    def update(self):
        self._build_cache()


def parse_hazard(spec: str, unreal_life_length: int = 110) -> Hazard:
    """Parse a hazard specification string into a Hazard object.
    
    Supported formats:
        "piecewise, t1@c1;t2@c2;..." -> PiecewiseHazard
        "exp, L"                     -> ExpHazard
        "gomp, L, B"                 -> PiecewiseHazard (converted)
    """
    parts = [p.strip() for p in spec.split(',')]
    htype = parts[0].lower()
    
    if htype == 'piecewise':
        pairs = parts[1].split(';')
        years = []
        coefs = []
        for pair in pairs:
            pair = pair.strip()
            if '@' in pair:
                y, c = pair.split('@')
                years.append(float(y.strip()))
                coefs.append(float(c.strip()))
        return PiecewiseHazard(np.array(coefs), np.array(years), unreal_life_length)
    else:
        L = float(parts[1])
        return ExpHazard(L)