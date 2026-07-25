"""Reproducible random number generation for the simulation model.

Replaces C# Tech.Rnd with numpy.random.Generator, supporting both sequential
and parallel-safe access patterns. Seeds are logged for reproducibility.
"""

import time as _time
from typing import Optional

import numpy as np


class RandomState:
    """Thread-safe, seed-tracked random state wrapper."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            self._seq = np.random.SeedSequence()
            self.seed = int(self._seq.generate_state(1)[0] & 0x7FFFFFFF)  # int32 for display
        else:
            self.seed = int(seed) & 0x7FFFFFFF
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed)))
    
    def reset(self, seed: Optional[int] = None):
        """Reset the generator with a new seed."""
        if seed is not None:
            self.seed = seed
        else:
            self._entropy_rng = np.random.SeedSequence()
            self.seed = self._entropy_rng.entropy
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed)))
    
    def next_double(self) -> float:
        """Generate a uniform random number in [0, 1). 
        Equivalent to C# Tech.NextDouble(false).
        """
        return self.rng.uniform(0.0, 1.0)
    
    def next_doubles(self, n: int) -> np.ndarray:
        """Generate n uniform random numbers. Vectorized batch version."""
        return self.rng.uniform(0.0, 1.0, size=n)
    
    def check_by_prob(self, prob: float) -> int:
        """Bernoulli trial: return 1 with probability `prob`, else 0.
        Equivalent to C# Tech.CheckByProb(prob).
        """
        return 1 if self.rng.uniform() < prob else 0
    
    def check_by_prob_batch(self, prob: float, n: int) -> np.ndarray:
        """Vectorized Bernoulli trials for n agents."""
        return (self.rng.uniform(0.0, 1.0, size=n) < prob).astype(np.int8)
    
    def exponential(self, scale: float, size: int = 1) -> np.ndarray:
        """Sample from Exponential(1/scale). Vectorized."""
        return self.rng.exponential(scale=scale, size=size)
    
    def normal(self, loc: float, scale: float, size: int = 1) -> np.ndarray:
        """Sample from Normal(loc, scale). Vectorized."""
        return self.rng.normal(loc=loc, scale=scale, size=size)


# Global random state instance
_global_random: Optional[RandomState] = None


def get_random() -> RandomState:
    """Get or create the global random state."""
    global _global_random
    if _global_random is None:
        _global_random = RandomState()
    return _global_random


def set_random(seed: int):
    """Set the global random state with a specific seed."""
    global _global_random
    _global_random = RandomState(seed)