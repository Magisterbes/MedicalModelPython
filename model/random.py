"""Reproducible random number generation — minimal core.

Replaces C# Tech.Rnd. Seeds are logged for reproducibility.
Only `next_double()` and `next_doubles(n)` are used outside numba.
Numba functions use their own simple LCG (`_uniform` in population.py).
"""

from typing import Optional
import numpy as np


class RandomState:
    """Minimal random state wrapper with seed tracking."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seq = np.random.SeedSequence()
            self.seed = int(seq.generate_state(1)[0] & 0x7FFFFFFF)
        else:
            self.seed = int(seed) & 0x7FFFFFFF
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed)))
    
    def next_double(self) -> float:
        """Uniform [0, 1). Equivalent to C# Tech.NextDouble."""
        return self.rng.uniform(0.0, 1.0)
    
    def next_doubles(self, n: int) -> np.ndarray:
        """n uniform [0, 1) numbers."""
        return self.rng.uniform(0.0, 1.0, size=n)


_global_random: Optional[RandomState] = None


def get_random() -> RandomState:
    global _global_random
    if _global_random is None:
        _global_random = RandomState()
    return _global_random


def set_random(seed: int):
    global _global_random
    _global_random = RandomState(seed)