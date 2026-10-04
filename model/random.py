"""Reproducible random number generation — minimal core.

Replaces C# Tech.Rnd. Seeds are logged for reproducibility.
Only `next_double()` and `next_doubles(n)` are used outside numba.
Numba functions use their own simple LCG (`_uniform` in population.py).

The state is **thread-local**. The web app serves concurrent runs from separate
threads, and a module-level singleton let one run re-seed the stream another run
was drawing from — silently breaking reproducibility. Every drawing site is
preceded by `set_random(...)` in the same thread (`Simulation.__init__` and
`Simulation.from_params`), so a seeded run yields exactly the same numbers as
before.
"""

from typing import Optional
import threading
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


# One RandomState per thread: concurrent runs in the same process no longer share
# (and re-seed) a single stream.
_local = threading.local()


def get_random() -> RandomState:
    """Random state of the *current thread*, created on first use."""
    rng = getattr(_local, 'random', None)
    if rng is None:
        rng = RandomState()
        _local.random = rng
    return rng


def set_random(seed: int) -> None:
    """(Re)seed the current thread's random state."""
    _local.random = RandomState(seed)