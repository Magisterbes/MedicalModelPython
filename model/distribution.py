"""Empirical discrete distribution from CDF or PDF data.

Replaces C# Distribution class, ported to numpy vectorized operations.
Supports generation of random samples via inverse CDF method.
"""

from typing import Optional
import numpy as np


class Distribution:
    """Discrete empirical distribution for age sampling.
    
    Built from either CDF or PDF arrays. Normalization is automatic.
    Random sampling uses the inverse CDF method.
    
    Parameters
    ----------
    input_array : np.ndarray
        Array of probability mass (PDF) or cumulative probabilities (CDF).
    input_type : str
        Either 'cdf' or 'pdf'.
    
    Attributes
    ----------
    cdf : np.ndarray
        Normalized cumulative distribution function (values in [0, 1]).
    pdf : np.ndarray
        Normalized probability mass function.
    n : int
        Number of discrete bins (ages).
    normalization_coef : float
        Normalization coefficient (max of CDF or sum of PDF).
    """
    
    def __init__(self, input_array: np.ndarray, input_type: str):
        input_array = np.asarray(input_array, dtype=np.float64)
        
        if input_type == 'cdf':
            self.normalization_coef = float(np.max(input_array))
            if self.normalization_coef == 0:
                self.normalization_coef = 1.0
            self.cdf = input_array / self.normalization_coef
            self.pdf = np.diff(self.cdf, prepend=0.0)
        elif input_type == 'pdf':
            self.normalization_coef = float(np.sum(input_array))
            if self.normalization_coef == 0:
                self.normalization_coef = 1.0
            self.pdf = input_array / self.normalization_coef
            self.cdf = np.cumsum(self.pdf)
        else:
            raise ValueError(f"Unknown input_type: {input_type}. Use 'cdf' or 'pdf'.")
        
        # Ensure CDF ends at exactly 1.0
        if self.cdf[-1] > 0:
            self.cdf = self.cdf / self.cdf[-1]
        else:
            self.cdf[-1] = 1.0
        
        self.n = len(self.cdf)
    
    def generate_random(self, size: int = 1) -> np.ndarray:
        """Generate random samples from the distribution.
        
        Uses vectorized inverse CDF method via np.searchsorted.
        Equivalent to C# Distribution.GenerateRandom(), but batched.
        
        Parameters
        ----------
        size : int
            Number of samples to generate.
        
        Returns
        -------
        np.ndarray
            Array of sampled bin indices (int), shape (size,).
        """
        from .random import get_random
        rng = get_random()
        uniforms = rng.rng.uniform(0.0, 1.0, size=size)
        indices = np.searchsorted(self.cdf, uniforms, side='right')
        indices = np.clip(indices, 0, self.n - 1)
        return indices
    
    def generate_single(self) -> int:
        """Generate a single random sample. Scalar equivalent for back-compat."""
        from .random import get_random
        rng = get_random()
        u = rng.next_double()
        idx = int(np.searchsorted(self.cdf, u, side='right'))
        return min(idx, self.n - 1)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Alias for generate_random."""
        return self.generate_random(size)


def risks_to_distribution(risks: np.ndarray) -> np.ndarray:
    """Convert annual mortality risks to probability mass of death age.
    
    Given risks q(a) = P(death at age a | alive at a), compute:
        P(T = a) = prod_{j=0}^{a-1} (1 - q(j)) * q(a)
    
    Equivalent to C# Parameters.FromRisksToDistribution().
    
    Parameters
    ----------
    risks : np.ndarray
        Annual mortality risks q(a) for ages 0..n-1.
    
    Returns
    -------
    np.ndarray
        Probability mass function of death age.
    """
    risks = np.asarray(risks, dtype=np.float64)
    survival = np.cumprod(1.0 - risks)
    survival = np.concatenate([[1.0], survival[:-1]])
    pmf = survival * risks
    return pmf