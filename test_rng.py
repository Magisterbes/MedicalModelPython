"""Guard tests for the per-agent random generator.

The cancer-history kernel draws each of an agent's quantities from
``_uniform(seed + k)`` for successive small ``k``, so ``_uniform`` must map nearby
indices to essentially independent values. A previous LCG + single xorshift did
not (corr(u(s), u(s+1)) ~ +0.36), which coupled the stage draw to lead time, cure,
cancer-death age and the aggressiveness label. These tests pin the invariant.
"""
import numpy as np

from model.population import _uniform

N = 50_000
KS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def _draws():
    s = np.arange(N, dtype=np.int64)
    u0 = np.array([_uniform(int(x)) for x in s])
    return s, u0


def test_uniform_moments():
    _, u0 = _draws()
    # SE of the mean for uniform[0,1] is ~1/sqrt(12*N) = 0.0013 at N=50k; allow 5 SE.
    assert abs(u0.mean() - 0.5) < 0.006, 'mean %.5f' % u0.mean()
    assert abs(u0.var() - 1.0 / 12.0) < 0.005, 'var %.6f' % u0.var()


def test_neighbouring_indices_are_independent():
    s, u0 = _draws()
    for k in KS:
        uk = np.array([_uniform(int(x) + k) for x in s])
        corr = np.corrcoef(u0, uk)[0, 1]
        # The old LCG gave +0.36; true independence gives ~0 (SE ~ 1/sqrt(N) = 0.0045).
        assert abs(corr) < 0.015, 'corr(u(s), u(s+%d)) = %+.4f' % (k, corr)
        below = (uk < u0).mean()
        # The old LCG gave 0.12; independence gives 0.5 (SE ~ 0.0022).
        assert abs(below - 0.5) < 0.015, 'P(u(s+%d) < u(s)) = %.4f' % (k, below)


if __name__ == '__main__':
    test_uniform_moments()
    test_neighbouring_indices_are_independent()
    print('RNG guard tests passed: uniform moments + neighbouring-index decorrelation')
