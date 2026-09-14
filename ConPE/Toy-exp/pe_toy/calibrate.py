"""Calibrating the variation degree rho (spec is silent on the step size s).

`select` is not variance-preserving.  Resampling proportional to
nearest-neighbour vote counts contracts the population: in D=128, 39% of the
Voronoi cells of a 500-point synthetic set drawn from p itself receive zero
votes, so resampling systematically pulls mass toward the dense core.  A real
sample from p is therefore *not* a fixed point of the PE loop.  Measured
one-round contraction of the per-coordinate variance:

    D =   2  ->  ~0.5%      D = 32  ->  ~2%      D = 128 ->  ~3-4%

Standard PE tolerates this because its variation step is strong.  In the toy
the variation degree is a free parameter, so we set it the way a practitioner
would tune PE: strong enough that the *baseline* arm is stationary at the
target distribution.  Writing c for the one-round variance contraction of the
selection step and using tau ~= rho * sd,

    stationary:  c * var + tau^2 = var   =>   rho* = sqrt(1 - c)

This is estimated from the `blind` arm alone, so it cannot favour a directed
arm, and it is re-estimated per (D, sigma) because sigma changes how peaked
the selection weights are (heavy noise pushes selection toward uniform, which
contracts less).
"""

from __future__ import annotations

import numpy as np

from .pipeline import dp_vote, select


def contraction(P, Z_target, sigma, n_syn, rng, reps=8):
    """One-round variance contraction factor c of the selection step."""
    cs = []
    for _ in range(reps):
        Z = Z_target[rng.choice(len(Z_target), size=n_syn, replace=False)] \
            if len(Z_target) > n_syn else Z_target
        _, v = dp_vote(P, Z, sigma, rng)
        par = select(Z, v, n_syn, rng)
        cs.append(Z[par].var(0).mean() / Z.var(0).mean())
    return float(np.mean(cs))


def calibrate_rho(P, Z_target, sigma, n_syn, seed=0, reps=8, lo=0.02, hi=0.6):
    c = contraction(P, Z_target, sigma, n_syn, np.random.default_rng(seed), reps)
    return float(np.clip(np.sqrt(max(1.0 - c, 1e-6)), lo, hi)), c
