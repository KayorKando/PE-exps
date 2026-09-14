"""Health gate (spec v0.2 section 9.0) -- blocking.

A cell is healthy iff, for every arm in it:

  1. FD_T < FD_0;
  2. FD is non-increasing over the last 3 rounds (one seed-s.d. of slack);
  3. n_eff/n_syn > 0.3 at t = T;
  4. mean pairwise distance in Z_T is within [0.5, 2]x that of a real sample
     of the same size (catches collapse and dispersion alike).

Results from an unhealthy cell are not reported as evidence for or against any
question.  Conditions 1, 3, 4 are per-run and live here; condition 2 needs a
cross-seed s.d. and is evaluated per cell in analyze.py.

`check_health` is also called inside the loop so a diverging run aborts early
rather than producing numbers that would later be discarded.
"""

from __future__ import annotations

import numpy as np

N_EFF_MIN = 0.3
SPREAD_LO, SPREAD_HI = 0.5, 2.0
ABORT_FD_FACTOR = 3.0
ABORT_SPREAD = (0.2, 5.0)
_PAIR_SUB = 400


def mean_pairwise(Z, rng=None, sub=_PAIR_SUB):
    """Mean pairwise distance over a subsample.

    Uses the |a|^2 + |b|^2 - 2ab form: the naive Z[:,None,:] - Z[None,:,:]
    materialises an (sub, sub, D) array -- 983 MB at sub=400, D=768, allocated
    every round, which is what OOM-killed the D=768 stage.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if len(Z) > sub:
        Z = Z[rng.choice(len(Z), sub, replace=False)]
    sq = (Z**2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    d = np.sqrt(np.maximum(d2, 0.0))
    iu = np.triu_indices(len(Z), 1)
    return float(d[iu].mean())


def in_loop(fd_t, fd_0, spread_t, spread_ref, t):
    """Cheap divergence check run every round; returns an abort reason or None."""
    if t < 2:
        return None
    if fd_t > ABORT_FD_FACTOR * fd_0:
        return f"FD blew up at t={t} ({fd_t:.1f} > {ABORT_FD_FACTOR}x FD_0={fd_0:.1f})"
    r = spread_t / max(spread_ref, 1e-12)
    if not (ABORT_SPREAD[0] < r < ABORT_SPREAD[1]):
        return f"spread ratio {r:.2f} left {ABORT_SPREAD} at t={t}"
    return None


def per_run(rows, spread_ref, n_syn):
    """Conditions 1, 3 and 4 for a completed run."""
    fd0, fdT = rows[0]["frechet"], rows[-1]["frechet"]
    n_eff = rows[-1].get("n_eff_select", np.nan) / n_syn
    ratio = rows[-1].get("spread", np.nan) / max(spread_ref, 1e-12)
    checks = {
        "h1_fd_improved": bool(fdT < fd0),
        "h3_n_eff": bool(n_eff > N_EFF_MIN),
        "h4_spread": bool(SPREAD_LO <= ratio <= SPREAD_HI),
    }
    return {
        **checks,
        "fd_0": fd0,
        "fd_T": fdT,
        "n_eff_frac": float(n_eff),
        "spread_ratio": float(ratio),
        "healthy_run": all(checks.values()),
    }


def cell_condition2(fd_curves, slack=None):
    """Condition 2: FD non-increasing over the last 3 rounds, mean over seeds,
    with one seed-s.d. of slack."""
    A = np.asarray(fd_curves, float)
    if A.ndim == 1:
        A = A[None]
    m = A.mean(0)
    sd = A.std(0, ddof=1).mean() if len(A) > 1 else 0.0
    slack = sd if slack is None else slack
    tail = m[-3:]
    return bool(all(b <= a + slack for a, b in zip(tail, tail[1:])))
