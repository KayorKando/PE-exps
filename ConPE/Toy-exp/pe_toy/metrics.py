"""Metrics (spec section 7).

7.1 transferable (survive the move to text / images):
    1. Frechet distance in embedding space
    2. precision / recall / coverage  (Kynkaanniemi et al. 2019, k=5;
       coverage as in Naeem et al. 2020)
    3. recall per t=0 vote-count quintile  (the direct test of Q5)
    4. queries-to-target                    (computed in analyze.py)

7.2 toy-only (config A, ground truth available):
    5. cos(g_hat, grad log(p/q_t))  split by gated / not
    6. gate false-positive rate and power
    7. fraction of children landing at higher p-density than their parent

The metrics module is the only place allowed to touch P, the true densities,
or the noise-free votes.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

CHUNK = 512


def _pairwise(A, B):
    a2 = (A**2).sum(1)[:, None]
    b2 = (B**2).sum(1)[None, :]
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _kth_nn_radius(X, k):
    """Distance from each row of X to its k-th nearest *other* row."""
    n = X.shape[0]
    out = np.empty(n)
    for i in range(0, n, CHUNK):
        d = _pairwise(X[i : i + CHUNK], X)
        d[np.arange(d.shape[0]), np.arange(i, min(i + CHUNK, n))] = np.inf
        out[i : i + CHUNK] = np.partition(d, kth=k - 1, axis=1)[:, k - 1]
    return out


def frechet_distance(X, Y, eps=1e-6):
    """||mu_x - mu_y||^2 + Tr(Sx + Sy - 2 (Sx Sy)^{1/2})  (the FID formula)."""
    mx, my = X.mean(0), Y.mean(0)
    Sx = np.cov(X, rowvar=False)
    Sy = np.cov(Y, rowvar=False)
    diff = mx - my
    covmean, _ = linalg.sqrtm(Sx @ Sy, disp=False)
    if not np.isfinite(covmean).all():
        off = eps * np.eye(Sx.shape[0])
        covmean = linalg.sqrtm((Sx + off) @ (Sy + off), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(Sx) + np.trace(Sy) - 2.0 * np.trace(covmean))


class Evaluator:
    """Holds everything about the reference set that does not change per round."""

    def __init__(self, reference, Z0, v_true_0, k_prdc=5, n_buckets=5, rng=None):
        self.ref = reference
        self.k = k_prdc
        self.ref_radius = _kth_nn_radius(reference, k_prdc)

        # Bucket reference points by the t=0 vote count of their nearest
        # synthetic point (spec 7.1.3).  Noise-free counts: metrics may see them.
        nearest0 = np.argmin(_pairwise(reference, Z0), axis=1)
        self.ref_vote0 = v_true_0[nearest0]
        # Rank-based quintiles; ties (many zero-vote cells) are broken by rank
        # order so the buckets stay equal-sized.
        order = np.argsort(np.argsort(self.ref_vote0, kind="stable"), kind="stable")
        self.bucket = np.minimum(
            (order * n_buckets) // len(order), n_buckets - 1
        )
        self.n_buckets = n_buckets
        self.bucket_vote_range = [
            (
                float(self.ref_vote0[self.bucket == b].min()),
                float(self.ref_vote0[self.bucket == b].max()),
            )
            for b in range(n_buckets)
        ]

    def __call__(self, Z):
        out = {"frechet": frechet_distance(Z, self.ref)}

        fake_radius = _kth_nn_radius(Z, self.k)
        d = _pairwise(self.ref, Z)  # (n_ref, n_fake)

        # precision: fake points inside the reference manifold
        out["precision"] = float((d <= self.ref_radius[:, None]).any(axis=0).mean())
        # recall: reference points inside the fake manifold
        in_fake = (d <= fake_radius[None, :]).any(axis=1)
        out["recall"] = float(in_fake.mean())
        # coverage: reference points whose own k-NN ball contains a fake point
        out["coverage"] = float(
            (d.min(axis=1) <= self.ref_radius).mean()
        )

        # Kynkaanniemi recall saturates at 1.0 whenever the fake set is
        # over-dispersed (its k-NN radii swallow everything) -- which is
        # exactly what q_0 is here.  Naeem coverage does not, so Q5 is read
        # off coverage and recall is reported alongside it.
        covered = d.min(axis=1) <= self.ref_radius
        for b in range(self.n_buckets):
            m = self.bucket == b
            out[f"recall_q{b + 1}"] = float(in_fake[m].mean())
            out[f"coverage_q{b + 1}"] = float(covered[m].mean())
        return out


def reference_ceiling(evaluator, mixture_or_pool, n, rng):
    """Metrics for a *real* sample of size n -- the best any arm could do.

    Frechet distance between two finite samples is biased away from 0, so the
    floor is not 0 and has to be measured to read the numbers.
    """
    if hasattr(mixture_or_pool, "sample"):
        S = mixture_or_pool.sample(n, rng)
    else:
        pool = mixture_or_pool
        S = pool[rng.choice(len(pool), size=n, replace=False)]
    return evaluator(S)


# --------------------------------------------------------------------------
# toy-only diagnostics
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Direction quality (spec v0.2 section 7.2) -- metrics 5, 5b, 7
# --------------------------------------------------------------------------


def kde_bandwidth_cv(Z, grid=None, folds=5, rng=None):
    """Bandwidth by k-fold CV on held-out log-likelihood (spec 6.4)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n, D = Z.shape
    nn = _kth_nn_radius(Z, 1)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    # duplicate parents are common, so the median NN distance can be 0
    med = float(np.median(nn)) if len(nn) else 1.0
    if med <= 0:
        med = float(Z.std(axis=0).mean()) or 1.0
    if grid is None:
        grid = med * np.geomspace(0.25, 8.0, 12)
    idx = rng.permutation(n)
    parts = np.array_split(idx, folds)
    best, best_ll = grid[0], -np.inf
    for h in grid:
        ll = 0.0
        for f in range(folds):
            te = parts[f]
            tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            d2 = _pairwise(Z[te], Z[tr]) ** 2
            lk = -0.5 * d2 / h**2 - D * np.log(h)
            m = lk.max(axis=1, keepdims=True)
            ll += float(np.sum(m[:, 0] + np.log(np.exp(lk - m).sum(axis=1))))
        if ll > best_ll:
            best, best_ll = h, ll
    return float(best)


def score_ratio(Z, mixture, q_law=None, Z_pop=None, h=None):
    """grad log (p/q) at each row of Z.

    q is the *analytic* law of the population when one is available (t=0), and
    a Gaussian KDE otherwise.  Run #1 read this quantity through a KDE only,
    which is what spec v0.2 D2 flags as unreliable.
    """
    if q_law is not None:
        return mixture.score(Z) - q_law.score(Z)
    return mixture.score(Z) - kde_score(Z, Z_pop, h)


def _cos(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    c = np.full(len(a), np.nan)
    c[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
    return c


def geometry_corrected(gate):
    """Metric 5b: M_w^+ g_i with M_w = sum_j ||z_j - z_i|| d_j d_j^T.

    For a locally linear vote field g_i ~ M_w grad v, so M_w^+ g_i removes the
    part of the direction error that comes from an anisotropic neighbourhood
    rather than from vote noise.
    """
    n, k, D = gate.dirs.shape
    out = np.zeros((n, D))
    for i in range(n):
        idx = np.flatnonzero(gate.valid[i])
        if len(idx) < 2:
            continue
        Di = gate.dirs[i, idx]
        Mw = (Di * gate.dists[i, idx, None]).T @ Di  # (D, D)
        out[i] = np.linalg.pinv(Mw, rcond=1e-8) @ gate.g[i]
    return out


def permutation_null(gate, v_tilde, target, n_perm=200, rng=None, mask=None):
    """Shuffle v_tilde across each parent's neighbours and recompute g.

    Gives the null distribution of the *mean* cosine, which is the statistic
    the Q0 gate is read against.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    v_knn = v_tilde[gate.knn]
    n, k = v_knn.shape
    mask = np.ones(n, bool) if mask is None else mask
    out = np.empty(n_perm)
    for b in range(n_perm):
        perm = np.argsort(rng.random((n, k)), axis=1)
        vp = np.take_along_axis(v_knn, perm, axis=1)
        vp = np.where(gate.valid, vp, np.nan)
        y = vp - np.nanmean(vp, axis=1, keepdims=True)
        y = np.nan_to_num(y)
        gp = np.einsum("nk,nkd->nd", y, gate.dirs)
        out[b] = np.nanmean(_cos(gp, target)[mask])
    return out


def direction_quality(gate, Z, target, D, v_tilde=None, n_perm=200, rng=None,
                      keep_quantile=0.5):
    """Metrics 5 and 5b with their null baselines (spec 7.2).

    Restricted to points where the reference vector is actually present:
    ||grad log(p/q)|| above `keep_quantile`.  Reporting the cosine where the
    reference is ~0 is what made run #1's metric 5 uninformative.
    """
    tn = np.linalg.norm(target, axis=1)
    thr = np.quantile(tn, keep_quantile)
    keep = tn >= thr
    cos5 = _cos(gate.g, target)
    cos5b = _cos(geometry_corrected(gate), target)
    out = {
        "cos5": float(np.nanmean(cos5[keep])),
        "cos5_all": float(np.nanmean(cos5)),
        "cos5b": float(np.nanmean(cos5b[keep])),
        "excluded_frac": float(1.0 - keep.mean()),
        "grad_norm_p10": float(np.percentile(tn, 10)),
        "grad_norm_p50": float(np.percentile(tn, 50)),
        "grad_norm_p90": float(np.percentile(tn, 90)),
        "random_baseline": float(np.sqrt(2.0 / (np.pi * D))),
        "n_kept": int(keep.sum()),
    }
    if v_tilde is not None:
        pn = permutation_null(gate, v_tilde, target, n_perm=n_perm, rng=rng,
                              mask=keep)
        out["perm_mean"] = float(pn.mean())
        out["perm_sd"] = float(pn.std(ddof=1))
        out["perm_z"] = float((out["cos5"] - pn.mean()) / max(pn.std(ddof=1), 1e-12))
        out["perm_p"] = float((pn >= out["cos5"]).mean())
    return out


def metric7_votefield(child, parents, Z, v_true):
    """Fraction of children landing at higher p/q than their parent.

    p/q is proxied by the noise-free vote field: cell j collects
    N * integral_{cell j} p, and the cell volume scales as 1/(n_syn q), so
    v_j ~ N/n_syn * p(z_j)/q(z_j).  Both parent and child are scored in the
    same fixed partition Z_t, so the comparison is well posed at any D.
    """
    d = _pairwise(child, Z)
    v_child = v_true[np.argmin(d, axis=1)]
    v_par = v_true[parents]
    hi = float((v_child > v_par).mean())
    lo = float((v_child < v_par).mean())
    return {
        "m7_higher": hi,
        "m7_lower": lo,
        "m7_tie": float(1.0 - hi - lo),
        "m7_ratio": float(hi / (hi + lo)) if (hi + lo) > 0 else float("nan"),
    }


def metric7_analytic(child, parents, Z, mixture, h):
    """Same quantity with p analytic and q a KDE -- trustworthy at D=2 only."""
    lr_c = mixture.log_prob(child) - _kde_logprob(child, Z, h)
    lr_p = mixture.log_prob(Z[parents]) - _kde_logprob(Z[parents], Z, h)
    return float((lr_c > lr_p).mean())


def _kde_logprob(X, Z, h):
    D = Z.shape[1]
    d2 = _pairwise(X, Z) ** 2
    lk = -0.5 * d2 / h**2 - D * np.log(h)
    m = lk.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(lk - m).sum(axis=1))) - np.log(len(Z))


def kde_score(Z_query, Z_pop, h):
    d = Z_query[:, None, :] - Z_pop[None, :, :]
    logk = -0.5 * (d**2).sum(-1) / h**2
    logk -= logk.max(axis=1, keepdims=True)
    w = np.exp(logk)
    w /= w.sum(axis=1, keepdims=True)
    return -np.einsum("qn,qnd->qd", w, d) / h**2


def cos_to_truth(g, Z, mixture, h):
    """cos(g_hat_i, grad log(p/q_t)(z_i)) for every point of Z (metric 5)."""
    target = mixture.score(Z) - kde_score(Z, Z, h)
    tn = np.linalg.norm(target, axis=1)
    gn = np.linalg.norm(g, axis=1)
    ok = (tn > 1e-12) & (gn > 1e-12)
    c = np.full(len(Z), np.nan)
    c[ok] = (g[ok] * target[ok]).sum(1) / (gn[ok] * tn[ok])
    return c


def noncentrality(v_true, gate, sigma):
    """lambda_i = ||whitened true signal||^2 / sigma^2 -- how non-flat the true
    vote profile around parent i really is.  Used to split parents into
    "flat" and "non-flat" for the gate FPR / power split (metric 6)."""
    y = v_true[gate.knn]
    y = y - y.mean(axis=1, keepdims=True)
    lam = np.empty(len(y))
    gram = np.einsum("nkd,nld->nkl", gate.dirs, gate.dirs)
    gram = gram - gram.mean(axis=1, keepdims=True)
    gram = gram - gram.mean(axis=2, keepdims=True)
    for i in range(len(y)):
        w, U = np.linalg.eigh(gram[i])
        w = np.clip(w, 0.0, None)
        r = int((w > 1e-8 * w.max()).sum()) if w.max() > 0 else 0
        if r == 0:
            lam[i] = 0.0
            continue
        Ur = U[:, -r:]
        lam[i] = float(((Ur.T @ y[i]) ** 2).sum()) / sigma**2
    return lam


def gate_fpr_power(reject, lam, flat_thresh=1.0):
    flat = lam < flat_thresh
    return (
        float(reject[flat].mean()) if flat.any() else np.nan,
        float(reject[~flat].mean()) if (~flat).any() else np.nan,
        float(flat.mean()),
    )
