"""Consensus test (spec v0.2 section 5).

For parent i with kNN set K(i) in Z_t:

    d_j = (z_j - z_i)/||z_j - z_i||,   vbar = mean_{l in K(i)} vtilde_l
    g_i = sum_j (vtilde_j - vbar) d_j
    M = sum_j d_j d_j^T,  s = sum_j d_j,  Sigma = M - s s^T / k
    Q_i = g_i^T Sigma^+ g_i / var_null

Implementation note (exact algebra, not an approximation).  Write the k x D
matrix Dm with rows d_j, the centring projector C = I_k - 11^T/k, and
A = C Dm.  Then g_i = A^T vtilde and Sigma = A^T A, so with the thin SVD
A = U S V^T of numerical rank r,  Sigma^+ = V S^-2 V^T  and

    Q_i = ||U^T vtilde||^2 / var_null .

Everything therefore lives in the k x k Gram matrix A A^T -- no D x D matrix is
ever formed, and the null draws pass through the same whitening as the observed
statistic.  Because U has orthonormal columns, Q under the null is exactly
chi^2_r, which is the bug check spec section 5 asks for (figure d).

v0.2 changes:
  * duplicate neighbours (||z_j - z_i|| < DUP_TOL) are dropped and k reduced
    per parent, instead of producing an arbitrary unit direction;
  * the null variance is a parameter, so section 4.2's two variants can both be run:
    (a) var_null = sigma^2                      -- DP term only, as v0.1
    (b) var_null = sigma^2 + N_hat / n_syn      -- plus finite-sample vote noise
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RANK_TOL = 1e-8
DUP_TOL = 1e-8


def knn_indices(Z, k):
    """(n, k) indices of the k nearest *other* points of Z."""
    n = Z.shape[0]
    k = min(k, n - 1)
    sq = (Z**2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.fill_diagonal(d2, np.inf)
    return np.argpartition(d2, kth=k - 1, axis=1)[:, :k]


@dataclass
class GateResult:
    g: np.ndarray  # (n, D) consensus vector
    g_norm: np.ndarray  # (n,)
    Q: np.ndarray  # (n,) test statistic
    rank: np.ndarray  # (n,) numerical rank r_i
    k_eff: np.ndarray  # (n,) neighbours left after dropping duplicates
    p_mc: np.ndarray  # (n,) Monte-Carlo p-value
    p_chi2: np.ndarray  # (n,) chi^2_r fast-path p-value
    knn: np.ndarray  # (n, k)
    dirs: np.ndarray  # (n, k, D) unit directions (zero where invalid)
    dists: np.ndarray  # (n, k) neighbour distances
    valid: np.ndarray  # (n, k) bool, False for dropped duplicates
    proj: np.ndarray  # (n, k) <d_j, ghat_i>
    var_null: float
    Q_null_sample: np.ndarray  # pooled MC null draws (for figure d)
    degenerate_null: bool = False  # var_null == 0: test undefined, rejects all

    def reject(self, alpha, use="mc"):
        p = self.p_mc if use == "mc" else self.p_chi2
        return p < alpha


def consensus(Z, v_tilde, sigma, k=10, B=2000, rng=None, mc=True,
              null_keep=200, var_extra=0.0):
    """g_i, Q_i and p_i for every point of Z.

    var_extra implements spec section 4.2(b): pass N/n_syn to add the
    finite-sample (multinomial) vote variance to the null.  var_extra=0
    reproduces v0.1 behaviour (variant (a)).
    """
    from scipy.stats import chi2

    rng = np.random.default_rng() if rng is None else rng
    n, D = Z.shape
    knn = knn_indices(Z, k)
    k = knn.shape[1]

    diff = Z[knn] - Z[:, None, :]  # (n, k, D)
    dists = np.linalg.norm(diff, axis=-1)
    valid = dists > DUP_TOL
    dirs = np.where(valid[..., None], diff / np.maximum(dists, 1e-300)[..., None], 0.0)

    var_null = float(sigma**2 + var_extra)
    degenerate = var_null <= 0.0

    v_knn = v_tilde[knn]
    g = np.zeros((n, D))
    Q = np.zeros(n)
    rank = np.zeros(n, dtype=int)
    k_eff = valid.sum(axis=1)
    p_mc = np.full(n, np.nan)
    null_pool = []

    for i in range(n):
        idx = np.flatnonzero(valid[i])
        ke = len(idx)
        if ke < 2:  # nothing to compare
            continue
        Di = dirs[i, idx]  # (ke, D)
        y = v_knn[i, idx] - v_knn[i, idx].mean()  # C v, over surviving nbrs
        g[i] = y @ Di

        gram = Di @ Di.T
        gram = gram - gram.mean(axis=0, keepdims=True)
        gram = gram - gram.mean(axis=1, keepdims=True)  # C G C
        w, U = np.linalg.eigh(gram)
        w = np.clip(w, 0.0, None)
        r = int((w > RANK_TOL * w.max()).sum()) if w.max() > 0 else 0
        rank[i] = r
        if r == 0:
            p_mc[i] = 1.0
            continue
        Ur = U[:, -r:]
        proj_y = Ur.T @ y
        if degenerate:
            # Variant (a) with sigma = 0: the null has no spread at all, so
            # any non-zero signal is infinitely significant.  This is the
            # over-rejection spec section 4.2 predicts, not a crash.
            Q[i] = np.inf if proj_y @ proj_y > 0 else 0.0
            p_mc[i] = 0.0 if np.isinf(Q[i]) else 1.0
            continue
        Q[i] = float(proj_y @ proj_y) / var_null
        if mc:
            eps = rng.standard_normal((B, ke)) * np.sqrt(var_null)
            eps = eps - eps.mean(axis=1, keepdims=True)
            qb = ((eps @ Ur) ** 2).sum(axis=1) / var_null
            p_mc[i] = float((qb >= Q[i]).mean())
            if len(null_pool) < null_keep:
                null_pool.append(qb)

    if degenerate:
        p_chi2 = p_mc.copy()
    else:
        p_chi2 = np.where(rank > 0, chi2.sf(Q, np.maximum(rank, 1)), 1.0)
    if not mc and not degenerate:
        p_mc = p_chi2.copy()

    g_norm = np.linalg.norm(g, axis=1)
    ghat = g / np.maximum(g_norm, 1e-12)[:, None]
    proj = np.einsum("nkd,nd->nk", dirs, ghat)
    proj = np.where(valid, proj, np.nan)

    return GateResult(
        g=g, g_norm=g_norm, Q=Q, rank=rank, k_eff=k_eff,
        p_mc=p_mc, p_chi2=p_chi2, knn=knn, dirs=dirs, dists=dists,
        valid=valid, proj=proj, var_null=var_null,
        Q_null_sample=np.concatenate(null_pool) if null_pool else np.empty(0),
        degenerate_null=degenerate,
    )
