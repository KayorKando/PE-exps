"""The swappable variation step (spec v0.2 section 6).

Frozen signature, every arm:

    vary(parents, Z_t, v_tilde, cfg) -> Z_{t+1}

`parents` is the integer index array from `select` (indices into Z_t, with
repeats).  One child per parent, so the generator-call budget is identical
across arms.

v0.2 changes
------------
section 6.0  Exploration noise is **manifold-local**, not isotropic.  In D>=128 an
        isotropic step is a random walk off the data manifold and was a main
        cause of the run #1 divergence (D1); an LLM rewrite, by contrast,
        stays on the text manifold by construction.  For each parent we take
        the sample covariance C_i of its k_cov=20 nearest neighbours,
        regularise it as C_i + lambda*tr(C_i)/D*I, trace-normalise to D, and
        draw xi ~ N(0, tau^2 C~_i).

        Step scale s = 0.25 * median nearest-neighbour distance in Z_t, and
        the **median displacement is rescaled to exactly s for every arm**, so
        the movement budget is equal by construction rather than by a
        variance argument.

section 6.6  Control arms `random_pair` and `centroid`.  A result is attributable
        to consensus only if `contrastive` beats both.

Privacy rule (section 2): nothing here reads P, true densities, or noise-free
votes.  Enforced mechanically -- those live behind `cfg.oracle`, which is None
unless the arm is `directed_truth`, the declared config-A-only ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEGENERATE_G = 1e-9
K_COV = 20
COV_LAMBDA = 0.1


@dataclass
class OracleAccess:
    """Ground truth. Only `directed_truth` (a diagnostic) may hold one."""

    mixture: object
    bandwidth: float | None = None


@dataclass
class VaryContext:
    sigma: float
    rng: np.random.Generator
    k: int = 10
    alpha: float = 0.05
    theta: float = 0.0
    theta_mode: str = "fixed"  # "fixed" | "uniform"
    step_frac: float = 0.25  # s = step_frac * nn_ref
    s_ref: float | None = None  # absolute NN scale; None = adaptive (v0.2 literal)
    jitter_frac: float = 0.10
    noise_model: str = "manifold"  # "manifold" | "pca" | "isotropic"
    pca_m: int = 20
    oracle: OracleAccess | None = None
    gate: object | None = None
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def nn_scale(Z):
    """Median nearest-neighbour distance in Z."""
    sq = (Z**2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(np.maximum(d2.min(axis=1), 0.0))))


def step_scale(Z, step_frac, s_ref=None):
    """Step length s = step_frac * nn_ref.

    spec 6.0 sets nn_ref = median NN distance of Z_t, i.e. re-measured every
    round.  That rule is self-extinguishing: the step is what holds points
    apart, so a smaller step gives closer points gives a smaller step.  Run #2
    measured s falling from 4.2e-2 to exactly 0 by round 7 at D=2 (over half
    the population becoming exact duplicates), for every arm.  Keying nn_ref
    to Z_0 instead -- a fixed, non-private scale, measured once -- removes the
    feedback and keeps the step meaningful for the whole budget.  Pass
    s_ref=None to reproduce the spec-literal adaptive rule.
    """
    return float(step_frac * (nn_scale(Z) if s_ref is None else s_ref))


class LocalCov:
    """Sampler for xi ~ N(0, C~_i), the trace-normalised local covariance.

    C_i is the sample covariance of z_i's k_cov nearest neighbours, so with
    X_i the (k_cov, D) centred neighbour block,

        C_i    = X_i^T X_i / (k_cov - 1)          -- rank <= k_cov
        C_reg  = C_i + lam * tr(C_i)/D * I
        C~_i   = C_reg * D / tr(C_reg)            -- trace(C~_i) = D

    Forming C_i explicitly costs O(n D^2): at n=500, D=768 that is 2.4 GB for
    the covariances and as much again for their Cholesky factors, which is what
    OOM-killed the first D=768 attempt.  C_i is only rank k_cov, so we never
    form it -- we sample through the factors instead:

        xi_raw = X_i^T e / sqrt(k_cov - 1) + sqrt(lam * tr(C_i)/D) * g
        xi     = xi_raw * sqrt(D / (tr(C_i) * (1 + lam)))

    with e ~ N(0, I_k), g ~ N(0, I_D).  Cov(xi_raw) = C_reg exactly, so this is
    the same distribution, at O(n k_cov D) memory (61 MB at D=768).
    """

    def __init__(self, Z, k_cov=K_COV, lam=COV_LAMBDA):
        n, D = Z.shape
        self.D, self.lam = D, lam
        k_cov = min(k_cov, n - 1)
        sq = (Z**2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
        np.fill_diagonal(d2, np.inf)
        nb = np.argpartition(d2, kth=k_cov - 1, axis=1)[:, :k_cov]
        X = Z[nb] - Z[nb].mean(axis=1, keepdims=True)  # (n, k_cov, D)
        self.X = X / np.sqrt(max(k_cov - 1, 1))
        self.k_cov = k_cov
        self.tr = np.maximum((X**2).sum(axis=(1, 2)) / max(k_cov - 1, 1), 1e-300)

    def draw(self, parents, rng):
        Xp = self.X[parents]  # (m, k_cov, D)
        m = len(parents)
        e = rng.standard_normal((m, self.k_cov))
        xi = np.einsum("mk,mkd->md", e, Xp)
        iso = np.sqrt(self.lam * self.tr[parents] / self.D)[:, None]
        xi = xi + iso * rng.standard_normal((m, self.D))
        scale = np.sqrt(self.D / (self.tr[parents] * (1.0 + self.lam)))
        return xi * scale[:, None]


def _pca_basis(Z, m):
    Zc = Z - Z.mean(0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    m = min(m, Vt.shape[0])
    scale = S[:m] / max(S[:m].max(), 1e-300)
    return Vt[:m], scale


class NoiseModel:
    """Draws exploration noise for a whole round (spec 6.0)."""

    def __init__(self, Z, cfg):
        self.kind = cfg.noise_model
        self.D = Z.shape[1]
        if self.kind == "manifold":
            self.cov = LocalCov(Z)
        elif self.kind == "pca":
            self.V, self.scale = _pca_basis(Z, cfg.pca_m)
        elif self.kind != "isotropic":
            raise ValueError(cfg.noise_model)

    def draw(self, parents, rng):
        n = len(parents)
        if self.kind == "manifold":
            return self.cov.draw(parents, rng)
        if self.kind == "pca":
            m = self.V.shape[0]
            e = rng.standard_normal((n, m)) * self.scale
            xi = e @ self.V
            return xi * np.sqrt(self.D / max((self.scale**2).sum(), 1e-300))
        return rng.standard_normal((n, self.D))


def _rescale_to_median(disp, s):
    """Force the median displacement length to exactly s (equal budget)."""
    nrm = np.linalg.norm(disp, axis=1)
    med = float(np.median(nrm))
    if med < 1e-300:
        return disp
    return disp * (s / med)


def _perp(axis, rng):
    u = rng.standard_normal(axis.shape)
    u -= (u * axis).sum(1, keepdims=True) * axis
    nrm = np.linalg.norm(u, axis=1, keepdims=True)
    bad = nrm[:, 0] < 1e-12
    if bad.any():
        u[bad] = rng.standard_normal((int(bad.sum()), axis.shape[1]))
        u[bad] -= (u[bad] * axis[bad]).sum(1, keepdims=True) * axis[bad]
        nrm = np.linalg.norm(u, axis=1, keepdims=True)
    return u / np.maximum(nrm, 1e-12)


def _kde_score(Zq, Zp, h):
    d = Zq[:, None, :] - Zp[None, :, :]
    logk = -0.5 * (d**2).sum(-1) / h**2
    logk -= logk.max(axis=1, keepdims=True)
    w = np.exp(logk)
    w /= w.sum(axis=1, keepdims=True)
    return -np.einsum("qn,qnd->qd", w, d) / h**2


def _thetas(cfg, n):
    if cfg.theta_mode == "uniform":
        return cfg.rng.uniform(0.0, cfg.theta, size=n)
    return np.full(n, cfg.theta)


def _gate_mask(cfg, parents, gated):
    if not gated:
        return np.ones(len(parents), dtype=bool)
    return cfg.gate.p_mc[parents] < cfg.alpha


def _measure_theta(disp, axis, mask):
    """Empirical angle between realised displacement and the suggested axis --
    the quantity spec section 10 requires the LLM stage to measure."""
    if not mask.any():
        return np.nan
    d = disp[mask]
    nrm = np.linalg.norm(d, axis=1)
    ok = nrm > 1e-12
    if not ok.any():
        return np.nan
    c = (d[ok] * axis[mask][ok]).sum(1) / nrm[ok]
    return float(np.median(np.arccos(np.clip(c, -1.0, 1.0))))


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def blind(parents, Z, v_tilde, cfg):
    """Spec 6.1 -- standard PE: undirected manifold-local variation."""
    s = step_scale(Z, cfg.step_frac, cfg.s_ref)
    nm = NoiseModel(Z, cfg)
    disp = _rescale_to_median(nm.draw(parents, cfg.rng) / np.sqrt(Z.shape[1]), s)
    cfg.stats.update(step_s=s, directed_frac=0.0,
                     median_disp=float(np.median(np.linalg.norm(disp, axis=1))))
    return Z[parents] + disp


def _axis_for(kind, parents, Z, cfg, gr, n, D):
    """Unit axis per parent, plus bookkeeping."""
    axis = np.zeros((n, D))
    info = dict(one_sided_frac=0.0, axis_flip_frac=0.0)
    rows = np.arange(n)

    if kind == "centroid":  # 6.6, ignores votes
        nb = gr.knn[parents]
        w = gr.valid[parents].astype(float)
        cen = (Z[nb] * w[..., None]).sum(1) / np.maximum(w.sum(1), 1)[:, None]
        a = cen - Z[parents]
        nrm = np.linalg.norm(a, axis=1)
        usable = nrm > DEGENERATE_G
        axis[usable] = a[usable] / nrm[usable, None]
        return axis, usable, info

    if kind == "random_pair":  # 6.6, ignores votes
        usable = np.zeros(n, bool)
        for i in range(n):
            idx = np.flatnonzero(gr.valid[parents[i]])
            if len(idx) < 2:
                continue
            jp, jm = cfg.rng.choice(idx, size=2, replace=False)
            a = Z[gr.knn[parents[i], jp]] - Z[gr.knn[parents[i], jm]]
            nr = np.linalg.norm(a)
            if nr > DEGENERATE_G:
                axis[i] = a / nr
                usable[i] = True
        return axis, usable, info

    if kind == "truth":  # 6.4, config A + D=2 only
        h = cfg.oracle.bandwidth or nn_scale(Z)
        zp = Z[parents]
        a = cfg.oracle.mixture.score(zp) - _kde_score(zp, Z, h)
        nrm = np.linalg.norm(a, axis=1)
        usable = nrm > DEGENERATE_G
        axis[usable] = a[usable] / nrm[usable, None]
        return axis, usable, info

    ghat = gr.g / np.maximum(gr.g_norm, 1e-12)[:, None]
    usable = gr.g_norm[parents] > DEGENERATE_G

    if kind == "oracle":  # 6.3
        axis[usable] = ghat[parents][usable]
        return axis, usable, info

    if kind == "contrastive":  # 6.2
        proj = gr.proj[parents]  # NaN where the neighbour was dropped
        jp = np.nanargmax(np.where(np.isnan(proj), -np.inf, proj), axis=1)
        jm = np.nanargmin(np.where(np.isnan(proj), np.inf, proj), axis=1)
        zp = Z[gr.knn[parents][rows, jp]]
        zm = Z[gr.knn[parents][rows, jm]]
        a = zp - zm
        nrm = np.linalg.norm(a, axis=1)
        with np.errstate(invalid="ignore"):
            one_sided = (np.nanmin(proj, axis=1) > 0) | (np.nanmax(proj, axis=1) < 0)
        fall_back = (one_sided | (nrm < DEGENERATE_G)) & usable
        ok = usable & ~fall_back
        axis[ok] = a[ok] / nrm[ok, None]
        axis[fall_back] = ghat[parents][fall_back]
        dot = (axis * ghat[parents]).sum(1)
        flip = ok & (dot < 0)
        axis[flip] *= -1.0
        info = dict(one_sided_frac=float(fall_back.mean()),
                    axis_flip_frac=float(flip.mean()))
        return axis, usable, info

    raise ValueError(kind)


def _directed(parents, Z, v_tilde, cfg, kind, gated):
    D = Z.shape[1]
    n = len(parents)
    s = step_scale(Z, cfg.step_frac, cfg.s_ref)
    gr = cfg.gate
    axis, usable, info = _axis_for(kind, parents, Z, cfg, gr, n, D)

    directed = _gate_mask(cfg, parents, gated) & usable
    theta = _thetas(cfg, n)
    nm = NoiseModel(Z, cfg)
    # E||xi||^2 = trace(C~) = D, so xi/sqrt(D) has unit expected length and is
    # directly comparable to the unit axis vector.  Without this the jitter
    # scales as sqrt(D) and swamps the axis (realised theta ~47deg at D=128
    # for a nominal theta=0), and gated parents -- who get pure xi -- would
    # move sqrt(D) times further than directed ones after the global rescale.
    xi = nm.draw(parents, cfg.rng) / np.sqrt(D)

    # unit-scale raw displacement, then one global rescale so that the median
    # displacement equals s -- identical budget for every arm
    disp = cfg.jitter_frac * xi
    if directed.any():
        m = directed
        a = axis[m]
        r = _perp(a, cfg.rng)
        th = theta[m][:, None]
        disp[m] = disp[m] + (np.cos(th) * a + np.sin(th) * r)
    if (~directed).any():
        disp[~directed] = xi[~directed]
    disp = _rescale_to_median(disp, s)

    cfg.stats.update(
        step_s=s,
        directed_frac=float(directed.mean()),
        degenerate_frac=float((~usable).mean()),
        median_disp=float(np.median(np.linalg.norm(disp, axis=1))),
        measured_theta=_measure_theta(disp, axis, directed),
        **info,
    )
    cfg.stats["_axis"] = axis
    cfg.stats["_directed"] = directed
    return Z[parents] + disp


DIRECTED_KINDS = ("contrastive", "oracle", "truth", "random_pair", "centroid")


def make_arm(name):
    """Resolve an arm name to a `vary` function with the frozen signature."""
    if name == "blind":
        blind.needs_gate = False
        blind.needs_oracle = False
        blind.kind = "blind"
        return blind
    kind, _, mode = name.partition("-")
    mode = mode or "always"
    if kind not in DIRECTED_KINDS:
        raise ValueError(f"unknown arm {name!r}")
    if mode not in ("always", "gated"):
        raise ValueError(f"unknown gating mode {mode!r}")
    gated = mode == "gated"

    def arm(parents, Z, v_tilde, cfg, _k=kind, _g=gated):
        return _directed(parents, Z, v_tilde, cfg, _k, _g)

    arm.__name__ = f"vary_{name.replace('-', '_')}"
    # every directed arm needs the kNN structure; only vote-using arms need g
    arm.needs_gate = True
    arm.needs_oracle = kind == "truth"
    arm.uses_votes = kind in ("contrastive", "oracle")
    arm.kind = kind
    return arm


ARMS = (
    "blind",
    "random_pair",
    "centroid",
    "contrastive-always",
    "contrastive-gated",
    "oracle-always",
    "oracle-gated",
    "truth-always",
)
