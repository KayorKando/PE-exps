"""The PE loop (spec v0.2 section 2).

    Z_0        = init(n_syn)
    for t in 1..T:
        v_tilde = dp_vote(P, Z_t, sigma)        # + finite-sample noise, section 4.2
        parents = select(Z_t, v_tilde, n_syn)   # vote floor, section 4.3
        Z_{t+1} = vary(parents, Z_t, v_tilde, cfg)
        health  = check_health(Z_{t+1}, Z_0, t) # section 9.0, aborts on divergence
        log_metrics(Z_{t+1}, t)

`dp_vote` and `select` are identical across arms.  P is touched only inside
`dp_vote` and inside the metrics module.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from . import health as H
from . import metrics as M
from .data import make_dataset
from .gate import consensus
from .privacy import derive_sigma
from .vary import OracleAccess, VaryContext, make_arm, nn_scale

CHUNK = 2048


def dp_vote(P, Z, sigma, rng):
    """Each private point votes for its nearest synthetic point; Gaussian noise
    per count.  Returns (noise-free counts, noisy counts).

    The noise-free counts go to the metrics module only -- VaryContext carries
    v_tilde and nothing else.

    Note (spec 4.2): even at sigma = 0 the counts are a multinomial draw over N
    private points, so sd(v_j) ~ sqrt(N/n_syn).  "Noise-free" means sigma = 0,
    not noiseless.
    """
    n = Z.shape[0]
    nearest = np.empty(len(P), dtype=int)
    zsq = (Z**2).sum(1)
    for i in range(0, len(P), CHUNK):
        blk = P[i : i + CHUNK]
        d2 = (blk**2).sum(1)[:, None] + zsq[None, :] - 2.0 * (blk @ Z.T)
        nearest[i : i + CHUNK] = np.argmin(d2, axis=1)
    v_true = np.bincount(nearest, minlength=n).astype(float)
    return v_true, v_true + rng.normal(0.0, sigma, size=n)


def vote_floor(sigma, survival=0.10):
    """Aug-PE's threshold kappa (spec 4.3).

    Interpretation note: the spec says "the 10th percentile of the null vote
    distribution", but a 10th percentile would let 90% of noise-only points
    through, the opposite of the stated purpose ("stop noise-only points from
    being resampled").  Implemented as the threshold that lets through
    `survival` = 10% of noise-only points, i.e. the 90th percentile of the null
    N(0, sigma^2).  With sigma = 0 the floor is 0 and inert.
    """
    from scipy.stats import norm

    return float(sigma * norm.ppf(1.0 - survival)) if sigma > 0 else 0.0


def select(Z, v_tilde, n_syn, rng, kappa=0.0):
    """Resample parents proportional to max(v_tilde, 0), after the vote floor."""
    w = np.where(v_tilde >= kappa, v_tilde, 0.0)
    w = np.maximum(w, 0.0)
    tot = w.sum()
    p = None if tot <= 0 else w / tot
    return rng.choice(len(Z), size=n_syn, replace=True, p=p)


@dataclass
class RunConfig:
    arm: str = "blind"
    config: str = "A"
    D: int = 128
    N: int = 20000
    n_syn: int = 500
    n_ref: int = 2000
    T: int = 10
    eps: float = 1.0
    delta: float | None = None  # default 1/N
    accountant: str = "rdp"
    var_variant: str = "b"  # spec 4.2: "a" = sigma^2 only, "b" = + N/n_syn
    vote_floor_survival: float = 0.10
    noise_model: str = "manifold"  # spec 6.0
    step_frac: float = 0.25
    step_ref: str = "z0"  # "z0" (fixed) | "adaptive" (spec 6.0 literal)
    jitter_frac: float = 0.10
    k: int = 10
    alpha: float = 0.05
    theta_deg: float = 0.0
    theta_mode: str = "fixed"
    mc_B: int = 2000
    use_mc_pvalue: bool = True
    seed: int = 0
    corpus: str = "yelp"

    def key(self):
        return "|".join(f"{k}={v}" for k, v in sorted(asdict(self).items()))


def run(cfg: RunConfig, verbose=False):
    t_start = time.time()
    rng = np.random.default_rng(10_000 + cfg.seed)
    delta = cfg.delta if cfg.delta is not None else 1.0 / cfg.N

    if np.isinf(cfg.eps):
        sigma = 0.0
        spend = dict(eps=np.inf, delta=delta, T=cfg.T, sigma=0.0,
                     accountant="none", rdp_order=np.nan,
                     sigma_rdp=0.0, sigma_analytic=0.0)
    else:
        sp = derive_sigma(cfg.eps, delta, cfg.T, cfg.accountant)
        sigma, spend = sp.sigma, sp.as_dict()

    ds = make_dataset(config=cfg.config, D=cfg.D, N=cfg.N, n_syn=cfg.n_syn,
                      n_ref=cfg.n_ref, seed=cfg.seed, corpus=cfg.corpus)
    Z = ds.Z0.copy()
    D = cfg.D

    # spec 4.2: the finite-sample term is part of the noise, not a nuisance
    var_extra = (cfg.N / cfg.n_syn) if cfg.var_variant == "b" else 0.0
    kappa = vote_floor(sigma, cfg.vote_floor_survival)

    v_true_0, _ = dp_vote(ds.P, Z, 0.0, rng)
    ev = M.Evaluator(ds.reference, Z, v_true_0, k_prdc=5)
    vote_q0 = np.clip(
        np.searchsorted(np.quantile(v_true_0, [0.2, 0.4, 0.6, 0.8]), v_true_0), 0, 4
    )

    arm_fn = make_arm(cfg.arm)
    is_truth = getattr(arm_fn, "needs_oracle", False)
    if is_truth and ds.mixture is None:
        raise ValueError("truth arm requires config A")

    ref_pool = (ds.mixture.sample(cfg.n_syn, np.random.default_rng(999))
                if ds.mixture is not None
                else ds.P[np.random.default_rng(999).choice(len(ds.P), cfg.n_syn,
                                                            replace=False)])
    ceiling = ev(ref_pool)
    spread_ref = H.mean_pairwise(ref_pool)

    bandwidth = None
    if is_truth:
        bandwidth = M.kde_bandwidth_cv(Z, rng=np.random.default_rng(7))

    snr = float(v_true_0.std() / np.sqrt(sigma**2 + cfg.N / cfg.n_syn))
    s_ref = nn_scale(Z) if cfg.step_ref == "z0" else None

    row0 = dict(round=0, queries=0, **ev(Z))
    row0["spread"] = H.mean_pairwise(Z)
    rows = [row0]
    abort = None

    for t in range(1, cfg.T + 1):
        v_true, v_tilde = dp_vote(ds.P, Z, sigma, rng)
        parents = select(Z, v_tilde, cfg.n_syn, rng, kappa=kappa)

        vcfg = VaryContext(
            sigma=sigma, rng=rng, k=cfg.k, alpha=cfg.alpha,
            theta=np.deg2rad(cfg.theta_deg), theta_mode=cfg.theta_mode,
            step_frac=cfg.step_frac, s_ref=s_ref, jitter_frac=cfg.jitter_frac,
            noise_model=cfg.noise_model,
            oracle=OracleAccess(ds.mixture, bandwidth) if is_truth else None,
        )
        gr = consensus(Z, v_tilde, sigma, k=cfg.k, B=cfg.mc_B, rng=rng,
                       mc=cfg.use_mc_pvalue, var_extra=var_extra)
        vcfg.gate = gr
        Z_next = arm_fn(parents, Z, v_tilde, vcfg)

        row = dict(round=t, queries=t * cfg.n_syn, **ev(Z_next))
        row.update({k: v for k, v in vcfg.stats.items() if not k.startswith("_")})
        row["spread"] = H.mean_pairwise(Z_next)

        wsel = np.where(v_tilde >= kappa, np.maximum(v_tilde, 0.0), 0.0)
        row["n_eff_select"] = (float(wsel.sum() ** 2 / (wsel**2).sum())
                               if wsel.sum() > 0 else float(cfg.n_syn))
        row["uniq_parent_frac"] = float(len(np.unique(parents)) / cfg.n_syn)
        row["floor_zeroed_frac"] = float((v_tilde < kappa).mean())
        row["distinct_frac"] = float(
            len(np.unique(np.round(Z_next, 9), axis=0)) / cfg.n_syn
        )

        # ---- gate diagnostics (spec 5) ----
        rej = gr.reject(cfg.alpha, "mc" if cfg.use_mc_pvalue else "chi2")
        row["gate_reject_rate"] = float(rej.mean())
        row["gate_k_eff"] = float(gr.k_eff.mean())
        row["gate_rank"] = float(np.mean(gr.rank))
        row["gate_degenerate_null"] = bool(gr.degenerate_null)
        for q in range(5):  # reject rate by vote quintile: density filter?
            m = vote_q0 == q
            row[f"gate_reject_q{q + 1}"] = float(rej[m].mean()) if m.any() else np.nan
        if ds.mixture is not None and gr.var_null > 0:
            lam = M.noncentrality(v_true, gr, np.sqrt(gr.var_null))
            fpr, pwr, flat = M.gate_fpr_power(rej, lam)
            row.update(gate_fpr=fpr, gate_power=pwr, flat_frac=flat)

        # ---- direction quality (spec 7.2) ----
        row.update(M.metric7_votefield(Z_next, parents, Z, v_true))
        if ds.mixture is not None:
            q_law = ds.q0 if t == 1 else None  # analytic law only at t=0
            h = bandwidth or M.kde_bandwidth_cv(Z, rng=np.random.default_rng(7))
            if t == 1 or D == 2:
                tgt = M.score_ratio(Z, ds.mixture, q_law=q_law, Z_pop=Z, h=h)
                dq = M.direction_quality(gr, Z, tgt, D, v_tilde=None)
                row["cos5"] = dq["cos5"]
                row["cos5b"] = dq["cos5b"]
                row["grad_norm_p50"] = dq["grad_norm_p50"]
            if D == 2:
                row["m7_analytic"] = M.metric7_analytic(Z_next, parents, Z,
                                                        ds.mixture, h)

        rows.append(row)
        stop = H.in_loop(row["frechet"], rows[0]["frechet"], row["spread"],
                         spread_ref, t)
        Z = Z_next
        if verbose:
            print(f"  t={t:2d} FD={row['frechet']:8.3f} cov={row['coverage']:.3f}")
        if stop:
            abort = stop
            break

    hp = H.per_run(rows, spread_ref, cfg.n_syn)
    return dict(
        cfg=asdict(cfg),
        sigma=float(sigma),
        privacy=spend,
        kappa=kappa,
        var_null=float(sigma**2 + var_extra),
        mean_count=float(cfg.N / cfg.n_syn),
        vote_snr=snr,
        ceiling=ceiling,
        spread_ref=spread_ref,
        rows=rows,
        health=hp,
        aborted=abort,
        wall_s=time.time() - t_start,
    )
