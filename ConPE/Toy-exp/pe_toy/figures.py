"""Figures (spec v0.2 section 12.3).

(a) FD vs round, all arms, in the healthy cells
(b) cos vs round and vs N (Q0), with the permutation-null band
(c) per-quintile coverage vs `random_pair`
(d) MC null vs chi^2_r overlay, at eps=1 and eps=inf
(e) D=2 scatter of one round: parent, z_+/z_-, child, and the true gradient
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze import RESULTS, cell, dims, final, index, load, ms, series

FIGS = pathlib.Path(__file__).resolve().parent.parent / "figures"
COLORS = {
    "blind": "#444444",
    "random_pair": "#e08a1e",
    "centroid": "#8e6bbf",
    "contrastive-always": "#1b7fd4",
    "contrastive-gated": "#7fbcea",
    "oracle-always": "#d1495b",
    "oracle-gated": "#eda6ae",
    "truth-always": "#2e8b57",
}


def fig_a(idx):
    ds = dims()
    if not ds:
        return
    fig, axes = plt.subplots(1, len(ds), figsize=(5.6 * len(ds), 4.3), squeeze=False)
    for ax, D in zip(axes[0], ds):
        for arm, col in COLORS.items():
            c = cell(idx, "A", D, 1.0, arm)
            if not c:
                continue
            L = min(len(r["rows"]) for r in c.values())
            S = np.array([series(r, "frechet")[:L] for r in c.values()])
            m, sd = S.mean(0), S.std(0, ddof=1)
            x = np.arange(L)
            ax.plot(x, m, color=col, label=arm, lw=1.8)
            ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.13, lw=0)
        b = cell(idx, "A", D, 1.0, "blind")
        if b:
            fl = ms([r["ceiling"]["frechet"] for r in b.values()])[0]
            ax.axhline(fl, ls=":", c="k", lw=1, label=f"real-sample floor ({fl:.3f})")
        ax.set_title(f"config A, D={D}, eps=1 (health gate: PASS)")
        ax.set_xlabel("round")
        ax.set_ylabel("Frechet distance")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7.5)
    fig.suptitle("(a) Frechet distance vs round, all arms, theta=0", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "a_frechet_vs_round.png", dpi=150)
    plt.close(fig)


def fig_b():
    rows = [json.loads(l) for l in (RESULTS / "stage0.jsonl").read_text().splitlines()
            if l.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, D in zip(axes, (2, 128)):
        for eps, col, lab in ((None, "#1b7fd4", "eps=inf"), (1.0, "#d1495b", "eps=1")):
            Ns, mu, sd, pm, ps = [], [], [], [], []
            for N in sorted({r["N"] for r in rows}):
                sel = [r for r in rows if r["D"] == D and r["N"] == N
                       and r["eps"] == eps]
                if not sel:
                    continue
                Ns.append(N)
                mu.append(np.mean([r["cos5"] for r in sel]))
                sd.append(np.std([r["cos5"] for r in sel], ddof=1))
                pm.append(np.mean([r["perm_mean"] for r in sel]))
                ps.append(np.mean([r["perm_sd"] for r in sel]))
            if not Ns:
                continue
            ax.errorbar(Ns, mu, yerr=sd, marker="o", color=col, lw=1.8,
                        capsize=3, label=f"cos, {lab}")
            pm, ps = np.array(pm), np.array(ps)
            ax.fill_between(Ns, pm - 3 * ps, pm + 3 * ps, color=col, alpha=0.15,
                            lw=0, label=f"permutation null +-3sd, {lab}")
        rb = np.sqrt(2.0 / (np.pi * D))
        ax.axhline(rb, ls="--", c="k", lw=1,
                   label=f"random-direction |cos| = {rb:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("N (private points)")
        ax.set_ylabel(r"$\cos(\hat g,\ \nabla\log(p/q_0))$")
        ax.set_title(f"D={D}, t=0")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("(b) Q0: direction quality at t=0 vs N, against the permutation null",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "b_q0_cos_vs_N.png", dpi=150)
    plt.close(fig)


def fig_c(idx):
    ds = dims()
    if not ds:
        return
    arms = ["random_pair", "centroid", "contrastive-always", "contrastive-gated"]
    fig, axes = plt.subplots(1, len(ds), figsize=(5.6 * len(ds), 4.3), squeeze=False)
    for ax, D in zip(axes[0], ds):
        w = 0.8 / len(arms)
        for i, arm in enumerate(arms):
            c = cell(idx, "A", D, 1.0, arm)
            if not c:
                continue
            m = [ms([final(r, f"coverage_q{q}") for r in c.values()])[0]
                 for q in range(1, 6)]
            e = [ms([final(r, f"coverage_q{q}") for r in c.values()])[1]
                 for q in range(1, 6)]
            ax.bar(np.arange(5) + i * w, m, w, yerr=e, capsize=2, label=arm,
                   color=COLORS[arm])
        ax.set_xticks(np.arange(5) + 0.4)
        ax.set_xticklabels([f"Q{q}" for q in range(1, 6)])
        ax.set_xlabel("t=0 vote-count quintile (Q1 = lowest)")
        ax.set_ylabel("coverage")
        ax.set_title(f"config A, D={D}, eps=1")
        ax.grid(alpha=0.25, axis="y")
    axes[0][0].legend(fontsize=7.5)
    fig.suptitle("(c) Q5: per-quintile coverage, read against random_pair", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "c_quintile_coverage.png", dpi=150)
    plt.close(fig)


def fig_d():
    from scipy.stats import chi2

    from .data import make_dataset
    from .gate import consensus
    from .pipeline import dp_vote
    from .privacy import derive_sigma

    ds = make_dataset("A", D=8, N=20000, seed=0)
    rng = np.random.default_rng(0)
    N, n_syn = 20000, 500
    cases = [("eps=1", derive_sigma(1.0, 1 / N, 10).sigma), ("eps=inf", 0.0)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for row, (lab, sigma) in enumerate(cases):
        for ax, k in zip(axes[row], (5, 10, 20)):
            Q, mc, r = [], None, None
            for rep in range(30):
                # flat TRUE profile: every cell has the same expected count
                v = float(N / n_syn) + (rng.normal(0, sigma, n_syn) if sigma > 0
                                        else 0.0)
                v = v + rng.normal(0, np.sqrt(N / n_syn), n_syn)  # finite-sample term
                res = consensus(ds.Z0, v, sigma, k=k, mc=(rep == 0), B=2000,
                                rng=rng, var_extra=N / n_syn)
                Q.append(res.Q)
                if rep == 0:
                    mc, r = res.Q_null_sample, int(res.rank[0])
            Q = np.concatenate(Q)
            Q = Q[np.isfinite(Q)]
            grid = np.linspace(0, chi2.ppf(0.999, r), 200)
            ax.hist(Q, bins=60, density=True, alpha=0.45, color="#1b7fd4",
                    label="observed Q (flat true profile)")
            if mc is not None and len(mc):
                ax.hist(mc, bins=60, density=True, histtype="step",
                        color="#d1495b", lw=1.4, label="Monte-Carlo null")
            ax.plot(grid, chi2.pdf(grid, r), "k--", lw=1.5, label=f"chi2_{r}")
            ax.set_title(f"{lab}, k={k}, r={r}")
            ax.set_xlabel("Q")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6.5)
        axes[row][0].set_ylabel("density")
    fig.suptitle("(d) consensus statistic under the null, with the spec 4.2(b) "
                 "variance inflation", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "d_null_overlay.png", dpi=150)
    plt.close(fig)


def fig_e():
    from .data import make_dataset
    from .gate import consensus
    from .metrics import score_ratio
    from .pipeline import dp_vote, select
    from .privacy import derive_sigma
    from .vary import VaryContext, make_arm, nn_scale

    ds = make_dataset("A", D=2, N=20000, seed=0)
    sigma = derive_sigma(1.0, 1 / 20000, 10).sigma
    rng = np.random.default_rng(0)
    Z = ds.Z0.copy()
    s_ref = nn_scale(Z)
    _, v = dp_vote(ds.P, Z, sigma, rng)
    par = select(Z, v, 500, rng)
    gr = consensus(Z, v, sigma, k=10, B=500, rng=rng, var_extra=40.0)
    cfg = VaryContext(sigma=sigma, rng=rng, gate=gr, s_ref=s_ref)
    child = make_arm("contrastive-always")(par, Z, v, cfg)

    proj = gr.proj[par]
    jp = np.nanargmax(np.where(np.isnan(proj), -np.inf, proj), axis=1)
    jm = np.nanargmin(np.where(np.isnan(proj), np.inf, proj), axis=1)
    rows = np.arange(len(par))
    zp, zm = Z[gr.knn[par][rows, jp]], Z[gr.knn[par][rows, jm]]
    truth = score_ratio(Z, ds.mixture, q_law=ds.q0)
    truth = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.scatter(ds.P[::10, 0], ds.P[::10, 1], s=5, c="#dddddd", label="private P", zorder=1)
    ax.scatter(Z[:, 0], Z[:, 1], s=9, c="#999999", label="$Z_t$", zorder=2)
    sel = rng.choice(len(par), 22, replace=False)
    L = 6 * cfg.stats["step_s"]
    for i in sel:
        ax.annotate("", xy=child[i], xytext=Z[par][i],
                    arrowprops=dict(arrowstyle="->", color="#1b7fd4", lw=1.4), zorder=6)
        ax.annotate("", xy=Z[par][i] + L * truth[par[i]], xytext=Z[par][i],
                    arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.2,
                                    ls="--", alpha=0.9), zorder=5)
        ax.plot([zm[i, 0], zp[i, 0]], [zm[i, 1], zp[i, 1]], c="#bbb", lw=0.7,
                ls="--", zorder=3)
    ax.scatter(Z[par][sel, 0], Z[par][sel, 1], s=50, facecolors="none",
               edgecolors="k", lw=1.3, label="parents", zorder=7)
    ax.scatter(zp[sel, 0], zp[sel, 1], s=38, marker="^", c="#2e8b57", label="$z_+$", zorder=7)
    ax.scatter(zm[sel, 0], zm[sel, 1], s=38, marker="v", c="#d1495b", label="$z_-$", zorder=7)
    ax.scatter(child[sel, 0], child[sel, 1], s=38, marker="*", c="#1b7fd4",
               label="children", zorder=8)
    ax.plot([], [], c="#2e8b57", ls="--", label=r"true $\nabla\log(p/q_0)$")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("(e) one round of contrastive directed variation, D=2\n"
                 "grey dashed = contrastive axis $z_-\\to z_+$; blue = parent$\\to$child; "
                 "green dashed = true gradient")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGS / "e_d2_scatter.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    idx = index(load(2, 3, 4))
    if idx:
        fig_a(idx); print("  a done")
        fig_c(idx); print("  c done")
    fig_b(); print("  b done")
    fig_d(); print("  d done")
    fig_e(); print("  e done")
    print(f"figures in {FIGS}")


if __name__ == "__main__":
    main()
