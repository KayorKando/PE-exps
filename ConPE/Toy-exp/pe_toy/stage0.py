"""Stage 0 -- Q0, the blocking gate (spec v0.2 section 8, section 9.1).

Is the direction estimate g_hat informative about grad log(p/q) at all, in the
regime where a direction exists?  One round, no loop.

Why this is measurable now and was not in run #1 (spec D2):
  * measured at t=0, where q_0 is far from p so grad log(p/q_0) is large and
    well determined -- not at t=T, where q_T ~= p and the reference vector is
    numerical noise;
  * q_0 is the *analytic* law of Z_0 (heaviest two mixture components
    convolved with the q0 noise), so no KDE enters the reference vector.
    That also makes the measurement valid at D=128, not only D=2.
  * restricted to points where ||grad log(p/q_0)|| is above its median;
  * read against a permutation null (shuffle v_tilde across each parent's
    neighbours), which is the honest baseline for a signed mean cosine.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .data import make_dataset
from .gate import consensus
from .metrics import direction_quality, score_ratio
from .pipeline import dp_vote
from .privacy import derive_sigma

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
SEEDS = tuple(range(10))
N_GRID = (2_000, 20_000, 100_000)


def q0_cell(D, N, seed, eps=float("inf"), k=10, n_syn=500, n_perm=200,
            var_b=True):
    rng = np.random.default_rng(20_000 + seed)
    ds = make_dataset("A", D=D, N=N, n_syn=n_syn, n_ref=2000, seed=seed)
    if np.isinf(eps):
        sigma = 0.0
    else:
        sigma = derive_sigma(eps, 1.0 / N, 10).sigma

    v_true, v_tilde = dp_vote(ds.P, ds.Z0, sigma, rng)
    var_extra = (N / n_syn) if var_b else 0.0
    gr = consensus(ds.Z0, v_tilde, sigma, k=k, mc=False, rng=rng,
                   var_extra=var_extra)
    target = score_ratio(ds.Z0, ds.mixture, q_law=ds.q0)
    out = direction_quality(gr, ds.Z0, target, D, v_tilde=v_tilde,
                            n_perm=n_perm, rng=rng)

    # spec 4.2: report the total vote SNR, not sigma alone
    snr = float(v_true.std() / np.sqrt(sigma**2 + N / n_syn))
    out.update(D=D, N=N, seed=seed, eps=eps, sigma=float(sigma), k=k,
               vote_snr=snr, mean_count=N / n_syn,
               k_eff_mean=float(gr.k_eff.mean()))
    return out


def run_stage0(dims=(2, 128), Ns=N_GRID, seeds=SEEDS, eps_list=(float("inf"), 1.0)):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "stage0.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["D"], r["N"], r["seed"], r["eps"]))
    with open(path, "a") as fh:
        for D in dims:
            for N in Ns:
                for eps in eps_list:
                    for s in seeds:
                        if (D, N, s, eps if not np.isinf(eps) else None) in done \
                           or (D, N, s, eps) in done:
                            continue
                        r = q0_cell(D, N, s, eps=eps)
                        r["eps"] = None if np.isinf(eps) else eps
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
                print(f"  Q0: D={D} N={N} done", flush=True)
    return path


def report(out=print):
    path = RESULTS / "stage0.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out("\n" + "=" * 100)
    out("STAGE 0 / Q0 -- is g_hat informative about grad log(p/q) at t=0?")
    out("    Pass (spec 9.1): cos exceeds the permutation null by > 3 permutation-s.d.")
    out("    at D=2, eps=inf, N=100k, and increases monotonically in N.")
    out("=" * 100)
    out(f"   {'D':>4} {'eps':>5} {'N':>8} {'vote SNR':>9} {'cos (m5)':>14} "
        f"{'cos (m5b)':>14} {'perm null':>16} {'z':>7} {'rand base':>10}")
    verdict = {}
    for D in sorted({r["D"] for r in rows}):
        for eps in sorted({r["eps"] for r in rows}, key=lambda x: (x is not None, x)):
            for N in sorted({r["N"] for r in rows}):
                sel = [r for r in rows if r["D"] == D and r["N"] == N
                       and r["eps"] == eps]
                if not sel:
                    continue
                f = lambda key: (float(np.mean([r[key] for r in sel])),
                                 float(np.std([r[key] for r in sel], ddof=1)))
                c, cs = f("cos5")
                cb, cbs = f("cos5b")
                pm, _ = f("perm_mean")
                ps, _ = f("perm_sd")
                z, zs = f("perm_z")
                snr, _ = f("vote_snr")
                rb = sel[0]["random_baseline"]
                out(f"   {D:4d} {str(eps):>5} {N:8d} {snr:9.2f} "
                    f"{c:8.3f}±{cs:.3f} {cb:8.3f}±{cbs:.3f} "
                    f"{pm:8.3f}±{ps:.3f} {z:7.1f} {rb:10.3f}")
                verdict[(D, eps, N)] = (c, z, cs)
    out("")
    key = (2, None, 100_000)
    if key in verdict:
        c, z, _ = verdict[key]
        mono = [verdict[(2, None, n)][0] for n in N_GRID if (2, None, n) in verdict]
        is_mono = all(b >= a - 1e-9 for a, b in zip(mono, mono[1:]))
        out(f"   D=2, eps=inf, N=100k:  cos={c:.3f}, {z:.1f} permutation-s.d. above null")
        out(f"   monotone in N: {mono} -> {is_mono}")
        out(f"   Q0 VERDICT: {'PASS' if (z > 3 and is_mono) else 'FAIL -- the line dies here'}")
    return verdict
