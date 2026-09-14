"""Sweeps that are worth a bigger machine than an 8 GB laptop.

Nothing here is a new method -- every sweep drives the existing pe_toy code at
parameters the local box cannot afford.  Each one targets a specific open
question left by run #2 (see SUMMARY.md).

  1. nsyn_health   -- Can the section 9.0 health gate be satisfied at D >= 32 by
                      raising n_syn?  Run #2 showed n_eff/n_syn is flat in
                      n_syn for a *perfect* population, but never tested the
                      full loop above n_syn = 500.  This is the blocking
                      question for v0.3: without a healthy non-trivial cell,
                      Q1 cannot be asked at all.
  2. q0_scale      -- Does the D=128 direction-quality plateau (cos ~ 0.09,
                      flat from N=2k to 100k) survive N = 1e6?  If it does, the
                      high-D weakness is geometric and no data budget fixes it.
  3. configA_768   -- The spec 3.4 breakage check at D=768 on config A, which run
                      #2 never ran (only config B reached 768).

Cost on a Colab CPU runtime, measured by extrapolating local timings
(n_syn=2000 at D=128 is 21.8 s / run, peak 0.51 GB; cost is ~linear in n_syn):

    nsyn_health   ~2.5 h   (the big one; n_syn=8000 cells dominate)
    q0_scale      ~25 min  (memory-bound: N=1e6 at D=768 is a 6 GB array)
    configA_768   ~20 min

Usage:
    python -m colab.scale_up nsyn_health --jobs 4
    python -m colab.scale_up all --jobs 4
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from pe_toy.pipeline import RunConfig, run
from pe_toy.run_grid import MP, RESULTS, run_many

SEEDS = tuple(range(5))


def _cfg(**kw):
    d = asdict(RunConfig())
    d.update(kw)
    return d


# --------------------------------------------------------------------------


def nsyn_health(jobs):
    """Does a larger synthetic population make D >= 32 healthy?

    Run #2 measured n_eff/n_syn for a real sample as 0.37 / 0.22 / 0.08 at
    D = 32 / 64 / 128, independent of n_syn from 500 to 8000.  That says the
    vote histogram's hubness is geometric.  What it does not settle is whether
    the *loop* becomes healthy anyway once n_syn is large enough that even a
    small effective fraction is an adequate population.  Health is judged by
    spec 9.0, on the control arms, exactly as in stage 1.
    """
    cfgs = []
    for D in (32, 64, 128):
        for n_syn in (500, 2000, 8000):
            for arm in ("blind", "random_pair", "contrastive-gated"):
                for s in SEEDS:
                    cfgs.append(_cfg(config="A", D=D, n_syn=n_syn, eps=1.0,
                                     arm=arm, seed=s))
    run_many(cfgs, RESULTS / "scale_nsyn_health.jsonl", jobs, "nsyn_health")


def configA_768(jobs):
    """spec 3.4's D=768 breakage check on config A (never run in run #2)."""
    cfgs = []
    for eps in (1.0, float("inf")):
        for arm in ("blind", "random_pair", "centroid",
                    "contrastive-always", "contrastive-gated", "oracle-always"):
            for s in SEEDS:
                cfgs.append(_cfg(config="A", D=768, eps=eps, arm=arm, seed=s))
    run_many(cfgs, RESULTS / "scale_configA_768.jsonl", jobs, "configA_768")


def q0_scale(jobs):
    """Q0 at larger N and more dimensions -- is the D>=128 plateau real?

    Serial on purpose: N=1e6 at D=768 is a 6 GB private matrix, so parallel
    workers would each hold their own copy.
    """
    from pe_toy.stage0 import q0_cell

    path = RESULTS / "scale_q0.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["D"], r["N"], r["seed"]))
    with open(path, "a") as fh:
        for D in (2, 8, 32, 128, 768):
            for N in (100_000, 1_000_000):
                if D == 768 and N > 100_000:
                    continue  # 6 GB of private points; skip unless you have RAM
                for s in range(5):
                    if (D, N, s) in done:
                        continue
                    t = time.time()
                    r = q0_cell(D, N, s, eps=float("inf"))
                    r["eps"] = None
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(f"  Q0 D={D} N={N} seed={s} cos={r['cos5']:.3f} "
                          f"z={r['perm_z']:.1f} ({time.time() - t:.0f}s)", flush=True)


def report():
    """Health verdicts for the n_syn sweep, plus the Q0 plateau check."""
    import collections

    from pe_toy.health import cell_condition2

    p = RESULTS / "scale_nsyn_health.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if "error" not in r]
        cells = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            c = r["cfg"]
            cells[(c["D"], c["n_syn"])][c["arm"]].append(r)
        print("\nn_syn sweep -- can a bigger population buy health at D >= 32?")
        print(f"   {'D':>5} {'n_syn':>7} {'arm':>18} {'FD_0':>9} {'FD_T':>9} "
              f"{'n_eff':>7} {'healthy':>8}")
        for key in sorted(cells):
            for arm, rs in sorted(cells[key].items()):
                g = lambda f: float(np.mean([r["health"][f] for r in rs]))
                cur = [[x["frechet"] for x in r["rows"]] for r in rs]
                L = min(len(c) for c in cur)
                ok = (all(r["health"]["healthy_run"] for r in rs)
                      and cell_condition2([c[:L] for c in cur]))
                print(f"   {key[0]:5d} {key[1]:7d} {arm:>18} {g('fd_0'):9.2f} "
                      f"{g('fd_T'):9.2f} {g('n_eff_frac'):7.2f} "
                      f"{('OK' if ok else '--'):>8}")

    p = RESULTS / "scale_q0.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        print("\nQ0 at scale -- cos(g_hat, grad log(p/q_0)) at t=0, eps=inf")
        print(f"   {'D':>5} {'N':>9} {'cos':>16} {'perm z':>8} {'random |cos|':>13}")
        for D in sorted({r["D"] for r in rows}):
            for N in sorted({r["N"] for r in rows}):
                sel = [r for r in rows if r["D"] == D and r["N"] == N]
                if not sel:
                    continue
                m = np.mean([r["cos5"] for r in sel])
                sd = np.std([r["cos5"] for r in sel], ddof=1)
                z = np.mean([r["perm_z"] for r in sel])
                print(f"   {D:5d} {N:9d} {m:9.3f}±{sd:.3f} {z:8.1f} "
                      f"{sel[0]['random_baseline']:13.3f}")


SWEEPS = {"nsyn_health": nsyn_health, "configA_768": configA_768,
          "q0_scale": q0_scale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", choices=list(SWEEPS) + ["all", "report"])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.sweep == "report":
        report()
        return
    names = list(SWEEPS) if a.sweep == "all" else [a.sweep]
    for n in names:
        print(f"=== {n} ===", flush=True)
        t = time.time()
        SWEEPS[n](a.jobs)
        print(f"=== {n} done in {time.time() - t:.0f}s ===", flush=True)
    report()


if __name__ == "__main__":
    main()
