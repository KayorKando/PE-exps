"""Stage-gated grid (spec v0.2 section 8).

Strictly sequential; each stage states whether its gate passed.

  Stage 0  Q0   -- is g_hat informative at all?          (pe_toy/stage0.py)
  Stage 1  health gate: FD_T < FD_0 for every arm in every cell
  Stage 2  Q2'  -- contrastive vs random_pair / centroid
  Stage 3  Q1, Q3 -- add oracle and truth
  Stage 4  Q2, Q4, Q5 -- sweeps over eps, k, alpha, theta
  Stage 5  config B, transferable metrics only

Healthy cells are decided by stage 1 and cached in results/healthy_cells.json;
later stages read conclusions only from cells listed there.
"""

from __future__ import annotations

import os

# Must precede the numpy import: 6 workers on 8 cores would each spin up a
# full BLAS thread pool and thrash.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np

from .pipeline import RunConfig, run

# macOS defaults to "spawn", which re-imports __main__ in every worker and
# stalls; fork is faster and reliable for this read-only-parent workload.
MP = mp.get_context("fork")

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SEEDS = tuple(range(10))  # spec 8: 10 seeds for any pass/fail cell
THETAS = (0, 15, 30, 45, 60, 90)
CONTROLS = ["blind", "random_pair", "centroid"]
METHOD = ["contrastive-always", "contrastive-gated"]
UPPER = ["oracle-always", "oracle-gated", "truth-always"]

# Candidate cells for the health gate.  D=128 is the spec's read-off cell;
# D=8/32/64 bracket the dimension where nearest-neighbour vote hubness makes
# the gate unsatisfiable, which is the thing stage 1 has to establish.
HEALTH_CELLS = [("A", 2), ("A", 8), ("A", 32), ("A", 64), ("A", 128), ("B", 128)]


def _one(cfg_dict):
    cfg = RunConfig(**cfg_dict)
    try:
        return run(cfg)
    except Exception as e:
        return {"cfg": cfg_dict, "error": f"{type(e).__name__}: {e}"}


def run_many(cfgs, out_path, jobs=6, label=""):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(RunConfig(**json.loads(line)["cfg"]).key())
    todo = [c for c in cfgs if RunConfig(**c).key() not in done]
    if not todo:
        print(f"  {label}: all {len(cfgs)} runs cached", flush=True)
        return
    print(f"  {label}: {len(todo)} runs ({len(cfgs) - len(todo)} cached)", flush=True)
    t0, n = time.time(), 0
    with open(out_path, "a") as fh, ProcessPoolExecutor(
        max_workers=jobs, mp_context=MP
    ) as ex:
        for f in as_completed([ex.submit(_one, c) for c in todo]):
            fh.write(json.dumps(f.result()) + "\n")
            fh.flush()
            n += 1
            if n % 20 == 0 or n == len(todo):
                print(f"    {n}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)


def base(config, D, eps, **kw):
    d = asdict(RunConfig(config=config, D=D, eps=eps))
    d.update(kw)
    return d


# --------------------------------------------------------------------------


def stage0(jobs):
    from .stage0 import report, run_stage0

    run_stage0()
    report()


def stage1(jobs):
    """Health gate.  Blocking: no Q is read from a cell that fails."""
    cfgs = []
    for config, D in HEALTH_CELLS:
        for eps in (1.0, float("inf")):
            for arm in ("blind", "contrastive-gated", "random_pair"):
                for s in SEEDS[:5]:
                    cfgs.append(base(config, D, eps, arm=arm, seed=s))
    run_many(cfgs, RESULTS / "stage1.jsonl", jobs, "stage1")
    return healthy_cells(refresh=True)


CONTROL_ARMS = ("blind", "random_pair")


def healthy_cells(refresh=False, mode="controls"):
    """Cells satisfying spec 9.0, under one of two readings.

    mode="strict"   -- every arm, exactly as spec 9.0 is written.
    mode="controls" -- the *control* arms (blind, random_pair) only.

    The distinction matters and is not cosmetic.  Condition 3
    (n_eff/n_syn > 0.3) is failed at D=2 and D=8 by `contrastive-gated`
    alone, because the directed step contracts the population and so sharpens
    the vote histogram.  That contraction is the behaviour under test -- it is
    most of what Q2' asks about -- not a defect of the harness that section 9.0's
    preamble tells us to "fix and re-run".  Certifying a cell on the controls
    keeps the gate doing its job (is the shared loop sound?) while leaving the
    method's own contraction measurable rather than self-censoring.  Both
    readings are reported.
    """
    path = RESULTS / f"healthy_cells_{mode}.json"
    if path.exists() and not refresh:
        return [tuple(x) for x in json.loads(path.read_text())]
    from .health import cell_condition2

    rows = [json.loads(l) for l in (RESULTS / "stage1.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if "error" not in r]
    cells = {}
    for r in rows:
        c = r["cfg"]
        key = (c["config"], c["D"], c["eps"])
        cells.setdefault(key, {}).setdefault(c["arm"], []).append(r)
    ok = []
    for key, arms in sorted(cells.items(), key=lambda kv: str(kv[0])):
        judged = arms if mode == "strict" else {
            a: v for a, v in arms.items() if a in CONTROL_ARMS
        }
        if not judged:
            continue
        good = True
        for arm, rs in judged.items():
            if not all(r["health"]["healthy_run"] for r in rs):
                good = False
            curves = [[x["frechet"] for x in r["rows"]] for r in rs]
            L = min(len(c) for c in curves)
            if not cell_condition2([c[:L] for c in curves]):
                good = False
        if good:
            ok.append(key)
    path.write_text(json.dumps(ok))
    return [tuple(x) for x in ok]


def _healthy_A_dims(cells):
    return [D for (cfg, D, eps) in cells if cfg == "A" and eps == 1.0]


def stage2(jobs):
    """Q2': does the consensus content matter? contrastive vs random_pair/centroid."""
    dims = _healthy_A_dims(healthy_cells()) or [2]
    cfgs = []
    for D in dims:
        for arm in CONTROLS + METHOD:
            for s in SEEDS:
                cfgs.append(base("A", D, 1.0, arm=arm, seed=s))
    run_many(cfgs, RESULTS / "stage2.jsonl", jobs, "stage2")


def stage3(jobs):
    """Q1, Q3: add the oracle and truth arms."""
    dims = _healthy_A_dims(healthy_cells()) or [2]
    cfgs = []
    for D in dims:
        for arm in UPPER:
            if arm == "truth-always" and D != 2:
                continue  # spec 6.4: truth is reported at D=2 only
            for s in SEEDS:
                cfgs.append(base("A", D, 1.0, arm=arm, seed=s))
        for eps in (float("inf"),):
            for arm in CONTROLS + METHOD + ["oracle-always"]:
                for s in SEEDS[:5]:
                    cfgs.append(base("A", D, eps, arm=arm, seed=s))
    run_many(cfgs, RESULTS / "stage3.jsonl", jobs, "stage3")


def stage4(jobs):
    """Q2, Q4, Q5: sweeps over eps, k, alpha, theta (+ the 4.2 (a)/(b) check)."""
    dims = _healthy_A_dims(healthy_cells()) or [2]
    cfgs = []
    for D in dims:
        for th in THETAS:
            for arm in ("contrastive-always", "contrastive-gated", "random_pair"):
                for s in SEEDS:
                    cfgs.append(base("A", D, 1.0, arm=arm, theta_deg=th, seed=s))
        for eps in (0.5, 4.0):
            for arm in CONTROLS + METHOD:
                for s in SEEDS[:5]:
                    cfgs.append(base("A", D, eps, arm=arm, seed=s))
        for k in (5, 20):
            for arm in ("contrastive-gated", "random_pair-gated"):
                for s in SEEDS[:5]:
                    cfgs.append(base("A", D, 1.0, arm=arm, k=k, seed=s))
        for al in (0.01, 0.2):
            for arm in ("contrastive-gated", "random_pair-gated"):
                for s in SEEDS[:5]:
                    cfgs.append(base("A", D, 1.0, arm=arm, alpha=al, seed=s))
        # Q2 needs random_pair in both gating modes at the default settings
        for arm in ("random_pair-gated",):
            for s in SEEDS:
                cfgs.append(base("A", D, 1.0, arm=arm, seed=s))
        # spec 4.2: variant (a) vs (b) for the null variance, at eps=1 and inf
        for eps in (1.0, float("inf")):
            for va in ("a", "b"):
                for s in SEEDS[:5]:
                    cfgs.append(base("A", D, eps, arm="contrastive-gated",
                                     var_variant=va, seed=s))
    run_many(cfgs, RESULTS / "stage4.jsonl", jobs, "stage4")


def stage5(jobs):
    """Config B, transferable metrics only."""
    cfgs = []
    for D in (128, 768):
        for eps in (1.0, float("inf")):
            for arm in CONTROLS + METHOD + ["oracle-always"]:
                for s in SEEDS[:5]:
                    cfgs.append(base("B", D, eps, arm=arm, seed=s))
    run_many(cfgs, RESULTS / "stage5.jsonl", min(jobs, 4), "stage5")


STAGES = {"0": stage0, "1": stage1, "2": stage2, "3": stage3, "4": stage4,
          "5": stage5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=list(STAGES))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    for s in (a.stages or list(STAGES)):
        print(f"=== stage {s} ===", flush=True)
        t = time.time()
        STAGES[s](a.jobs)
        print(f"=== stage {s} done in {time.time() - t:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
