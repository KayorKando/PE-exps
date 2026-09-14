"""Runs one task on the VM, logging to /content/logs/<task>.log.

Launched detached by launch.py -- never invoked directly by `colab exec`.

Why detached: `colab exec` streams the kernel's output and gives up with
"TimeoutError: Timeout waiting for output" if a step is silent for too long.
Embedding 24k sentences on 2 CPU cores is silent for ~10 min, and the
scale-up sweeps run for hours, so anything long has to be a background
process on the VM that short `exec` calls poll. This also means a dropped
laptop connection cannot kill the job.
"""

import os
import subprocess
import sys
import time

WORK = "/content/toyexp"
LOGS = "/content/logs"


def jobs():
    import multiprocessing as mp
    # config A peaks ~0.5 GB/run, config B D=768 ~0.8 GB; the VM has ~12 GB,
    # so cores are the binding constraint, not memory.
    return max(1, min(4, mp.cpu_count()))


def sh(*cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(list(cmd), cwd=WORK)
    if r.returncode:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}")


def task_setup():
    from pe_toy.data import embed_corpus, fetch_corpus, make_dataset, pca_pool
    sents = fetch_corpus("yelp", 24000)
    print(f"sentences {len(sents)}", flush=True)
    t = time.time()
    print(f"embeddings {embed_corpus(sents).shape} ({time.time() - t:.0f}s)", flush=True)
    for D in (128, 768):
        pca_pool("yelp", D)
        print(f"pca pool D={D} ready", flush=True)
    for D in (128, 768):
        for s in range(5):
            t = time.time()
            make_dataset("B", D=D, N=20000, n_syn=500, n_ref=2000, seed=s)
            print(f"  Z0 D={D} seed={s}  {time.time() - t:.0f}s", flush=True)


def task_grid():
    j = str(jobs())
    for stages in (["0"], ["1"], ["2", "3", "4"], ["5"]):
        sh(sys.executable, "-m", "pe_toy.run_grid", *stages, "--jobs", j)
    sh(sys.executable, "-m", "pe_toy.analyze")
    sh(sys.executable, "-m", "pe_toy.figures")


def task_scaleup():
    j = str(jobs())
    for s in ("configA_768", "nsyn_health"):
        sh(sys.executable, "-m", "colab.scale_up", s, "--jobs", j)
    sh(sys.executable, "-m", "colab.scale_up", "report")


def task_q0scale():
    sh(sys.executable, "-m", "colab.scale_up", "q0_scale", "--jobs", "1")


TASKS = {"setup": task_setup, "grid": task_grid, "scaleup": task_scaleup,
         "q0scale": task_q0scale}

if __name__ == "__main__":
    name = sys.argv[1]
    os.chdir(WORK)
    sys.path.insert(0, WORK)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    t0 = time.time()
    print(f"=== {name} starting, jobs={jobs()} ===", flush=True)
    try:
        TASKS[name]()
        print(f"=== {name} DONE in {time.time() - t0:.0f}s ===", flush=True)
    except SystemExit as e:
        print(f"=== {name} FAILED: {e} ===", flush=True)
        raise
