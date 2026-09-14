# pe_toy — Consensus-Guided Variation for Private Evolution

Implementation of `spec.md` **v0.2**. Read `SUMMARY.md` first: it names which
gates fired. Short version: Q0 passes decisively, Q2′ fails, and the binding
constraint is `select`, not the swappable step.

## Layout

| file | spec § | what it is |
|---|---|---|
| `pe_toy/pipeline.py` | §2, §4.2–4.3 | the loop: `dp_vote`, vote floor, `select`, `vary`, in-loop health check |
| `pe_toy/data.py` | §3 | config A (mixture, with **analytic `p` and `q₀`**) and config B (Yelp + all-mpnet-base-v2, PCA cached to disk) |
| `pe_toy/privacy.py` | §4.1 | σ from (ε, δ, T): RDP accountant + exact analytic Gaussian cross-check |
| `pe_toy/gate.py` | §5 | consensus test; duplicate-neighbour handling; §4.2 (a)/(b) null variance |
| `pe_toy/vary.py` | §6 | arms incl. controls `random_pair`, `centroid`; manifold-local noise |
| `pe_toy/health.py` | §9.0 | the blocking health gate |
| `pe_toy/metrics.py` | §7 | FD, precision/recall/coverage, metrics 5, 5b, 7, permutation null |
| `pe_toy/stage0.py` | §8 stage 0 | Q0, the cheapest kill — one round, no loop |
| `pe_toy/run_grid.py` | §8 | the stage-gated grid |
| `pe_toy/analyze.py` | §12.2 | one table per stage, each stating whether its gate passed |
| `pe_toy/figures.py` | §12.3 | figures (a)–(e) |

## Reproducing

```bash
python3 -m pe_toy.run_grid 0            # Q0 gate   (~4 min)
python3 -m pe_toy.run_grid 1 --jobs 5   # health    (~12 min)
python3 -m pe_toy.run_grid 2 3 4 --jobs 6
python3 -m pe_toy.run_grid 5 --jobs 3   # config B; downloads corpus + encoder
python3 -m pe_toy.analyze               # -> results/tables.txt
python3 -m pe_toy.figures               # -> figures/
```

Runs are cached by config key in `results/stage*.jsonl`; re-running skips
completed cells. Healthy cells are cached in `results/healthy_cells_*.json`.

**Memory:** a config A run peaks at ~0.5 GB, a config B D=768 run at ~0.8 GB.
Size `--jobs` from that; on an 8 GB box 3 is the safe ceiling.

**First run of config B** fits a 2-component full-covariance GMM on a
20000×768 matrix (~60 s per (D, seed)) to build `Z0`. The result is cached to
`.cache/z0_*.npy`, so it happens once, not once per run. Warm the cache
serially before a parallel sweep, or several workers will race on the same fit:

```bash
python3 -c "from pe_toy.data import make_dataset
for D in (128,768):
    for s in range(5): make_dataset('B', D=D, N=20000, n_syn=500, n_ref=2000, seed=s)"
```

## Running on Colab

**Colab is the primary path** — the sweeps peg every core for tens of minutes
and make an 8 GB laptop unusable while they run.

`colab/PE_toy_colab.ipynb` does the whole workflow there: mount Drive, fetch the
code, build the caches, run stages 0–5, produce tables and figures, and hand
back a zip. Work is mirrored to Drive after each stage by `colab/sync.py`, so a
dropped runtime costs nothing — re-run the cell and each stage resumes from its
`results/*.jsonl` cache. A plain CPU runtime is right; no GPU is used.

If you do run locally, `--jobs` defaults to 4 (was 6, which pegged all 8 cores);
drop to 2 to keep the machine responsive.

### Scale-up sweeps

`colab/PE_toy_colab.ipynb` reproduces the whole grid on a Colab CPU runtime
(~40 min) and drives the scale-up sweeps in `colab/scale_up.py`, which do not
fit on an 8 GB laptop:

| sweep | question | cost |
|---|---|---|
| `nsyn_health` | Can a bigger `n_syn` buy health at D ≥ 32? **Blocking for v0.3.** | ~2.5 h |
| `q0_scale` | Is the D=128 direction plateau real at N = 1e6? | ~25 min, high-RAM |
| `configA_768` | The §3.4 breakage check on config A, never run in run #2 | ~20 min |

Build the upload with `bash make_bundle.sh` (64 KB, code only — `.cache/` is
rebuilt on Colab faster than it uploads).

## The `vary` interface (the only swappable step)

```python
vary(parents, Z_t, v_tilde, cfg) -> Z_{t+1}
```

`parents` is the integer index array from `select` (indices into `Z_t`, with
repeats). `cfg` is a `VaryContext` carrying σ, the RNG, k, α, θ, the step scale
and the per-round `GateResult` — **and nothing private**. §2's rule is enforced
mechanically: `P`, the true densities and the noise-free votes live behind
`cfg.oracle`, which is `None` for every arm except `directed_truth`.

## Deviations from the spec, and why

Each of these is a place where the spec as written could not be executed
literally. All are load-bearing for the results.

1. **Step scale (§6.0).** `s = 0.25 × median NN distance of Z_t` is
   self-extinguishing — the step is what holds points apart, so it drives itself
   to zero (measured: s → exactly 0 by round 7, >50% duplicate points, every
   arm). `s` is keyed to `nn(Z₀)` instead: fixed, non-private, measured once.
   `step_ref="adaptive"` reproduces the literal rule.
2. **Noise scale (§6.0).** `trace(C̃) = D` makes the jitter √D long against a
   unit axis, so nominal θ=0 realised as 47° at D=128. ξ is normalised to unit
   expected length; θ now tracks nominal to <1°.
3. **Vote floor (§4.3).** "the 10th percentile of the null vote distribution"
   would pass 90% of noise-only points, the opposite of the stated purpose.
   Implemented as the threshold passing 10% of them (the 90th percentile).
4. **Health gate (§9.0).** Applying condition 3 to *every* arm voids a cell
   whenever the method contracts — which is the thing Q2′ measures. Both the
   literal reading and a controls-only reading are computed and reported; with
   the step fix they agree.
5. **Config B reference (§7).** Using all N=20,000 private points as the metrics
   reference makes each call an O(N²) k-NN problem. A random `n_ref` subset is
   used, matching config A's reference size.
6. **Metric 5 at t=0 is analytic.** `q₀` is the heaviest two components
   convolved with the initialisation noise, so `∇log(p/q₀)` needs no KDE. This
   is what made Q0 readable, and it makes the measurement valid at D=128 too,
   not only D=2.

## Known-good checks

- Analytic Gaussian mechanism reproduces the published σ = 3.7306 at ε=1, δ=1e-5.
- Mixture `score()` matches finite differences to 2e-9; `q₀` matches the
  empirical law of `Z₀` (mean 0.003, cov 0.4% relative).
- Gate statistic Q is exactly χ²_r per parent under a flat true profile
  (KS p-values uniform over 300 independent noise replications, k = 5/10/20).
- Empirical gate FPR 0.055–0.069 against nominal 0.05 at ε ≤ 1.
- `LocalCov` low-rank sampler matches the explicit covariance construction to
  Monte-Carlo error, with trace(C̃) = D exactly.
- `health.mean_pairwise` matches the naive O(n²D) form exactly.
