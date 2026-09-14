# Toy Experiment v0.2 — Which gates fired, and what was concluded

Built to `spec.md` v0.2. 1,290 runs. Full tables: `results/tables.txt`;
figures: `figures/`; raw per-round records: `results/stage*.jsonl`.
Run #1 (v0.1) results are archived under `results.v01/`.

---

## Gate summary

| Stage | Gate | Fired? |
|---|---|---|
| **0 — Q0** | cos must beat the permutation null by >3 s.d. and rise with N | **PASS** (cos 0.669, z = 14.8, monotone) |
| **1 — health** | FD_T < FD_0 for every arm in every cell | **PASS at D ∈ {2, 8} only**; D ≥ 32 and config B fail and are unfixable within §2 |
| **2 — Q2′** | contrastive must beat `random_pair` **and** `centroid` on FD *and* metric 7 | **FAIL** — passes on metric 7 (10/10 seeds, every cell), fails on FD |
| 3 — Q1 | contrastive beats blind | **FAIL** (FD 0.649 vs 0.472 at D=8); metric has no resolution anyway |
| 3 — Q3 | share of oracle gain retained | ~94% — the rendering is **not** the bottleneck |
| 4 — Q2 | keep the gate? | **DROP** — gated ≈ always, and `random_pair` shows the same margin |
| 4 — Q4 | θ\* | **No θ\*** — FD *improves* as the axis is ignored (Spearman −0.94 at D=8) |
| 4 — Q5 | gains concentrate in low-vote regions | **FAIL** — losses are *largest* in the bottom quintile |

**The headline.** v0.2's central hypothesis splits cleanly in two, and the two
halves disagree:

- **The consensus direction is real.** Q0 passes decisively, and `contrastive`
  beats both vote-free controls on metric 7 in 10/10 seeds in every healthy cell.
  v0.1's null on this was a measurement artifact, exactly as D2 predicted.
- **Acting on it makes the synthetic data worse.** The same arm loses to
  `random_pair` on Fréchet distance (0/10 seeds at D=8) and on coverage
  (0.346 vs 0.543). Moving every child up the density ratio is mode-seeking: it
  raises per-point p/q while destroying coverage.

So the method does what it claims mechanically, and that mechanism is the wrong
objective. This is the Q2′ null, and per §9 it is the most informative outcome
available. No machinery was added to rescue it.

---

## Q0 — the direction estimate is informative (PASS)

cos(ĝ, ∇log(p/q₀)) at t=0, restricted to points where ‖∇log(p/q₀)‖ is above its
median, against a permutation null (shuffle ṽ across each parent's neighbours):

| D | ε | N | vote SNR | cos | perm null | z |
|---|---|---|---|---|---|---|
| 2 | ∞ | 2k | 2.00 | 0.523 ± 0.039 | 0.019 ± 0.045 | 11.2 |
| 2 | ∞ | 20k | 5.36 | 0.653 ± 0.062 | 0.016 ± 0.043 | 14.7 |
| 2 | ∞ | 100k | 11.78 | **0.669 ± 0.051** | 0.017 ± 0.044 | **14.8** |
| 2 | 1 | 2k | 0.40 | 0.129 ± 0.069 | −0.001 ± 0.045 | 2.9 |
| 2 | 1 | 100k | 8.74 | 0.644 ± 0.047 | 0.011 ± 0.044 | 14.4 |
| 128 | ∞ | 20k | 34.54 | 0.093 ± 0.004 | 0.001 ± 0.004 | 21.3 |

Three things follow.

1. **v0.2's N = 20,000 change was necessary and sufficient.** At v0.1's
   N = 2,000, ε = 1 the direction really was uninformative (cos 0.129, z = 2.9).
   At N = 100k it is 0.644. Run #1's null on direction quality was a
   finite-sample artifact, not a property of the method.
2. **D=128 is a different problem.** cos ≈ 0.09 is 20 permutation-s.d. above
   null but barely above the random-direction magnitude √(2/πD) = 0.071 — and it
   is **flat in N** (0.089 / 0.093 / 0.087). The high-dimensional weakness is
   geometric, not noise, so no amount of data or privacy budget fixes it.
3. **The M_w⁺ correction should not be adopted** (answers the §11 deferred
   question). Metric 5b equals metric 5 at D=2 (0.670 vs 0.669) and is *four
   times worse* at D=128 (0.021 vs 0.087). Neighbourhood geometry is not the
   source of the direction error.

Measuring q₀ analytically rather than by KDE is what made this readable: q₀ is
the heaviest two mixture components convolved with the initialisation noise, so
it is available in closed form, and D2's "reference vector is KDE jitter"
objection disappears.

## Stage 1 — the health gate, and two harness defects v0.2 did not anticipate

Healthy: **config A at D = 2 and D = 8**, for every arm, under both the literal
§9.0 reading and a controls-only reading. Everything else fails.

Getting there required fixing two defects, both in v0.2's own new machinery:

1. **§6.0's step rule is self-extinguishing.** `s = 0.25 × median NN distance of
   Z_t` re-measures the step from the population the step itself holds apart:
   smaller step → closer points → smaller step. Measured s collapsing
   4.2e-2 → **exactly 0 by round 7** at D=2, with >50% of the population becoming
   exact duplicates, for *every* arm. Every metric then froze after ~3 effective
   rounds. Fixed by keying s to nn(Z₀) — a fixed, non-private scale measured
   once. After the fix `distinct_frac` = 1.00 and n_eff rises 0.29 → 0.53.
2. **The §6.0 noise scaled as √D.** With trace(C̃) = D the jitter has length √D
   against a unit axis, so at D=128 a nominal θ=0 realised as **47°**, and
   gated parents (who get pure ξ) moved √D× further than directed ones. Fixed by
   normalising ξ to unit expected length. Measured θ now tracks nominal to <1°.

**D ≥ 32 cannot be fixed by any lever §8 sanctions.** n_eff/n_syn for a
*perfect* population (a real sample from p) — the ceiling any arm could reach:

| D | 2 | 8 | 32 | 64 | 128 | 768 |
|---|---|---|---|---|---|---|
| n_eff/n_syn | 0.76 | 0.77 | 0.37 | 0.22 | **0.08** | 0.06 |

and this is **independent of n_syn** (0.08 at every n_syn from 500 to 8,000 at
D=128). The cause is nearest-neighbour vote hubness: one synthetic point
captured 1,355 of 20,000 votes. Pure resampling with noise-free votes and no
variation at all still drives FD 99.8 → 284 at D=128. §6.0, §4.3 and the noise
model all act on `vary`; the defect is in `dp_vote` + `select`, which §2 freezes.
Verified across 3 noise models × 4 step sizes × floor on/off × σ=0.

## Q2′ — the consensus content matters, but not in the right direction (FAIL)

config A, D=8, ε=1, θ=0, 10 seeds:

| arm | final FD | coverage | metric 7 | n_eff/n |
|---|---|---|---|---|
| blind | 0.472 ± 0.077 | 0.572 | 0.329 | 0.70 |
| random_pair | 0.495 ± 0.074 | 0.543 | 0.362 | 0.71 |
| centroid | 1.526 ± 0.377 | 0.147 | 0.151 | 0.34 |
| contrastive-always | 0.649 ± 0.095 | 0.346 | **0.560** | 0.55 |
| contrastive-gated | 0.637 ± 0.082 | 0.376 | **0.560** | 0.57 |

- metric 7: contrastive beats random_pair **10/10** seeds (+0.199) and centroid
  **10/10** (+0.409). Same at D=2. The votes are contributing real information,
  and `centroid` — pure vote-free contraction — is the *worst* arm, so this is
  not the contraction artifact D3 hypothesised.
- FD: contrastive loses to random_pair **0/10** (D=8) and 3/10 (D=2).
- §9.1 requires both ⇒ FAIL.

## Q1, Q3

Q1 fails: contrastive is worse than blind on FD (0.649 vs 0.472). Separately,
**the queries-to-target metric has no resolution left**: with N=20,000 every arm
reaches blind's final FD in 1.0–2.9 rounds and blind itself takes 1.0–1.5, out of
a T=10 budget. The healthy cells are healthy but trivial; the non-trivial cells
are unhealthy. **No cell in the v0.2 grid is both healthy and non-trivial**, and
that, not any arm's performance, is what blocks Q1.

Q3 **reverses run #1's finding**: oracle vs contrastive is 0.595 vs 0.560 on
metric 7 (≈94% retained) and 0.619 vs 0.649 on FD. The contrastive rendering is
essentially lossless. Run #1's "the rendering destroys the gain" was an artifact
of the contaminated harness. **The prompt design (§6.2) is not the problem.**

## Q2 — drop the test

gated beats always 6/10 seeds with a margin inside one seed-s.d. (+0.012 at
D=8), and `random_pair`-gated shows the *same* margin (6/10, +0.016). Per §9.1
that makes gating a generic regulariser, not a consensus test ⇒ **drop**.

The competing "density filter" explanation is also ruled out: reject rate is
flat across t=0 vote quintiles (0.60–0.67 at D=2; 0.83–0.85 at D=8).

**§4.2 verdict: adopt variant (b).** Variant (a) at ε=∞ has a degenerate null
and rejects **100%** of parents, exactly as §4.2 predicted. With (b) the FPR is
0.055–0.069 against nominal 0.05 at ε ≤ 1, and power rises 0.52 → 0.69 → 0.88
across ε = 0.5 / 1 / 4. The gate remains correctly calibrated, as v0.2 said not
to re-litigate; re-run for the record and it holds.

## Q4 — no θ\*, and the sign is wrong

At D=2 FD is insensitive to θ. At D=8 both contrastive arms give
Spearman(FD, θ) = **−0.94** with range > 2 seed-s.d. (0.649 at θ=0 → 0.458 at
θ=90°), while `random_pair` is flat (+0.37, range < 2 s.d.). The θ-dependence
exists *only* for the consensus arm and says the same thing as Q2′: the more
faithfully the generator follows the consensus axis, the worse the result. There
is no fidelity bar to carry to the LLM stage.

## Q5 — the coverage claim is backwards

Per-quintile coverage gain over `random_pair`, D=8, ε=1:

| quintile | Q1 (lowest votes) | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| contrastive-always | **−0.479** | −0.367 | −0.319 | −0.338 | −0.349 |

Every gain is negative, and the loss is *largest* in the bottom quintile. The
method does not fill coverage holes; it deepens them. FAIL in 3 of 4 cells.

---

## Recommendations for v0.3

1. **The blocker is `select`, and it is not in the swappable step.** Three
   consecutive specs have now placed the fix in `vary` (v0.1 step size, v0.2
   §6.0 noise model + §4.3 floor). Measurements say the loss is in the
   nearest-neighbour vote histogram. v0.3 should unfreeze `dp_vote`/`select` and
   test hubness-corrected voting or soft/multi-vote assignment — on the `blind`
   arm alone, as a harness change, before any arm is compared again.
2. **Change the objective, not the direction.** Q0 and metric 7 say the
   direction estimate works. Q2′, Q4 and Q5 say maximising p/q per child is the
   wrong target because it is mode-seeking. If the claim is "fills coverage
   holes", the arm should optimise coverage directly — e.g. move *away* from
   high-vote neighbours, or weight by inverse local synthetic density.
   `contrastive` as specified moves the wrong way for the stated goal.
3. **Build a cell that is both healthy and non-trivial.** D ∈ {2, 8} converge in
   one round; D ≥ 32 diverges. Neither can answer Q1. Either harden the healthy
   cells (worse q₀, smaller budget, fewer rounds) or fix `select` so D=32 becomes
   usable — D=32 already sits at n_eff = 0.37 and is the nearest candidate.
4. **Drop the gate** (Q2), and **drop the M_w⁺ correction** from consideration
   (Q0 metric 5b). Both are now measured, not guessed.
5. **The prompt/rendering design is not on the critical path** (Q3, ≈94%
   retained). Do not spend the LLM budget redesigning §6.2.
6. **Keep the controls.** `random_pair` and `centroid` are what turned this run
   from "contrastive beats blind, ship it" into a correct negative — and they
   also cleared `contrastive` of the contraction-artifact charge, which the FD
   numbers alone would have sustained. Carry both to the LLM stage as §10 says.
