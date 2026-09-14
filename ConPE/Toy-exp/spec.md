# Toy Experiment Spec — Consensus-Guided Variation for Private Evolution

Kando Hsieh · Eli Lab · v0.2 · 2026-09-10
Supersedes v0.1 (2026-09-10). Changes driven by run #1 (566 runs).

---

## 0. What changed and why

Run #1 could not answer Q1. Three defects, all in the harness rather than in the method:

| Defect | Evidence from run #1 | Fix |
|---|---|---|
| **D1. Shared loop diverges at $D \ge 128$** | Every arm, including `blind`, `truth-always`, and every $\varepsilon = \infty$ cell, ends with $\text{FD}_T > \text{FD}_0$. `truth-always` reaches coverage 1.000 with FD 112 vs $\text{FD}_0 = 98.9$ — the population is over-dispersed, not converged. `n_eff/n_syn` drops to 0.09–0.23. | Manifold-local variation noise (§6.0), smaller step, vote floor. Health gate (§9.0) blocks all read-off until fixed. |
| **D2. Direction quality measured where no direction exists** | metric 5 read at $t=T$; in $D=2$, $\varepsilon=\infty$, $\text{FD}_T = 0.03$ vs floor 0.01, so $q_T \approx p$ and $\nabla\log(p/q) \approx 0$ — the reference vector is KDE jitter. Contradicted by `oracle-gated` reaching blind's final FD in 3.4 rounds vs blind's 6.2. | Measure per round, gate on $\|\nabla\log(p/q)\|$, report the $t=0$ value as primary (§7.2). |
| **D3. No control isolating "consensus" from "move toward an existing neighbour"** | `contrastive` beats `blind` on FD and precision (0.99 vs 0.78) in exactly the cells where `blind` is diffusing; $\theta = 45°$ beats $\theta = 0$ in $D{=}128$. Both consistent with a contraction artifact. | New arms `random_pair` and `centroid` (§6.6), mandatory in every comparison. |

Two results from run #1 survive and are **not** re-litigated: the gate is correctly calibrated (FPR 0.048–0.072 against nominal $\alpha$ = 0.01/0.05/0.05→0.2, power ≈ 0.5 at $\varepsilon = 1$, $k = 10$), and the MC null overlays $\chi^2_r$. Re-run once for the record; do not re-tune.

Also new in v0.2: finite-sample vote noise is now modelled explicitly (§4.2), because $\varepsilon = \infty$ cells still carry Poisson vote noise of the same order as the signal (mean ≈ 4 votes per synthetic point, s.d. ≈ 2, at $N/n_\text{syn} = 4$).

---

## 1. Questions and kill criteria

Q0 is new and blocking. Q1–Q5 are unchanged in intent; their decision rules are tightened.

| # | Question | Kill / gate condition |
|---|----------|-----------------------|
| **Q0** | Is the direction estimate $\hat g_i$ informative about $\nabla \log(p/q)$ at all, in the regime where a direction exists? | If $\cos(\hat g, \nabla\log(p/q))$ at $t=0$, $\varepsilon=\infty$, $D=2$, $N$ large is not clearly above the random-direction baseline $\approx 1/\sqrt{D}$, **the line dies here.** Cheapest possible kill; one round, no full loop. |
| Q1 | Does directed variation beat blind at $\theta = 0$? | Fail at $D=128$ in a *healthy* loop ⇒ stop. |
| **Q2′** | Does the consensus content matter — does `contrastive` beat `random_pair`? | If `contrastive` ≈ `random_pair`, the votes contribute nothing and the method reduces to "step toward a neighbour". Report as such; the deck's central claim is void. |
| Q2 | Does gating add anything over always-directed? | If gated ≈ always **and** `random_pair` gated ≈ `random_pair` always, drop the test. |
| Q3 | How much oracle gain survives contrastive rendering? | Informative. |
| Q4 | Minimum axis fidelity $\theta^*$. | Informative. Only reported if the FD-vs-$\theta$ curve is monotone and its range exceeds 2 seed-s.d.; otherwise report "insensitive to $\theta$" and treat as evidence for D3. |
| Q5 | Does the benefit concentrate in low-vote regions? | Only read in a healthy loop, and only against `random_pair`, not against `blind`. |

Order of execution is Q0 → health gate → Q2′ → Q1 → Q2, Q3, Q4, Q5.

---

## 2. Pipeline interface

Unchanged from v0.1 §2, with one addition:

```
Z_0        = init(n_syn)
for t in 1..T:
    v_tilde = dp_vote(P, Z_t, sigma)              # now includes finite-sample noise, §4.2
    parents = select(Z_t, v_tilde, n_syn)         # vote floor, §4.3
    Z_{t+1} = vary(parents, Z_t, v_tilde, cfg)    # only swappable step
    health  = check_health(Z_{t+1}, Z_0, t)       # NEW, §9.0 — aborts the run on divergence
    log_metrics(Z_{t+1}, t)
```

`check_health` runs inside the loop and aborts early with a flag rather than producing numbers that will later be voided.

**Rule (unchanged):** no arm may read `P`, true densities, or noise-free votes inside `vary`.

---

## 3. Data

### 3.1 Private distribution $p$
Config A (synthetic mixture) and Config B (real sentence-encoder embeddings) as in v0.1 §3.1.

### 3.2 Sizes — CHANGED
- $N = 20{,}000$ private points (was 2,000).
- $n_\text{syn} = 500$ (unchanged), so $N/n_\text{syn} = 40$ votes per synthetic point on average, Poisson s.d. ≈ 6.3, relative noise 0.16 (was 0.5).
- Additionally run $N \in \{2{,}000,\ 20{,}000,\ 100{,}000\}$ for Q0 only, to separate finite-sample vote noise from DP noise as the cause of any low $\cos$.

### 3.3 Initial $q_0$
Unchanged (deliberately misaligned; head modes only).

### 3.4 Dimensions
$D \in \{2, 128, 768\}$. $D = 2$ is now the **primary diagnostic** cell for Q0 (the only place ground-truth $\nabla\log(p/q)$ is trustworthy); $D = 128$ remains the read-off cell for Q1–Q5. $D = 768$ is a breakage check.

---

## 4. Noise model

### 4.1 DP noise
$\sigma$ derived from $(\varepsilon, \delta, T, N)$ via the Gaussian mechanism accountant, exactly as v0.1 §4. Defaults $\varepsilon \in \{0.5, 1, 4, \infty\}$, $\delta = 1/N$, $T = 10$.

### 4.2 Finite-sample vote noise — NEW
$\varepsilon = \infty$ does **not** mean noiseless votes: the count $v_j$ is a multinomial draw over $N$ private points, so $\text{sd}(v_j) \approx \sqrt{N/n_\text{syn}}$. Report the total vote SNR
$$\text{SNR} = \frac{\text{sd across } j \text{ of } \mathbb E[v_j]}{\sqrt{\sigma^2 + N/n_\text{syn}}}$$
alongside $\sigma$ in every table. Any claim about "no noise" refers to $\sigma = 0$ only.

The hypothesis test (§5) currently models the DP term only. In the $\sigma \to 0$ limit its null is therefore wrong. Two options, both to be run:
- (a) leave the test as specified and report that its calibration degrades as $\sigma \to 0$ (expected: over-rejection);
- (b) inflate the null variance to $\sigma^2 + \hat N/n_\text{syn}$ and re-check FPR.
If (b) restores calibration at $\varepsilon = \infty$, adopt it and note it as a correction to the deck.

### 4.3 Vote floor — NEW
Apply Aug-PE's threshold: zero out $\tilde v_j$ below a floor $\kappa$ before `select`, to stop noise-only points from being resampled. Default $\kappa$ = the 10th percentile of the null vote distribution. This is part of standard PE and its absence contributed to D1.

---

## 5. Gating

Formulas unchanged from v0.1 §5 ($g_i$, $M$, $s$, $\Sigma = M - \frac1k ss^\top$, $Q_i = \sigma^{-2} g_i^\top \Sigma^+ g_i$, MC null $B = 2000$, $\chi^2_r$ fast path).

Additions:
- **Duplicate handling — NEW.** When $\|z_j - z_i\| < 10^{-8}$ (duplicate parents are common: unique-parent fraction was 0.25–0.57 in run #1), the neighbour is **dropped** from the kNN set and $k$ reduced accordingly, rather than yielding a NaN or an arbitrary $\hat d_j$. Log the effective $k$ distribution per round. Silent handling here is a candidate cause of a degraded $\hat g$.
- **Variance term** per §4.2(b) when that variant is active.
- Log per round: reject rate, effective $k$, and reject rate **split by vote quintile** — if rejections concentrate in high-vote regions, gating is acting as a density filter rather than a consensus test, which is a competing explanation for Q2.

---

## 6. `vary` implementations

### 6.0 Shared variation noise — CHANGED (fixes D1)
All arms add exploration noise $\xi$. In v0.1 this was isotropic, $\xi \sim \mathcal N(0, \tau^2 I)$, which in $D \ge 128$ is a random walk off the data manifold — a poor stand-in for an LLM rewrite, which stays on the text manifold by construction.

Replace with manifold-local noise: let $C_i$ be the sample covariance of $z_i$'s $k_\text{cov} = 20$ nearest neighbours in $Z_t$, regularised as $C_i + \lambda \text{tr}(C_i)/D \cdot I$ with $\lambda = 0.1$. Then
$$\xi \sim \mathcal N(0, \tau^2 \, \tilde C_i), \qquad \tilde C_i = C_i / \text{tr}(C_i) \cdot D$$
so the trace is normalised and $\tau$ retains its meaning as a step scale. Alternative to run as a check: project $\xi$ onto the top-$m$ PCA subspace of $Z_t$ ($m = 20$).

Set $\tau$ so that the **median displacement per round equals $s$** for every arm (fair movement budget), with $s = 0.25 \times$ median nearest-neighbour distance in $Z_t$ (was a free parameter; now scale-adaptive and much smaller).

### 6.1 `blind` — baseline
$z' = z_i + \xi$ with $\xi$ per §6.0.

### 6.2 `directed_contrastive` — the method
$z_+ = \arg\max_j \langle \hat d_j, \hat g_i\rangle$, $z_- = \arg\min_j \langle \hat d_j, \hat g_i\rangle$, axis $\hat a = (z_+ - z_-)/\|z_+ - z_-\|$,
$$z' = z_i + s(\cos\theta\,\hat a + \sin\theta\,\hat r) + \xi', \qquad \xi' \sim \mathcal N(0, \tau_\text{small}^2 \tilde C_i)$$
$\theta$ sweep unchanged. One-sided case: use $\hat g_i$ from the available side; log frequency.

### 6.3 `directed_oracle` — upper bound (toy only)
$\hat a = \hat g_i$ directly.

### 6.4 `directed_truth` — ceiling (config A only)
$\hat a = \widehat{\nabla \log(p/q_t)}$ normalised. **Changed:** $q_t$ estimated by KDE with a bandwidth chosen by cross-validation, and this arm is reported only for $D = 2$. A 500-point KDE in 128 dimensions does not estimate a density gradient; run #1's $D{=}128$ `truth` numbers were not a ceiling and should not have been read as one.

### 6.5 Gating variants
Each directed arm in `always` ($\alpha = 1$) and `gated` ($\alpha = 0.05$) versions.

### 6.6 Control arms — NEW (fixes D3)
- **`random_pair`**: identical to 6.2 except $z_+, z_-$ are drawn uniformly at random from the kNN set, ignoring votes entirely. Isolates "step along an axis between two existing neighbours" from "step along the consensus axis". **This is the true baseline for Q2′ and Q5.**
- **`centroid`**: $\hat a = (\bar z_{kNN} - z_i)/\|\cdot\|$, ignoring votes. Isolates pure contraction toward local mass.

Both carry the same $s$, $\theta$, and noise as 6.2. A result is only attributable to consensus if `contrastive` beats *both*.

---

## 7. Metrics

### 7.1 Transferable (computable at the LLM and image stages)
1. **Fréchet distance** to a held-out reference sample.
2. **Precision / recall / coverage** (Kynkäänniemi, $k = 5$). **Note:** the recall estimator saturates for over-dispersed populations; always report coverage and precision together with FD, and never read recall alone as quality.
3. **Low-vote-region coverage**, by $t{=}0$ vote quintile, computed **against `random_pair`**, not against `blind`.
4. **Queries-to-target**: `vary` calls until FD falls below `blind`'s final FD. Void whenever the health gate flags the cell.

### 7.2 Direction quality — CHANGED (fixes D2)
5. **$\cos(\hat g_i, \nabla \log(p/q_t)(z_i))$**, config A, $D = 2$ only.
   - Reported **per round**, with $t = 0$ as the headline value.
   - Restricted to primes with $\|\nabla\log(p/q_t)(z_i)\|$ above the 50th percentile; points where the reference is near zero are excluded, and the excluded fraction is reported.
   - Compared against the random-direction baseline $\mathbb E[\cos] \approx \sqrt{2/(\pi D)}$ and against a permutation null (shuffle $\tilde v$ across neighbours, recompute $\hat g$).
   - Report $\|\nabla\log(p/q_t)\|$ percentiles alongside, so a low $\cos$ can be attributed to a vanishing reference rather than to a bad estimate.
5b. **Geometry-corrected cosine — NEW.** For a locally linear vote field, $g_i \approx M_w \nabla v$ with $M_w = \sum_j \|z_j - z_i\| \hat d_j \hat d_j^\top$. Report $\cos(M_w^{+} g_i, \nabla\log(p/q))$ as well. The gap between 5 and 5b is the share of direction error attributable to neighbourhood geometry rather than to vote noise, and it tells you whether a $M_w^{+}$ correction belongs in the method.
6. **Gate FPR and power** (config A, ground-truth flat vs non-flat vote profile), by $\varepsilon$ and by vote quintile.
7. **Fraction of children landing at higher $p/q$ than their parent.** Promoted to the primary direction-quality metric for $D \ge 128$, where metric 5 is unavailable ($p$ analytic, $q$ proxied by the vote field). Reported per arm, including `random_pair`.

---

## 8. Grid

Fixed: $n_\text{syn} = 500$, $T = 10$, one child per parent, seeds $\in \{0..4\}$ (seeds $\{0..9\}$ for any cell used in a pass/fail decision).

| Factor | Values |
|---|---|
| Data config | A, B |
| $D$ | 2, 128, 768 |
| $N$ | 20,000 (2,000 / 100,000 for Q0 only) |
| Arm | blind, **random_pair**, **centroid**, contrastive-always, contrastive-gated, oracle-always, oracle-gated, truth ($D{=}2$ only) |
| $\theta$ | 0, 15, 30, 45, 60, 90° |
| $\varepsilon$ | 0.5, 1, 4, $\infty$ |
| $k$ | 5, 10, 20 |
| $\alpha$ | 0.01, 0.05, 0.2 |

**Run order — strictly sequential, stop where a gate fires:**

- **Stage 0 (Q0, ~1 hour):** single round, no loop. Config A, $D = 2$, $\varepsilon = \infty$, $N \in \{2\text{k}, 20\text{k}, 100\text{k}\}$. Compute metric 5, 5b, and the permutation null at $t = 0$. Then $D = 128$ likewise. **Gate: $\cos$ must exceed both baselines. If not, stop and write the negative result.**
- **Stage 1 (health):** config A and B, $D \in \{2, 128\}$, arms {blind, contrastive-gated}, $\varepsilon \in \{1, \infty\}$. **Gate: $\text{FD}_T < \text{FD}_0$ for every arm in every cell** (§9.0). Iterate on $\tau$, $s$, $\kappa$, and the noise model until it holds. No Q is read before this passes.
- **Stage 2 (Q2′):** config A, $D = 128$, $\varepsilon = 1$, $\theta = 0$, arms {blind, random_pair, centroid, contrastive-always, contrastive-gated}. **Gate: contrastive must beat random_pair and centroid.**
- **Stage 3 (Q1, Q3):** add oracle and truth arms.
- **Stage 4 (Q2, Q4, Q5):** sweeps over $\varepsilon$, $k$, $\alpha$, $\theta$; per-quintile analysis vs `random_pair`.
- **Stage 5:** config B, $D \in \{128, 768\}$, transferable metrics only.

---

## 9. Decision rules

### 9.0 Health gate — NEW, blocking
A cell is **healthy** iff, for every arm in it:
1. $\text{FD}_T < \text{FD}_0$;
2. $\text{FD}$ is non-increasing over the last 3 rounds (allowing one seed-s.d. of slack);
3. $n_\text{eff}/n_\text{syn} > 0.3$ at $t = T$;
4. mean pairwise distance in $Z_T$ is within $[0.5, 2] \times$ that of a real sample of the same size (catches both collapse and dispersion).

Results from an unhealthy cell are not reported as evidence for or against any Q — not even as "void" rows in a results table. Fix the harness and re-run.

### 9.1 Per-question rules
- **Q0 pass:** $\cos$ at $t{=}0$, $D{=}2$, $\varepsilon=\infty$, $N{=}100$k exceeds the permutation null by more than 3 permutation-s.d., and shows a monotone increase in $N$.
- **Q2′ pass:** `contrastive` beats `random_pair` on FD **and** on metric 7 by more than one seed-s.d., on 7 of 10 seeds.
- **Q1 pass:** in a healthy $D{=}128$ cell, `contrastive` reaches `blind`'s final FD in $\le 0.75T$ rounds on 7 of 10 seeds.
- **Q2 keep-the-test:** `gated` beats `always` by more than one seed-s.d. at $\varepsilon \le 1$, **and** the same margin does not appear between `random_pair`-gated and `random_pair`-always (which would show gating is a generic contraction regulariser, not a consensus test).
- **Q4 report $\theta^*$** only if FD-vs-$\theta$ is monotone over $\ge 3$ consecutive $\theta$ values with total range $> 2$ seed-s.d.
- **Q5 pass:** relative coverage gain over `random_pair` in the bottom vote quintile exceeds that in the top quintile, in a healthy cell, on both config A and config B.

A null at Q0 or Q2′ is the most informative outcome available and should be written up as such. Do not add machinery to rescue it.

---

## 10. Transfer checklist (toy → LLM → image)

Unchanged from v0.1 §10. What changes at transfer: `vary` (prompt with parent + positive/negative example; or image-variation call), `embed` (sentence encoder / CLIP), and $\theta$ becomes measured (embed the child, project displacement onto $\hat a$, take $\arccos$).

What does not change: `dp_vote`, `select`, gating, $\sigma$ derivation, metrics 1–4, arms (including `random_pair`, which transfers directly and should be run at the LLM stage too), decision rules.

Pre-registered LLM-stage bar: if measured median $\theta > \theta^*$, do not spend the full budget; report the fidelity gap and either redesign the prompt or stop.

---

## 11. Out of scope for v0.2

Variant re-filtering, cross-round memory, multiple children per parent, adaptive $\alpha$, Laplace mechanism, convergence theory. Also deferred: the $M_w^{+}$ correction as a *method* component — v0.2 only measures whether it would help (metric 5b).

---

## 12. Deliverables

1. `pe_toy/` — `pipeline.py`, `data.py`, `privacy.py`, `gate.py`, `vary.py`, `metrics.py`, `health.py` (new), `run_grid.py`. Every arm is one function with the §2 signature.
2. Stage-gated results: one table per stage, each stating whether its gate passed.
3. Figures: (a) FD vs round, all arms, healthy $D{=}128$ cell; (b) $\cos$ vs round and vs $N$ (Q0), with permutation null band; (c) per-quintile coverage vs `random_pair`; (d) MC null vs $\chi^2_r$ overlay, at $\varepsilon = 1$ and $\varepsilon = \infty$; (e) $D{=}2$ scatter of one round showing parent, $z_\pm$, child, and the true gradient arrow.
4. A one-page summary naming which gates fired and what was concluded.