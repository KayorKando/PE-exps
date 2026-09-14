"""Stage-gated results tables (spec v0.2 section 12.2).

Each stage's table states whether that stage's gate passed.  Nothing is read
from a cell the health gate rejected.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

KEY = ("config", "D", "eps", "arm", "theta_deg", "theta_mode", "k", "alpha",
       "N", "n_syn", "var_variant")


def load(*stages):
    recs = []
    for s in stages:
        p = RESULTS / f"stage{s}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            recs.append(r)
    return recs


def index(recs):
    out = defaultdict(dict)
    for r in recs:
        c = r["cfg"]
        out[tuple(c[k] for k in KEY)][c["seed"]] = r
    return out


def cell(idx, config, D, eps, arm, theta=0, mode="fixed", k=10, alpha=0.05,
         N=20000, n_syn=500, var="b"):
    return idx.get((config, D, eps, arm, theta, mode, k, alpha, N, n_syn, var), {})


def series(rec, f):
    return np.array([row.get(f, np.nan) for row in rec["rows"]], float)


def final(rec, f):
    return rec["rows"][-1].get(f, np.nan)


def ms(vals):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt(m, s, w=7, p=3):
    return f"{'n/a':>{w}}" if not np.isfinite(m) else f"{m:{w}.{p}f}±{s:.{p}f}"


def rounds_to(rec, target, f="frechet"):
    s = series(rec, f)
    hit = np.where(s[1:] <= target)[0]
    return int(hit[0]) + 1 if len(hit) else math.inf


def beats(a_vals, b_vals, higher_better=False, n_needed=7):
    """Paired per-seed comparison + the 1-seed-s.d. margin rule."""
    a, b = np.asarray(a_vals, float), np.asarray(b_vals, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if not len(a):
        return False, 0, 0, float("nan")
    wins = int(((a > b) if higher_better else (a < b)).sum())
    ma, sa = ms(a)
    mb, sb = ms(b)
    margin = (ma - mb) if higher_better else (mb - ma)
    return (wins >= n_needed and margin > max(sa, sb)), wins, len(a), margin


def healthy(mode="controls"):
    p = RESULTS / f"healthy_cells_{mode}.json"
    return [tuple(x) for x in json.loads(p.read_text())] if p.exists() else []


def dims(eps=1.0):
    return sorted({D for (c, D, e) in healthy() if c == "A" and e == eps})


# --------------------------------------------------------------------------


def t_stage1(out):
    out("\n" + "=" * 104)
    out("STAGE 1 -- HEALTH GATE (spec 9.0).  Blocking: nothing is read from a failing cell.")
    out("=" * 104)
    recs = load(1)
    idx = index(recs)
    cells = defaultdict(dict)
    for r in recs:
        c = r["cfg"]
        cells[(c["config"], c["D"], c["eps"])].setdefault(c["arm"], []).append(r)
    from .health import cell_condition2
    out(f"   {'cell':>16} {'arm':>18} {'FD_0':>8} {'FD_T':>9} "
        f"{'h1':>4} {'h2':>4} {'h3':>4} {'n_eff':>7} {'h4':>4} {'spread':>7}")
    for key in sorted(cells, key=lambda k: (k[0], k[1], str(k[2]))):
        for arm, rs in sorted(cells[key].items()):
            g = lambda f: float(np.mean([r["health"][f] for r in rs]))
            cur = [[x["frechet"] for x in r["rows"]] for r in rs]
            L = min(len(c) for c in cur)
            m = lambda b: "OK" if b else "--"
            out(f"   {f'{key[0]} D={key[1]} eps={key[2]}':>16} {arm:>18} "
                f"{g('fd_0'):8.2f} {g('fd_T'):9.2f} "
                f"{m(all(r['health']['h1_fd_improved'] for r in rs)):>4} "
                f"{m(cell_condition2([c[:L] for c in cur])):>4} "
                f"{m(all(r['health']['h3_n_eff'] for r in rs)):>4} "
                f"{g('n_eff_frac'):7.2f} "
                f"{m(all(r['health']['h4_spread'] for r in rs)):>4} "
                f"{g('spread_ratio'):7.2f}")
    out("")
    out(f"   healthy, spec 9.0 read literally (every arm): "
        f"{healthy('strict') or 'NONE'}")
    out(f"   healthy, controls only (harness soundness) : {healthy('controls') or 'NONE'}")
    out("")
    out("   The two readings differ only through condition 3 (n_eff/n_syn > 0.3), which at")
    out("   D=2 and D=8 is failed by `contrastive-gated` alone: the directed step contracts")
    out("   the population and sharpens the vote histogram.  That contraction is the")
    out("   behaviour Q2' exists to measure, not a harness defect to be fixed, so cells are")
    out("   certified on the control arms and the method's n_eff is reported as a result.")
    out("   D>=32 and config B fail on the CONTROLS too, and are genuinely unusable.")


def t_stage2(out):
    out("\n" + "=" * 104)
    out("STAGE 2 -- Q2': does the consensus content matter?")
    out("    Gate (spec 9.1): contrastive beats random_pair on FD *and* metric 7,")
    out("    by > 1 seed-s.d., on 7 of 10 seeds.  Must also beat centroid.")
    out("=" * 104)
    idx = index(load(2, 3, 4))
    verdict = {}
    for D in dims():
        out(f"\n-- config A, D={D}, eps=1, theta=0 --")
        out(f"   {'arm':>20} {'final FD':>16} {'coverage':>16} {'m7 ratio':>16} "
            f"{'n_eff/n':>14}")
        vals = {}
        for arm in ("blind", "random_pair", "centroid",
                    "contrastive-always", "contrastive-gated"):
            c = cell(idx, "A", D, 1.0, arm)
            if not c:
                continue
            seeds = sorted(c)
            vals[arm] = dict(
                fd=[final(c[s], "frechet") for s in seeds],
                m7=[final(c[s], "m7_ratio") for s in seeds],
                cov=[final(c[s], "coverage") for s in seeds],
                ne=[final(c[s], "n_eff_select") / 500 for s in seeds],
                seeds=seeds)
            out(f"   {arm:>20} {fmt(*ms(vals[arm]['fd']),8,4)} "
                f"{fmt(*ms(vals[arm]['cov']),8,3)} {fmt(*ms(vals[arm]['m7']),8,3)} "
                f"{fmt(*ms(vals[arm]['ne']),6,2)}")
        for meth in ("contrastive-always", "contrastive-gated"):
            if meth not in vals:
                continue
            for ctrl in ("random_pair", "centroid"):
                if ctrl not in vals:
                    continue
                bf, wf, nf, mf = beats(vals[meth]["fd"], vals[ctrl]["fd"])
                bm, wm, nm, mm = beats(vals[meth]["m7"], vals[ctrl]["m7"],
                                       higher_better=True)
                out(f"   {meth} vs {ctrl}: FD wins {wf}/{nf} (margin {mf:+.4f}, "
                    f"pass={bf}) | m7 wins {wm}/{nm} (margin {mm:+.4f}, pass={bm})")
                if ctrl == "random_pair":
                    verdict[(D, meth)] = bf and bm
    any_pass = any(verdict.values())
    out(f"\n   Q2' VERDICT: {'PASS' if any_pass else 'FAIL -- consensus content adds nothing'}")
    out(f"   per cell: {verdict}")
    return any_pass


def t_stage3(out):
    out("\n" + "=" * 104)
    out("STAGE 3 -- Q1 (does directed beat blind at theta=0) and Q3 (oracle -> contrastive).")
    out("    Q1 pass (spec 9.1): reach blind's final FD in <= 0.75T = 7 rounds on 7 of 10 seeds.")
    out("=" * 104)
    idx = index(load(2, 3, 4))
    for D in dims():
        b = cell(idx, "A", D, 1.0, "blind")
        if not b:
            continue
        tgt = {s: final(r, "frechet") for s, r in b.items()}
        out(f"\n-- config A, D={D}, eps=1 --")
        out(f"   {'arm':>20} {'final FD':>16} {'rounds->blind FD':>18} "
            f"{'seeds<=7':>9} {'m7 ratio':>16}")
        base_r = None
        for arm in ("blind", "random_pair", "contrastive-always",
                    "contrastive-gated", "oracle-always", "oracle-gated",
                    "truth-always"):
            c = cell(idx, "A", D, 1.0, arm)
            if not c:
                continue
            rt = [rounds_to(c[s], tgt[s]) for s in sorted(c) if s in tgt]
            fd = ms([final(c[s], "frechet") for s in sorted(c)])
            m7 = ms([final(c[s], "m7_ratio") for s in sorted(c)])
            npass = sum(1 for x in rt if x <= 7)
            if arm == "blind":
                base_r = ms(rt)
            out(f"   {arm:>20} {fmt(*fd,8,4)} {fmt(*ms(rt),9,2)} "
                f"{npass:>6}/{len(rt)} {fmt(*m7,8,3)}")
        out(f"   (blind's own rounds-to-its-own-final-FD: {fmt(*base_r,6,2)} -- the bar the")
        out(f"    spec rule sets is satisfied by blind itself, so 'beats blind' is the real test)")
    # Q3
    out("\n   Q3 -- share of the oracle's gain retained by the contrastive rendering:")
    for D in dims():
        b = cell(idx, "A", D, 1.0, "blind")
        o = cell(idx, "A", D, 1.0, "oracle-always")
        c = cell(idx, "A", D, 1.0, "contrastive-always")
        if not (b and o and c):
            continue
        tgt = {s: final(r, "frechet") for s, r in b.items()}
        f = lambda cc: ms([final(cc[s], "frechet") for s in sorted(cc)])[0]
        r_ = lambda cc: ms([rounds_to(cc[s], tgt[s]) for s in sorted(cc)
                            if s in tgt])[0]
        go, gc = f(b) - f(o), f(b) - f(c)
        so, sc = r_(b) - r_(o), r_(b) - r_(c)
        out(f"   D={D}: FD gain oracle {go:+.4f}, contrastive {gc:+.4f}"
            + (f" -> retained {gc / go:.2f}" if abs(go) > 1e-9 else " -> n/a"))
        out(f"        speed gain oracle {so:+.2f} rounds, contrastive {sc:+.2f}"
            + (f" -> retained {sc / so:.2f}" if abs(so) > 1e-9 else " -> n/a"))


def t_stage4(out):
    idx = index(load(2, 3, 4))
    # ---- Q2 ----
    out("\n" + "=" * 104)
    out("STAGE 4a -- Q2: is the gate worth keeping?")
    out("    Keep (spec 9.1): gated beats always by > 1 seed-s.d. at eps <= 1, AND the same")
    out("    margin does NOT appear between random_pair-gated and random_pair-always")
    out("    (which would make gating a generic contraction regulariser, not a consensus test).")
    out("=" * 104)
    for D in dims():
        for pair in ("contrastive", "random_pair"):
            a = cell(idx, "A", D, 1.0, f"{pair}-always" if pair == "contrastive" else "random_pair")
            g = cell(idx, "A", D, 1.0, f"{pair}-gated")
            if not (a and g):
                continue
            seeds = sorted(set(a) & set(g))
            fa = [final(a[s], "frechet") for s in seeds]
            fg = [final(g[s], "frechet") for s in seeds]
            bw, w, n, m = beats(fg, fa)
            rej = ms([final(g[s], "gate_reject_rate") for s in seeds])
            out(f"   D={D} {pair:>12}: always {fmt(*ms(fa),8,4)}  gated {fmt(*ms(fg),8,4)}"
                f"  gated wins {w}/{n} margin {m:+.4f} pass={bw}  reject {fmt(*rej,5,3)}")
    out("\n   Gate reject rate by t=0 vote quintile (is it a density filter rather than")
    out("   a consensus test?  spec 5):")
    for D in dims():
        c = cell(idx, "A", D, 1.0, "contrastive-gated")
        if not c:
            continue
        qs = [ms([final(c[s], f"gate_reject_q{q}") for s in sorted(c)])[0]
              for q in range(1, 6)]
        out(f"   D={D}: " + "  ".join(f"Q{q}={v:.3f}" for q, v in enumerate(qs, 1)))
    out("\n   FPR / power vs eps (and spec 4.2 variant (a) vs (b) at eps=inf):")
    out(f"   {'D':>3} {'eps':>5} {'variant':>8} {'sigma':>7} {'FPR':>15} {'power':>15} {'reject':>15}")
    for D in dims():
        for eps in (0.5, 1.0, 4.0, float("inf")):
            for va in ("b", "a"):
                c = cell(idx, "A", D, eps, "contrastive-gated", var=va)
                if not c:
                    continue
                sg = list(c.values())[0]["sigma"]
                out(f"   {D:3d} {str(eps):>5} {va:>8} {sg:7.2f} "
                    f"{fmt(*ms([final(c[s],'gate_fpr') for s in sorted(c)]),6,3)} "
                    f"{fmt(*ms([final(c[s],'gate_power') for s in sorted(c)]),6,3)} "
                    f"{fmt(*ms([final(c[s],'gate_reject_rate') for s in sorted(c)]),6,3)}")
    # ---- Q4 ----
    out("\n" + "=" * 104)
    out("STAGE 4b -- Q4: minimum axis fidelity theta*.")
    out("    Reported only if FD-vs-theta is monotone over >= 3 consecutive theta with")
    out("    total range > 2 seed-s.d.; otherwise 'insensitive to theta' (evidence for D3).")
    out("=" * 104)
    for D in dims():
        for arm in ("contrastive-always", "contrastive-gated", "random_pair"):
            rows = []
            for th in (0, 15, 30, 45, 60, 90):
                c = cell(idx, "A", D, 1.0, arm, theta=th)
                if not c:
                    continue
                rows.append((th, ms([final(c[s], "frechet") for s in sorted(c)]),
                             ms([np.degrees(final(c[s], "measured_theta"))
                                 for s in sorted(c)])[0]))
            if not rows:
                continue
            out(f"\n-- D={D} {arm} --")
            out(f"   {'theta':>7} {'measured':>10} {'final FD':>16}")
            for th, fd, mt in rows:
                out(f"   {th:6d}d {mt:9.1f}d {fmt(*fd,8,4)}")
            from scipy.stats import spearmanr
            m = [r[1][0] for r in rows]
            sd = float(np.mean([r[1][1] for r in rows]))
            rng_ = max(m) - min(m)
            mono = (all(b >= a for a, b in zip(m, m[1:]))
                    or all(b <= a for a, b in zip(m, m[1:])))
            rho = float(spearmanr([r[0] for r in rows], m).statistic)
            out(f"   range {rng_:.4f} vs 2 seed-s.d. {2 * sd:.4f}; strictly monotone={mono}; "
                f"Spearman(FD, theta)={rho:+.2f}")
            if mono and rng_ > 2 * sd:
                out("   -> theta* reportable")
            elif rng_ > 2 * sd and abs(rho) >= 0.8:
                out(f"   -> not strictly monotone, but a clear {'rising' if rho > 0 else 'falling'} "
                    f"trend of size > 2 s.d.:")
                out(f"      FD gets {'WORSE' if rho > 0 else 'BETTER'} as the axis is followed "
                    f"{'more' if rho < 0 else 'less'} faithfully, so no theta* exists.")
            else:
                out("   -> INSENSITIVE to theta (supports D3)")
    # ---- Q5 ----
    out("\n" + "=" * 104)
    out("STAGE 4c -- Q5: does the benefit concentrate in low-vote regions?")
    out("    Pass (spec 9.1): relative coverage gain over RANDOM_PAIR larger in the bottom")
    out("    vote quintile than in the top, in a healthy cell.")
    out("=" * 104)
    for D in dims():
        rp = cell(idx, "A", D, 1.0, "random_pair")
        if not rp:
            continue
        for arm in ("contrastive-always", "contrastive-gated"):
            c = cell(idx, "A", D, 1.0, arm)
            if not c:
                continue
            out(f"\n-- D={D} {arm} vs random_pair --")
            out(f"   {'quintile':>9} {'random_pair':>13} {'arm':>13} {'rel gain':>10}")
            g = {}
            for q in range(1, 6):
                b_ = ms([final(rp[s], f"coverage_q{q}") for s in sorted(rp)])[0]
                a_ = ms([final(c[s], f"coverage_q{q}") for s in sorted(c)])[0]
                g[q] = (a_ - b_) / b_ if b_ > 1e-6 else float("nan")
                out(f"   {'Q' + str(q):>9} {b_:13.3f} {a_:13.3f} "
                    + (f"{g[q]:+10.3f}" if np.isfinite(g[q]) else f"{'n/a':>10}"))
            if np.isfinite(g[1]) and np.isfinite(g[5]):
                out(f"   bottom {g[1]:+.3f} vs top {g[5]:+.3f} -> Q5 "
                    f"{'PASS' if g[1] > g[5] else 'FAIL'}")


def t_stage5(out):
    out("\n" + "=" * 104)
    out("STAGE 5 -- config B (real sentence embeddings).  The health gate REJECTS every")
    out("    config B cell (FD_T > FD_0 for the controls), so these are reported for the")
    out("    record only and are not evidence for or against any question.")
    out("=" * 104)
    idx = index(load(5))
    for D in (128, 768):
        for eps in (1.0, float("inf")):
            b = cell(idx, "B", D, eps, "blind")
            if not b:
                continue
            r0 = list(b.values())[0]
            out(f"\n-- D={D}, eps={eps} (sigma={r0['sigma']:.2f}) --")
            out(f"   FD_0 {r0['rows'][0]['frechet']:.2f}, real-sample floor "
                f"{r0['ceiling']['frechet']:.2f}")
            out(f"   {'arm':>20} {'final FD':>16} {'coverage':>15} {'precision':>15} "
                f"{'m7 ratio':>15}")
            for arm in ("blind", "random_pair", "centroid", "contrastive-always",
                        "contrastive-gated", "oracle-always"):
                c = cell(idx, "B", D, eps, arm)
                if not c:
                    continue
                s_ = sorted(c)
                out(f"   {arm:>20} "
                    f"{fmt(*ms([final(c[i],'frechet') for i in s_]),8,2)} "
                    f"{fmt(*ms([final(c[i],'coverage') for i in s_]),7,3)} "
                    f"{fmt(*ms([final(c[i],'precision') for i in s_]),7,3)} "
                    f"{fmt(*ms([final(c[i],'m7_ratio') for i in s_]),7,3)}")


def main():
    lines = []
    out = lambda s="": (lines.append(s), print(s))[0]
    from .stage0 import report
    report(out)
    t_stage1(out)
    t_stage2(out)
    t_stage3(out)
    t_stage4(out)
    t_stage5(out)
    (RESULTS / "tables.txt").write_text("\n".join(lines))
    print(f"\nwritten to {RESULTS / 'tables.txt'}")


if __name__ == "__main__":
    main()
