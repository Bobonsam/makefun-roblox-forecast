"""Step-4 candidate evaluation on the SAME 5 in-history pseudo-window cuts.

Candidates: kernel (main.forecast, Step-3 knobs), empirical (main2) at three
budgets, and blends w_emp in {0.3, 0.5, 0.7} using the best-budget empirical.
Practice-window scores are computed report-only at the end.
"""
import json, sys
sys.path.insert(0, "/home/hatch/workspace/makefun/roblox-repo2")
import main as K
import main2 as E
from local_scorer import score, load_reads

KIT = "/home/hatch/workspace/makefun/roblox-up-and-coming-forecast-01-kit/input/history.ndjson"
raw = [json.loads(l) for l in open(KIT) if l.strip()]
raw.sort(key=lambda d: d["at"])
CUTS = [13, 17, 21, 25, 29]
WIN_READS = load_reads("/home/hatch/workspace/makefun/roblox-window-reads.ndjson")

def to_scorer_reads(reads):
    out = []
    for d in reads:
        chart = sorted(d["up_and_coming"], key=lambda r: r["rank"])
        ids = [r["universe_id"] for r in chart]
        out.append({"at": d.get("at"), "ids": ids,
                    "place_of": {g: i + 1 for i, g in enumerate(ids)}})
    return out

def blend(fa, fb, w):
    """w * fb + (1-w) * fa, per (game, place); top10 by blended totals/exp place."""
    pa = {(r["universe_id"], r["place"]): r["p"] for r in fa["probabilities"]}
    pb = {(r["universe_id"], r["place"]): r["p"] for r in fb["probabilities"]}
    rows, totals, wsum, esum = [], {}, {}, {}
    for (g, k) in sorted(set(pa) | set(pb)):
        p = (1 - w) * pa.get((g, k), 0.0) + w * pb.get((g, k), 0.0)
        if p >= 0.0012:
            rows.append({"universe_id": g, "place": k, "p": round(p, 6)})
            totals[g] = totals.get(g, 0.0) + p
            wsum[g] = wsum.get(g, 0.0) + p
            esum[g] = esum.get(g, 0.0) + p * k
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    exp = {g: (esum[g] / wsum[g] if wsum[g] else 99.0) for g in totals}
    by_total = sorted(totals, key=lambda g: (-totals[g], exp[g], g))
    top10 = sorted(by_total[:10], key=lambda g: (exp[g], g))
    return {"probabilities": rows, "top10": top10}

# per-cut parent forecasts
cases = []
for i in CUTS:
    trunc = raw[: i + 1]
    t_cut = K.parse_time(trunc[-1]["at"])
    fwd = [d for d in raw[i + 1:] if (K.parse_time(d["at"]) - t_cut).total_seconds() <= 72 * 3600]
    cases.append((trunc, to_scorer_reads(fwd)))

def mean_bt(fcs):
    return sum(score(fc, sr)["log_score"] for fc, (_, sr) in zip(fcs, cases)) / len(cases)

kernel_fcs = [K.forecast(trunc, 10, 72.0) for trunc, _ in cases]
print(f"KERNEL            backtest {mean_bt(kernel_fcs):.3f}   practice {score(K.forecast(raw, 10, 72.0), WIN_READS)['log_score']:.3f}")

emp_by_budget = {}
for budget in (0.74, 0.80, 0.86):
    fcs = [E.empirical_forecast(trunc, 10, 72.0, budget=budget) for trunc, _ in cases]
    emp_by_budget[budget] = fcs
    print(f"EMPIRICAL b={budget}  backtest {mean_bt(fcs):.3f}   practice {score(E.empirical_forecast(raw, 10, 72.0, budget=budget), WIN_READS)['log_score']:.3f}")

best_budget = max(emp_by_budget, key=lambda b: mean_bt(emp_by_budget[b]))
print(f"best empirical budget on backtests: {best_budget}")
emp_full = E.empirical_forecast(raw, 10, 72.0, budget=best_budget)
kernel_full = K.forecast(raw, 10, 72.0)
for w in (0.3, 0.5, 0.7):
    fcs = [blend(kf, ef, w) for kf, ef in zip(kernel_fcs, emp_by_budget[best_budget])]
    prac = score(blend(kernel_full, emp_full, w), WIN_READS)["log_score"]
    print(f"BLEND w_emp={w}    backtest {mean_bt(fcs):.3f}   practice {prac:.3f}")

# per-cut detail for the three families at their best settings
print("\nper-cut (kernel / empirical-best / blend w=0.5):")
blend_fcs = [blend(kf, ef, 0.5) for kf, ef in zip(kernel_fcs, emp_by_budget[best_budget])]
for idx, (_, sr) in enumerate(cases):
    print(f"  cut {CUTS[idx]}: {score(kernel_fcs[idx], sr)['log_score']:.3f} / "
          f"{score(emp_by_budget[best_budget][idx], sr)['log_score']:.3f} / "
          f"{score(blend_fcs[idx], sr)['log_score']:.3f}")
