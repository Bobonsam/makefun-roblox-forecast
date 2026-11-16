"""Backtest-only knob grid (iteration 1). Monkeypatches main's 4 knobs; the
practice window reads are never touched here."""
import json, sys, itertools
sys.path.insert(0, "/home/hatch/workspace/makefun/roblox-repo2")
import main as M
from local_scorer import score

KIT = "/home/hatch/workspace/makefun/roblox-up-and-coming-forecast-01-kit/input/history.ndjson"
raw = [json.loads(l) for l in open(KIT) if l.strip()]
raw.sort(key=lambda d: d["at"])
CUTS = [13, 17, 21, 25, 29]

def to_scorer_reads(reads):
    out = []
    for d in reads:
        chart = sorted(d["up_and_coming"], key=lambda r: r["rank"])
        ids = [r["universe_id"] for r in chart]
        out.append({"at": d.get("at"), "ids": ids,
                    "place_of": {g: i + 1 for i, g in enumerate(ids)}})
    return out

CASES = []
for i in CUTS:
    trunc = raw[: i + 1]
    t_cut = M.parse_time(trunc[-1]["at"])
    fwd = [d for d in raw[i + 1:] if (M.parse_time(d["at"]) - t_cut).total_seconds() <= 72 * 3600]
    CASES.append((trunc, to_scorer_reads(fwd)))

def evaluate():
    tot = 0.0
    for trunc, sr in CASES:
        tot += score(M.forecast(trunc, 10, 72.0), sr)["log_score"]
    return tot / len(CASES)

results = []
for budget in (0.80, 0.86, 0.92):
    for sigma in (2.6, 3.4, 4.4):
        for damp in (0.15, 0.35, 0.60):
            for pull in (0.0, 0.18, 0.35):
                M.NAMED_BUDGET, M.KERNEL_SIGMA, M.VELOCITY_DAMP, M.PLAYING_PULL = budget, sigma, damp, pull
                s = evaluate()
                results.append((s, budget, sigma, damp, pull))
results.sort(reverse=True)
for s, b, sg, d, p in results[:12]:
    print(f"backtest {s:.3f}  budget {b}  sigma {sg}  damp {d}  pull {p}")
print("...")
for s, b, sg, d, p in results[-3:]:
    print(f"backtest {s:.3f}  budget {b}  sigma {sg}  damp {d}  pull {p}")
