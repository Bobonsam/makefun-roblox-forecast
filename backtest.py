"""Pseudo-window backtests INSIDE the history only (helper, not the entry).

For each cut index i: pretend the history ends at read i, run main.forecast on
the truncated history, and score against the real subsequent history reads
that fall within the following 72h. Never touches the practice window reads.
Also scores the kit persistence baseline on the same cuts for reference.
"""
import json, sys
sys.path.insert(0, "/home/hatch/workspace/makefun/roblox-repo2")
import main as M
from local_scorer import score, load_reads
from datetime import datetime

KIT = "/home/hatch/workspace/makefun/roblox-up-and-coming-forecast-01-kit/input/history.ndjson"
raw = [json.loads(l) for l in open(KIT) if l.strip()]
raw.sort(key=lambda d: d["at"])

def to_scorer_reads(reads):
    out = []
    for d in reads:
        chart = sorted(d["up_and_coming"], key=lambda r: r["rank"])
        ids = [r["universe_id"] for r in chart]
        out.append({"at": d.get("at"), "ids": ids,
                    "place_of": {g: i + 1 for i, g in enumerate(ids)}})
    return out

def baseline(truncated):
    last = sorted(truncated[-1]["up_and_coming"], key=lambda r: r["rank"])
    chart = [r["universe_id"] for r in last]
    probs = []
    for place in range(1, 11):
        probs.append({"universe_id": chart[place - 1], "place": place, "p": 0.5})
        if place == 1:
            probs.append({"universe_id": chart[1], "place": 1, "p": 0.3})
            continue
        probs.append({"universe_id": chart[place - 2], "place": place, "p": 0.15})
        if place < len(chart):
            probs.append({"universe_id": chart[place], "place": place, "p": 0.15})
    return {"probabilities": probs, "top10": chart[:10]}

def run(cuts):
    mine, base = [], []
    for i in cuts:
        trunc = raw[: i + 1]
        t_cut = M.parse_time(trunc[-1]["at"])
        fwd = [d for d in raw[i + 1:] if (M.parse_time(d["at"]) - t_cut).total_seconds() <= 72 * 3600]
        if not fwd:
            continue
        fc = M.forecast(trunc, 10, 72.0)
        sr = to_scorer_reads(fwd)
        m = score(fc, sr)["log_score"]
        b = score(baseline(trunc), sr)["log_score"]
        mine.append(m); base.append(b)
        print(f"cut #{i} ({trunc[-1]['at']}, {len(fwd)} fwd reads): mine {m:.3f}  baseline {b:.3f}")
    print(f"MEAN mine {sum(mine)/len(mine):.3f}  baseline {sum(base)/len(base):.3f}  n={len(mine)}")
    return sum(mine) / len(mine)

if __name__ == "__main__":
    cuts = [int(x) for x in sys.argv[1:]] or [13, 17, 21, 25, 29]
    run(cuts)
