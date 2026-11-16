"""Step 6 evaluation: EXPANDED pseudo-window backtests + 3-way blend grid.

Windows: every cut i (>=10 history reads) with >=3 forward reads within 72h.
Parents per window come only from reads <= cut (no leakage). The shipped
blend is main.forecast itself. Practice scores are report-only.
"""
import json, sys, itertools
sys.path.insert(0, "/home/hatch/workspace/makefun/roblox-repo2")
import main as M
import riser as R
from local_scorer import score, load_reads

KIT = "/home/hatch/workspace/makefun/roblox-up-and-coming-forecast-01-kit/input/history.ndjson"
raw = [json.loads(l) for l in open(KIT) if l.strip()]
raw.sort(key=lambda d: d["at"])
WIN_READS = load_reads("/home/hatch/workspace/makefun/roblox-window-reads.ndjson")

def to_scorer_reads(reads):
    out = []
    for d in reads:
        chart = sorted(d["up_and_coming"], key=lambda r: r["rank"])
        ids = [r["universe_id"] for r in chart]
        out.append({"at": d.get("at"), "ids": ids,
                    "place_of": {g: i + 1 for i, g in enumerate(ids)}})
    return out

WINDOWS = []
for i in range(9, len(raw) - 1):
    trunc = raw[: i + 1]
    t_cut = M.parse_time(trunc[-1]["at"])
    fwd = [d for d in raw[i + 1:] if (M.parse_time(d["at"]) - t_cut).total_seconds() <= 72 * 3600]
    if len(fwd) >= 3:
        WINDOWS.append((i, trunc, to_scorer_reads(fwd)))
print(f"expanded windows: {len(WINDOWS)}  cuts {[w[0] for w in WINDOWS]}")
print(f"forward reads per window: {[len(w[2]) for w in WINDOWS]}")

def blend3(fk, fe, fr, wk, we, wr):
    pk = {(r["universe_id"], r["place"]): r["p"] for r in fk["probabilities"]}
    pe = {(r["universe_id"], r["place"]): r["p"] for r in fe["probabilities"]}
    pr = {(r["universe_id"], r["place"]): r["p"] for r in fr["probabilities"]}
    rows, totals, wsum, esum = [], {}, {}, {}
    for (g, k) in sorted(set(pk) | set(pe) | set(pr)):
        p = wk * pk.get((g, k), 0.0) + we * pe.get((g, k), 0.0) + wr * pr.get((g, k), 0.0)
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

parents = []   # per window: (kernel, emp, riser, shipped, scorer_reads)
sums = {"kernel": 0.0, "empirical": 0.0, "riser": 0.0, "shipped": 0.0}
for i, trunc, sr in WINDOWS:
    kf = M.kernel_forecast(trunc, 10, 72.0)
    ef = M.empirical_forecast(trunc, 10, 72.0)
    rf = R.riser_forecast(trunc, 10, 72.0)
    sf = M.forecast(trunc, 10, 72.0)
    parents.append((kf, ef, rf, sr))
    for name, fc in (("kernel", kf), ("empirical", ef), ("riser", rf), ("shipped", sf)):
        sums[name] += score(fc, sr)["log_score"]
n = len(WINDOWS)
print("\n== expanded-backtest means (the number to beat = shipped) ==")
for name in ("shipped", "kernel", "empirical", "riser"):
    print(f"  {name:10s} {sums[name]/n:.3f}")
shipped_mean = sums["shipped"] / n

print("\n== 3-way grid (step 0.1) on expanded set, top 12 ==")
grid = []
for wk_i in range(0, 11):
    for we_i in range(0, 11 - wk_i):
        wr_i = 10 - wk_i - we_i
        wk, we, wr = wk_i / 10, we_i / 10, wr_i / 10
        tot = 0.0
        for kf, ef, rf, sr in parents:
            tot += score(blend3(kf, ef, rf, wk, we, wr), sr)["log_score"]
        grid.append((tot / n, wk, we, wr))
grid.sort(reverse=True)
for s, wk, we, wr in grid[:12]:
    print(f"  {s:.3f}  wk {wk} we {we} wr {wr}")
best = grid[0]
print(f"\nbest {best[0]:.3f} vs shipped {shipped_mean:.3f}  -> margin {best[0]-shipped_mean:+.3f} (ship threshold +0.15)")

print("\n== practice (report-only) ==")
kf = M.kernel_forecast(raw, 10, 72.0)
ef = M.empirical_forecast(raw, 10, 72.0)
rf = R.riser_forecast(raw, 10, 72.0)
sf = M.forecast(raw, 10, 72.0)
print(f"  shipped blend practice {score(sf, WIN_READS)['log_score']:.3f}")
print(f"  riser standalone practice {score(rf, WIN_READS)['log_score']:.3f}")
_, bwk, bwe, bwr = best
print(f"  best-grid blend ({bwk}/{bwe}/{bwr}) practice {score(blend3(kf, ef, rf, bwk, bwe, bwr), WIN_READS)['log_score']:.3f}")

print("\n== sanity: shipped blend's mass ranking of the practice window's risers ==")
totals = {}
for row in sf["probabilities"]:
    totals[row["universe_id"]] = totals.get(row["universe_id"], 0.0) + row["p"]
ranking = sorted(totals, key=lambda g: -totals[g])
rank_of = {g: i + 1 for i, g in enumerate(ranking)}
from collections import Counter
freq = Counter()
for rd in WIN_READS:
    for g in rd["ids"][:10]:
        freq[g] += 1
names = {}
for h in raw:
    for r in h["up_and_coming"]:
        names[r["universe_id"]] = r["name"]
for g, c in freq.most_common():
    print(f"  actual {c}/10 reads  blend mass-rank {rank_of.get(g, 'unnamed')}  in blend top10: {g in sf['top10']}  {names.get(g, '(post-cut birth)')}")
mult, dbg = R.virality_multipliers(raw, 10, 72.0)
print("\nage entry-rate ratios (practice history):", [round(x, 2) for x in dbg["ratios"]],
      "n per bucket:", dbg["n_b"], "entries:", dbg["e_b"], "pooled rate:", round(dbg["rate_all"], 4))
top_mult = sorted(mult.items(), key=lambda kv: -kv[1])[:8]
print("highest virality multipliers at practice close:", [(g, names.get(g, '?')[:26], round(m, 2)) for g, m in top_mult])
