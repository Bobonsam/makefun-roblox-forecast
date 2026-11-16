"""My own exploratory analysis for roblox-repo2 (helper, not the entry).
Reads only the kit practice history + the rebuilt practice window reads.
"""
import json, math, statistics
from datetime import datetime
from collections import Counter, defaultdict

KIT_HIST = "/home/hatch/workspace/makefun/roblox-up-and-coming-forecast-01-kit/input/history.ndjson"
WIN = "/home/hatch/workspace/makefun/roblox-window-reads.ndjson"

def parse_at(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def load(path, full):
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            chart = sorted(d["up_and_coming"], key=lambda r: r["rank"])
            out.append({
                "at": parse_at(d["at"]), "at_s": d["at"],
                "ids": [r["universe_id"] for r in chart],
                "rank": {r["universe_id"]: r["rank"] for r in chart},
                "playing": {r["universe_id"]: r.get("playing") for r in chart} if full else {},
                "names": {r["universe_id"]: r.get("name") for r in chart} if full else {},
            })
    return out

hist = load(KIT_HIST, True)
win = load(WIN, False)
close = hist[-1]
print(f"history reads={len(hist)} {hist[0]['at_s']} .. {hist[-1]['at_s']}")
print(f"window reads={len(win)} {win[0]['at_s']} .. {win[-1]['at_s']}")
gaps = [(b["at"]-a["at"]).total_seconds()/3600 for a, b in zip(hist, hist[1:])]
print(f"history gaps h: min {min(gaps):.2f} med {statistics.median(gaps):.2f} max {max(gaps):.2f} mean {statistics.mean(gaps):.2f}")

# ---- consecutive-pair churn inside history ----
print("\n== consecutive pairs (history, 33 pairs) ==")
top10_overlap, exact_stay, changed_places, num1_changes = [], [], [], 0
full_overlap_frac, per_hour_top10_loss = [], []
for a, b in zip(hist, hist[1:]):
    dt = (b["at"]-a["at"]).total_seconds()/3600
    t10a, t10b = a["ids"][:10], b["ids"][:10]
    ov = len(set(t10a) & set(t10b))
    top10_overlap.append(ov)
    exact = sum(1 for k in range(10) if t10a[k] == t10b[k])
    exact_stay.append(exact)
    changed_places.append(10 - exact)
    if t10a[0] != t10b[0]:
        num1_changes += 1
    full_overlap_frac.append(len(set(a["ids"]) & set(b["ids"])) / len(set(a["ids"]) | set(b["ids"])))
    per_hour_top10_loss.append((10 - ov) / dt)
print(f"top10 overlap: mean {statistics.mean(top10_overlap):.2f} med {statistics.median(top10_overlap)} min {min(top10_overlap)} max {max(top10_overlap)}")
print(f"exact-place stays in top10: mean {statistics.mean(exact_stay):.2f}/10  (changed places mean {statistics.mean(changed_places):.2f})")
print(f"#1 changed in {num1_changes}/33 pairs")
print(f"full-chart Jaccard overlap: mean {statistics.mean(full_overlap_frac):.3f}")
print(f"top10 members lost per hour: mean {statistics.mean(per_hour_top10_loss):.3f} med {statistics.median(per_hour_top10_loss):.3f}")
# rank displacement for games staying in top10
disp = []
for a, b in zip(hist, hist[1:]):
    for g in set(a["ids"][:10]) & set(b["ids"][:10]):
        disp.append(abs(a["rank"][g] - b["rank"][g]))
print(f"|rank move| for top10 survivors: mean {statistics.mean(disp):.2f} med {statistics.median(disp)} p90 {sorted(disp)[int(0.9*len(disp))]} max {max(disp)}; P(move=0)={disp.count(0)/len(disp):.2f} P(<=1)={sum(1 for x in disp if x<=1)/len(disp):.2f} P(<=2)={sum(1 for x in disp if x<=2)/len(disp):.2f} P(<=3)={sum(1 for x in disp if x<=3)/len(disp):.2f}")

# ---- where do new top10 members come from? ----
print("\n== origin of new top10 members (history pairs) ==")
origins = Counter()
for a, b in zip(hist, hist[1:]):
    for g in set(b["ids"][:10]) - set(a["ids"][:10]):
        r = a["rank"].get(g)
        if r is None: origins["absent_from_chart"] += 1
        elif r <= 15: origins["rank_11_15"] += 1
        elif r <= 20: origins["rank_16_20"] += 1
        elif r <= 30: origins["rank_21_30"] += 1
        elif r <= 50: origins["rank_31_50"] += 1
        else: origins["rank_51+"] += 1
tot = sum(origins.values())
for k, v in origins.most_common():
    print(f"  {k}: {v} ({v/tot:.1%})  total new={tot}")

# ---- close -> window decay ----
print("\n== close (7 Nov 20:53) vs each window read ==")
close_ids = set(close["ids"]); close_t10 = close["ids"][:10]
hist_seen = set()
for h in hist: hist_seen |= set(h["ids"])
hist_t10_seen = set()
for h in hist: hist_t10_seen |= set(h["ids"][:10])
print("close top10:", [(i+1, close['names'][g], g) for i, g in enumerate(close_t10)])
for w in win:
    hrs = (w["at"]-close["at"]).total_seconds()/3600
    t10 = w["ids"][:10]
    ov = len(set(t10) & set(close_t10))
    exact = sum(1 for k in range(10) if t10[k] == close_t10[k])
    not_in_close_chart = [g for g in t10 if g not in close_ids]
    not_in_close_t10 = [g for g in t10 if g not in set(close_t10)]
    never_in_hist = [g for g in t10 if g not in hist_seen]
    print(f"  +{hrs:5.1f}h {w['at_s']}: top10 overlap w/ close top10 = {ov}/10, exact places = {exact}/10, "
          f"new-to-top10 = {len(not_in_close_t10)}, absent-from-close-chart = {len(not_in_close_chart)}, never-in-history = {len(never_in_hist)}")
# window top10 census
cnt = Counter()
for w in win:
    for g in w["ids"][:10]: cnt[g] += 1
print(f"distinct games in window top10 across 10 reads: {len(cnt)}")
print("window top10 frequency:", [(g, c, close['names'].get(g, '?'), close['rank'].get(g)) for g, c in cnt.most_common()])
win_only = [g for g in cnt if g not in close_ids]
print(f"window-top10 games absent from close chart entirely: {win_only} (count {len(win_only)})")
win_new_t10 = [g for g in cnt if g not in set(close_t10)]
print(f"window-top10 games not in close top10: {len(win_new_t10)} distinct; their close ranks: {[(g, close['rank'].get(g)) for g in win_new_t10]}")

# ---- #1 in window ----
print("\n#1 per window read:", [(w['at_s'], w['ids'][0], close['names'].get(w['ids'][0], '?'), close['rank'].get(w['ids'][0])) for w in win])
print("close #1 (Survive Your Ideas 10769671316) ranks in window:", [w['rank'].get(10769671316) for w in win])

# ---- playing / velocity signal inside history ----
print("\n== playing signal ==")
# correlation: playing growth over last interval vs next rank change, pooled over history
import math
xs, ys = [], []
for i in range(1, len(hist)-1):
    prev, cur, nxt = hist[i-1], hist[i], hist[i+1]
    dt1 = (cur["at"]-prev["at"]).total_seconds()/3600
    for g in set(cur["ids"][:30]):
        if g in prev["playing"] and g in cur["playing"] and cur["playing"].get(g) and prev["playing"].get(g):
            if g in nxt["rank"] and g in cur["rank"]:
                growth = math.log(cur["playing"][g]/prev["playing"][g]) / dt1  # per hour
                xs.append(growth); ys.append(cur["rank"][g]-nxt["rank"][g])  # + = climbed
def corr(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    den = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return num/den if den else float('nan')
print(f"pooled top30: corr(log-playing growth/h, next rank gain) = {corr(xs, ys):.3f}  n={len(xs)}")
# level signal: playing rank vs chart rank at close
cl = sorted(close["ids"], key=lambda g: -(close["playing"].get(g) or 0))
print("close: top10 by playing:", [(g, close['names'][g], close['rank'][g], close['playing'][g]) for g in cl[:10]])
# last-interval velocity at close for close top10 + ranks 11-25
prev = hist[-2]
dt = (close["at"]-prev["at"]).total_seconds()/3600
print(f"last interval dt={dt:.2f}h; close ranks 1-25 velocity (playing ratio/h) and rank delta last interval:")
for g in close["ids"][:25]:
    pr, cr = prev["rank"].get(g), close["rank"][g]
    ratio = (close["playing"][g]/prev["playing"][g])**(1/dt) if prev["playing"].get(g) else None
    print(f"  rank {cr:2d} {close['names'][g][:34]:34s} playing {close['playing'][g]:7d} ratio/h {ratio:.3f} rankdelta {None if pr is None else pr-cr}")

# ---- tenure / first-seen ----
print("\n== tenure ==")
first_seen = {}
for i, h in enumerate(hist):
    for g in h["ids"]:
        if g not in first_seen: first_seen[g] = i
for g in close_t10:
    print(f"  close top10 {close['names'][g][:30]:30s} first seen read #{first_seen[g]} of 34, appearances {sum(1 for h in hist if g in h['rank'])}")
# survival: games in top10 at read i, still top10 at i+k
for k in (1, 2, 4, 8):
    surv = []
    for i in range(len(hist)-k):
        a, b = hist[i], hist[i+k]
        surv.append(len(set(a['ids'][:10]) & set(b['ids'][:10])))
    print(f"  top10 overlap after {k} reads (~{k*statistics.median(gaps):.0f}h): mean {statistics.mean(surv):.2f}/10")
