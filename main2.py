"""Empirical transition forecaster for roblox-up-and-coming-forecast-01.

Standalone alternative to the kernel model in main.py (Step 4 candidate).

Idea: instead of a parametric kernel around a predicted centre, estimate each
close game's place distribution directly from the history's realised
transitions. All ordered read pairs (i, j) with lag in (0, window] are pooled
into lag buckets (fractions of the window). A game's outcome at j (place
1..places, or off-top-places) is counted by its rank BAND at i and its recent
trajectory CLASS at i (climbing / flat / falling). Distributions are shrunk
hierarchically: cell -> band x bucket -> band (global) -> global marginal,
so thin cells fall back to broader evidence. At the close, each game's
distribution is the bucket-width-weighted average over lag buckets (a uniform
horizon mix over the scoring window), then places are normalised per place
to NAMED_BUDGET, leaving an unnamed remainder for post-cut births (Step-2
measurement: 8/100 practice top-10 slots were games absent at the cut).

Everything is derived from the input at runtime. Stdlib only, deterministic.
"""

import json
import math
import os
from datetime import datetime

NAMED_BUDGET = 0.80    # per-place named sum before thresholding (backtest-tuned)
P_MIN = 0.0012
GAME_MASS_CAP = 1.10
MAX_ROWS_PER_PLACE = 100

# rank bands at the conditioning read
BANDS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 18),
         (19, 28), (29, 45), (46, 70), (71, 10 ** 9)]
# lag bucket edges as fractions of the window (uniform horizon mix weights)
BUCKET_FRACS = [0.0, 1 / 12, 1 / 6, 1 / 3, 2 / 3, 1.0]
SHRINK_CELL = 25.0     # pseudo-count pulling a cell toward band x bucket
SHRINK_BAND = 40.0     # pseudo-count pulling band x bucket toward band-global
SHRINK_GLOBAL = 30.0   # pseudo-count pulling band-global toward global marginal


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def band_of(rank):
    for bi, (lo, hi) in enumerate(BANDS):
        if lo <= rank <= hi:
            return bi
    return len(BANDS) - 1


def load_inputs(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda d: d["at"])
    return task, reads


def window_hours(task):
    w = task.get("window") or {}
    if w.get("from") and w.get("until"):
        return (parse_time(w["until"]) - parse_time(w["from"])).total_seconds() / 3600.0
    if w.get("slots") and w.get("slot_minutes"):
        return w["slots"] * w["slot_minutes"] / 60.0
    return 72.0


def empirical_forecast(reads, places, win_hours, budget=NAMED_BUDGET):
    times = [parse_time(r["at"]) for r in reads]
    hours = [(t - times[0]).total_seconds() / 3600.0 for t in times]
    rank_maps = []
    for r in reads:
        rank_maps.append({row["universe_id"]: row["rank"]
                          for row in r.get("up_and_coming") or []})
    n_reads = len(reads)
    n_bands = len(BANDS)
    n_buckets = len(BUCKET_FRACS) - 1
    OFF = places  # outcome index for "not in top places"

    # trajectory class at each read for each game present there:
    # rank ~8-24h earlier minus rank now; >=3 climbing, <=-3 falling, else flat
    classes = []
    for i in range(n_reads):
        ref = None
        for j in range(i - 1, -1, -1):
            if hours[i] - hours[j] >= 8.0:
                ref = j
                break
            if hours[i] - hours[j] >= 4.0:
                ref = j  # keep the earliest read at least 4h back as fallback
        cmap = {}
        for g, rk in rank_maps[i].items():
            cls = 1  # flat
            if ref is not None and g in rank_maps[ref]:
                delta = rank_maps[ref][g] - rk
                if delta >= 3:
                    cls = 0  # climbing
                elif delta <= -3:
                    cls = 2  # falling
            cmap[g] = cls
        classes.append(cmap)

    # counts[bucket][band][cls] -> [places outcomes..., OFF]
    def blank():
        return [0.0] * (places + 1)
    counts = [[[blank() for _ in range(3)] for _ in range(n_bands)]
              for _ in range(n_buckets)]
    band_bucket = [[blank() for _ in range(n_bands)] for _ in range(n_buckets)]
    band_global = [blank() for _ in range(n_bands)]
    global_counts = blank()
    bucket_has_pairs = [False] * n_buckets

    edges = [f * win_hours for f in BUCKET_FRACS]
    for i in range(n_reads):
        for j in range(i + 1, n_reads):
            lag = hours[j] - hours[i]
            if lag <= 0 or lag > win_hours:
                continue
            b = n_buckets - 1
            for bi in range(n_buckets):
                if edges[bi] < lag <= edges[bi + 1] or (bi == 0 and lag <= edges[1]):
                    b = bi
                    break
            bucket_has_pairs[b] = True
            rj = rank_maps[j]
            for g, rk in rank_maps[i].items():
                out = rj.get(g)
                o = (out - 1) if (out is not None and out <= places) else OFF
                bd = band_of(rk)
                cl = classes[i][g]
                counts[b][bd][cl][o] += 1.0
                band_bucket[b][bd][o] += 1.0
                band_global[bd][o] += 1.0
                global_counts[o] += 1.0

    def normalised(counts_vec, prior_vec, shrink):
        n = sum(counts_vec)
        if prior_vec is None:
            tot = n + shrink
            return [(c + shrink / len(counts_vec)) / tot for c in counts_vec]
        return [(c + shrink * p) / (n + shrink) for c, p in zip(counts_vec, prior_vec)]

    gtot = sum(global_counts)
    global_dist = [(c + 1.0) / (gtot + places + 1) for c in global_counts]
    band_global_dist = [normalised(band_global[bd], global_dist, SHRINK_GLOBAL)
                        for bd in range(n_bands)]
    band_bucket_dist = [[normalised(band_bucket[b][bd], band_global_dist[bd], SHRINK_BAND)
                         for bd in range(n_bands)] for b in range(n_buckets)]
    cell_dist = [[[normalised(counts[b][bd][cl], band_bucket_dist[b][bd], SHRINK_CELL)
                   for cl in range(3)] for bd in range(n_bands)]
                 for b in range(n_buckets)]

    # horizon mix: bucket width weights, restricted to buckets that saw pairs
    widths = [BUCKET_FRACS[b + 1] - BUCKET_FRACS[b] for b in range(n_buckets)]
    wsum = sum(w for b, w in enumerate(widths) if bucket_has_pairs[b]) or 1.0
    mix = [(w / wsum if bucket_has_pairs[b] else 0.0) for b, w in enumerate(widths)]

    # per-game horizon-averaged distributions at the close
    close_map = rank_maps[-1]
    last_seen = {}
    for i in range(n_reads):
        for g in rank_maps[i]:
            last_seen[g] = i
    pool = set(close_map)
    for g, i in last_seen.items():
        if n_reads - 1 - i <= 2 and rank_maps[i][g] <= 30:
            pool.add(g)  # recent absentees can come back

    game_dist = {}
    for g in sorted(pool):
        if g in close_map:
            rk, cl, discount = close_map[g], classes[-1][g], 1.0
        else:
            i = last_seen[g]
            rk, cl = rank_maps[i][g], classes[i][g]
            discount = 0.5 ** (n_reads - 1 - i)
        bd = band_of(rk)
        dist = [0.0] * (places + 1)
        for b in range(n_buckets):
            if mix[b] == 0.0:
                continue
            cd = cell_dist[b][bd][cl]
            for o in range(places + 1):
                dist[o] += mix[b] * cd[o]
        game_dist[g] = [d * discount for d in dist]

    # per-place normalisation to the named budget
    probabilities = []
    totals = {g: 0.0 for g in pool}
    for k in range(1, places + 1):
        col = {g: game_dist[g][k - 1] for g in pool}
        tot = sum(col.values())
        if tot <= 0:
            continue
        rows = []
        for g in sorted(pool):
            p = budget * col[g] / tot
            if p >= P_MIN:
                rows.append((g, p))
        rows.sort(key=lambda gp: (-gp[1], gp[0]))
        rows = rows[:MAX_ROWS_PER_PLACE]
        for g, p in rows:
            probabilities.append({"universe_id": g, "place": k, "p": round(p, 6)})
            totals[g] += p

    over = {g: t for g, t in totals.items() if t > GAME_MASS_CAP}
    if over:
        fixed = []
        for row in probabilities:
            if row["universe_id"] in over:
                row = dict(row, p=round(row["p"] * GAME_MASS_CAP / over[row["universe_id"]], 6))
            fixed.append(row)
        probabilities = [r for r in fixed if r["p"] > 0]

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    exp_place = {}
    for g in pool:
        s = sum(game_dist[g][:places])
        exp_place[g] = (sum((k + 1) * game_dist[g][k] for k in range(places)) / s
                        if s > 0 else 99.0)
    by_total = sorted(pool, key=lambda g: (-totals[g], exp_place[g], g))
    top10 = sorted(by_total[:places], key=lambda g: (exp_place[g], g))
    return {"probabilities": probabilities, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_inputs(input_dir)
    places = int(task.get("places", 10))
    out_name = task.get("output", "forecast.json")
    result = empirical_forecast(reads, places, window_hours(task))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, out_name), "w") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
