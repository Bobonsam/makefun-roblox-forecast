"""Roblox Up-and-Coming top-10 forecaster (roblox-up-and-coming-forecast-01).

Final model (Step 6): a three-way blend, weights chosen on an EXPANDED set of
22 in-history pseudo-window backtests (new blend -37.543 vs the Step-4/5
two-way blend's -37.702 on the same windows; practice score report-only).

1. KERNEL (weight 0.5): each game's centre blends its last rank with its
   recency-weighted trajectory, a damped rank-velocity extrapolation and a
   small capped pull toward its playing-count rank; its place distribution
   is a horizon-averaged two-component Laplace mixture (narrow early, wide
   late). Playing GROWTH alone is not used (measured corr ~0.05: noise).
2. EMPIRICAL (weight 0.2): realised transitions pooled from all read pairs,
   by lag bucket (fractions of the window), conditioned on rank band x
   trajectory class, with hierarchical shrinkage cell -> band x bucket ->
   band -> global. A game's close distribution is the bucket-width-weighted
   (uniform horizon) average of its cell distributions.
3. RISER (weight 0.3): the empirical forecast reweighted per place by a
   virality multiplier m_g = age entry-rate ratio x acceleration multiplier.
   Age (time since first appearance in the history; details.created is not
   carried by the archive reads) conditions top-places entry rates: games
   first seen <72h ago enter at ~1.25x the pooled rate, older games at
   ~0.5x. Acceleration is the slope of log-playing (and of rank) over the
   last ~15h minus the prior ~45h, z-scored across the pool -- the in-data
   fingerprint of a game starting to blow up. (A visits/day-since-creation
   proxy was dropped: details is null throughout the practice history, so
   it could not be built or validated.)
BLEND: per (game, place), p = 0.5 * kernel + 0.2 * empirical + 0.3 * riser.
The grid's argmax (0.6 kernel / 0.0 empirical / 0.4 riser, -37.538) sits on
the same plateau; the shipped weights are the plateau centre and keep all
three parents.

Both parents normalise each place to a named budget below 1: the unnamed
remainder is shared by every unnamed game on each read's chart and is the
only cover for post-cut births (Step-2 measurement: 8/100 practice top-10
slots were games absent at the cut, 36/100 were outside the close top 10).

Everything (places, output name, window length) is read from task.json and
the history at runtime. Standard library only, no network, deterministic.
"""

import json
import math
import os
from datetime import datetime

# --- kernel knobs (tuned on in-history pseudo-window backtests only) ---
K_NAMED_BUDGET = 0.74
K_KERNEL_SIGMA = 2.6
K_VELOCITY_DAMP = 0.60
K_PLAYING_PULL = 0.06
TRAJ_HOURS = 48.0
VEL_HOURS = 30.0
OCC_HOURS = 72.0
POOL_HOURS = 96.0

# --- empirical knobs ---
E_NAMED_BUDGET = 0.80
BANDS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 18),
         (19, 28), (29, 45), (46, 70), (71, 10 ** 9)]
BUCKET_FRACS = [0.0, 1 / 12, 1 / 6, 1 / 3, 2 / 3, 1.0]
SHRINK_CELL = 25.0
SHRINK_BAND = 40.0
SHRINK_GLOBAL = 30.0

# --- riser ---
AGE_EDGES = [24.0, 72.0, 168.0]   # hours since first seen in the history
RECENT_H = 15.0
LONG_H = 60.0

# --- shared / blend ---
W_KERNEL = 0.5
W_EMPIRICAL = 0.2
W_RISER = 0.3
P_MIN = 0.0012
GAME_MASS_CAP = 1.10
MAX_ROWS_PER_PLACE = 100


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


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


# ============================ kernel model ============================

def game_features(reads):
    t0 = parse_time(reads[0]["at"])
    times = [(parse_time(r["at"]) - t0).total_seconds() / 3600.0 for r in reads]
    series = {}
    details_by_id = {}
    for idx, r in enumerate(reads):
        for row in r.get("up_and_coming") or []:
            g = row["universe_id"]
            series.setdefault(g, []).append((times[idx], row["rank"], row.get("playing")))
        det = r.get("details")
        if isinstance(det, dict):
            for k, v in det.items():
                if isinstance(v, dict):
                    details_by_id[int(k)] = v
    return times, series, details_by_id


def weighted_mean_rank(points, t_last, half_life=36.0):
    num = den = 0.0
    for t, rank, _ in points:
        w = 0.5 ** ((t_last - t) / half_life)
        num += w * rank
        den += w
    return num / den if den else None


def rank_velocity(points, t_last):
    pts = [(t, rank) for t, rank, _ in points if t_last - t <= VEL_HOURS]
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < 5.0:
        return 0.0
    n = len(pts)
    mt = sum(p[0] for p in pts) / n
    mr = sum(p[1] for p in pts) / n
    cov = sum((p[0] - mt) * (p[1] - mr) for p in pts)
    var = sum((p[0] - mt) ** 2 for p in pts)
    return cov / var if var > 0 else 0.0


def kernel_forecast(reads, places, win_hours):
    times, series, details_by_id = game_features(reads)
    t_last = times[-1]
    mean_horizon = win_hours / 2.0

    last_read = reads[-1]
    close_rank = {}
    close_playing = {}
    for row in last_read.get("up_and_coming") or []:
        close_rank[row["universe_id"]] = row["rank"]
        close_playing[row["universe_id"]] = row.get("playing")
    playing_rank = {}
    ranked_by_playing = sorted(
        (g for g, p in close_playing.items() if p is not None),
        key=lambda g: (-close_playing[g], g),
    )
    for i, g in enumerate(ranked_by_playing):
        playing_rank[g] = i + 1

    pool = []
    for g, pts in series.items():
        recent = [p for p in pts if t_last - p[0] <= POOL_HOURS]
        if not recent:
            continue
        best_recent = min(p[1] for p in recent)
        if best_recent <= 45 or playing_rank.get(g, 999) <= 45 or g in close_rank and close_rank[g] <= 45:
            pool.append(g)
    pool.sort()

    centres, scales, amps = {}, {}, {}
    for g in pool:
        pts = series[g]
        last_t, last_rank, _ = pts[-1]
        reads_since = sum(1 for t in times if t > last_t)
        traj = weighted_mean_rank([p for p in pts if t_last - p[0] <= TRAJ_HOURS], t_last)
        if traj is None:
            traj = float(last_rank)
        mu = 0.55 * last_rank + 0.45 * traj
        mu += min(12.0, 3.0 * reads_since)

        vel = rank_velocity(pts, t_last)
        shift = vel * mean_horizon * K_VELOCITY_DAMP
        shift = max(-9.0, min(10.0, shift))
        mu += shift

        if g in playing_rank:
            pulled = mu * (1.0 - K_PLAYING_PULL) + playing_rank[g] * K_PLAYING_PULL
            pulled = max(mu - 10.0, min(mu + 10.0, pulled))
            mu = pulled
        mu = max(1.0, min(60.0, mu))

        recent_ranks = [p[1] for p in pts if t_last - p[0] <= OCC_HOURS]
        if len(recent_ranks) >= 3:
            mean_r = sum(recent_ranks) / len(recent_ranks)
            std_r = math.sqrt(sum((x - mean_r) ** 2 for x in recent_ranks) / len(recent_ranks))
        else:
            std_r = 4.0
        scale = K_KERNEL_SIGMA * (1.0 + 0.5 * min(1.0, std_r / 6.0))

        occ_pts = [p for p in pts if t_last - p[0] <= OCC_HOURS]
        occ_reads = sum(1 for t in times if t_last - t <= OCC_HOURS)
        hits = sum(1 for p in occ_pts if p[1] <= 15)
        freq = hits / occ_reads if occ_reads else 0.0
        amp = 0.30 + 0.90 * freq
        if reads_since:
            amp *= 0.7 ** reads_since

        det = details_by_id.get(g)
        if det and det.get("created"):
            try:
                created = parse_time(det["created"])
                close_dt = parse_time(reads[-1]["at"])
                age_days = (close_dt - created).total_seconds() / 86400.0
                left_days = 28.0 - age_days
                win_days = win_hours / 24.0
                if left_days < win_days:
                    amp *= max(0.15, min(1.0, left_days / win_days)) if left_days > 0 else 0.15
            except Exception:
                pass

        centres[g], scales[g], amps[g] = mu, scale, amp

    probabilities = []
    totals = {g: 0.0 for g in pool}
    for k in range(1, places + 1):
        weights = {}
        for g in pool:
            mu, b, amp = centres[g], scales[g], amps[g]
            d = abs(k - mu)
            q = amp * (0.45 * math.exp(-d / (0.6 * b)) + 0.55 * math.exp(-d / (1.5 * b)))
            q += 0.004 * amp
            weights[g] = q
        tot = sum(weights.values())
        rows = []
        for g in pool:
            p = K_NAMED_BUDGET * weights[g] / tot
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
    by_total = sorted(pool, key=lambda g: (-totals[g], centres[g], g))
    top10 = sorted(by_total[:places], key=lambda g: (centres[g], g))
    return {"probabilities": probabilities, "top10": top10}


# ============================ empirical model ============================

def band_of(rank):
    for bi, (lo, hi) in enumerate(BANDS):
        if lo <= rank <= hi:
            return bi
    return len(BANDS) - 1


def empirical_forecast(reads, places, win_hours, budget=E_NAMED_BUDGET):
    times = [parse_time(r["at"]) for r in reads]
    hours = [(t - times[0]).total_seconds() / 3600.0 for t in times]
    rank_maps = []
    for r in reads:
        rank_maps.append({row["universe_id"]: row["rank"]
                          for row in r.get("up_and_coming") or []})
    n_reads = len(reads)
    n_bands = len(BANDS)
    n_buckets = len(BUCKET_FRACS) - 1
    OFF = places

    classes = []
    for i in range(n_reads):
        ref = None
        for j in range(i - 1, -1, -1):
            if hours[i] - hours[j] >= 8.0:
                ref = j
                break
            if hours[i] - hours[j] >= 4.0:
                ref = j
        cmap = {}
        for g, rk in rank_maps[i].items():
            cls = 1
            if ref is not None and g in rank_maps[ref]:
                delta = rank_maps[ref][g] - rk
                if delta >= 3:
                    cls = 0
                elif delta <= -3:
                    cls = 2
            cmap[g] = cls
        classes.append(cmap)

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

    widths = [BUCKET_FRACS[b + 1] - BUCKET_FRACS[b] for b in range(n_buckets)]
    wsum = sum(w for b, w in enumerate(widths) if bucket_has_pairs[b]) or 1.0
    mix = [(w / wsum if bucket_has_pairs[b] else 0.0) for b, w in enumerate(widths)]

    close_map = rank_maps[-1]
    last_seen = {}
    for i in range(n_reads):
        for g in rank_maps[i]:
            last_seen[g] = i
    pool = set(close_map)
    for g, i in last_seen.items():
        if n_reads - 1 - i <= 2 and rank_maps[i][g] <= 30:
            pool.add(g)

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


# ============================ riser model ============================

def _series_slope(pts, t_lo, t_hi, idx):
    sel = [(p[0], p[idx]) for p in pts if t_lo <= p[0] <= t_hi and p[idx] is not None]
    if len(sel) < 3 or sel[-1][0] - sel[0][0] < 4.0:
        return None
    n = len(sel)
    mt = sum(s[0] for s in sel) / n
    my = sum(s[1] for s in sel) / n
    var = sum((s[0] - mt) ** 2 for s in sel)
    if var <= 0:
        return None
    return sum((s[0] - mt) * (s[1] - my) for s in sel) / var


def _age_bucket(age_h):
    for bi, e in enumerate(AGE_EDGES):
        if age_h < e:
            return bi
    return len(AGE_EDGES)


def virality_multipliers(reads, places, win_hours):
    times = [parse_time(r["at"]) for r in reads]
    hours = [(t - times[0]).total_seconds() / 3600.0 for t in times]
    t_last = hours[-1]
    rank_maps = [{row["universe_id"]: row["rank"] for row in r.get("up_and_coming") or []}
                 for r in reads]
    series = {}
    first_seen = {}
    for idx, r in enumerate(reads):
        for row in r.get("up_and_coming") or []:
            g = row["universe_id"]
            series.setdefault(g, []).append((hours[idx], row["rank"], row.get("playing")))
            if g not in first_seen:
                first_seen[g] = hours[idx]

    n_b = [0] * (len(AGE_EDGES) + 1)
    e_b = [0] * (len(AGE_EDGES) + 1)
    for i in range(len(reads)):
        for j in range(i + 1, len(reads)):
            lag = hours[j] - hours[i]
            if lag <= 0 or lag > win_hours:
                continue
            for g, rk in rank_maps[i].items():
                if rk <= places:
                    continue
                b = _age_bucket(hours[i] - first_seen[g])
                n_b[b] += 1
                out = rank_maps[j].get(g)
                if out is not None and out <= places:
                    e_b[b] += 1
    n_all, e_all = sum(n_b), sum(e_b)
    rate_all = e_all / n_all if n_all else 0.0
    ratios = []
    for b in range(len(AGE_EDGES) + 1):
        if rate_all <= 0 or n_b[b] == 0:
            ratios.append(1.0)
        else:
            r = (e_b[b] + 10.0 * rate_all) / (n_b[b] + 10.0) / rate_all
            ratios.append(max(0.5, min(2.5, r)))

    accel = {}
    for g, pts in series.items():
        lp = [(t, rk, math.log(max(p, 1)) if p is not None else None) for t, rk, p in pts]
        sp_r = _series_slope(lp, t_last - RECENT_H, t_last, 2)
        sp_l = _series_slope(lp, t_last - LONG_H, t_last - RECENT_H, 2)
        sr_r = _series_slope(pts, t_last - RECENT_H, t_last, 1)
        sr_l = _series_slope(pts, t_last - LONG_H, t_last - RECENT_H, 1)
        vals = []
        if sp_r is not None and sp_l is not None:
            vals.append(sp_r - sp_l)
        if sr_r is not None and sr_l is not None:
            vals.append(-(sr_r - sr_l))
        if vals:
            accel[g] = sum(vals) / len(vals)
    mult_accel = {}
    if len(accel) >= 5:
        vs = list(accel.values())
        mean = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vs) / len(vs)) or 1.0
        for g, v in accel.items():
            z = max(-1.5, min(2.0, (v - mean) / sd))
            mult_accel[g] = max(0.4, min(3.0, math.exp(0.6 * z)))
    mult = {}
    for g in series:
        age = t_last - first_seen[g]
        m = ratios[_age_bucket(age)] * mult_accel.get(g, 1.0)
        mult[g] = max(0.25, min(4.0, m))
    return mult


def riser_forecast(reads, places, win_hours):
    emp = empirical_forecast(reads, places, win_hours)
    mult = virality_multipliers(reads, places, win_hours)
    by_place = {}
    for row in emp["probabilities"]:
        by_place.setdefault(row["place"], []).append(row)
    probabilities = []
    totals = {}
    for k in sorted(by_place):
        raw = [(row["universe_id"], row["p"] * mult.get(row["universe_id"], 1.0))
               for row in by_place[k]]
        tot = sum(v for _, v in raw)
        if tot <= 0:
            continue
        rows = []
        for g, v in raw:
            p = E_NAMED_BUDGET * v / tot
            if p >= P_MIN:
                rows.append((g, p))
        rows.sort(key=lambda gp: (-gp[1], gp[0]))
        rows = rows[:MAX_ROWS_PER_PLACE]
        for g, p in rows:
            probabilities.append({"universe_id": g, "place": k, "p": round(p, 6)})
            totals[g] = totals.get(g, 0.0) + p
    over = {g: t for g, t in totals.items() if t > GAME_MASS_CAP}
    if over:
        fixed = []
        for row in probabilities:
            if row["universe_id"] in over:
                row = dict(row, p=round(row["p"] * GAME_MASS_CAP / over[row["universe_id"]], 6))
            fixed.append(row)
        probabilities = [r for r in fixed if r["p"] > 0]
        totals = {}
        for r in probabilities:
            totals[r["universe_id"]] = totals.get(r["universe_id"], 0.0) + r["p"]
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    esum, wsum = {}, {}
    for r in probabilities:
        esum[r["universe_id"]] = esum.get(r["universe_id"], 0.0) + r["p"] * r["place"]
        wsum[r["universe_id"]] = wsum.get(r["universe_id"], 0.0) + r["p"]
    exp = {g: (esum[g] / wsum[g] if wsum[g] else 99.0) for g in totals}
    by_total = sorted(totals, key=lambda g: (-totals[g], exp[g], g))
    top10 = sorted(by_total[:places], key=lambda g: (exp[g], g))
    return {"probabilities": probabilities, "top10": top10}


# ============================ blend ============================

def blend_forecasts(kernel_fc, emp_fc, riser_fc, places):
    pk = {(r["universe_id"], r["place"]): r["p"] for r in kernel_fc["probabilities"]}
    pe = {(r["universe_id"], r["place"]): r["p"] for r in emp_fc["probabilities"]}
    pr = {(r["universe_id"], r["place"]): r["p"] for r in riser_fc["probabilities"]}
    rows, totals, wsum, esum = [], {}, {}, {}
    for (g, k) in sorted(set(pk) | set(pe) | set(pr)):
        p = (W_KERNEL * pk.get((g, k), 0.0) + W_EMPIRICAL * pe.get((g, k), 0.0)
             + W_RISER * pr.get((g, k), 0.0))
        if p >= P_MIN:
            rows.append({"universe_id": g, "place": k, "p": round(p, 6)})
            totals[g] = totals.get(g, 0.0) + p
            wsum[g] = wsum.get(g, 0.0) + p
            esum[g] = esum.get(g, 0.0) + p * k
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    exp = {g: (esum[g] / wsum[g] if wsum[g] else 99.0) for g in totals}
    by_total = sorted(totals, key=lambda g: (-totals[g], exp[g], g))
    top10 = sorted(by_total[:places], key=lambda g: (exp[g], g))
    return {"probabilities": rows, "top10": top10}


def forecast(reads, places, win_hours):
    return blend_forecasts(kernel_forecast(reads, places, win_hours),
                           empirical_forecast(reads, places, win_hours),
                           riser_forecast(reads, places, win_hours), places)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_inputs(input_dir)
    places = int(task.get("places", 10))
    out_name = task.get("output", "forecast.json")
    result = forecast(reads, places, window_hours(task))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, out_name), "w") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
