"""RISER component (Step 6 candidate): a virality tilt on the empirical model.

Data-internal proxies only (the entry runs offline on provided data):
(a) visits/day since creation -- DROPPED: details (created, visits) is null in
    every practice read, so the proxy can neither be built nor validated here.
(b) recent acceleration: slope of log-playing over the last ~15h minus its
    slope over the prior ~45h (same for rank, sign-flipped), z-scored across
    the pool -- the in-data fingerprint of a game starting to blow up.
(c) age-conditioned entry rates: from all history pairs, P(a game outside the
    top places enters them) by age bucket, where age is time since the game's
    first appearance in the history (details.created unavailable). Young
    buckets earn a ratio vs the pooled entry rate, shrunk toward 1.

The riser reweights the empirical forecast's rows per place by
m_g = age_ratio(g) * accel_mult(g), then renormalises per place to the
empirical budget. Place shapes stay empirical; only WHO holds the mass tilts
toward young accelerators.
"""

import math
import sys
sys.path.insert(0, "/home/hatch/workspace/makefun/roblox-repo2")
from main import empirical_forecast, parse_time, E_NAMED_BUDGET, P_MIN, GAME_MASS_CAP, MAX_ROWS_PER_PLACE

AGE_EDGES = [24.0, 72.0, 168.0]   # hours since first seen
RECENT_H = 15.0
LONG_H = 60.0


def _slope(pts, t_lo, t_hi, idx):
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

    # (c) age-conditioned entry rates from all pairs within the window
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

    # (b) acceleration at the close, z-scored across games with enough data
    accel = {}
    for g, pts in series.items():
        lp = [(t, rk, math.log(max(p, 1)) if p is not None else None) for t, rk, p in pts]
        sp_r = _slope(lp, t_last - RECENT_H, t_last, 2)
        sp_l = _slope(lp, t_last - LONG_H, t_last - RECENT_H, 2)
        sr_r = _slope(pts, t_last - RECENT_H, t_last, 1)
        sr_l = _slope(pts, t_last - LONG_H, t_last - RECENT_H, 1)
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
    return mult, {"ratios": ratios, "n_b": n_b, "e_b": e_b, "rate_all": rate_all}


def riser_forecast(reads, places, win_hours):
    emp = empirical_forecast(reads, places, win_hours)
    mult, _dbg = virality_multipliers(reads, places, win_hours)
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
