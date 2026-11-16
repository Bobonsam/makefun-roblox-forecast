"""My own local scorer for roblox-up-and-coming-forecast-01, written from RULES.txt.

Scoring (RULES.txt, 'Scoring' + 'The answer'):
- A forecast is ONE static object {probabilities, top10} scored against EVERY read
  of the window (final: 72 hourly reads x 10 places = 720 place-scores;
  practice: 10 archive reads x 10 places = 100 place-scores).
- At a read, for place k: p(game) = its row p if named at k, else the unnamed
  share = (1 - sum of place-k rows) / (# games on THAT read's chart not named at k).
- Place contribution = ln(max(p, 0.0001)). Read score = sum over k=1..10.
- Log score = mean of read scores (equivalently total / #reads). Higher is better.
- Also: top-10 hits (mean |top10 ∩ read top10|) and agreement tie-break
  (per read, sum over shared games of 10 - |place distance|, then mean).

Usage:
  python3 local_scorer.py <forecast.json> <reads.ndjson>
  reads.ndjson lines: {"at": ..., "up_and_coming": [{"rank":..,"universe_id":..}, ...]}
  (kit history lines also work: extra fields are ignored.)
"""
import json, math, sys

FLOOR = 0.0001
PLACES = 10

def load_reads(path):
    reads = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            chart = sorted(d["up_and_coming"], key=lambda r: r["rank"])
            ids = [r["universe_id"] for r in chart]
            reads.append({"at": d.get("at"), "ids": ids,
                          "place_of": {g: i + 1 for i, g in enumerate(ids)}})
    return reads

def score(forecast, reads):
    # index named probabilities: place -> {game: p}
    named = {}
    for row in forecast["probabilities"]:
        named.setdefault(row["place"], {})[row["universe_id"]] = row["p"]
    top10 = forecast["top10"]
    my_place = {g: i + 1 for i, g in enumerate(top10)}

    total = 0.0
    hits_total = 0
    agree_total = 0
    per_read = []
    for rd in reads:
        chart_set = set(rd["ids"])
        read_score = 0.0
        for k in range(1, PLACES + 1):
            actual = rd["ids"][k - 1]
            rows = named.get(k, {})
            s = sum(rows.values())
            named_on_chart = sum(1 for g in rows if g in chart_set)
            n_unnamed = len(chart_set) - named_on_chart
            if actual in rows:
                p = rows[actual]
            else:
                p = (1.0 - s) / n_unnamed if n_unnamed > 0 else 0.0
            read_score += math.log(max(p, FLOOR))
        total += read_score
        per_read.append((rd["at"], read_score))
        read_t10 = rd["ids"][:PLACES]
        hits_total += len(set(top10) & set(read_t10))
        agree_total += sum(10 - abs(my_place[g] - rd["place_of"][g])
                           for g in set(top10) & set(read_t10))
    n = len(reads)
    return {
        "log_score": total / n,
        "typical_p": math.exp((total / n) / PLACES),
        "top10_hits": hits_total / n,
        "agreement": agree_total / n,
        "reads": n,
        "place_scores": n * PLACES,
        "per_read": per_read,
    }

if __name__ == "__main__":
    fc = json.load(open(sys.argv[1]))
    rds = load_reads(sys.argv[2])
    r = score(fc, rds)
    print(f"log_score  {r['log_score']:.6f}")
    print(f"typical_p  {r['typical_p']:.6f}")
    print(f"top10_hits {r['top10_hits']:.2f} / read")
    print(f"agreement  {r['agreement']:.2f} / read")
    print(f"reads {r['reads']}  place-scores {r['place_scores']}")
    for at, s in r["per_read"]:
        print(f"  {at}  {s:.3f}")
