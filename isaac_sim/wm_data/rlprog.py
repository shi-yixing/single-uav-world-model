import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
ep = [r for r in rows if "episode/score" in r]
st = [r for r in rows if "epstats/log/goal/max" in r]
print("episodes logged:", len(ep), " last step:", rows[-1].get("step"))
for r in st[-6:]:
    print(r.get("step"), "goal %.2f coll %.2f height %.2f timeout %.2f" % (r["epstats/log/goal/max"], r["epstats/log/collision/max"],
          r["epstats/log/height/max"], r["epstats/log/timeout/max"]), "fps", round(r.get("fps/policy", 0), 1))
k = max(1, len(ep) // 8)
for i in range(0, len(ep), k):
    chunk = ep[i:i + k]
    print("  ep %4d-%4d step~%7d  score %.1f  len %.0f" % (i, i + len(chunk), chunk[-1]["step"],
          sum(c["episode/score"] for c in chunk) / len(chunk), sum(c["episode/length"] for c in chunk) / len(chunk)))
