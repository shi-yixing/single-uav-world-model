import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scenegen as sg
runs = [("rl_base", "Baseline DreamerV3"), ("rl_map30", "DreamerV3 + map loss (λ=30)")]
root = sys.argv[1]
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for col, (suite, k) in enumerate([("utrap", 0), ("utrap", 1), ("ctrap", 0), ("ctrap", 1)]):
    for row, (run, title) in enumerate(runs):
        eps = json.load(open(f"{root}/{run}/eval/{suite}.json"))["episodes"]
        e = eps[k]
        sc = sg.sample_test_scene(np.random.default_rng(e["seed"]), suite)
        a = ax[row, col]
        for cx, cy, hx, hy, yaw in sc.boxes:
            R = sg.rot2(yaw)
            pts = [np.array([cx, cy]) + R @ np.array([sx * hx, sy * hy]) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
            a.add_patch(Polygon(pts, color="#888"))
        t = np.array(e["traj"])
        a.plot(t[:, 0], t[:, 1], color="#C44E52" if e["outcome"] != "goal" else "#55A868", lw=2)
        a.plot(*sc.start, "o", color="#4C72B0"); a.plot(*sc.goal, "*", color="#55A868", ms=14)
        if e["outcome"] == "collision":
            a.plot(t[-1, 0], t[-1, 1], "x", color="k", ms=10, mew=2)
        a.set_xlim(-10, 10); a.set_ylim(-10, 10); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{suite.upper()} #{k}: {e['outcome']}", fontsize=10)
        if col == 0:
            a.set_ylabel(title, fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=120)
