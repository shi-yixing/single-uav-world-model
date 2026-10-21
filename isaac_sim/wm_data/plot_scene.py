import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scenegen as sg
fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
for a, (k, s) in zip(ax, [("deadend", 200000), ("deadend", 200001), ("nested", 201000), ("nested", 201001)]):
    sc = sg.sample_test_scene(np.random.default_rng(s), k)
    for cx, cy, hx, hy, yaw in sc.boxes:
        R = sg.rot2(yaw)
        a.add_patch(Polygon([np.array([cx, cy]) + R @ np.array([u * hx, v * hy]) for u, v in [(-1, -1), (1, -1), (1, 1), (-1, 1)]], color="#888"))
    p = sc.planner.plan(sc.start, sc.goal)
    a.plot(p[:, 0], p[:, 1], "--", color="#4C72B0"); a.plot(*sc.start, "o", color="#4C72B0"); a.plot(*sc.goal, "*", color="#55A868", ms=14)
    a.arrow(*sc.start, *(1.2 * np.array([np.cos(sc.start_yaw), np.sin(sc.start_yaw)])), head_width=0.4, color="k")
    a.set_xlim(-11, 11); a.set_ylim(-11, 11); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(k)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=110)
