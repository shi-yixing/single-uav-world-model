import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open(sys.argv[1]))
G = {"baseline (3 seeds)": ["base_s0", "base_s1", "base_s2"], "map loss λ=10 (3 seeds)": ["map10_s0", "map10_s1", "map10_s2"],
     "map loss λ=30 (1 seed)": ["map30_s0"], "current frame only": ["frame"]}
cols = ["#4C72B0", "#55A868", "#C44E52", "#8C8C8C"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
groups = ["vis", "mem", "unseen"]
for k, (name, ns) in enumerate(G.items()):
    m = [np.mean([r[n][g]["ap"] for n in ns]) for g in groups]
    s = [np.std([r[n][g]["ap"] for n in ns]) for g in groups]
    ax[0].bar(np.arange(3) + 0.2 * k, m, 0.2, yerr=s if len(ns) > 1 else None, capsize=3, label=name, color=cols[k])
    bins = list(r[ns[0]]["mem_by_age"])
    mm = np.array([np.mean([r[n]["mem_by_age"][b]["ap"] for n in ns]) for b in bins])
    ss = np.array([np.std([r[n]["mem_by_age"][b]["ap"] for n in ns]) for b in bins])
    ax[1].plot(range(len(bins)), mm, marker="o", color=cols[k], label=name)
    if len(ns) > 1:
        ax[1].fill_between(range(len(bins)), mm - ss, mm + ss, color=cols[k], alpha=0.2)
ax[0].set_xticks(np.arange(3) + 0.3); ax[0].set_xticklabels(["visible now", "memory", "never seen"])
ax[0].set_ylabel("obstacle average precision"); ax[0].set_title("Probe AP by cell group (mean ± std)"); ax[0].legend(fontsize=8)
labels = ["≤0.5 s", "0.5–1 s", "1–2 s", "2–4 s", "4–8 s", ">8 s"]
ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels)
ax[1].set_xlabel("time since the cell was last visible"); ax[1].set_ylabel("obstacle AP (memory cells)")
ax[1].set_title("Memory decay"); ax[1].set_ylim(0, 0.45); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=140)
