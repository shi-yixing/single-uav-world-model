import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [json.loads(l) for l in open(sys.argv[1])]
st = [r for r in rows if "epstats/log/goal/max" in r]
ep = [r for r in rows if "episode/score" in r]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = [r["step"] / 1e3 for r in st]
for k, c in (("goal", "#55A868"), ("collision", "#C44E52"), ("height", "#8172B2"), ("timeout", "#8C8C8C")):
    ax[0].plot(x, [r[f"epstats/log/{k}/max"] for r in st], marker=".", color=c, label=k)
ax[0].set_xlabel("environment steps (k)"); ax[0].set_ylabel("fraction of episodes"); ax[0].set_ylim(0, 1)
ax[0].set_title("Episode outcomes"); ax[0].legend()
w = 50
sc = [r["episode/score"] for r in ep]
sm = [sum(sc[max(0, i - w):i + 1]) / len(sc[max(0, i - w):i + 1]) for i in range(len(sc))]
ax[1].plot([r["step"] / 1e3 for r in ep], sm, color="#4C72B0")
ax[1].set_xlabel("environment steps (k)"); ax[1].set_ylabel("episode return (50-episode mean)"); ax[1].set_title("Return")
fig.suptitle(sys.argv[3] if len(sys.argv) > 3 else "")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130)
