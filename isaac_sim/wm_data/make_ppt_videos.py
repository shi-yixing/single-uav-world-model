"""Presentation videos (English captions, 1280 px wide, H.264/yuv420p).

  A: label video of one episode  -> depth | ego-centric labels | top-down map
  B: imagination vs reality      -> real depth | reconstruction / imagined depth | abs error

    python make_ppt_videos.py --ep ep_00497_env06_trap_goal.npz --out $Y/outputs/ppt_media
"""
import argparse
import os
import subprocess

import cv2
import numpy as np

import scenegen as sg

p = argparse.ArgumentParser()
p.add_argument("--ep", required=True)
p.add_argument("--data", default="/media/gsk/disk4t/yixing/data/wm_v1")
p.add_argument("--feats", default="/media/gsk/disk4t/yixing/runs/wm50m/features")
p.add_argument("--out", default="/media/gsk/disk4t/yixing/outputs/ppt_media")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
d = np.load(os.path.join(args.data, args.ep))
T = len(d["reward"])
W = 1280
FONT = cv2.FONT_HERSHEY_SIMPLEX


def writer(path, w, h, fps=20):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                             "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", path],
                            stdin=subprocess.PIPE)


def text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def depth_rgb(depth, size):
    g = ((sg.D_MAX - np.clip(depth.astype(np.float32), sg.D_MIN, sg.D_MAX)) / (sg.D_MAX - sg.D_MIN) * 255)
    img = cv2.cvtColor(cv2.applyColorMap(g.astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- video A
COL = {"seen_free": (110, 110, 110), "vis_free": (40, 70, 140), "obs_unseen": (70, 20, 20),
       "obs_mem": (255, 150, 0), "obs_vis": (230, 30, 30)}


def ego_rgb(occ, seen, vis, S):
    img = np.zeros((64, 64, 3), np.uint8)
    img[seen & ~occ] = COL["seen_free"]; img[vis & ~occ] = COL["vis_free"]
    img[occ & ~seen] = COL["obs_unseen"]; img[occ & seen & ~vis] = COL["obs_mem"]; img[occ & vis] = COL["obs_vis"]
    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(img, (S // 2, S // 2), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)
    return img


def world_rgb(t, S):
    scale = S / (2 * sg.EXTENT)
    px = lambda xy: (int((xy[0] + sg.EXTENT) * scale), int((sg.EXTENT - xy[1]) * scale))
    img = np.full((S, S, 3), 235, np.uint8)
    for cx, cy, hx, hy, yaw in d["boxes"]:
        R = sg.rot2(yaw)
        pts = np.array([px(np.array([cx, cy]) + R @ np.array([a * hx, b * hy])) for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        cv2.fillPoly(img, [pts], (90, 90, 100))
    cv2.polylines(img, [np.array([px(q) for q in d["pos"][:t + 1, :2]])], False, (220, 40, 40), 2, cv2.LINE_AA)
    cv2.circle(img, px(d["start"]), 6, (40, 120, 220), -1)
    cv2.circle(img, px(d["goal"]), 7, (40, 180, 60), -1)
    yaw = float(d["yaw"][t]); p0 = d["pos"][t, :2]
    fov = [px(p0)] + [px(p0 + sg.D_MAX * np.array([np.cos(yaw + a), np.sin(yaw + a)]))
                      for a in np.linspace(-sg.HFOV / 2, sg.HFOV / 2, 12)]
    over = img.copy(); cv2.fillPoly(over, [np.array(fov)], (120, 150, 230))
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    cv2.circle(img, px(p0), 5, (220, 40, 40), -1)
    return img


def video_a():
    S = 400                                   # label + map panels are S x S; depth panel 480 x 288 centred
    H = 40 + S + 70
    out = os.path.join(args.out, "video_A_trap_labels.mp4")
    ff = writer(out, W, H)
    for t in range(T):
        f = np.full((H, W, 3), 25, np.uint8)
        dep = depth_rgb(d["depth"][t], (480, 288))
        f[40 + (S - 288) // 2:40 + (S - 288) // 2 + 288, 0:480] = dep
        f[40:40 + S, 480:480 + S] = ego_rgb(d["occ"][t], d["seen"][t], d["vis"][t], S)
        f[40:40 + S, 880:880 + S] = world_rgb(t, S)
        text(f, "Depth input (48x80, 0.6-6 m)", (10, 28))
        text(f, "Ego-centric labels (9.6 m x 9.6 m)", (490, 28))
        text(f, "Top-down view", (890, 28))
        text(f, f"t = {t / 20:5.2f} s", (10, 40 + S - 10))
        mem = int((d["occ"][t] & d["seen"][t] & ~d["vis"][t]).sum())
        text(f, f"memory cells: {mem}", (490, 40 + S - 10), 0.55)
        # legend
        y = 40 + S + 28
        items = [("visible obstacle", COL["obs_vis"]), ("remembered obstacle (seen, out of view)", COL["obs_mem"]),
                 ("never-seen obstacle", COL["obs_unseen"]), ("visible free", COL["vis_free"]), ("seen free", COL["seen_free"])]
        x = 10
        for name, c in items:
            cv2.rectangle(f, (x, y - 12), (x + 16, y + 4), c, -1)
            text(f, name, (x + 22, y), 0.45)
            x += 30 + int(cv2.getTextSize(name, FONT, 0.45, 1)[0][0])
        text(f, "Scripted flight in Isaac Sim (trap mode): enter the trap, turn around, fly out to the goal (green)",
             (10, y + 30), 0.45, (200, 200, 200))
        ff.stdin.write(f.tobytes())
    ff.stdin.close(); ff.wait()
    return out


# ---------------------------------------------------------------- video B
def video_b():
    fz = np.load(os.path.join(args.feats, args.ep))
    rec = fz["recon_depth"].astype(np.float32) * (sg.D_MAX - sg.D_MIN) + sg.D_MIN
    img = fz["img_depth"].astype(np.float32) * (sg.D_MAX - sg.D_MIN) + sg.D_MIN
    C = int(fz["chunk"]); Hh = C // 2
    true = d["depth"].astype(np.float32)
    pw, ph = 408, 246               # even frame height for yuv420p
    H = 50 + ph + 90
    out = os.path.join(args.out, "video_B_imagination.mp4")
    ff = writer(out, W, H)
    maes = []
    for t in range(T):
        k = t % C
        imagining = k >= Hh and not np.isnan(img[t]).any()
        pred = img[t] if imagining else rec[t]
        err = np.abs(pred - true[t])
        mae = float(err.mean()); maes.append((imagining, mae))
        f = np.full((H, W, 3), 25, np.uint8)
        x0 = [8, 8 + pw + 16, 8 + 2 * (pw + 16)]
        f[50:50 + ph, x0[0]:x0[0] + pw] = depth_rgb(true[t], (pw, ph))
        f[50:50 + ph, x0[1]:x0[1] + pw] = depth_rgb(pred, (pw, ph))
        e = cv2.applyColorMap(np.clip(err / 2.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        f[50:50 + ph, x0[2]:x0[2] + pw] = cv2.resize(cv2.cvtColor(e, cv2.COLOR_BGR2RGB), (pw, ph), interpolation=cv2.INTER_NEAREST)
        text(f, "Real depth", (x0[0], 38))
        text(f, "Imagined depth" if imagining else "Reconstruction", (x0[1], 38))
        text(f, "Absolute error (0-2 m)", (x0[2], 38))
        if imagining:
            cv2.rectangle(f, (x0[1] - 4, 46), (x0[1] + pw + 3, 50 + ph + 3), (230, 40, 40), 4)
            status, col = f"IMAGINING (actions only)  step {k - Hh + 1}/{C - Hh}", (255, 90, 90)
        else:
            status, col = f"OBSERVING (camera input)  step {k + 1}/{Hh}", (120, 220, 120)
        y = 50 + ph + 32
        text(f, status, (8, y), 0.65, col, 2)
        text(f, f"MAE {mae:.2f} m", (x0[2], y), 0.65, (255, 255, 255), 2)
        # MAE strip over the last 160 steps
        sx, sw, sy, sh = 8, W - 16, y + 12, 36
        cv2.rectangle(f, (sx, sy), (sx + sw, sy + sh), (60, 60, 60), 1)
        hist = maes[-160:]
        for j, (im, m) in enumerate(hist):
            xx = sx + int(j / 160 * sw); hh = int(min(m / 1.0, 1) * sh)
            cv2.line(f, (xx, sy + sh), (xx, sy + sh - hh), (230, 70, 70) if im else (110, 200, 110), 3)
        text(f, f"t = {t / 20:5.2f} s", (W - 130, 38))
        ff.stdin.write(f.tobytes())
    ff.stdin.close(); ff.wait()
    return out


for path in (video_a(), video_b()):
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB")
