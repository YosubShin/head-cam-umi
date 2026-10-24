#!/usr/bin/env python3
"""Offline analysis of the operator's head motion during glove sessions, for the
active-camera idea: how much the head rotates vs translates, where the ball
(gripper) sits in the head image, and whether the head leads the hand.

Inputs: per-frame head-camera poses from the taped tiles
(calib/<stem>_cam_world.json, tiles_pose.py solve --per-frame) and per-frame
ball poses in the head camera (scratchpad head_ball_poses.json, labels_lib.body_frames).

  python scripts/head_motion_analysis.py yam_glove5_b0587b.take1 yam_glove5_b0587b
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S = Path("/tmp/claude-1000/-home-yosub-co/ab14273a-b823-4f30-b268-81870532d533/scratchpad")
intr = json.load(open(ROOT / "calib/B0587b_f3_intrinsics.json"))
K = np.array(intr["camera_matrix"]); W, H = 3840, 2160
HFOV = 2 * np.degrees(np.arctan(W / 2 / K[0, 0]))

eps = json.load(open(ROOT / "data/raw/yam_glove5_glove_episodes.json"))["episodes"]
ball_all = json.load(open(S / "head_ball_poses.json"))


def to_T(rv, tv):
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(np.asarray(rv, float))[0]; T[:3, 3] = np.asarray(tv, float).ravel(); return T


def yaw_pitch(view_dir_world):
    """View direction (unit vector in world) -> yaw about world z, pitch below horizontal (deg)."""
    d = view_dir_world / np.linalg.norm(view_dir_world)
    return np.degrees(np.arctan2(d[1], d[0])), np.degrees(np.arcsin(-d[2]))


per_ep = []
all_offaxis, all_px = [], []
lead_rows = []
for stem in sys.argv[1:]:
    cw = json.load(open(ROOT / f"calib/{stem}_cam_world.json"))
    ts = [e["wall"] for e in json.load(open(ROOT / f"data/raw/{stem}_ts.json"))]
    # single-tile PnP is ambiguous (planar flips put the camera under the table); keep >= 2 tiles, tight reprojection
    head = {r["k"]: (np.array(r["T_cam_world"]), r["rms"], r["tiles"]) for r in cw["per_frame"]
            if r["tiles"] >= 2 and r["rms"] <= 5.0}
    ball = {r["k"]: to_T(r["rvec"], r["tvec"]) for r in ball_all[stem]}
    ks = sorted(set(head) & set(ball))
    print(f"{stem}: {len(head)} head poses, {len(ball)} ball poses, {len(ks)} joint frames")
    for e in eps:
        kk = [k for k in ks if e["t_start"] <= ts[k] <= e["t_end"]]
        if len(kk) < 10:
            continue
        T_wc = np.array([np.linalg.inv(head[k][0]) for k in kk])          # camera pose in world
        pos = T_wc[:, :3, 3] * 1000                                          # mm
        view = T_wc[:, :3, 2]                                                # optical axis in world
        yp = np.array([yaw_pitch(v) for v in view])
        # ball in camera frame -> off-axis angle and pixel position
        pb = np.array([ball[k][:3, 3] for k in kk])                          # m, camera frame
        off = np.degrees(np.arccos(np.clip(pb[:, 2] / np.linalg.norm(pb, axis=1), -1, 1)))
        px = (K @ (pb / pb[:, 2:3]).T).T[:, :2]
        dist_mm = np.linalg.norm(pb, axis=1) * 1000
        # ball in world (for the lead analysis)
        pbw = np.array([(T_wc[i] @ np.append(pb[i], 1))[:3] * 1000 for i in range(len(kk))])
        t = np.array([ts[k] for k in kk])
        per_ep.append(dict(
            ep=e["episode_index"], n=len(kk), dur=e["duration_s"],
            pos_std=pos.std(0).tolist(), pos_range=(pos.max(0) - pos.min(0)).tolist(),
            path_mm=float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
            yaw_range=float(np.ptp(yp[:, 0])), pitch_range=float(np.ptp(yp[:, 1])),
            yaw_std=float(yp[:, 0].std()), pitch_std=float(yp[:, 1].std()),
            dist_med=float(np.median(dist_mm)),
            offaxis_med=float(np.median(off)), offaxis_p90=float(np.percentile(off, 90)),
            in_center30=float(np.mean((np.abs(px[:, 0] - W / 2) < 0.15 * W) & (np.abs(px[:, 1] - H / 2) < 0.15 * H))),
            px_std=px.std(0).tolist()))
        all_offaxis.extend(off.tolist()); all_px.extend(px.tolist())
        lead_rows.append((t, yp[:, 0], pbw, pos))

pe = per_ep
def med(key, idx=None):
    v = np.array([p[key] if idx is None else p[key][idx] for p in pe]); return np.median(v), np.percentile(v, 90)

print(f"\n== {len(pe)} episodes with joint head+ball poses; head camera HFOV {HFOV:.0f} deg ==")
print("head TRANSLATION within an episode (mm):")
for i, ax in enumerate("xyz"):
    m, p = med("pos_range", i); print(f"  range {ax}: median {m:5.0f}  p90 {p:5.0f}")
m, p = med("path_mm"); print(f"  path length per episode: median {m:.0f} mm, p90 {p:.0f} mm")
print("head ROTATION within an episode (deg):")
m, p = med("yaw_range"); print(f"  yaw range:   median {m:5.1f}  p90 {p:5.1f}")
m, p = med("pitch_range"); print(f"  pitch range: median {m:5.1f}  p90 {p:5.1f}")
m, _ = med("dist_med"); print(f"head-to-ball distance: median {m:.0f} mm")
# translation expressed as the angle it subtends at the working distance
tr = np.array([np.linalg.norm(p["pos_std"]) for p in pe]); d = np.array([p["dist_med"] for p in pe])
rot = np.array([np.hypot(p["yaw_std"], p["pitch_std"]) for p in pe])
print(f"translation std as an angle at working distance: median {np.degrees(np.median(tr / d)):.1f} deg  vs  rotation std median {np.median(rot):.1f} deg")
print("ball in the head image:")
m, p = med("offaxis_med"); print(f"  off-axis angle: median {m:.1f} deg (p90 over episodes {p:.1f});  all frames p90 {np.percentile(all_offaxis, 90):.1f} deg")
m, _ = med("in_center30"); print(f"  frames with the ball in the central 30% box: median {100 * m:.0f}%")
px = np.array(all_px); print(f"  ball pixel position: mean ({px[:, 0].mean():.0f}, {px[:, 1].mean():.0f}) of {W}x{H}; std ({px[:, 0].std():.0f}, {px[:, 1].std():.0f})")

# does the head lead the hand? cross-correlate head yaw rate with the ball's horizontal velocity direction
print("head-leads-hand: correlation of head yaw rate with ball lateral velocity (world), by lag")
lags = np.arange(-1.0, 1.01, 0.1)
corr = {l: [] for l in lags}
for t, yaw, pbw, pos in lead_rows:
    if len(t) < 30:
        continue
    tt = np.arange(t[0], t[-1], 0.1)
    yaw_i = np.interp(tt, t, np.unwrap(np.radians(yaw)))
    # ball bearing from the head position, in world yaw
    rel = np.array([np.interp(tt, t, pbw[:, i] - pos[:, i]) for i in range(2)]).T
    bear = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    dy, db = np.gradient(yaw_i), np.gradient(bear)
    for l in lags:
        s = int(round(l / 0.1))
        a, b = (dy[:-s], db[s:]) if s > 0 else (dy[-s:], db[:len(db) + s]) if s < 0 else (dy, db)
        if len(a) > 10 and a.std() > 0 and b.std() > 0:
            corr[l].append(np.corrcoef(a, b)[0, 1])
best = max(lags, key=lambda l: np.nanmedian(corr[l]) if corr[l] else -1)
for l in lags[::2]:
    print(f"  lag {l:+.1f}s: median r {np.nanmedian(corr[l]):+.2f}")
print(f"  best lag {best:+.1f}s (positive = head turns before the hand moves that way)")
json.dump(pe, open(S / "head_motion_per_episode.json", "w"))
