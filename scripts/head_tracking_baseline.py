#!/usr/bin/env python3
"""Gripper-tracking baselines for an active camera, scored against the operator's
actual head direction on held-out glove episodes.

Controllers (all produce a camera view direction, yaw/pitch, no roll):
  static      : one fixed direction per rig (median human direction over training episodes)
  centre      : point the camera at the ball
  offset      : point at the ball plus a fixed angular offset fitted on training episodes
                (the operator keeps the gripper lower-left of centre, not centred)
Errors are the angle between the controller's view direction and the human's, per frame,
reported on held-out episodes. Two base assumptions: the human's actual head position per
frame (direction-only question) and a fixed base at the per-episode median head position
(what a pan-tilt unit would have).

  python scripts/head_tracking_baseline.py yam_glove5_b0587b.take1 yam_glove5_b0587b
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S = Path("/tmp/claude-1000/-home-yosub-co/ab14273a-b823-4f30-b268-81870532d533/scratchpad")
eps = json.load(open(ROOT / "data/raw/yam_glove5_glove_episodes.json"))["episodes"]
ball_all = json.load(open(S / "head_ball_poses.json"))


def to_T(rv, tv):
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(np.asarray(rv, float))[0]; T[:3, 3] = np.asarray(tv, float).ravel(); return T


def yp(d):  # unit world direction -> yaw, pitch (rad)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.arctan2(d[..., 1], d[..., 0]), np.arcsin(-d[..., 2])


def ang_err(y1, p1, y2, p2):  # angle between two directions given as yaw/pitch (deg)
    d1 = np.stack([np.cos(p1) * np.cos(y1), np.cos(p1) * np.sin(y1), -np.sin(p1)], -1)
    d2 = np.stack([np.cos(p2) * np.cos(y2), np.cos(p2) * np.sin(y2), -np.sin(p2)], -1)
    return np.degrees(np.arccos(np.clip((d1 * d2).sum(-1), -1, 1)))


episodes = []  # per episode: human yaw/pitch, head pos, ball world pos
for stem in sys.argv[1:]:
    cw = json.load(open(ROOT / f"calib/{stem}_cam_world.json"))
    ts = [e["wall"] for e in json.load(open(ROOT / f"data/raw/{stem}_ts.json"))]
    head = {r["k"]: np.array(r["T_cam_world"]) for r in cw["per_frame"] if r["tiles"] >= 2 and r["rms"] <= 5.0}
    ball = {r["k"]: to_T(r["rvec"], r["tvec"]) for r in ball_all[stem]}
    ks = sorted(set(head) & set(ball))
    for e in eps:
        kk = [k for k in ks if e["t_start"] <= ts[k] <= e["t_end"]]
        if len(kk) < 10:
            continue
        T_wc = np.array([np.linalg.inv(head[k]) for k in kk])
        pos = T_wc[:, :3, 3]; view = T_wc[:, :3, 2]
        pb_c = np.array([ball[k][:3, 3] for k in kk])
        pb_w = np.einsum("nij,nj->ni", T_wc[:, :3, :3], pb_c) + pos
        hy, hp = yp(view)
        episodes.append(dict(ep=e["episode_index"], pos=pos, hy=hy, hp=hp, pb=pb_w))

rng = np.random.default_rng(0)
idx = rng.permutation(len(episodes)); n_tr = int(0.8 * len(episodes))
train, test = [episodes[i] for i in idx[:n_tr]], [episodes[i] for i in idx[n_tr:]]
print(f"{len(episodes)} episodes: {len(train)} train, {len(test)} held out")


def bearing(ep, fixed_base):
    base = np.median(ep["pos"], 0) if fixed_base else ep["pos"]
    by, bp = yp(ep["pb"] - base)
    return by, bp


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


for fixed_base in (False, True):
    label = "fixed base (pan-tilt at the episode's median head position)" if fixed_base else "human's actual head position"
    # fit on train: static direction, and the offset (human minus ball bearing)
    sy = np.median(np.concatenate([e["hy"] for e in train])); sp = np.median(np.concatenate([e["hp"] for e in train]))
    dy, dp = [], []
    for e in train:
        by, bp = bearing(e, fixed_base); dy.append(wrap(e["hy"] - by)); dp.append(e["hp"] - bp)
    oy, op = np.median(np.concatenate(dy)), np.median(np.concatenate(dp))
    # partial-gain tracker: view = static + g * (ball bearing - mean bearing), g fitted by least squares on train
    BY = np.concatenate([bearing(e, fixed_base)[0] for e in train]); BP = np.concatenate([bearing(e, fixed_base)[1] for e in train])
    HY = np.concatenate([e["hy"] for e in train]); HP = np.concatenate([e["hp"] for e in train])
    my, mp = np.median(BY), np.median(BP)
    gy = float(np.sum(wrap(BY - my) * wrap(HY - sy)) / np.sum(wrap(BY - my) ** 2)); gp = float(np.sum((BP - mp) * (HP - sp)) / np.sum((BP - mp) ** 2))
    res = {"static": [], "centre": [], "offset": [], "gain": []}
    for e in test:
        by, bp = bearing(e, fixed_base)
        res["static"].append(ang_err(np.full_like(e["hy"], sy), np.full_like(e["hp"], sp), e["hy"], e["hp"]))
        res["centre"].append(ang_err(by, bp, e["hy"], e["hp"]))
        res["offset"].append(ang_err(by + oy, bp + op, e["hy"], e["hp"]))
        res["gain"].append(ang_err(sy + gy * wrap(by - my), sp + gp * (bp - mp), e["hy"], e["hp"]))
    print(f"\n== base: {label} ==")
    print(f"  fitted offset: yaw {np.degrees(oy):+.1f} deg, pitch {np.degrees(op):+.1f} deg (ball relative to the view axis); fitted gains yaw {gy:.2f}, pitch {gp:.2f}")
    print(f"  {'controller':10s} {'median err':>10s} {'p90 err':>8s} {'RMS':>6s}   (deg, held-out frames)")
    for k, v in res.items():
        v = np.concatenate(v); print(f"  {k:10s} {np.median(v):10.1f} {np.percentile(v, 90):8.1f} {np.sqrt((v ** 2).mean()):6.1f}")
