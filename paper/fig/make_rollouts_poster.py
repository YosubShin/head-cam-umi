#!/usr/bin/env python3
"""Poster rollout strip, a 2 x 2 grid: two rows per placement, three moments each, from the fixed workspace camera with the policy's own view inset.
Placement 15: arm 2 (head camera) succeeds, arm 5 (wrist-camera UMI) drops the block during the carry.
Placement 16: arm 3 (48 teleop episodes alone) never grasps, arm 4 (48 teleop + 194 glove) succeeds.
The world camera exists only from placement 15 on, and 15 is the only placement >= 15 where arm 5 dropped.

  python fig/make_rollouts_poster.py      # writes fig/rollouts_poster.jpg
"""
import glob, json
from pathlib import Path
import cv2, numpy as np

G = Path(__file__).resolve().parents[2] / "rollout_logs/grid_20261016"
OUT = Path(__file__).resolve().parent / "rollouts_poster.jpg"
ROWS = [  # scene, policy, label, times (s)
    (15, "mix_gHs3", "2  head camera, 172 teleop + 194 glove", [8, 13, 17]),
    (15, "mix_FA", "5  wrist-camera UMI", [8, 13, 16]),
    (16, "head_Hgt48", "3  48 teleop episodes alone", [6, 10, 19]),
    (16, "mix_gHs3_48", "4  48 teleop + 194 glove", [6, 9, 13]),
]
NICE = {"success": "success", "no_grasp": "no grasp", "dropped_before_bin": "dropped during the carry", "dropped_at_bin": "dropped at the bin", "timeout": "timeout"}
R = {(r["policy"], r["scene"]): r for r in map(json.loads, open(G / "trials.jsonl"))}

def frame_at(path, t, fps):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(t * fps), n - 1)); ok, im = cap.read(); return im

def label(im, text, org, scale, thick=2):
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thick + 3, cv2.LINE_AA)
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (20, 20, 20), thick, cv2.LINE_AA)

rows = []
for sc, n, name, times in ROWS:
    w = glob.glob(str(G / f"{n}_S{sc}r0_*_world.mp4"))[0]; p = [x for x in glob.glob(str(G / f"{n}_S{sc}r0_*.mp4")) if "world" not in x][0]
    row = []
    for t in times:
        W = frame_at(w, t, 10); P = frame_at(p, t, 15)
        W = (W if W is not None else np.zeros((540, 960, 3), np.uint8))[40:500, 120:900].copy()
        if P is not None:
            ph = 170; pw = int(ph * P.shape[1] / P.shape[0]); P = cv2.resize(P, (pw, ph), interpolation=cv2.INTER_AREA)
            cv2.rectangle(P, (0, 0), (pw - 1, ph - 1), (255, 255, 255), 3); W[-ph:, -pw:] = P
        label(W, f"{t} s", (12, W.shape[0] - 16), 1.2)
        row.append(W)
    strip = np.hstack(row); label(strip, f"{name}: {NICE[R[(n, sc)]['outcome']]}   (placement {sc})", (14, 52), 1.6, 3)
    rows.append(strip)
gap = np.full((rows[0].shape[0] * 2, 24, 3), 255, np.uint8)
out = np.hstack([np.vstack(rows[:2]), gap, np.vstack(rows[2:])])  # placement 15 left, 16 right
cv2.imwrite(str(OUT), out, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(OUT, out.shape)
