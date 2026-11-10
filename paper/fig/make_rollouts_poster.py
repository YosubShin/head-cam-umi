#!/usr/bin/env python3
"""Poster rollout strip: one placement (23), three rows, each the policy's OWN view: the head camera on the
moving positioner (arm 2), the wrist fisheye with head-camera labels (arm 8; the UMI arm 5 never dropped at a
placement where the fixed camera failed to grasp) and the fixed forward camera (arm 7).
Twelve moments per row on one shared time axis (every 2 s from 2 to 24 s); a row is black after its episode
ended, so the rows also show relative speed. The fixed camera's observation is square; the head and wrist frames
are centre-cropped to square so the three rows share one aspect ratio.

  python fig/make_rollouts_poster.py      # writes fig/rollouts_poster.jpg
"""
import glob, json
from pathlib import Path
import cv2, numpy as np

G = Path(__file__).resolve().parents[2] / "rollout_logs/grid_20261016"
OUT = Path(__file__).resolve().parent / "rollouts_poster.jpg"
SC = 23
H, W = 360, 360
TIMES = list(range(2, 26, 2))
ROWS = [  # policy, label
    ("mix_gHs3", "head camera on the moving positioner"),
    ("mix_gHA", "wrist fisheye, head labels"),
    ("mix_FB", "fixed forward camera"),
]
NICE = {"success": "success", "no_grasp": "no grasp", "dropped_before_bin": "dropped during the carry", "dropped_at_bin": "dropped at the bin", "timeout": "timeout"}
R = {(r["policy"], r["scene"]): r for r in map(json.loads, open(G / "trials.jsonl"))}

def frame_at(cap, t, fps):
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); i = int(t * fps)
    if i >= n: return None  # episode over: the cell stays black
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, im = cap.read(); return im

def label(im, text, org, scale, thick=2):
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2 * thick + 4, cv2.LINE_AA)
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (20, 20, 20), thick, cv2.LINE_AA)

rows = []
for n, name in ROWS:
    p = [x for x in glob.glob(str(G / f"{n}_S{SC}r0_*.mp4")) if "world" not in x][0]
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS)
    row = []
    for t in TIMES:
        im = frame_at(cap, t, fps)
        if im is None: im = np.zeros((H, W, 3), np.uint8)
        im = cv2.resize(im, (int(round(im.shape[1] * H / im.shape[0])), H), interpolation=cv2.INTER_AREA)
        x0 = (im.shape[1] - W) // 2; cell = im[:, x0:x0 + W].copy()  # centre crop to square
        label(cell, f"{t} s", (12, H - 16), 1.1)
        row.append(cell)
    strip = np.hstack(row); label(strip, f"{name}: {NICE[R[(n, SC)]['outcome']]}", (14, 48), 1.5, 3)
    rows.append(strip)
w = max(r.shape[1] for r in rows)
out = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])
cv2.imwrite(str(OUT), out, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(OUT, out.shape)
