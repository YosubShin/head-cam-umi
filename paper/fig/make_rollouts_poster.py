#!/usr/bin/env python3
"""Poster rollout strip: one placement (23), three rows, each the policy's OWN view: the head camera on the
moving positioner (arm 2), the wrist fisheye with head-camera labels (arm 8; the UMI arm 5 never dropped at a
placement where the fixed camera failed to grasp) and the fixed forward camera (arm 7).
Twelve moments per row at the same scale. The fixed camera's observation is square; the head and wrist frames
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
ROWS = [  # policy, label, times (s)
    ("mix_gHs3", "head camera on the moving positioner", [2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18]),
    ("mix_gHA", "wrist fisheye, head labels", [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    ("mix_FB", "fixed forward camera", [2, 5, 8, 11, 14, 17, 20, 24, 28, 32, 36, 40]),
]
NICE = {"success": "success", "no_grasp": "no grasp", "dropped_before_bin": "dropped during the carry", "dropped_at_bin": "dropped at the bin", "timeout": "timeout"}
R = {(r["policy"], r["scene"]): r for r in map(json.loads, open(G / "trials.jsonl"))}

def frame_at(cap, t, fps):
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(t * fps), n - 1)); ok, im = cap.read(); return im

def label(im, text, org, scale, thick=2):
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2 * thick + 4, cv2.LINE_AA)
    cv2.putText(im, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (20, 20, 20), thick, cv2.LINE_AA)

rows = []
for n, name, times in ROWS:
    p = [x for x in glob.glob(str(G / f"{n}_S{SC}r0_*.mp4")) if "world" not in x][0]
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS)
    row = []
    for t in times:
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
