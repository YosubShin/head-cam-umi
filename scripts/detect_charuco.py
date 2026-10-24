#!/usr/bin/env python3
"""Pass 1: ChArUco detection over sampled frames of a calibration clip.

Saves per-frame charuco corners/ids to an .npz plus a summary CSV.
Board: letter fallback, 10x7, 25.0mm square / 18.0mm marker, DICT_5X5_250.
"""
import sys
import cv2
import numpy as np
import csv

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/raw/C0754.MP4"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT_NPZ = "calib/C0754_detections.npz"
OUT_CSV = "calib/C0754_detections.csv"

SQUARE_M = 0.0250
MARKER_M = 0.0180

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
board = cv2.aruco.CharucoBoard((10, 7), SQUARE_M, MARKER_M, d)
detector = cv2.aruco.CharucoDetector(board)

cap = cv2.VideoCapture(VIDEO)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"{VIDEO}: {n_total} frames {W}x{H}, stride {STRIDE}", flush=True)

frames = {}  # idx -> (corners Nx2, ids N)
rows = []
idx = -1
while True:
    ok = cap.grab()
    if not ok:
        break
    idx += 1
    if idx % STRIDE:
        continue
    ok, img = cap.retrieve()
    if not ok:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cc, ci, mc, mi = detector.detectBoard(gray)
    n = 0 if cc is None else len(cc)
    if n >= 6:
        pts = cc.reshape(-1, 2)
        ids = ci.flatten()
        frames[idx] = (pts.astype(np.float32), ids.astype(np.int32))
        cx, cy = pts.mean(axis=0)
        # apparent scale: mean nearest-neighbor distance between adjacent grid corners
        gcols = 9
        grid = {int(i): p for i, p in zip(ids, pts)}
        dists = []
        for i, p in grid.items():
            j = i + 1
            if j in grid and (j % gcols):  # same row neighbor
                dists.append(np.linalg.norm(grid[j] - p))
        sq_px = float(np.mean(dists)) if dists else float("nan")
    else:
        cx = cy = sq_px = float("nan")
    rows.append((idx, n, cx, cy, sq_px))
    if (idx // STRIDE) % 100 == 0:
        print(f"frame {idx}/{n_total}: {n} corners", flush=True)
cap.release()

np.savez_compressed(
    OUT_NPZ,
    frame_indices=np.array(sorted(frames.keys()), dtype=np.int32),
    **{f"c{k}": v[0] for k, v in frames.items()},
    **{f"i{k}": v[1] for k, v in frames.items()},
)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_idx", "n_corners", "centroid_x", "centroid_y", "square_px"])
    w.writerows(rows)

det = [r for r in rows if r[1] >= 6]
print(f"done: {len(det)}/{len(rows)} sampled frames with >=6 corners", flush=True)
