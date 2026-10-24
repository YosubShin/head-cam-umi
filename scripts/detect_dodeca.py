#!/usr/bin/env python3
"""P0 pass 1: detect DICT_4X4_50 markers (dodecahedron faces) in every frame.

Saves per-frame corners for all detected markers to an .npz + summary CSV.
"""
import sys
import cv2
import numpy as np
import csv

import os
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/raw/C0755.MP4"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 2
F_START = int(sys.argv[3]) if len(sys.argv) > 3 else 0
F_END = int(sys.argv[4]) if len(sys.argv) > 4 else None
STEM = os.path.splitext(os.path.basename(VIDEO))[0]
SUFFIX = f"_{F_START}_{F_END}" if (F_START or F_END) else ""
OUT_NPZ = f"calib/{STEM}_dodeca_detections{SUFFIX}.npz"
OUT_CSV = f"calib/{STEM}_dodeca_detections{SUFFIX}.csv"

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
# APRILTAG refinement: markedly more robust to MJPEG artifacts and blur than
# SUBPIX (2026-08-31 A/B on b0591_feas: pair RMS 7.0 -> 4.3mm, coverage 4.5x)
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
params.cornerRefinementWinSize = 5
# allow the dictionary's full 1-bit error correction (DICT_4X4_50 min
# distance 4 -> id swaps impossible at 1 bit). A/B 2026-09-12 on frames
# where 22 decoded but 20 didn't: default 0.6 corrects 0 bits and rates
# <1.0 floor to 0; 1.0 recovers 31.8% of the misses, 3/400 frames gained
# a spurious UNKNOWN id (ignored downstream, but never assign id 17).
params.errorCorrectionRate = 1.0
detector = cv2.aruco.ArucoDetector(d, params)

cap = cv2.VideoCapture(VIDEO)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{VIDEO}: {n_total} frames, stride {STRIDE}", flush=True)

store = {}  # frame_idx -> (ids array, corners array Nx4x2)
rows = []
idx = -1
while True:
    ok = cap.grab()
    if not ok:
        break
    idx += 1
    if idx < F_START or idx % STRIDE:
        continue
    if F_END is not None and idx > F_END:
        break
    ok, img = cap.retrieve()
    if not ok:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids):
        ids_f = ids.flatten().astype(np.int32)
        c = np.stack([c.reshape(4, 2) for c in corners]).astype(np.float32)
        store[idx] = (ids_f, c)
        # mean side length in px, per frame
        side = float(np.mean([np.linalg.norm(cc[i] - cc[(i + 1) % 4])
                              for cc in c for i in range(4)]))
        rows.append((idx, len(ids_f), " ".join(map(str, sorted(ids_f))), round(side, 2)))
    else:
        rows.append((idx, 0, "", 0.0))
    if (idx // STRIDE) % 200 == 0:
        print(f"frame {idx}/{n_total}: {0 if idx not in store else len(store[idx][0])} markers", flush=True)
cap.release()

np.savez_compressed(
    OUT_NPZ,
    frame_indices=np.array(sorted(store.keys()), dtype=np.int32),
    **{f"i{k}": v[0] for k, v in store.items()},
    **{f"c{k}": v[1] for k, v in store.items()},
)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_idx", "n_markers", "ids", "mean_side_px"])
    w.writerows(rows)

nz = [r for r in rows if r[1] > 0]
from collections import Counter
cnt = Counter()
for k, (ids_f, _) in store.items():
    cnt.update(ids_f.tolist())
print(f"done: {len(nz)}/{len(rows)} frames with detections", flush=True)
print("per-id frame counts:", dict(sorted(cnt.items())), flush=True)
