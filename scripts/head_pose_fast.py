#!/usr/bin/env python3
"""Per-frame head-camera pose from the taped tiles, fast: detect 36h11 tiles on a
half-resolution frame, refine the corners on the full-resolution frame, then the
same multi-tile PnP as tiles_pose.py. Writes calib/<stem>_cam_world.json in the
same per-frame format.

  python scripts/head_pose_fast.py yam_glove5_b0587b --stride 4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels_lib as L  # noqa: E402

# tiles_pose.py parses argv at import time, so its solver is copied here verbatim
def tag_obj(edge_m):
    h = edge_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def to_T(rv, tv):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))
    T[:3, 3] = np.asarray(tv, float).ravel()
    return T


def solve_frame(corners_by_id, layout, K, dist, last=None):
    """Multi-tile PnP: all corners of all decoded tiles vs world coordinates.

    ONE tile: IPPE_SQUARE needs the canonical centred square, not world corners (world corners gave ~190 px
    garbage and lost every single-tile frame). Solve tile->cam on the canonical square, take BOTH IPPE
    solutions, compose with the tile's world pose and keep the one continuous with `last` (T_cam_world of
    the previous accepted frame); without a `last` a lone tile is skipped (the flip is undecidable)."""
    edge = layout["edge_mm"] / 1000.0
    obj_t = tag_obj(edge)
    O, I, TW = [], [], []
    for i, c in corners_by_id.items():
        t = layout["tiles"].get(str(i))
        if t is None:
            continue
        Tw = to_T(t["rvec"], np.array(t["xyz_mm"]) / 1000.0)
        O.append((Tw[:3, :3] @ obj_t.T).T + Tw[:3, 3])
        I.append(c.reshape(4, 2).astype(float))
        TW.append(Tw)
    if not O:
        return None
    if len(O) == 1:
        if last is None:
            return None
        last_T, dt = last                                       # previous ACCEPTED pose (any tile count), seconds since
        n_sol, rvs, tvs, errs = cv2.solvePnPGeneric(obj_t, I[0], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        best = None
        # a worn head moves: the continuity bound scales with the gap (0.6 m/s, 120 deg/s, floors 3 cm / 5 deg)
        dp_max, dr_max = 0.03 + 0.6 * dt, 5 + 120 * dt
        for rv, tv, e in zip(rvs, tvs, np.asarray(errs).ravel()):
            T_cw = to_T(rv, tv) @ np.linalg.inv(TW[0])          # cam<-tile . tile<-world
            dp = np.linalg.norm(T_cw[:3, 3] - last_T[:3, 3])
            dr = np.degrees(np.arccos(np.clip((np.trace(last_T[:3, :3].T @ T_cw[:3, :3]) - 1) / 2, -1, 1)))
            score = dp / dp_max + dr / dr_max
            if dp <= dp_max and dr <= dr_max and (best is None or score < best[1]):
                best = (T_cw, score, float(e))
        if best is None:
            return None
        return best[0], best[2], 1

    def fit(Os, Is):
        Oc, Ic = np.concatenate(Os), np.concatenate(Is)
        flags = cv2.SOLVEPNP_IPPE_SQUARE if len(Oc) == 4 else cv2.SOLVEPNP_SQPNP
        ok, rv, tv = cv2.solvePnP(Oc, Ic, K, dist, flags=flags)
        if not ok:
            return None
        if len(Oc) > 4:
            ok, rv, tv = cv2.solvePnP(Oc, Ic, K, dist, rvec=rv, tvec=tv, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(Oc, rv, tv, K, dist)
        per_tile = np.sqrt(((proj.reshape(-1, 2) - Ic) ** 2).sum(1)).reshape(-1, 4).mean(1)
        return rv, tv, per_tile

    r = fit(O, I)
    if r is None:
        return None
    rv, tv, per_tile = r
    # a tile that was bumped after the layout was taken disagrees with the
    # others by many px: drop the worst and refit while >= 3 tiles remain and
    # it reprojects far worse than the rest (taped tiles never trip this)
    while len(O) >= 4 and per_tile.max() > max(8.0, 4 * np.median(per_tile)):
        j = int(np.argmax(per_tile))
        O.pop(j); I.pop(j)
        r = fit(O, I)
        if r is None:
            return None
        rv, tv, per_tile = r
    return to_T(rv, tv), float(np.sqrt((per_tile ** 2).mean())), len(O)



ap = argparse.ArgumentParser()
ap.add_argument("stem"); ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--intr", default="calib/B0587b_f3_intrinsics.json")
ap.add_argument("--tags", default="calib/tags36h11_yosub.json")
ap.add_argument("--layout", default="calib/workspace_tiles_pilot_v2.json")
ap.add_argument("--out", default=None, help="default calib/<stem>_cam_world.json")
a = ap.parse_args()
ROOT = Path(__file__).resolve().parent.parent
K, dist = L.load_intrinsics(a.intr)
layout = json.load(open(a.layout))
ids_ok = set(int(i) for i in json.load(open(a.tags))["ids"])
p = cv2.aruco.DetectorParameters(); p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

rows = []
last_T, last_k, anchor_k = None, None, None          # last accepted pose; last multi-tile (anchor) frame
FPS = 24.0
cap = cv2.VideoCapture(str(ROOT / f"data/raw/{a.stem}.mkv")); k = 0; misses = 0
while misses < 50:
    if not cap.grab():
        ok, _ = cap.read(); misses += 1
        if not ok and misses >= 5:
            break
        k += 1; continue
    misses = 0
    if k % a.stride == 0:
        ok, im = cap.retrieve()
        if ok:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cs, ids, _ = det.detectMarkers(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
            if ids is not None:
                by = {}
                for i, c in zip(ids.ravel(), cs):
                    if int(i) in ids_ok:
                        c2 = (c.reshape(4, 2) * 2).astype(np.float32)
                        by[int(i)] = cv2.cornerSubPix(g, c2, (5, 5), (-1, -1), crit).reshape(1, 4, 2)
                # single-tile frames chain frame-to-frame from the last accepted pose, for up to 10 s past the
                # last multi-tile anchor (and at most 1 s since the last accepted frame of any kind)
                chain_ok = (last_k is not None and k - last_k <= 1.0 * FPS
                            and anchor_k is not None and k - anchor_k <= 10.0 * FPS)
                r = solve_frame(by, layout, K, dist, last=(last_T, (k - last_k) / FPS) if chain_ok else None) if by else None
                if r is not None:
                    T, e, n = r
                    rows.append({"k": k, "T_cam_world": T.tolist(), "rms": float(e), "tiles": int(n)})
                    if n >= 2 and e <= 5.0:
                        last_T, last_k, anchor_k = T, k, k
                    elif n == 1:
                        last_T, last_k = T, k
    k += 1
cap.release()
rms = [r["rms"] for r in rows]
print(f"{a.stem}: {len(rows)} frames solved of {k // a.stride} sampled; tiles/frame median "
      f"{np.median([r['tiles'] for r in rows]):.0f}; reprojection RMS median {np.median(rms):.2f} px")
out = Path(a.out) if a.out else ROOT / f"calib/{a.stem}_cam_world.json"
import os
tmp = str(out) + ".tmp"
json.dump({"stem": a.stem, "layout": a.layout, "stride": a.stride, "n_frames": len(rows),
           "rms_px_median": float(np.median(rms)), "n_single_tile": int(sum(r["tiles"] == 1 for r in rows)),
           "per_frame": rows}, open(tmp, "w"))
os.replace(tmp, out)                                       # readers never see a half-written file
print("wrote", out)
