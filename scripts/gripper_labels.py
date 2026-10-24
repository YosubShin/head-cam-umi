#!/usr/bin/env python3
"""Dodeca-frame gripper-aperture estimators — the base-frame fix for the
cross-pose pooling artifact, plus the single-tail variant.

Per frame:
  1. fused dodeca body pose from >=2 faces (bundle model, temporal seeding)
  2. each gripper marker's position transformed into the BODY frame, where
     the jaw-travel direction is a physical constant across all arm poses
     and camera positions
Estimators (both graded against the actuator, e8_meas, wall-clock aligned):
  pair        tail<->base relative vector in body frame, projected on the
              pair's global body-frame travel axis
  single_tail one tail's body-frame position projected on its own global
              travel axis (doctrine upgrade candidate: needs no base marker)

Usage:
  python scripts/gripper_labels.py <log.json> <stem> [intrinsics.json]
"""
import json
import sys
from itertools import product

import cv2
import numpy as np

LOGF, STEM = sys.argv[1], sys.argv[2]
INTR = sys.argv[3] if len(sys.argv) > 3 else "calib/B0587_intrinsics.json"
BUNDLE = "calib/dodeca_bundle_ideal.json"
TAILS = {11: 12.16, 12: 12.16, 19: 12.16, 20: 12.16}
BASES = {15: 12.16, 16: 12.16, 21: 12.16, 22: 12.16}
EDGE = {**TAILS, **BASES}

intr = json.load(open(INTR))
K = np.array(intr["camera_matrix"]); dist = np.array(intr["dist_coeffs"])
bundle = json.load(open(BUNDLE))
T_body = {int(k): np.array(v) for k, v in bundle["markers"].items()}
BM = bundle["marker_edge_mm"] / 2000.0
DOBJ = np.array([[-BM, BM, 0], [BM, BM, 0], [BM, -BM, 0], [-BM, -BM, 0]], np.float64)


def obj(edge_mm):
    h = edge_mm / 2000.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


log = json.load(open(LOGF))
meas = [(e["t_wall"], e["gripper_pos"]) for e in log["events"]
        if e["event"] == "e8_meas"]
mt, mv = map(np.array, zip(*meas))

d = np.load(f"calib/{STEM}_dodeca_detections.npz")
ts = json.load(open(f"data/raw/{STEM}_ts.json"))

rows = []  # (t_wall, grip, {marker_id: p_body_mm})
last_rvec = last_tvec = None
n_body = 0
for k in sorted(int(x) for x in d["frame_indices"]):
    if k >= len(ts):
        continue
    tw = ts[k]["wall"]
    if not (mt[0] <= tw <= mt[-1]):
        continue
    ids, cs = d[f"i{k}"], d[f"c{k}"]
    # --- fused dodeca body pose ---
    op, ip = [], []
    for i, cc in zip(ids, cs):
        i = int(i)
        if i in T_body:
            P = (T_body[i] @ np.hstack([DOBJ, np.ones((4, 1))]).T).T[:, :3]
            op.append(P); ip.append(cc.reshape(4, 2).astype(np.float64))
    if len(op) < 2:
        continue
    op_all, ip_all = np.concatenate(op), np.concatenate(ip)
    if last_rvec is not None:
        ok, rvec, tvec = cv2.solvePnP(op_all, ip_all, K, dist,
                                      rvec=last_rvec.copy(), tvec=last_tvec.copy(),
                                      useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(op_all, ip_all, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        continue
    proj, _ = cv2.projectPoints(op_all, rvec, tvec, K, dist)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - ip_all.reshape(-1, 2)) ** 2)
                        .sum(axis=1).mean()))
    if rms > 5.0:
        continue
    last_rvec, last_tvec = rvec, tvec
    Rb, _ = cv2.Rodrigues(rvec)
    tb = tvec.ravel()
    n_body += 1
    # --- gripper markers into body frame ---
    pb = {}
    for i, cc in zip(ids, cs):
        i = int(i)
        if i in EDGE:
            ok2, rv2, t2 = cv2.solvePnP(obj(EDGE[i]),
                                        cc.reshape(4, 2).astype(np.float64),
                                        K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                pb[i] = (Rb.T @ (t2.ravel() - tb)) * 1000.0  # mm, body frame
    rows.append((tw, float(np.interp(tw, mt, mv)), pb))
print(f"{n_body} frames with fused dodeca pose + actuator alignment")


def grade(series, name):
    """series: list of (vec3_mm_in_body, grip). Global axis + fit + holdout."""
    if len(series) < 80:
        print(f"  {name}: n={len(series)} (too few)")
        return None
    vecs = np.array([v for v, _ in series]); gs = np.array([g for _, g in series])
    ctr = vecs.mean(axis=0)
    _, _, vt = np.linalg.svd(vecs - ctr, full_matrices=False)
    dts = (vecs - ctr) @ vt[0]
    tr, te = slice(0, None, 2), slice(1, None, 2)
    A = np.vstack([gs[tr], np.ones_like(gs[tr])]).T
    (sl, ic), *_ = np.linalg.lstsq(A, dts[tr], rcond=None)
    r = dts[te] - (sl * gs[te] + ic)
    swing = abs(sl)
    ok = swing > 20 and r.std() < 0.15 * swing
    print(f"  {name}: n={len(series)}  swing {sl:+.1f}mm/stroke  "
          f"holdout RMS {r.std():.1f}mm  {'GOOD' if ok else 'rejected'}")
    return (name, len(series), r.std()) if ok else None


print("\n== PAIR estimator (tail<->base relative vector, body frame, "
      "ONE GLOBAL FIT across all poses):")
good_pairs = []
for t, b in product(sorted(TAILS), sorted(BASES)):
    series = [(pb[t] - pb[b], g) for _, g, pb in rows if t in pb and b in pb]
    res = grade(series, f"{t}-{b}")
    if res:
        good_pairs.append((t, b))

print("\n== SINGLE-TAIL estimator (tail position in body frame, no base "
      "marker needed):")
good_tails = []
for t in sorted(TAILS):
    series = [(pb[t], g) for _, g, pb in rows if t in pb]
    res = grade(series, f"tail {t}")
    if res:
        good_tails.append(t)

# coverage under each doctrine
n = len(rows)
cov_pair = sum(1 for _, _, pb in rows
               if any(t in pb and b in pb for t, b in good_pairs)) / max(n, 1)
cov_tail = sum(1 for _, _, pb in rows
               if any(t in pb for t in good_tails)) / max(n, 1)
print(f"\ncoverage over dodeca-valid frames: pair doctrine {100*cov_pair:.1f}%  "
      f"single-tail doctrine {100*cov_tail:.1f}%")
