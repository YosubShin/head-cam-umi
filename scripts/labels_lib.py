#!/usr/bin/env python3
"""Importable label stack for the offline dataset constructor.

Same machinery as gripper_labels.py (the validated grading CLI), packaged as
functions plus the inverse step the constructor needs: after calibrating each
estimator against the actuator, PREDICT a gripper value per frame from
whatever markers that frame offers (median across calibrated estimators).

Frames of reference:
  forward cam  — gripper markers are transformed into the DODECA BODY frame
                 (fused >=2-face pose, temporal seeding), where the jaw-travel
                 direction is a physical constant across arm poses.
  wrist cam    — the camera is rigid to the gripper base, so the CAMERA frame
                 is already a body frame: pass T_body=None and marker
                 positions stay in camera coordinates (tips 13/14 pipeline).

Estimator keys: "pair:T-B" (tail<->base relative vector) and "one:M"
(single-marker position), both projected on that estimator's global travel
axis fitted across the whole session.
"""
import json
from itertools import product

import cv2
import numpy as np

TAILS = {11: 12.16, 12: 12.16, 19: 12.16, 20: 12.16}
BASES = {15: 12.16, 16: 12.16, 21: 12.16, 22: 12.16}
TIPS = {13: 6.27, 14: 6.27}


def marker_obj(edge_mm):
    h = edge_mm / 2000.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def load_intrinsics(path):
    d = json.load(open(path))
    return np.array(d["camera_matrix"]), np.array(d["dist_coeffs"])


def load_bundle(path):
    b = json.load(open(path))
    T = {int(k): np.array(v) for k, v in b["markers"].items()}
    return T, marker_obj(b["marker_edge_mm"])


def load_gripper_bundle(path="calib/gripper_bundle.json"):
    """Calibrated BASE markers (rigid to the gripper body) as extra bundle
    members for the fused pose — usable when the dodeca is occluded.
    Returns {id: (T_marker->body 4x4, edge_mm)} or {} if uncalibrated."""
    try:
        b = json.load(open(path))
    except OSError:
        return {}
    return {int(k): (np.array(v), b["edge_mm"])
            for k, v in b["markers"].items()}


def solve_body_pose(ids, cs, K, dist, T_body, dobj, extra=None, last=None,
                    rms_max=5.0):
    """ONE frame's fused body pose in the camera frame, from 4x4 detections.

    ids: marker ids; cs: matching 4x2 corner arrays in SENSOR px (scale
    corners from a downscaled image first); extra: calibrated base markers
    {id: (T_marker->body, edge_mm)} (load_gripper_bundle); last: (rvec, tvec)
    of the running track (warm start + the single-marker fallback's gate).
    Returns (rvec, tvec, rms_px, n_faces, n_extra) or None. Same solve and
    gates as body_frames (live use: uvc_record.py POSE_PUB publisher)."""
    op, ip, n_extra = [], [], 0
    for i, cc in zip(ids, cs):
        i = int(i)
        if i in T_body:
            P = (T_body[i] @ np.hstack([dobj, np.ones((4, 1))]).T).T[:, :3]
        elif extra and i in extra:
            Tm, edge = extra[i]
            P = (Tm @ np.hstack([marker_obj(edge), np.ones((4, 1))]).T).T[:, :3]
            n_extra += 1
        else:
            continue
        op.append(P)
        ip.append(np.asarray(cc, dtype=np.float64).reshape(4, 2))
    single = len(op) == 1 and n_extra >= 1
    if len(op) < 2 and not single:
        return None
    if single and last is None:
        return None
    op_all, ip_all = np.concatenate(op), np.concatenate(ip)
    if last is not None:
        ok, rvec, tvec = cv2.solvePnP(op_all, ip_all, K, dist,
                                      rvec=last[0].copy(), tvec=last[1].copy(),
                                      useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(op_all, ip_all, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    if last is not None and not single:
        ok2, rv2, tv2 = cv2.solvePnP(op_all, ip_all, K, dist,
                                     flags=cv2.SOLVEPNP_SQPNP)
        if ok2:
            p1, _ = cv2.projectPoints(op_all, rvec, tvec, K, dist)
            p2, _ = cv2.projectPoints(op_all, rv2, tv2, K, dist)
            e1 = ((p1.reshape(-1, 2) - ip_all) ** 2).sum(1).mean()
            e2 = ((p2.reshape(-1, 2) - ip_all) ** 2).sum(1).mean()
            if e2 < e1:
                rvec, tvec = rv2, tv2
    if single:
        if np.linalg.norm(tvec.ravel() - last[1].ravel()) > 0.05 \
                or np.linalg.norm(rvec.ravel() - last[0].ravel()) > 0.5:
            return None
        rms = -1.0
    else:
        proj, _ = cv2.projectPoints(op_all, rvec, tvec, K, dist)
        rms = float(np.sqrt(((proj.reshape(-1, 2) - ip_all) ** 2)
                            .sum(axis=1).mean()))
        if rms > (rms_max if n_extra == 0 else rms_max * 1.6):
            return None
    return rvec, tvec, rms, len(op) - n_extra, n_extra


def body_frames(det, ts, K, dist, T_body, dobj, markers, rms_max=5.0,
                extra=None):
    """Per-frame fused dodeca pose + marker positions in the body frame.

    det: npz from detect_dodeca.py; ts: list of {"wall": ...} per frame;
    markers: {id: edge_mm} of gripper markers to localize.
    T_body=None skips the dodeca solve (wrist cam: camera frame IS the body
    frame — identity pose, every detected marker localized).

    Returns list of dicts {k, t_wall, rvec, tvec, rms, pb} where pb maps
    marker id -> position (mm) in the body frame. rvec/tvec are the dodeca
    body pose in the camera frame (None in the T_body=None mode).
    """
    rows = []
    last_rvec = last_tvec = None
    for k in sorted(int(x) for x in det["frame_indices"]):
        if k >= len(ts):
            continue
        ids, cs = det[f"i{k}"], det[f"c{k}"]
        if T_body is None:
            Rb, tb = np.eye(3), np.zeros(3)
            rvec = tvec = None
            rms = 0.0
        else:
            op, ip = [], []
            n_extra = 0
            for i, cc in zip(ids, cs):
                i = int(i)
                if i in T_body:
                    P = (T_body[i] @ np.hstack([dobj, np.ones((4, 1))]).T).T[:, :3]
                    op.append(P)
                    ip.append(cc.reshape(4, 2).astype(np.float64))
                elif extra and i in extra:
                    # calibrated base marker: rigid to the body, joins the
                    # same fused solve (coarser — calibration error adds in)
                    Tm, edge = extra[i]
                    P = (Tm @ np.hstack([marker_obj(edge),
                                         np.ones((4, 1))]).T).T[:, :3]
                    op.append(P)
                    ip.append(cc.reshape(4, 2).astype(np.float64))
                    n_extra += 1
            single = len(op) == 1 and n_extra >= 1
            if len(op) < 2 and not single:
                continue
            if single and last_rvec is None:
                continue  # single-marker pose only continues a track
            op_all, ip_all = np.concatenate(op), np.concatenate(ip)
            if last_rvec is not None:
                ok, rvec, tvec = cv2.solvePnP(
                    op_all, ip_all, K, dist, rvec=last_rvec.copy(),
                    tvec=last_tvec.copy(), useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, rvec, tvec = cv2.solvePnP(op_all, ip_all, K, dist,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            if last_rvec is not None and not single:
                # The warm start is a guess, not a prior: once ITERATIVE
                # settles in a wrong local minimum the gate below rejects
                # the frame, last_rvec never updates, and every following
                # frame starts from the same bad guess — glove3 2026-10-07
                # lost a whole episode (0/206 frames) that fuses at 1.6 px
                # from a cold start. Always also solve globally (SQPNP, no
                # guess) and keep whichever reprojects better.
                ok2, rv2, tv2 = cv2.solvePnP(op_all, ip_all, K, dist,
                                             flags=cv2.SOLVEPNP_SQPNP)
                if ok2:
                    p1, _ = cv2.projectPoints(op_all, rvec, tvec, K, dist)
                    p2, _ = cv2.projectPoints(op_all, rv2, tv2, K, dist)
                    e1 = ((p1.reshape(-1, 2) - ip_all) ** 2).sum(1).mean()
                    e2 = ((p2.reshape(-1, 2) - ip_all) ** 2).sum(1).mean()
                    if e2 < e1:
                        rvec, tvec = rv2, tv2
            if single:
                # ONE calibrated base marker (design intent 2026-09-14:
                # gripper-back keeps tracking the EEF when the dodeca is
                # occluded, at degraded accuracy). A 4-point solve fits
                # ~exactly, so RMS can't gate it — gate on temporal
                # consistency with the running track instead.
                if np.linalg.norm(tvec.ravel() - last_tvec.ravel()) > 0.05 \
                        or np.linalg.norm(rvec.ravel()
                                          - last_rvec.ravel()) > 0.5:
                    continue
                rms = -1.0   # marker: single-fallback row
            else:
                proj, _ = cv2.projectPoints(op_all, rvec, tvec, K, dist)
                rms = float(np.sqrt(
                    ((proj.reshape(-1, 2) - ip_all.reshape(-1, 2)) ** 2)
                    .sum(axis=1).mean()))
                if rms > (rms_max if n_extra == 0 else rms_max * 1.6):
                    continue  # looser gate for base-marker-leaning solves
            last_rvec, last_tvec = rvec, tvec
            Rb, _ = cv2.Rodrigues(rvec)
            tb = tvec.ravel()
        pb = {}
        # collect per-id first: a DUPLICATED id in one frame (today a
        # phantom decode; with the planned wedge-face id reuse, a second
        # physical instance) must never silently overwrite the real one.
        # Until wedge geometry exists for projection-based assignment,
        # drop ambiguous ids for the frame.
        seen = {}
        for i, cc in zip(ids, cs):
            seen.setdefault(int(i), []).append(cc)
        for i, ccs in seen.items():
            if i not in markers:
                continue
            if len(ccs) > 1:
                continue  # ambiguous instance — skip this id this frame
            ok2, _, t2 = cv2.solvePnP(marker_obj(markers[i]),
                                      ccs[0].reshape(4, 2).astype(np.float64),
                                      K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                pb[i] = (Rb.T @ (t2.ravel() - tb)) * 1000.0
        rows.append({"k": k, "t_wall": ts[k]["wall"], "rvec": rvec,
                     "tvec": tvec, "rms": rms, "pb": pb})
    return rows


def _series(rows, key, mt, mv, tol=None):
    """Vector series for one estimator over actuator-covered frames.

    tol: max |t_frame - nearest measurement| in seconds. None keeps the
    dense-actuator behaviour (interpolate anywhere inside the series).
    Sparse caliper holds (glove sessions: events only DURING holds) need a
    tol, or every transit frame between two holds gets an interpolated ramp
    value and corrupts the fit.
    """
    kind, ids = key.split(":")
    out = []
    for r in rows:
        tw = r["t_wall"]
        if not (mt[0] <= tw <= mt[-1]):
            continue
        if tol is not None:
            j = np.searchsorted(mt, tw)
            near = min(abs(tw - mt[max(j - 1, 0)]),
                       abs(tw - mt[min(j, len(mt) - 1)]))
            if near > tol:
                continue
        pb = r["pb"]
        if kind == "pair":
            t, b = (int(x) for x in ids.split("-"))
            if t in pb and b in pb:
                out.append((pb[t] - pb[b], float(np.interp(tw, mt, mv))))
        else:
            m = int(ids)
            if m in pb:
                out.append((pb[m], float(np.interp(tw, mt, mv))))
    return out


def fit_estimators(rows, mt, mv, tails=None, bases=None, singles=None,
                   min_n=80, min_swing=20.0, max_rms_frac=0.15, verbose=True,
                   meas_tol=None):
    """Calibrate every candidate estimator vs the actuator (even/odd holdout,
    same accept gate as gripper_labels.py) and keep the GOOD ones.

    Returns {key: est} where est carries the global travel axis and the
    inverse linear map used by predict_aperture. mm-RMS thresholds match the
    validated grader; est["grms"] is the holdout RMS in GRIPPER units.
    """
    tails = TAILS if tails is None else tails
    bases = BASES if bases is None else bases
    singles = list(tails) if singles is None else singles
    keys = [f"pair:{t}-{b}" for t, b in product(sorted(tails), sorted(bases))] \
        + [f"one:{m}" for m in sorted(singles)]
    ests = {}
    for key in keys:
        ser = _series(rows, key, mt, mv, tol=meas_tol)
        if len(ser) < min_n:
            if verbose:
                print(f"  {key}: n={len(ser)} (too few)")
            continue
        vecs = np.array([v for v, _ in ser])
        gs = np.array([g for _, g in ser])
        ctr = vecs.mean(axis=0)
        _, _, vt = np.linalg.svd(vecs - ctr, full_matrices=False)
        dts = (vecs - ctr) @ vt[0]
        tr, te = slice(0, None, 2), slice(1, None, 2)
        A = np.vstack([gs[tr], np.ones_like(gs[tr])]).T
        (sl, ic), *_ = np.linalg.lstsq(A, dts[tr], rcond=None)
        res = dts[te] - (sl * gs[te] + ic)
        # the fit must SPAN the stroke: samples from one caliper hold fit a
        # line through a point with a meaningless slope and a flattering
        # holdout (glove1 2026-10-07: pair 2-3 seen only at the closed hold
        # scored 0.1 mm and drove every width label to ~0)
        span = float(gs.max() - gs.min())
        ok = (abs(sl) > min_swing and res.std() < max_rms_frac * abs(sl)
              and span >= 0.4)
        if verbose:
            why = "" if ok else (" (single-hold fit)" if span < 0.4 else "")
            print(f"  {key}: n={len(ser)}  swing {sl:+.1f}mm/stroke  "
                  f"holdout RMS {res.std():.1f}mm  "
                  f"{'GOOD' if ok else 'rejected' + why}")
        if ok:
            ests[key] = {"ctr": ctr, "axis": vt[0], "slope": sl, "icpt": ic,
                         "rms_mm": float(res.std()),
                         "grms": float(res.std() / abs(sl)), "n": len(ser)}
    return ests


def predict_aperture(pb, ests):
    """Gripper value from one frame's body-frame markers: median over the
    calibrated estimators available in this frame. Returns (value, n_used);
    (None, 0) when no estimator's markers are visible."""
    preds = []
    for key, e in ests.items():
        kind, ids = key.split(":")
        if kind == "pair":
            t, b = (int(x) for x in ids.split("-"))
            if t in pb and b in pb:
                v = pb[t] - pb[b]
            else:
                continue
        else:
            m = int(ids)
            if m in pb:
                v = pb[m]
            else:
                continue
        d = float((v - e["ctr"]) @ e["axis"])
        preds.append((d - e["icpt"]) / e["slope"])
    if not preds:
        return None, 0
    return float(np.median(preds)), len(preds)
