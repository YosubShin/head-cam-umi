#!/usr/bin/env python3
"""Pair the glove5 head-camera poses with the glove pose, in the neutral head frame, for the viewpoint model.

Per head frame (tile-solved, >= 2 tiles, rms <= 5 px, inside an episode): the glove body pose from the forward
cam's dodeca detections (same label stack as build_dataset, glove bundles) at the nearest wall time (<= 40 ms),
moved into the tile frame with the forward cam's tile pose for that take, then everything expressed in the
NEUTRAL head frame N (so101_cam_park.head_median_pose = where the SO-101 camera parks).

Writes calib/head_residual_glove5.npz with
  X: [grasp point in N (mm, 3) | approach axis in N (3) | grasp point 0.5 s earlier in N (mm, 3)]
  Y: [head position offset from N (mm, 3) | head view (yaw, pitch) deg in N (2)]
  ep: episode id (take-prefixed), t: wall time
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import labels_lib as L  # noqa: E402
import so101_cam_park as P  # noqa: E402
from build_dataset import load_stem  # noqa: E402

TAKES = [("yam_glove5_b0587.take1", "yam_glove5_b0587b.take1"), ("yam_glove5_b0587", "yam_glove5_b0587b")]
HIST_S = 0.5


def bearing_deg(v):
    return np.degrees(np.array([np.arctan2(v[0], v[2]), np.arctan2(v[1], v[2])]))


def main():
    mm = json.load(open(ROOT / "calib/marker_map_glove_yosub.json"))
    L.TAILS = {int(k): v for k, v in mm["tails"].items()}; L.BASES = {int(k): v for k, v in mm["bases"].items()}; L.TIPS = {int(k): v for k, v in mm["tips"].items()}
    K, dist = L.load_intrinsics(str(ROOT / "calib/B0587_intrinsics.json"))
    T_body, dobj = L.load_bundle(str(ROOT / "calib/glove_yosub_dodeca_bundle.json"))
    extra = L.load_gripper_bundle(str(ROOT / "calib/glove_yosub_gripper_bundle.json"))
    p_body = np.append(np.array(json.load(open(ROOT / "calib/eef_target_glove_yosub.json"))["p_body_mm"]) / 1000.0, 1.0)
    if "--neutral-all" in sys.argv:
        T_N = P.head_median_pose()                                     # all-frame median (idle frames included)
    else:                                                               # default: the median head pose DURING episodes
        T_N = np.array(json.load(open(ROOT / "calib/head_pose_episode_median.json"))["T_world_cam"])
    T_Ni = np.linalg.inv(T_N); print("neutral frame N = camera at", np.round(T_N[:3, 3] * 1000), "mm (tile frame)")
    eps = json.load(open(ROOT / "data/raw/yam_glove5_glove_episodes.json"))["episodes"]

    X, Y, EP, TT = [], [], [], []
    for fwd, head in TAKES:
        ts_f, det_f = load_stem(fwd, str(ROOT / "data/raw"), str(ROOT / "calib"))
        rows = L.body_frames(det_f, ts_f, K, dist, T_body, dobj, {}, extra=extra)
        tP = np.array([r["t_wall"] for r in rows])
        fw = json.load(open(ROOT / f"calib/{fwd}_cam_world.json"))
        pf = [r for r in fw["per_frame"] if r["T_cam_world"] is not None and r["tiles"] >= 3 and r["rms"] <= 5] if fw.get("per_frame") else []
        Tcw = np.array([r["T_cam_world"] for r in pf]) if pf else np.array([fw["T_cam_world"]])
        T_world_fwd = np.linalg.inv(np.median(Tcw, 0)) if len(Tcw) > 1 else np.linalg.inv(Tcw[0])   # static camera: median of frames
        if len(Tcw) > 1:                                                                             # re-orthonormalise after the median
            U, _, Vt = np.linalg.svd(T_world_fwd[:3, :3]); T_world_fwd[:3, :3] = U @ Vt
        print(f"{fwd}: {len(rows)} glove poses; forward cam at {np.round(T_world_fwd[:3, 3] * 1000)} mm in the tile frame ({len(Tcw)} tile solves)")

        hw = np.array([e["wall"] for e in json.load(open(ROOT / f"data/raw/{head}_ts.json"))])
        hp = json.load(open(ROOT / f"calib/{head}_cam_world.json"))["per_frame"]
        good = {r["k"]: np.linalg.inv(np.array(r["T_cam_world"])) for r in hp if r["T_cam_world"] is not None and r["tiles"] >= 2 and r["rms"] <= 5.0 and r["k"] < len(hw)}
        # glove grasp point in N per head frame (for the history feature)
        pN_by_k = {}
        for k, T_head in good.items():
            j = int(np.searchsorted(tP, hw[k])); cand = [c for c in (j - 1, j) if 0 <= c < len(tP)]
            if not cand:
                continue
            j = min(cand, key=lambda c: abs(tP[c] - hw[k]))
            if abs(tP[j] - hw[k]) > 0.04:
                continue
            r = rows[j]; R, _ = cv2.Rodrigues(np.asarray(r["rvec"], float).reshape(3, 1))
            T_fwd_body = np.eye(4); T_fwd_body[:3, :3] = R; T_fwd_body[:3, 3] = np.asarray(r["tvec"], float).ravel()
            T_world_body = T_world_fwd @ T_fwd_body
            p = (T_Ni @ T_world_body @ p_body)[:3] * 1000                                   # grasp point in N, mm
            ax = T_Ni[:3, :3] @ T_world_body[:3, :3] @ (p_body[:3] / np.linalg.norm(p_body[:3]))   # approach axis in N
            pN_by_k[k] = (p, ax, T_head)
        ks = np.array(sorted(pN_by_k)); kw = hw[ks]
        n_take = 0
        for i, k in enumerate(ks):
            t = hw[k]; ep = next((e["episode_index"] for e in eps if e["t_start"] <= t <= e["t_end"]), None)
            if ep is None:
                continue
            p, ax, T_head = pN_by_k[k]
            jh = int(np.searchsorted(kw, t - HIST_S)); jh = min(max(jh, 0), len(ks) - 1)
            ph = pN_by_k[ks[jh]][0] if abs(kw[jh] - (t - HIST_S)) < 0.2 else p
            TN_head = T_Ni @ T_head
            X.append(np.concatenate([p, ax, ph])); Y.append(np.concatenate([TN_head[:3, 3] * 1000, bearing_deg(TN_head[:3, 2])]))
            EP.append(f"{head[-5:]}_{ep}" if head.endswith("take1") else f"main_{ep}"); TT.append(t); n_take += 1
        print(f"  {head}: {len(good)} head poses, {len(pN_by_k)} paired with a glove pose, {n_take} inside episodes")
    X, Y = np.array(X), np.array(Y); EP = np.array(EP)
    out = ROOT / "calib/head_residual_glove5.npz"; np.savez(out, X=X, Y=Y, ep=EP, t=np.array(TT), T_N=T_N)
    print(f"wrote {out}: {len(X)} samples, {len(set(EP))} episodes")
    print("  head position offset from N (mm): std", np.round(Y[:, :3].std(0), 1), " view (yaw, pitch) std deg", np.round(Y[:, 3:].std(0), 1))
    print("  grasp point in N (mm): median", np.round(np.median(X[:, :3], 0)), " bearing median deg", np.round(np.median([bearing_deg(x[:3]) for x in X], 0), 1))


if __name__ == "__main__":
    main()
