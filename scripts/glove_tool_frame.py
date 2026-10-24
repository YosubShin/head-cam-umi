#!/usr/bin/env python3
"""Glove ball -> virtual flange transform, from marker geometry alone.

The joint-action build maps a ball pose to a robot flange through the
hand-eye's T_flange<-body. That transform is the ROBOT ball's mount; the glove
ball sits on its gripper differently (2026-10-07: targets came out rotated
~175 deg and 21 cm high, IK residual 310 mm, base joint pinned at its limit).

Fix: build the same TOOL frame on both grippers from what the forward camera
sees — origin = fingertip-marker midpoint, z = finger axis (gripper-base
marker cluster -> tips), x = jaw axis (tip->tip, sign such that the ball lies
on +y) — then
    T_flange<-body_glove = T_flange<-body_robot . T_body<-tool_robot . T_tool<-body_glove
so the glove's tool frame lands exactly where the robot's does.

Writes calib/handeye_glove_<unit>.json (T_cam_base copied from --handeye).
"""
import argparse
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import labels_lib as L  # noqa: E402


def tool_frame(stems, bundle, gripper_bundle, marker_map, intrinsics, label):
    mm = json.load(open(marker_map))
    bases = {int(k): v for k, v in mm["bases"].items()}
    tails = {int(k): v for k, v in mm["tails"].items()}
    tips = {int(k): v for k, v in mm["tips"].items()}
    K, dist = L.load_intrinsics(intrinsics)
    Tb, dobj = L.load_bundle(bundle)
    extra = L.load_gripper_bundle(gripper_bundle)
    tipA, tipB, basec = [], [], []
    ta, tb = sorted(tips)
    for stem in stems:
        det = np.load(f"calib/{stem}_dodeca_detections.npz")
        ts = json.load(open(f"data/raw/{stem}_ts.json"))
        rows = {r["k"]: r for r in L.body_frames(det, ts, K, dist, Tb, dobj,
                                                 {**bases, **tails}, extra=extra)
                if r["rms"] >= 0}
        tdet = np.load(f"calib/{stem}_tips16h5_detections.npz")
        for k in tdet["frame_indices"]:
            k = int(k)
            r = rows.get(k)
            if r is None:
                continue
            ids = [int(x) for x in tdet[f"i{k}"]]
            if ta not in ids or tb not in ids:
                continue
            Rb, _ = cv2.Rodrigues(r["rvec"])
            tc = r["tvec"].ravel()
            pos = {}
            for i, cc in zip(ids, tdet[f"c{k}"]):
                if i in tips:
                    ok, _, t2 = cv2.solvePnP(L.marker_obj(tips[i]),
                                             cc.reshape(4, 2).astype(np.float64),
                                             K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if ok:
                        pos[i] = (Rb.T @ (t2.ravel() - tc)) * 1000.0
            if ta in pos and tb in pos:
                tipA.append(pos[ta])
                tipB.append(pos[tb])
                bb = [r["pb"][b] for b in bases if b in r["pb"]]
                if bb:
                    basec.append(np.mean(bb, axis=0))
    tipA, tipB, basec = map(np.array, (tipA, tipB, basec))
    print(f"{label}: {len(tipA)} frames with both tips + ball; base cluster "
          f"frames {len(basec)}")
    o = np.median((tipA + tipB) / 2, axis=0)
    jaw = np.median(tipB - tipA, axis=0)
    spread = np.linalg.norm(tipB - tipA, axis=1)
    print(f"   tip separation median {np.median(spread):.1f} mm "
          f"(p5 {np.percentile(spread, 5):.1f}, p95 {np.percentile(spread, 95):.1f})")
    bc = np.median(basec, axis=0)
    z = o - bc
    z /= np.linalg.norm(z)
    x = jaw - (jaw @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if (-o) @ y < 0:          # ball (body origin) must lie on +y: flip jaw sign
        x, y = -x, -y
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, o / 1000.0
    print(f"   tool origin (body, mm) {np.round(o, 1)}  finger axis {np.round(z, 2)}  "
          f"jaw axis {np.round(x, 2)}  base->tips {np.linalg.norm(o - bc):.0f} mm")
    return T                   # T_body<-tool


ap = argparse.ArgumentParser()
ap.add_argument("--unit", default="yosub")
ap.add_argument("--handeye", default="calib/handeye_yam_he7.json")
ap.add_argument("--robot-stems", default="yam_pilot5_b0587.take1")
ap.add_argument("--glove-stems", default="yam_glove2_b0587.take2,yam_glove3_b0587")
a = ap.parse_args()
Tr = tool_frame(a.robot_stems.split(","), "calib/dodeca_bundle_ideal.json",
                "calib/gripper_bundle.json", "calib/marker_map_v3.json",
                "calib/B0587_intrinsics.json", "ROBOT")
Tg = tool_frame(a.glove_stems.split(","), f"calib/glove_{a.unit}_dodeca_bundle.json",
                f"calib/glove_{a.unit}_gripper_bundle.json",
                f"calib/marker_map_glove_{a.unit}.json",
                "calib/B0587_intrinsics.json", "GLOVE")
he = json.load(open(a.handeye))
Tfb_r = np.array(he["T_flange_body"])
Tfb_g = Tfb_r @ Tr @ np.linalg.inv(Tg)
out = {"T_cam_base": he["T_cam_base"], "T_flange_body": Tfb_g.tolist(),
       "source": {"handeye": a.handeye, "robot_stems": a.robot_stems,
                  "glove_stems": a.glove_stems,
                  "T_body_tool_robot": Tr.tolist(), "T_body_tool_glove": Tg.tolist()},
       "note": "glove ball -> virtual flange via matched tool frames "
               "(fingertip midpoint, finger axis, jaw axis); see scripts/glove_tool_frame.py"}
dst = f"calib/handeye_glove_{a.unit}.json"
json.dump(out, open(dst, "w"), indent=1)
rel = np.linalg.inv(Tfb_r) @ Tfb_g
ang = np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3]) - 1) / 2, -1, 1)))
print(f"glove ball sits {ang:.0f} deg rotated and {np.linalg.norm(rel[:3, 3]) * 1000:.0f} mm "
      f"offset from the robot ball's mount -> wrote {dst}")
