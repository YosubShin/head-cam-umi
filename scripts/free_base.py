#!/usr/bin/env python3
"""Virtual robot base for a glove session recorded on ANY rig.

Absolute joint targets need the ball pose in the robot's base frame. On our
rig that is the hand-eye (camera -> base). A rig without a robot has no
hand-eye, so — as labs do when converting UMI data to joint targets — we
treat the base pose as a free variable and choose the one under which the
demonstrations are reachable with the robot's usual posture:

  minimise over T_cam_base:  mean_frames [ IK residual (unreachable -> big)
                              + lambda * ||(q - mu) / sigma||^2 (posture prior)
                              + joint-limit penalty ]

with the posture prior (mu, sigma) from the teleop corpus
(calib/robot_joint_prior.json). The ball -> flange transform comes from the
glove's tool frame (rig-independent). Initialised at our own hand-eye and
optimised over a 6-DoF se(3) perturbation (Nelder-Mead; the objective is
cheap: a few hundred IK solves per evaluation).

This is only valid for policies whose observations do not reveal the base —
the wrist-camera arm. A fixed forward camera needs the real hand-eye.

Usage:
  python scripts/free_base.py <forward_stem> [<forward_stem> ...] --unit <unit> --out calib/handeye_virtual_<rig>.json
     [--init calib/handeye_yam_he7.json] [--intr calib/B0587_intrinsics.json] [--frames 300] [--perturb 0]
--perturb <m> starts from the init shifted by a random <m>-metre offset and
~30 degrees (a self-test: how close does the solve return to the true
hand-eye on our own data?).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path.home() / "co/i2rt"))
import labels_lib as L  # noqa: E402
from i2rt.robots.kinematics import Kinematics  # noqa: E402
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml  # noqa: E402


def se3(x):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(x[3:], float).reshape(3, 1))
    T[:3, 3] = x[:3]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--unit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="calib/handeye_yam_he7.json")
    ap.add_argument("--intr", default="calib/B0587_intrinsics.json")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.05, help="posture prior weight")
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--plane", action="store_true", help="constrain the base's height and tilt to the TABLE "
                    "plane (fingertip lower envelope) using the reference from our rig; solve x, y, yaw only")
    ap.add_argument("--plane-ref", default="calib/base_above_grasp_plane.json",
                    help="base height/tilt above the table plane measured on our rig (written with --calibrate-plane)")
    ap.add_argument("--calibrate-plane", action="store_true", help="our rig: measure the base pose relative to the "
                    "table plane from the init hand-eye and write --plane-ref")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    mm = json.load(open(f"calib/marker_map_glove_{a.unit}.json"))
    L.TAILS = {int(k): v for k, v in mm["tails"].items()}
    L.BASES = {int(k): v for k, v in mm["bases"].items()}
    K, dist = L.load_intrinsics(a.intr)
    Tb, dobj = L.load_bundle(f"calib/glove_{a.unit}_dodeca_bundle.json")
    extra = L.load_gripper_bundle(f"calib/glove_{a.unit}_gripper_bundle.json")
    glove_he = json.load(open(f"calib/handeye_glove_{a.unit}.json"))
    Binv = np.linalg.inv(np.array(glove_he["T_flange_body"]))       # rig-independent: glove ball -> virtual flange
    init = json.load(open(a.init))
    T0 = np.array(init["T_cam_base"])
    prior = json.load(open("calib/robot_joint_prior.json"))
    mu, sig = np.array(prior["mean"]), np.array(prior["std"])
    kin = Kinematics(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")
    lo, hi = np.array([-3.0, -0.1, -0.1, -1.7, -1.6, -2.0]), np.array([3.0, 3.1, 3.2, 1.7, 1.6, 2.1])  # YAM approx limits

    # ball poses in the camera frame (M), sampled across the session(s)
    Ms = []
    for stem in a.stems:
        det = np.load(f"calib/{stem}_dodeca_detections.npz")
        ts = json.load(open(f"data/raw/{stem}_ts.json"))
        marks_p = Path(f"data/raw/{stem.split('_b0587')[0]}_glove_episodes.json")
        eps = json.load(open(marks_p))["episodes"] if marks_p.exists() else None
        for r in L.body_frames(det, ts, K, dist, Tb, dobj, {}, extra=extra):
            if r["rms"] < 0:
                continue
            if eps and not any(e["t_start"] <= r["t_wall"] <= e["t_end"] for e in eps):
                continue
            Rm, _ = cv2.Rodrigues(np.asarray(r["rvec"], np.float64))
            M = np.eye(4); M[:3, :3], M[:3, 3] = Rm, r["tvec"].ravel()
            Ms.append(M)
    Ms = np.array(Ms)
    sel = Ms[rng.choice(len(Ms), size=min(a.frames, len(Ms)), replace=False)]

    # ---- table plane (optional, --plane): the lower envelope of the fingertip
    # (tool-origin) trajectory. The tips touch the table band at every pick, so
    # the 7% of points farthest along "down" lie on one plane; "down" is refined
    # from the camera axis to the plane normal in a few iterations. On our rig
    # this plane is within 4 deg / 8 mm of the tile-measured table (2026-10-08).
    plane = None
    if a.plane:
        T_body_tool = np.array(glove_he["source"]["T_body_tool_glove"])
        pts = np.array([(M @ T_body_tool)[:3, 3] for M in Ms])
        down = np.array([0, 0, 1.0])
        for _ in range(4):
            d = pts @ down
            band = pts[d >= np.percentile(d, 93)]
            c = band.mean(0)
            _, _, vt = np.linalg.svd(band - c)
            n = vt[2]
            if n @ (-c) < 0:      # normal toward the camera: up
                n = -n
            down = -n
        res = np.abs((band - c) @ n) * 1000
        plane = (c, n)
        print(f"table plane from the fingertip lower envelope: {len(band)} pts, residual p90 {np.percentile(res, 90):.1f} mm")

    seed_q = mu.copy()

    def base_from_plane(x3):
        """base pose from (x, y, yaw) on the plane frame: height and tilt from the reference."""
        c, n = plane
        ref = json.load(open(a.plane_ref))
        ex = T0[:3, 0] - (T0[:3, 0] @ n) * n; ex /= np.linalg.norm(ex); ey = np.cross(n, ex)
        Pw = np.eye(4); Pw[:3, 0], Pw[:3, 1], Pw[:3, 2], Pw[:3, 3] = ex, ey, n, c   # plane frame in the camera
        yaw = x3[2]; Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        Tpb = np.eye(4); Tpb[:3, :3] = Rz @ np.array(ref["R_plane_base"]); Tpb[:3, 3] = [x3[0], x3[1], ref["height_m"]]
        return Pw @ Tpb

    if plane is not None and a.calibrate_plane:
        c, n = plane
        ex = T0[:3, 0] - (T0[:3, 0] @ n) * n; ex /= np.linalg.norm(ex); ey = np.cross(n, ex)
        Pw = np.eye(4); Pw[:3, 0], Pw[:3, 1], Pw[:3, 2], Pw[:3, 3] = ex, ey, n, c
        Tpb = np.linalg.inv(Pw) @ T0
        tilt = np.degrees(np.arccos(np.clip(Tpb[2, 2], -1, 1)))
        # strip the yaw: R_plane_base with yaw removed is what another rig reuses
        yaw = np.arctan2(Tpb[1, 0], Tpb[0, 0]); Rz = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])
        json.dump({"height_m": float(Tpb[2, 3]), "R_plane_base": (Rz @ Tpb[:3, :3]).tolist(), "tilt_deg": float(tilt),
                   "note": "robot base relative to the plane of the fingertip lower-envelope table plane, measured on the pilot rig from its hand-eye; "
                           "yaw removed. Another rig: height and tilt reused, x/y/yaw solved."}, open(a.plane_ref, "w"), indent=1)
        print(f"base is {Tpb[2, 3]*1000:.0f} mm above the table plane, tilted {tilt:.1f} deg; wrote {a.plane_ref} (x,y,yaw of the init: {Tpb[0,3]*1000:.0f}, {Tpb[1,3]*1000:.0f} mm, {np.degrees(yaw):.1f} deg)")
        return

    def cost(x, verbose=False):
        T_cam_base = base_from_plane(x) if plane is not None else se3(x) @ T0
        Ainv = np.linalg.inv(T_cam_base)
        res, post, lim, q = [], [], 0, seed_q.copy()
        for M in sel:
            F = Ainv @ M @ Binv
            _, qq = kin.ik(F, "grasp_site", init_q=q.copy(), pos_threshold=1e-3, ori_threshold=1e-2, max_iters=80)
            q = np.array(qq[:6], float)
            Fk = np.asarray(kin.fk(q)).reshape(4, 4)
            res.append(np.linalg.norm(Fk[:3, 3] - F[:3, 3]))
            post.append(np.mean(((q - mu) / sig) ** 2))
            lim += np.sum(np.clip(lo - q, 0, None) ** 2 + np.clip(q - hi, 0, None) ** 2)
        res, post = np.array(res), np.array(post)
        c = res.mean() + a.lam * post.mean() + 10 * lim / len(sel)
        if verbose:
            print(f"   IK residual median {np.median(res)*1000:.1f} mm, >1 cm in {np.mean(res > 0.01):.0%}; "
                  f"posture z^2 mean {post.mean():.2f}; limit penalty {lim:.3f}")
        return c

    x0 = np.zeros(6)
    if plane is not None:
        # start at the init's x, y, yaw in the plane frame
        c, n = plane; ex = T0[:3, 0] - (T0[:3, 0] @ n) * n; ex /= np.linalg.norm(ex); ey = np.cross(n, ex)
        Pw = np.eye(4); Pw[:3, 0], Pw[:3, 1], Pw[:3, 2], Pw[:3, 3] = ex, ey, n, c
        Tpb = np.linalg.inv(Pw) @ T0
        x0 = np.array([Tpb[0, 3], Tpb[1, 3], np.arctan2(Tpb[1, 0], Tpb[0, 0])])
    if a.perturb > 0 and plane is not None:
        x0 = x0 + np.array([rng.normal() * a.perturb, rng.normal() * a.perturb, np.radians(30) * rng.choice([-1, 1])])
        print(f"self-test (plane mode): starting ~{a.perturb*1000:.0f} mm and 30 deg of yaw away")
    elif a.perturb > 0:
        d = rng.normal(size=3); d = d / np.linalg.norm(d) * a.perturb
        rv = rng.normal(size=3); rv = rv / np.linalg.norm(rv) * np.radians(30)
        x0 = np.concatenate([d, rv])
        print(f"self-test: starting {a.perturb*1000:.0f} mm / 30 deg away from the init hand-eye")
    print("start:"); c0 = cost(x0, True)
    r = minimize(cost, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400, "initial_simplex": None})
    r2 = minimize(cost, r.x, method="Powell", options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 2000})
    x = r2.x if r2.fun < r.fun else r.x
    print("end:"); c1 = cost(x, True)
    T_cam_base = base_from_plane(x) if plane is not None else se3(x) @ T0
    dT = np.linalg.inv(T0) @ T_cam_base
    ang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
    print(f"cost {c0:.4f} -> {c1:.4f}; solved base differs from the init hand-eye by "
          f"{np.linalg.norm(dT[:3, 3])*1000:.0f} mm / {ang:.1f} deg")
    out = {"T_cam_base": T_cam_base.tolist(), "T_flange_body": glove_he["T_flange_body"],
           "note": "VIRTUAL base from reachability + posture prior (free_base.py); valid for wrist-observation policies only",
           "source": {"stems": a.stems, "init": a.init, "frames": int(len(sel)), "lam": a.lam, "perturb_m": a.perturb,
                      "cost_start": float(c0), "cost_end": float(c1),
                      "offset_from_init_mm": float(np.linalg.norm(dT[:3, 3]) * 1000), "offset_from_init_deg": float(ang)}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
