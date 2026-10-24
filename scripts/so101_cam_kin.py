#!/usr/bin/env python3
"""Kinematic calibration of the SO-101 camera positioner from tile-solved camera poses.

Model (base -> camera):
  T_world_base · Π_{i=1..5} [URDF origin_i · Rot(axis_i, s_i·q_i + off_i)] · T_yaw · Rot(z, s_6·q_6 + off_6) · T_cam
where links 1-5 are the vanilla SO-101 (calib/so101/so101_new_calib.urdf), T_yaw is the unknown
yaw-joint frame in the gripper_link (custom mount: fitted; initialised at the URDF gripper joint),
off_i are the encoder-to-model offsets (LeRobot zeros each joint mid-travel), s_i = ±1 the encoder
sign, and T_cam the unknown camera offset. Everything is fitted by least squares against the
(joint readings, tile-solved camera pose) pairs recorded by cam_arm_poses.py.

  python scripts/so101_cam_kin.py fit calib/so101_cam_poses_*.json          # -> calib/so101_cam_kin.json
  python scripts/so101_cam_kin.py check calib/so101_cam_poses_<stamp>.json  # residuals of a saved fit on new data
"""
import itertools
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "calib/so101/so101_new_calib.urdf"
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
OUT = ROOT / "calib/so101_cam_kin.json"


def T_from(xyz, rpy):
    T = np.eye(4); T[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix(); T[:3, 3] = xyz; return T


def load_urdf():
    root = ET.parse(URDF).getroot(); J = {}
    for j in root.findall("joint"):
        o = j.find("origin"); ax = j.find("axis")
        J[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                T=T_from([float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()]),
                                axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3))
    return J


def rotvec_T(rv, t):
    T = np.eye(4); T[:3, :3] = Rot.from_rotvec(rv).as_matrix(); T[:3, 3] = t; return T


def axis_T(axis, ang):
    T = np.eye(4); T[:3, :3] = Rot.from_rotvec(np.asarray(axis) / np.linalg.norm(axis) * ang).as_matrix(); return T


class Model:
    """Parameter vector: base rv(3) t(3) | off(6) | yaw: p(3) dir angles(2) | cam rv(3) t(3)  -> 23"""
    def __init__(self, J, signs):
        self.J = J; self.signs = np.asarray(signs, float)
        self.chain = [J[n] for n in CHAIN]; self.grip = J["gripper"]

    def unpack(self, x):
        base = rotvec_T(x[0:3], x[3:6]); off = x[6:12]
        p = x[12:15]; th, ph = x[15], x[16]
        d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        # yaw frame: origin p, z axis along d (minimal rotation from z)
        Ry = Rot.align_vectors([d], [[0, 0, 1]])[0].as_matrix()
        Ty = np.eye(4); Ty[:3, :3] = Ry; Ty[:3, 3] = p
        cam = rotvec_T(x[17:20], x[20:23])
        return base, off, Ty, cam

    def fk(self, x, q_enc_deg):
        base, off, Ty, cam = self.unpack(x)
        q = np.radians(np.asarray(q_enc_deg)) * self.signs + off
        T = base.copy()
        for jj, qi in zip(self.chain, q[:5]):
            T = T @ jj["T"] @ axis_T(jj["axis"], qi)
        T = T @ Ty @ axis_T([0, 0, 1], q[5]) @ cam
        return T

    def x0(self, base_guess, cam_guess):
        g = self.grip["T"]; dz = g[:3, :3] @ np.array([0, 0, 1.0])
        th, ph = np.arccos(np.clip(dz[2], -1, 1)), np.arctan2(dz[1], dz[0])
        return np.concatenate([Rot.from_matrix(base_guess[:3, :3]).as_rotvec(), base_guess[:3, 3], np.zeros(6), g[:3, 3], [th, ph],
                               Rot.from_matrix(cam_guess[:3, :3]).as_rotvec(), cam_guess[:3, 3]])


def residuals(x, model, Q, Tm):
    r = []
    for q, T in zip(Q, Tm):
        P = model.fk(x, q); D = np.linalg.inv(T) @ P
        r.append(np.concatenate([D[:3, 3] * 1000.0, np.degrees(Rot.from_matrix(D[:3, :3]).as_rotvec()) * 5.0]))  # mm, 5 mm per deg
    return np.concatenate(r)


def load_samples(paths, min_tiles=2, max_rms=5.0):
    Q, Tm = [], []
    for p in paths:
        for r in json.load(open(p))["samples"]:
            if r["T_cam_world"] is None or r["tiles"] < min_tiles or r["rms"] > max_rms:
                continue
            Q.append([r["joints_deg"][n] for n in CHAIN + ["wrist_yaw"]]); Tm.append(np.linalg.inv(np.array(r["T_cam_world"])))  # camera in world
    return np.array(Q), np.array(Tm)


def fit(paths):
    J = load_urdf(); Q, Tm = load_samples(paths)
    print(f"{len(Q)} samples from {len(paths)} file(s)")
    # base guess: camera positions' centroid minus ~0.3 m, upright; camera guess: 70 mm off the yaw axis
    base_guess = np.eye(4); base_guess[:3, 3] = Tm[:, :3, 3].mean(0) - np.array([0, 0.3, 0.25])
    best = None
    rng = np.random.default_rng(0)
    for signs in itertools.product([1, -1], repeat=6):
        m = Model(J, signs)
        for trial in range(2):
            cg = np.eye(4); cg[:3, :3] = Rot.random(random_state=rng.integers(1 << 30)).as_matrix(); cg[:3, 3] = [0.07, 0, 0]
            x0 = m.x0(base_guess, cg); x0[0:3] = Rot.random(random_state=rng.integers(1 << 30)).as_rotvec() if trial else 0
            try:
                sol = least_squares(residuals, x0, args=(m, Q, Tm), max_nfev=40, loss="soft_l1", f_scale=20.0)
            except Exception:
                continue
            c = float(np.mean(sol.fun ** 2))
            if best is None or c < best[0]:
                best = (c, signs, sol.x)
    c, signs, x = best
    m = Model(J, signs)
    sol = least_squares(residuals, x, args=(m, Q, Tm), max_nfev=2000, loss="soft_l1", f_scale=20.0)
    x = sol.x
    r = residuals(x, m, Q, Tm).reshape(-1, 6)
    tr = np.linalg.norm(r[:, :3], axis=1); ro = np.linalg.norm(r[:, 3:], axis=1) / 5.0
    base, off, Ty, cam = m.unpack(x)
    print(f"signs {signs}; residual vs tiles: translation median {np.median(tr):.1f} mm (p90 {np.percentile(tr, 90):.1f}), rotation median {np.median(ro):.2f} deg (p90 {np.percentile(ro, 90):.2f})")
    print(f"joint offsets (deg): {np.round(np.degrees(off), 1)}")
    print(f"camera offset from the yaw axis: {np.linalg.norm(cam[:3, 3][:2]) * 1000:.0f} mm radial, {cam[2, 3] * 1000:.0f} mm along it (expect ~70 mm radial)")
    print(f"yaw joint origin in gripper_link (mm): {np.round(Ty[:3, 3] * 1000)}  vs URDF gripper joint {np.round(J['gripper']['T'][:3, 3] * 1000)}")
    print(f"SO-101 base in the tile frame (mm): {np.round(base[:3, 3] * 1000)}")
    json.dump({"signs": list(signs), "x": x.tolist(), "chain": CHAIN, "urdf": str(URDF.relative_to(ROOT)), "sources": [str(p) for p in paths],
               "residual_mm_median": float(np.median(tr)), "residual_deg_median": float(np.median(ro)),
               "T_world_base": base.tolist(), "T_grip_yaw": Ty.tolist(), "T_yaw_cam": cam.tolist(), "offsets_deg": np.degrees(off).tolist()},
              open(OUT, "w"), indent=1)
    print("wrote", OUT)


def fit_prior(paths, base_yam_mm=(-305.0, 13.0, 546.0)):
    """Constrained fit: signs +1 (URDF so101_new_calib matches the LeRobot calibration), offsets of joints 1-5
    within +-35 deg, base initialised from the measured position relative to the YAM base (pan zero facing +x of
    the YAM) and kept within 0.25 m / 40 deg of it; yaw axis and camera offset free. Several restarts."""
    J = load_urdf(); Q, Tm = load_samples(paths)
    he = json.load(open(ROOT / "calib/handeye.json")); T_cam_base = np.array(he["T_cam_base"])
    T_cam_world = np.array(json.load(open(ROOT / "calib/forward_cam_world_20261012.json"))["T_cam_world"])
    T_world_yam = np.linalg.inv(np.linalg.inv(T_cam_base) @ T_cam_world)          # YAM base pose in the tile frame
    bp = json.load(open(paths[0])).get("T_world_base_prior")
    if bp is not None:        # pose run on a re-mounted arm: start from the previous model's base, rigidly corrected at that run's home
        prior = np.array(bp); print("base prior from the pose run's home correction (tile frame, mm):", np.round(prior[:3, 3] * 1000))
    else:
        prior = T_world_yam @ rotvec_T([0, 0, 0], np.array(base_yam_mm) / 1000.0)
    m = Model(J, [1, 1, 1, 1, 1, 1])
    rv0, t0 = Rot.from_matrix(prior[:3, :3]).as_rotvec(), prior[:3, 3]
    lo = np.concatenate([rv0 - 0.7, t0 - 0.25, np.full(5, -np.radians(35)), [-np.pi], J["gripper"]["T"][:3, 3] - 0.12, [0, -np.pi], np.full(3, -np.pi), np.full(3, -0.15)])
    hi = np.concatenate([rv0 + 0.7, t0 + 0.25, np.full(5, np.radians(35)), [np.pi], J["gripper"]["T"][:3, 3] + 0.12, [np.pi, np.pi], np.full(3, np.pi), np.full(3, 0.15)])
    rng = np.random.default_rng(1); best = None
    for trial in range(24):
        cg = np.eye(4); cg[:3, :3] = Rot.random(random_state=rng.integers(1 << 30)).as_matrix(); cg[:3, 3] = [0.07, 0, 0]
        x0 = m.x0(prior, cg); x0[11] = rng.uniform(-np.pi, np.pi)
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        try:
            sol = least_squares(residuals, x0, args=(m, Q, Tm), bounds=(lo, hi), max_nfev=300, loss="soft_l1", f_scale=15.0)
        except Exception as e:
            continue
        c = float(np.mean(sol.fun ** 2))
        if best is None or c < best[0]:
            best = (c, sol.x); print(f"  trial {trial}: cost {c:.1f}", flush=True)
    x = least_squares(residuals, best[1], args=(m, Q, Tm), bounds=(lo, hi), max_nfev=3000, loss="soft_l1", f_scale=15.0).x
    r = residuals(x, m, Q, Tm).reshape(-1, 6); tr = np.linalg.norm(r[:, :3], axis=1); ro = np.linalg.norm(r[:, 3:], axis=1) / 5.0
    base, off, Ty, cam = m.unpack(x)
    print(f"residual vs tiles: translation median {np.median(tr):.1f} mm (p90 {np.percentile(tr, 90):.1f}), rotation median {np.median(ro):.2f} deg (p90 {np.percentile(ro, 90):.2f})")
    print(f"joint offsets (deg): {np.round(np.degrees(off), 1)}")
    print(f"camera offset from the yaw axis: {np.linalg.norm(cam[:3, 3][:2]) * 1000:.0f} mm radial, {cam[2, 3] * 1000:.0f} mm along it (expect ~70 radial)")
    pb = (np.linalg.inv(T_world_yam) @ np.append(base[:3, 3], 1))[:3] * 1000
    print(f"SO-101 base in the YAM frame (mm): {np.round(pb)}  (measured {base_yam_mm}); base z axis in YAM frame {np.round(np.linalg.inv(T_world_yam)[:3, :3] @ base[:3, :3] @ [0, 0, 1], 2)}")
    json.dump({"signs": [1] * 6, "x": x.tolist(), "chain": CHAIN, "urdf": str(URDF.relative_to(ROOT)), "sources": [str(p) for p in paths],
               "residual_mm_median": float(np.median(tr)), "residual_deg_median": float(np.median(ro)),
               "T_world_base": base.tolist(), "T_grip_yaw": Ty.tolist(), "T_yaw_cam": cam.tolist(), "offsets_deg": np.degrees(off).tolist()}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


def check(paths):
    d = json.load(open(OUT)); J = load_urdf(); m = Model(J, d["signs"]); x = np.array(d["x"])
    Q, Tm = load_samples(paths); r = residuals(x, m, Q, Tm).reshape(-1, 6)
    tr = np.linalg.norm(r[:, :3], axis=1); ro = np.linalg.norm(r[:, 3:], axis=1) / 5.0
    print(f"{len(Q)} samples: encoder-predicted camera pose vs tiles: translation median {np.median(tr):.1f} mm (p90 {np.percentile(tr, 90):.1f}), rotation median {np.median(ro):.2f} deg")


def cam_pose(q_enc_deg):
    """Camera pose in the tile frame from the six encoder readings (deg), using the saved fit."""
    d = json.load(open(OUT)); m = Model(load_urdf(), d["signs"]); return m.fk(np.array(d["x"]), q_enc_deg)


if __name__ == "__main__":
    cmd, paths = sys.argv[1], sys.argv[2:]
    {"fit": fit, "fit_prior": fit_prior, "check": check}[cmd](paths)
