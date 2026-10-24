#!/usr/bin/env python3
"""Robot base <-> tile-world link from the HEAD camera alone (no forward cam).

Chain per frame:  T_world_body = inv(T_cam_world[tiles]) . T_cam_body[dodeca+base markers]
FK per frame:     T_base_body  = FK(q_yam) . T_flange_body (calib/handeye.json)
Fit:              T_world_base = argmin sum || o_world - (R o_base + t) ||   (Kabsch, trimmed)

The result replaces inv(forward_cam_world) . handeye.T_cam_base as the base-to-tile link
used by build_dataset.py --base-world, so joint actions from head-cam labels land in the
robot's base frame with the bias measured here instead of the Sept hand-eye's.

Usage:
  python scripts/base_world_fit.py yam_head14_b0587b --track data/raw/yam_head14_so101cam_track.jsonl \
      [--cam-world calib/yam_head14_b0587b_cam_world.json] [--skeleton ~/.cache/huggingface/lerobot/yam_head14] \
      [--gripper-bundle calib/gripper_bundle_yam_head14_b0587b.json] [--out calib/base_world_20261015.json]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "co/i2rt"))
import labels_lib as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("stem", help="head stream stem, e.g. yam_head14_b0587b")
ap.add_argument("--track", required=True, help="so101_cam_track.py jsonl with q_yam ticks")
ap.add_argument("--cam-world", default=None, help="default calib/<stem>_cam_world.json (head_pose_fast.py)")
ap.add_argument("--intr", default="calib/B0587b_f3_intrinsics.json")
ap.add_argument("--bundle", default="calib/dodeca_bundle_ideal.json")
ap.add_argument("--gripper-bundle", default="calib/gripper_bundle.json")
ap.add_argument("--handeye", default="calib/handeye.json", help="T_flange_body (gripper ball on the robot flange)")
ap.add_argument("--skeleton", default=None, help="LeRobot dataset root with meta/episode_wallclock.jsonl: fit on in-episode frames only")
ap.add_argument("--min-tiles", type=int, default=2); ap.add_argument("--max-rms", type=float, default=5.0)
ap.add_argument("--shift", type=float, nargs=3, default=[-0.2, 0.2, 0.0333], metavar=("MIN", "MAX", "STEP"),
                help="s: scan the camera-vs-joints time shift and keep the best (+ = frames lag the joints)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

K, dist = L.load_intrinsics(a.intr)
T_body, dobj = L.load_bundle(a.bundle)
extra = L.load_gripper_bundle(a.gripper_bundle)
he = json.load(open(a.handeye)); B = np.array(he["T_flange_body"])
p_body = np.append(np.array(json.load(open(ROOT / "calib/eef_target.json"))["p_body_mm"]) / 1000.0, 1.0)

ts = json.load(open(ROOT / f"data/raw/{a.stem}_ts.json")); tw = np.array([e["wall"] for e in ts])
det = np.load(ROOT / f"calib/{a.stem}_dodeca_detections.npz", allow_pickle=True)
rows = L.body_frames(det, ts, K, dist, T_body, dobj, {}, extra=extra)
print(f"{a.stem}: {len(rows)} fused body poses of {len(ts)} frames")

cw = json.load(open(a.cam_world or ROOT / f"calib/{a.stem}_cam_world.json"))
cam = {r["k"]: np.array(r["T_cam_world"]) for r in cw["per_frame"]
       if r["tiles"] >= a.min_tiles and r["rms"] <= a.max_rms}
print(f"tile poses: {len(cam)} frames (tiles >= {a.min_tiles}, rms <= {a.max_rms} px) of {cw['n_frames']} solved, stride {cw['stride']}")

# joint ticks
tq, qq = [], []
for line in open(a.track):
    r = json.loads(line)
    if r.get("header") or r.get("q_yam") is None:
        continue
    tq.append(r["t"]); qq.append(r["q_yam"])
tq, qq = np.array(tq), np.array(qq)
order = np.argsort(tq); tq, qq = tq[order], qq[order]
print(f"tracker: {len(tq)} joint ticks, {(tq[-1] - tq[0]) / 60:.1f} min")

from i2rt.robots.kinematics import Kinematics  # noqa: E402
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml  # noqa: E402
kin = Kinematics(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")

win = None
if a.skeleton:
    eps = [json.loads(l) for l in open(Path(a.skeleton).expanduser() / "meta/episode_wallclock.jsonl")]
    win = np.array([[e["start_unix"], e["end_unix"]] for e in eps])


def T_from(rvec, tvec):
    T = np.eye(4); T[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1)); T[:3, 3] = np.asarray(tvec, float).ravel(); return T


def fk_at(t):
    j = np.searchsorted(tq, t)
    if j <= 0 or j >= len(tq) or tq[j] - tq[j - 1] > 0.25:
        return None
    w = (t - tq[j - 1]) / (tq[j] - tq[j - 1]); q = (1 - w) * qq[j - 1] + w * qq[j]
    return kin.fk(q) @ B


# per-frame pairs (world-frame body pose from the head chain, base-frame body pose from FK)
pairs = []   # (t, T_world_body, k)
for r in rows:
    k = r["k"]
    if k not in cam:
        continue
    t = r["t_wall"]
    if win is not None and not ((t >= win[:, 0]) & (t <= win[:, 1])).any():
        continue
    pairs.append((t, np.linalg.inv(cam[k]) @ T_from(r["rvec"], r["tvec"]), k))
print(f"frames with head-chain pose AND tile pose{' in-episode' if win is not None else ''}: {len(pairs)}")
if len(pairs) < 200:
    sys.exit("too few frames to fit")


def kabsch(P, Q):
    """R, t with Q ~ R P + t (least squares)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def fit(shift):
    Tw, Tb = [], []
    for t, T_wb, _ in pairs:
        T_bb = fk_at(t - shift)
        if T_bb is None:
            continue
        Tw.append(T_wb); Tb.append(T_bb)
    Tw, Tb = np.array(Tw), np.array(Tb)
    P, Q = Tb[:, :3, 3], Tw[:, :3, 3]
    keep = np.ones(len(P), bool)
    for _ in range(4):                      # trimmed Kabsch
        R, t = kabsch(P[keep], Q[keep])
        res = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        keep = res <= max(3 * np.median(res[keep]), 0.004)
    R, t = kabsch(P[keep], Q[keep])
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
    res = np.linalg.norm((P @ R.T + t) - Q, axis=1)
    # grasp-point residual + rotation residual with the fitted link
    g_w = np.einsum("nij,j->ni", Tw, p_body)[:, :3]
    g_b = np.einsum("nij,j->ni", Tb, p_body)[:, :3]
    gres = np.linalg.norm((g_b @ R.T + t) - g_w, axis=1)
    rres = []
    for Tw_i, Tb_i in zip(Tw, Tb):
        Rrel = Tw_i[:3, :3].T @ R @ Tb_i[:3, :3]
        rres.append(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    return T, res, gres, np.array(rres), keep


best = None
lo, hi, st = a.shift
for shift in np.arange(lo, hi + 1e-9, st):
    T, res, gres, rres, keep = fit(shift)
    print(f"  shift {shift * 1000:+5.0f} ms: body-origin residual median {np.median(res) * 1000:5.1f} mm  p90 {np.percentile(res, 90) * 1000:5.1f}"
          f"  grasp-point median {np.median(gres) * 1000:5.1f} mm  rotation median {np.median(rres):4.2f} deg  (n={len(res)}, inliers {keep.mean():.2f})")
    if best is None or np.median(res) < best[1]:
        best = (shift, np.median(res), T, res, gres, rres)
shift, _, T, res, gres, rres = best
print(f"\nbest shift {shift * 1000:+.0f} ms")
print(f"T_world_base:\n{np.round(T, 4)}")

# compare with the Sept link through the forward camera
try:
    T_world_fwd = np.linalg.inv(np.array(json.load(open(ROOT / "calib/forward_cam_world_20261012.json"))["T_cam_world"]))
    T_old = T_world_fwd @ np.array(he["T_cam_base"])
    dT = np.linalg.inv(T_old) @ T
    ang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
    print(f"vs forward-cam link (inv(forward_cam_world) . handeye): translation {np.round((T[:3, 3] - T_old[:3, 3]) * 1000, 1)} mm "
          f"(|{np.linalg.norm(T[:3, 3] - T_old[:3, 3]) * 1000:.1f}|), rotation {ang:.2f} deg")
except OSError:
    pass
print(f"residuals with the fit: body origin median {np.median(res) * 1000:.1f} / p90 {np.percentile(res, 90) * 1000:.1f} mm;"
      f" grasp point median {np.median(gres) * 1000:.1f} / p90 {np.percentile(gres, 90) * 1000:.1f} mm; rotation median {np.median(rres):.2f} / p90 {np.percentile(rres, 90):.2f} deg")
out = a.out or ROOT / f"calib/base_world_{a.stem}.json"
json.dump({"T_world_base": T.tolist(), "T_base_world": np.linalg.inv(T).tolist(), "source": a.stem, "track": a.track,
           "cam_world": str(a.cam_world or f"calib/{a.stem}_cam_world.json"), "gripper_bundle": a.gripper_bundle,
           "shift_s": float(shift), "n": int(len(res)), "residual_mm": {"origin_median": float(np.median(res) * 1000), "origin_p90": float(np.percentile(res, 90) * 1000),
                                                                           "grasp_median": float(np.median(gres) * 1000), "grasp_p90": float(np.percentile(gres, 90) * 1000)},
           "rotation_deg": {"median": float(np.median(rres)), "p90": float(np.percentile(rres, 90))},
           "note": "robot base -> tile world from the HEAD cam chain (tiles x dodeca) vs FK; replaces inv(forward_cam_world).handeye.T_cam_base"},
          open(out, "w"), indent=1)
print("wrote", out)
