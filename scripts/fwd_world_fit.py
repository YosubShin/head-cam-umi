#!/usr/bin/env python3
"""Forward (fixed) camera -> tile world, fitted from frames where BOTH the head cam (tiles x dodeca) and the
forward cam see the gripper ball: T_world_fwd = argmin sum || o_world(head chain) - (R o_fwd + t) ||  (trimmed Kabsch).
Writes calib/forward_cam_world_<tag>.json in the forward_cam_world_20261012.json format ({"T_cam_world": world->cam}).
usage: fwd_world_fit.py <head_stem> <fwd_stem> --tag 20261015 [--gripper-bundle ...]"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import labels_lib as L
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("head_stem"); ap.add_argument("fwd_stem"); ap.add_argument("--tag", required=True)
ap.add_argument("--head-intr", default="calib/B0587b_f3_intrinsics.json"); ap.add_argument("--fwd-intr", default="calib/B0587_intrinsics.json")
ap.add_argument("--bundle", default="calib/dodeca_bundle_ideal.json"); ap.add_argument("--gripper-bundle", default="calib/gripper_bundle.json")
ap.add_argument("--cam-world", default=None); ap.add_argument("--min-tiles", type=int, default=2); ap.add_argument("--max-rms", type=float, default=5.0)
ap.add_argument("--tol", type=float, default=0.03, help="s: head/forward frame pairing tolerance")
a = ap.parse_args()
T_body, dobj = L.load_bundle(a.bundle); extra = L.load_gripper_bundle(a.gripper_bundle)
def rows(stem, intr):
    K, d = L.load_intrinsics(intr); ts = json.load(open(ROOT / f"data/raw/{stem}_ts.json"))
    det = np.load(ROOT / f"calib/{stem}_dodeca_detections.npz", allow_pickle=True)
    return L.body_frames(det, ts, K, d, T_body, dobj, {}, extra=extra), ts
def T_from(rv, tv):
    T = np.eye(4); T[:3, :3], _ = cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1)); T[:3, 3] = np.asarray(tv, float).ravel(); return T
rh, ts_h = rows(a.head_stem, a.head_intr); rf, ts_f = rows(a.fwd_stem, a.fwd_intr)
cw = {r["k"]: np.array(r["T_cam_world"]) for r in json.load(open(a.cam_world or ROOT / f"calib/{a.head_stem}_cam_world.json"))["per_frame"]
      if r["tiles"] >= a.min_tiles and r["rms"] <= a.max_rms}
head = [(r["t_wall"], np.linalg.inv(cw[r["k"]]) @ T_from(r["rvec"], r["tvec"])) for r in rh if r["k"] in cw]
th = np.array([t for t, _ in head]); Th = np.array([T for _, T in head])
tf = np.array([r["t_wall"] for r in rf]); Tf = np.array([T_from(r["rvec"], r["tvec"]) for r in rf])
j = np.searchsorted(th, tf); j = np.clip(j, 1, len(th) - 1); near = np.where(np.abs(th[j - 1] - tf) < np.abs(th[j] - tf), j - 1, j)
ok = np.abs(th[near] - tf) <= a.tol
P = Tf[ok][:, :3, 3]; Q = Th[near[ok]][:, :3, 3]; Rf = Tf[ok][:, :3, :3]; Rh = Th[near[ok]][:, :3, :3]
print(f"head-chain poses {len(th)}, forward poses {len(tf)}, paired within {a.tol*1000:.0f} ms: {ok.sum()}")
def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0); U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc)); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, qc - R @ pc
keep = np.ones(len(P), bool)
for _ in range(5):
    R, t = kabsch(P[keep], Q[keep]); res = np.linalg.norm(P @ R.T + t - Q, axis=1); keep = res <= max(3 * np.median(res[keep]), 0.004)
R, t = kabsch(P[keep], Q[keep]); res = np.linalg.norm(P @ R.T + t - Q, axis=1)
rres = [np.degrees(np.arccos(np.clip((np.trace(Rh[i].T @ R @ Rf[i]) - 1) / 2, -1, 1))) for i in range(len(P))]
T_world_fwd = np.eye(4); T_world_fwd[:3, :3], T_world_fwd[:3, 3] = R, t
old = json.load(open(ROOT / "calib/forward_cam_world_20261012.json")); T_old = np.linalg.inv(np.array(old["T_cam_world"]))
dT = np.linalg.inv(T_old) @ T_world_fwd; ang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
print(f"residual (forward chain vs head chain, tile frame): origin median {np.median(res)*1000:.1f} / p90 {np.percentile(res, 90)*1000:.1f} mm; rotation median {np.median(rres):.2f} deg; inliers {keep.mean():.2f}")
print(f"vs forward_cam_world_20261012: translation {np.round((T_world_fwd[:3, 3] - T_old[:3, 3])*1000, 1)} mm, rotation {ang:.2f} deg")
out = ROOT / f"calib/forward_cam_world_{a.tag}.json"
json.dump({"stem": a.fwd_stem, "layout": old.get("layout"), "T_cam_world": np.linalg.inv(T_world_fwd).tolist(), "n_frames": int(len(P)),
           "residual_mm_median": float(np.median(res) * 1000), "residual_mm_p90": float(np.percentile(res, 90) * 1000), "rotation_deg_median": float(np.median(rres)),
           "note": f"fit vs the HEAD-cam chain (tiles x dodeca) on {a.head_stem}: forward body poses mapped into the tile frame; replaces the 2026-10-12 tile solve"},
          open(out, "w"), indent=1); print("wrote", out)
