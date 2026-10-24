#!/usr/bin/env python3
"""Hand-eye calibration: recover T_cam<-base and T_flange<-body from any
pilot session (camera-frame dodeca poses paired with joint states).

Per frame: T_cam<-body = T_cam<-base . T_base<-flange(q) . T_flange<-body
(AXB problem; A and B constant). Solved with cv2.calibrateRobotWorldHandEye,
output assignment disambiguated EMPIRICALLY by reconstruction residual
(OpenCV's frame-naming conventions are treacherous; residuals are not).

Stages (different envs own different deps):
  export  (lerobot env)  pair vision poses with skeleton joints -> npz
  fk      (i2rt venv)    MuJoCo FK per joint vector -> appends to npz
  solve   (lerobot env)  solve + validate -> calib/handeye.json

Usage:
  python scripts/handeye.py export <take_stem> <skeleton_repo_id>
  ~/co/i2rt/.venv/bin/python scripts/handeye.py fk
  python scripts/handeye.py solve
"""
import json
import sys

import numpy as np

NPZ = "calib/handeye_pairs.npz"


def stage_export(stem, skel_repo, intrinsics="calib/B0587_intrinsics.json"):
    sys.path.insert(0, "scripts")
    import labels_lib as L
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    K, dist = L.load_intrinsics(intrinsics)
    T_body, dobj = L.load_bundle("calib/dodeca_bundle_ideal.json")
    det = np.load(f"calib/{stem}_dodeca_detections.npz")
    ts = json.load(open(f"data/raw/{stem}_ts.json"))
    rows = L.body_frames(det, ts, K, dist, T_body, dobj, {})  # faces only:
    # base markers excluded — keep the vision side at max purity here

    import os
    walls_p = os.path.expanduser(
        f"~/.cache/huggingface/lerobot/{skel_repo}/meta/episode_frames_wallclock.jsonl")
    walls = [json.loads(l) for l in open(walls_p)]
    sk = LeRobotDataset(skel_repo)
    jn = [n for n in sk.meta.features["observation.state"]["names"]
          if n.startswith("joint")]
    ji = [sk.meta.features["observation.state"]["names"].index(n) for n in jn]
    jt, jq = [], []
    for e in walls:
        ep = e["episode_index"]
        f0 = sk.meta.episodes[ep]["dataset_from_index"]
        f1 = sk.meta.episodes[ep]["dataset_to_index"]
        for k, tw in enumerate(e["t_wall"]):
            j = min(f0 + k, f1 - 1)
            s = sk[j]["observation.state"].numpy()
            jt.append(tw)
            jq.append(s[ji])
    jt, jq = np.array(jt), np.array(jq)
    o = np.argsort(jt)
    jt, jq = jt[o], jq[o]

    import cv2
    Ms, Qs, Tw = [], [], []
    for r in rows:
        i = np.searchsorted(jt, r["t_wall"])
        i = min(max(i, 0), len(jt) - 1)
        if i > 0 and abs(jt[i - 1] - r["t_wall"]) < abs(jt[i] - r["t_wall"]):
            i -= 1
        if abs(jt[i] - r["t_wall"]) > 0.025:
            continue
        R, _ = cv2.Rodrigues(r["rvec"])
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, r["tvec"].ravel()
        Ms.append(M)
        Qs.append(jq[i])
        Tw.append(r["t_wall"])
    # diversity subsample: stride to ~600 pairs
    n = len(Ms)
    sel = np.linspace(0, n - 1, min(600, n)).astype(int)
    np.savez(NPZ, M=np.array(Ms)[sel], q=np.array(Qs)[sel],
             t=np.array(Tw)[sel])
    print(f"exported {len(sel)} pairs (of {n} matched) -> {NPZ}")


def stage_fk():
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import ArmType, GripperType, \
        combine_arm_and_gripper_xml
    z = dict(np.load(NPZ))
    kin = Kinematics(combine_arm_and_gripper_xml(
        ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")
    z["F"] = np.array([kin.fk(q) for q in z["q"]])
    np.savez(NPZ, **z)
    p = z["F"][:, :3, 3]
    print(f"FK done: {len(z['F'])} poses, workspace span "
          f"{np.round(p.max(0) - p.min(0), 3)} m")


def stage_solve():
    import cv2
    z = np.load(NPZ)
    M, F = z["M"], z["F"]
    Rw2c = [m[:3, :3] for m in M]
    tw2c = [m[:3, 3] for m in M]
    # try both directions for the middle input; pick by residual
    best = None
    for mid_inv in (False, True):
        Ks = [np.linalg.inv(f) if mid_inv else f for f in F]
        try:
            R1, t1, R2, t2 = cv2.calibrateRobotWorldHandEye(
                Rw2c, tw2c, [k[:3, :3] for k in Ks], [k[:3, 3] for k in Ks])
        except cv2.error as e:
            print(f"(mid_inv={mid_inv} failed: {e})")
            continue
        X1 = np.eye(4); X1[:3, :3], X1[:3, 3] = R1, t1.ravel()
        X2 = np.eye(4); X2[:3, :3], X2[:3, 3] = R2, t2.ravel()
        # candidate assignments: M ?= A . F' . B with A,B from {X1,X2}+invs
        for A, B, name in [
            (X2, X1, "A=X2,B=X1"),
            (np.linalg.inv(X2), X1, "A=X2^-1,B=X1"),
            (X2, np.linalg.inv(X1), "A=X2,B=X1^-1"),
            (np.linalg.inv(X2), np.linalg.inv(X1), "A=X2^-1,B=X1^-1"),
            (X1, X2, "A=X1,B=X2"),
            (np.linalg.inv(X1), X2, "A=X1^-1,B=X2"),
            (X1, np.linalg.inv(X2), "A=X1,B=X2^-1"),
            (np.linalg.inv(X1), np.linalg.inv(X2), "A=X1^-1,B=X2^-1"),
        ]:
            Fm = [np.linalg.inv(f) if mid_inv else f for f in F]
            pred = np.array([A @ f @ B for f in Fm])
            dp = np.linalg.norm(pred[:, :3, 3] - M[:, :3, 3], axis=1)
            tr = np.einsum("nii->n",
                           np.transpose(pred[:, :3, :3], (0, 2, 1))
                           @ M[:, :3, :3])
            ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
            score = np.median(dp)
            if best is None or score < best[0]:
                best = (score, np.median(ang), A, B, name, mid_inv, dp, ang)
    score, angm, A, B, name, mid_inv, dp, ang = best
    print(f"best assignment: {name} mid_inv={mid_inv}")
    print(f"reconstruction residual: pos median {1000*score:.1f}mm "
          f"p95 {1000*np.percentile(dp,95):.1f}mm | rot median {angm:.2f}deg "
          f"p95 {np.percentile(ang,95):.2f}deg")
    out = {"T_cam_base": A.tolist(), "T_flange_body": B.tolist(),
           "middle_inverted": bool(mid_inv), "assignment": name,
           "pos_median_mm": 1000 * score, "rot_median_deg": angm,
           "n_pairs": len(dp),
           "note": "T_cam<-body = T_cam_base . FK(q)[grasp_site] . "
                   "T_flange_body (middle inverted if flagged)"}
    import os as _o
    dst = _o.environ.get("HANDEYE_OUT", "calib/handeye.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "export":
        stage_export(sys.argv[2], sys.argv[3])
    elif stage == "fk":
        stage_fk()
    elif stage == "solve":
        stage_solve()
