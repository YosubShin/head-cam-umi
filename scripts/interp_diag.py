#!/usr/bin/env python3
"""Interpolation-vs-ground-truth diagnostic + twin grip-fusion coverage.

Quantifies how wrong the fiducial labels are where the forward cam was
blind and the build interpolated through the gap, against robot ground
truth (FK of measured joints via hand-eye for EEF; measured actuator for
gripper) — overall and inside the grasp region (commanded gripper below
threshold). Also grades the twin (b0587b) grip-fusion alternative on the
same gap frames, and prints per-episode post-fusion coverage (the rescue
table for the min-coverage gate).

Stages (same split as bridge_roundtrip: lerobot env has the dataset,
i2rt venv has FK):
  export: ~/miniconda3/envs/lerobot_alohamini/bin/python \
      scripts/interp_diag.py export --skeleton ~/.cache/huggingface/lerobot/yam_pilot2 \
      --forward yam_pilot2_b0587 --side yam_pilot2_b0587b --eps 30-56
  fk:     ~/co/i2rt/.venv/bin/python scripts/interp_diag.py fk
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

NPZ = "calib/interp_diag.npz"
LABEL_TOL = 0.12


def quat_from_rvec(rv):
    import cv2
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.array([1.0, 0, 0, 0])
    ax = rv.ravel() / th
    return np.concatenate([[np.cos(th / 2)], ax * np.sin(th / 2)])


def interp_pose(tq, tP, tvecs, quats):
    q = np.array([np.interp(tq, tP, quats[:, i]) for i in range(4)]).T
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    t = np.array([np.interp(tq, tP, tvecs[:, i]) for i in range(3)]).T
    return t, q


def rot_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def series_from_rows(rows):
    tP = np.array([r["t_wall"] for r in rows])
    tv = np.array([r["tvec"].ravel() for r in rows])
    qs = np.array([quat_from_rvec(r["rvec"]) for r in rows])
    for i in range(1, len(qs)):
        if qs[i] @ qs[i - 1] < 0:
            qs[i] = -qs[i]
    return tP, tv, qs


def stage_export(argv):
    sys.path.insert(0, str(Path(__file__).parent))
    import labels_lib as L
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--forward", required=True)
    ap.add_argument("--side", help="twin stem for grip fusion (optional)")
    ap.add_argument("--eps", required=True, help="e.g. 30-56 or 30,31,40")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--det-dir", default="calib")
    ap.add_argument("--intrinsics", default="calib/B0587_intrinsics.json")
    ap.add_argument("--side-intrinsics", default="calib/B0587b_intrinsics.json")
    ap.add_argument("--bundle", default="calib/dodeca_bundle_ideal.json")
    ap.add_argument("--marker-map", help="marker-epoch override json "
                                         "(pilot3+: calib/marker_map_v2.json)")
    ap.add_argument("--fuse-side-pose", metavar="TWIN_HANDEYE_JSON",
                    help="fill forward POSE gaps with twin-camera body "
                         "poses mapped through the two hand-eyes")
    a = ap.parse_args(argv)
    if a.marker_map:
        mm = json.load(open(a.marker_map))
        L.TAILS = {int(k): v for k, v in mm["tails"].items()}
        L.BASES = {int(k): v for k, v in mm["bases"].items()}
        L.TIPS = {int(k): v for k, v in mm["tips"].items()}
        print(f"== marker map {a.marker_map} "
              f"(tails {sorted(L.TAILS)}, bases {sorted(L.BASES)})")
    eps = ([int(x) for x in a.eps.split(",")] if "," in a.eps or "-" not in a.eps
           else list(range(int(a.eps.split("-")[0]), int(a.eps.split("-")[1]) + 1)))

    # skeleton: walls + measured joints/gripper + commanded gripper
    root = Path(a.skeleton)
    walls = {}
    for line in open(root / "meta" / "episode_frames_wallclock.jsonl"):
        e = json.loads(line)
        walls[e["episode_index"]] = np.array(e["t_wall"])
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    skel = LeRobotDataset("interp_diag_skel", root=root)
    snames = skel.meta.features["observation.state"]["names"]
    anames = skel.meta.features["action"]["names"]
    ji = [i for i, n in enumerate(snames) if n.startswith("joint")]
    gi = next(i for i, n in enumerate(snames) if "gripper" in n)
    ga = next(i for i, n in enumerate(anames) if "gripper" in n)
    out = {"eps": np.array(eps), "label_tol": LABEL_TOL}
    mt_all, mv_all = [], []
    for ep in sorted(walls):
        f0 = skel.meta.episodes[ep]["dataset_from_index"]
        f1 = skel.meta.episodes[ep]["dataset_to_index"]
        s = np.stack([skel[i]["observation.state"].numpy() for i in range(f0, f1)])
        m = min(len(s), len(walls[ep]))
        mt_all.append(walls[ep][:m])
        mv_all.append(s[:m, gi])
        if ep in eps:
            aa = np.stack([skel[i]["action"].numpy() for i in range(f0, f1)])
            out[f"w_{ep}"] = walls[ep][:m]
            out[f"q_{ep}"] = s[:m][:, ji]
            out[f"gm_{ep}"] = s[:m, gi]
            out[f"gc_{ep}"] = aa[:m, ga]
    mt = np.concatenate(mt_all)
    mv = np.concatenate(mv_all)
    o = np.argsort(mt)
    mt, mv = mt[o], mv[o]

    # forward-cam label stack (mirrors build_dataset pipeline B)
    K, dist = L.load_intrinsics(a.intrinsics)
    T_body, dobj = L.load_bundle(a.bundle)
    extra = L.load_gripper_bundle()

    def stem_stack(stem, Ki, di, tag):
        det = np.load(f"{a.det_dir}/{stem}_dodeca_detections.npz")
        ts = json.load(open(f"{a.raw_dir}/{stem}_ts.json"))
        rows = L.body_frames(det, ts, Ki, di, T_body, dobj,
                             {**L.TAILS, **L.BASES}, extra=extra)
        print(f"== {tag} {stem}: fused pose in {len(rows)}/{len(ts)} frames")
        ests = L.fit_estimators(rows, mt, mv, verbose=False)
        gt, gv = [], []
        for r in rows:
            g, _ = L.predict_aperture(r["pb"], ests)
            if g is not None:
                gt.append(r["t_wall"])
                gv.append(g)
        print(f"   {len(ests)} GOOD estimators, grip on {len(gt)} frames")
        return rows, np.array(gt), np.array(gv)

    rows_f, gt_f, gv_f = stem_stack(a.forward, K, dist, "forward")
    tP, tv, qs = series_from_rows(rows_f)

    if a.side:
        K2, d2 = L.load_intrinsics(a.side_intrinsics)
        rows_s, gt_s, gv_s = stem_stack(a.side, K2, d2, "side")
        out.update(gt_s=gt_s, gv_s=gv_s)
    else:
        rows_s, gt_s = [], np.array([])

    if a.fuse_side_pose and rows_s:
        # twin POSE fusion: twin-frame body poses -> forward frame via the
        # two hand-eyes, filling forward pose gaps (cable occlusions etc.)
        heF = json.load(open(__import__("os").environ.get("HANDEYE", "calib/handeye.json")))
        heT = json.load(open(a.fuse_side_pose))
        X = np.array(heF["T_cam_base"]) @ np.linalg.inv(
            np.array(heT["T_cam_base"]))   # fwd<-twin
        import cv2 as _cv
        add_t, add_tv, add_q = [], [], []
        for r in rows_s:
            t = r["t_wall"]
            if len(tP) and np.abs(tP - t).min() <= LABEL_TOL:
                continue
            Rt, _ = _cv.Rodrigues(r["rvec"])
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = Rt, r["tvec"].ravel()
            Mf = X @ M
            rv2, _ = _cv.Rodrigues(Mf[:3, :3])
            add_t.append(t)
            add_tv.append(Mf[:3, 3])
            add_q.append(quat_from_rvec(rv2))
        if add_t:
            o = np.argsort(np.concatenate([tP, add_t]))
            tP = np.concatenate([tP, add_t])[o]
            tv = np.concatenate([tv, np.array(add_tv)])[o]
            qs = np.concatenate([qs, np.array(add_q)])[o]
            for i in range(1, len(qs)):
                if qs[i] @ qs[i - 1] < 0:
                    qs[i] = -qs[i]
            print(f"== side POSE fusion: +{len(add_t)} gap samples "
                  f"-> {len(tP)} total")
    out.update(tP=tP, tv=tv, qs=qs, gt_f=gt_f, gv_f=gv_f)

    # rescue table: per-episode coverage before/after twin grip fusion
    print(f"\n== per-episode coverage (label_tol {LABEL_TOL}s, gate 70%):")
    print(f"{'ep':>4} {'pose':>6} {'grip':>6} {'grip+twin':>9}  verdict")
    for ep in eps:
        w = out[f"w_{ep}"]
        near = lambda s: (np.abs(w[:, None] - s[None, :]).min(1)
                          if len(s) else np.full(len(w), np.inf))
        cp = float((near(tP) <= LABEL_TOL).mean())
        cg = float((near(gt_f) <= LABEL_TOL).mean())
        gfu = np.concatenate([gt_f, gt_s]) if len(gt_s) else gt_f
        cgf = float((near(gfu) <= LABEL_TOL).mean())
        was = "in" if (cp >= .7 and cg >= .7) else "OUT"
        now = "in" if (cp >= .7 and cgf >= .7) else "OUT"
        verdict = {("OUT", "in"): "RESCUED", ("OUT", "OUT"): "still out",
                   ("in", "in"): "", ("in", "OUT"): "?"}[(was, now)]
        print(f"{ep:>4} {cp:>6.1%} {cg:>6.1%} {cgf:>9.1%}  {verdict}")
    np.savez(NPZ, **out)
    print(f"\nwrote {NPZ} — now run the fk stage in the i2rt venv")


def stage_fk(argv):
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp-frac", type=float, default=0.35,
                    help="grasp region = commanded gripper <= min + frac*stroke")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path.home() / "co/i2rt"))
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import (ArmType, GripperType,
                                   combine_arm_and_gripper_xml)
    he = json.load(open(__import__("os").environ.get("HANDEYE", "calib/handeye.json")))
    A = np.array(he["T_cam_base"])
    B = np.array(he["T_flange_body"])
    kin = Kinematics(combine_arm_and_gripper_xml(
        ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")
    try:
        p_tip = (np.array(json.load(open("calib/eef_target.json"))["p_body_mm"])
                 .ravel() / 1000.0)
    except Exception:
        p_tip = np.zeros(3)

    z = np.load(NPZ)
    tol = float(z["label_tol"])
    tP, tv, qs = z["tP"], z["tv"], z["qs"]
    gt_f, gv_f = z["gt_f"], z["gv_f"]
    has_twin = "gt_s" in z
    if has_twin:
        gt_s, gv_s = z["gt_s"], z["gv_s"]
        # fusion = forward primary; twin samples only in forward gaps
        far = np.abs(gt_s[:, None] - gt_f[None, :]).min(1) > tol \
            if len(gt_f) else np.ones(len(gt_s), bool)
        gt_u = np.concatenate([gt_f, gt_s[far]])
        gv_u = np.concatenate([gv_f, gv_s[far]])
        o = np.argsort(gt_u)
        gt_u, gv_u = gt_u[o], gv_u[o]

    gm_all = np.concatenate([z[f"gm_{ep}"] for ep in z["eps"]])
    stroke = float(np.percentile(gm_all, 99) - np.percentile(gm_all, 1))
    gc_all = np.concatenate([z[f"gc_{ep}"] for ep in z["eps"]])
    thresh = gc_all.min() + a.grasp_frac * (gc_all.max() - gc_all.min())
    print(f"gripper stroke (p1..p99 measured): {stroke:.1f} units; "
          f"grasp = cmd <= {thresh:.1f} "
          f"(min {gc_all.min():.1f}, max {gc_all.max():.1f})")

    rec = {k: [] for k in ["ep", "t", "interp_p", "interp_g", "twin_fill",
                           "grasp", "tip_mm", "ball_mm", "rot_deg",
                           "gerr", "gerr_fused"]}
    for ep in z["eps"]:
        w, q6 = z[f"w_{ep}"], z[f"q_{ep}"]
        gm, gc = z[f"gm_{ep}"], z[f"gc_{ep}"]
        tvi, qi = interp_pose(w, tP, tv, qs)
        gfi = np.interp(w, gt_f, gv_f)
        gui = np.interp(w, gt_u, gv_u) if has_twin else gfi
        near_p = np.abs(w[:, None] - tP[None, :]).min(1)
        near_g = np.abs(w[:, None] - gt_f[None, :]).min(1)
        near_u = (np.abs(w[:, None] - gt_u[None, :]).min(1)
                  if has_twin else near_g)
        for j, t in enumerate(w):
            M = A @ kin.fk(q6[j].astype(float)) @ B     # GT body pose in cam
            Rg, tg = M[:3, :3], M[:3, 3]
            Rf = rot_from_quat(qi[j])
            tf = tvi[j]
            rec["ep"].append(int(ep))
            rec["t"].append(t)
            rec["interp_p"].append(near_p[j] > tol)
            rec["interp_g"].append(near_g[j] > tol)
            rec["twin_fill"].append(bool(near_g[j] > tol and near_u[j] <= tol))
            rec["grasp"].append(gc[j] <= thresh)
            rec["ball_mm"].append(np.linalg.norm(tf - tg) * 1000)
            rec["tip_mm"].append(
                np.linalg.norm((Rf @ p_tip + tf) - (Rg @ p_tip + tg)) * 1000)
            cosang = (np.trace(Rf.T @ Rg) - 1) / 2
            rec["rot_deg"].append(
                np.degrees(np.arccos(np.clip(cosang, -1, 1))))
            rec["gerr"].append(abs(gfi[j] - gm[j]))
            rec["gerr_fused"].append(abs(gui[j] - gm[j]))
    r = {k: np.array(v) for k, v in rec.items()}
    np.savez("calib/interp_diag_frames.npz", **r)

    def line(name, mask, pose=True):
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:<38} n=0")
            return
        if pose:
            print(f"  {name:<38} n={n:<6} tip median {np.median(r['tip_mm'][mask]):5.1f}mm  "
                  f"p95 {np.percentile(r['tip_mm'][mask], 95):5.1f}mm  "
                  f"rot median {np.median(r['rot_deg'][mask]):4.1f}deg")
        else:
            ge, gu = r["gerr"][mask], r["gerr_fused"][mask]
            print(f"  {name:<38} n={n:<6} grip median {np.median(ge):5.1f} "
                  f"({100*np.median(ge)/stroke:4.1f}% stroke)  p95 {np.percentile(ge,95):5.1f}"
                  f"  | fused median {np.median(gu):5.1f} ({100*np.median(gu)/stroke:4.1f}%)"
                  f"  p95 {np.percentile(gu,95):5.1f}")

    g = r["grasp"]
    for region, msk in [("ALL FRAMES", np.ones(len(g), bool)), ("GRASP REGION", g)]:
        print(f"\n== {region} ==")
        print(" EEF pose error vs FK ground truth (at the between-tips point):")
        line("pose: detected frames", msk & ~r["interp_p"])
        line("pose: interpolated frames", msk & r["interp_p"])
        print(" gripper error vs measured actuator (units; fused = twin fills gaps):")
        line("grip: detected frames", msk & ~r["interp_g"], pose=False)
        line("grip: gap frames (interp | twin-fill)", msk & r["interp_g"], pose=False)
        line("grip:   gap subset the twin measures", msk & r["twin_fill"], pose=False)
    print("\nper-episode gap grip error (median, units | % stroke, interp -> fused):")
    for ep in np.unique(r["ep"]):
        m = (r["ep"] == ep) & r["interp_g"]
        if m.sum() == 0:
            continue
        a1, a2 = np.median(r["gerr"][m]), np.median(r["gerr_fused"][m])
        print(f"  ep{ep}: n={int(m.sum()):<4} {a1:5.1f} ({100*a1/stroke:4.1f}%) -> "
              f"{a2:5.1f} ({100*a2/stroke:4.1f}%)  twin fills "
              f"{int((r['twin_fill'] & (r['ep']==ep)).sum())}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("export", "fk"):
        raise SystemExit(__doc__)
    (stage_export if sys.argv[1] == "export" else stage_fk)(sys.argv[2:])
