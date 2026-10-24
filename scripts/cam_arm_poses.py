#!/usr/bin/env python3
"""SO-101 camera positioner, bring-up test 1: repeatability and pose sampling against the tiles.

Mount the arm so its 4K camera sees the four taped workspace tiles. With torque off, place the
arm by hand at a sensible 'home' view and press ENTER. The script then visits N poses around home
(small joint perturbations), each R times in shuffled order, and at every visit reads the joints
and solves the camera pose from the tiles. It prints the repeatability (spread of camera position
and orientation over the repeats of each pose) and saves every (joints, T_cam_world) pair, which
is also the input for the hand-eye solve.

  python scripts/cam_arm_poses.py --port /dev/ttyACM0 --cam /dev/v4l/by-path/pci-0000:10:00.3-usb-0:2.1:1.0-video-index0
  python scripts/cam_arm_poses.py --report calib/so101_cam_poses_<stamp>.json      # stats only
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tile_solver import TileSolver  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "wrist_yaw"]
# perturbation half-ranges (deg) per joint around the hand-placed home; roll stays put (keeps the horizon)
HALF = {"shoulder_pan": 12, "shoulder_lift": 8, "elbow_flex": 8, "wrist_flex": 8, "wrist_roll": 0, "wrist_yaw": 10}


# joint box (deg) for model-sampled poses: the union of what the arm-up runs (130807, 131418) visited, widened ~15 deg,
# roll widened on purpose (it never varied before, so its offset was unobservable)
BOX = {"shoulder_pan": (-20, 70), "shoulder_lift": (-60, 30), "elbow_flex": (-15, 50), "wrist_flex": (10, 100), "wrist_roll": (-30, 20), "wrist_yaw": (-45, 35)}
HEAD_MM = np.array([303.0, 423.0, 750.0])          # glove5 median head position, tile frame


def sample_head_volume(n, seed=0, span_mm=150.0, min_tiles=3, margin=0.1, verbose=True, home=None, T_home_meas=None):
    """Camera poses spanning the operator's head volume, converted to joints with the rough kinematic model.
    Each target: position = head median + U(+-span), looking at a point around the tile centroid, roll +-15 deg.
    Accept when IK (within BOX) lands within 30 mm / 5 deg of the target and the model predicts >= min_tiles tile
    centres inside the central (1-2*margin) of the image. Returns a list of joint dicts.
    home + T_home_meas (joints and tile-solved camera pose at the hand-placed home): the model's base pose is first
    corrected by the rigid transform that makes the model agree with the measurement at home, so the sampler still
    lands the tiles in view after the arm has been re-mounted."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot
    import so101_cam_kin as K
    d = json.load(open(K.OUT)); m = K.Model(K.load_urdf(), d["signs"]); x = np.array(d["x"])
    if home is not None and T_home_meas is not None:
        qh = [home[j] for j in JOINTS]; D = T_home_meas @ np.linalg.inv(m.fk(x, qh))
        base = D @ K.rotvec_T(x[0:3], x[3:6]); x[0:3] = Rot.from_matrix(base[:3, :3]).as_rotvec(); x[3:6] = base[:3, 3]
        if verbose:
            print(f"model base corrected from the home measurement: moved {np.linalg.norm(D[:3, 3]) * 1000:.0f} mm, {np.degrees(np.linalg.norm(Rot.from_matrix(D[:3, :3]).as_rotvec())):.1f} deg")
    solver = TileSolver(); Kc = solver.K
    tiles = np.array([t["xyz_mm"] for t in solver.layout["tiles"].values()]) / 1000.0
    lo = np.array([BOX[j][0] for j in JOINTS]); hi = np.array([BOX[j][1] for j in JOINTS])
    rng = np.random.default_rng(seed); out = []; tried = 0
    q_seed = np.array([22.0, -1.0, 15.0, 39.0, -4.0, -6.0])     # arm-up home of run 131418

    def err(q, T_t):
        D = np.linalg.inv(T_t) @ m.fk(x, q)
        return np.concatenate([D[:3, 3] * 1000, np.degrees(Rot.from_matrix(D[:3, :3]).as_rotvec()) * 5])

    while len(out) < n and tried < 40 * n:
        tried += 1
        if verbose and tried % 100 == 0:
            print(f"  sampling head-volume poses: {len(out)}/{n} accepted after {tried} tries", flush=True)
        p = HEAD_MM / 1000 + rng.uniform(-span_mm, span_mm, 3) / 1000
        aim = tiles.mean(0) + rng.uniform(-0.15, 0.15, 3) * [1, 1, 0]
        z = aim - p; z /= np.linalg.norm(z)                              # camera z looks at the aim point
        up = np.array([0, 0, -1.0]); xax = np.cross(up, z); xax /= np.linalg.norm(xax); yax = np.cross(z, xax)
        R = np.stack([xax, yax, z], 1) @ Rot.from_rotvec([0, 0, np.radians(rng.uniform(-15, 15))]).as_matrix()
        T_t = np.eye(4); T_t[:3, :3] = R; T_t[:3, 3] = p
        best = None
        for s in range(4):
            q0 = q_seed if s == 0 else np.clip(q_seed + rng.uniform(-30, 30, 6), lo + 1, hi - 1)
            sol = least_squares(err, q0, args=(T_t,), bounds=(lo, hi), max_nfev=300)
            if best is None or sol.cost < best.cost:
                best = sol
        e = err(best.x, T_t); tr, ro = np.linalg.norm(e[:3]), np.linalg.norm(e[3:]) / 5
        if tr > 30 or ro > 5:
            continue
        T = m.fk(x, best.x); pc = (np.linalg.inv(T) @ np.c_[tiles, np.ones(4)].T)[:3].T   # tiles in camera frame
        vis = 0
        for c in pc:
            if c[2] > 0.2:
                u, v = Kc[0, 0] * c[0] / c[2] + Kc[0, 2], Kc[1, 1] * c[1] / c[2] + Kc[1, 2]
                if margin * 3840 < u < (1 - margin) * 3840 and margin * 2160 < v < (1 - margin) * 2160:
                    vis += 1
        if vis < min_tiles:
            continue
        out.append({j: float(v) for j, v in zip(JOINTS, best.x)})
    if verbose:
        print(f"sampled {len(out)} poses in {tried} tries (span +-{span_mm:.0f} mm around the head median, >= {min_tiles} tiles predicted)")
    return out


def report(path):
    d = json.load(open(path))
    groups = {}
    for r in d["samples"]:
        if r["T_cam_world"] is None:
            continue
        groups.setdefault(r["pose"], []).append(r)
    pos_sp, rot_sp = [], []
    for k, rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        T = np.array([np.linalg.inv(np.array(r["T_cam_world"])) for r in rs])   # camera pose in world
        p = T[:, :3, 3] * 1000
        pos_sp.append(np.linalg.norm(p - p.mean(0), axis=1).mean())
        R = T[:, :3, :3]; Rm = R[0]
        ang = [np.degrees(np.arccos(np.clip((np.trace(Rm.T @ Ri) - 1) / 2, -1, 1))) for Ri in R[1:]]
        rot_sp.append(np.mean(ang))
    n_ok = sum(1 for r in d["samples"] if r["T_cam_world"] is not None)
    print(f"{path}: {len(d['samples'])} visits, {n_ok} solved, {len(groups)} poses")
    if pos_sp:
        print(f"  repeatability (mean deviation from the pose's mean, over repeats): position median {np.median(pos_sp):.1f} mm, p90 {np.percentile(pos_sp, 90):.1f} mm;"
              f" orientation median {np.median(rot_sp):.2f} deg, p90 {np.percentile(rot_sp, 90):.2f} deg")
    else:
        print("  (single visit per pose: no repeatability estimate)")
    rms = [r["rms"] for r in d["samples"] if r["T_cam_world"] is not None]
    print(f"  tile reprojection RMS median {np.median(rms):.2f} px; tiles per solve median {np.median([r['tiles'] for r in d['samples'] if r['T_cam_world'] is not None]):.0f}")
    # coverage of the human head volume (glove5 median +- 10 cm, tile frame) by the sampled camera positions
    P = np.array([np.linalg.inv(np.array(r["T_cam_world"]))[:3, 3] * 1000 for r in d["samples"] if r["T_cam_world"] is not None and r["tiles"] >= 2])  # 1-tile solves are ambiguous
    head = np.array([303.0, 423.0, 750.0])
    lo, hi = P.min(0), P.max(0)
    print(f"  camera positions sampled (tile frame, mm, solves with >= 2 tiles, n={len(P)}): min {np.round(lo)}  max {np.round(hi)}")
    print(f"  human head median (303, 423, 750) +- 100 mm: " + ("inside" if np.all(lo <= head - 100) and np.all(hi >= head + 100) else "NOT fully inside") + " the sampled box; "
          f"nearest sample to the head median {np.min(np.linalg.norm(P - head, axis=1)):.0f} mm away")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="/dev/ttyACM0"); ap.add_argument("--id", default="so101_cam")
    ap.add_argument("--cam", default="/dev/v4l/by-path/pci-0000:10:00.3-usb-0:2.1:1.0-video-index0")
    ap.add_argument("--poses", type=int, default=20); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--move_s", type=float, default=1.5, help="interpolation time per move")
    ap.add_argument("--settle_s", type=float, default=0.6)
    ap.add_argument("--frames", type=int, default=3, help="frames solved per visit (median pose)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the perturbation half-ranges (3 = map the reachable envelope)")
    ap.add_argument("--head-volume", action="store_true", help="sample poses across the operator head volume with the kinematic model (needs calib/so101_cam_kin.json) instead of perturbing home; visited in nearest-neighbour order")
    ap.add_argument("--span-mm", type=float, default=150.0, help="--head-volume: half-range of camera positions around the head median")
    ap.add_argument("--report", default=None, help="only print stats for an existing JSON")
    ap.add_argument("--record-rest", action="store_true", help="record the arm's rest (parked) pose with torque off, then exit")
    a = ap.parse_args()
    if a.report:
        report(a.report); return

    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    cfg = SOFollowerRobotConfig(port=a.port, id=a.id, arm_profile="so-arm-cam-6dof", use_degrees=True,
                                max_relative_target=20.0, disable_torque_on_disconnect=False)   # never drop the arm on exit
    REST = ROOT / "calib/so101_cam_rest.json"
    robot = SOFollower(cfg); robot.connect(calibrate=False)

    # camera: opened streaming first, exposure ritual applied on the live stream (controls only latch while frames flow),
    # auto-exposure fallback on black frames — the same ArmCam the tracker uses
    import so101_cam_park as P
    cam = P.ArmCam(a.cam); cap = cam.cap; solver = cam.solver

    def grab_solve():
        res = []
        for _ in range(a.frames):
            for _ in range(2):
                cap.grab()                      # flush stale frames
            ok, im = cap.read()
            if not ok:
                continue
            r = solver.solve(im)
            if r is not None:
                res.append(r)
        if not res:
            return None
        res.sort(key=lambda r: r[1])
        return res[len(res) // 2]               # median-rms solution

    def move_to(target, seconds):
        q0 = joints(); n = max(2, int(seconds * 30))
        for i in range(1, n + 1):
            w = i / n
            robot.send_action({f"{j}.pos": q0[j] + w * (target[j] - q0[j]) for j in JOINTS})
            time.sleep(seconds / n)

    def joints():
        o = robot.get_observation(); return {j: float(o[f"{j}.pos"]) for j in JOINTS}

    if a.record_rest:
        robot.bus.disable_torque()
        input("Torque is OFF. Fold the arm into its parked/rest position (resting on itself, not on the camera), then ENTER…")
        json.dump(joints(), open(REST, "w")); print("wrote", REST); robot.bus.enable_torque(); robot.disconnect(); return
    if not REST.exists():
        print(f"[no rest pose recorded ({REST.name}); the arm will be left holding at home with torque ON at the end — run --record-rest once]")
    import select
    robot.bus.disable_torque()
    print("Torque is OFF. Place the arm by hand so the camera sees all four tiles (the 'home' view); the line below shows what it sees. ENTER when it reads 4.")
    while True:
        im = cam.frame(); r = solver.solve(im) if im is not None else None
        n = r[2] if r else 0; bright = float(im.mean()) if im is not None else float("nan")
        print(f"\r  tiles seen: {n}   frame brightness {bright:.0f}/255   {'OK' if n >= 3 else ''}      ", end="", flush=True)
        if select.select([sys.stdin], [], [], 0.2)[0]:
            sys.stdin.readline(); break
    print()
    home = joints(); print("home (deg):", {k: round(v, 1) for k, v in home.items()})
    robot.bus.enable_torque()
    r0 = grab_solve()
    if r0 is None:
        sys.exit("no tiles solved at home — adjust the view")
    print(f"home solve: {r0[2]} tiles, {r0[1]:.2f} px RMS; camera at {np.round(np.linalg.inv(r0[0])[:3, 3] * 1000)} mm in the tile frame")

    rng = random.Random(a.seed)
    lim = {m: (robot.calibration[m].range_min, robot.calibration[m].range_max) for m in JOINTS}
    base_prior = None
    try:                      # where the model's base sits after a rigid correction at home: the fit's starting point on a re-mounted arm
        import so101_cam_kin as K
        from scipy.spatial.transform import Rotation as Rot
        d = json.load(open(K.OUT)); m = K.Model(K.load_urdf(), d["signs"]); x = np.array(d["x"])
        D = np.linalg.inv(r0[0]) @ np.linalg.inv(m.fk(x, [home[j] for j in JOINTS])); base_prior = (D @ K.rotvec_T(x[0:3], x[3:6])).tolist()
    except Exception as e:
        print(f"[no base prior from the current model: {e}]")
    if a.head_volume:
        poses = sample_head_volume(a.poses, a.seed, span_mm=a.span_mm, min_tiles=3, home=home, T_home_meas=np.linalg.inv(r0[0]))
        # visit in a nearest-neighbour order from home so consecutive moves stay short
        rem = list(range(len(poses))); cur = np.array([home[j] for j in JOINTS]); ordered = []
        while rem:
            k = min(rem, key=lambda i: np.linalg.norm(np.array([poses[i][j] for j in JOINTS]) - cur)); ordered.append(poses[k]); rem.remove(k); cur = np.array([poses[k][j] for j in JOINTS])
        poses = ordered
    else:
        poses = [{j: home[j] + rng.uniform(-HALF[j] * a.scale, HALF[j] * a.scale) for j in JOINTS} for _ in range(a.poses)]
    order = [(k, r) for r in range(a.reps) for k in range(len(poses))]
    if not a.head_volume:
        rng.shuffle(order)
    stamp = time.strftime("%Y%m%d_%H%M%S"); out = ROOT / f"calib/so101_cam_poses_{stamp}.json"
    samples = []
    try:
        for n, (k, rep) in enumerate(order):
            move_to(poses[k], a.move_s); time.sleep(a.settle_s)
            q = joints(); r = grab_solve()
            samples.append({"pose": k, "rep": rep, "target_deg": poses[k], "joints_deg": q,
                            "T_cam_world": r[0].tolist() if r else None, "rms": r[1] if r else None, "tiles": r[2] if r else None,
                            "t": time.time()})
            print(f"[{n + 1}/{len(order)}] pose {k} rep {rep}: " + (f"{r[2]} tiles, {r[1]:.2f} px" if r else "no solve"), flush=True)
            if (n + 1) % 10 == 0:
                json.dump({"home_deg": home, "T_world_base_prior": base_prior, "samples": samples}, open(out, "w"))
    finally:
        json.dump({"home_deg": home, "T_world_base_prior": base_prior, "samples": samples}, open(out, "w"))
        move_to(home, 2.0)
        if REST.exists():
            print("parking…"); move_to(json.load(open(REST)), 4.0); time.sleep(0.5); robot.bus.disable_torque()
        else:
            print("holding at home with torque ON (no rest pose recorded)")
        robot.disconnect(); cap.release()
    print("wrote", out); report(out)


if __name__ == "__main__":
    main()
