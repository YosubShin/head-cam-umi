#!/usr/bin/env python3
"""Drive the SO-101 camera to follow the YAM gripper (optionally with the operator's head gain).

Start: park the camera at the glove5 median head pose, closed-loop against the tiles
(so101_cam_park.closed_loop_park). Then CALIBRATE the rotation mapping at the park pose with the
tiles: command three small view rotations through the kinematic model, measure what the camera
actually did, and fit the rotation + scale between the model's camera axes and the real ones
(the model's camera roll is poorly identified, so a commanded "yaw" would otherwise come out as
yaw+pitch). A verification wiggle then reports the residual. Then loop at --hz:

  YAM joints (ZMQ broadcast from yam_zmq_host, port obs_port+1)
    -> gripper point in the tile frame (i2rt FK + hand-eye + forward-cam tile pose)
    -> bearing of the gripper in the PARKED camera frame
    -> desired view offset = gain * (bearing - ref_bearing)
         default gain 1/1 = keep the gripper where it was at park (centred-ish); --human-gain = .36/.12
    -> low-pass + rate limit + |offset| cap
    -> joint command by damped least squares on the kinematic model, anchored to the tile-verified park pose
       and corrected by the fitted rotation mapping

Identical at teleop time and at policy-rollout time: both only need the YAM joint state.
--static: gains zero (parked camera, same logging). --release_cam: free the arm camera for the recorder
after the calibration. Log: data/raw/<name>_so101cam_track.jsonl. Ctrl-C parks the arm.

  python scripts/so101_cam_track.py --name track_test                 # keeps the camera: prints tiles vs model
  python scripts/so101_cam_track.py --name yam_head15 --hand-home        # collection / rollout: viewpoint model, camera released, box + 50 deg/s (defaults since 2026-10-16)
(run in the lerobot_alohamini conda env: lerobot + mink/mujoco for the YAM FK)
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path.home() / "co/i2rt"))
import so101_cam_park as P  # noqa: E402

YAM_JOINTS = [f"joint{i}.pos" for i in range(1, 7)]
REF_BEARING_DEG = (2.2, 8.0)         # median gripper-point bearing in the parked view over pilot_all5 teleop joints (zero offset there);
                                     # the human equivalent (ball at px (1224, 989), f 2539) was (-14.3, -1.9) in the human's own view


def bearing_deg(v):
    """(yaw, pitch) deg of a vector in camera coordinates (x right, y down, z forward)."""
    return np.degrees(np.array([np.arctan2(v[0], v[2]), np.arctan2(v[1], v[2])]))


def reach_check(fk, T_N, n=40, tol_mm=10.0, tol_deg=1.5):
    """How much of the operator's in-episode head volume (calib/head_residual_glove5.npz) the arm reaches from this base
    placement: IK of n sampled head poses on the (base-corrected) model; prints the reachable fraction, the median
    shortfall direction in robot axes and the joints that limit it."""
    d = np.load(ROOT / "calib/head_residual_glove5.npz"); Y = d["Y"]; rng = np.random.default_rng(0)
    T_dataN = np.array(d["T_N"]); idx = rng.choice(len(Y), size=min(n, len(Y)), replace=False)
    try:
        R_wy = np.array(json.load(open(ROOT / "calib/handeye.json"))["T_cam_base"])[:3, :3]
        R_wy = np.linalg.inv(np.array(json.load(open(ROOT / "calib/forward_cam_world_20261012.json"))["T_cam_world"]))[:3, :3] @ R_wy
    except Exception:
        R_wy = np.eye(3)
    q0 = np.zeros(6); ok = 0; short = []; lim = {}
    lo, hi = P.LIMITS
    for i in idx:
        T = target_pose(T_N, Y[i, 3:]); T[:3, 3] = T_N[:3, 3] + T_dataN[:3, :3] @ Y[i, :3] / 1000.0
        q = P.ik_model(fk, T, q0, restarts=3, quiet=True); q0 = q
        t, r = P.pose_err(T, fk(q)); t_w = T[:3, :3] @ t            # shortfall (reached - target) in tile axes, mm
        if np.linalg.norm(t) <= tol_mm and r <= tol_deg:
            ok += 1
        else:
            short.append(R_wy.T @ t_w); m = np.minimum(q - lo, hi - q)
            for k in np.where(m < 3)[0]:
                lim[P.JOINTS[k]] = lim.get(P.JOINTS[k], 0) + 1
    print(f"reach check: {ok}/{len(idx)} in-episode head poses reachable within {tol_mm:.0f} mm / {tol_deg} deg from this base placement")
    if short:
        med = np.median(np.array(short), 0)
        print(f"  unreached poses land a median {np.round(med)} mm (fwd, left, up; negative = camera stops short of the head in that direction); "
              f"joints at a limit: {lim or 'none'}")
    return ok / len(idx)


def target_pose(T_park, off_deg):
    """Park pose rotated so the view direction moves by (yaw, pitch) deg in the park camera frame; roll kept."""
    y, p = np.radians(off_deg)
    d_c = np.array([np.tan(y), np.tan(p), 1.0]); d = T_park[:3, :3] @ (d_c / np.linalg.norm(d_c))
    x0 = T_park[:3, 0]; x = x0 - np.dot(x0, d) * d; x /= np.linalg.norm(x); yax = np.cross(d, x)
    T = T_park.copy(); T[:3, :3] = np.stack([x, yax, d], 1)
    return T


def view_offset_deg(T_park, T):
    """Measured (yaw, pitch) of T's view direction in the park camera frame."""
    return bearing_deg(T_park[:3, :3].T @ T[:3, 2])


class RotMap:
    """Relative rotation (about the park pose, expressed in the park camera frame) actually produced by the
    camera = s * R_c applied to the rotation the model thinks it commanded. Fitted from wiggles; identity by default."""

    def __init__(self):
        self.R_c = np.eye(3); self.s = 1.0

    def fit(self, pairs):
        A = np.array([a for a, b in pairs]); B = np.array([b for a, b in pairs])
        U, _, Vt = np.linalg.svd(B.T @ A); d = np.sign(np.linalg.det(U @ Vt))
        self.R_c = U @ np.diag([1, 1, d]) @ Vt; self.s = float(np.linalg.norm(B, axis=1).sum() / max(np.linalg.norm(A, axis=1).sum(), 1e-9))
        res = np.degrees(np.linalg.norm(B - self.s * (self.R_c @ A.T).T, axis=1))
        return res

    def to_model(self, T_park, T_real_target):
        """Model-space target pose that should produce T_real_target."""
        b = Rot.from_matrix(T_park[:3, :3].T @ T_real_target[:3, :3]).as_rotvec()
        a = self.R_c.T @ b / self.s
        T = T_real_target.copy(); T[:3, :3] = T_park[:3, :3] @ Rot.from_rotvec(a).as_matrix(); return T   # translation passes through

    def to_real(self, T_park, T_model):
        """Real pose predicted from a model-space pose."""
        a = Rot.from_matrix(T_park[:3, :3].T @ T_model[:3, :3]).as_rotvec()
        T = T_model.copy(); T[:3, :3] = T_park[:3, :3] @ Rot.from_rotvec(self.s * (self.R_c @ a)).as_matrix(); return T


class YamHand:
    """Gripper point in the tile frame from the YAM joint broadcast."""

    def __init__(self, remote_ip, port):
        import zmq
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
        self.kin = Kinematics(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")
        he = json.load(open(ROOT / "calib/handeye.json"))
        T_world_fwd = np.linalg.inv(np.array(json.load(open(ROOT / "calib/forward_cam_world_20261012.json"))["T_cam_world"]))
        self.A = T_world_fwd @ np.array(he["T_cam_base"]); self.B = np.array(he["T_flange_body"])
        self.p_body = np.append(np.array(json.load(open(ROOT / "calib/eef_target.json"))["p_body_mm"]) / 1000.0, 1.0)
        ctx = zmq.Context.instance(); self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1); self.sock.setsockopt_string(zmq.SUBSCRIBE, ""); self.sock.connect(f"tcp://{remote_ip}:{port}")
        self.poller = zmq.Poller(); self.poller.register(self.sock, zmq.POLLIN)
        self.q = None; self.t = 0.0

    def update(self):
        """Latest YAM joints (rad), gripper point in the tile frame (m) and the gripper body pose (4x4, tile frame),
        or (None, None, None) if nothing fresh."""
        import zmq
        while self.poller.poll(0):
            o = json.loads(self.sock.recv_string(zmq.NOBLOCK)); self.q = np.array([float(o[n]) for n in YAM_JOINTS]); self.t = time.time()
        if self.q is None or time.time() - self.t > 1.0:
            return None, None, None
        T_world_body = self.A @ self.kin.fk(self.q) @ self.B
        return self.q, (T_world_body @ self.p_body)[:3], T_world_body


class MarkerHand:
    """Gripper point in the tile frame from the forward camera's LIVE marker pose (uvc_record.py POSE_PUB, the offline
    label chain: fused dodeca + calibrated base markers), so the head follows a glove exactly as it follows the robot."""

    def __init__(self, unit, port, remote_ip="127.0.0.1", max_age=0.5):
        import zmq
        he = json.load(open(ROOT / "calib/handeye.json"))
        self.T_world_fwd = np.linalg.inv(np.array(json.load(open(ROOT / "calib/forward_cam_world_20261012.json"))["T_cam_world"]))
        self.A = self.T_world_fwd @ np.array(he["T_cam_base"])        # robot axes in the tile frame (box clamp), either unit
        tgt = "calib/eef_target.json" if unit == "robot" else f"calib/eef_target_glove_{unit}.json"
        self.p_body = np.append(np.array(json.load(open(ROOT / tgt))["p_body_mm"]) / 1000.0, 1.0)
        ctx = zmq.Context.instance(); self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1); self.sock.setsockopt_string(zmq.SUBSCRIBE, ""); self.sock.connect(f"tcp://{remote_ip}:{port}")
        self.poller = zmq.Poller(); self.poller.register(self.sock, zmq.POLLIN)
        self.T = None; self.t = 0.0; self.max_age = max_age; self.n_msgs = 0; self.n_pose = 0; self.last = None
        print(f"hand source: forward-camera markers ({unit} unit, tcp port {port}); target {np.round(self.p_body[:3] * 1000)} mm in the body frame")

    def update(self):
        """(None, gripper point in the tile frame (m), body pose 4x4 in the tile frame) or (None, None, None) when the
        newest pose is older than max_age (no recorder / unit out of view)."""
        import cv2
        import zmq
        while self.poller.poll(0):
            m = json.loads(self.sock.recv_string(zmq.NOBLOCK)); self.n_msgs += 1; self.last = m
            if m.get("rvec") is not None:
                R, _ = cv2.Rodrigues(np.array(m["rvec"], dtype=float)); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = m["tvec"]
                self.T = self.T_world_fwd @ T; self.t = m["t"]; self.n_pose += 1
        if self.T is None or time.time() - self.t > self.max_age:
            return None, None, None
        return None, (self.T @ self.p_body)[:3], self.T


class Viewpoint:
    """Learned head viewpoint (scripts/head_residual_fit.py --save): from the gripper geometry in the neutral head
    frame N to the head's position offset (mm, in N) and view (yaw, pitch deg in N)."""

    def __init__(self, path, kind="ridge"):
        m = json.load(open(path)); self.kind = kind
        self.ridge = {k: np.array(v) for k, v in m["ridge"].items()}
        self.mlp = {k: np.array(v) for k, v in m["mlp"].items() if k != "layers"}; self.layers = [(np.array(w), np.array(b)) for w, b in m["mlp"]["layers"]]

    @staticmethod
    def feats(p, ax, ph):
        b = lambda v: np.degrees([np.arctan2(v[0], v[2]), np.arctan2(v[1], v[2])])
        return np.concatenate([p, ax, ph, b(p), b(ph), [np.linalg.norm(p)]])

    def predict(self, p_N_mm, axis_N, p_hist_N_mm):
        f = self.feats(p_N_mm, axis_N, p_hist_N_mm)
        if self.kind == "ridge":
            r = self.ridge; return np.append((f - r["mu"]) / r["sd"], 1.0) @ r["W"]
        h = (f - self.mlp["mu"]) / self.mlp["sd"]
        for i, (w, bb) in enumerate(self.layers):
            h = h @ w.T + bb
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h * self.mlp["ysd"] + self.mlp["ymu"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", default="/dev/ttyACM0"); ap.add_argument("--id", default="so101_cam"); ap.add_argument("--cam", default=P.ARM_CAM)
    ap.add_argument("--remote_ip", default="127.0.0.1"); ap.add_argument("--obs_port", type=int, default=5602, help="host obs port; the broadcast is on obs_port+1")
    ap.add_argument("--gain-yaw", type=float, default=1.0); ap.add_argument("--gain-pitch", type=float, default=1.0)
    ap.add_argument("--human-gain", action="store_true", help="use the operator's head gain .36 yaw / .12 pitch")
    ap.add_argument("--ref-bearing", type=float, nargs=2, default=REF_BEARING_DEG, help="deg: gripper bearing (park frame) that maps to zero offset")
    ap.add_argument("--static", action="store_true", help="gains 0: parked camera with the same logging")
    ap.add_argument("--ctrl-port", type=int, default=5612, help="ZMQ REP port: {\"mode\": \"track\"|\"static\"} switches tracking at run time without redoing the solve (eval_grid uses it per arm); 0 = off")
    ap.add_argument("--no-yam", action="store_true", help="do not subscribe to the YAM (park + calibration + hold only)")
    ap.add_argument("--hand-source", default="yam", choices=["yam", "markers"],
                    help="what the head follows: the YAM joint broadcast (FK) or the forward camera's live marker pose published by the "
                         "recorder (uvc_record.py POSE_PUB; works for the glove and the robot alike)")
    ap.add_argument("--marker-unit", default="robot", help="--hand-source markers: 'robot' or a glove unit name (eef_target_glove_<unit>.json)")
    ap.add_argument("--pose-port", type=int, default=5611); ap.add_argument("--pose-max-age", type=float, default=0.5, help="s: older poses count as no hand")
    ap.add_argument("--hz", type=float, default=15.0); ap.add_argument("--tau", type=float, default=0.15, help="s, low-pass on the desired offset (setpoint shaping, outside the feedback loop). 0.4 s put the camera ~0.7 s behind the gripper; the human head trails the hand by ~0 ms")
    ap.add_argument("--max-rate", type=float, default=50.0, help="deg/s cap on the view offset (default = the head12-14 collection setting)")
    ap.add_argument("--max-off", type=float, default=20.0, help="deg cap on |offset| (keeps the table in view)")
    ap.add_argument("--park-rounds", type=int, default=3, help="wiggle-fit / park rounds (each park improves the map, each map improves the park)")
    ap.add_argument("--park-tol-mm", type=float, default=8.0); ap.add_argument("--park-tol-deg", type=float, default=2.5, help="the arm repeats its own pose to ~6 mm / 1 deg (backlash + flex); stop the park loops there instead of random-walking below it")
    ap.add_argument("--wiggle", type=float, default=6.0, help="deg: rotation-mapping calibration amplitude (0 = skip, model used as is)")
    ap.add_argument("--viewpoint", default="calib/head_residual_model.json", help="learned head-viewpoint model JSON (head_residual_fit.py --save): position + view from the gripper geometry; 'none' = bearing-only tracking")
    ap.add_argument("--viewpoint-kind", default="ridge", choices=["ridge", "mlp"])
    ap.add_argument("--pos-rate", type=float, default=250.0, help="mm/s cap on the camera position offset (viewpoint mode); human head p90 speed is 122 mm/s")
    ap.add_argument("--box", type=float, nargs=6, default=[-110, 125, -90, 100, -55, 50], metavar=("FWD_MIN", "FWD_MAX", "LEFT_MIN", "LEFT_MAX", "UP_MIN", "UP_MAX"),
                    help="mm, ROBOT axes (forward, left, up) relative to the parked camera (= the operator's in-episode median head pose): box the camera "
                         "may move in. Default follows the operator's head range (p5-p95) except on the right, stopped at 4 cm because the forward camera "
                         "sits ~10 cm right / 15 cm below the park")
    ap.add_argument("--meas-tau", type=float, default=0.0, help="s, low-pass on the measured joints used for the pose estimate. 0 = none: a filter inside the loop adds lag and the loop rings after a fast move (0.3 s gave 60 %% overshoot in simulation and a 1.2 Hz limit cycle on the arm); the command deadband handles reading flicker")
    ap.add_argument("--loop-gain", type=float, default=0.5, help="per-tick gain of the damped-LS joint correction (0.8 rang with a filtered measurement; 0.5 settles in ~0.7 s with ~3 %% overshoot)")
    ap.add_argument("--deadband", type=float, default=0.15, help="deg: a new servo command is sent only when some joint changes by at least this much")
    ap.add_argument("--limit-margin", type=float, default=3.0, help="deg inside the calibrated joint range the TRACKING command may go (park/IK keep the 12 deg margin). 12 lost ~35 mm of reach when the head retreats; 3 recovers it")
    ap.add_argument("--lead", type=float, default=12.0, help="deg: how far the integrated joint command may lead the measured joints (servo lag under load)")
    ap.add_argument("--p-gain", type=int, default=32); ap.add_argument("--i-gain", type=int, default=0); ap.add_argument("--d-gain", type=int, default=32)
    ap.add_argument("--accel", type=int, default=0, help="servo goal ramp (STS3215 Acceleration register, x 8.8 deg/s^2; 0 = none). ~5 smooths the 15 Hz stop-go without adding much lag")
    ap.add_argument("--release_cam", action=argparse.BooleanOptionalAction, default=True, help="close the arm camera after the calibration (the recorder / evaluator opens it); --no-release_cam keeps it for the once-a-second tiles-vs-model print")
    ap.add_argument("--reach-check", type=int, default=40, metavar="N", help="after --hand-home: IK N in-episode operator head poses on the corrected model and report the reachable fraction (0 = skip)")
    ap.add_argument("--hand-home", action="store_true", help="after a re-mount: torque off, point the camera at the tiles by hand, ENTER; corrects the model's base before the IK")
    a = ap.parse_args()
    if a.viewpoint and a.viewpoint.lower() == "none":
        a.viewpoint = None
    gains = np.zeros(2) if a.static else (np.array([0.36, 0.12]) if a.human_gain else np.array([a.gain_yaw, a.gain_pitch])); ref = np.array(a.ref_bearing)

    if a.viewpoint:      # park at the model's own neutral frame (the operator's median head pose DURING episodes)
        T_head = np.array(json.load(open(a.viewpoint))["T_N"]); print("park target = the viewpoint model's neutral frame:", np.round(T_head[:3, 3] * 1000), "mm")
    else:
        T_head = P.head_median_pose()
    fk = P.load_model(); arm = P.Arm(a.port, a.id, a.p_gain, a.i_gain, a.d_gain, accel=a.accel); cam = P.ArmCam(a.cam)
    out = ROOT / f"data/raw/{a.name}_so101cam_track.jsonl"; logf = open(out, "w")
    hand = None if a.no_yam else (MarkerHand(a.marker_unit, a.pose_port, a.remote_ip, a.pose_max_age) if a.hand_source == "markers"
                                  else YamHand(a.remote_ip, a.obs_port + 1))
    stop = {"flag": False}; signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    rmap = RotMap()
    try:
        if a.hand_home:
            fk = P.hand_home(arm, cam, fk)
            if a.reach_check:
                reach_check(fk, T_head, a.reach_check)
        # The model's rotation axes are off by 35-50 deg about the camera's x axis (the wrist-yaw axis direction was never
        # identified on this mount), so a park loop on the raw model walks in circles. Park coarsely, fit the rotation map
        # from wiggles, then park again on the corrected model.
        seed = ROOT / "calib/so101_cam_park_seed.json"       # IK branch that is known to park well on this mount (wrist nearly straight)
        q_seed = np.array(json.load(open(seed))["q_park_deg"]) if seed.exists() else None
        q_park, T_park = P.closed_loop_park(arm, cam, fk, T_head, iters=20, q0=q_seed, restarts=1 if q_seed is not None else 8, tol_mm=a.park_tol_mm, tol_deg=a.park_tol_deg)
        if T_park is None:
            sys.exit("could not verify the park pose against the tiles")
        D = T_park @ np.linalg.inv(fk(q_park))                      # anchors the model to the measured park pose
        est = lambda q: D @ fk(q)                                    # model-space estimate of the camera pose
        print(f"coarse park: {np.linalg.norm(P.pose_err(T_head, T_park)[0]):.1f} mm / {P.pose_err(T_head, T_park)[1]:.2f} deg from the head median")

        def go_model(T_t_model, steps=10):
            """Drive the model estimate to T_t_model (integrated command, see the loop); return
            (tile-measured pose, model's own estimate of the reached pose)."""
            qc = arm.joints().copy()
            for _ in range(steps):
                qn = arm.joints(); dq = P.dls_step(fk, qn, T_t_model, P.err6(T_t_model, est(qn)), gain=1.0) - qn
                qc = np.clip(qc + dq, qn - a.lead, qn + a.lead); arm.move_to(qc, 0.25)
            time.sleep(0.5); s = cam.solve(); return (s[0] if s else None), est(arm.joints())

        calib = {}
        if a.wiggle > 0:
            w = a.wiggle; e_mm = e_deg = np.inf
            # The map is only as good as the park it is measured at (0.7 deg residuals at a 4 mm park, 7 deg at a 25 mm one),
            # and the park is only as good as the map that steers it: alternate the two until the park is tight.
            for rnd in range(a.park_rounds):
                pairs = []
                # pairs: (what the MODEL thinks it rotated by, what the TILES say it rotated by), both about the park pose
                for name, off in (("yaw", (w, 0.0)), ("pitch", (0.0, w)), ("both", (-w, w)), ("neg", (-w, -w))):
                    T_m, T_e = go_model(target_pose(T_park, off))
                    if T_m is None:
                        print(f"  wiggle {name}: no tiles — skipped"); continue
                    a_rv = Rot.from_matrix(T_park[:3, :3].T @ T_e[:3, :3]).as_rotvec(); b_rv = Rot.from_matrix(T_park[:3, :3].T @ T_m[:3, :3]).as_rotvec()
                    pairs.append((a_rv, b_rv))
                    calib[f"r{rnd}_{name}"] = {"commanded": off, "model_reached": view_offset_deg(T_park, T_e).round(2).tolist(), "measured": view_offset_deg(T_park, T_m).round(2).tolist()}
                    print(f"  wiggle {name} {off}: model reached {np.round(view_offset_deg(T_park, T_e), 2)}, tiles measured {np.round(view_offset_deg(T_park, T_m), 2)}")
                if len(pairs) < 2:
                    print("  too few wiggles solved — keeping the previous rotation map"); break
                res = rmap.fit(pairs)
                print(f"  rotation map fitted (round {rnd + 1}): axes rotated {np.degrees(np.linalg.norm(Rot.from_matrix(rmap.R_c).as_rotvec())):.1f} deg, scale {rmap.s:.2f}, fit residual {np.round(res, 2)} deg")
                T_park0 = T_park.copy(); fk_c = lambda q: rmap.to_real(T_park0, est(q))      # model with the measured rotation behaviour
                q_new, T_new = P.closed_loop_park(arm, cam, fk_c, T_head, iters=10, restarts=1, move_s=1.5, q0=q_park, tol_mm=a.park_tol_mm, tol_deg=a.park_tol_deg)
                if T_new is None:
                    print("  lost the tiles during the park — keeping the previous park"); break
                q_park, T_park = q_new, T_new
                D = T_park @ np.linalg.inv(fk(q_park)); est = lambda q: D @ fk(q)             # re-anchor at the new park
                e_mm, e_deg = np.linalg.norm(P.pose_err(T_head, T_park)[0]), P.pose_err(T_head, T_park)[1]
                print(f"parked (round {rnd + 1}): {e_mm:.1f} mm / {e_deg:.2f} deg from the head median")
                if e_mm < a.park_tol_mm and e_deg < a.park_tol_deg:
                    break
            if e_mm < 8 and e_deg < 4:
                json.dump({"q_park_deg": q_park.round(2).tolist(), "note": f"park of {a.name}: {e_mm:.1f} mm / {e_deg:.2f} deg"}, open(seed, "w"), indent=1)
            for name, off in (("yaw", (w, 0.0)), ("pitch", (0.0, w))):           # verification through the corrected mapping
                T_m, T_e = go_model(rmap.to_model(T_park, target_pose(T_park, off)))
                got = view_offset_deg(T_park, T_m).round(2).tolist() if T_m is not None else None
                calib[f"verify_{name}"] = {"commanded": off, "measured": got}; print(f"  verify {name} {off} -> tiles measured {got}")
            arm.move_to(q_park, 1.0); time.sleep(0.3)
        if a.release_cam:
            cam.release(); print("camera released for the recorder")
        logf.write(json.dumps({"header": True, "name": a.name, "gains": gains.tolist(), "ref_bearing_deg": ref.tolist(), "static": a.static, "hand_source": a.hand_source, "marker_unit": a.marker_unit,
                               "T_world_cam_park": T_park.tolist(), "q_park": q_park.tolist(), "calib": calib, "viewpoint": a.viewpoint, "viewpoint_kind": a.viewpoint_kind,
                               "rot_map": {"R_c": rmap.R_c.tolist(), "s": rmap.s}, "t0": time.time()}) + "\n")

        vp = Viewpoint(a.viewpoint, a.viewpoint_kind) if a.viewpoint else None
        if vp is not None:
            p_body_dir = hand.p_body[:3] / np.linalg.norm(hand.p_body[:3]) if hand else None
            R_wy = hand.A[:3, :3] if hand else np.eye(3)                 # robot (YAM) axes in the tile frame
            box = np.array(a.box).reshape(3, 2)
        # tracking clips the joint command closer to the calibrated range than park/IK do (P.LIMITS, 12 deg margin):
        # with 12 deg the elbow's folded side ran out on ~1/3 of the in-episode ticks (head retreating/rising when the
        # gripper is near home) and the camera sat ~35 mm forward+low of the target; 3 deg recovers it (head9 log, IK re-solve)
        track_limits = P.joint_limits(margin_deg=a.limit_margin)
        print(f"tracking joint margin {a.limit_margin:.0f} deg: elbow to {track_limits[1][2]:.0f}, shoulder_lift to {track_limits[0][1]:.0f} deg")
        off_cmd = np.zeros(2); pos_cmd = np.zeros(3); dt = 1.0 / a.hz; last_print = 0.0; n_hand = 0; tick = 0; last_tiles = None; hist = []; q = q_park.copy(); qf = None; q_sent = None
        print(("viewpoint model: " + a.viewpoint if vp else "tracking") + " — Ctrl-C to park and exit" if not a.static else "holding (static) — Ctrl-C to park and exit")
        Ti = np.linalg.inv(T_park)
        mode_static = bool(a.static); ctrl = None
        if a.ctrl_port:
            import zmq
            ctrl = zmq.Context.instance().socket(zmq.REP); ctrl.bind(f"tcp://*:{a.ctrl_port}")
            print(f"control socket tcp://*:{a.ctrl_port}: send {{\"mode\": \"track\"|\"static\"}} (or \"status\")")
        while not stop["flag"]:
            t0 = time.time(); q_yam, p_hand, T_body = (hand.update() if hand else (None, None, None))
            if ctrl is not None:
                try:
                    while True:
                        m = json.loads(ctrl.recv_string(zmq.NOBLOCK))
                        if m.get("mode") in ("track", "static"):
                            if (m["mode"] == "static") != mode_static:
                                mode_static = m["mode"] == "static"; print(f"\n[ctrl] mode -> {m['mode']}")
                        ctrl.send_string(json.dumps({"ok": True, "mode": "static" if mode_static else "track", "tick": tick}))
                except zmq.Again:
                    pass
            pos_des = np.zeros(3)
            if p_hand is not None:
                p_N = (Ti[:3, :3] @ p_hand + Ti[:3, 3]) * 1000; b = bearing_deg(p_N); n_hand += 1
                hist.append((t0, p_N)); hist = [h for h in hist if t0 - h[0] <= 0.6]
                if vp is not None and not mode_static:
                    ph = next((p for (th, p) in hist if t0 - th >= 0.45), hist[0][1])
                    y = vp.predict(p_N, Ti[:3, :3] @ T_body[:3, :3] @ p_body_dir, ph)
                    pos_des = T_park[:3, :3] @ y[:3]                      # offset in N's axes -> tile-frame axes (mm)
                    pos_des = R_wy @ np.clip(R_wy.T @ pos_des, box[:, 0], box[:, 1]); off_des = y[3:]   # clamp in robot axes
                else:
                    off_des = np.zeros(2) if mode_static else gains * (b - ref)
            else:
                b = None; off_des = np.zeros(2)                       # no hand: drift back to the park pose
            off_des = np.clip(off_des, -a.max_off, a.max_off)
            step = (off_des - off_cmd) * min(1.0, dt / a.tau)
            off_cmd = off_cmd + np.clip(step, -a.max_rate * dt, a.max_rate * dt)
            pstep = (pos_des - pos_cmd) * min(1.0, dt / a.tau)
            pos_cmd = pos_cmd + np.clip(pstep, -a.pos_rate * dt, a.pos_rate * dt)
            T_real = target_pose(T_park, off_cmd); T_real[:3, 3] = T_park[:3, 3] + pos_cmd / 1000.0
            T_t_model = rmap.to_model(T_park, T_real)
            qn = arm.joints()
            # the servo readings are quantised (0.088 deg) and flicker by a count; stepping the controller from the raw
            # reading makes the arm hunt visibly when the target sits at a joint limit. Filter the reading for the estimate.
            qf = qn.copy() if (qf is None or a.meas_tau <= 0) else qf + (qn - qf) * min(1.0, dt / a.meas_tau)
            # integrate the COMMAND (not the measured joints): under load the servos sit a few degrees off their goal,
            # and a step added to the measured position never overcomes that; the lead cap is the anti-windup
            dq = P.dls_step(fk, qf, T_t_model, P.err6(T_t_model, est(qf)), max_deg=a.max_rate * dt * 1.5, gain=a.loop_gain, limits=track_limits) - qf
            q = np.clip(q + dq, qn - a.lead, qn + a.lead)
            if q_sent is None or np.abs(q - q_sent).max() >= a.deadband:     # do not chase the servo's own resolution
                arm.command(q); q_sent = q.copy()
            rec = {"t": t0, "q_so101": q.tolist(), "q_meas": qn.round(2).tolist(), "q_yam": q_yam.tolist() if q_yam is not None else None,
                   "p_hand_mm": (p_hand * 1000).round(1).tolist() if p_hand is not None else None,
                   "bearing": b.round(2).tolist() if b is not None else None, "off_des": off_des.round(2).tolist(), "off_cmd": off_cmd.round(2).tolist(),
                   "pos_des": pos_des.round(1).tolist(), "pos_cmd": pos_cmd.round(1).tolist(), "static": mode_static,
                   "off_est": view_offset_deg(T_park, rmap.to_real(T_park, est(qn))).round(2).tolist(),
                   "pos_est": ((est(qn)[:3, 3] - T_park[:3, 3]) * 1000).round(1).tolist()}
            if not a.release_cam and tick % max(1, int(a.hz)) == 0:   # once a second, the tiles' opinion
                s = cam.solve(frames=1); last_tiles = view_offset_deg(T_park, s[0]).round(2).tolist() if s else None; rec["off_tiles"] = last_tiles
            logf.write(json.dumps(rec) + "\n"); tick += 1
            if t0 - last_print > 1.0:
                print(f"\r hand {'ok ' if b is not None else 'none'} bearing {np.round(b, 1) if b is not None else '--'}  view cmd {np.round(off_cmd, 1)} est {rec['off_est']}"
                      f"  pos cmd {np.round(pos_cmd)} est {rec['pos_est']}  tiles {last_tiles if last_tiles is not None else '--'}      ", end="", flush=True); last_print = t0
            time.sleep(max(0.0, dt - (time.time() - t0)))
        print(f"\n{tick} ticks, hand seen on {n_hand}; wrote {out}")
    finally:
        logf.close(); print("parking…"); arm.park_and_close()
        if not a.release_cam:
            cam.release()


if __name__ == "__main__":
    main()
