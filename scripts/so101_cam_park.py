#!/usr/bin/env python3
"""Park the SO-101 camera at a target pose in the tile frame, closed-loop against the tiles.

Default target: the operator's median head pose from the glove5 head-camera sessions
(position and orientation, tile frame). The rough kinematic model (calib/so101_cam_kin.json)
gives an IK start; then: move, solve the tiles, correct with the model's Jacobian, repeat
until within --tol_mm / --tol_deg. Then HOLD with torque on, logging the encoders and the
tile-solved camera pose every few seconds to data/raw/<name>_so101cam_log.json, until Ctrl-C;
on exit the arm parks at calib/so101_cam_rest.json.

  python scripts/so101_cam_park.py --name yam_teleop_head1 --release_cam
  python scripts/so101_cam_park.py --target calib/head_pose_median.json ...   # any T_world_cam

The pieces (robot, camera, tile solve, IK, closed-loop park) are functions so that
so101_cam_track.py can reuse them.
"""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import so101_cam_kin as K  # noqa: E402
from tile_solver import TileSolver  # noqa: E402

JOINTS = K.CHAIN + ["wrist_yaw"]
HEAD_STEMS = ["yam_glove5_b0587b.take1", "yam_glove5_b0587b"]
ARM_CAM = "/dev/v4l/by-path/pci-0000:10:00.3-usb-0:2.1:1.0-video-index0"
REST = ROOT / "calib/so101_cam_rest.json"


def head_median_pose():
    """Median camera pose (position + chordal-mean rotation) over the glove5 head frames with >= 2 tiles."""
    Ts = []
    for stem in HEAD_STEMS:
        for r in json.load(open(ROOT / f"calib/{stem}_cam_world.json"))["per_frame"]:
            if r["tiles"] >= 2 and r["rms"] <= 5.0:
                Ts.append(np.linalg.inv(np.array(r["T_cam_world"])))
    Ts = np.array(Ts)
    T = np.eye(4); T[:3, 3] = np.median(Ts[:, :3, 3], 0)
    M = Ts[:, :3, :3].mean(0); U, _, Vt = np.linalg.svd(M); T[:3, :3] = U @ Vt
    return T


def pose_err(T_target, T):
    """(translation error mm (3), rotation error deg) of T relative to T_target, in T_target's frame."""
    D = np.linalg.inv(T_target) @ T
    return D[:3, 3] * 1000.0, np.degrees(np.linalg.norm(Rot.from_matrix(D[:3, :3]).as_rotvec()))


def err6(T_target, T):
    """6-vector residual: mm (3) and deg*5 (3)."""
    D = np.linalg.inv(T_target) @ T
    return np.concatenate([D[:3, 3] * 1000.0, np.degrees(Rot.from_matrix(D[:3, :3]).as_rotvec()) * 5.0])


def load_model():
    d = json.load(open(K.OUT)); model = K.Model(K.load_urdf(), d["signs"]); x = np.array(d["x"])
    return lambda q: model.fk(x, q)


class Arm:
    """SO-101 camera positioner: joints in degrees, interpolated moves, rest parking. Torque is never dropped on exit."""

    def __init__(self, port="/dev/ttyACM0", id="so101_cam", p_gain=32, i_gain=0, d_gain=32, accel=0):
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower
        self.robot = SOFollower(SOFollowerRobotConfig(port=port, id=id, arm_profile="so-arm-cam-6dof", use_degrees=True,
                                                      max_relative_target=20.0, disable_torque_on_disconnect=False))
        self.robot.connect(calibrate=False)
        # LeRobot's follower configure() halves the servo P gain (16) with I=0 to avoid teleop shakiness; a pure
        # low-gain P loop leaves a large steady-state error under the extended camera arm's load, so the shoulder
        # and elbow sit short of their goals. The camera arm is not a teleop follower: restore the gains.
        with self.robot.bus.torque_disabled():
            for m in self.robot.bus.motors:
                self.robot.bus.write("P_Coefficient", m, int(p_gain)); self.robot.bus.write("I_Coefficient", m, int(i_gain)); self.robot.bus.write("D_Coefficient", m, int(d_gain))
                # goal-position ramp inside the servo (STS3215 'Acceleration', units of 100 steps/s^2 = 8.8 deg/s^2; 0 = none):
                # with 15 Hz goal updates and no ramp the servo sprints to each new goal and stops, a 15 Hz stop-go that
                # shows as a shaky picture even though the commanded trajectory is smooth
                self.robot.bus.write("Acceleration", m, int(accel))
        self.robot.bus.enable_torque()

    def joints(self):
        o = self.robot.get_observation(); return np.array([float(o[f"{j}.pos"]) for j in JOINTS])

    def command(self, q):
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})

    def move_to(self, q_target, seconds):
        q0 = self.joints(); n = max(2, int(seconds * 30)); q_target = np.asarray(q_target, float)
        for i in range(1, n + 1):
            self.command(q0 + (i / n) * (q_target - q0)); time.sleep(seconds / n)

    def park_and_close(self):
        if REST.exists():
            r = json.load(open(REST)); self.move_to([r[j] for j in JOINTS], 4.0); time.sleep(0.5); self.robot.bus.disable_torque()
        self.robot.disconnect()


class ArmCam:
    """4K MJPG capture of the arm camera + tile solve (median of a few frames)."""

    def __init__(self, dev=ARM_CAM):
        self.dev = dev
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160); self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(10):
            self.cap.read()
        # UVC controls on these cameras latch only while frames are flowing (forward-cam-umi/uvc_ritual.py),
        # so the exposure ritual is applied AFTER the stream is up, not before opening the device
        sys.path.insert(0, str(Path.home() / "co/forward-cam-umi/scripts"))
        try:
            from uvc_ritual import apply as _ritual
            _ritual(dev)
        except Exception as e:
            print(f"[ritual skipped: {e}]")
        time.sleep(0.5); m = self._brightness()
        if m < 8:                                                     # black: try the camera's own auto exposure on the live stream
            subprocess.run(["v4l2-ctl", "-d", dev, "-c", "auto_exposure=3"], capture_output=True)
            time.sleep(1.0); m2 = self._brightness()
            if m2 < 8:
                print(f"WARNING: arm camera frames are black (mean {m:.1f} after the ritual, {m2:.1f} with auto exposure) — the camera's "
                      "ISP is in its stuck state (controls read back fine but do nothing). Replug its USB and rerun.")
            else:
                print(f"note: frames were black at the ritual exposure (mean {m:.1f}); running on auto exposure instead (mean {m2:.0f}) — fine for the tiles")
        self.solver = TileSolver(); self.K = self.solver.K

    def _brightness(self):
        for _ in range(4):
            self.cap.grab()
        ok, im = self.cap.read(); return float(im.mean()) if ok else float("nan")

    def frame(self):
        self.cap.grab(); self.cap.grab(); ok, im = self.cap.read()
        return im if ok else None

    def solve(self, frames=3):
        """(T_world_cam, rms_px, n_tiles) from the median-rms solve over a few frames, >= 2 tiles; None if nothing."""
        res = []
        for _ in range(frames):
            im = self.frame()
            if im is not None:
                r = self.solver.solve(im)
                if r is not None and r[2] >= 2:
                    res.append(r)
        if not res:
            return None
        res.sort(key=lambda r: r[1]); T_cw, rms, nt = res[len(res) // 2]
        return np.linalg.inv(T_cw), rms, nt

    def release(self):
        self.cap.release()


def joint_limits(robot_id="so101_cam", margin_deg=12.0):
    """(lo, hi) in degrees from the LeRobot calibration ranges (DEGREES mode is centred on the range midpoint),
    pulled in by margin_deg. Joints parked at a limit cannot move further, which stalls the controller."""
    p = Path.home() / f".cache/huggingface/lerobot/calibration/robots/so_follower/{robot_id}.json"
    cal = json.load(open(p)); half = np.array([(cal[j]["range_max"] - cal[j]["range_min"]) / 2 * 360 / 4095 for j in JOINTS])
    return -(half - margin_deg), half - margin_deg


LIMITS = joint_limits()


def ik_model(fk, T_target, q0, restarts=8, limits=None, quiet=False):
    """Joint angles (deg) that put the model's camera at T_target, best of several restarts, within the joint limits."""
    lo, hi = limits if limits is not None else LIMITS; rng = np.random.default_rng(0); sol = None
    for s in range(restarts):
        qs = np.clip(q0 if s == 0 else q0 + rng.uniform(-60, 60, 6), lo + 1, hi - 1)
        cand = least_squares(lambda q: err6(T_target, fk(q)), qs, bounds=(lo, hi), max_nfev=400)
        # prefer branches that keep every joint >= 8 deg from its limit: a solution parked at a limit cannot be
        # corrected against the tiles in that direction (the loop then circles). 1 deg short of 8 ~ a 1.4 mm residual.
        m = np.minimum(cand.x - lo, hi - cand.x).min(); score = cand.cost + 0.5 * (2.0 * max(0.0, 8.0 - m)) ** 2
        if sol is None or score < best:
            sol, best = cand, score
    q = sol.x; t, r = pose_err(T_target, fk(q)); margin = np.minimum(q - lo, hi - q)
    if not quiet and (np.linalg.norm(t) > 10 or r > 1.5 or margin.min() < 3):
        print(f"  IK note: residual {np.linalg.norm(t):.1f} mm / {r:.2f} deg within joint limits; smallest margin {margin.min():.0f} deg on {JOINTS[int(np.argmin(margin))]}")
    return q


def jacobian(fk, q, T_ref, h=0.5):
    """d err6(T_ref, fk(q)) / dq, numerical (6x6)."""
    Jm = np.zeros((6, 6))
    for k in range(6):
        dq = np.zeros(6); dq[k] = h
        Jm[:, k] = (err6(T_ref, fk(q + dq)) - err6(T_ref, fk(q - dq))) / (2 * h)
    return Jm


def dls_step(fk, q, T_target, e_meas, damping=4.0, max_deg=8.0, gain=0.8, limits=None):
    """One damped-least-squares joint correction that drives the MEASURED residual e_meas (err6 of the camera
    vs T_target) to zero, using the model's Jacobian at q. `limits` (lo, hi) defaults to LIMITS (12 deg margin)."""
    lo, hi = limits if limits is not None else LIMITS
    Jm = jacobian(fk, q, T_target)
    dq = -np.linalg.solve(Jm.T @ Jm + damping * np.eye(6), Jm.T @ e_meas)
    return np.clip(q + np.clip(dq, -max_deg, max_deg) * gain, lo, hi)


def base_corrected_fk(fk, q_home, T_home_meas):
    """fk wrapped so the model agrees with a tile measurement at q_home (rigid base correction).
    Use after the arm has been re-mounted and calib/so101_cam_kin.json still holds the old base pose."""
    D = T_home_meas @ np.linalg.inv(fk(q_home))
    print(f"model base corrected from the home measurement: moved {np.linalg.norm(D[:3, 3]) * 1000:.0f} mm, {np.degrees(np.linalg.norm(Rot.from_matrix(D[:3, :3]).as_rotvec())):.1f} deg")
    return lambda q: D @ fk(q)


def hand_home(arm, cam, fk):
    """Torque off, let the user point the camera at the tiles by hand, ENTER; returns the base-corrected fk."""
    import select
    arm.robot.bus.disable_torque()
    print("Torque is OFF. Hold the arm so the camera sees at least three tiles (a 'home' view); the line below shows what it sees. ENTER when it reads 3-4 tiles.")
    s = None
    while True:
        im = cam.frame(); r = cam.solver.solve(im) if im is not None else None
        n = r[2] if r else 0; bright = float(im.mean()) if im is not None else float("nan")
        print(f"\r  tiles seen: {n}   frame brightness {bright:.0f}/255   {'OK' if n >= 3 else ''}      ", end="", flush=True)
        if select.select([sys.stdin], [], [], 0.2)[0]:
            sys.stdin.readline(); break
    print()
    q = arm.joints(); arm.robot.bus.enable_torque(); arm.command(q); time.sleep(0.3)
    s = cam.solve()
    if s is None:
        im = cam.frame(); p = ROOT / "data/raw/hand_home_fail.jpg"
        if im is not None:
            cv2.imwrite(str(p), cv2.resize(im, (1280, 720)))
        sys.exit(f"no tile solve at the hand-placed home — need >= 2 tiles in view (frame saved to {p})")
    print(f"home solve: {s[2]} tiles, {s[1]:.2f} px RMS")
    return base_corrected_fk(fk, q, s[0])


def closed_loop_park(arm, cam, fk, T_target, tol_mm=3.0, tol_deg=0.5, iters=8, verbose=True, restarts=8, move_s=3.0, q0=None, lead_deg=12.0):
    """Move to the model's IK solution, then correct against the tiles. Returns (q, T_meas or None).
    The correction is integrated on the COMMANDED joints: under the camera's load the servos settle short of their
    goal, so a step re-based on the measured joints sinks the arm a little further every iteration."""
    q = ik_model(fk, T_target, arm.joints() if q0 is None else np.asarray(q0, float), restarts=restarts)
    t, r = pose_err(T_target, fk(q))
    if verbose:
        print(f"IK (model): predicted residual {np.linalg.norm(t):.1f} mm, {r:.2f} deg; joints {np.round(q, 1)}")
    arm.move_to(q, move_s); time.sleep(0.8); s = None; best = None; q_cmd = q.copy(); it = -1
    for it in range(iters):
        s = cam.solve()
        if s is None:
            print("  no tile solve (need >= 2 tiles in view) — stopping the correction here"); break
        T_meas, rms, nt = s; t, r = pose_err(T_target, T_meas); qc = arm.joints(); score = np.linalg.norm(t) + 5.0 * r
        if verbose:
            print(f"  iter {it}: error {np.linalg.norm(t):.1f} mm, {r:.2f} deg ({nt} tiles, {rms:.2f} px)")
        if best is None or score < best[0]:
            best = (score, q_cmd.copy(), T_meas, it)
        if np.linalg.norm(t) < tol_mm and r < tol_deg:
            break
        dq = dls_step(fk, qc, T_target, err6(T_target, T_meas)) - qc        # correction from the measured state
        q_cmd = np.clip(q_cmd + dq, qc - lead_deg, qc + lead_deg)             # ... applied to the command (anti-windup: lead cap)
        arm.move_to(q_cmd, 1.0); time.sleep(0.6)
    if best is not None and s is not None and best[3] != it:          # the loop wandered off after its best pose: go back there
        arm.move_to(best[1], 1.0); time.sleep(0.8); s2 = cam.solve()
        if s2 is not None:
            t, r = pose_err(T_target, s2[0])
            if verbose:
                print(f"  back at the best iterate ({best[3]}, same command): error {np.linalg.norm(t):.1f} mm, {r:.2f} deg")
            s = s2
    return arm.joints(), (s[0] if s else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="session name; log goes to data/raw/<name>_so101cam_log.json")
    ap.add_argument("--port", default="/dev/ttyACM0"); ap.add_argument("--id", default="so101_cam")
    ap.add_argument("--cam", default=ARM_CAM)
    ap.add_argument("--target", default=None, help="JSON with T_world_cam (4x4, metres); default = glove5 median head pose")
    ap.add_argument("--tol_mm", type=float, default=3.0); ap.add_argument("--tol_deg", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=8); ap.add_argument("--log_every", type=float, default=3.0)
    ap.add_argument("--release_cam", action="store_true", help="close the camera once parked (so the recorder can open it)")
    ap.add_argument("--hand-home", action="store_true", help="after a re-mount: torque off, point the camera at the tiles by hand, ENTER; corrects the model's base before the IK")
    a = ap.parse_args()

    T_target = np.array(json.load(open(a.target))["T_world_cam"]) if a.target else head_median_pose()
    json.dump({"T_world_cam": T_target.tolist()}, open(ROOT / "calib/head_pose_median.json", "w"), indent=1)
    print("target camera position (tile frame, mm):", np.round(T_target[:3, 3] * 1000), " view dir:", np.round(T_target[:3, 2], 2))
    fk = load_model(); arm = Arm(a.port, a.id); cam = ArmCam(a.cam)
    try:
        if a.hand_home:
            fk = hand_home(arm, cam, fk)
        qh, T_meas = closed_loop_park(arm, cam, fk, T_target, a.tol_mm, a.tol_deg, a.iters)
        print("parked; holding. joints (deg):", np.round(qh, 1), "| camera error vs target:",
              (f"{np.linalg.norm(pose_err(T_target, T_meas)[0]):.1f} mm, {pose_err(T_target, T_meas)[1]:.2f} deg" if T_meas is not None else "n/a"))
        log = {"name": a.name, "T_world_cam_target": T_target.tolist(), "joints_deg": {j: float(v) for j, v in zip(JOINTS, qh)},
               "T_world_cam_tiles": T_meas.tolist() if T_meas is not None else None, "samples": []}
        out = ROOT / f"data/raw/{a.name}_so101cam_log.json"
        if a.release_cam:
            cam.release(); print("camera released for the recorder")
        stop = {"flag": False}
        signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
        print("holding — Ctrl-C to park and exit")
        while not stop["flag"]:
            rec = {"t": time.time(), "joints_deg": arm.joints().tolist()}
            if not a.release_cam:
                s = cam.solve(); rec["T_world_cam_tiles"] = s[0].tolist() if s else None
            arm.command(qh)                                   # keep holding the parked joints
            log["samples"].append(rec); json.dump(log, open(out, "w"))
            time.sleep(a.log_every)
        print("wrote", out)
    finally:
        print("parking…"); arm.park_and_close()
        if not a.release_cam:
            cam.release()


if __name__ == "__main__":
    main()
