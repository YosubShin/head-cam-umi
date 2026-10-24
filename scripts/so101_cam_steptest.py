#!/usr/bin/env python3
"""Does the SO-101 camera arm actually follow a joint-position step under its own load?

From the hand-placed pose (torque off, you hold the arm roughly where it parks for tracking, ENTER),
each listed joint gets a +step then a -step goal; the measured position is read for --hold seconds
after each step. Reports the steady-state error (goal - measured) and the settle time per joint,
for the given servo P/I gains. Run twice to compare gains:

  python scripts/so101_cam_steptest.py --p 16 --i 0      # LeRobot's default follower gains
  python scripts/so101_cam_steptest.py --p 32 --i 8
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import so101_cam_park as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="/dev/ttyACM0"); ap.add_argument("--id", default="so101_cam")
    ap.add_argument("--joints", default="shoulder_lift,elbow_flex,wrist_flex,shoulder_pan")
    ap.add_argument("--step", type=float, default=10.0); ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--p", type=int, default=32); ap.add_argument("--i", type=int, default=0); ap.add_argument("--d", type=int, default=32)
    a = ap.parse_args()
    arm = P.Arm(a.port, a.id, p_gain=a.p, i_gain=a.i, d_gain=a.d)
    try:
        arm.robot.bus.disable_torque()
        input("Torque is OFF. Hold the arm about where it parks for tracking (camera over the table), then ENTER… ")
        q0 = arm.joints(); arm.robot.bus.enable_torque(); arm.command(q0); time.sleep(1.0)
        print(f"gains P={a.p} I={a.i} D={a.d}; start joints {np.round(q0, 1)}")
        print(f"{'joint':14s} {'dir':>4s} {'steady err':>11s} {'settle(s)':>10s} {'current mA':>11s}")
        for j in a.joints.split(","):
            k = P.JOINTS.index(j)
            for sgn in (+1, -1):
                q = arm.joints().copy(); goal = q.copy(); goal[k] += sgn * a.step
                arm.command(goal); t0 = time.time(); tr = []; settle = None
                while time.time() - t0 < a.hold:
                    m = arm.joints()[k]; tr.append((time.time() - t0, m))
                    if settle is None and abs(goal[k] - m) < 1.0:
                        settle = time.time() - t0
                    time.sleep(0.05)
                err = goal[k] - tr[-1][1]
                cur = arm.robot.bus.sync_read("Present_Current").get(j, float("nan")) * 6.5
                print(f"{j:14s} {('+' if sgn > 0 else '-'):>4s} {err:11.2f} {(f'{settle:.2f}' if settle is not None else 'never'):>10s} {cur:11.0f}")
            arm.command(q0); time.sleep(1.0)
    finally:
        arm.park_and_close()


if __name__ == "__main__":
    main()
