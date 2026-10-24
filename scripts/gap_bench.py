#!/usr/bin/env python3
"""Bench fit: jaw GAP (caliper, mm) -> gripper ACTUATOR units, on the robot.

The glove has no actuator, so its width labels come from caliper holds. For
those labels to share units with the robot-collected corpus (Bj's gripper
channel is the actuator reading, ~0.2 closed .. ~1.0 open), we need the
robot's own gap<->actuator relation once. Tip-marker distances can't give it
directly (marker centres sit an unknown offset from the jaw faces), so measure
it: hold the follower at 3-5 apertures via the leader, caliper the gap between
the jaw faces, and this script pairs each with the live actuator readback.

Run with the ZMQ host up (teleop mode is fine):
  python scripts/gap_bench.py            # prompts per hold, writes the fit
Output: calib/gap_to_actuator.json  {"a", "b", "rms_actuator", "points"}
        actuator = a * gap_mm + b
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yam_zmq_client import YamZmqClient, YamZmqClientConfig  # noqa: E402

PILOT = Path(__file__).resolve().parent.parent
OUT = PILOT / "calib/gap_to_actuator.json"


def main():
    robot = YamZmqClient(YamZmqClientConfig(remote_ip="127.0.0.1",
                                            zmq_cmd_port=5601,
                                            zmq_obs_port=5602, cameras={}))
    robot.connect()
    time.sleep(0.5)
    pts = []
    print("Set an aperture with the leader, caliper the gap between the jaw "
          "FACES, type it in mm. Blank line to finish (need >= 3).")
    while True:
        s = input(f"[{len(pts) + 1}] gap mm (blank = done): ").strip()
        if not s:
            break
        try:
            mm = float(s)
        except ValueError:
            print("   number please")
            continue
        vals = []
        for _ in range(20):                       # 1 s of readback, averaged
            vals.append(float(robot.get_observation()["gripper.pos"]))
            time.sleep(0.05)
        a = float(np.mean(vals))
        print(f"   actuator {a:.4f} (sd {np.std(vals):.4f})")
        if np.std(vals) > 0.01:
            print("   !! gripper not still — hold steady and re-enter")
            continue
        pts.append((mm, a))
    if len(pts) < 3:
        raise SystemExit("need at least 3 points")
    mm = np.array([p[0] for p in pts])
    ac = np.array([p[1] for p in pts])
    a, b = np.polyfit(mm, ac, 1)
    res = ac - (a * mm + b)
    fit = {"a": float(a), "b": float(b), "rms_actuator": float(res.std()),
           "points": [{"gap_mm": float(m), "actuator": float(x)}
                      for m, x in pts],
           "note": "actuator = a*gap_mm + b; jaw-FACE gap by caliper; "
                   f"measured {time.strftime('%Y-%m-%d')}"}
    OUT.write_text(json.dumps(fit, indent=1))
    print(f"\nactuator = {a:.5f} * gap_mm + {b:.4f}   "
          f"(rms {res.std():.4f} actuator units over {len(pts)} points)"
          f"\nwrote {OUT}")
    if abs(res).max() > 0.03:
        print("!! a point deviates > 0.03 — the relation may be nonlinear "
              "or a caliper reading is off; consider re-measuring")


if __name__ == "__main__":
    main()
