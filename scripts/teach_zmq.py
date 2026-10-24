#!/usr/bin/env python3
"""Record experiment poses by teleoperating the follower (yam_zmq_host running).

Subscribes to the host's observation stream and snapshots the follower's
measured joints each time you press Enter. Output is e_runner-compatible
e_poses.json (arm joints in radians; gripper stored as the host's normalized
0..1 value, noted in the file).

Usage:  python3 scripts/teach_zmq.py --n 8 [--obs_port 5602] [--out e2/e_poses.json]
Then:   stop the zmq host (it owns the CAN bus) and run e_runner modes.
"""
import argparse
import json
import os
import threading
import time

import zmq

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--obs_port", type=int, default=5602)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--out", default="e2/e_poses.json")
args = ap.parse_args()

ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.connect(f"tcp://{args.host}:{args.obs_port}")
sock.setsockopt(zmq.RCVTIMEO, 2000)

latest = {}
lock = threading.Lock()

def pump():
    global latest
    while True:
        try:
            msg = sock.recv_json()
        except zmq.Again:
            continue
        with lock:
            latest = msg

threading.Thread(target=pump, daemon=True).start()

ARM = [f"joint{i}.pos" for i in range(1, 7)]
time.sleep(1.0)
with lock:
    if not latest:
        print("waiting for observations from host...")
while True:
    with lock:
        if any(k in latest for k in ARM):
            break
    time.sleep(0.2)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
poses = []
print("Teleop the follower to each pose. Enter=record, q=finish.")
while len(poses) < args.n:
    s = input(f"[{len(poses)}/{args.n}] pose ready> ").strip()
    if s == "q":
        break
    with lock:
        obs = dict(latest)
    q = [obs.get(k) for k in ARM]
    if any(v is None for v in q):
        print("  missing joints in observation, keys:", sorted(obs)[:12])
        continue
    poses.append({"qpos": q + [obs.get("gripper.pos", 0.0)],
                  "gripper_units": "normalized_0_1",
                  "t_wall": time.time()})
    print(f"  recorded joints {[round(v, 3) for v in q]}")

with open(args.out, "w") as f:
    json.dump({"poses": poses, "source": "teach_zmq",
               "note": "arm joints rad; gripper normalized 0..1 (zmq host units)"},
              f, indent=2)
print(f"wrote {args.out} ({len(poses)} poses)")
