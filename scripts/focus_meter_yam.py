#!/usr/bin/env python3
"""Live sharpness meter for focusing a locally attached camera. Ctrl-C to stop.

Direct V4L2 capture at full resolution (much more sensitive than the 320x240
ZMQ stream), so run it BEFORE starting yam_zmq_host.py — one process owns the
camera at a time.

Put a textured target (printed text / checkerboard / a block) ~20 cm from the
lens and rotate the barrel until the number peaks; then aim at something >1 m
away and confirm it stays high. Center crop is weighted separately since the
fisheye periphery is always soft.

Usage: python focus_meter_yam.py [/dev/video0]
"""
import sys
import time

import cv2

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
cap = cv2.VideoCapture(dev)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
client = None
if cap.isOpened():
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{dev} @ {w}x{h} — rotate the lens barrel; Ctrl-C to stop")
else:
    # device busy — a running yam_zmq_host owns it; read its ZMQ stream instead
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from yam_zmq_client import YamZmqClient, YamZmqClientConfig
    client = YamZmqClient(YamZmqClientConfig(
        remote_ip="127.0.0.1", cameras={"wrist": (240, 320, 3)}))
    client.connect()
    w, h = 320, 240
    print(f"{dev} busy — reading 'wrist' via the running host ({w}x{h} JPEG; "
          "numbers run lower than direct capture) — Ctrl-C to stop")

best_c = 0.0
try:
    while True:
        if client is not None:
            frame = client.get_observation()["wrist"]
            ok = frame is not None and frame.any()
        else:
            ok, frame = cap.read()
        if not ok:
            print("\rframe grab failed", end="", flush=True)
            time.sleep(0.2)
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch, cw = h // 3, w // 3
        center = g[ch : 2 * ch, cw : 2 * cw]
        s_full = cv2.Laplacian(g, cv2.CV_64F).var()
        s_c = cv2.Laplacian(center, cv2.CV_64F).var()
        best_c = max(best_c, s_c)
        bar = "#" * min(60, int(s_c / 8))
        print(f"\rcenter {s_c:7.1f} (best {best_c:7.1f})  full {s_full:7.1f}  {bar}".ljust(110),
              end="", flush=True)
        time.sleep(0.15)
except KeyboardInterrupt:
    print()
finally:
    cap.release()
    if client is not None:
        client.disconnect()
