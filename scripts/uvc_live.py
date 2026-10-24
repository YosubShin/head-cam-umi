#!/usr/bin/env python3
"""Live UVC camera view with ArUco overlay (B0591 etc.) — aiming/focus tool.

Same as d405_live.py but for plain V4L2 cameras; display via ffplay
(headless OpenCV). Ctrl-C quits; s<Enter> saves a frame.

Usage: python scripts/uvc_live.py [/dev/video0]
Lock focus/exposure first, e.g.:
  v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0 -c focus_absolute=200
  v4l2-ctl -d /dev/video0 -c auto_exposure=1 -c exposure_time_absolute=30 -c gain=40
"""
import select
import subprocess
import sys

import cv2
import numpy as np


from uvc_ritual import apply as apply_ritual

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
cap = cv2.VideoCapture(dev)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 30)  # UNSET fps negotiates a broken mode on the fisheye (10s+ stalls) — always request one
if not cap.isOpened():
    sys.exit(f"cannot open {dev}")
for _ in range(5):
    cap.read()  # stream must be running before controls stick
apply_ritual(dev)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
p = cv2.aruco.DetectorParameters()
p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
det = cv2.aruco.ArucoDetector(d, p)

viewer = subprocess.Popen(
    ["ffplay", "-loglevel", "quiet", "-window_title", f"{dev} live",
     "-f", "rawvideo", "-pixel_format", "bgr24",
     "-video_size", f"{w}x{h}", "-i", "-"],
    stdin=subprocess.PIPE)
n_saved = 0
print(f"{dev} @ {w}x{h} — Ctrl-C quits, s<Enter> saves", flush=True)
try:
    while True:
        ok, img = cap.read()
        if not ok:
            continue
        corners, ids, _ = det.detectMarkers(img)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for i, c in zip(ids.ravel(), corners):
                cc = c.reshape(4, 2)
                e = np.mean([np.linalg.norm(cc[j] - cc[(j + 1) % 4]) for j in range(4)])
                cv2.putText(img, f"{int(i)}:{e:.0f}px",
                            (int(cc[:, 0].mean()), int(cc[:, 1].min()) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # sharpness of center crop (focus aid)
        g = cv2.cvtColor(img[h//3:2*h//3, w//3:2*w//3], cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(g, cv2.CV_64F).var()
        n = 0 if ids is None else len(ids)
        cv2.putText(img, f"{n} markers  sharp {sharp:.0f}", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        try:
            viewer.stdin.write(img.tobytes())
        except BrokenPipeError:
            break
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower().startswith("s"):
                fn = f"uvc_live_{n_saved:03d}.png"
                cv2.imwrite(fn, img)
                print("saved", fn, flush=True)
                n_saved += 1
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    try:
        viewer.stdin.close()
    except Exception:
        pass
    viewer.terminate()
