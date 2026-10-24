#!/usr/bin/env python3
"""Rapid exposure sweep for streak comparison: one capture session, values
switched live, ~0.4s apart — spin a toy in view and run this; all frames see
nearly the same speed.

Usage: python scripts/exposure_sweep.py [/dev/video0] [steps]
Steps are exposure values, optionally with a per-step gain as e:g —
e.g. "20:0,20:30,20:50,20:80" sweeps gain at fixed exposure, and
"1:80,5:60,20:50,50:20" pairs each exposure with a compensating gain.
Bare numbers keep the previous gain (initial 0).
Saves /tmp/sweep_e<exp>_g<gain>.png and prints brightness + flat-patch noise.
"""
import subprocess
import sys
import time

import cv2
import numpy as np

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
steps = []
g_cur = 0
for tok in (sys.argv[2] if len(sys.argv) > 2 else "1,5,10,20,50,100,200").split(","):
    if ":" in tok:
        e_, g_cur = (int(x) for x in tok.split(":"))
    else:
        e_ = int(tok)
    steps.append((e_, g_cur))

def ctl(**kw):
    subprocess.run(["v4l2-ctl", "-d", dev] +
                   sum((["-c", f"{k}={v}"] for k, v in kw.items()), []),
                   check=True)

ctl(auto_exposure=1, gain=0, sharpness=0)
cap = cv2.VideoCapture(dev)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 30)  # UNSET fps negotiates a broken mode on the fisheye (10s+ stalls) — always request one
for _ in range(10):
    cap.read()  # warm up

print("sweeping — keep the spinner going…", flush=True)
t0 = time.time()
for e, gn in steps:
    ctl(exposure_time_absolute=e, gain=gn)
    for _ in range(8):      # flush buffered frames so the new setting lands
        cap.read()
    ok, img = cap.read()
    if not ok:
        print(f"e={e} g={gn}: grab failed"); continue
    cv2.imwrite(f"/tmp/sweep_e{e}_g{gn}.png", img)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # noise proxy: std of high-pass residual in the brightest flat region
    blur = cv2.GaussianBlur(g, (0, 0), 3)
    hp = g.astype(float) - blur
    flat = (g > np.percentile(g, 60)) & (g < 250)  # bright but unclipped
    print(f"e={e:>4} g={gn:>3}: mean {g.mean():5.0f}  "
          f"flat-patch noise {hp[flat].std():5.2f}  t={time.time()-t0:.1f}s",
          flush=True)
cap.release()
print("done — frames at /tmp/sweep_e*.png")
