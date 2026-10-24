#!/usr/bin/env python3
"""Probe whether a UVC camera supports FIXED shutter + AUTO gain.

Procedure (controls latch only mid-stream, so everything happens while
frames flow):
  0. inventory every control whose name mentions gain/auto/exposure
  1. baseline: auto_exposure=1, exposure fixed, gain=0 — you vary the room
     light for ~20 s; brightness should track the light with NO adaptation
     (gain readback frozen at 0)
  2. for each candidate auto-gain control found: enable it and repeat.
     Verdict per second: gain readback moving BY ITSELF while
     exposure_time_absolute readback stays pinned = the combo works.
     Brightness alone is never trusted (ISP tone-mapping trap).

If step 2 finds no candidate control, the camera can't do it natively —
fall back to a slow software AGC (1 Hz v4l2-ctl gain nudges).

Usage: python scripts/agc_probe.py /dev/videoN [exposure=10] [phase_secs=20]
Follow up with a hand-wave blur test: blur width must NOT grow when the
light dims (proof the ISP isn't secretly stretching the shutter).
"""
import re
import subprocess
import sys
import time

import cv2
import numpy as np

DEV = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
EXP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SECS = int(sys.argv[3]) if len(sys.argv) > 3 else 20


def ctl_set(kv):
    r = subprocess.run(["v4l2-ctl", "-d", DEV, "-c", kv],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"  (cannot set {kv}: {r.stderr.strip()})")
    return r.returncode == 0


def ctl_get(name):
    r = subprocess.run(["v4l2-ctl", "-d", DEV, "-C", name],
                       capture_output=True, text=True)
    m = re.search(r":\s*(-?\d+)", r.stdout)
    return int(m.group(1)) if m else None


def watch(cap, label):
    print(f"\n== {label}")
    print("   VARY THE LIGHT NOW (dim it, brighten it, shade the scene)…")
    print(f"   {'t':>4} {'mean':>6} {'exposure':>9} {'gain':>6}")
    t0 = time.time()
    gains = []
    while time.time() - t0 < SECS:
        te = time.time() + 1.0
        img = None
        while time.time() < te:          # drain so readings are current
            ok, img = cap.read()
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        band = g[(g > np.percentile(g, 40)) & (g < 250)]
        e_rb, g_rb = ctl_get("exposure_time_absolute"), ctl_get("gain")
        gains.append(g_rb)
        print(f"   {time.time()-t0:4.0f} {band.mean() if band.size else 0:6.0f}"
              f" {e_rb!s:>9} {g_rb!s:>6}", flush=True)
    return gains


# ---- 0: control inventory ----------------------------------------------------
out = subprocess.run(["v4l2-ctl", "-d", DEV, "-L"],
                     capture_output=True, text=True).stdout
print(f"== controls on {DEV} mentioning gain/auto/exposure:")
names = []
for line in out.splitlines():
    if re.search(r"gain|auto|exposure", line, re.I):
        print("  " + line.rstrip())
        m = re.match(r"\s*(\w+)", line)
        if m:
            names.append(m.group(1))
auto_gain_cands = [n for n in names
                   if "gain" in n.lower() and "auto" in n.lower()]

# ---- stream + baseline -------------------------------------------------------
cap = cv2.VideoCapture(DEV)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FPS, 30)  # UNSET fps negotiates a broken mode on the fisheye (10s+ stalls) — always request one
for _ in range(10):
    cap.read()
ctl_set("auto_exposure=1")
ctl_set(f"exposure_time_absolute={EXP}")
ctl_set("gain=0")
ctl_set("backlight_compensation=0")
base_gains = watch(cap, f"BASELINE — manual e={EXP}, gain=0 "
                        "(brightness should track light, gain frozen)")

# ---- candidate auto-gain modes ----------------------------------------------
if not auto_gain_cands:
    print("\n== no dedicated auto-gain control found.")
    print("   Trying auto_exposure menu modes with a pinned shutter "
          "(some ISPs then auto-gain only):")
    for mode in (3, 2):  # aperture priority / shutter priority variants
        if not ctl_set(f"auto_exposure={mode}"):
            continue
        ctl_set(f"exposure_time_absolute={EXP}")
        gains = watch(cap, f"auto_exposure={mode} with exposure pinned to {EXP}")
        e_now = ctl_get("exposure_time_absolute")
        print(f"   -> exposure readback after: {e_now} "
              f"({'PINNED' if e_now == EXP else 'ISP TOOK OVER — unusable'})")
        if e_now == EXP and len(set(gains)) > 1:
            print("   -> gain moved by itself with exposure pinned: "
                  "combo WORKS in this mode")
    ctl_set("auto_exposure=1")
else:
    for c in auto_gain_cands:
        ctl_set(f"auto_exposure=1")
        ctl_set(f"exposure_time_absolute={EXP}")
        ctl_set(f"{c}=1")
        gains = watch(cap, f"{c}=1 with manual e={EXP}")
        e_now = ctl_get("exposure_time_absolute")
        moved = len(set(gains)) > 1
        print(f"   -> gain {'MOVED BY ITSELF' if moved else 'never moved'}, "
              f"exposure readback {e_now} "
              f"({'pinned' if e_now == EXP else 'CHANGED'})")
        print(f"   -> VERDICT: fixed-shutter+auto-gain "
              f"{'SUPPORTED' if moved and e_now == EXP else 'not proven'}")
        ctl_set(f"{c}=0")
cap.release()
print("\nDone. Final check regardless of verdict: hand-wave blur test — "
      "blur width must not grow when the light dims.")
