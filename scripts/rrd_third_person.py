#!/usr/bin/env python3
"""Cut third-person clips of grid trials out of a saved Rerun recording.

eval_grid.py streams the forward camera to the Rerun viewer as 'live/forward'
(~6 Hz, half resolution). If the viewer's recording was saved (.rrd), this
matches those frames to trials.jsonl by wall clock and writes one mp4 per trial
into <out_dir>/third_person/<policy>_S<scene>r<rep>_<outcome>.mp4.

  python rrd_third_person.py rollout_logs/grid_final /mnt/nvme/data.rrd [more.rrd ...]
"""
import datetime as dt
import json
import os
import sys
import zoneinfo

import cv2
import numpy as np
from rerun.experimental import RrdReader

out_dir, rrds = sys.argv[1], sys.argv[2:]
out = os.path.join(out_dir, "third_person"); os.makedirs(out, exist_ok=True)
tz = zoneinfo.ZoneInfo("America/Los_Angeles")
trials = []
for line in open(os.path.join(out_dir, "trials.jsonl")):
    r = json.loads(line)
    t0 = dt.datetime.strptime(r["stamp"], "%Y%m%d_%H%M%S").replace(tzinfo=tz).timestamp()
    trials.append((t0, t0 + r["elapsed_s"], r))
frames = []
for p in rrds:
    rd = RrdReader(p)
    for e in rd.recordings():
        for ch in rd.stream(store=e).filter(content="/live/forward"):
            rb = ch.to_record_batch()
            for t, b in zip(rb.column("log_time").to_pylist(), rb.column("EncodedImage:blob")):
                frames.append((t.timestamp(), np.asarray(b.as_py()[0], dtype=np.uint8)))
frames.sort(key=lambda x: x[0])
ft = np.array([f[0] for f in frames])
print(f"{len(frames)} forward frames from {len(rrds)} file(s)")
n = 0
for t0, t1, r in trials:
    i0, i1 = np.searchsorted(ft, t0 - 0.5), np.searchsorted(ft, t1 + 0.5)
    if i1 - i0 < 10:
        continue
    sel = frames[i0:i1]
    fps = max(1.0, (len(sel) - 1) / max(sel[-1][0] - sel[0][0], 1e-3))
    im0 = cv2.imdecode(sel[0][1], cv2.IMREAD_COLOR); h, w = im0.shape[:2]
    name = f"{r['policy']}_S{r['scene']}r{r['rep']}_{r['outcome']}.mp4"
    vw = cv2.VideoWriter(os.path.join(out, name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _, b in sel:
        vw.write(cv2.imdecode(b, cv2.IMREAD_COLOR))
    vw.release(); n += 1
    print(f"  {name:48s} {len(sel):4d} frames  {fps:.1f} fps  {t1 - t0:.0f}s")
print(f"clips written: {n} of {len(trials)} trials")
