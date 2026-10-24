#!/usr/bin/env python3
"""Print per-episode seek times inside a session's raw mkvs.

Maps each saved episode's wall-clock window (skeleton sidecar) to video
timestamps in every camera stem of that session, so you can jump straight
to an episode in any player.

Usage: python scripts/episode_times.py <dataset> [stem ...]
  e.g. python scripts/episode_times.py yam_sess1
Stems default to every data/raw/<dataset>_*.mkv found.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
dataset = sys.argv[1]
stems = sys.argv[2:] or sorted(
    os.path.basename(p)[:-4]
    for p in glob.glob(f"data/raw/{dataset}_*.mkv"))
walls = os.path.expanduser(
    f"~/.cache/huggingface/lerobot/{dataset}/meta/episode_wallclock.jsonl")
eps = [json.loads(line) for line in open(walls)]

for stem in stems:
    ts = np.array([e["wall"]
                   for e in json.load(open(f"data/raw/{stem}_ts.json"))])
    pts = np.array([float(x) for x in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         f"data/raw/{stem}.mkv"],
        capture_output=True, text=True).stdout.split()])
    n = min(len(ts), len(pts))
    print(f"\n{stem}.mkv:")
    for ep in eps:
        k0 = min(int(np.searchsorted(ts, ep["start_unix"])), n - 1)
        k1 = min(int(np.searchsorted(ts, ep["end_unix"])), n - 1)
        t0, t1 = pts[k0], pts[k1]
        print(f"  ep{ep['episode_index']}: {int(t0//60)}:{t0%60:05.2f} -> "
              f"{int(t1//60)}:{t1%60:05.2f}   "
              f"(ffplay -ss {t0:.0f} data/raw/{stem}.mkv)")
