#!/usr/bin/env python3
"""Spot-check page for built LeRobot datasets: per episode, a composite video
of every observation stream side by side with the label series (joints +
gripper) drawn underneath and a moving cursor, plus an index.html.

Usage: spotcheck_dataset.py REPO [REPO ...] --episodes 0,10,28 --out data/sanity/spotcheck
The datasets live in ~/.cache/huggingface/lerobot (v3 layout: several
episodes per mp4, located by from/to timestamps).
"""
import argparse
import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HF = Path(os.path.expanduser("~/.cache/huggingface/lerobot"))
PANEL_H, TILE_H = 220, 384


def episode_frames(root, ep, key):
    """Decode one episode of one video stream (h264 re-encode via ffmpeg,
    since the stored AV1 may not decode in cv2)."""
    f = sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))
    meta = pd.concat([pd.read_parquet(x) for x in f]).set_index("episode_index").loc[ep]
    ci, fi = int(meta[f"videos/{key}/chunk_index"]), int(meta[f"videos/{key}/file_index"])
    t0, t1 = float(meta[f"videos/{key}/from_timestamp"]), float(meta[f"videos/{key}/to_timestamp"])
    src = root / f"videos/{key}/chunk-{ci:03d}/file-{fi:03d}.mp4"
    with tempfile.TemporaryDirectory() as td:
        seg = f"{td}/seg.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.4f}", "-to", f"{t1:.4f}",
                        "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        "-pix_fmt", "yuv420p", seg], check=True)
        cap = cv2.VideoCapture(seg)
        frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
        cap.release()
    return frames


def labels(root, ep):
    f = sorted(glob.glob(str(root / "data/*/*.parquet")))
    df = pd.concat([pd.read_parquet(x, columns=["episode_index", "frame_index", "action", "observation.state"]) for x in f])
    df = df[df.episode_index == ep].sort_values("frame_index")
    return np.stack(df["action"].values), np.stack(df["observation.state"].values)


def panel(acts, j, width, names):
    """Label series with a cursor at frame j."""
    n, d = acts.shape
    img = np.full((PANEL_H, width, 3), 255, np.uint8)
    cols = [(200, 60, 60), (60, 160, 60), (60, 60, 220), (160, 60, 160), (60, 160, 160), (160, 120, 0), (0, 0, 0)]
    lo, hi = acts.min(0), acts.max(0)
    xs = np.linspace(10, width - 10, n)
    for k in range(d):
        rng = max(hi[k] - lo[k], 1e-3)
        ys = PANEL_H - 15 - (acts[:, k] - lo[k]) / rng * (PANEL_H - 40)
        pts = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, cols[k % len(cols)], 2 if k == d - 1 else 1)
        cv2.putText(img, f"{names[k]} [{lo[k]:+.2f},{hi[k]:+.2f}]", (10 + (k % 4) * 300, 14 + (k // 4) * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, cols[k % len(cols)], 1)
    x = int(xs[min(j, n - 1)])
    cv2.line(img, (x, 30), (x, PANEL_H - 10), (0, 0, 0), 1)
    return img


# glove takes in build order: (episode-marks file, raw stem suffix) — used to
# pull RAW streams (e.g. the head cam, not in any dataset) by wall clock
GLOVE_TAKES = [("data/raw/yam_glove2_glove_episodes.json", "yam_glove2_{cam}.take2", range(0, 30)),
               ("data/raw/yam_glove2_glove_episodes.json", "yam_glove2_{cam}", range(30, 35)),
               ("data/raw/yam_glove3_glove_episodes.json", "yam_glove3_{cam}", range(0, 30)),
               ("data/raw/yam_glove4_glove_episodes.json", "yam_glove4_{cam}", range(0, 7))]


def episode_wall_map(root, fps=15):
    """dataset episode index -> (raw stem pattern, t_start, t_end). Marks are
    walked in build order; a mark whose duration does not match the dataset
    episode's length (+-3 frames) was skipped by the build's coverage gate."""
    f = sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))
    lens = pd.concat([pd.read_parquet(x, columns=["episode_index", "length"]) for x in f]).set_index("episode_index")["length"]
    out, di = {}, 0
    for marks_file, pat, rng in GLOVE_TAKES:
        marks = json.load(open(marks_file))["episodes"]
        for mi in rng:
            if di not in lens.index:
                break
            m = marks[mi]
            if abs(round((m["t_end"] - m["t_start"]) * fps) - int(lens[di])) <= 3:
                out[di] = (pat, m["t_start"], m["t_end"])
                di += 1
    return out


def raw_frames(stem, t0, n, fps=15, height=TILE_H):
    ts = np.array([x["wall"] for x in json.load(open(f"data/raw/{stem}_ts.json"))])
    cap = cv2.VideoCapture(f"data/raw/{stem}.mkv")
    frames = []
    for j in range(n):
        k = int(np.clip(np.searchsorted(ts, t0 + j / fps), 0, len(ts) - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = cap.read()
        if not ok:
            break
        h, w = im.shape[:2]
        frames.append(cv2.resize(im, (int(w * height / h), height), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def render(repos, ep, out_mp4, fps=15, raw_cams=()):
    """ep indexes the FIRST repo; other repos are aligned by wall clock (their
    coverage gates may have skipped different episodes)."""
    tiles, acts = [], None
    wm0 = episode_wall_map(HF / repos[0])
    for repo in repos:
        root = HF / repo
        info = json.load(open(root / "meta/info.json"))
        keys = [k for k in info["features"] if k.startswith("observation.images")]
        names = info["features"]["action"]["names"]
        ep_r = ep
        if repo != repos[0] and ep in wm0:
            wm = episode_wall_map(root)
            match = [i for i, (_, t0, _) in wm.items() if abs(t0 - wm0[ep][1]) < 0.5]
            if not match:
                continue
            ep_r = match[0]
        try:
            a, _ = labels(root, ep_r)
        except KeyError:
            continue
        if acts is None:
            acts = a
        for key in keys:
            fr = episode_frames(root, ep_r, key)
            if not fr:
                continue
            h, w = fr[0].shape[:2]
            s = TILE_H / h
            fr = [cv2.resize(x, (int(w * s), TILE_H)) for x in fr]
            for x in fr[:1]:
                pass
            tiles.append((f"{repo}:{key.split('.')[-1]} ep{ep_r}", fr))
    if acts is None or not tiles:
        return None
    n = min(len(acts), *[len(f) for _, f in tiles])
    if raw_cams:
        wm = episode_wall_map(HF / repos[0])
        if ep in wm:
            pat, t0, _ = wm[ep]
            for cam in raw_cams:
                fr = raw_frames(pat.format(cam=cam), t0, n)
                if fr:
                    tiles.append((f"raw:{cam}", fr))
                    n = min(n, len(fr))
    width = sum(f[0].shape[1] for _, f in tiles)
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{width}x{TILE_H + PANEL_H}", "-r", str(fps), "-i", "pipe:0",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                           out_mp4], stdin=subprocess.PIPE)
    for j in range(n):
        row = []
        for label, fr in tiles:
            im = fr[j].copy()
            cv2.putText(im, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            row.append(im)
        frame = np.vstack([np.hstack(row), panel(acts, j, width, names)])
        pr.stdin.write(frame.tobytes())
    pr.stdin.close()
    pr.wait()
    return n


ap = argparse.ArgumentParser()
ap.add_argument("repos", nargs="+")
ap.add_argument("--episodes", default="0,10,28,35,50")
ap.add_argument("--out", default="data/sanity/spotcheck")
ap.add_argument("--raw", default="", help="comma list of RAW camera suffixes to add by wall "
                                           "clock (e.g. b0587b for the head cam)")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rows = []
for ep in (int(x) for x in a.episodes.split(",")):
    mp4 = out / f"ep{ep:03d}.mp4"
    n = render(a.repos, ep, str(mp4), raw_cams=[c for c in a.raw.split(",") if c])
    if n:
        rows.append((ep, n))
        print(f"ep{ep}: {n} frames -> {mp4}", flush=True)
html = ["<html><body style='font-family:sans-serif'>", f"<h2>spot check: {' + '.join(a.repos)}</h2>",
        "<p>left to right: every observation stream of each dataset, then raw streams (head cam) aligned by wall clock; below: action series (joints + gripper, per-episode min/max) with a cursor.</p>"]
for ep, n in rows:
    html.append(f"<h3>episode {ep} ({n} frames, {n/15:.1f} s)</h3><video controls width='1400' src='ep{ep:03d}.mp4'></video>")
html.append("</body></html>")
(out / "index.html").write_text("\n".join(html))
print(f"index -> {out}/index.html")
