#!/usr/bin/env python3
"""Summarise a placement-grid rollout log (eval_grid.py trials.jsonl).

Per policy: successes, Wilson 95% CI, time-to-success (policy time = ticks/fps,
immune to control-loop jitter), failure modes. Then the paired picture, which
is what the shared-scene design buys: per-scene wins/ties/losses against a
reference arm, an exact McNemar test on scene-rep pairs, and success by
table region from the saved block placements.

  python grid_analysis.py rollout_logs/grid_final --ref Aj
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

import numpy as np

SHORT = {"no_grasp": "n", "dropped_before_bin": "d", "dropped_at_bin": "b", "timeout": "t"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b (A wins) and c (B wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--ref", default="Aj")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="drop trials carrying camera_moved=true (the forward arm's bumped-camera scenes)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    rs = [json.loads(l) for l in open(out / "trials.jsonl")]
    if args.exclude_flagged:
        rs = [r for r in rs if not r.get("camera_moved")]
    names = list(dict.fromkeys(r["policy"] for r in rs))
    if args.ref in names:
        names.remove(args.ref); names.insert(0, args.ref)
    ok = lambda r: r["outcome"] == "success"  # noqa: E731

    print(f"{len(rs)} trials, {len(set(r['scene'] for r in rs))} scenes, policies: {', '.join(names)}")
    flagged = sum(1 for r in rs if r.get("camera_moved"))
    if flagged:
        print(f"  ({flagged} trials flagged camera_moved; rerun with --exclude-flagged to drop them)")
    print("\n== per policy ==")
    print(f"{'policy':14s} {'succ':>7s} {'rate':>5s}  {'95% CI':>11s}  {'tts med [IQR] s':>18s}   fails")
    for n in names:
        x = [r for r in rs if r["policy"] == n]
        s = [r for r in x if ok(r)]
        lo, hi = wilson(len(s), len(x))
        tts = np.array([r["ticks"] / args.fps for r in s])
        t = (f"{np.median(tts):5.1f} [{np.percentile(tts, 25):.1f}-{np.percentile(tts, 75):.1f}]"
             if len(tts) else "    -")
        fails = defaultdict(int)
        for r in x:
            if not ok(r):
                fails[SHORT[r["outcome"]]] += 1
        print(f"{n:14s} {len(s):3d}/{len(x):<3d} {100 * len(s) / max(len(x), 1):4.0f}%  "
              f"[{100 * lo:3.0f}-{100 * hi:3.0f}]%  {t:>18s}   "
              + " ".join(f"{k}{v}" for k, v in sorted(fails.items())))

    print(f"\n== paired against {args.ref} (same scene & rep) ==")
    print(f"{'policy':14s} {'wins':>4s} {'ties':>4s} {'loss':>4s}  {'diff':>6s}  McNemar p   scenes won/lost (per-scene rate)")
    key = lambda r: (r["scene"], r["rep"])  # noqa: E731
    ref = {key(r): ok(r) for r in rs if r["policy"] == args.ref}
    for n in names[1:]:
        mine = {key(r): ok(r) for r in rs if r["policy"] == n}
        common = sorted(set(ref) & set(mine))
        b = sum(mine[k] and not ref[k] for k in common)   # n wins
        c = sum(ref[k] and not mine[k] for k in common)   # ref wins
        ties = len(common) - b - c
        diff = (sum(mine[k] for k in common) - sum(ref[k] for k in common)) / max(len(common), 1)
        # per-scene rate comparison
        sc = defaultdict(lambda: [0, 0, 0])
        for k in common:
            sc[k[0]][0] += mine[k]; sc[k[0]][1] += ref[k]; sc[k[0]][2] += 1
        sw = sum(v[0] > v[1] for v in sc.values()); sl = sum(v[0] < v[1] for v in sc.values())
        print(f"{n:14s} {b:4d} {ties:4d} {c:4d}  {100 * diff:+5.0f}%  {mcnemar_exact(b, c):8.3f}   {sw}/{sl} of {len(sc)}")

    pl_p = out / "placements.json"
    if pl_p.exists():
        pl = {int(k): v for k, v in json.load(open(pl_p)).items()}
        xs = np.array([v[0] for v in pl.values()])
        cut = np.median(xs)
        print(f"\n== by table region (forward-frame x; split at median {cut:.0f} px: near/left vs far/right) ==")
        for n in names:
            parts = []
            for lab, sel in (("x<med", lambda x: x < cut), ("x>=med", lambda x: x >= cut)):
                x = [r for r in rs if r["policy"] == n and r["scene"] in pl and sel(pl[r["scene"]][0])]
                parts.append(f"{lab} {sum(map(ok, x)):2d}/{len(x):<2d}")
            print(f"{n:14s} " + "   ".join(parts))

    print("\n== scene table (S success, n/d/b/t failures; reps joined by /) ==")
    scenes = sorted(set(r["scene"] for r in rs))
    print("scene " + " ".join(f"{n[:10]:>10s}" for n in names))
    for s in scenes:
        row = []
        for n in names:
            o = [r for r in sorted((r for r in rs if r["scene"] == s and r["policy"] == n), key=lambda r: r["rep"])]
            row.append("/".join("S" if ok(r) else SHORT[r["outcome"]] for r in o)
                       + ("*" if any(r.get("camera_moved") for r in o) else ""))
        print(f"{s:5d} " + " ".join(f"{v:>10s}" for v in row))
    print("(* = camera_moved)")


if __name__ == "__main__":
    main()
