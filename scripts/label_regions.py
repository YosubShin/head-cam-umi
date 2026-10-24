#!/usr/bin/env python3
"""Where in the workspace are the fiducial labels worst?

Bj and Cf come from the same --emit-quad pass, so they are row-aligned frame
for frame: same episodes, same frames, same videos. The only difference is the
instrument behind the labels:

  Bj state = IK-of-fiducial joints  + tail-estimator gripper width
  Cf state = measured encoder joints + measured actuator width

So |Bj - Cf| per frame IS the label error, with no policy in the loop. Binned
by workspace position (FK of the GT joints) this answers "do the labels
degrade where the arm and gripper back occlude the tips?".

Usage: PYTHONPATH=~/co/i2rt python scripts/label_regions.py [--stride N]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

HF = Path.home() / ".cache/huggingface/lerobot"


def load(repo, stride):
    fs = sorted(glob.glob(f"{HF/repo}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df.iloc[::stride]
    return (np.stack(df["observation.state"].values),
            np.stack(df["action"].values),
            df["episode_index"].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--bins", type=int, default=3)
    args = ap.parse_args()

    s_bj, a_bj, ep_bj = load("pilot_all7_Bj", args.stride)
    s_cf, a_cf, ep_cf = load("pilot_all7_Cf", args.stride)
    assert len(s_bj) == len(s_cf) and (ep_bj == ep_cf).all(), \
        "Bj/Cf rows are not aligned — cannot compare labels"

    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import (ArmType, GripperType,
                                   combine_arm_and_gripper_xml)
    kin = Kinematics(combine_arm_and_gripper_xml(
        ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")

    def fk(q6):
        return np.array([kin.fk(q.astype(float))[:3, 3] for q in q6])

    p_bj, p_cf = fk(s_bj[:, :6]), fk(s_cf[:, :6])
    tip_mm = np.linalg.norm(p_bj - p_cf, axis=1) * 1000.0
    stroke = float(s_cf[:, 6].max() - s_cf[:, 6].min())
    grip_pct = np.abs(s_bj[:, 6] - s_cf[:, 6]) / stroke * 100.0
    grasp = a_cf[:, 6] < 0.5

    x, y = p_cf[:, 0], p_cf[:, 1]
    print(f"{len(tip_mm)} frames (stride {args.stride}), "
          f"{grasp.sum()} in grasp region; actuator stroke {stroke:.3f}")
    print(f"OVERALL   tip {np.median(tip_mm):6.1f} mm (p95 "
          f"{np.quantile(tip_mm, .95):6.1f})   grip "
          f"{np.median(grip_pct):5.1f}% of stroke (p95 "
          f"{np.quantile(grip_pct, .95):5.1f}%)")

    for axis, vals, lo_name, hi_name in (
            ("lateral y", y, "y-low = RIGHT", "y-high = LEFT"),
            ("reach x", x, "x-low = NEAR", "x-high = FAR")):
        edges = np.quantile(vals, np.linspace(0, 1, args.bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        print(f"\n--- by {axis}  ({lo_name} .. {hi_name}) ---")
        print(f"  {'region':22s} {'n':>6s} {'tip med':>8s} {'tip p95':>8s} "
              f"{'grip med':>9s} {'grip p95':>9s}")
        for tag, mask in (("all", np.ones_like(grasp)), ("grasp", grasp)):
            for i in range(args.bins):
                m = mask & (vals >= edges[i]) & (vals < edges[i + 1])
                if m.sum() == 0:
                    continue
                rng = (f"[{edges[i]:+.2f},{edges[i+1]:+.2f}]"
                       if np.isfinite(edges[i]) and np.isfinite(edges[i + 1])
                       else f"bin{i}")
                lab = f"{tag} {rng}"
                print(f"  {lab:22s} {m.sum():6d} {np.median(tip_mm[m]):8.1f} "
                      f"{np.quantile(tip_mm[m], .95):8.1f} "
                      f"{np.median(grip_pct[m]):9.1f} "
                      f"{np.quantile(grip_pct[m], .95):9.1f}")


if __name__ == "__main__":
    main()
