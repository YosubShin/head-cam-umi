#!/usr/bin/env python3
"""Spatial breakdown of action-prediction error over the val episodes.

Why not val_curve.py: that returns one scalar per checkpoint. To ask "is the
Bj label gap concentrated where the arm occludes the gripper tips?" we need
per-frame error attributed to a workspace location, split by channel (joints
vs gripper) and restricted to the grasp region.

Metric: mean |predicted - ground truth| over the whole predicted chunk, in
radians (gripper in actuator units). Deterministic — uses the policy's
inference path (ACT: zero latent), not the stochastic training loss, so
numbers are comparable across runs and interpretable physically.

Location: FK of the frame's own measured joints -> tool point in robot base
frame (needs i2rt on PYTHONPATH; it lives outside the lerobot env).

Usage:
  PYTHONPATH=~/co/i2rt python scripts/val_regions.py \
      train_runs/all7_Bj_act50/checkpoints/048000/pretrained_model [more...]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_val_loss import _decode_policy_config, _load_policy  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import (LeRobotDataset,  # noqa: E402
                                              LeRobotDatasetMetadata)

GRASP_MAX = 0.5      # action gripper below this = grasp intent (closing/closed)
N_BINS = 3           # lateral bins: right / centre / left


def fk_points(q6):
    """Tool-point xyz in base frame for an (N,6) array of joint vectors."""
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import (ArmType, GripperType,
                                   combine_arm_and_gripper_xml)
    kin = Kinematics(combine_arm_and_gripper_xml(
        ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site")
    return np.array([kin.fk(q.astype(float))[:3, 3] for q in q6])


def red_fractions(batch):
    """Per-frame visible area of the RED BLOCK in each observation stream.

    The policy/label error metrics say nothing about whether the OBJECT is
    visible — they only see fiducials and actions. The task block is red on a
    neutral table, so an HSV-free RGB mask is a direct, assumption-light
    measure of object visibility per view. Returned as fraction of pixels.
    """
    out = {}
    for k, v in batch.items():
        if not k.startswith("observation.images") or not torch.is_tensor(v):
            continue
        img = v.float()
        if img.ndim == 5:                    # (B, T, C, H, W) obs history
            img = img[:, -1]
        if img.shape[1] != 3:                # (B, H, W, C) -> channels first
            img = img.permute(0, 3, 1, 2)
        if img.max() > 1.5:
            img = img / 255.0
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        mask = (r > 0.30) & (r > 1.6 * g) & (r > 1.6 * b)
        out["vis_" + k.split(".")[-1]] = mask.float().mean(dim=(1, 2)).numpy()
    return out


def per_frame_errors(ck):
    cfg = json.loads((ck / "train_config.json").read_text())
    repo = cfg["dataset"]["repo_id"]
    root = cfg["dataset"]["root"] or str(
        Path.home() / ".cache/huggingface/lerobot" / repo)
    val_eps = json.loads(
        (Path(root) / "meta/val_episodes.json").read_text())["val_episodes"]
    pcfg = _decode_policy_config(ck)
    meta = LeRobotDatasetMetadata(repo, root=root)
    ds = LeRobotDataset(repo, root=root, episodes=val_eps,
                        delta_timestamps=resolve_delta_timestamps(pcfg, meta),
                        video_backend="pyav")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    policy = _load_policy(ck).to("cuda").eval()
    pre = None
    if (ck / "policy_preprocessor.json").exists():
        from lerobot.policies import make_pre_post_processors
        pre, _ = make_pre_post_processors(policy.config, str(ck))

    rows = []
    with torch.no_grad():
        for batch in loader:
            gt = batch["action"].clone()                    # (B, chunk, 7) raw
            state = batch["observation.state"].clone()      # (B, [hist,] 7)
            b = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v)
                     else v) for k, v in batch.items()}
            if pre is not None:
                b = pre(b)
            pred = policy.predict_action_chunk(b).float().cpu()
            # undo action normalization so errors are in physical units
            st = policy.config.normalization_mapping
            stats = meta.stats["action"]
            if str(st.get("ACTION", "")).endswith("MEAN_STD"):
                m = torch.tensor(stats["mean"], dtype=torch.float32)
                s = torch.tensor(stats["std"], dtype=torch.float32)
                pred = pred * s + m
            else:                                          # MIN_MAX -> [-1,1]
                lo = torch.tensor(stats["min"], dtype=torch.float32)
                hi = torch.tensor(stats["max"], dtype=torch.float32)
                pred = (pred + 1) / 2 * (hi - lo) + lo
            n = min(pred.shape[1], gt.shape[1])
            err = (pred[:, :n] - gt[:, :n]).abs()           # (B, n, 7)
            # padded chunk steps repeat the last action: including them
            # dilutes real differences, so mask exactly as training does
            if "action_is_pad" in batch:
                keep = (~batch["action_is_pad"][:, :n]).float().unsqueeze(-1)
            else:
                keep = torch.ones(err.shape[0], n, 1)
            err = err * keep
            denom = keep.sum(dim=1).clamp(min=1)             # (B,1) valid steps
            if state.ndim == 3:                             # obs history
                state = state[:, -1]
            h = min(15, n)     # executed horizon (1 s at 15 fps)
            keep_h = keep[:, :h]
            rows.append({
                "err_all": (err.sum(dim=(1, 2)) / (denom.squeeze(-1) * 7)).numpy(),
                "err_joints": (err[:, :, :6].sum(dim=(1, 2))
                               / (denom.squeeze(-1) * 6)).numpy(),
                "err_grip": (err[:, :, 6].sum(dim=1)
                             / denom.squeeze(-1)).numpy(),
                "err_h15": ((err[:, :h] * keep_h).sum(dim=(1, 2))
                            / (keep_h.sum(dim=1).clamp(min=1).squeeze(-1) * 7)
                            ).numpy(),
                "q": state[:, :6].numpy(),
                "g_state": state[:, 6].numpy(),
                "g_act": gt[:, 0, 6].numpy(),
                **red_fractions(batch),
            })
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    out["repo"] = repo
    return out


def report(name, d):
    xyz = fk_points(d["q"])
    y = xyz[:, 1]                       # lateral axis in base frame
    grasp = d["g_act"] < GRASP_MAX
    edges = np.quantile(y, np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    labels = ["y-low (right)", "y-mid", "y-high (left)"][:N_BINS]
    print(f"\n=== {name}  [{d['repo']}]  {len(y)} val frames, "
          f"{grasp.sum()} in grasp region")
    print(f"    overall  all {d['err_all'].mean():.4f}  "
          f"joints {d['err_joints'].mean():.4f} rad  "
          f"grip {d['err_grip'].mean():.4f}  "
          f"first-15-steps {d['err_h15'].mean():.4f}")
    print(f"    lateral y edges (m): "
          f"{[round(float(e), 3) for e in edges[1:-1]]}")
    hdr = (f"{'region':16s} {'n':>6s} {'all':>8s} {'joints':>8s} "
           f"{'grip':>8s} {'h15':>8s}")
    for tag, mask in (("ALL FRAMES", np.ones_like(grasp)),
                      ("GRASP REGION", grasp)):
        print(f"  -- {tag} --\n  {hdr}")
        for i, lab in enumerate(labels):
            m = mask & (y >= edges[i]) & (y < edges[i + 1])
            if m.sum() == 0:
                continue
            print(f"  {lab:16s} {m.sum():6d} {d['err_all'][m].mean():8.4f} "
                  f"{d['err_joints'][m].mean():8.4f} "
                  f"{d['err_grip'][m].mean():8.4f} "
                  f"{d['err_h15'][m].mean():8.4f}")
    # --- object visibility: is the block actually in view, and does error
    # --- track its disappearance?
    for key in sorted(k for k in d if k.startswith("vis_")):
        vis = d[key]
        gone = vis < 2e-4          # <0.02% of pixels = effectively not visible
        print(f"  -- OBJECT VISIBILITY [{key[4:]}] --")
        print(f"     block invisible in {100 * gone.mean():5.1f}% of frames "
              f"({100 * (gone & grasp).mean():.1f}% of all frames are "
              f"invisible AND in grasp); median visible area "
              f"{100 * np.median(vis):.3f}% of pixels")
        qs = np.quantile(vis[~gone], [0.33, 0.66]) if (~gone).sum() else [0, 0]
        buckets = [("invisible", gone),
                   ("small", ~gone & (vis <= qs[0])),
                   ("medium", ~gone & (vis > qs[0]) & (vis <= qs[1])),
                   ("large", ~gone & (vis > qs[1]))]
        print(f"     {'bucket':10s} {'n':>6s} {'err all':>8s} {'err h15':>8s}"
              f" {'grasp n':>8s} {'grasp err':>10s}")
        for name, m in buckets:
            if m.sum() == 0:
                continue
            gm = m & grasp
            ge = f"{d['err_all'][gm].mean():10.4f}" if gm.sum() else f"{'-':>10}"
            print(f"     {name:10s} {m.sum():6d} {d['err_all'][m].mean():8.4f} "
                  f"{d['err_h15'][m].mean():8.4f} {gm.sum():8d} {ge}")
    return {"y": y, "grasp": grasp, "edges": edges, **d}


if __name__ == "__main__":
    res = {}
    for arg in sys.argv[1:]:
        ck = Path(arg)
        # .../train_runs/<run>/checkpoints/<step>/pretrained_model
        name = "/".join(ck.parts[-4:-2]) if len(ck.parts) >= 4 else str(ck)
        res[name] = report(name, per_frame_errors(ck))
    if len(res) == 2:
        (na, a), (nb, b) = res.items()
        print(f"\n=== per-region ratio {nb} / {na} "
              f"(>1 means {nb} worse) ===")
        e = a["edges"]
        for i, lab in enumerate(["y-low (right)", "y-mid", "y-high (left)"]):
            for tag in ("all frames", "grasp"):
                ma = (a["y"] >= e[i]) & (a["y"] < e[i + 1])
                mb = (b["y"] >= e[i]) & (b["y"] < e[i + 1])
                if tag == "grasp":
                    ma, mb = ma & a["grasp"], mb & b["grasp"]
                if ma.sum() == 0 or mb.sum() == 0:
                    continue
                r = b["err_all"][mb].mean() / a["err_all"][ma].mean()
                rg = b["err_grip"][mb].mean() / a["err_grip"][ma].mean()
                print(f"  {lab:16s} {tag:11s} all x{r:.2f}  grip x{rg:.2f}")
