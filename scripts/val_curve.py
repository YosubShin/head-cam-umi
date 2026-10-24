#!/usr/bin/env python3
"""Val loss for one checkpoint, with train-parity data loading.

Builds the val DataLoader using lerobot's OWN resolve_delta_timestamps
(exactly what lerobot-train uses), then reuses the preprocessor-applying
compute_checkpoint_val_loss from aic. Val episodes: dataset
meta/val_episodes.json.

Usage: python scripts/val_curve.py <checkpoint>/pretrained_model
"""
import json
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_val_loss import (_decode_policy_config,  # noqa: E402
                              compute_checkpoint_val_loss)
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import (LeRobotDataset,  # noqa: E402
                                              LeRobotDatasetMetadata)

ck = Path(sys.argv[1])
cfg = json.loads((ck / "train_config.json").read_text())
# VAL_REPO=<repo> scores the checkpoint on ANOTHER dataset's val split (same
# feature keys/action space): e.g. a glove-trained policy on the robot val
# episodes, the distribution it must deploy into. Normalisation stays the
# checkpoint's own (as deployed).
repo = os.environ.get("VAL_REPO", cfg["dataset"]["repo_id"])
root = (str(Path.home() / ".cache/huggingface/lerobot" / repo)
        if "VAL_REPO" in os.environ
        else cfg["dataset"]["root"] or str(Path.home() / ".cache/huggingface/lerobot" / repo))
val_eps = json.loads((Path(root) / "meta/val_episodes.json").read_text())["val_episodes"]

pcfg = _decode_policy_config(ck)
meta = LeRobotDatasetMetadata(repo, root=root)
delta = resolve_delta_timestamps(pcfg, meta)
ds = LeRobotDataset(repo, root=root, episodes=val_eps,
                    delta_timestamps=delta, video_backend="pyav")
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
loss = compute_checkpoint_val_loss(checkpoint_dir=ck, val_loader=loader)
print(f"val_loss = {loss:.6f}")
