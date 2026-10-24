#!/bin/bash
# Everything after the recording, in one go: detect -> export -> fk -> solve.
# Usage: scripts/handeye_run.sh <dataset> [out_json]
#   dataset   the --dataset you gave record_session.py (robot mode)
#   out_json  default calib/handeye_<dataset>.json
# Then compare against the previous hand-eye before switching builds to it.
set -euo pipefail
cd "$(dirname "$0")/.."
DS=${1:?dataset name}
OUT=${2:-calib/handeye_${DS}.json}
P=~/miniconda3/envs/lerobot_alohamini/bin/python
STEM=${DS}_b0587
[ -f data/raw/$STEM.mkv ] || { echo "no data/raw/$STEM.mkv (recording not finalised?)"; exit 1; }
[ -f calib/${STEM}_dodeca_detections.npz ] || $P scripts/detect_parallel.py data/raw/$STEM.mkv
$P scripts/handeye.py export $STEM $DS
~/co/i2rt/.venv/bin/python scripts/handeye.py fk
HANDEYE_OUT=$OUT $P scripts/handeye.py solve
$P - "$OUT" <<'PY'
import json, sys, numpy as np
new = json.load(open(sys.argv[1])); old = json.load(open("calib/handeye_p5cur.json"))
A, B = np.array(new["T_cam_base"]), np.array(old["T_cam_base"])
dt = np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000
R = A[:3, :3] @ B[:3, :3].T; ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
print(f"\nvs handeye_p5cur: forward cam moved {dt:.1f} mm / {ang:.2f} deg; "
      f"new fit residual pos {new['pos_median_mm']:.1f} mm rot {new['rot_median_deg']:.2f} deg on {new['n_pairs']} pairs "
      f"(p5cur was {old['pos_median_mm']:.1f} mm / {old['rot_median_deg']:.2f} deg)")
PY
