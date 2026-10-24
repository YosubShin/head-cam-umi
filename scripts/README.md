# Software: calibrate, record, solve, build, deploy

The Python side of the system, collection kit and robot side. Everything runs from the
**repository root** (`python scripts/...`) and reads/writes `calib/` and
`data/raw/` relative to it — keep that layout.

Status: the robot-side path (same code, robot-mounted markers) is validated
against robot forward kinematics — 7.7 mm / 1.7° median, see
[yam-umi/pos-tracking/dynamic-accuracy.md](https://github.com/YosubShin/yam-umi/blob/main/pos-tracking/dynamic-accuracy.md).
The glove-session path (steps 4–6) has been run end to end on four glove
sessions (Oct 2026, ~70 episodes): recorded, solved, built into training
datasets and trained. The gotchas below are the ones it surfaced.

## 0. Install

```
pip install -r scripts/requirements.txt      # opencv-contrib, numpy, scipy
sudo apt install ffmpeg v4l-utils            # ffmpeg/ffprobe + v4l2-ctl
```
Linux with the cameras on USB (UVC). Tested with Python 3.12.

## 1. Measurements — one caliper reading per printed group

Printers rescale by a few percent and the error does not announce itself:
a 3% size error is a 3% range error in every pose. **Caliper one black square
per group** and declare it in the config. Nominal is not enough.

| what | file | fields |
|---|---|---|
| glove markers (sheet `markers/glove_markers_v4.pdf`) | `calib/marker_map_glove_<unit>.json` | `tails`, `bases`, `tips` sizes in mm; `ball_edge_mm` |
| workspace tags (`markers/apriltag_36h11_60mm_marker_75mm_tile_final.pdf`) | `calib/tags36h11_<unit>.json` | `edge_mm`; **use tags 1–5 only** |
| ChArUco board (`markers/charuco_letter_8.5x11.pdf`) | flags to `calibrate_uvc.py` | `--square-mm`, `--marker-mm` |

Two built units are recorded, both from the same sheet and both complete:
`calib/marker_map_glove_{yuan,yosub}.json` with `calib/tags36h11_{yuan,yosub}.json`.
They differ by ~1% in every group — the normal print-to-print variance — which
is why a new unit always gets its **own** file and its own bundle solves
(step 6): copy one of them, replace every size with your calipers, rename.

Two ID traps, already handled if you follow the commands below:
`4x4-27` is a real ball face but aliases to AprilTag `16h5-1` (a robot tip
marker) — detect glove footage with `--keep-27`; workspace tag `36h11-0`
aliases to `4x4-37` (a glove tail) — never place tag 0.

## 2. Cameras: pin by USB port, then lock focus and exposure

Copy `calib/cameras.example.json` to `calib/cameras.json` and edit it for
your machine. Each entry is `[suffix, by_path, name_fragment_or_null,
"WxH", "fps"]`; find by-path links with `ls -l /dev/v4l/by-path/`. Suffixes
the pipeline understands: `b0587` (fixed forward 4K), `b0587b` (second
B0587 — fixed twin or head-worn), `fish` (wrist fisheye). The two B0587s
share a serial, so `b0587b` has no name fallback; if its port changes the
recorder resolves it as "the B0587 that is not the forward one".

Focus is mechanical on the B0587 and intrinsics are focus-dependent:
lock the barrel with the live sharpness meter, mark it, and only then
calibrate.
```
python scripts/focus_tune.py /dev/v4l/by-path/<forward>
python scripts/focus_tune.py /dev/v4l/by-path/<head> --ref /dev/v4l/by-path/<forward>
```
It opens a window: the whole frame on top and a 1:1 crop of actual sensor
pixels below (the crop follows the largest decoded marker; `n` in the
terminal steps to the next one), so you see the blur you are turning out. Aim at a marker (a workspace tile at ~1 m is ideal) and minimise
**edge px**, the black/white transition width on the marker's edges —
2–3 px is a sharp B0587 at 4K; the "sharp" number is a scene-dependent
Laplacian meter with peak-hold, useful only against itself while you turn.
`--ref` puts a known-good camera beside yours on the same target, which is
the only honest "as sharp as the forward cam" check. It matters more than it
looks: a head-worn B0587 at 7 px edges decoded a second ball face in 5% of
frames (no pose); the same unit at 4 px decoded two faces in 92% and fused a
pose in 99% — a calibration RMS of 0.6 px had said nothing about it. The meter captures at
the recording size (4K): a 1080p preview hides half the blur.
The exposure/gain ritual (`scripts/uvc_ritual.py`) is applied automatically
by the recorder and the calibrator: manual exposure, `exposure_time_absolute=2`
(0.2 ms), gain 0. Verify behaviourally — a hand waved in front of the lens
must be frozen, not smeared; readback alone lies on these modules.

## 3. Intrinsics (once per camera, or after any barrel movement)

Wave the ChArUco board through the view — near/far, corners, tilted — until
it has collected enough frames; it calibrates, validates on a holdout and
writes `calib/<name>_intrinsics.json`, naming the camera from its device
name (B0587 → `B0587_intrinsics.json`; pass `--name B0587b` for the second
unit). Existing files are backed up, never overwritten.
```
python scripts/calibrate_uvc.py /dev/v4l/by-path/<forward> --square-mm 25.0 --marker-mm <caliper>
python scripts/calibrate_uvc.py /dev/v4l/by-path/<second>  --name B0587b --square-mm 25.0 --marker-mm <caliper>
python scripts/calibrate_uvc.py /dev/v4l/by-path/<fish>    --model rational --square-mm 25.0 --marker-mm <caliper>
```
Expect ~0.7 px RMS at 4K on the holdout; the 1080p fisheye lands near 2.7 px
with `--model rational` (required for it). **Calibrate the glove's wrist camera
under its own `--name`** (e.g. `--name fisheye_glove_<unit>`): the wrist-observation
arm reads gripper width from the 16h5 tip markers that camera sees, which is
metric PnP, and the auto-derived name `fisheye` would overwrite another unit's
file. Only a wrist stream used purely as observations can skip calibration.

## 4. Record a glove session

No robot, no host. Cameras roll continuously; you mark episode boundaries.
Hold the glove in the workspace when it starts — the pre-flight decodes a
live frame from every camera and aborts on zero markers (lens smudge,
exposure not latched, wrong device).
```
python scripts/record_session.py --glove --dataset glove_yuan_01 --num_episodes 30 --task "<task>"
```
`ENTER` starts an episode, `ENTER` ends it, `d` discards the one you just
did, `q` finishes. Outputs in `data/raw/`: `<dataset>_<cam>.mkv` + `_ts.json`
per camera and `<dataset>_glove_episodes.json` (boundaries).

**Width holds — once per unit, not per session.** The first session on a new
glove unit (and any session after you move or reprint the tip/back markers)
runs with `--width_holds`. At the first prompt, clamp the jaws on a rigid
**spacer** of caliper-measured width (the gap must not change during the
hold; fingers can't hold an exact opening while you move, a spacer can),
type the value (`37.2`, or `m 37.2`) and ENTER; the recorder logs that gap
until you press ENTER again. During the ~15–20 s hold rotate the whole glove
slowly so the forward camera sees **both jaws'** tail cubes and the base
markers from a few angles — every marker face only gets a width estimator if
it was seen at two or more holds, and a one-sided hold leaves the other jaw
blind. Five apertures (closed, ~20, ~40, ~60, ~80 mm) beat three: the open
end extrapolates badly otherwise. The wrist camera must decode **both** tip
markers at every hold (check with `tag_contrast.py` first). Holds land in
`<dataset>_glove_log.json`, which every later build of that unit reuses via
`--log`. Without the flag the prompt never asks. Caliper values are logged
as raw mm unless `calib/gap_to_actuator.json` exists (that file comes from a
robot bench measurement and maps mm to the robot's gripper units — we
supply it).

Also record one **calibration take** per unit (`--num_episodes 1`): a single
long episode slowly rotating the ball through every orientation at working
distance, with the gripper-back markers visible part of the time. Step 6
solves the ball's geometry from it.

## 4b. Collecting for the pilot on another machine

The hand-off sequence — camera mapping, calibration, geometry, holds,
recording, detection, a one-command session check, camera→tiles, and what
to ship — is in [`docs/pilot-task.md`](../docs/pilot-task.md), with the task
definition, reference frames and the tile layout that puts your labels in
the same frame as ours.

## 5. Detection

```
python scripts/detect_parallel.py data/raw/<dataset>_b0587.mkv --keep-27     # and _b0587b, _fish
```
Writes `calib/<stem>_dodeca_detections.npz` (ArUco 4x4) and
`calib/<stem>_tips16h5_detections.npz`. Uses all cores; `--workers N` to
limit.

## 6. Solve the unit's geometry (once per physical glove)

Face-to-ID layout on the ball is not prescribed; it is recovered from the
calibration take. Three passes, outputs are specific to the physical unit:
```
python scripts/bundle_dodeca.py --npz=calib/glove_cal_b0587_dodeca_detections.npz \
    --intr=calib/B0587_intrinsics.json --marker-mm=<ball caliper> --ball-ids=25-35 --max-vel=600 --out=calib/tmp_bundle_<unit>.json
python scripts/bundle_dodeca_ideal.py --npz=calib/glove_cal_b0587_dodeca_detections.npz \
    --intr=calib/B0587_intrinsics.json --init=calib/tmp_bundle_<unit>.json --marker-mm=<ball caliper> \
    --max-vel=600 --out=calib/glove_<unit>_dodeca_bundle.json
python scripts/calibrate_gripper_bundle.py glove_cal_b0587 --bundle=calib/glove_<unit>_dodeca_bundle.json \
    --marker-map=calib/marker_map_glove_<unit>.json --out=calib/glove_<unit>_gripper_bundle.json
```
`--max-vel=600` admits moving frames: the default 150 px/s stillness gate
was tuned for a slow-shutter camera, and on a rotation take at the 0.2 ms
shutter it discards most frames and drops the rarely-shown faces from the
bundle entirely (a first solve at the default kept 9/11 faces). Expect all
11 faces at ~0.5 px RMS — a built glove unit solved at 0.52 px over 597
frames; the robot's ball, included as a format example
(`calib/*_robot.example.json`), sits at 0.71 px on 465 frames.

## What to send back for dataset building

`data/raw/<dataset>_*` (all cameras, mkv + ts.json), the two
`_glove_*.json` sidecars, `calib/<name>_intrinsics.json` for each camera,
`calib/marker_map_glove_<unit>.json`, `calib/tags36h11_<unit>.json`, and the
calibration take. Dataset construction (labels → joint actions via IK) needs
the robot's hand-eye and kinematics, so it runs on the robot side.

## Command index

Every command prints its full documentation with `--help`. Run from the
repository root.

| command | does | step |
|---|---|---|
| `cameras_setup.py [--list]` | preview every camera, assign forward / head / wrist roles, write `calib/cameras.json` | 2 |
| `focus_tune.py <dev> [--ref <dev>]` | live window + marker edge-width meter to lock a barrel-focused lens; `--ref` shows a known-good camera beside it | 2 |
| `calibrate_uvc.py <dev> [--name] [--model rational] --square-mm --marker-mm` | live ChArUco intrinsics → `calib/<name>_intrinsics.json` | 3 |
| `record_session.py --glove --dataset D --num_episodes N` | multi-camera session: pre-flight, ritual, episode marks, width holds | 4 |
| `uvc_record.py <name> [dev] [WxH] [fps]` | one camera's raw MJPEG + wall clocks (launched by record_session) | 4 |
| `detect_parallel.py <video.mkv> [--keep-27] [--workers N]` | ArUco 4x4 + AprilTag 16h5 detection caches | 5 |
| `bundle_dodeca.py --npz= --intr= --marker-mm= --ball-ids=` | pass 2: free bundle of the ball's faces | 6 |
| `bundle_dodeca_ideal.py --npz= --init= --intr= --marker-mm= --out=` | pass 3: dodecahedron-prior refinement → the bundle the pipeline uses | 6 |
| `calibrate_gripper_bundle.py <stem> --bundle= --marker-map= --out=` | gripper-back markers in the ball's body frame | 6 |
| `session_check.py <dataset> --unit <unit>` | per-episode pose/width coverage, hold sanity, tile visibility → OK / problems | 4b |
| `tiles_pose.py layout\|solve …` | export the workspace-tile layout from a static camera; camera→world pose from the tiles | 4b |
| `pack_session.py <dataset> … --unit <unit> [--dest]` | hand-off manifest + rsync line (sessions are 7–14 GB) | 4b |

Libraries (not commands): `labels_lib.py` — the label stack (fused ball
pose, body-frame aperture estimators); `uvc_ritual.py` — per-camera
exposure/gain/focus controls, selected by device name.

## Where to edit

This directory is the single source of truth for every script of the
system, kit and robot side alike, and for the measured configs in `calib/`.
The private experiment tree that produced the early numbers links to these
files with relative symlinks rather than keeping copies, so a change
committed here is immediately what runs.

## Robot side

The steps after a session, in the order the paper's chain ran them. Robot-side
scripts run in a LeRobot environment with the
[lerobot_alohamini](https://github.com/YosubShin/lerobot_alohamini) fork
checked out (`LEROBOT_ALOHAMINI`, default `~/lerobot_alohamini`); the YAM's
forward kinematics and the ZMQ host use the i2rt SDK's own venv where noted.

1. **Teleoperation.** `yam_zmq_host.py` owns the CAN buses and runs the
   bilateral teleop loop (i2rt venv); `record_yam_teleop.py` records LeRobot
   episodes from it while `record_session.py` records every camera with wall
   clocks (`docs/COLLECTION_RUNBOOK.md`). Start `so101_cam_track.py` first so
   the positioner follows the gripper during teleoperation exactly as it will
   in rollouts.
2. **Hand-eye** (once per rig state): `scripts/handeye_run.sh <dataset>` runs
   detect → export → FK → solve (`cv2.calibrateRobotWorldHandEye`) on any
   session with the ball on the robot and writes `calib/handeye_<dataset>.json`.
3. **Base ↔ tile world** (once): `base_world_fit.py` fits the link from the
   head camera's own frames (tiles + ball) against FK over a session; v3 of
   the paper sits at 6.5 mm / 1.55° median over 5065 frames.
4. **Glove tool frame** (once per glove unit): `glove_tool_frame.py` builds the
   same tool frame on the robot's and the glove's ball so IK targets land
   where the robot's do → `calib/handeye_glove_<unit>.json`.
5. **Head-camera pose per frame:** `head_pose_fast.py <stem>` (tiles,
   half-res detect, full-res corner refinement) → `calib/<stem>_cam_world.json`.
6. **Dataset:** `build_dataset.py --pipeline B --forward <head stem> ...`
   joins recordings, labels and (for teleop) the skeleton into a LeRobot
   dataset with joint-space actions via IK; `spotcheck_dataset.py` renders a
   per-episode check page. `--help` lists the gates and the width maps.
7. **Viewpoint model** (once per collector): `head_residual_data.py` pairs
   head and glove poses in the neutral head frame; `head_residual_fit.py
   --save calib/head_residual_model.json` fits the ridge model and reports the
   held-out error against the static and gain-tracker baselines.
8. **Positioner:** `cam_arm_poses.py` samples tile-verified poses,
   `so101_cam_kin.py fit` fits the kinematics, `so101_cam_park.py` parks
   closed-loop against the tiles, `so101_cam_track.py` follows the gripper
   (`--static` for the parked ablation, `--release_cam` to hand the camera to
   the recorder).
9. **Training:** LeRobot ACT, 50k steps, with `calib/train_eps_*.txt` as the
   episode lists; `val_curve.py <checkpoint>` scores a checkpoint on the
   dataset's held-out episodes with train-parity loading.
10. **Rollouts:** `evaluate_yam.py --arm <name>` for single rollouts,
    `eval_grid.py --policies a,b,... --scenes 25` for the placement grid
    (ghost-aligned re-placement, shuffled arm order, one-key scoring, append-only
    `trials.jsonl`); `camera_realign.py` puts the fixed camera back between
    days; `grid_analysis.py <dir> --ref <arm>` gives Wilson intervals, paired
    wins/losses and exact McNemar tests.

Robot-side command index:

| command | does | step |
|---|---|---|
| `yam_zmq_host.py --leader_channel … --follower_channel …` | CAN owner, bilateral teleop, policy mode (i2rt venv) | 1 |
| `record_yam_teleop.py --dataset D --num_episodes N` | LeRobot episodes from the host's commanded target | 1 |
| `handeye_run.sh <dataset>` / `handeye.py export\|fk\|solve` | hand-eye from a session with the ball on the robot | 2 |
| `base_world_fit.py <head stem> --track …` | base ↔ tile world from the head camera | 3 |
| `glove_tool_frame.py` | glove ball → virtual flange | 4 |
| `head_pose_fast.py <stem> [--stride N]` | per-frame head-camera pose from the tiles | 5 |
| `build_dataset.py --pipeline B …` | recordings + labels (+ skeleton) → LeRobot dataset | 6 |
| `spotcheck_dataset.py <repo>` | per-episode composite check video | 6 |
| `head_residual_data.py` / `head_residual_fit.py [--save]` | viewpoint model data and fit | 7 |
| `cam_arm_poses.py` / `so101_cam_kin.py fit\|check` | positioner pose sampling and kinematic fit | 8 |
| `so101_cam_park.py` / `so101_cam_track.py` | park at the median head pose; follow the gripper | 8 |
| `so101_cam_steptest.py` | servo gain check under load | 8 |
| `val_curve.py <ckpt>` | validation loss for one checkpoint | 9 |
| `evaluate_yam.py --arm A` / `eval_grid.py --policies …` | rollouts; the placement grid | 10 |
| `camera_realign.py [--ref]` | put the fixed camera back to a reference frame | 10 |
| `grid_analysis.py <log dir> --ref A` | grid report with paired tests | 10 |
| `free_base.py`, `fwd_world_fit.py`, `interp_diag.py`, `label_regions.py`, `val_regions.py`, `vis_causal.py`, `head_motion_analysis.py`, `head_tracking_baseline.py`, `gap_bench.py` | analyses behind individual paper numbers; each documents itself with `--help` | — |

## Troubleshooting (things that actually happened)

- **Wrist fisheye washed out / saturated, tags not decoding, controls look
  fine.** The Arducam 1080P Low Light module's ISP can get stuck in a state
  where exposure/brightness writes are accepted, read back correctly, and
  make no difference to the image (blacks lifted to ~180 while the scene
  still has real blacks). **Replug the camera.** Seen repeatedly on two
  units; nothing in software reaches it. `scripts/tag_contrast.py <dev>`
  shows per-tag black/white levels live: healthy tags read blacks ≲40,
  contrast ≳150 under lab light.
- **A recorder from an earlier run is still alive.** If a session is killed
  rather than finished, its `uvc_record` processes can outlive it, keep the
  cameras, and keep writing `data/raw/.<stem>_seg*.mkv`; the next run with
  the same dataset name then reads *those* segments as if they were its own
  (pre-flights pass, deliverables are stitched from the wrong process).
  `record_session.py` now refuses to start while a recorder for the stem is
  alive (`kill -TERM <pid>` first — it finishes its remux). Check with
  `pgrep -af uvc_record.py` if anything looks odd.
- **A camera dropped mid-session.** In glove mode a watchdog now beeps if a
  stream stops producing segments for 90 s. Episodes recorded after the drop
  have no stream from that camera; the build gates them per stream.
- **Episode marks across runs.** `<dataset>_glove_episodes.json` is
  cumulative: a second run with the same name continues the numbering and
  never overwrites. (Before 2026-10-06 it overwrote — 30 episodes' marks
  were lost and had to be reconstructed from marker visibility.)
- **`input()` never responds at the prompt.** The terminal is in raw mode
  from an earlier process; the glove loop runs `stty sane` first and prints
  the tty flags it found. Also: the preview windows take keyboard focus —
  click the terminal.
