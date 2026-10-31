# Raw glove sessions with the head camera

The four glove sessions behind the paper's head-camera policies (198
demonstrations), as recorded: continuous multi-camera video with wall clocks
and hand-marked episode boundaries. Nothing here is trimmed, re-encoded or
labelled; labels are computed from these files by the pipeline in
[`scripts/README.md`](../scripts/README.md).

Copies: `gs://head-cam-umi-raw/pilot/raw/` (Google Cloud Storage, project
`videosync-413908`, us-west1) and `/mnt/nas/nvme_archive/pilot/raw/` on the
lab NAS, which the repo reaches as `data/raw/`. To work from the bucket:

```
gcloud storage cp 'gs://head-cam-umi-raw/pilot/raw/yam_glove6_*' data/raw/
```

## Sessions

| session | recorded (PDT) | episodes | episode length (min / median / max) | takes | size |
|---|---|---|---|---|---|
| `yam_glove5` | 2026-10-08 15:09 and 15:44 | 115 | 4.5 / 7.5 / 13.6 s | `.take1` (eps 0–48, 26 min) + main (eps 49–114, 28 min) | 95 GB |
| `yam_glove6` | 2026-10-15 12:41 and 12:54 | 49 | 1.8 / 6.3 / 8.6 s | `.take1` (eps 0–29, 11 min) + main (eps 30–48, 6 min) | 31 GB |
| `yam_glove7` | 2026-10-15 13:46 | 30 | 4.2 / 6.5 / 8.2 s | main only (13 min) | 24 GB |
| `yam_glove8` | 2026-10-15 14:40 | 4 | 5.6 / 5.9 / 6.7 s | main only (2.4 min) | 4.2 GB |

All four are the same collector (glove unit `yosub`) on the paper's pick-and-
place task, with the workspace tiles in place. Earlier glove sessions
(`glove_cal_yosub`, `yam_glove1`–`yam_glove4`, 6–7 October) also carry a
`_b0587b` stream but predate the head-camera setup the paper used; they are
not in the bucket.

## Files per session

Every session is a flat set of files sharing the stem `<session>`, one group
per camera:

| file | what |
|---|---|
| `<session>_b0587b.mkv` | **head camera**: the Arducam B0587 worn on the collector's head (GoPro strap), 3840×2160 at 25 fps |
| `<session>_fish.mkv` | **wrist camera**: the glove's fisheye (Arducam 1080P low-light), 1920×1080 at 30 fps, looking along the jaws at the 16h5 tip tags |
| `<session>_b0587.mkv` | **fixed camera**: the second B0587 on a stand looking across the workspace, 3840×2160 at 25 fps; the ablation / fallback view |
| `<session>_<cam>_ts.json` | one entry per video frame, in order: `{"wall": <epoch seconds>}` from `time.time()` on the recording machine. Frame *k* of the `.mkv` is entry *k* |
| `<session>_<cam>_meta.json` | the v4l2 device the stream came from and any controls set at open (empty: the exposure/focus ritual is applied after the stream starts and logged instead) |
| `<session>_<cam>_recorder.log` | `uvc_record.py` console: the control ritual that was applied (gain 0, sharpness 0, manual exposure, `exposure_time_absolute` 2), running frame counts, the segment concatenation and the final wall-clock range |
| `<session>_glove_episodes.json` | the episode boundaries (below) |

The videos are the cameras' own MJPEG frames copied into Matroska without
re-encoding (`uvc_record.py`), so a frame is exactly what the sensor
delivered. Camera streams are not frame-locked to each other; the wall
clocks are the only alignment, and the three cameras of a take start within
0.25 s of each other.

**Takes.** `yam_glove5` and `yam_glove6` were each recorded in two runs of
`record_session.py` under the same session name. When the second run started,
the first run's files were renamed `<session>_<cam>.take1.*` and the second
run wrote the plain `<session>_<cam>.*`. Both takes are complete recordings
with their own `_ts.json`; the episode file is shared and numbers straight
through (see below).

## Episode boundaries

There is no robot and no in-video marker for episodes; the cameras roll
continuously and the collector marks episodes at the terminal while
recording (`record_session.py --glove`): **Enter** starts an episode, **Enter**
ends it, **d** discards the one just finished, **q** ends the session. Each
mark is a `time.time()` stamp on the same clock as the `_ts.json` files, so
an episode is simply the frames of each camera whose wall clock falls in
`[t_start, t_end]`:

```json
{"dataset": "yam_glove6",
 "episodes": [{"episode_index": 0, "t_start": 1792093339.816, "t_end": 1792093341.647, "duration_s": 1.83},
              ...]}
```

Things to know when slicing:

- **Indices are contiguous and cumulative.** A discarded episode is never
  written and its index is reused, and a second run of the same session name
  continues the numbering rather than restarting (the file is read back and
  appended). Which take holds an episode follows from its wall clock: an
  episode lies inside exactly one take's `_ts.json` range.
- **Idle time between episodes is in the video** (block resets, glove
  re-positioning). Only the marked intervals are demonstrations. The marked
  intervals total 866 s, 311 s, 191 s and 24 s for the four sessions out of
  54, 17, 13 and 2.4 min of footage.
- **The first episode of `yam_glove6` is 1.8 s long**: a false start that was
  kept in the file. The build's per-episode gates (pose coverage, width
  coverage) are what decide whether an episode enters a dataset, not the
  marks.
- **Frames carry no labels.** Pose and width per frame come from the
  fiducials; see the next section.

## What is not in the bucket

- Detection caches (`calib/<stem>_dodeca_detections.npz`,
  `_tips16h5_detections.npz`) and per-frame camera-pose solves
  (`calib/<stem>_cam_world.json`). They are regenerable from the videos and
  were not archived.
- Calibration (intrinsics, marker sizes, bundles, tile layout, base↔world):
  tracked in this repo under [`calib/`](../calib/README.md).
- The built LeRobot datasets and checkpoints.

## Processing pointers

The whole chain, with the files each step reads, is in
[`scripts/README.md`](../scripts/README.md); this is the order for a glove
session, with the calibration files that apply to these four.

1. **Detect fiducials in every stream.**
   `python scripts/detect_parallel.py data/raw/<session>_b0587b.mkv --keep-27`
   and the same for `_b0587` and `_fish` (and the `.take1` stems). Writes the
   ArUco 4x4 cache (ball faces 25–35, tails and bases 36–49) and the AprilTag
   16h5 cache (tips 2 and 3). `--keep-27` is required: 4x4-27 is a real ball
   face on this glove unit. Known phantoms: 16h5 tips decode as a spurious
   4x4-17, and 36h11 tag 0 as 4x4-37; both are handled by the maps.
2. **Head-camera pose per frame from the tiles.**
   `python scripts/head_pose_fast.py <session>_b0587b` solves the AprilTag
   36h11 tiles (IDs 1–5, `calib/tags36h11_yosub.json`, layout
   `calib/workspace_tiles_pilot_v3.json`) in each head frame and writes
   `calib/<stem>_cam_world.json` (`per_frame[k].T_cam_world`, with the tile
   count and RMS used as quality gates downstream). Head intrinsics:
   `calib/B0587b_f3_intrinsics.json` is the file for every head session from
   2026-10-09; `yam_glove5` is from 2026-10-08, so check which `B0587b`
   intrinsics its solves were made with before reusing them.
3. **Fixed camera → world** (fallback poses, ablations).
   One static solve per rig state: `calib/forward_cam_world_yam_glove5_b0587.json`
   and `..._b0587.take1.json` for glove5, `calib/forward_cam_world_20261015.json`
   for glove6–8 (`tiles_pose.py solve`, intrinsics `calib/B0587_intrinsics.json`).
4. **Glove pose and tool frame.** The ball's face layout and the gripper-back
   markers for this unit are `calib/glove_yosub_dodeca_bundle.json` and
   `calib/glove_yosub_gripper_bundle.json` (sizes in
   `calib/marker_map_glove_yosub.json`); the virtual flange and tool point are
   `calib/handeye_glove_yosub.json` and `calib/eef_target_glove_yosub.json`.
   `calib/base_world_v3_20261016.json` ties the tile world to the robot base so
   glove poses become robot IK targets.
5. **Jaw width.** From the 16h5 tips in `_fish` with
   `calib/fisheye_glove_yosub_intrinsics.json`, mapped into gripper command
   space by `calib/width_to_cmd_glove_v2.json` (the paper's map).
6. **Build the dataset.** One call per take, for example
   ```
   python scripts/build_dataset.py --pipeline B \
       --forward yam_glove6_b0587b --wrist yam_glove6_fish \
       --episode-marks data/raw/yam_glove6_glove_episodes.json \
       --cam-world calib/yam_glove6_b0587b_cam_world.json \
       --grip-map calib/width_to_cmd_glove_v2.json --out <repo_id>
   ```
   (`--help` lists the pose gates, the fixed-camera fallback flags and the
   width options; `.take1` stems build the same way). Without
   `--episode-marks` a whole take becomes one episode.
   `spotcheck_dataset.py` renders a per-episode check page. The paper's
   training episode lists are `calib/train_eps_pilot_*.txt`.
7. **Viewpoint model.** `head_residual_data.py` pairs the head and glove poses
   of glove5 (`calib/head_residual_glove5.npz`) and `head_residual_fit.py`
   fits `calib/head_residual_model.json`, which places the positioner.
8. **Sanity check a session** before any of this:
   `python scripts/session_check.py <session> --unit yosub` reports per-episode
   pose and width coverage and tile visibility.
