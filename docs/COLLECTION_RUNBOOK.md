# Multi-pipeline collection runbook

One teleop session → three datasets from the same episodes:
- **A** classic-UMI sim: pose = forward-cam dodeca, gripper = wrist-cam tips, obs = wrist
- **B** glove-UMI sim: pose + gripper = forward cam, obs = forward
- **C** ground truth: actions = actuator/FK, obs = wrist

Everything is joined offline by wall clock (single machine → single clock).
The recorder writes per-frame wall clocks to
`meta/episode_frames_wallclock.jsonl`; uvc_record writes `<stem>_ts.json`.

## Pre-flight (once, before the first session)

- [ ] **Fisheye ChArUco calibration** — never done; blocks pipeline A only.
      Plug fisheye in (device indices shuffle — tools pick rituals by NAME):
      `python scripts/calibrate_uvc.py /dev/videoN --width 1920 --height 1080`
- [ ] **B0587 focus barrel witness mark** + quick ChArUco spot-check that the
      0.744px calibration still holds (focus breathing ≈ 2% fx per step).
- [ ] **Lighting decision**: lab light → default ritual (e=2, gain 0).
      Dim/ceiling light → CERTIFIED (2026-09-03): e=10/gain=20 via the
      `UVC_RITUAL` override in uvc_ritual.py — aperture grading at parity
      with bright lab (14 GOOD estimators, 88% coverage, 6% of stroke).
      Keep whichever lighting you pick constant across the whole session.
- [ ] **Scene**: red block first; then the 20/40/60 mm width set (the
      discriminating task for gripper-label quality A vs B).
- [ ] Verify wrist fisheye actually sees tips 13/14 at the working pose.

## Session command card

Terminal 1 — camera-free host (owns both CAN buses; start once, outlives
sessions):
```
~/co/i2rt/.venv/bin/python yam_zmq_host.py --leader_channel can_leader \
    --follower_channel can_follower
```

Terminal 2 — everything else, one command (`scripts/record_session.py`
resolves cameras by NAME, starts both uvc_records, runs the interactive
teleop recorder, and shuts the recorders down cleanly afterwards):
```
python scripts/record_session.py --dataset yam_sess1 --num_episodes 20 \
    --episode_time 60 --task "pick red block"
```
Camera stems land as `data/raw/<dataset>_b0587.*` and `<dataset>_fish.*`
(recorder logs beside them). Never Ctrl-C during the final
"remux in progress" print — 4K remux takes a while.

Disk behavior: cameras record continuously (~1 GB/min combined) into 60 s
segments; a janitor deletes segments older than 5 min that overlap no saved
episode, so pauses/idle never accumulate. At session end the survivors are
concatenated into the deliverable mkv + per-frame `_ts.json` (pruned idle
time appears as wall-clock gaps; frame indices stay aligned). A crash loses
at most the current segment. The janitor also warns LOUDLY if a camera
stops producing segments mid-session.

**Session sizing.** `data/raw` is a symlink to `/mnt/nvme/pilot/raw`
(~500 GB free): episode footage is retained at ~1.2 GB/min (both cameras)
and the end-of-session concat needs 2x that peak, so even a 100-episode
single session (~230 GB peak) fits. Still prefer 2-3 sessions for a
100-episode corpus (a bumped camera or bad setting costs one session, not
all of it; `--resume` on the teleop recorder accumulates), and delete or
archive a session's raw mkvs once its datasets are built and verified.

Manual per-process fallback: run the two `uvc_record.py` and
`record_yam_teleop.py --no_cameras` commands yourself (stop order:
recorder q → camera recorders Ctrl-C).

## Post-session build

```
python scripts/detect_dodeca.py data/raw/b0587_sess1.mkv    # forward dets
python scripts/detect_dodeca.py data/raw/fisheye_sess1.mkv  # wrist dets (tips)

SKEL=~/.cache/huggingface/lerobot/yam_sess1
python scripts/build_dataset.py --pipeline B --skeleton $SKEL \
    --forward b0587_sess1 --out yam_sess1_B --fps 15 --obs-size 640x360
python scripts/build_dataset.py --pipeline A --skeleton $SKEL \
    --forward b0587_sess1 --wrist fisheye_sess1 \
    --wrist-intrinsics calib/fisheye_intrinsics.json --out yam_sess1_A --fps 15
python scripts/build_dataset.py --pipeline C --skeleton $SKEL \
    --wrist fisheye_sess1 --out yam_sess1_C --fps 15
```

The constructor prints the per-session aperture calibration table (same gate
as gripper_labels.py) and per-episode pose/grip coverage; episodes under
`--min-coverage` (default 70%) are skipped with a notice.

Status: pipeline B path smoke-tested end-to-end on b0587_e8 (glove mode,
2026-09-02); A and C paths are structurally complete but **untested until the
first real session** — budget 30 min of debugging on the first build.

## Where the code lives

Every script in this runbook is in [`scripts/`](../scripts/) of this
repository (kit procedure for glove sessions, intrinsics and solves:
[`scripts/README.md`](../scripts/README.md)). The private experiment tree
that produced the early numbers reaches the same files through relative
symlinks, so there is one copy to edit.
