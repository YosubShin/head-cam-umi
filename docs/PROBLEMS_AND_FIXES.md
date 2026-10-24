# Marker-Tracking Campaign: Problems Encountered → Fixes

Running ledger of every failure mode hit while building the fiducial
tracking system (2026-08-27 → 09-02), its root cause, and where the fix
lives. Statuses: **FIXED-TOOL** (encoded in a script/pipeline default),
**FIXED-PROC** (procedure/checklist), **PENDING-HW** (hardware on the way),
**BY-DESIGN** (absent in the glove design), **OPEN**.

## Camera / capture

| problem | root cause | fix | status |
|---|---|---|---|
| Detection collapsed above ~0.25 m/s (fake "cliff") | recorder decoded camera MJPEG and re-encoded with mp4v; inter-frame codec crushed moving regions | `uvc_record.py` stream-copies the camera's own MJPEG (mkv + wall-clock ts + settings snapshot) | FIXED-TOOL |
| Exposure knob "did nothing" | HDR ISP tone-maps output brightness to a target (~1–2 s settling) — brightness is a useless probe | probe exposure with STREAKS (`exposure_target.html` bar-count, spinner, rapid `exposure_sweep.py` outruns the AGC); knob is real: 100 µs units, linear | FIXED-PROC |
| e>330 all identical | integration capped at the 33 ms frame period (`exposure_dynamic_framerate=0`) | known ceiling; operate 0.1–33 ms | FIXED-PROC |
| Settings mysteriously lost between sessions | (a) replug resets UVC controls; (b) `cv2.VideoCapture` resets controls on open; (c) writes only physically latch AFTER streaming starts — readback shows accepted values regardless | all tools apply the ritual MID-STREAM and print readback; recorder snapshots `_meta.json`; behavioral check = **hand-wave blur test** in `uvc_live.py`; readback is never proof | FIXED-TOOL |
| Manual short exposure graded WORSE than auto (mode 1 vs 3) | at first: the value never latched (race above); once latched, verified 2 ms won decisively | ritual final: `auto_exposure=1, e=20 (2 ms), gain=50, sharpness=0`; envelope 99–100% det to 0.5 m/s, 90% to 1.0 | FIXED-TOOL |
| Weak gain (0–100 ≈ tens of %) can't brighten short exposures | hardware limit | light the scene; 4K cam has ~7× light budget (F1.65 + 2 µm STARVIS 2) | PENDING-HW |
| Focus changes silently invalidate intrinsics | focus breathing ~2% fx per step; plus AF re-enabled by the cv2 reset during one calibration | fixed-focus lock in ritual; recalibrate after ANY focus change; calibrator locks post-stream and prints readback | FIXED-TOOL |
| MJPEG artifacts limited corner precision | compression ringing at marker edges | `CORNER_REFINE_APRILTAG` pipeline default (7.0→4.3 mm, coverage 4.5×) | FIXED-TOOL |
| mkv unseekable in players | wall-clock PTS via `-copyts` broke seek logic | recorder remuxes deliverable to zero-based time; wall clock lives in `_ts.json` | FIXED-TOOL |
| D405s graded 14–18 mm | pixel budget: 11 px/deg (depth-camera optics) | D405 = ego/observation fleet only; never a tracker | FIXED-PROC |
| Narrow FOV clips gripper at frame edge (pose-5 failure) | B0591 65° true FOV | 4K/88° camera incoming; certify via CAMERA_CERTIFICATION.md gauntlet on arrival | PENDING-HW |
| Fisheye stalled 10 s+ periodically (calibration crawled; `select() timeout` spam) | fps left UNSET at open → driver negotiates a broken mode on this camera (51 frames/45 s, 36 s dead; explicit fps → clean 30 fps, zero stalls). NOT autosuspend, NOT the Renesas card | all live tools now `cap.set(CAP_PROP_FPS, 30)`; `uvc_record` already passes ffmpeg an explicit rate | FIXED-TOOL |
| Fisheye calibration overwrote `B0591_intrinsics.json` with garbage | `calibrate_uvc.py --name` defaulted to the literal `B0591` | name auto-derived from sysfs device name; existing output backed up before write; views saved to `_charuco_views.npz` (`--refit` = no re-waving); bogus file quarantined as `INVALID_*`; B0591 needs a 10-min recalibration when next used | FIXED-TOOL |
| Fisheye pinhole calibration unusable (7.2 px holdout) | 5-coeff model can't fit fisheye distortion | `--model rational` (8-coeff, stays solvePnP-compatible): 2.67 px holdout, 2.60 px central where tips live; richer models overfit; residual likely rolling-shutter shear from hand-waved board — stationary-view gating is the next lever if pipeline-A grading demands it | FIXED-TOOL |

## Markers / geometry

| problem | root cause | fix | status |
|---|---|---|---|
| Marker sizes drift between prints (~4%) | printer scaling per run | caliper EVERY print; sizes live in EDGE maps + memory | FIXED-PROC |
| Tips 13/14 invisible to external cameras | mounted facing the wrist camera | orientation is a design axis: outward tips 17/18 added; glove co-designs facing with the viewer | FIXED-PROC / BY-DESIGN |
| One whole side occluded ~half the time; anchors sank with their side | wrist-yaw motor, protruding cable, one-sided camera | v2 markers: +y top faces (19–22) readable from above; second mirrored camera for the union; **glove has no motor/cable occluders** | FIXED-PROC / BY-DESIGN |
| Tail↔base 3D distance nonlinear/compressed | bolt asymmetry: tails not coplanar with base markers | travel-axis projection (SVD of pair-vector track), never 3D norm | FIXED-TOOL |
| "id19 pose-sick" (40–58 mm RMS pairs) | ANALYSIS ARTIFACT: global SVD axis sign flips with viewpoint; pooling +47 and −45 swings faked noise. Marker was healthy (2.6 mm within-pose) | per-pose fits interim; proper fix = express tail in the BASE MARKER's own frame (travel direction constant across viewpoints) — goes in the dataset constructor | OPEN (designed) |
| Aperture calibration "never done"? | it is implicit: per-pair linear fits vs actuator (`e8_meas`), holdout-validated, every session | glove corollary: projection is metric, slope = 2× for symmetric jaws → calibration = ONE on-camera fully-closed reference | FIXED-TOOL |
| Coverage misread (31% vs reality) | six-pose stress set includes gripper-facing-away poses; aimed vantage measures 98% | quote coverage BY REGIME; deployment doctrine: exactly one base↔tail pair visible suffices | FIXED-PROC |

## Analysis / method

| problem | root cause | fix | status |
|---|---|---|---|
| Motion vs occlusion conflated in detection stats | same "det%" metric absorbed both | deconfound with the unoccluded dodeca (rigid pairs = built-in ground truth) — `dodeca_dynamics.py` | FIXED-TOOL |
| Per-speed bins misaligned with real motion | commands finish before the arm does (tracking lag) | bin by MEASURED image velocity from the marker track | FIXED-TOOL |
| Cross-pose bias read as noise | viewing-geometry-dependent depth bias per pose | decompose within-pose vs cross-pose before concluding; base-frame estimator removes the cause | FIXED-PROC |
| Single-marker aperture inference failed (25 mm RMS) | single-pose IPPE depth noise dominates absolute positions | pair distances only (common-mode depth cancels); anchors give single-TAIL fallback | FIXED-PROC |
| Sony "4 mm" vs honest grading | favorable-subset evaluation | actuator-referenced grading over ALL co-visible frames (`aperture_union.py`) | FIXED-TOOL |

## Meta-lessons (the ones that kept repeating)

1. **Readback/spec/brightness are testimony; only behavior is evidence.**
   Every major misdiagnosis came from trusting a reported value.
2. **Deconfound before concluding** — occlusion+re-encode+mode stacked into
   one fake "blur cliff"; three separate fixes.
3. **Aggregate statistics hide structure** — per-pose, per-regime,
   per-velocity decompositions overturned three verdicts (user-driven each
   time).
4. Certification procedure for any new camera: `CAMERA_CERTIFICATION.md`.
