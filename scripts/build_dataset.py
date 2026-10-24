#!/usr/bin/env python3
"""Offline LeRobot dataset constructor — joins standalone camera recordings
(uvc_record mkv + _ts.json wall clocks + detect_dodeca npz) with the teleop
skeleton (record_yam_teleop --no_cameras) into training datasets.

Pipelines (same episodes, three label/observation recipes):
  A  classic-UMI sim: EEF pose from FORWARD-cam dodeca, gripper from
     WRIST-cam tips (13/14), observations = wrist frames
  B  glove-UMI sim:   EEF pose + gripper from FORWARD cam,
     observations = forward frames (cropped/downscaled)
  C  ground truth:    actions = skeleton actuator/FK verbatim,
     observations = wrist frames

Label stack: labels_lib.py (fused dodeca body pose; body-frame aperture
estimators calibrated per session against the skeleton's gripper channel).
Action/state vector for A/B: [x y z rx ry rz grip] — dodeca body pose in the
forward camera frame (m, rodrigues) + gripper in actuator units.

Timeline: skeleton per-frame wall clocks (meta/episode_frames_wallclock.jsonl)
subsampled to --fps. Without --skeleton (glove mode / smoke test), pass
--log <e_runner json> for gripper calibration; the timeline is then the
forward camera's own frames and the whole recording is one episode.

Usage examples:
  # pipeline B from a teleop session
  python scripts/build_dataset.py --pipeline B --skeleton ~/.cache/huggingface/lerobot/yam_sess1 \
      --forward b0587_sess1 --out yam_sess1_B --fps 15 --obs-size 640x360
  # glove-mode smoke test on an e-run
  python scripts/build_dataset.py --pipeline B --log e2/log_e8_1788402701.json \
      --forward b0587_e8 --out e8_smoke_B --fps 12.5 --max-frames 300
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels_lib as L  # noqa: E402

POSE_NAMES = ["eef.x", "eef.y", "eef.z", "eef.rx", "eef.ry", "eef.rz",
              "gripper"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pipeline", choices=["A", "B", "C"], required=True)
    p.add_argument("--skeleton", help="LeRobot root of record_yam_teleop session")
    p.add_argument("--log", help="e_runner log json (glove mode: no skeleton; "
                                 "gripper calibration from its e8_meas events)")
    p.add_argument("--forward", help="uvc_record stem of the forward tracker cam")
    p.add_argument("--wrist", help="uvc_record stem of the wrist fisheye")
    p.add_argument("--out", required=True, help="output repo_id")
    p.add_argument("--out-root", help="output root dir (default HF cache)")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--slow", type=float, default=1.0,
                   help="time-stretch factor: sample the timeline at "
                        "fps*slow but declare --fps, so demos play back "
                        "slow x slower at deploy (slow2x lever, gated by "
                        "the fps-10 probe 2026-09-17)")
    p.add_argument("--obs-size", default="640x360", help="WxH of emitted frames")
    p.add_argument("--obs-stem", default=None,
                   help="use THIS recording as the observation stream instead of the "
                        "pipeline's default (e.g. the head cam yam_glove5_b0587b); labels, "
                        "width and fovea sources are unchanged")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--det-dir", default="calib")
    p.add_argument("--intrinsics", default="calib/B0587_intrinsics.json")
    p.add_argument("--wrist-intrinsics", help="fisheye intrinsics json "
                                              "(required for pipeline A)")
    p.add_argument("--bundle", default="calib/dodeca_bundle_ideal.json")
    p.add_argument("--cam-world", default=None, metavar="JSON",
                   help="MOVING label camera (head cam): per-frame camera "
                        "pose from the tiles (head_pose_fast.py "
                        "calib/<stem>_cam_world.json). Body poses from "
                        "--forward are carried into the TILE frame before "
                        "interpolation; frames without a usable tile pose "
                        "(--cam-world-min-tiles/--cam-world-rms, bridged "
                        "across gaps <= --cam-world-bridge s) are dropped")
    p.add_argument("--cam-world-min-tiles", type=int, default=2)
    p.add_argument("--cam-world-rms", type=float, default=5.0)
    p.add_argument("--cam-world-bridge", type=float, default=0.5)
    p.add_argument("--pose-fallback", default=None, metavar="STEM",
                   help="with --cam-world: a FIXED camera's stream whose "
                        "fused body poses fill the frames where the moving "
                        "camera has no tile pose (mapped into the tile frame "
                        "through --pose-fallback-world). Gripper labels stay "
                        "with --forward")
    p.add_argument("--pose-fallback-intrinsics",
                   default="calib/B0587_intrinsics.json")
    p.add_argument("--pose-fallback-world",
                   default="calib/forward_cam_world_20261015.json",
                   help="{T_cam_world} of the fallback camera in the tile frame")
    p.add_argument("--pose-fixed-only", action="store_true",
                   help="with --pose-fallback: discard the moving camera's "
                        "poses and label EVERY frame from the fixed camera "
                        "(conventional external-tracking setup); the moving "
                        "camera then only supplies the gripper labels")
    p.add_argument("--pose-fallback-tol", type=float, default=0.1,
                   help="s: a fallback row is added only when no head-chain "
                        "row lies within this of its time")
    p.add_argument("--base-world", default=None, metavar="JSON",
                   help="with --cam-world --joint-actions: T_world_base "
                        "(base_world_fit.py) — robot base <-> tile link "
                        "replacing the forward cam's hand-eye")
    p.add_argument("--gripper-bundle", default="calib/gripper_bundle.json",
                   help="calibrated BASE markers in the ball's body frame "
                        "(single-base pose fallback). Glove sessions MUST "
                        "pass the glove's own (calibrate_gripper_bundle.py "
                        "--bundle=<glove ball> --marker-map=<glove map>).")
    p.add_argument("--label-tol", type=float, default=0.12,
                   help="max s between an output frame and its nearest label; "
                        "beyond this the frame counts as a coverage gap")
    p.add_argument("--min-coverage", type=float, default=0.7,
                   help="episodes below this pose+grip coverage are skipped")
    p.add_argument("--split-gap", type=float, default=0.0,
                   help="instead of gating whole episodes, cut each episode "
                        "at label gaps longer than this many (real) seconds "
                        "and emit the visible stretches as separate episodes "
                        "(epNsK); shorter gaps are interpolated as usual. "
                        "0 = off (gate whole episodes with --min-coverage)")
    p.add_argument("--min-segment", type=float, default=1.0,
                   help="with --split-gap: drop visible stretches shorter "
                        "than this many (real) seconds")
    p.add_argument("--max-frames", type=int, help="debug: cap emitted frames")
    p.add_argument("--task", default="YAM manipulation")
    p.add_argument("--eef-crop", type=int, default=0,
                   help="emit observation.images.eef: a square crop of this "
                        "many NATIVE px around the fused dodeca position "
                        "(forward-obs pipelines B and C'; 0 = off). Keep the "
                        "same value for B and C' so their obs match.")
    p.add_argument("--eef-target", default="calib/eef_target.json",
                   help="body-frame point (json p_body_mm) the fovea crop is "
                        "centred on; per BALL MOUNT — the glove unit has its "
                        "own (calib/eef_target_glove_<unit>.json)")
    p.add_argument("--eef-size", type=int, default=224,
                   help="emitted eef-crop resolution (square)")
    p.add_argument("--joint-actions", action="store_true",
                   help="pipeline B: convert the fiducial pose labels to "
                        "ABSOLUTE JOINT actions via offline warm-started IK "
                        "(hand-eye + mink; validated 4-6mm vs the real "
                        "follower). State becomes measured joints. Matches "
                        "C'/SO-101 action space exactly -> pure label-source "
                        "comparison.")
    p.add_argument("--exclude-eps", default="",
                   help="comma list of SKELETON episode indices to never "
                        "emit (bad recordings); safer than deleting from "
                        "the skeleton, which would desync the wall-clock "
                        "sidecars")
    p.add_argument("--emit-c", metavar="REPO_ID",
                   help="pipeline B only: ALSO emit the C' control dataset "
                        "(actuator actions, identical obs) in the same pass "
                        "— halves build time vs two runs")
    p.add_argument("--append", action="store_true",
                   help="append to an existing output dataset (multi-take "
                        "sessions: run once per take; the obs-coverage gate "
                        "routes each episode to the take that recorded it)")
    p.add_argument("--side", help="side-camera stem (b0587b): emits "
                                  "observation.images.side as an EEF-centered "
                                  "crop via the side camera's own pose solve")
    p.add_argument("--side-intrinsics", default="calib/B0587b_intrinsics.json")
    p.add_argument("--fuse-side-grip", metavar="STEM",
                   help="pipeline B: fill forward-cam GRIP gaps with width "
                        "measurements from this side camera (b0587b). Width "
                        "is body-frame geometry, so no cross-camera extrinsics "
                        "needed; side samples are used only where the forward "
                        "cam has no sample within --label-tol. Obs streams are "
                        "unaffected (stays bit-matched to the unfused build).")
    p.add_argument("--width-units-mm", action="store_true",
                   help="glove mode: accept a raw-mm width log without "
                        "calib/gap_to_actuator.json (glove-only datasets; "
                        "never cotrain the result with robot data)")
    p.add_argument("--episode-marks", metavar="JSON",
                   help="glove mode: wall-clock episode boundaries written by "
                        "record_session.py --glove ({'episodes': [{'episode_"
                        "index', 't_start', 't_end'}]}). Without this the "
                        "whole recording becomes ONE episode.")
    p.add_argument("--fit-intrinsics", default=None, metavar="JSON",
                   help="intrinsics for the --fit-stems streams when they "
                        "come from a DIFFERENT camera than --forward (body-"
                        "frame geometry is camera independent; the solve "
                        "is not). Default: same as --intrinsics")
    p.add_argument("--fit-stems", metavar="STEMS",
                   help="comma list of extra forward stems whose body-frame "
                        "rows AUGMENT the aperture-estimator fit (short "
                        "takes borrowing a richer take's calibration; "
                        "predictions still use only this take's frames)")
    p.add_argument("--grip-map", metavar="JSON",
                   help="piecewise-linear map applied to the gripper label series "
                        "(json with x_width / y_cmd): e.g. glove measured width -> "
                        "the robot's COMMANDED gripper, so a glove grasp lands at the "
                        "value teleop commands (the printed jaws do not squeeze; the "
                        "robot's do)")
    p.add_argument("--grip-smooth", type=int, default=0, metavar="N",
                   help="median filter (N camera frames) on the gripper label series; "
                        "fiducial width jitters frame to frame, a commanded channel does not")
    p.add_argument("--grip-from-skeleton", default=None,
                   choices=["commanded", "measured"],
                   help="pipeline B with --skeleton: take the gripper label "
                        "from the skeleton's actuator channel instead of the "
                        "marker aperture estimator (for a take whose marker-"
                        "to-actuator relation is broken, e.g. jaw deflection "
                        "under squeeze). 'commanded' = leader target, the "
                        "level the marker estimator lands at on a grasp")
    p.add_argument("--grip-from-wrist", action="store_true",
                   help="pipeline B: take the gripper WIDTH from the wrist "
                        "camera's tip markers (pipeline A's instrument) while "
                        "pose and observations stay on the forward cam. The "
                        "glove's wrist cam is part of the glove, so this is "
                        "still an electronics-free label; needs --wrist and "
                        "--wrist-intrinsics")
    p.add_argument("--fit-wrist-stems", metavar="STEMS",
                   help="pipeline A: comma list of extra wrist stems whose "
                        "tip rows AUGMENT the tip-estimator fit (glove "
                        "sessions: the caliper holds live in the unit's "
                        "first session; later sessions borrow them)")
    p.add_argument("--fit-side-stems", metavar="STEMS",
                   help="same, for the --fuse-side-grip estimator fit")
    p.add_argument("--marker-map", metavar="JSON",
                   help="marker-epoch override (e.g. calib/marker_map_v2."
                        "json for pilot3+ hardware): replaces the built-in "
                        "TAILS/BASES/TIPS id->edge maps; a tips_dict of "
                        "'16h5' makes pipeline A read the {stem}_tips16h5_"
                        "detections cache instead of the 4x4 one. Use ONLY "
                        "for takes recorded with the matching hardware.")
    p.add_argument("--emit-quad", action="store_true",
                   help="pipeline B one-pass family emit: --out is a BASE "
                        "name and four datasets {base}_Bj/_B/_Cf/_Bjt are "
                        "written from a single decode+encode (obs are "
                        "identical across them by construction). Requires "
                        "--joint-actions and --fuse-side-grip; supersedes "
                        "--emit-c. Bjt's label gate uses the FUSED grip "
                        "coverage, so an episode can enter Bjt only.")
    p.add_argument("--fuse-side-pose", metavar="TWIN_HANDEYE_JSON",
                   help="quad mode: fill forward POSE gaps for the Bjt arm "
                        "with twin-camera body poses mapped through the two "
                        "hand-eyes (calib/handeye.json + this file). Bj "
                        "stays single-camera pure; Bjt gets its own IK "
                        "chain on the fused series.")
    p.add_argument("--force-eps", default="",
                   help="comma-separated skeleton episode indices exempt from "
                        "the pose/grip label-coverage gate (labels are "
                        "interpolated through the gaps as usual); the "
                        "obs-camera gate still applies")
    p.add_argument("--square-obs", action="store_true",
                   help="crop the obs to the CENTER SQUARE (side = short "
                        "frame dimension) before resizing — the region shown "
                        "bright in the teleop preview; makes obs identical "
                        "across portrait/landscape mounts")
    p.add_argument("--rotate", default="none",
                   choices=["cw", "ccw", "180", "none"],
                   help="rotate emitted obs frames (portrait-mounted camera; "
                        "the mkv stores unrotated sensor frames)")
    return p.parse_args()


def load_stem(stem, raw_dir, det_dir, cache="dodeca"):
    ts = json.load(open(f"{raw_dir}/{stem}_ts.json"))
    det = np.load(f"{det_dir}/{stem}_{cache}_detections.npz")
    return ts, det


class SeqReader:
    """Sequential mkv frame access by index (output times are monotonic)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.idx = -1
        self.frame = None

    def get(self, k):
        if k < self.idx:
            raise ValueError(f"non-monotonic frame request {k} < {self.idx}")
        while self.idx < k:
            ok, f = self.cap.read()
            if not ok:
                return None
            self.idx += 1
            self.frame = f
        return self.frame


def interp_pose(tq, tP, tvecs, quats):
    """Pose at times tq from samples (tP, tvec, quat): componentwise interp,
    sign-aligned normalized-quaternion blend (adequate at 25 Hz sampling)."""
    q = np.array([np.interp(tq, tP, quats[:, i]) for i in range(4)]).T
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    t = np.array([np.interp(tq, tP, tvecs[:, i]) for i in range(3)]).T
    return t, q


def quat_from_rvec(rvec):
    a = np.linalg.norm(rvec)
    if a < 1e-12:
        return np.array([1.0, 0, 0, 0])
    ax = rvec.ravel() / a
    return np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * ax])


def rvec_from_quat(q):
    q = q / np.linalg.norm(q)
    w = np.clip(q[0], -1, 1)
    a = 2 * np.arccos(w)
    s = np.sqrt(max(1 - w * w, 1e-16))
    return (q[1:] / s) * a if a > 1e-9 else np.zeros(3)


def main():
    args = parse_args()
    tips_cache = "dodeca"
    if args.marker_map:
        mm = json.load(open(args.marker_map))
        L.TAILS = {int(k): v for k, v in mm["tails"].items()}
        L.BASES = {int(k): v for k, v in mm["bases"].items()}
        L.TIPS = {int(k): v for k, v in mm["tips"].items()}
        if mm.get("tips_dict") == "16h5":
            tips_cache = "tips16h5"
        print(f"== marker map {args.marker_map}: tails {sorted(L.TAILS)}, "
              f"bases {sorted(L.BASES)}, tips {sorted(L.TIPS)} "
              f"({mm.get('tips_dict', '4x4')})")
    if args.pipeline in ("A", "B") and not args.forward:
        raise SystemExit("--forward stem required for pipelines A/B")
    if args.pipeline == "A" and not args.wrist:
        raise SystemExit("--wrist stem required for pipeline A")
    if args.pipeline == "C" and not (args.wrist or args.forward):
        raise SystemExit("pipeline C needs an obs camera: --wrist or "
                         "--forward (forward obs = the C' control for B)")
    if args.pipeline == "A" and not args.wrist_intrinsics:
        raise SystemExit("--wrist-intrinsics required for pipeline A "
                         "(fisheye must be ChArUco-calibrated first)")
    if not args.skeleton and not args.log:
        raise SystemExit("need --skeleton or --log (gripper calibration source)")
    ow, oh = (int(x) for x in args.obs_size.split("x"))

    # ---- actuator series + episode timelines ---------------------------------
    if args.skeleton:
        root = Path(args.skeleton)
        walls = {}
        for line in open(root / "meta" / "episode_frames_wallclock.jsonl"):
            e = json.loads(line)
            walls[e["episode_index"]] = np.array(e["t_wall"])
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        skel = LeRobotDataset(args.out + "_skel_src", root=root)
        anames = skel.meta.features["action"]["names"]
        snames = skel.meta.features["observation.state"]["names"]
        gidx = next(i for i, n in enumerate(snames) if "gripper" in n)
        acts, states = {}, {}   # episode -> (n_frames, dim)
        for ep in sorted(walls):
            f0 = skel.meta.episodes[ep]["dataset_from_index"]
            f1 = skel.meta.episodes[ep]["dataset_to_index"]
            a = np.stack([skel[i]["action"].numpy() for i in range(f0, f1)])
            s = np.stack([skel[i]["observation.state"].numpy()
                          for i in range(f0, f1)])
            if len(a) != len(walls[ep]):
                print(f"  ep{ep}: {len(a)} frames vs {len(walls[ep])} wall "
                      f"clocks — trimming to min")
                m = min(len(a), len(walls[ep]))
                a, s, walls[ep] = a[:m], s[:m], walls[ep][:m]
            acts[ep], states[ep] = a, s
        # aperture calibration must target the MEASURED gripper (the camera
        # sees the physical jaws); actions stay the COMMANDED targets.
        mt = np.concatenate([walls[e] for e in sorted(walls)])
        mv = np.concatenate([states[e][:, gidx] for e in sorted(walls)])
        order = np.argsort(mt)
        mt, mv = mt[order], mv[order]
        episodes = [(ep, walls[ep], acts[ep], states[ep])
                    for ep in sorted(walls)]
    else:
        # glove mode: no skeleton. Joint actions still work — the IK is
        # warm-started from a corpus-typical configuration instead of the
        # episode's measured joints, and the names are the YAM's.
        anames = snames = [f"joint{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
        log = json.load(open(args.log))
        meas = [(e["t_wall"], e["gripper_pos"]) for e in log["events"]
                if e["event"] == "e8_meas"]
        mt, mv = map(np.array, zip(*meas))
        if log.get("units") == "mm":
            # recorded before the robot bench fit existed: caliper gap in mm.
            # Convert now so the gripper channel shares the robot corpus's
            # actuator units (cotraining would otherwise mix mm with ~0.2..1.0)
            try:
                g2a = json.load(open("calib/gap_to_actuator.json"))
            except OSError:
                raise SystemExit(
                    "glove log is in raw mm and calib/gap_to_actuator.json is "
                    "missing — run scripts/gap_bench.py on the robot first "
                    "(or pass --width-units-mm to build a glove-only dataset "
                    "whose gripper channel stays in mm)") \
                    if not args.width_units_mm else None
            if not args.width_units_mm:
                mv = g2a["a"] * mv + g2a["b"]
                print(f"== glove width holds converted mm -> actuator units "
                      f"(a={g2a['a']:.5f}, b={g2a['b']:.4f})")
            else:
                print("== glove width holds kept in RAW mm (--width-units-mm): "
                      "do NOT cotrain this dataset with the robot corpus")
        episodes = None  # timeline comes from the forward cam below

    # ---- forward-cam label stack (pipelines A and B) -------------------------
    pose_rows, ests = [], {}
    eef_on = args.eef_crop > 0
    global EEF_TGT
    EEF_TGT = np.zeros((1, 3))
    if eef_on:
        try:
            EEF_TGT = (np.array(json.load(open(args.eef_target))
                                ["p_body_mm"]).reshape(1, 3) / 1000.0)
        except Exception:
            print(f"(no {args.eef_target} — eef crop centers on the ball)")
    if eef_on and not (args.pipeline == "B"
                       or (args.pipeline == "C" and not args.wrist
                           and args.forward)):
        raise SystemExit("--eef-crop needs forward-cam observations "
                         "(pipeline B, or C' with --forward and no --wrist)")
    need_pose = args.pipeline in ("A", "B") or eef_on
    if need_pose:
        K, dist = L.load_intrinsics(args.intrinsics)
        T_body, dobj = L.load_bundle(args.bundle)
        ts_f, det_f = load_stem(args.forward, args.raw_dir, args.det_dir)
        markers = {**L.TAILS, **L.BASES} if args.pipeline == "B" else {}
        extra = L.load_gripper_bundle(args.gripper_bundle)
        print(f"== forward cam {args.forward}: fusing dodeca poses "
              f"(+{len(extra)} calibrated base markers) …")
        pose_rows = L.body_frames(det_f, ts_f, K, dist, T_body, dobj, markers,
                                  extra=extra)
        print(f"   {len(pose_rows)} frames with fused pose "
              f"({100 * len(pose_rows) / max(len(ts_f), 1):.1f}% of recording)")
        if args.pipeline == "B":
            fit_rows = pose_rows
            if args.fit_stems:
                # width estimators are body-frame geometry (camera-pose
                # independent): a short take may borrow a richer take's
                # frames for the FIT while predicting on its own frames
                K_x, dist_x = (L.load_intrinsics(args.fit_intrinsics)
                               if args.fit_intrinsics else (K, dist))
                for st in args.fit_stems.split(","):
                    ts_x, det_x = load_stem(st, args.raw_dir, args.det_dir)
                    fit_rows = fit_rows + L.body_frames(
                        det_x, ts_x, K_x, dist_x, T_body, dobj, markers,
                        extra=extra)
                print(f"== estimator fit augmented with "
                      f"{len(fit_rows) - len(pose_rows)} frames from "
                      f"{args.fit_stems}")
            print("== calibrating aperture estimators vs actuator:")
            # glove --log = sparse caliper holds: only frames within 0.3 s of
            # a measurement may enter the fit (see labels_lib._series)
            ests = L.fit_estimators(fit_rows, mt, mv,
                                    meas_tol=0.3 if args.log else None)
            print(f"   {len(ests)} GOOD estimators")

    # side camera: its own fused pose -> EEF-centered crop stream
    side = None
    if args.side:
        K2, dist2 = L.load_intrinsics(args.side_intrinsics)
        Tb2, dobj2 = L.load_bundle(args.bundle)
        ts_s, det_s = load_stem(args.side, args.raw_dir, args.det_dir)
        rows_s = L.body_frames(det_s, ts_s, K2, dist2, Tb2, dobj2, {},
                               extra=L.load_gripper_bundle(args.gripper_bundle))
        print(f"== side cam {args.side}: fused pose in {len(rows_s)}/"
              f"{len(ts_s)} frames")
        tP2 = np.array([r["t_wall"] for r in rows_s])
        tv2 = np.array([r["tvec"].ravel() for r in rows_s])
        q2 = np.array([quat_from_rvec(r["rvec"]) for r in rows_s])
        for i in range(1, len(q2)):
            if q2[i] @ q2[i - 1] < 0:
                q2[i] = -q2[i]
        tw_s = np.array([e["wall"] for e in ts_s])
        side = {"K": K2, "dist": dist2, "tP": tP2, "tv": tv2, "q": q2,
                "tw": tw_s,
                "reader": SeqReader(f"{args.raw_dir}/{args.side}.mkv")}

    # pipeline A gripper: tips in the wrist camera frame (camera IS body frame)
    tip_rows = []
    grip_wrist = args.pipeline == "A" or args.grip_from_wrist
    if args.grip_from_wrist and not (args.wrist and args.wrist_intrinsics):
        raise SystemExit("--grip-from-wrist needs --wrist and --wrist-intrinsics")
    if grip_wrist:
        Kw, distw = L.load_intrinsics(args.wrist_intrinsics)
        ts_w, det_w = load_stem(args.wrist, args.raw_dir, args.det_dir,
                                cache=tips_cache)
        tip_rows = L.body_frames(det_w, ts_w, Kw, distw, None, None, L.TIPS)
        print(f"== wrist cam {args.wrist}: tips in {len(tip_rows)} frames")
        fit_tip_rows = tip_rows
        if args.fit_wrist_stems:
            # tips are wrist-camera-frame geometry (the camera rides on the
            # body), so another session of the same unit can lend its hold
            # frames to the fit; predictions still use this take's frames
            for st in args.fit_wrist_stems.split(","):
                ts_x, det_x = load_stem(st, args.raw_dir, args.det_dir,
                                        cache=tips_cache)
                fit_tip_rows = fit_tip_rows + L.body_frames(
                    det_x, ts_x, Kw, distw, None, None, L.TIPS)
            print(f"== tip fit augmented with "
                  f"{len(fit_tip_rows) - len(tip_rows)} frames from "
                  f"{args.fit_wrist_stems}")
        print("== calibrating tip estimators vs actuator:")
        # 13<->14 as a PAIR too: the relative vector cancels common-mode
        # IPPE depth noise, usually grading tighter than either single tip
        tk = sorted(L.TIPS)   # tip PAIR ids (13/14 in v1; 16h5 0/1 in v2)
        ests = L.fit_estimators(fit_tip_rows, mt, mv,
                                tails={tk[0]: L.TIPS[tk[0]]},
                                bases={tk[1]: L.TIPS[tk[1]]},
                                singles=list(L.TIPS),
                                meas_tol=0.3 if args.log else None)

    # ---- moving label camera: body poses -> tile frame -----------------------
    series_rows = pose_rows if need_pose else []
    if need_pose and args.cam_world:
        if eef_on or args.side or args.fuse_side_pose:
            raise SystemExit("--cam-world: eef crop / side / fused-side "
                             "poses need camera-frame poses")
        cwj = json.load(open(args.cam_world))
        cw_rows = sorted((r for r in cwj["per_frame"]
                          if r["tiles"] >= args.cam_world_min_tiles
                          and r["rms"] <= args.cam_world_rms),
                         key=lambda r: r["k"])
        kc = np.array([r["k"] for r in cw_rows])
        tc = np.array([ts_f[k]["wall"] for k in kc])
        Tc = [np.array(r["T_cam_world"]) for r in cw_rows]   # world -> cam
        qc = np.array([quat_from_rvec(cv2.Rodrigues(T[:3, :3])[0])
                       for T in Tc])
        for i in range(1, len(qc)):
            if qc[i] @ qc[i - 1] < 0:
                qc[i] = -qc[i]
        pc = np.array([T[:3, 3] for T in Tc])
        series_rows = []
        n_exact = n_bridged = n_drop = 0
        for r in pose_rows:
            k, t = r["k"], r["t_wall"]
            j = int(np.searchsorted(kc, k))
            if j < len(kc) and kc[j] == k:
                T_cw = Tc[j]
                n_exact += 1
            else:
                if j == 0 or j >= len(kc) or tc[j] - tc[j - 1] > args.cam_world_bridge:
                    n_drop += 1
                    continue
                w = (t - tc[j - 1]) / (tc[j] - tc[j - 1])
                qi = (1 - w) * qc[j - 1] + w * qc[j]
                qi /= np.linalg.norm(qi)
                T_cw = np.eye(4)
                T_cw[:3, :3], _ = cv2.Rodrigues(rvec_from_quat(qi))
                T_cw[:3, 3] = (1 - w) * pc[j - 1] + w * pc[j]
                n_bridged += 1
            T_cb = np.eye(4)
            T_cb[:3, :3], _ = cv2.Rodrigues(np.asarray(r["rvec"], np.float64))
            T_cb[:3, 3] = np.asarray(r["tvec"], np.float64).ravel()
            T_wb = np.linalg.inv(T_cw) @ T_cb
            r2 = dict(r)
            r2["rvec"], _ = cv2.Rodrigues(T_wb[:3, :3])
            r2["tvec"] = T_wb[:3, 3].reshape(3, 1)
            series_rows.append(r2)
        print(f"== moving camera ({args.cam_world}): {len(cw_rows)} tile poses; "
              f"body poses carried to the tile frame: {n_exact} exact + "
              f"{n_bridged} bridged, {n_drop} dropped "
              f"({100 * len(series_rows) / max(len(pose_rows), 1):.1f}% kept)")
        if args.pose_fallback and args.pose_fixed_only:
            print(f"== --pose-fixed-only: dropping the {len(series_rows)} moving-camera poses")
            series_rows = []
        if args.pose_fallback:
            Kx, dx = L.load_intrinsics(args.pose_fallback_intrinsics)
            ts_x, det_x = load_stem(args.pose_fallback, args.raw_dir,
                                    args.det_dir)
            rows_x = L.body_frames(det_x, ts_x, Kx, dx, T_body, dobj, {},
                                   extra=extra)
            T_wx = np.linalg.inv(np.array(json.load(open(
                args.pose_fallback_world))["T_cam_world"]))
            t_have = np.array(sorted(r["t_wall"] for r in series_rows))
            n_add = 0
            for r in rows_x:
                t = r["t_wall"]
                j = int(np.searchsorted(t_have, t))
                near = min(abs(t_have[j - 1] - t) if j > 0 else np.inf,
                           abs(t_have[j] - t) if j < len(t_have) else np.inf)
                if near <= args.pose_fallback_tol:
                    continue
                T_cb = np.eye(4)
                T_cb[:3, :3], _ = cv2.Rodrigues(np.asarray(r["rvec"], np.float64))
                T_cb[:3, 3] = np.asarray(r["tvec"], np.float64).ravel()
                T_wb = T_wx @ T_cb
                r2 = dict(r)
                r2["rvec"], _ = cv2.Rodrigues(T_wb[:3, :3])
                r2["tvec"] = T_wb[:3, 3].reshape(3, 1)
                r2["fallback"] = True
                series_rows.append(r2)
                n_add += 1
            series_rows.sort(key=lambda r: r["t_wall"])
            print(f"== pose fallback {args.pose_fallback}: {len(rows_x)} fixed-cam "
                  f"poses, {n_add} added where the head chain had none "
                  f"(series now {len(series_rows)} rows, "
                  f"{100 * n_add / max(len(series_rows), 1):.1f}% fallback)")

    # ---- per-camera-frame label series ---------------------------------------
    if need_pose:
        tP = np.array([r["t_wall"] for r in series_rows])
        tvecs = np.array([r["tvec"].ravel() for r in series_rows])
        quats = np.array([quat_from_rvec(r["rvec"]) for r in series_rows])
        for i in range(1, len(quats)):          # hemisphere alignment
            if quats[i] @ quats[i - 1] < 0:
                quats[i] = -quats[i]
        grip_src = tip_rows if grip_wrist else pose_rows
        gt, gv = [], []
        for r in grip_src:
            g, _ = L.predict_aperture(r["pb"], ests)
            if g is not None:
                gt.append(r["t_wall"])
                gv.append(g)
        gt, gv = np.array(gt), np.array(gv)
        print(f"== gripper labels on {len(gt)} camera frames")
        if args.grip_from_skeleton:
            if not args.skeleton:
                raise SystemExit("--grip-from-skeleton needs --skeleton")
            gi_a = next(i for i, n in enumerate(anames) if "gripper" in n)
            src = acts if args.grip_from_skeleton == "commanded" else states
            gt = np.concatenate([walls[e] for e in sorted(walls)])
            gv = np.concatenate([src[e][:, gi_a if args.grip_from_skeleton == "commanded" else gidx]
                                 for e in sorted(walls)])
            o_ = np.argsort(gt)
            gt, gv = gt[o_], gv[o_]
            print(f"== gripper labels REPLACED by the skeleton's "
                  f"{args.grip_from_skeleton} actuator channel "
                  f"({len(gt)} frames, p2/p50 "
                  f"{np.percentile(gv, [2, 50]).round(3).tolist()})")
        if args.grip_smooth and len(gv) > args.grip_smooth:
            from scipy.ndimage import median_filter
            order = np.argsort(gt)
            gv_s = median_filter(gv[order], size=args.grip_smooth, mode="nearest")
            gv = gv_s[np.argsort(order)]
            print(f"== gripper series median-filtered over {args.grip_smooth} frames")
        if args.grip_map:
            gm = json.load(open(args.grip_map))
            gv = np.interp(gv, np.array(gm["x_width"]), np.array(gm["y_cmd"]))
            print(f"== gripper series mapped through {args.grip_map}: "
                  f"p2/p50/p98 -> {np.percentile(gv, [2, 50, 98]).round(3).tolist()}")
        if args.fuse_side_grip:
            if args.pipeline != "B":
                raise SystemExit("--fuse-side-grip is a pipeline-B option")
            K2f, d2f = L.load_intrinsics(args.side_intrinsics)
            ts_sf, det_sf = load_stem(args.fuse_side_grip, args.raw_dir,
                                      args.det_dir)
            rows_sf = L.body_frames(det_sf, ts_sf, K2f, d2f, T_body, dobj,
                                    {**L.TAILS, **L.BASES}, extra=extra)
            print(f"== fuse-side {args.fuse_side_grip}: fused pose in "
                  f"{len(rows_sf)}/{len(ts_sf)} frames; calibrating its own "
                  "aperture estimators:")
            rows_sf_fit = rows_sf
            if args.fit_side_stems:
                for st in args.fit_side_stems.split(","):
                    ts_x, det_x = load_stem(st, args.raw_dir, args.det_dir)
                    rows_sf_fit = rows_sf_fit + L.body_frames(
                        det_x, ts_x, K2f, d2f, T_body, dobj,
                        {**L.TAILS, **L.BASES}, extra=extra)
            ests_sf = L.fit_estimators(rows_sf_fit, mt, mv,
                                       meas_tol=0.3 if args.log else None)
            gt2, gv2 = [], []
            for r in rows_sf:
                g, _ = L.predict_aperture(r["pb"], ests_sf)
                if g is not None:
                    gt2.append(r["t_wall"])
                    gv2.append(g)
            gt2, gv2 = np.array(gt2), np.array(gv2)
            # forward stays the primary instrument: only fill its gaps
            if len(gt) and len(gt2):
                far = np.abs(gt2[:, None] - gt[None, :]).min(1) > args.label_tol
                gt2, gv2 = gt2[far], gv2[far]
            if len(gt2):
                o = np.argsort(np.concatenate([gt, gt2]))
                gtF = np.concatenate([gt, gt2])[o]
                gvF = np.concatenate([gv, gv2])[o]
            else:
                gtF, gvF = gt.copy(), gv.copy()
            print(f"== side grip fusion: +{len(gt2)} gap-filling samples "
                  f"-> {len(gtF)} fused")
            if not args.emit_quad:
                # legacy single-output fusion: fused series IS the series
                gt, gv = gtF, gvF
        tPF, tvF, qsF = tP, tvecs, quats
        if args.fuse_side_pose:
            # twin POSE fusion (Bjt only — Bj stays single-camera pure):
            # twin-frame body poses -> forward frame through the two
            # hand-eyes, filling forward pose gaps (cable occlusion,
            # frame-edge clipping). Validated 2026-09-14: ep26 pose
            # 25.7->91.6%, fused-sample accuracy ~13mm median vs FK.
            if not args.fuse_side_grip:
                raise SystemExit("--fuse-side-pose needs --fuse-side-grip "
                                 "(provides the twin detections)")
            heF = json.load(open(__import__("os").environ.get("HANDEYE", "calib/handeye.json")))
            heT = json.load(open(args.fuse_side_pose))
            Xft = np.array(heF["T_cam_base"]) @ np.linalg.inv(
                np.array(heT["T_cam_base"]))
            at, atv, aq = [], [], []
            for r in rows_sf:
                t = r["t_wall"]
                if len(tP) and np.abs(tP - t).min() <= args.label_tol:
                    continue
                Rt, _ = cv2.Rodrigues(r["rvec"])
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = Rt, r["tvec"].ravel()
                Mf = Xft @ M
                rv2, _ = cv2.Rodrigues(Mf[:3, :3])
                at.append(t)
                atv.append(Mf[:3, 3])
                aq.append(quat_from_rvec(rv2))
            if at:
                o = np.argsort(np.concatenate([tP, at]))
                tPF = np.concatenate([tP, at])[o]
                tvF = np.concatenate([tvecs, np.array(atv)])[o]
                qsF = np.concatenate([quats, np.array(aq)])[o]
                for i in range(1, len(qsF)):
                    if qsF[i] @ qsF[i - 1] < 0:
                        qsF[i] = -qsF[i]
            print(f"== side POSE fusion: +{len(at)} gap samples "
                  f"-> {len(tPF)} fused")
        if (args.pipeline != "C" and (len(tP) < 2 or len(gt) < 2)) \
                or (args.pipeline == "C" and eef_on and len(tP) < 2):
            # a stub take (aborted session) overlaps no episodes: nothing
            # to emit, exit cleanly so multi-take chains continue
            print("== no usable label frames for this stem — 0 episodes")
            return

    # ---- output timeline ------------------------------------------------------
    if episodes is None and args.episode_marks:
        # glove mode with a real session: wall-clock episode boundaries from
        # the recorder's sidecar. Clip to LABEL coverage only (tP) — in glove
        # mode mt/mv is a short caliper-calibration window, not a full series.
        marks = json.load(open(args.episode_marks))["episodes"]
        episodes = []
        for m in marks:
            t0 = max(m["t_start"], tP[0])
            t1 = min(m["t_end"], tP[-1])
            if t1 - t0 < 1.0:
                print(f"  ep{m['episode_index']}: outside label coverage "
                      f"({t1 - t0:.1f}s usable) — SKIPPED")
                continue
            episodes.append((m["episode_index"],
                             np.arange(t0, t1, 1.0 / (args.fps * args.slow)),
                             None, None))
        print(f"== {len(episodes)}/{len(marks)} marked episodes inside "
              f"label coverage")
        if not episodes:
            raise SystemExit("no marked episode overlaps the label stack")
    elif episodes is None:
        t0, t1 = max(tP[0], mt[0]), min(tP[-1], mt[-1])
        tq = np.arange(t0, t1, 1.0 / (args.fps * args.slow))
        episodes = [(0, tq, None, None)]
    if args.split_gap > 0 and args.pipeline != "C":
        # blind-region split: a frame is blind when no pose or grip label
        # lies within --label-tol of it; blind runs longer than --split-gap
        # cut the episode, the visible stretches become episodes of their own
        def nearest(ts, tq):
            if len(ts) == 0:
                return np.full(len(tq), np.inf)
            i = np.clip(np.searchsorted(ts, tq), 1, len(ts) - 1)
            return np.minimum(np.abs(tq - ts[i - 1]), np.abs(tq - ts[i]))
        dt = 1.0 / (args.fps * args.slow)
        split_eps, n_tot, n_gap, n_short, n_interp, n_segs = [], 0, 0, 0, 0, 0
        for ep, twq, a_, s_ in episodes:
            blind = (nearest(tP, twq) > args.label_tol) \
                | (nearest(gt, twq) > args.label_tol)
            n_tot += len(twq)
            cuts, i = [], 0
            while i < len(twq):
                if blind[i]:
                    j = i
                    while j < len(twq) and blind[j]:
                        j += 1
                    if (j - i) * dt > args.split_gap:
                        cuts.append((i, j))
                        n_gap += j - i
                    else:
                        n_interp += j - i
                    i = j
                else:
                    i += 1
            bounds, prev = [], 0
            for a, b in cuts:
                bounds.append((prev, a))
                prev = b
            bounds.append((prev, len(twq)))
            segs = [(a, b) for a, b in bounds if b > a]
            kept = [(a, b) for a, b in segs if (b - a) * dt >= args.min_segment]
            n_short += sum(b - a for a, b in segs if (a, b) not in kept)
            n_segs += len(kept)
            whole = len(kept) == 1 and kept[0] == (0, len(twq))
            for k, (a, b) in enumerate(kept):
                split_eps.append((ep if whole else f"{ep}s{k}",
                                  twq[a:b], None, None))
        print(f"== blind split (gap > {args.split_gap}s, segment >= "
              f"{args.min_segment}s): {len(episodes)} episodes -> {n_segs} "
              f"segments; {n_tot} frames: {n_gap} dropped in blind gaps "
              f"({100 * n_gap / max(n_tot, 1):.1f}%), {n_short} dropped in "
              f"short segments ({100 * n_short / max(n_tot, 1):.1f}%), "
              f"{n_interp} interpolated ({100 * n_interp / max(n_tot, 1):.1f}%)")
        episodes = split_eps
    if args.pipeline == "B":
        obs_stem = args.forward
    elif args.pipeline == "A":
        obs_stem = args.wrist
    else:  # C: wrist if present, else forward (the C' control for B)
        obs_stem = args.wrist or args.forward
    if args.obs_stem:
        obs_stem = args.obs_stem
        print(f"== observation stream overridden: {obs_stem}")
    ts_obs = json.load(open(f"{args.raw_dir}/{obs_stem}_ts.json"))
    tw_obs = np.array([e["wall"] for e in ts_obs])
    reader = SeqReader(f"{args.raw_dir}/{obs_stem}.mkv")
    # dual-obs pipeline C (Cwf reference rig): wrist PRIMARY + forward
    # full-frame SECONDARY at the same emitted size (DP requires equal
    # image shapes across streams)
    obs2 = None
    if args.pipeline == "C" and args.wrist and args.forward:
        ts2 = json.load(open(f"{args.raw_dir}/{args.forward}_ts.json"))
        obs2 = {"tw": np.array([e["wall"] for e in ts2]),
                "reader": SeqReader(f"{args.raw_dir}/{args.forward}.mkv"),
                "key": "observation.images."
                       + args.forward.split("_")[-1].split(".")[0]}

    # ---- emit -----------------------------------------------------------------
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    jact = None
    ik_seed0 = None
    if args.joint_actions:
        if args.pipeline not in ("A", "B"):
            raise SystemExit("--joint-actions needs pipeline A/B")
        if episodes[0][2] is None:
            try:
                ik_seed0 = np.array(json.load(open("calib/ik_seed_yam.json"))
                                    ["q_seed"], dtype=float)
            except OSError:
                raise SystemExit("--joint-actions without --skeleton needs "
                                 "calib/ik_seed_yam.json (IK warm start)")
            print("== glove mode joint actions: IK warm-started from "
                  "calib/ik_seed_yam.json, first frame settled unguarded")
        sys.path.insert(0, str(Path.home() / "co/i2rt"))
        he = json.load(open(__import__("os").environ.get("HANDEYE", "calib/handeye.json")))
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import (ArmType, GripperType,
                                       combine_arm_and_gripper_xml)
        ji_s = [i for i, n in enumerate(snames) if n.startswith("joint")]
        gi_s = next(i for i, n in enumerate(anames) if "gripper" in n)
        ja_s = [i for i, n in enumerate(anames) if n.startswith("joint")]
        if args.cam_world and not args.base_world:
            raise SystemExit("--cam-world --joint-actions needs --base-world "
                             "(tile frame -> robot base; base_world_fit.py)")
        Ainv_j = np.linalg.inv(np.array(he["T_cam_base"]))
        if args.base_world:
            Ainv_j = np.linalg.inv(np.array(
                json.load(open(args.base_world))["T_world_base"]))
            print(f"== joint actions through the tile frame: "
                  f"T_base_world from {args.base_world}")
        jact = {"Ainv": Ainv_j,
                "Binv": np.linalg.inv(np.array(he["T_flange_body"])),
                "kin": Kinematics(combine_arm_and_gripper_xml(
                    ArmType.YAM, GripperType.NO_GRIPPER), "grasp_site"),
                "ji_s": ji_s, "gi_s": gi_s, "ja_s": ja_s, "seed": None}
    if args.pipeline == "C":
        # action = COMMANDED target (leader), state = MEASURED follower pose
        adim, anames_out = episodes[0][2].shape[1], anames
        sdim, snames_out = episodes[0][3].shape[1], snames
    elif jact is not None:
        # fiducial labels expressed as ABSOLUTE JOINT actions (offline IK);
        # state = measured joints -> action space/state match C' exactly
        adim, anames_out = len(anames), anames
        sdim, snames_out = len(snames), snames
    else:
        adim, anames_out = 7, POSE_NAMES
        sdim, snames_out = adim, anames_out
    features = {
        "action": {"dtype": "float32", "shape": (adim,), "names": anames_out},
        "observation.state": {"dtype": "float32", "shape": (sdim,),
                              "names": snames_out},
        f"observation.images.{obs_stem.split('_')[-1].split('.')[0]}": {
            "dtype": "video", "shape": (oh, ow, 3),
            "names": ["height", "width", "channels"]},
    }
    if obs2 is not None:
        features[obs2["key"]] = {
            "dtype": "video", "shape": (oh, ow, 3),
            "names": ["height", "width", "channels"]}
    if eef_on:
        features["observation.images.eef"] = {
            "dtype": "video", "shape": (args.eef_size, args.eef_size, 3),
            "names": ["height", "width", "channels"]}
    if side is not None:
        features["observation.images.side"] = {
            "dtype": "video", "shape": (args.eef_size, args.eef_size, 3),
            "names": ["height", "width", "channels"]}
    imkey = f"observation.images.{obs_stem.split('_')[-1].split('.')[0]}"
    ROT = {"cw": cv2.ROTATE_90_CLOCKWISE,
           "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
           "180": cv2.ROTATE_180}.get(args.rotate)
    import os as _os

    def _open(repo, feats):
        outdir = args.out_root or _os.path.expanduser(
            f"~/.cache/huggingface/lerobot/{repo}")
        if args.append and _os.path.exists(f"{outdir}/meta/tasks.parquet"):
            print(f"== appending to existing {repo}")
            return LeRobotDataset.resume(repo_id=repo, root=outdir,
                                         image_writer_threads=4)
        if args.append and _os.path.exists(outdir):
            # 0-episode shell from a stub take: replace it
            import shutil as _sh
            _sh.rmtree(outdir)
            print(f"== replacing empty shell {repo}")
        return LeRobotDataset.create(repo_id=repo, fps=round(args.fps),
                                     features=feats, root=args.out_root,
                                     robot_type="yam", use_videos=True,
                                     image_writer_threads=4)

    quad_base = None
    if args.emit_quad:
        if args.pipeline != "B" or episodes[0][2] is None \
                or jact is None or not args.fuse_side_grip:
            raise SystemExit("--emit-quad needs pipeline B, --skeleton, "
                             "--joint-actions and --fuse-side-grip")
        if args.emit_c:
            raise SystemExit("--emit-quad supersedes --emit-c")
        quad_base = args.out
        args.out = quad_base + "_Bj"

    ds = _open(args.out, features)
    ds2 = None
    if args.emit_c:
        if args.pipeline != "B" or episodes[0][2] is None:
            raise SystemExit("--emit-c needs pipeline B with a --skeleton")
        feats_c = dict(features)
        feats_c["action"] = {"dtype": "float32",
                             "shape": (episodes[0][2].shape[1],),
                             "names": anames}
        feats_c["observation.state"] = {"dtype": "float32",
                                        "shape": (episodes[0][3].shape[1],),
                                        "names": snames}
        ds2 = _open(args.emit_c, feats_c)
    quad = None
    if quad_base:
        feats_pose = dict(features)
        feats_pose["action"] = {"dtype": "float32", "shape": (7,),
                                "names": POSE_NAMES}
        feats_pose["observation.state"] = {"dtype": "float32", "shape": (7,),
                                           "names": POSE_NAMES}
        feats_c = dict(features)
        feats_c["action"] = {"dtype": "float32",
                             "shape": (episodes[0][2].shape[1],),
                             "names": anames}
        feats_c["observation.state"] = {"dtype": "float32",
                                        "shape": (episodes[0][3].shape[1],),
                                        "names": snames}
        quad = {"B": _open(quad_base + "_B", feats_pose),
                "Cf": _open(quad_base + "_Cf", feats_c),
                "Bjt": _open(quad_base + "_Bjt", dict(features))}
        print(f"== quad emit: {quad_base}_{{Bj,B,Cf,Bjt}}")
    print(f"== emitting to {ds.root.resolve()}"
          + (f" + {args.emit_c}" if ds2 else ""))

    def _ik_guarded(J, rv_, tv_):
        """warm-started IK + branch-flip guard + slew clamp (state in J)."""
        Rm, _ = cv2.Rodrigues(np.asarray(rv_, dtype=np.float64))
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = Rm, tv_
        F_t = J["Ainv"] @ M @ J["Binv"]
        _, qik = J["kin"].ik(F_t, "grasp_site", init_q=J["seed"].copy(),
                             pos_threshold=1e-3, ori_threshold=1e-2,
                             max_iters=60)
        jstep = float(np.max(np.abs(np.array(qik[:6]) - J["seed"])))
        if jstep > 0.35 and J["Fprev"] is not None:
            q2 = J["seed"].copy()
            F0 = J["Fprev"]
            rrel, _ = cv2.Rodrigues(F0[:3, :3].T @ F_t[:3, :3])
            NS = 8
            for k in range(1, NS + 1):
                Fk = np.eye(4)
                Rk, _ = cv2.Rodrigues(rrel * (k / NS))
                Fk[:3, :3] = F0[:3, :3] @ Rk
                Fk[:3, 3] = F0[:3, 3] + (F_t[:3, 3] - F0[:3, 3]) * (k / NS)
                _, qs2 = J["kin"].ik(Fk, "grasp_site", init_q=q2.copy(),
                                     pos_threshold=1e-3, ori_threshold=1e-2,
                                     max_iters=120)
                q2 = np.array(qs2[:6], dtype=float)
            j2 = float(np.max(np.abs(q2 - J["seed"])))
            if j2 < jstep:
                qik, jstep = q2, j2
                J["fixed"] += 1
        if jstep > 0.35:
            d = np.array(qik[:6], dtype=float) - J["seed"]
            qik = J["seed"] + d * (0.3 / max(jstep, 1e-9))
            J["jumps"] += 1
        J["Fprev"] = F_t
        J["seed"] = np.array(qik[:6], dtype=float)
        return qik

    jactT = dict(jact) if (quad is not None and args.fuse_side_pose
                           and jact is not None) else None

    excl = {int(x) for x in args.exclude_eps.split(",") if x.strip()}
    force = {int(x) for x in args.force_eps.split(",") if x.strip()}
    emitted = 0
    for ep, twq, ep_acts, ep_states in episodes:
        if ep in excl:
            print(f"  ep{ep}: EXCLUDED (--exclude-eps)")
            continue
        # subsample the skeleton timeline to the output fps
        if ep_acts is not None:
            skel_fps = (len(twq) - 1) / max(twq[-1] - twq[0], 1e-9)
            step = max(int(round(skel_fps / (args.fps * args.slow))), 1)
            sel = np.arange(0, len(twq), step)
            twq_s, acts_s, states_s = twq[sel], ep_acts[sel], ep_states[sel]
        else:
            twq_s, acts_s, states_s = twq, None, None
        gaps_p = gaps_g = gaps_o = gaps_gF = gaps_pF = 0
        n_ep = 0
        for j, t in enumerate(twq_s):
            if args.max_frames and emitted >= args.max_frames:
                break
            tv = None
            if args.pipeline == "C":
                vec = acts_s[j].astype(np.float32)
                svec = states_s[j].astype(np.float32)
                if eef_on:
                    tv, q = interp_pose([t], tP, tvecs, quats)
            else:
                near_p = np.abs(tP - t).min() if len(tP) else np.inf
                near_g = np.abs(gt - t).min() if len(gt) else np.inf
                gaps_p += near_p > args.label_tol
                gaps_g += near_g > args.label_tol
                tv, q = interp_pose([t], tP, tvecs, quats)
                rv = rvec_from_quat(q[0])
                g = float(np.interp(t, gt, gv))
                vec = np.concatenate([tv[0], rv, [g]]).astype(np.float32)
                svec = vec
                pose_vec, gF = vec, None
                if quad is not None:
                    near_gF = np.abs(gtF - t).min() if len(gtF) else np.inf
                    gaps_gF += near_gF > args.label_tol
                    gF = float(np.interp(t, gtF, gvF))
                    if args.fuse_side_pose:
                        near_pF = np.abs(tPF - t).min() if len(tPF) \
                            else np.inf
                        gaps_pF += near_pF > args.label_tol
                if jact is not None:
                    if j == 0:  # per-episode warm-start from measured joints
                        for J in ([jact, jactT] if jactT else [jact]):
                            if states_s is not None:
                                J["seed"] = states_s[0][J["ji_s"]].astype(float)
                            else:
                                # glove: no measured joints. Settle the first
                                # frame from the corpus-typical seed WITHOUT
                                # the slew clamp (it would otherwise creep
                                # toward the true pose at 0.3 rad/frame)
                                J["seed"] = ik_seed0.copy()
                                Rm0, _ = cv2.Rodrigues(np.asarray(rv, np.float64))
                                M0 = np.eye(4)
                                M0[:3, :3], M0[:3, 3] = Rm0, tv[0]
                                F0 = J["Ainv"] @ M0 @ J["Binv"]
                                for _ in range(3):
                                    _, q0 = J["kin"].ik(
                                        F0, "grasp_site",
                                        init_q=J["seed"].copy(),
                                        pos_threshold=1e-3,
                                        ori_threshold=1e-2, max_iters=300)
                                    J["seed"] = np.array(q0[:6], dtype=float)
                            J["Fprev"] = None
                            J["jumps"] = J["fixed"] = 0
                    qik = _ik_guarded(jact, rv, tv[0])
                    vec = np.zeros(adim, dtype=np.float32)
                    vec[jact["ja_s"]] = jact["seed"]
                    vec[jact["gi_s"]] = g
                    # glove-purity: state is ALSO fiducial-derived (the IK
                    # joints), so the arm uses only what a glove provides
                    svec = vec.copy()
                    if jactT is not None:
                        # Bjt's own chain on the twin-FUSED pose series
                        tvT, qT = interp_pose([t], tPF, tvF, qsF)
                        _ik_guarded(jactT, rvec_from_quat(qT[0]), tvT[0])
            ki = int(np.searchsorted(tw_obs, t))
            ki = min(max(ki, 0), len(tw_obs) - 1)
            if ki > 0 and abs(tw_obs[ki - 1] - t) < abs(tw_obs[ki] - t):
                ki -= 1
            miss1 = abs(tw_obs[ki] - t) > args.label_tol
            miss2 = False
            if obs2 is not None:
                k2 = int(np.clip(np.searchsorted(obs2["tw"], t), 0,
                                 len(obs2["tw"]) - 1))
                if k2 > 0 and abs(obs2["tw"][k2 - 1] - t) <                         abs(obs2["tw"][k2] - t):
                    k2 -= 1
                miss2 = abs(obs2["tw"][k2] - t) > args.label_tol
            gaps_o += miss1 or miss2
            img = reader.get(ki)
            if img is None:
                break
            H_s, W_s = img.shape[:2]        # sensor dims (pre-rotation)
            if ROT is not None:
                img = cv2.rotate(img, ROT)
            frame = {"action": vec, "observation.state": svec,
                     "task": args.task}
            if eef_on and tv is not None:
                # crop around the between-tips target (calib/eef_target.json,
                # body frame), projected into the obs camera — the "virtual
                # wrist cam" stream. Same recipe at train and deploy (the
                # robot carries the same markers).
                px, _ = cv2.projectPoints(EEF_TGT,
                                          rvec_from_quat(q[0]),
                                          tv[0].astype(np.float64), K, dist)
                cx, cy = px.ravel()          # sensor coords -> map through
                if ROT == cv2.ROTATE_90_CLOCKWISE:        # the obs rotation
                    cx, cy = H_s - 1 - cy, cx
                elif ROT == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    cx, cy = cy, W_s - 1 - cx
                elif ROT == cv2.ROTATE_180:
                    cx, cy = W_s - 1 - cx, H_s - 1 - cy
                H0, W0 = img.shape[:2]
                cr = min(args.eef_crop, W0, H0)
                x0 = int(min(max(cx - cr / 2, 0), W0 - cr))
                y0 = int(min(max(cy - cr / 2, 0), H0 - cr))
                eef = cv2.resize(img[y0:y0 + cr, x0:x0 + cr],
                                 (args.eef_size, args.eef_size),
                                 interpolation=cv2.INTER_AREA)
                frame["observation.images.eef"] = \
                    cv2.cvtColor(eef, cv2.COLOR_BGR2RGB)
            if args.square_obs:
                Hc, Wc = img.shape[:2]
                sq = min(Hc, Wc)
                xq, yq = (Wc - sq) // 2, (Hc - sq) // 2
                img = img[yq:yq + sq, xq:xq + sq]
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
            frame[imkey] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if obs2 is not None:
                img2 = obs2["reader"].get(k2)
                if img2 is None:
                    break
                img2 = cv2.resize(img2, (ow, oh),
                                  interpolation=cv2.INTER_AREA)
                frame[obs2["key"]] = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            if side is not None:
                kj = int(np.clip(np.searchsorted(side["tw"], t), 0,
                                 len(side["tw"]) - 1))
                if kj > 0 and abs(side["tw"][kj - 1] - t) < \
                        abs(side["tw"][kj] - t):
                    kj -= 1
                simg = side["reader"].get(kj)
                if simg is None:
                    break
                Hs, Ws = simg.shape[:2]
                cr2 = min(args.eef_crop or 1200, Ws, Hs)
                if len(side["tP"]):
                    stv, sq_ = interp_pose([t], side["tP"], side["tv"],
                                           side["q"])
                    px2, _ = cv2.projectPoints(EEF_TGT,
                                               rvec_from_quat(sq_[0]),
                                               stv[0].astype(np.float64),
                                               side["K"], side["dist"])
                    cx2, cy2 = px2.ravel()
                else:
                    cx2, cy2 = Ws / 2, Hs / 2
                x2 = int(min(max(cx2 - cr2 / 2, 0), Ws - cr2))
                y2 = int(min(max(cy2 - cr2 / 2, 0), Hs - cr2))
                sc = cv2.resize(simg[y2:y2 + cr2, x2:x2 + cr2],
                                (args.eef_size, args.eef_size),
                                interpolation=cv2.INTER_AREA)
                frame["observation.images.side"] = \
                    cv2.cvtColor(sc, cv2.COLOR_BGR2RGB)
            frame2 = None
            if ds2 is not None:
                # copy BEFORE ds.add_frame — lerobot mutates the dict
                frame2 = dict(frame)
                frame2["action"] = acts_s[j].astype(np.float32)
                frame2["observation.state"] = states_s[j].astype(np.float32)
            qframes = None
            if quad is not None:
                # copies BEFORE any add_frame — lerobot mutates the dict
                if jactT is not None:
                    vBjt = np.zeros_like(vec)
                    vBjt[jactT["ja_s"]] = jactT["seed"]
                else:
                    vBjt = vec.copy()
                vBjt[jact["gi_s"]] = gF
                qframes = {
                    "B": {**frame, "action": pose_vec,
                          "observation.state": pose_vec.copy()},
                    "Cf": {**frame, "action": acts_s[j].astype(np.float32),
                           "observation.state":
                               states_s[j].astype(np.float32)},
                    "Bjt": {**frame, "action": vBjt,
                            "observation.state": vBjt.copy()},
                }
            ds.add_frame(frame)
            if frame2 is not None:
                ds2.add_frame(frame2)
            if qframes is not None:
                for kq, dq in quad.items():
                    dq.add_frame(qframes[kq])
            n_ep += 1
            emitted += 1
        if n_ep == 0:
            continue
        cov_p, cov_g = 1 - gaps_p / n_ep, 1 - gaps_g / n_ep
        cov_o = 1 - gaps_o / n_ep
        if cov_o < args.min_coverage:
            # every pipeline needs obs frames — an episode whose camera
            # recording is missing (overwritten/pruned) must not emit
            # nearest-garbage images
            print(f"  ep{ep}: SKIPPED — obs-camera coverage "
                  f"{100*cov_o:.1f}% (< {100*args.min_coverage:.0f}%)")
            ds.clear_episode_buffer()
            if ds2 is not None:
                ds2.clear_episode_buffer()
            if quad is not None:
                for dq in quad.values():
                    dq.clear_episode_buffer()
            emitted -= n_ep
            continue
        plain_ok = (args.pipeline == "C"
                    or min(cov_p, cov_g) >= args.min_coverage
                    or ep in force)
        if quad is not None:
            cov_gF = 1 - gaps_gF / n_ep
            cov_pF = (1 - gaps_pF / n_ep) if args.fuse_side_pose else cov_p
            fused_ok = (min(cov_pF, cov_gF) >= args.min_coverage
                        or ep in force)
            if plain_ok:
                ds.save_episode()
                quad["B"].save_episode()
                quad["Cf"].save_episode()
            else:
                ds.clear_episode_buffer()
                quad["B"].clear_episode_buffer()
                quad["Cf"].clear_episode_buffer()
                print(f"  ep{ep}: Bj/B/Cf SKIPPED — pose {100*cov_p:.1f}%, "
                      f"grip {100*cov_g:.1f}% (< {100*args.min_coverage:.0f}%)")
            if fused_ok:
                quad["Bjt"].save_episode()
            else:
                quad["Bjt"].clear_episode_buffer()
                print(f"  ep{ep}: Bjt SKIPPED — pose {100*cov_p:.1f}%, "
                      f"fused grip {100*cov_gF:.1f}%")
            if not plain_ok and not fused_ok:
                emitted -= n_ep
                continue
            print(f"  ep{ep}: {n_ep} frames  pose {100*cov_p:.1f}%  "
                  f"grip {100*cov_g:.1f}%  fusedgrip {100*cov_gF:.1f}%"
                  + (f"  fusedpose {100*cov_pF:.1f}%"
                     if args.fuse_side_pose else "")
                  + ("  [FORCED]" if ep in force
                     and min(cov_p, cov_g) < args.min_coverage else "")
                  + (f"  [IK: {jact['fixed']} flips healed, "
                     f"{jact['jumps']} slew-clamped]"
                     if jact.get("fixed") or jact.get("jumps") else ""))
            if args.max_frames and emitted >= args.max_frames:
                break
            continue
        if not plain_ok:
            print(f"  ep{ep}: SKIPPED — pose coverage {100*cov_p:.1f}%, "
                  f"grip {100*cov_g:.1f}% (< {100*args.min_coverage:.0f}%)")
            ds.clear_episode_buffer()
            if ds2 is not None:
                ds2.clear_episode_buffer()
            emitted -= n_ep
            continue
        ds.save_episode()
        if ds2 is not None:
            ds2.save_episode()
        print(f"  ep{ep}: {n_ep} frames"
              + ("" if args.pipeline == "C" else
                 f"  pose coverage {100*cov_p:.1f}%  grip {100*cov_g:.1f}%")
              + ("  [FORCED past label gate]" if ep in force
                 and args.pipeline != "C"
                 and min(cov_p, cov_g) < args.min_coverage else "")
              + (f"  [IK: {jact['fixed']} flips healed, "
                 f"{jact['jumps']} slew-clamped]"
                 if jact is not None and (jact.get("fixed") or
                                          jact.get("jumps")) else ""))
        if args.max_frames and emitted >= args.max_frames:
            break
    ds.finalize()
    if ds2 is not None:
        ds2.finalize()
    if quad is not None:
        for dq in quad.values():
            dq.finalize()
    print(f"== done: {emitted} frames -> {ds.root.resolve()}"
          + (f" (+ {args.emit_c})" if ds2 else "")
          + (f" (+ {quad_base}_{{B,Cf,Bjt}})" if quad else ""))


if __name__ == "__main__":
    main()
