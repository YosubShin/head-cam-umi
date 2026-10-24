#!/usr/bin/env python3
"""One-command multi-pipeline recording session.

Orchestrates the three processes of a collection session:
  1. uvc_record on the forward B0587 (4K25) and wrist fisheye (1080p30),
     devices resolved by sysfs NAME (indices shuffle across replugs)
  2. record_yam_teleop --no_cameras in the foreground (Enter/r/q as usual)
  3. clean shutdown: recorder exits -> camera recorders get SIGTERM and
     finish their remux (do NOT kill them; 4K remux takes a minute)

The host (yam_zmq_host) stays external — start it first, it owns the CAN
buses and typically outlives many sessions:
    ~/co/i2rt/.venv/bin/python yam_zmq_host.py \
        --leader_channel can_leader --follower_channel can_follower

Usage:
  python scripts/record_session.py --dataset yam_sess0 --num_episodes 5 \
      --episode_time 30 --task "pick red block"
Camera stems land in data/raw/<dataset>_b0587.* and <dataset>_fish.*;
afterwards run build_dataset.py per COLLECTION_RUNBOOK.md.
"""
import argparse
import atexit
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path

# Resolve the DIRECTORY, not the file: this script may be reached through a
# symlink from another working tree (pilot/scripts -> forward-cam-umi), and
# the data/calib roots must belong to the tree it was invoked from.
PILOT = Path(__file__).parent.parent.resolve()
TELEOP = Path("~/lerobot_alohamini/examples/yam/record_yam_teleop.py").expanduser()

# Cameras are pinned by USB PORT (/dev/v4l/by-path): the two B0587 twins
# share the serial "SN0001", so names can't tell them apart — if a camera
# moves to a different port, update its by-path here.
# (suffix, by-path device, sysfs-name fragment fallback, res, fps)
CAMERAS = [
    ("b0587", "/dev/v4l/by-path/pci-0000:44:00.0-usb-0:1:1.0-video-index0",
     "B0587", "3840x2160", "25"),          # original, portrait mount
    ("b0587b", "/dev/v4l/by-path/pci-0000:10:00.3-usb-0:2.1:1.0-video-index0",
     None, "3840x2160", "25"),             # twin, landscape (no name fallback:
    ("fish", "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1:1.0-video-index0",
     "1080P Low Light", "1920x1080", "30"),  # a fallback could grab the twin's sibling)
]


# Another machine: put your cameras in calib/cameras.json instead of editing
# the table above — a list of [suffix, by_path, name_fragment_or_null, "WxH",
# "fps"]. Find by-path links with `ls -l /dev/v4l/by-path/`.
_CAM_JSON = PILOT / "calib/cameras.json"
if _CAM_JSON.exists():
    CAMERAS = [tuple(c) for c in json.load(open(_CAM_JSON))]
    print(f"cameras from {_CAM_JSON.name}: {[c[0] for c in CAMERAS]}")


def find_dev(fragment):
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            name = open(f"{node}/name").read().strip()
        except OSError:
            continue
        if fragment in name:
            dev = f"/dev/{os.path.basename(node)}"
            # only index-0 nodes of a camera expose the video stream
            caps = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats"],
                                  capture_output=True, text=True).stdout
            if "MJPG" in caps or "YUYV" in caps:
                return dev
    return None


def find_other_b0587(exclude):
    """Fallback for the second B0587 (twin, now the head-worn glasses cam):
    the two share serial 'SN0001', so names can't tell them apart — but 'the
    B0587 that is NOT at any other camera's pinned path' is unique when both
    are present. Returns its by-path link, or None if there isn't exactly
    one candidate (then fail loudly rather than grab the wrong camera)."""
    cands = []
    for bp in sorted(glob.glob("/dev/v4l/by-path/*video-index0")):
        if bp in exclude:
            continue
        node = os.path.basename(os.path.realpath(bp))
        try:
            name = open(f"/sys/class/video4linux/{node}/name").read()
        except OSError:
            continue
        if "B0587" in name:
            cands.append(bp)
    return cands[0] if len(cands) == 1 else None


def _tail_frame(stem):
    """Latest complete JPEG from the newest growing segment, decoded.
    Never frame 0 — pre-ritual boot frames are legitimately blown."""
    import cv2
    import numpy as _np
    segs = sorted((PILOT / "data/raw").glob(f".{stem}_seg*.mkv"))
    if not segs or segs[-1].stat().st_size < 500_000:
        return None
    raw = segs[-1].read_bytes()[-30_000_000:]
    i2 = raw.rfind(b"\xff\xd8\xff")
    i1 = raw.rfind(b"\xff\xd8\xff", 0, max(i2, 0))
    i = i1 if i1 >= 0 else i2
    j = raw.find(b"\xff\xd9", i) if i >= 0 else -1
    if i < 0 or j <= i:
        return None
    return cv2.imdecode(_np.frombuffer(raw[i:j + 2], _np.uint8),
                        cv2.IMREAD_COLOR)


_DETS = {}


def _frame_health(stem, img):
    """(brightness mean, decoded fiducial count). The marker count is the
    FUNCTIONAL check: exposure, focus, lens smudge, and framing all gate
    it, and the rig guarantees markers in view (dodeca/tails for the
    forward cams, 16h5 tips for the wrist fisheye)."""
    import cv2
    kind = "16h5" if "fish" in stem else "4x4"
    if kind not in _DETS:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        dic = (cv2.aruco.DICT_APRILTAG_16h5 if kind == "16h5"
               else cv2.aruco.DICT_4X4_50)
        _DETS[kind] = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dic), p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETS[kind].detectMarkers(gray)
    if ids is None or not len(ids):
        return float(gray.mean()), 0, None, None
    # contrast at the decoded markers: a stuck ISP (seen on the wrist
    # fisheye, fixed only by replugging) lifts the black cells to ~180 while
    # the frame mean looks normal and a tag or two still decodes — the
    # decoder then loses the rest. Healthy: blacks <= ~60, contrast >= ~150.
    import numpy as _np
    blacks, contrasts = [], []
    for c in corners:
        q = c.reshape(4, 2).astype(int)
        m = _np.zeros(gray.shape, _np.uint8)
        cv2.fillPoly(m, [q], 255)
        v = gray[m > 0]
        if v.size:
            b, w = _np.percentile(v, 5), _np.percentile(v, 95)
            blacks.append(b)
            contrasts.append(w - b)
    return (float(gray.mean()), len(ids),
            float(max(blacks)) if blacks else None,
            float(min(contrasts)) if contrasts else None)


# Fail thresholds. Measured 2026-10-06 on real frames: stuck-ISP wrist cam
# blacks 186 / contrast 60; healthy robot wrist 98 / 107; healthy glove wrist
# after replug 77 / 126. Thresholds sit between the clusters with margin.
WASHED_BLACK, WASHED_CONTRAST = 140.0, 75.0


def _washed(black, contrast):
    # The fault signature is LIFTED blacks. Low contrast alone is not it: a
    # marker sitting in shadow on the forward cam reads black 24 / white 98
    # (glove3 2026-10-06) and decodes fine, so contrast only counts when the
    # blacks are already well above a healthy level.
    return (black is not None and contrast is not None
            and (black > WASHED_BLACK
                 or (contrast < WASHED_CONTRAST and black > 100)))


def _janitor(dataset, stems, stop_ev, seg_secs=60, keep_recent=300, margin=30):
    """Prune idle camera segments: anything older than keep_recent seconds
    that overlaps no saved episode (±margin) gets deleted, so pauses and
    between-episode time never accumulate on disk. The 2 newest segments per
    camera are never touched (one is still being written)."""
    walls = (Path.home() / ".cache/huggingface/lerobot" / dataset
             / "meta/episode_wallclock.jsonl")
    freed = 0
    tick = 0
    bad_streak = {}
    while not stop_ev.wait(30.0):
        tick += 1
        wins = []
        if walls.exists():
            for line in open(walls):
                try:
                    e = json.loads(line)
                    wins.append((e["start_unix"] - margin,
                                 e["end_unix"] + margin))
                except Exception:
                    pass
        now = time.time()
        for stem in stems:
            segs = sorted(glob.glob(str(PILOT / "data/raw" / f".{stem}_seg*.mkv")))
            # recorder health: segments must keep rolling while we record
            if segs and now - os.path.getmtime(segs[-1]) > 90:
                print(f"\n{'!'*70}\n!! {stem}: NO NEW FRAMES for "
                      f"{now - os.path.getmtime(segs[-1]):.0f}s — camera "
                      f"recorder may be dead; episodes since then have no "
                      f"labels/obs from it\n{'!'*70}", flush=True)
            # in-session health watchdog (~every 2 min): the pre-flight
            # only guards session START; a mid-session lens smudge, bump,
            # or control reset must not silently burn the remaining
            # episodes (2026-09-11: 27 wrist-blind eps went undetected)
            if tick % 4 == 0 and segs:
                try:
                    img = _tail_frame(stem)
                    if img is not None:
                        m, nmk = _frame_health(stem, img)
                        bad = nmk == 0
                        bad_streak[stem] = bad_streak.get(stem, 0) + 1 \
                            if bad else 0
                        if bad:
                            print(f"\a\n{'!'*70}\n!! {stem}: LIVE HEALTH "
                                  f"CHECK FAILED (x{bad_streak[stem]}) — "
                                  f"frame mean {m:.0f}/255, {nmk} markers "
                                  f"decoded. Footage since the last good "
                                  f"check may be unusable — check the "
                                  f"lens/preview NOW.\n{'!'*70}", flush=True)
                except Exception:
                    pass
            for s in segs[:-2]:
                end = os.path.getmtime(s)
                start = end - seg_secs - 15
                if now - end < keep_recent:
                    continue
                if any(start < w1 and end > w0 for w0, w1 in wins):
                    continue
                try:
                    freed += os.path.getsize(s)
                    os.remove(s)
                except OSError:
                    pass
        if freed:
            print(f"\r[janitor] pruned {freed/1e9:.1f} GB of idle footage so far",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False,
                    help="required unless --cameras_only")
    ap.add_argument("--glove", action="store_true",
                    help="electronics-free glove session: no robot, no "
                         "skeleton. Cameras record continuously while you "
                         "mark episode boundaries interactively; writes "
                         "data/raw/<dataset>_glove_episodes.json for "
                         "build_dataset --episode-marks.")
    ap.add_argument("--glove_unit", default="yosub",
                    help="glove unit name: calib/glove_<unit>_*_bundle.json "
                         "for the live pose publisher")
    ap.add_argument("--width_holds", action="store_true",
                    help="glove: enable the 'm <gap_mm>' caliper width holds "
                         "at the episode prompt. Holds characterise the glove "
                         "UNIT's tip/back markers, not the session, so they "
                         "are only needed once per unit and again after any "
                         "marker change; builds reuse an earlier session's "
                         "<dataset>_glove_log.json via --log")
    ap.add_argument("--preview_all", action="store_true",
                    help="open a live preview for EVERY camera, not just the "
                         "forward primary (implied by --glove; --no_preview "
                         "still wins)")
    ap.add_argument("--cameras_only", action="store_true",
                    help="rehearsal mode: all camera recorders + previews, "
                         "no host/skeleton; Enter stops and finalizes")
    ap.add_argument("--num_episodes", type=int, default=30)
    ap.add_argument("--episode_time", type=int, default=60)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", default="YAM teleop demonstration")
    ap.add_argument("--zmq_obs_port", type=int, default=5602)
    ap.add_argument("--skip_cameras", action="store_true",
                    help="skeleton only (debug)")
    ap.add_argument("--no_preview", action="store_true",
                    help="disable the live preview windows")
    ap.add_argument("--cameras", default="b0587,b0587b,fish",
                    help="comma list of camera suffixes to record "
                         "(e.g. b0587,b0587b when the wrist cam is removed)")
    ap.add_argument("--forward_rotate", default="none",
                    choices=["cw", "ccw", "180", "none"],
                    help="rotate the forward-cam PREVIEW (for a portrait "
                         "mount; default none = landscape). Recording is "
                         "unaffected")
    args, extra = ap.parse_known_args()

    # the teleop recorder puts the tty in raw/no-echo mode for its
    # single-key controls; every abnormal exit (Ctrl-C races included) must
    # still restore the EXACT original terminal state. Direct termios (no
    # killable subprocess) + atexit backstop.
    tty_state = termios.tcgetattr(0) if sys.stdin.isatty() else None

    def _restore_tty():
        if tty_state is not None:
            try:
                termios.tcsetattr(0, termios.TCSAFLUSH, tty_state)
            except Exception:
                pass
    atexit.register(_restore_tty)

    if args.glove:
        args.cameras_only = True   # no robot/skeleton; we mark episodes below
        args.preview_all = True    # validate every stream by eye (no teleop
                                   # CPU load to protect in glove mode)
    if not args.dataset:
        if not args.cameras_only:
            raise SystemExit("--dataset is required (or use --cameras_only)")
        args.dataset = "rehearsal"

    # dataset-dir state must match the resume flag BEFORE cameras spin up:
    # a shell dir from a crashed run breaks both create (FileExists) and
    # resume (Hub-lookup 404)
    droot = Path.home() / ".cache/huggingface/lerobot" / args.dataset
    has_eps = (droot / "meta/tasks.parquet").exists()
    if args.cameras_only:
        droot = Path("/nonexistent")  # no skeleton in rehearsal mode
    if "--resume" in extra and not has_eps and not args.glove:
        raise SystemExit(
            f"--resume passed but {args.dataset} has no saved episodes — "
            f"if a shell dir exists, delete {droot} and run WITHOUT --resume")
    if "--resume" not in extra and droot.exists():
        raise SystemExit(
            f"{droot} already exists "
            + ("with saved episodes — add --resume to continue it, or pick "
               "a new --dataset name" if has_eps else
               "but holds no episodes (crashed run) — delete it: "
               f"rm -rf {droot}"))

    # ---- disk: ~1 GB/min continuous (4K25 + fisheye), 2x peak during remux --
    import shutil
    free_gb = shutil.disk_usage(PILOT / "data/raw").free / 1e9
    est_gb = args.num_episodes * (args.episode_time + 30) / 60 * 1.0 * 2
    if free_gb < max(est_gb, 30):
        raise SystemExit(
            f"only {free_gb:.0f} GB free — session needs ~{est_gb:.0f} GB "
            "(cameras record CONTINUOUSLY, ~1 GB/min, and remux doubles the "
            "peak). Free space or shorten the session.")
    print(f"disk: {free_gb:.0f} GB free (~1 GB/min while cameras roll; a "
          "janitor prunes idle footage >5 min old that belongs to no episode)")

    # ---- host reachable? -----------------------------------------------------
    if not args.cameras_only:
        s = socket.socket()
        s.settimeout(1.0)
        try:
            s.connect(("127.0.0.1", args.zmq_obs_port))
            s.close()
        except OSError:
            raise SystemExit(
                f"no host listening on port {args.zmq_obs_port} — start "
                "yam_zmq_host first (see this script's docstring)")

    # ---- camera recorders ----------------------------------------------------
    recs = []   # (stem, Popen, logfile)
    wanted = {c.strip() for c in args.cameras.split(",") if c.strip()}
    unknown = wanted - {c[0] for c in CAMERAS}
    if unknown:
        raise SystemExit(f"unknown camera suffix(es) {sorted(unknown)} — "
                         f"known: {[c[0] for c in CAMERAS]}")
    if not args.skip_cameras:
        def _bail(msg):
            # abort DURING the launch loop: recorders already started for
            # earlier cameras must die with us, or they keep the devices and
            # the next run trips over them (2026-10-08: two attempts in a row)
            for _, q, _ in recs:
                if q.poll() is None:
                    q.send_signal(signal.SIGTERM)
            raise SystemExit(msg)

        # zombie check for EVERY wanted stem before launching any of them
        for suffix, *_ in CAMERAS:
            if suffix not in wanted:
                continue
            stem0 = f"{args.dataset}_{suffix}"
            alive = subprocess.run(["pgrep", "-f", f"uvc_record.py {stem0} "],
                                   capture_output=True, text=True).stdout.split()
            if alive:
                raise SystemExit(
                    f"a recorder for {stem0} is already running (pid "
                    f"{', '.join(alive)}) — stop it first: kill -TERM "
                    f"{' '.join(alive)}  (it will finish its remux), then rerun")
        for suffix, bypath, frag, res, fps in CAMERAS:
            if suffix not in wanted:
                continue
            dev = bypath if os.path.exists(bypath) else \
                (find_dev(frag) if frag else None)
            if dev is None and suffix == "b0587b":
                # glasses cam: its port changes with the cable route
                dev = find_other_b0587({c[1] for c in CAMERAS
                                        if c[0] != suffix})
                if dev:
                    print(f"  b0587b resolved by exclusion at {dev} "
                          "(pin it in CAMERAS if this route sticks)")
            if dev is None:
                _bail(f"camera '{suffix}' not found at {bypath} — replugged to "
                      "a different port? Update CAMERAS in this script.")
            stem = f"{args.dataset}_{suffix}"
            # A recorder for this stem must not already be running: a
            # leftover from an earlier run keeps the camera, and this run
            # would then silently read ITS segments while believing they are
            # its own (2026-10-06: three runs shared one zombie recorder).
            alive = subprocess.run(["pgrep", "-f", f"uvc_record.py {stem} "],
                                   capture_output=True, text=True).stdout.split()
            if alive:
                _bail(f"a recorder for {stem} is already running (pid "
                      f"{', '.join(alive)}) — stop it first: kill -TERM "
                      f"{' '.join(alive)}  (it will finish its remux), then "
                      "rerun")
            # NEVER overwrite a previous recording with the same stem (a
            # --resume session reuses the dataset name): archive it aside
            # first. Lost ep0-4 raws to this on 2026-09-09.
            if (PILOT / "data/raw" / f"{stem}.mkv").exists():
                k = 1
                while (PILOT / "data/raw" / f"{stem}.take{k}.mkv").exists():
                    k += 1
                for ext in (".mkv", "_ts.json", "_meta.json",
                            "_recorder.log"):
                    src = PILOT / "data/raw" / f"{stem}{ext}"
                    if src.exists():
                        src.rename(PILOT / "data/raw"
                                   / f"{stem}.take{k}{ext}")
                # cached detections must follow their take, or the next
                # session's detections would collide with the stale ones
                for ext in ("_dodeca_detections.npz",
                            "_dodeca_detections.csv",
                            "_tips16h5_detections.npz"):
                    src = PILOT / "calib" / f"{stem}{ext}"
                    if src.exists():
                        src.rename(PILOT / "calib"
                                   / f"{stem}.take{k}{ext}")
                print(f"  (existing {stem} recording archived as "
                      f"{stem}.take{k}.*)")
            log = open(PILOT / "data/raw" / f"{stem}_recorder.log", "w")
            # previews only where they guide teleop: the forward primary
            # (main + EEF zoom). Twin/wrist record headless — less CPU,
            # fewer windows.
            primary = suffix == "b0587"
            show = (primary or args.preview_all) and not args.no_preview
            env = dict(os.environ, PREVIEW="1" if show else "0")
            if primary and args.forward_rotate != "none":
                env["ROTATE"] = args.forward_rotate
            if primary:
                env["ZOOM"] = "1"    # EEF-tracking zoom window for teleop
                env["SQUARE"] = "1"  # policy-obs square + out-of-range shade
                # live marker pose for the head tracker (so101_cam_track.py
                # --hand-source markers); POSE_PUB=0 in the env disables it
                if os.environ.get("POSE_PUB", "5611") != "0":
                    env["POSE_PUB"] = os.environ.get("POSE_PUB", "5611")
                    if args.glove:
                        env["POSE_BUNDLE"] = \
                            f"calib/glove_{args.glove_unit}_dodeca_bundle.json"
                        env["POSE_GRIPPER_BUNDLE"] = \
                            f"calib/glove_{args.glove_unit}_gripper_bundle.json"
            p = subprocess.Popen(
                [sys.executable, str(PILOT / "scripts/uvc_record.py"),
                 stem, dev, res, fps],
                cwd=PILOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                stdin=subprocess.DEVNULL,  # NEVER the terminal: ffmpeg would
                                           # read keys from it and switch the
                                           # tty to raw mode, killing input()
                start_new_session=True)  # keep tty Ctrl-C away from recorders
            recs.append((stem, p, log))
            print(f"recording {stem}: {dev} @ {res}x{fps} (log: {log.name})")
        time.sleep(4.0)  # streams up + rituals latched
        for stem, p, _ in recs:
            if p.poll() is not None:
                for _, q, _ in recs:
                    if q.poll() is None:
                        q.send_signal(signal.SIGTERM)
                raise SystemExit(f"{stem} recorder died at startup — see its log")
        # behavioral exposure pre-flight: control READBACK LIES (latch race —
        # the 2026-09-11 fish session read back e=2 yet recorded mean 236/255
        # blown-white for 27 episodes). Decode a real frame from each
        # camera's first segment and refuse to run on a blown/black stream.
        # Known-good scene means run 94-156; blown ~236.
        time.sleep(5.0)  # streams up AND ritual latched (~1.5s in)
        dead = [(stem, q.returncode) for stem, q, _ in recs if q.poll() is not None]
        if dead:
            for _, q, _ in recs:
                if q.poll() is None:
                    q.send_signal(signal.SIGTERM)
            raise SystemExit(
                f"recorder(s) died at startup: {dead} — device busy (a "
                "recorder from an earlier run still alive? `pgrep -af "
                "uvc_record.py`) or wrong device node; see the stem's "
                "_recorder.log")

        def _health(stem):
            for _ in range(10):
                img = _tail_frame(stem)
                if img is not None:
                    return _frame_health(stem, img)
                time.sleep(1.0)
            return None, None, None, None

        devs = {f"{args.dataset}_{s}": d for s, d, *_ in CAMERAS}

        def _abort(msg):
            for _, q, _ in recs:
                if q.poll() is None:
                    q.send_signal(signal.SIGTERM)
            raise SystemExit(msg)

        if args.glove:
            # On the robot rig the markers are always in view, so one decode
            # a few seconds in is a valid health check. On a glove session
            # NOTHING is in view until the operator holds the glove up, so
            # wait for every stream instead of aborting on the first frame.
            from uvc_ritual import apply as _ritual
            print("\nGLOVE PRE-FLIGHT — hold the glove where ALL cameras see "
                  "it (fixed forward, glasses, wrist); waiting up to 120 s "
                  "for markers on every stream…", flush=True)
            pending = {stem for stem, _, _ in recs}
            healed = set()
            t_end = time.time() + 120
            while pending and time.time() < t_end:
                for stem in sorted(pending):
                    mean, nmk, black, contrast = _health(stem)
                    if nmk and _washed(black, contrast):
                        # decodes, but the ISP is in its stuck state: keep
                        # waiting so a replug mid-pre-flight can fix it
                        print(f"\a  {stem}: {nmk} markers but WASHED OUT "
                              f"(tag blacks {black:.0f}/255, contrast "
                              f"{contrast:.0f}; healthy <= {WASHED_BLACK:.0f}"
                              f" / >= {WASHED_CONTRAST:.0f}) — REPLUG this "
                              "camera; a brightness/exposure tweak will not "
                              "fix it", flush=True)
                    elif nmk:
                        print(f"  {stem}: {nmk} markers decoded "
                              f"(mean {mean:.0f}/255, tag blacks {black:.0f},"
                              f" contrast {contrast:.0f}) OK", flush=True)
                        pending.discard(stem)
                    elif (mean is not None and not 25 <= mean <= 210
                          and stem not in healed and stem in devs):
                        # exposure anomaly: the mid-stream latch race is
                        # transient — re-apply the ritual once
                        healed.add(stem)
                        print(f"  {stem}: mean {mean:.0f}/255, no markers — "
                              "re-applying ritual", flush=True)
                        try:
                            _ritual(devs[stem])
                        except Exception as e:
                            print(f"  (ritual re-apply failed: {e})")
                if pending:
                    time.sleep(2.0)
            if pending:
                _abort(f"{sorted(pending)}: no fiducials decoded within 120 s "
                       "— is the glove in that camera's view? lens smudge? "
                       "wrong device? (check its preview window)")

        for stem, _, _ in ([] if args.glove else recs):
            mean, nmk, black, contrast = _health(stem)
            if mean is not None and (not 25 <= mean <= 210 or nmk == 0) \
                    and stem in devs:
                # self-heal: the mid-stream latch race is transient — apply
                # the ritual again on the live stream and re-check before
                # aborting (2026-09-13: b0587 latched only on 2nd apply)
                print(f"{stem}: startup frame mean {mean:.0f}/255, "
                      f"{nmk} markers — re-applying ritual and re-checking…")
                from uvc_ritual import apply as _ritual
                try:
                    _ritual(devs[stem])
                except Exception as e:
                    print(f"  (ritual re-apply failed: {e})")
                time.sleep(3.0)
                mean, nmk, black, contrast = _health(stem)
            if mean is None:
                print(f"*** {stem}: no startup frame for the health check — "
                      "verify its preview by eye before recording ***")
                continue
            # markers are the abort criterion (functional truth: exposure,
            # focus, smudge, framing all gate them); brightness only
            # diagnoses WHY — the tone-mapping fisheye runs legitimately
            # bright (healthy scenes near mean 200), so mean alone must
            # never abort a stream that still decodes fiducials
            fail = None
            if nmk == 0:
                why = ("blown white" if mean > 210 else
                       "near black" if mean < 25 else
                       "exposure OK — suspect lens smudge/focus/framing")
                fail = (f"{stem} HEALTH CHECK FAILED: ZERO fiducials decode "
                        f"(frame mean {mean:.0f}/255, {why}). Wipe/replug "
                        "and rerun; markers are always in view on this rig.")
            elif _washed(black, contrast):
                fail = (f"{stem} HEALTH CHECK FAILED: markers decode but the "
                        f"image is WASHED OUT (tag blacks {black:.0f}/255, "
                        f"contrast {contrast:.0f}) — the camera's ISP is in "
                        "its stuck state; REPLUG it and rerun (controls read "
                        "back fine and do nothing in this state)")
            elif not 25 <= mean <= 210:
                print(f"*** {stem}: brightness unusual (mean {mean:.0f}/255)"
                      f" but {nmk} markers decode — proceeding; check the "
                      "preview ***")
            if fail:
                _abort(fail)
            print(f"{stem}: health OK (mean {mean:.0f}/255, "
                  f"{nmk} markers decoded)")
        print("all cameras rolling; rituals applied mid-stream\n")

    stop_ev = threading.Event()
    if recs and not args.cameras_only:
        # no janitor in rehearsal mode: with no episodes saved, it would
        # prune ALL footage older than 5 min
        threading.Thread(target=_janitor,
                         args=(args.dataset, [s for s, _, _ in recs], stop_ev),
                         daemon=True).start()
    elif recs and args.glove:
        # glove mode: no pruning (no skeleton to define episode windows), but
        # the dead-stream alarm is essential — a wrist camera that drops out
        # mid-session must not go unnoticed for 7 minutes (2026-10-06)
        def _glove_watchdog():
            stems = [s for s, _, _ in recs]
            while not stop_ev.wait(30.0):
                now = time.time()
                for stem in stems:
                    segs = sorted(glob.glob(str(PILOT / "data/raw" / f".{stem}_seg*.mkv")))
                    if segs and now - os.path.getmtime(segs[-1]) > 90:
                        print(f"\a\n{'!'*70}\n!! {stem}: NO NEW FRAMES for "
                              f"{now - os.path.getmtime(segs[-1]):.0f}s — the "
                              "camera dropped (replug it); episodes since "
                              f"then have no {stem.split('_')[-1]} stream\n"
                              f"{'!'*70}", flush=True)
        threading.Thread(target=_glove_watchdog, daemon=True).start()

    # ---- skeleton recorder (foreground, interactive) -------------------------
    code = 0
    try:
        if args.glove:
            # No robot, so nothing supplies episode boundaries: mark them by
            # hand here and build_dataset --episode-marks slices the cameras'
            # wall clocks with them. Same clock (time.time()) as the *_ts.json
            # the recorders write, so the two align directly.
            # FORCE a cooked terminal before the first prompt. Restoring the
            # saved state is not enough: a previous process in this same
            # terminal (a Ctrl-C'd focus_tune, a crashed run) can leave the
            # tty in raw/no-echo mode — bash still works (readline manages
            # its own echo) but every input() here looks dead (2026-10-06).
            # In raw mode Enter sends '\r' (ICRNL off), so input() never
            # returns while Ctrl-C still works — exactly the observed
            # symptom. `stty sane` resets every relevant flag at once.
            if sys.stdin.isatty():
                def _flags():
                    a = termios.tcgetattr(0)
                    return (f"ICANON={bool(a[3] & termios.ICANON):d} "
                            f"ECHO={bool(a[3] & termios.ECHO):d} "
                            f"ICRNL={bool(a[0] & termios.ICRNL):d}")
                before = _flags()
                subprocess.run(["stty", "sane"], stdin=sys.stdin,
                               check=False)
                print(f"  terminal: {before} -> {_flags()} (after stty sane)")
            out = PILOT / "data/raw" / f"{args.dataset}_glove_episodes.json"
            # CUMULATIVE across runs of the same dataset name: a second run
            # must never overwrite the first run's marks (2026-10-06: 30
            # episodes' boundaries lost that way). Episodes keep counting
            # up; wall clocks tie each one to whichever take recorded it.
            marks = []
            if out.exists():
                try:
                    marks = json.load(open(out)).get("episodes", [])
                except Exception:
                    marks = []
                if marks:
                    print(f"  resuming {out.name}: {len(marks)} episodes "
                          "already marked — new ones continue the numbering")
            ep = len(marks)
            args.num_episodes += ep        # plan N MORE, not N total
            # width calibration: no actuator on a glove, so caliper holds
            # stand in. Each 'm <gap_mm>' logs e8_meas events every 0.5 s
            # for as long as the hold lasts (dense during holds, NOTHING
            # between them — the fit must only use frames near a
            # measurement). Gap mm -> robot actuator units via the bench fit
            # in calib/gap_to_actuator.json so glove labels share Bj's units.
            # Holds are OPT-IN (--width_holds): they characterise the glove
            # unit, not the session, so the usual session never asks.
            meas, log_out = [], PILOT / "data/raw" / f"{args.dataset}_glove_log.json"
            g2a = None
            if args.width_holds:
                try:
                    g2a = json.load(open(PILOT / "calib/gap_to_actuator.json"))
                except OSError:
                    print("!! calib/gap_to_actuator.json missing — caliper "
                          "values will be logged as RAW mm (run "
                          "scripts/gap_bench.py on the robot first)")
            hold_help = ("  'm <gap_mm>' + ENTER logs a caliper width hold "
                         "(keep the jaws still and in view until you press "
                         "ENTER again).\n" if args.width_holds else
                         "  (width holds off — pass --width_holds after a "
                         "marker change)\n")
            prompt_opts = "(q / m <mm>)" if args.width_holds else "(q)"
            print(f"\nGLOVE SESSION — {args.num_episodes} episodes planned.\n"
                  "  >>> CLICK THIS TERMINAL FIRST: the preview windows that just "
                  "opened take keyboard focus, so keystrokes go to them until "
                  "you click back here. <<<\n"
                  "  ENTER starts an episode, ENTER ends it.\n"
                  "  'd' + ENTER after an episode discards it, "
                  "'q' + ENTER finishes the session.\n" + hold_help)
            while ep < args.num_episodes:
                cmd = input(f"[{ep + 1}/{args.num_episodes}] ENTER to START "
                            f"{prompt_opts}: ").strip().lower()
                if cmd == "q":
                    break
                # a bare number at this prompt can only be a caliper value
                # (2026-10-07: "37" started a 27-s episode instead of a hold)
                try:
                    float(cmd)
                    cmd = "m " + cmd
                except ValueError:
                    pass
                if cmd.startswith("m") and not args.width_holds:
                    print("    width holds are off for this session — restart "
                          "with --width_holds if the markers changed")
                    continue
                if cmd.startswith("m"):
                    try:
                        mm = float(cmd[1:].strip())
                    except ValueError:
                        print("    usage: m 42.5")
                        continue
                    val = g2a["a"] * mm + g2a["b"] if g2a else mm
                    import select as _sel
                    print(f"    holding {mm:.1f} mm -> {val:.3f} "
                          f"{'actuator' if g2a else 'RAW mm'} … ENTER to end")
                    t_hold = time.time()
                    while True:
                        meas.append({"event": "e8_meas", "t_wall": time.time(),
                                     "gripper_pos": val, "gap_mm": mm})
                        if _sel.select([sys.stdin], [], [], 0.5)[0]:
                            sys.stdin.readline()
                            break
                    log_out.write_text(json.dumps({"dataset": args.dataset,
                                                   "units": "actuator" if g2a
                                                   else "mm", "events": meas},
                                                  indent=1))
                    print(f"    logged {time.time() - t_hold:.1f}s hold "
                          f"({len(meas)} events total) -> {log_out.name}")
                    continue
                t0 = time.time()
                resp = input(f"    recording… ENTER to END "
                             f"(then d to discard): ").strip().lower()
                t1 = time.time()
                if resp == "d":
                    print(f"    discarded ({t1 - t0:.1f}s)")
                    continue
                marks.append({"episode_index": ep, "t_start": t0,
                              "t_end": t1, "duration_s": round(t1 - t0, 2)})
                out.write_text(json.dumps({"dataset": args.dataset,
                                           "episodes": marks}, indent=1))
                print(f"    saved ep{ep} ({t1 - t0:.1f}s) -> {out.name}")
                ep += 1
            print(f"\n{len(marks)} episodes marked in {out}")
        elif args.cameras_only:
            input("REHEARSAL — previews live, everything recording. "
                  "Enter (or Ctrl-C) to stop + finalize… ")
        else:
            code = subprocess.call(
                [sys.executable, str(TELEOP), "--dataset", args.dataset,
                 "--no_cameras", "--fps", str(args.fps),
                 "--num_episodes", str(args.num_episodes),
                 "--episode_time", str(args.episode_time),
                 "--task_description", args.task, *extra],
                cwd=TELEOP.parent)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ev.set()
        # from here on, further Ctrl-C must not abort cleanup (tty restore,
        # remux waits) — this also ENFORCES "do not Ctrl-C during remux"
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _restore_tty()
        if recs:
            print("\nstopping camera recorders — remux in progress, "
                  "do NOT Ctrl-C…", flush=True)
            for _, p, _ in recs:
                if p.poll() is None:
                    p.send_signal(signal.SIGTERM)
            for stem, p, log in recs:
                try:
                    p.wait(timeout=1800)
                    print(f"  {stem}: done")
                except subprocess.TimeoutExpired:
                    print(f"  {stem}: STILL REMUXING after 30 min — leave it; "
                          f"check {log.name}")
                log.close()

    # recorders must be GONE now, or the next run will read their segments
    # as its own (2026-10-06: three runs shared one survivor)
    left = [stem for stem, q, _ in recs if q.poll() is None]
    for stem, q, _ in recs:
        if q.poll() is None:
            q.kill()
    if left:
        print(f"\n!! recorder(s) still alive at exit, killed: {left} — their "
              "last segments were NOT remuxed; check data/raw/.<stem>_seg*")
    print(f"\nsession complete (teleop exit {code}).")
    if recs:
        stems = "  ".join(s for s, _, _ in recs)
        print(f"camera stems: {stems}")
        print("next: detect_dodeca on both stems, then build_dataset.py "
              "(COLLECTION_RUNBOOK.md)")


if __name__ == "__main__":
    main()
