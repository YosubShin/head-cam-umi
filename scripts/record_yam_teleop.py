#!/usr/bin/env python3
"""Record LeRobot training episodes on a YAM follower via bilateral teleop.

Port of record_so101_teleop.py with one structural difference: yam_zmq_host.py
runs the bilateral teleop loop itself (leader in grav-comp with force feedback,
follower tracking), so this recorder has no local leader object and never sends
actions. The ACTION each tick is the host's currently commanded follower target
(`target.*` in the published obs) — teleop semantics, same as the SO-101 recipe
where action = leader pose and the follower's lag gives a natural plant lead.

Shares the SO-101 recorder's guards: pre-flight + in-episode obs-rate and
frozen-camera watchdogs with glitch-retry, Enter/r/q controls, live rate
display, wall-clock sidecar.

Setup: start the host first (it owns both CAN buses and the camera):
    ~/co/i2rt/.venv/bin/python yam_zmq_host.py \
        --leader_channel can_leader --follower_channel can_follower --cameras wrist:0

Controls:
    Enter  finish + save episode  |  between episodes: start the next
    r      discard current episode and redo
    q      quit session
"""
from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path

from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

ALOHAMINI = Path(os.environ.get("LEROBOT_ALOHAMINI", str(Path.home() / "lerobot_alohamini")))  # the lerobot fork: examples/alohamini helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ALOHAMINI / "examples/alohamini"))
from record_so101 import (  # noqa: E402
    _MIN_OBS_RATE_FRACTION,
    FocusKeyboard,
    _apply_stdin_controls,
    _frozen_cam_banner,
    _log_episode_wallclock,
    _measure_obs_rate,
    _obs_rate_banner,
    parse_bool,
)
from yam_zmq_client import YamZmqClient, YamZmqClientConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teleop-record episodes on a YAM follower")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode_time", type=int, default=60)
    p.add_argument("--task_description", type=str, default="YAM teleop demonstration")
    p.add_argument("--remote_ip", type=str, default="127.0.0.1")
    p.add_argument("--zmq_cmd_port", type=int, default=5601)
    p.add_argument("--zmq_obs_port", type=int, default=5602)
    p.add_argument("--cam_width", type=int, default=320)
    p.add_argument("--cam_height", type=int, default=240)
    p.add_argument("--cameras", type=str, default="wrist",
                   help="Comma list of host camera names to record")
    p.add_argument("--no_cameras", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push_to_hub", type=parse_bool, nargs="?", const=True, default=False)
    p.add_argument("--no_rerun", action="store_true")
    return p.parse_args()


class _View:
    """Throttled Rerun view (camera + measured/target joints)."""

    def __init__(self, cam: str, names: tuple[str, ...]):
        import rerun as rr
        self.rr, self.cam, self.names = rr, cam, names
        self._last_img = self._last_scalar = 0.0
        rr.init("yam_record", spawn=True)

    def log(self, obs: dict) -> None:
        import numpy as np
        rr, now = self.rr, time.perf_counter()
        frame = obs.get(self.cam)
        if (now - self._last_img >= 1 / 15 and isinstance(frame, np.ndarray)
                and frame.any()):
            self._last_img = now
            rr.log(f"{self.cam}/image", rr.Image(frame[:, :, ::-1]))
        if now - self._last_scalar >= 1 / 10:
            self._last_scalar = now
            for n in self.names:
                joint = n.removesuffix(".pos")
                rr.log(f"plots/{joint}/measured", rr.Scalars(float(obs[n])))
                tgt = obs.get(f"target.{n}")
                if tgt is not None:
                    rr.log(f"plots/{joint}/target", rr.Scalars(float(tgt)))


def _log_frame_wallclocks(dataset, frame_walls: list[float]) -> None:
    """Append this episode's per-frame unix times to
    meta/episode_frames_wallclock.jsonl — the offline dataset constructor
    (scripts/build_dataset.py) aligns standalone camera recordings
    (uvc_record *_ts.json wall clocks) to these, frame by frame. The
    episode-level sidecar alone leaves loop jitter unmodeled."""
    import json

    idx = dataset.meta.total_episodes - 1
    path = Path(dataset.root) / "meta" / "episode_frames_wallclock.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"episode_index": idx,
                            "t_wall": [round(t, 4) for t in frame_walls]}) + "\n")


def _wait_for_next_episode(kb, events, robot, fps, prompt=None, view=None) -> bool:
    """Between episodes: teleop keeps running on the host; just observe and
    wait for Enter. Returns False on quit."""
    kb.clear()
    time.sleep(0.2)
    kb.clear()
    if prompt is None:
        prompt = "RESET THE SCENE (teleop stays live), then Enter for the next episode"
    print("\n" + "▒" * 74)
    print(f"▒▒  ⏸  {prompt}")
    print("▒▒       q = quit session")
    print("▒" * 74 + "\n", flush=True)
    log_say("Press Enter when ready")
    interval = 1.0 / fps
    while not events["stop_recording"]:
        t0 = time.perf_counter()
        try:
            obs = robot.get_observation()
            if view is not None:
                view.log(obs)
        except Exception:
            pass
        pressed = kb.get_pressed()
        if "q" in pressed:
            events["stop_recording"] = True
            kb.clear()
            print("\n[q] quit.")
            return False
        if "\r" in pressed or "\n" in pressed:
            kb.clear()
            print("\n[Enter] starting episode…", flush=True)
            time.sleep(0.3)
            kb.clear()
            return True
        precise_sleep(max(interval - (time.perf_counter() - t0), 0.0))
    return False


def record_loop(*, robot, events, fps, control_time_s, dataset, single_task,
                kb, view=None, frame_walls=None) -> None:
    import numpy as np

    interval = 1.0 / fps
    start_t = time.perf_counter()
    frame_i = 0
    rate_check_every = max(int(fps * 3), 1)
    msgs0 = getattr(robot, "msgs_received", None)
    frozen_limit = int(fps * 1.0)
    prev_frames: dict[str, np.ndarray] = {}
    frozen_run: dict[str, int] = {}

    while (time.perf_counter() - start_t) < control_time_s:
        if kb is not None:
            _apply_stdin_controls(kb, events)
        if events.get("exit_early"):
            events["exit_early"] = False
            break
        if events.get("stop_recording") or events.get("rerecord_episode"):
            break

        t0 = time.perf_counter()
        obs = robot.get_observation()
        action = robot.get_target_action(obs)  # host's commanded target = action
        if view is not None:
            view.log(obs)

        for cam, frame in obs.items():
            if not (isinstance(frame, np.ndarray) and frame.ndim == 3):
                continue
            if cam in prev_frames and np.array_equal(frame, prev_frames[cam]):
                frozen_run[cam] = frozen_run.get(cam, 0) + 1
            else:
                frozen_run[cam] = 0
                prev_frames[cam] = frame.copy()
        stuck = [c for c, n in frozen_run.items() if n >= frozen_limit]
        if stuck:
            print(_frozen_cam_banner(stuck), flush=True)
            print("Episode discarded — will wait for recovery and let you redo it.",
                  flush=True)
            events["rerecord_episode"] = True
            events["glitch_abort"] = True
            break

        if dataset is not None:
            oframe = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            aframe = build_dataset_frame(dataset.features, action, prefix=ACTION)
            dataset.add_frame({**oframe, **aframe, "task": single_task})
            if frame_walls is not None:
                frame_walls.append(time.time())

        frame_i += 1
        if frame_i % rate_check_every == 0:
            elapsed = time.perf_counter() - start_t
            loop_hz = frame_i / elapsed
            obs_hz = loop_hz if msgs0 is None else (robot.msgs_received - msgs0) / elapsed
            if min(loop_hz, obs_hz) < fps * _MIN_OBS_RATE_FRACTION:
                print(_obs_rate_banner(min(loop_hz, obs_hz),
                                       fps * _MIN_OBS_RATE_FRACTION), flush=True)
                print("Episode discarded — will wait for recovery and let you redo it.",
                      flush=True)
                events["rerecord_episode"] = True
                events["glitch_abort"] = True
                break
        if frame_i % 10 == 0:
            elapsed = time.perf_counter() - start_t
            obs_hz = "" if msgs0 is None else \
                f"  obs={(robot.msgs_received - msgs0) / elapsed:4.1f}Hz"
            print(f"\r  recorded frames={frame_i}  t={elapsed:5.1f}s/"
                  f"{control_time_s:.0f}s{obs_hz}   ", end="", flush=True)

        precise_sleep(max(interval - (time.perf_counter() - t0), 0.0))
    if frame_i:
        print()


def main() -> None:
    args = parse_args()

    cam_shapes = {}
    if not args.no_cameras:
        cam_shapes = {c.strip(): (args.cam_height, args.cam_width, 3)
                      for c in args.cameras.split(",") if c.strip()}
    robot = YamZmqClient(YamZmqClientConfig(
        remote_ip=args.remote_ip, zmq_cmd_port=args.zmq_cmd_port,
        zmq_obs_port=args.zmq_obs_port, cameras=cam_shapes))

    use_videos = bool(cam_shapes)
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_videos)
    dataset_features = {**action_features, **obs_features}

    if args.resume:
        print("Resuming existing dataset:", args.dataset)
        dataset = LeRobotDataset.resume(
            repo_id=args.dataset, root=HF_LEROBOT_HOME / args.dataset,
            image_writer_threads=4)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset, fps=args.fps, features=dataset_features,
            robot_type=robot.name, use_videos=use_videos, image_writer_threads=4)
        print(f"Dataset created with id: {dataset.repo_id}")
    print(f"Local dataset path: {dataset.root.resolve()}")

    robot.connect()
    robot.set_mode("teleop")
    time.sleep(1.0)

    print("Pre-flight: measuring host observation rate (3 s)…", flush=True)
    obs_hz, frozen = _measure_obs_rate(robot)
    needed_hz = args.fps * _MIN_OBS_RATE_FRACTION
    if (obs_hz is not None and obs_hz < needed_hz) or frozen:
        if obs_hz is not None and obs_hz < needed_hz:
            print(_obs_rate_banner(obs_hz, needed_hz), flush=True)
        if frozen:
            print(_frozen_cam_banner(frozen), flush=True)
        robot.disconnect()
        raise SystemExit(1)
    print(f"Observation rate OK: {obs_hz:.1f} Hz (need >= {needed_hz:.1f}); "
          "cameras live: all", flush=True)

    view = None
    if cam_shapes and not args.no_rerun:
        view = _View(next(iter(cam_shapes)), robot.config.joint_names)

    listener, events = init_keyboard_listener()
    kb = FocusKeyboard()
    kb.start()

    print("\nYAM teleop record ready — the host keeps the follower mirroring "
          "the leader continuously.")
    print("  CONTROLS: Enter=save/start   r=discard+redo   q=quit")
    print("  Flow: Enter(start) → record → Enter(save) → reset scene → Enter(next)\n",
          flush=True)

    recorded_episodes = 0
    try:
        if not _wait_for_next_episode(
                kb, events, robot, args.fps,
                prompt="Position the leader at the start pose, then Enter for the first episode",
                view=view):
            return
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            print("\n" + "█" * 74)
            print(f"██  ▶  RECORDING episode {recorded_episodes + 1}/{args.num_episodes}"
                  f"      Enter=save   r=discard   q=quit")
            print("█" * 74 + "\n", flush=True)
            log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")
            events["exit_early"] = False
            events["rerecord_episode"] = False
            kb.clear()
            time.sleep(0.2)
            kb.clear()

            ep_start_unix = time.time()
            frame_walls: list[float] = []
            record_loop(robot=robot, events=events, fps=args.fps,
                        control_time_s=args.episode_time, dataset=dataset,
                        single_task=args.task_description, kb=kb, view=view,
                        frame_walls=frame_walls)
            ep_end_unix = time.time()

            if events["stop_recording"]:
                dataset.clear_episode_buffer()
                print("Quit — in-progress episode discarded (not saved).")
                break

            if events["rerecord_episode"]:
                log_say("Discarded episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                print("Episode discarded (not saved).")
                if events.pop("glitch_abort", False):
                    print("Waiting for the stream to recover (teleop stays live)…",
                          flush=True)
                    while True:
                        hz, frz = _measure_obs_rate(robot, seconds=2.0)
                        if (hz is None or hz >= needed_hz) and not frz:
                            print(f"Recovered: {hz:.1f} Hz, cameras live.", flush=True)
                            break
                        print(f"  not yet (rate={hz and f'{hz:.1f}'} Hz, "
                              f"frozen={frz or 'none'}) — retrying in 3 s", flush=True)
                        time.sleep(3)
                if not _wait_for_next_episode(kb, events, robot, args.fps, view=view):
                    break
                continue

            if not dataset.has_pending_frames():
                print("No frames captured — not saving.")
                dataset.clear_episode_buffer()
                if not _wait_for_next_episode(kb, events, robot, args.fps, view=view):
                    break
                continue

            print("\n… saving + encoding episode …", flush=True)
            dataset.save_episode()
            _log_episode_wallclock(dataset, ep_start_unix, ep_end_unix)
            _log_frame_wallclocks(dataset, frame_walls)
            recorded_episodes += 1
            print(f"\n✔✔✔  SAVED  —  {recorded_episodes}/{args.num_episodes} episodes\n",
                  flush=True)

            if recorded_episodes < args.num_episodes and not events["stop_recording"]:
                if not _wait_for_next_episode(kb, events, robot, args.fps, view=view):
                    break
    finally:
        log_say("Stop recording")
        kb.stop()
        try:
            robot.disconnect()
        except Exception as e:
            print(f"(robot disconnect warning: {e})")
        if listener is not None:
            listener.stop()
        dataset.finalize()

    print(f"Dataset saved locally at: {dataset.root.resolve()}")
    if args.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
