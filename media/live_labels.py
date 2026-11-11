"""Live label readout for the demo video: the pipeline's own solvers (scripts/labels_lib.py, scripts/head_pose_fast.py)
applied frame by frame to a downscaled head-camera frame. Camera pose in the tile world from the 36h11 tiles, gripper body
pose from the ball faces (+ the base cubes), tool point from calib/eef_target_glove_yosub.json. The jaw width is not solved
here: from the head camera the tails travel along the view direction, and the paper's width comes from a session-fitted
tail/base model (scripts/forward_width.py), so the demo shows the width label stored in the built dataset instead."""
import glob, json, os, sys
from pathlib import Path
import cv2, numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import labels_lib as L  # noqa: E402


def _head_pose_solver():
    src = (REPO / "scripts/head_pose_fast.py").read_text().split("\n\nap = argparse")[0]  # defs only; the script parses argv at import
    ns = {"__file__": str(REPO / "scripts/head_pose_fast.py")}; exec(src, ns); return ns["solve_frame"]


def euler_deg(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_euler("xyz", degrees=True)


class LiveLabels:
    def __init__(self, scale=0.5, fps=25.0, unit="yosub"):
        K, self.dist = L.load_intrinsics(REPO / "calib/B0587b_f3_intrinsics.json")
        self.K = K.copy(); self.K[:2] *= scale; self.fps = fps
        self.layout = json.load(open(REPO / "calib/workspace_tiles_pilot_v3.json"))
        self.tile_ids = set(int(i) for i in json.load(open(REPO / f"calib/tags36h11_{unit}.json"))["ids"])
        self.T_body, self.dobj = L.load_bundle(REPO / f"calib/glove_{unit}_dodeca_bundle.json")
        self.extra = L.load_gripper_bundle(REPO / f"calib/glove_{unit}_gripper_bundle.json")
        self.p_tool = np.array(json.load(open(REPO / f"calib/eef_target_glove_{unit}.json"))["p_body_mm"]) / 1000
        self.solve_frame = _head_pose_solver()
        self.last_T = None; self.last_k = None; self.anchor_k = None; self.last_body = None

    def update(self, k, c36, i36, c4, i4):
        """k: frame index (for the single-tile continuity gate); corners in the downscaled image's pixels."""
        out = {"cam": None, "tiles": 0, "body": None, "faces": 0, "tool_world": None, "R_world_body": None}
        by = {}
        if i36 is not None:
            for i, c in zip(i36.ravel(), c36):
                if int(i) in self.tile_ids: by[int(i)] = c.reshape(1, 4, 2).astype(np.float32)
        chain = (self.last_k is not None and k - self.last_k <= 1.0 * self.fps and self.anchor_k is not None and k - self.anchor_k <= 10.0 * self.fps)
        r = self.solve_frame(by, self.layout, self.K, self.dist, last=(self.last_T, (k - self.last_k) / self.fps) if chain else None) if by else None
        if r is not None:
            T, e, n = r; out["cam"], out["tiles"], out["cam_rms"] = T, n, e
            if n >= 2 and e <= 5.0: self.last_T, self.last_k, self.anchor_k = T, k, k
            elif n == 1: self.last_T, self.last_k = T, k
        if i4 is not None:
            ids = [int(i) for i in i4.ravel()]
            rb = L.solve_body_pose(ids, [c.reshape(4, 2) for c in c4], self.K, self.dist, self.T_body, self.dobj, extra=self.extra, last=self.last_body)
            if rb is not None:
                rvec, tvec, rms, nf, ne = rb; self.last_body = (rvec, tvec); R = cv2.Rodrigues(rvec)[0]; t = tvec.ravel()
                out["body"] = (R, t, rms); out["faces"] = nf
                if out["cam"] is not None:
                    T_wc = np.linalg.inv(out["cam"]); p_c = R @ self.p_tool + t
                    out["tool_world"] = T_wc[:3, :3] @ p_c + T_wc[:3, 3]; out["R_world_body"] = T_wc[:3, :3] @ R
        return out


def dataset_gripper_for_cut(ts_json, marks_json, parts, dataset="pilot_glove_HHs3"):
    """Per cut-frame gripper label (the 7th action channel of the built dataset) for a cut made of source frame ranges.
    parts: [(src_first, src_last, mark_episode, dataset_episode)]. The dataset episode is the paced resample of the marked
    interval, so a source frame maps by its fraction of the mark's wall-clock span."""
    import pandas as pd
    D = os.path.expanduser(f"~/.cache/huggingface/lerobot/{dataset}")
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/data/*/*.parquet"))])
    ts = [e["wall"] for e in json.load(open(ts_json))]; marks = {e["episode_index"]: e for e in json.load(open(marks_json))["episodes"]}
    out = []
    for a, b, mk, de in parts:
        g = np.array([r[6] for r in df[df.episode_index == de].action]); m = marks[mk]
        for s in range(a, b + 1):
            f = (ts[s] - m["t_start"]) / (m["t_end"] - m["t_start"]); f = min(max(f, 0.0), 1.0)
            out.append(float(g[int(round(f * (len(g) - 1)))]))
    return np.array(out)
