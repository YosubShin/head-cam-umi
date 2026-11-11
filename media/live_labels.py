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


# ---------- schematic 3D view of the tile world ----------
JAW_AXIS_BODY = np.array([0.649, 0.743, -0.161])  # the jaws' travel direction in the ball's body frame (from the tails' motion in glove6)


class View3D:
    """Camera (frustum) and gripper (triad, jaw symbol, tool point) in the tile world from a fixed virtual viewpoint,
    with fading trails. Colours are BGR."""
    def __init__(self, layout, size=(650, 470), eye=(1.55, -1.25, 1.12), target=(0.25, 0.15, 0.30), f_px=700,
                 teal=(0x8F, 0x7F, 0x2A), amber=(0x2B, 0x82, 0xD9), ink=(0x38, 0x2A, 0x1E), paper=(0xF7, 0xFA, 0xFB)):
        self.W, self.H = size; self.teal, self.amber, self.ink, self.paper = teal, amber, ink, paper
        eye, target = np.array(eye, float), np.array(target, float)
        fwd = target - eye; fwd /= np.linalg.norm(fwd); right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd]); self.t = -self.R @ eye
        self.K = np.array([[f_px, 0, self.W / 2], [0, f_px, self.H / 2 + 10], [0, 0, 1.0]])
        edge = layout["edge_mm"] / 1000; h = edge / 2; sq = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        self.tiles = {}
        for i, tl in layout["tiles"].items():
            Rt = cv2.Rodrigues(np.array(tl["rvec"], float))[0]; self.tiles[int(i)] = (Rt @ sq.T).T + np.array(tl["xyz_mm"]) / 1000
        self.grid = [((x, -0.4, 0), (x, 0.6, 0)) for x in np.arange(-0.3, 0.81, 0.1)] + [((-0.3, y, 0), (0.8, y, 0)) for y in np.arange(-0.4, 0.61, 0.1)]

    def px(self, P):
        P = np.asarray(P, float).reshape(-1, 3); c = (self.R @ P.T).T + self.t
        u = self.K[0, 0] * c[:, 0] / c[:, 2] + self.K[0, 2]; v = self.K[1, 1] * c[:, 1] / c[:, 2] + self.K[1, 2]
        return np.stack([u, v], 1)

    def line(self, im, a, b, col, w=1):
        p = self.px([a, b]).round().astype(int); cv2.line(im, tuple(p[0]), tuple(p[1]), col, w, cv2.LINE_AA)

    def render(self, cam_T, body_world, R_world_body, tool_world, grip_label, trail_cam, trail_tool):
        im = np.full((self.H, self.W, 3), self.paper, np.uint8)
        for a, b in self.grid: self.line(im, a, b, (0xDE, 0xDB, 0xD4))
        for i, c in self.tiles.items():
            p = self.px(c).round().astype(np.int32); cv2.polylines(im, [p.reshape(-1, 1, 2)], True, self.teal, 2, cv2.LINE_AA)
        o = np.zeros(3)
        for ax, lab in (([0.1, 0, 0], "x"), ([0, 0.1, 0], "y"), ([0, 0, 0.1], "z")):
            self.line(im, o, ax, self.ink, 2); p = self.px(np.array(ax) * 1.25)[0]; cv2.putText(im, lab, (int(p[0]) - 5, int(p[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ink, 1, cv2.LINE_AA)
        for trail, col in ((trail_cam, self.teal), (trail_tool, self.amber)):
            n = len(trail)
            for j in range(1, n):
                a = 0.15 + 0.85 * j / n; c = tuple(int(self.paper[k] * (1 - a) + col[k] * a) for k in range(3)); self.line(im, trail[j - 1], trail[j], c, 2)
        if cam_T is not None:
            T = np.linalg.inv(cam_T); C = T[:3, 3]; Rc = T[:3, :3]
            base = [C + Rc @ np.array([sx * 0.045, sy * 0.03, 0.09]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for b in base: self.line(im, C, b, self.teal, 2)
            for j in range(4): self.line(im, base[j], base[(j + 1) % 4], self.teal, 2)
            p = self.px(C)[0].round().astype(int); cv2.circle(im, tuple(p), 6, self.teal, -1, cv2.LINE_AA)
            cv2.putText(im, "camera", (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.teal, 1, cv2.LINE_AA)
        if tool_world is not None:
            Rb = R_world_body; B = body_world
            for k, col in enumerate(((0x40, 0x40, 0xC8), (0x40, 0xA0, 0x40), (0xC8, 0x60, 0x30))):  # body axes x red, y green, z blue (BGR)
                self.line(im, B, B + Rb[:, k] * 0.06, col, 2)
            self.line(im, B, tool_world, self.amber, 2)
            jaw = Rb @ JAW_AXIS_BODY; half = (0.015 + 0.04 * min(max((grip_label - 0.2) / 0.8, 0.0), 1.0)) if grip_label is not None else 0.03
            for s in (-1, 1):
                tip = tool_world + jaw * half * s; self.line(im, tip - Rb[:, 2] * 0.03, tip + Rb[:, 2] * 0.03, self.amber, 3)
            self.line(im, tool_world - jaw * half, tool_world + jaw * half, self.amber, 1)
            p = self.px(tool_world)[0].round().astype(int); cv2.circle(im, tuple(p), 6, self.amber, -1, cv2.LINE_AA)
            cv2.putText(im, "gripper", (p[0] + 10, p[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.amber, 1, cv2.LINE_AA)
        return im
