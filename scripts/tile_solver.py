"""Per-frame camera pose from the taped workspace tiles (36h11), fast path:
detect on a half-resolution frame, refine corners on the full frame, multi-tile PnP.
Shared by head_pose_fast.py (recorded video) and cam_arm_poses.py (live camera).
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def tag_obj(edge_m):
    h = edge_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def to_T(rv, tv):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))
    T[:3, 3] = np.asarray(tv, float).ravel()
    return T


def solve_frame(corners_by_id, layout, K, dist):
    """Multi-tile PnP over all decoded tiles; drops a bumped tile. Returns (T_cam_world, rms_px, n_tiles) or None."""
    edge = layout["edge_mm"] / 1000.0
    obj_t = tag_obj(edge)
    O, I = [], []
    for i, c in corners_by_id.items():
        t = layout["tiles"].get(str(i))
        if t is None:
            continue
        Tw = to_T(t["rvec"], np.array(t["xyz_mm"]) / 1000.0)
        O.append((Tw[:3, :3] @ obj_t.T).T + Tw[:3, 3])
        I.append(c.reshape(4, 2).astype(float))
    if not O:
        return None

    def fit(Os, Is):
        Oc, Ic = np.concatenate(Os), np.concatenate(Is)
        flags = cv2.SOLVEPNP_IPPE_SQUARE if len(Oc) == 4 else cv2.SOLVEPNP_SQPNP
        ok, rv, tv = cv2.solvePnP(Oc, Ic, K, dist, flags=flags)
        if not ok:
            return None
        if len(Oc) > 4:
            ok, rv, tv = cv2.solvePnP(Oc, Ic, K, dist, rvec=rv, tvec=tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(Oc, rv, tv, K, dist)
        per_tile = np.sqrt(((proj.reshape(-1, 2) - Ic) ** 2).sum(1)).reshape(-1, 4).mean(1)
        return rv, tv, per_tile

    r = fit(O, I)
    if r is None:
        return None
    rv, tv, per_tile = r
    while len(O) >= 4 and per_tile.max() > max(8.0, 4 * np.median(per_tile)):
        j = int(np.argmax(per_tile)); O.pop(j); I.pop(j)
        r = fit(O, I)
        if r is None:
            return None
        rv, tv, per_tile = r
    return to_T(rv, tv), float(np.sqrt((per_tile ** 2).mean())), len(O)


class TileSolver:
    def __init__(self, intr="calib/B0587b_f3_intrinsics.json", tags="calib/tags36h11_yosub.json",
                 layout="calib/workspace_tiles_pilot_v2.json"):
        import labels_lib as L
        self.K, self.dist = L.load_intrinsics(str(ROOT / intr) if not Path(intr).is_absolute() else intr)
        self.layout = json.load(open(ROOT / layout))
        self.ids_ok = set(int(i) for i in json.load(open(ROOT / tags))["ids"])
        p = cv2.aruco.DetectorParameters(); p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)
        self.crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    def solve(self, bgr):
        """(T_cam_world, rms_px, n_tiles) or None for one full-resolution BGR frame."""
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cs, ids, _ = self.det.detectMarkers(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        if ids is None:
            return None
        by = {}
        for i, c in zip(ids.ravel(), cs):
            if int(i) in self.ids_ok:
                c2 = (c.reshape(4, 2) * 2).astype(np.float32)
                by[int(i)] = cv2.cornerSubPix(g, c2, (5, 5), (-1, -1), self.crit).reshape(1, 4, 2)
        return solve_frame(by, self.layout, self.K, self.dist) if by else None
