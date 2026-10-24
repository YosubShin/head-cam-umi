#!/usr/bin/env python3
"""Pass 2: select diverse frames from detections, calibrate, validate on held-out frames.

Reads calib/C0754_detections.npz produced by detect_charuco.py.
Outputs calib/C0754_intrinsics.yaml (+ json), coverage map, per-frame error report.
"""
import json
import cv2
import numpy as np

NPZ = "calib/C0754_detections.npz"
W, H = 3840, 2160
SQUARE_M = 0.0250
MARKER_M = 0.0180
N_SELECT = 40      # total diverse frames
N_HOLDOUT = 10     # of which held out for validation
MIN_CORNERS = 20   # frames used for calibration need a decent chunk of board
MIN_FRAME_GAP = 10 # in original frame indices (~0.33s) to avoid near-duplicates

MAX_VEL = 150.0  # px/s; above this, motion smear + rolling-shutter shear corrupt corners

data = np.load(NPZ)
idxs = data["frame_indices"]
frames = {int(k): (data[f"c{k}"], data[f"i{k}"]) for k in idxs}

vel_data = np.load("calib/C0754_velocity.npz")
vel = dict(zip(vel_data["frame"].astype(int), vel_data["px_per_s"]))

# --- features for diversity: centroid (norm), log scale, tilt proxy ---
feats = {}
for k, (pts, ids) in frames.items():
    if len(pts) < MIN_CORNERS:
        continue
    if vel.get(k, np.inf) > MAX_VEL:
        continue
    cx, cy = pts.mean(axis=0)
    grid = {int(i): p for i, p in zip(ids, pts)}
    dists = []
    for i, p in grid.items():
        j = i + 1
        if j in grid and (j % 9):
            dists.append(np.linalg.norm(grid[j] - p))
    if not dists:
        continue
    sq = np.mean(dists)
    # tilt proxy: spread of neighbor distances (foreshortening varies across board when tilted)
    tilt = np.std(dists) / sq
    feats[k] = np.array([cx / W, cy / H, np.log(sq), tilt * 3.0])

cands = sorted(feats.keys())
print(f"{len(cands)} candidate frames with >={MIN_CORNERS} corners")

# --- greedy max-min (farthest point) selection in feature space ---
sel = [max(cands, key=lambda k: len(frames[k][0]))]  # seed: most corners
while len(sel) < min(N_SELECT, len(cands)):
    best, best_d = None, -1
    for k in cands:
        if any(abs(k - s) < MIN_FRAME_GAP for s in sel):
            continue
        d = min(np.linalg.norm(feats[k] - feats[s]) for s in sel)
        if d > best_d:
            best, best_d = k, d
    if best is None:
        break
    sel.append(best)
sel.sort()
print(f"selected {len(sel)} frames: {sel}")

# --- split: every 4th selected frame held out ---
holdout = sel[3::4][:N_HOLDOUT]
train = [k for k in sel if k not in holdout]
print(f"train {len(train)}, holdout {len(holdout)}")

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
board = cv2.aruco.CharucoBoard((10, 7), SQUARE_M, MARKER_M, d)

def match(k):
    pts, ids = frames[k]
    obj, img = board.matchImagePoints(pts.reshape(-1, 1, 2), ids.reshape(-1, 1))
    return obj, img

obj_all = [match(k)[0] for k in train]
img_all = [match(k)[1] for k in train]

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_all, img_all, (W, H), None, None
)

# one outlier-rejection pass: drop frames whose rms is far above the median
frame_rms = []
for i in range(len(train)):
    proj, _ = cv2.projectPoints(obj_all[i], rvecs[i], tvecs[i], K, dist)
    e = np.linalg.norm(proj.reshape(-1, 2) - img_all[i].reshape(-1, 2), axis=1)
    frame_rms.append(np.sqrt((e**2).mean()))
med = np.median(frame_rms)
keep = [i for i, r in enumerate(frame_rms) if r < max(2.5 * med, 1.0)]
if len(keep) < len(train):
    dropped = [train[i] for i in range(len(train)) if i not in keep]
    print(f"dropping {len(dropped)} outlier frames (rms > {max(2.5*med,1.0):.2f}px): {dropped}")
    train = [train[i] for i in keep]
    obj_all = [obj_all[i] for i in keep]
    img_all = [img_all[i] for i in keep]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_all, img_all, (W, H), None, None
    )
print(f"\nRMS reprojection (train): {rms:.4f} px")
print("K =\n", K)
print("dist =", dist.ravel())

# per-train-frame errors
per_frame = []
for i, k in enumerate(train):
    proj, _ = cv2.projectPoints(obj_all[i], rvecs[i], tvecs[i], K, dist)
    e = np.linalg.norm(proj.reshape(-1, 2) - img_all[i].reshape(-1, 2), axis=1)
    per_frame.append((k, "train", len(e), float(np.sqrt((e**2).mean())), float(e.max())))

# held-out validation: fixed intrinsics, solvePnP per frame
ho_errs = []
for k in holdout:
    obj, img = match(k)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    e = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    ho_errs.append(e)
    per_frame.append((k, "holdout", len(e), float(np.sqrt((e**2).mean())), float(e.max())))
ho_all = np.concatenate(ho_errs)
ho_rms = float(np.sqrt((ho_all**2).mean()))
print(f"held-out RMS ({len(holdout)} frames, {len(ho_all)} corners): {ho_rms:.4f} px, "
      f"p95 {np.percentile(ho_all,95):.3f}, max {ho_all.max():.3f}")

for k, split, n, frms, fmax in sorted(per_frame):
    print(f"  f{k:5d} {split:7s} n={n:2d} rms={frms:.3f} max={fmax:.3f}")

# --- coverage map: all detections vs selected ---
cov = np.zeros((H // 20, W // 20), np.float32)
cov_sel = np.zeros_like(cov)
for k, (pts, _) in frames.items():
    for x, y in pts:
        xi, yi = min(int(x) // 20, cov.shape[1]-1), min(int(y) // 20, cov.shape[0]-1)
        cov[yi, xi] += 1
        if k in sel:
            cov_sel[yi, xi] += 1
def render(c):
    c = np.sqrt(c / max(c.max(), 1))
    return cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
cv2.imwrite("calib/C0754_coverage_all.png", render(cov))
cv2.imwrite("calib/C0754_coverage_selected.png", render(cov_sel))
frac_all = float((cov > 0).mean())
frac_sel = float((cov_sel > 0).mean())
print(f"coverage (20px cells hit): all detections {frac_all:.1%}, selected {frac_sel:.1%}")

fov_x = 2 * np.degrees(np.arctan(W / (2 * K[0, 0])))
fov_y = 2 * np.degrees(np.arctan(H / (2 * K[1, 1])))
px_per_deg = K[0, 0] * np.pi / 180  # at image center
print(f"HFOV {fov_x:.2f} deg, VFOV {fov_y:.2f} deg, center px/deg {px_per_deg:.1f}")

out = {
    "video": "data/raw/C0754.MP4",
    "image_size": [W, H],
    "board": {"layout": [10, 7], "square_m": SQUARE_M, "marker_m": MARKER_M,
              "dict": "DICT_5X5_250", "note": "letter fallback board; square size from 300dpi PNG, not caliper-verified"},
    "camera_matrix": K.tolist(),
    "dist_coeffs": dist.ravel().tolist(),
    "rms_train_px": float(rms),
    "rms_holdout_px": ho_rms,
    "n_train": len(train), "n_holdout": len(holdout),
    "train_frames": train, "holdout_frames": holdout,
    "hfov_deg": float(fov_x), "vfov_deg": float(fov_y),
    "px_per_deg_center": float(px_per_deg),
    "capture_caveat": "clip recorded at 100M 4:2:0 8-bit (yuv420p), not the locked 200M 4:2:2 10-bit",
}
with open("calib/C0754_intrinsics.json", "w") as f:
    json.dump(out, f, indent=2)
fs = cv2.FileStorage("calib/C0754_intrinsics.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("image_width", W); fs.write("image_height", H)
fs.write("camera_matrix", K); fs.write("distortion_coefficients", dist)
fs.write("rms_train_px", rms); fs.write("rms_holdout_px", ho_rms)
fs.release()
print("wrote calib/C0754_intrinsics.{json,yaml}")
