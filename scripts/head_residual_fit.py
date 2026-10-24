#!/usr/bin/env python3
"""Learn where the operator's head goes from the glove pose (viewpoint model), held-out by episode.

Baselines and models, all predicting Y = [head position offset from N (mm, 3) | view (yaw, pitch) deg]:
  static        : position 0, view = train mean
  gain tracker  : position 0, view = a + g * bearing(grasp point)   (per-axis least squares; the human gain)
  ridge         : linear on the 9 features (+ bearing), ridge-regularised
  mlp           : 2 x 64 tanh, early-stopped on a train-internal split
Reports held-out median position error (mm) and view error (deg).

  python scripts/head_residual_fit.py                      # 15 held-out episodes (every 7th)
  python scripts/head_residual_fit.py --save calib/head_residual_model.json
(run in the lerobot_alohamini env for torch)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def bearing_deg(v):
    return np.degrees(np.stack([np.arctan2(v[:, 0], v[:, 2]), np.arctan2(v[:, 1], v[:, 2])], 1))


def feats(X):
    """9 raw features + bearing of the grasp point (2) + bearing of the history point (2) + distance (1)."""
    return np.concatenate([X, bearing_deg(X[:, :3]), bearing_deg(X[:, 6:9]), np.linalg.norm(X[:, :3], axis=1, keepdims=True)], 1)


def errs(Y, Yp):
    pos = np.linalg.norm(Y[:, :3] - Yp[:, :3], axis=1)
    view = np.linalg.norm(Y[:, 3:] - Yp[:, 3:], axis=1)
    return float(np.median(pos)), float(np.median(view)), float(np.percentile(pos, 90)), float(np.percentile(view, 90))


def fit_ridge(F, Y, lam):
    mu, sd = F.mean(0), F.std(0) + 1e-9; Fz = (F - mu) / sd; A = np.c_[Fz, np.ones(len(Fz))]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return {"mu": mu, "sd": sd, "W": W}


def pred_ridge(m, F):
    return np.c_[(F - m["mu"]) / m["sd"], np.ones(len(F))] @ m["W"]


def fit_mlp(F, Y, seed=0, hidden=64, epochs=300):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu, sd = F.mean(0), F.std(0) + 1e-9; ymu, ysd = Y.mean(0), Y.std(0) + 1e-9
    Fz = torch.tensor((F - mu) / sd, dtype=torch.float32); Yz = torch.tensor((Y - ymu) / ysd, dtype=torch.float32)
    idx = rng.permutation(len(Fz)); nv = len(idx) // 8; vi, ti = idx[:nv], idx[nv:]
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); best = (1e9, None)
    for ep in range(epochs):
        net.train(); perm = rng.permutation(ti)
        for b in range(0, len(perm), 256):
            bi = perm[b:b + 256]; loss = ((net(Fz[bi]) - Yz[bi]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(Fz[vi]) - Yz[vi]) ** 2).mean())
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    return {"net": net, "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd}


def pred_mlp(m, F):
    import torch
    with torch.no_grad():
        return m["net"](torch.tensor((F - m["mu"]) / m["sd"], dtype=torch.float32)).numpy() * m["ysd"] + m["ymu"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "calib/head_residual_glove5.npz"))
    ap.add_argument("--every", type=int, default=7, help="hold out every k-th episode")
    ap.add_argument("--lam", type=float, default=10.0); ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--save", default=None, help="fit on ALL episodes and save the model (ridge + mlp weights) here")
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True); X, Y, ep = d["X"], d["Y"], d["ep"]
    eps = sorted(set(ep)); held = set(eps[::a.every]); te = np.array([e in held for e in ep]); tr = ~te
    F = feats(X)
    print(f"{len(X)} samples, {len(eps)} episodes; held out {len(held)} episodes ({te.sum()} samples)")
    print(f"{'model':14s} {'pos med':>8s} {'pos p90':>8s} {'view med':>9s} {'view p90':>9s}")
    rows = {}

    def report(name, Yp):
        rows[name] = errs(Y[te], Yp); print(f"{name:14s} {rows[name][0]:8.1f} {rows[name][2]:8.1f} {rows[name][1]:9.2f} {rows[name][3]:9.2f}")

    Ys = np.zeros_like(Y[te]); Ys[:, 3:] = Y[tr][:, 3:].mean(0); report("static", Ys)
    B = bearing_deg(X[:, :3]); Yg = np.zeros_like(Y[te])
    for j in range(2):                                   # view_j = a + g * bearing_j
        A = np.c_[B[tr][:, j], np.ones(tr.sum())]; c = np.linalg.lstsq(A, Y[tr][:, 3 + j], rcond=None)[0]; Yg[:, 3 + j] = c[0] * B[te][:, j] + c[1]
        print(f"   gain {'yaw' if j == 0 else 'pitch'} {c[0]:.2f}")
    report("gain tracker", Yg)
    rm = fit_ridge(F[tr], Y[tr], a.lam); report("ridge", pred_ridge(rm, F[te]))
    mm = fit_mlp(F[tr], Y[tr], hidden=a.hidden); report("mlp", pred_mlp(mm, F[te]))
    # what the models do with position: per-axis std of the prediction vs the truth (does it move at all?)
    Yp = pred_mlp(mm, F[te]); print("   held-out head position std (mm) truth", np.round(Y[te][:, :3].std(0), 1), " mlp prediction", np.round(Yp[:, :3].std(0), 1),
                                 " corr per axis", np.round([np.corrcoef(Y[te][:, k], Yp[:, k])[0, 1] for k in range(3)], 2))
    if a.save:
        rm = fit_ridge(F, Y, a.lam); mm = fit_mlp(F, Y, hidden=a.hidden)
        json.dump({"ridge": {k: v.tolist() for k, v in rm.items()},
                   "mlp": {"mu": mm["mu"].tolist(), "sd": mm["sd"].tolist(), "ymu": mm["ymu"].tolist(), "ysd": mm["ysd"].tolist(),
                           "layers": [[w.detach().numpy().tolist() for w in (lay.weight, lay.bias)] for lay in mm["net"] if hasattr(lay, "weight")]},
                   "features": "X(9) + bearing(grasp) + bearing(hist) + dist", "T_N": d["T_N"].tolist(), "heldout": rows}, open(a.save, "w"))
        print("wrote", a.save)


if __name__ == "__main__":
    main()
