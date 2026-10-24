#!/usr/bin/env python3
"""Does losing sight of the BLOCK explain the forward camera's disadvantage?

A raw visibility-vs-error correlation is confounded: the block goes out of the
forward view mostly during particular task phases, and phases differ in
intrinsic difficulty. But Cf and Cw are built from the SAME episodes with the
SAME ground-truth labels, so each val frame has an error from both arms.
Differencing them,

    delta(i) = err_Cf(i) - err_Cw(i)

cancels whatever makes frame i intrinsically hard. Cf and Cw still differ in
resolution and viewpoint canonicalisation, but those are frame-independent, so
they shift delta by a CONSTANT. Any part of delta that varies with
forward-view block visibility is therefore attributable to the view losing the
object — the interaction is identified even though the main effect is not.

Usage: PYTHONPATH=~/co/i2rt python scripts/vis_causal.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from val_regions import per_frame_errors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # repo root: calib/, train_runs/, rollout_logs/
CF = ROOT / "train_runs/all7_Cf_act50/checkpoints/048000/pretrained_model"
CW = ROOT / "train_runs/all7_Cw_act50/checkpoints/048000/pretrained_model"


def main():
    cf = per_frame_errors(CF)
    cw = per_frame_errors(CW)
    assert len(cf["err_all"]) == len(cw["err_all"]), \
        f"frame counts differ: {len(cf['err_all'])} vs {len(cw['err_all'])}"
    # both arms carry GT labels, so their action channels must agree frame for
    # frame — this is the alignment proof, not an assumption
    dg = np.abs(cf["g_act"] - cw["g_act"])
    print(f"alignment check: max |Cf.g_act - Cw.g_act| = {dg.max():.4f} "
          f"(mean {dg.mean():.5f})")
    if dg.max() > 0.05:
        raise SystemExit("frames are NOT aligned — comparison invalid")

    vis = cf["vis_b0587"]
    grasp = cf["g_act"] < 0.5
    delta = cf["err_all"] - cw["err_all"]
    gone = vis < 2e-4

    print(f"\n{len(delta)} paired frames; mean delta (Cf-Cw) "
          f"{delta.mean():+.4f} rad  [Cf worse when positive]")
    for tag, m in (("all frames", np.ones_like(grasp)),
                   ("grasp region", grasp),
                   ("pre-grasp only", ~grasp)):
        d, v, g = delta[m], vis[m], gone[m]
        if m.sum() == 0:
            continue
        # Spearman on visible frames only (rank-based: immune to the heavy
        # right tail of visible-area)
        vv, dd = v[~g], d[~g]
        rank = lambda a: np.argsort(np.argsort(a)).astype(float)
        rho = (np.corrcoef(rank(vv), rank(dd))[0, 1]
               if len(vv) > 10 else float("nan"))
        print(f"\n  --- {tag} (n={m.sum()}) ---")
        print(f"    delta | block INVISIBLE in forward: {d[g].mean():+.4f} "
              f"(n={g.sum()})")
        print(f"    delta | block visible            : {d[~g].mean():+.4f} "
              f"(n={(~g).sum()})")
        print(f"    penalty attributable to losing the block: "
              f"{d[g].mean() - d[~g].mean():+.4f} rad")
        print(f"    Spearman rho(visible area, delta) among visible frames: "
              f"{rho:+.3f}  [negative = more visible, less Cf penalty]")
        if g.sum() > 10 and (~g).sum() > 10:
            # bootstrap the invisible-vs-visible contrast
            rng = np.random.default_rng(0)
            bs = [d[rng.integers(0, g.sum(), g.sum())[
                      np.zeros(1, int)]].mean() for _ in range(0)]  # noqa
            gi, vi = np.where(g)[0], np.where(~g)[0]
            diffs = [d[rng.choice(gi, len(gi))].mean()
                     - d[rng.choice(vi, len(vi))].mean() for _ in range(2000)]
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            print(f"    95% CI on that penalty: [{lo:+.4f}, {hi:+.4f}]"
                  f"{'  (excludes 0)' if lo * hi > 0 else '  (includes 0)'}")

    # how much of the total Cf-Cw gap could visibility account for?
    share = (delta[gone].mean() - delta[~gone].mean()) * gone.mean()
    print(f"\nInvisible frames are {100 * gone.mean():.1f}% of frames; their "
          f"excess penalty contributes {share:+.4f} rad of the "
          f"{delta.mean():+.4f} rad total gap "
          f"= {100 * share / delta.mean():.0f}% of it"
          if abs(delta.mean()) > 1e-9 else "")


if __name__ == "__main__":
    main()
