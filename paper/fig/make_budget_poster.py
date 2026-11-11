"""Poster budget curve: ONE label chain only (head-camera-labelled teleop, round-2 chain, scored on its own
targets), with and without the 194 glove segments, at 24/48/96/172 teleop episodes. The paper's budget.pdf
also overlays the encoder-labelled chain, whose loss is scored on encoder targets and so is not comparable
line-to-line; the poster omits it to avoid that reading. Writes budget_poster.pdf."""
import csv, collections
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
F = Path(__file__).resolve().parent
rows = collections.defaultdict(list)
for r in csv.DictReader(open(F.parents[1] / "docs/results/head_val_curves.csv")):
    if r["val_loss"] != "NA": rows[r["run"]].append(float(r["val_loss"]))
m = {k: min(v) for k, v in rows.items()}
n = [24, 48, 96, 172]
tele = [m[f"head_HHc{k}_act50@robotval"] for k in [24, 48, 96]] + [m["head_HHc_act50@robotval"]]
mix = [m[f"mix_HHs3c{k}_act50@robotval"] for k in [24, 48, 96]] + [m["mix_HHs3c_act50@robotval"]]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(n, tele, "o-", color="#444", lw=2, label="teleop only")
ax.plot(n, mix, "s-", color="#c0392b", lw=2, label="teleop + 194 glove demos")
for x, y in zip(n, tele): ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color="#444")
for x, y in zip(n, mix): ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color="#c0392b")
ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels(n); ax.minorticks_off()
ax.set_xlabel("teleoperated episodes in the training set"); ax.set_ylabel("robot val loss (6 held-out eps)")
ax.set_ylim(0.22, 0.62)
ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(F / "budget_poster.pdf")
print({"tele": tele, "mix": mix})
