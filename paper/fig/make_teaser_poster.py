"""Poster teaser: (a) collector with the head camera (iPhone video, 26 s), (b) the head camera's own frame with
detections (glove6, 182 s, square crop), (c) the SO-101 positioner carrying the same camera (phone photo,
2026-11-10) stacked above (d) the YAM from an iPhone rollout frame (336 s), so the robot arm is not hidden behind an inset.
Sources in fig/src; originals in ../media/originals (gitignored) and NAS yam-umi-media/originals."""
import cv2, numpy as np
from pathlib import Path
F = Path(__file__).parent
H, gap = 1080, 24
a = cv2.imread(str(F / "src/collector_iphone_26s.jpg"))[100:1900]                      # 1080 x 1800
b = cv2.imread(str(F / "src/headcam_glove6_182s_overlay.jpg"))[:, 280:1360]             # 1080 x 1080
r = cv2.imread(str(F / "src/rollout_iphone_336s.jpg"))
c = cv2.imread(str(F / "src/positioner_photo_20261110.jpg"))[300:1350, 100:1500]        # positioner + camera, 4:3
d = r[820:1900, 0:1080]                                                                 # YAM + table, 1:1
def fit_h(im, h): return cv2.resize(im, (int(round(im.shape[1] * h / im.shape[0])), h), interpolation=cv2.INTER_AREA)
def fit_w(im, w): return cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1]))), interpolation=cv2.INTER_AREA)
# column: c (4:3) over d (1:1), total height H
W = int(round((H - gap) / 1.75))
c, d = fit_w(c, W), fit_w(d, W)
col = np.full((H, W, 3), 255, np.uint8)
col[:c.shape[0]] = c
col[H - d.shape[0]:] = d
panels = [(fit_h(a, H), ["a"]), (fit_h(b, H), ["b"]), (col, ["c", "d"])]
out = np.full((H, sum(p.shape[1] for p, _ in panels) + gap * 2, 3), 255, np.uint8)
x = 0
for p, labs in panels:
    out[:, x:x + p.shape[1]] = p
    for lab in labs:
        y = 0 if lab != "d" else H - d.shape[0]
        cv2.rectangle(out, (x, y), (x + 110, y + 90), (255, 255, 255), -1)
        cv2.putText(out, f"({lab})", (x + 8, y + 68), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 4, cv2.LINE_AA)
    x += p.shape[1] + gap
cv2.imwrite(str(F / "teaser_poster.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(out.shape)
