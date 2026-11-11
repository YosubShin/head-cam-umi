#!/usr/bin/env python3
"""Demo-table loop for the UMI Arena: 1920x1080, 30 fps, no audio.  python media/make_demo_loop.py -> media/demo-loop.mp4
Inputs outside this repo: the phone originals in ../yam-umi/media/originals and the yam_glove6 head stream (data/raw/nas).
Segments: title; collection (phone, 1.6x); the head camera's own frame with the three label sources drawn;
deployment (phone + positioner camera); one placement seen by three policies (1.5x); end card with the QR."""
import subprocess, glob, json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
S = REPO / "media/.demo_cache"; S.mkdir(exist_ok=True)
PHONE = REPO.parent / "yam-umi/media/originals/data-collection_iphone_20261016.mov"
HEAD = REPO / "data/raw/nas/yam_glove6_b0587b.mkv"
POSTER = REPO / "paper/poster/poster.pdf"
G = REPO / "rollout_logs/grid_20261016"
OUT = REPO / "media/demo-loop.mp4"
W, H, FPS = 1920, 1080, 30
NAVY, PAPER, AMBER, TEAL, INK, MUTED, WHITE = (0x16,0x32,0x4F), (0xFB,0xFA,0xF7), (0xD9,0x82,0x2B), (0x2A,0x7F,0x8F), (0x1E,0x2A,0x38), (0x5C,0x66,0x70), (255,255,255)
MAG = (0xA0, 0x3C, 0xC8)
FB = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"; FR = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"
def font(sz, bold=False): return ImageFont.truetype(FB if bold else FR, sz)
def rgb2bgr(c): return (c[2], c[1], c[0])

def canvas(color=PAPER):
    return Image.new("RGB", (W, H), color)
def to_bgr(im): return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
def to_pil(bgr): return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# ---------- pre-cuts (cached) ----------
def sh(*a): subprocess.run(a, check=True)
if not (S / "seg_coll.mp4").exists():  # phone, 17-57 s of the take, 1.6x, portrait 608x1080
    sh("ffmpeg", "-v", "error", "-y", "-ss", "17", "-t", "40", "-i", str(PHONE), "-vf", "setpts=PTS/1.6,scale=608:1080", "-r", "30", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(S / "seg_coll.mp4"))
if not (S / "seg_head_raw.mp4").exists():  # glove6 main take 169-190 s: episodes 38 and 39 with the reset between
    sh("ffmpeg", "-v", "error", "-y", "-ss", "169", "-t", "21", "-i", str(HEAD), "-vf", "scale=1920:1080", "-r", "25", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(S / "seg_head_raw.mp4"))
if not (S / "qr_crop-1.png").exists():  # the poster's QR (github.com/YosubShin/head-cam-umi)
    sh("pdftoppm", "-r", "216", "-f", "1", "-l", "1", "-png", "-x", "8930", "-y", "80", "-W", "650", "-H", "650", str(POSTER), str(S / "qr_crop"))

# ---------- ffmpeg sink ----------
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
nframes = 0
def emit(bgr, n=1):
    global nframes
    for _ in range(n):
        ff.stdin.write(np.ascontiguousarray(bgr).tobytes()); nframes += 1
def fade(a, b, secs=0.4):
    n = int(secs * FPS)
    for i in range(1, n + 1):
        t = i / (n + 1); emit(cv2.addWeighted(a, 1 - t, b, t, 0))

# ---------- cards ----------
def text(d, xy, s, f, fill, anchor="la"): d.text(xy, s, font=f, fill=fill, anchor=anchor)
def wrap(d, s, f, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def title_card():
    im = canvas(NAVY); d = ImageDraw.Draw(im)
    text(d, (120, 330), "A Head Camera Is (Almost) All You Need", font(88, True), WHITE)
    text(d, (120, 450), "An electronics-free UMI glove labelled and observed by one moving camera", font(44), (0xC9,0xD3,0xDE))
    text(d, (120, 600), "Yosub Shin¹,  Yuan Liu²,  Igor Molybog¹", font(40, True), WHITE)
    text(d, (120, 660), "¹University of Hawai'i at Mānoa    ²Independent", font(34), (0xC9,0xD3,0xDE))
    text(d, (120, 860), "The UMI Arena  ·  CoRL 2026", font(34), AMBER)
    return to_bgr(im)

def end_card():
    im = canvas(NAVY); d = ImageDraw.Draw(im)
    qr = cv2.imread(str(S / "qr_crop-1.png"))
    g = (qr.min(axis=2) > 240); rows = np.where(g.mean(axis=1) > 0.5)[0]; cols = np.where(g.mean(axis=0) > 0.5)[0]
    y0, y1, x0, x1 = rows.min(), rows.max(), cols.min(), cols.max()
    qr = qr[y0:y1 + 1, x0:x1 + 1]; qr = cv2.resize(qr, (520, 520), interpolation=cv2.INTER_NEAREST)
    bgr = to_bgr(im); bgr[280:800, 1280:1800] = qr
    im = to_pil(bgr); d = ImageDraw.Draw(im)
    text(d, (120, 300), "Code, calibration, data", font(64, True), WHITE)
    text(d, (120, 400), "github.com/YosubShin/head-cam-umi", font(44), AMBER)
    text(d, (120, 470), "arXiv:2610.27622", font(44), AMBER)
    for i, s in enumerate(["$22 printed glove, 370 g, no electronics", "$122 4K head camera, 103 g on the head",
                           "One frame: the observation and all three labels", "Hardware and tools: MIT licence"]):
        text(d, (120, 580 + 58 * i), "•  " + s, font(34), (0xC9,0xD3,0xDE))
    text(d, (120, 900), "The UMI Arena  ·  CoRL 2026", font(34), (0x8A,0x9B,0xB0))
    return to_bgr(im)

def header_bar(im_bgr, step, title):
    """Top band: step number in amber, title in white."""
    im = to_pil(im_bgr); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 96], fill=NAVY)
    text(d, (48, 48), step, font(40, True), AMBER, "lm")
    text(d, (140, 48), title, font(40, True), WHITE, "lm")
    return to_bgr(im)

# ---------- 1 collection ----------
def seg_collection():
    cap = cv2.VideoCapture(str(S / "seg_coll.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = canvas(PAPER); d = ImageDraw.Draw(base)
    x0 = 760
    text(d, (x0, 180), "Collecting demonstrations", font(56, True), NAVY)
    bullets = [("The operator wears the head camera", "a $122 4K USB camera on a $10 strap, 103 g"),
               ("The glove is printed and electronics-free", "$22, 370 g: no wrist camera, IMU or encoder"),
               ("No protocol to follow", "what the operator sees is what the policy will see"),
               ("2–3× the demonstrations per minute of teleoperation", "6–8 s per demonstration, continuous recording")]
    y = 300
    for h, b in bullets:
        d.rectangle([x0, y + 14, x0 + 18, y + 32], fill=TEAL)
        text(d, (x0 + 40, y), h, font(38, True), NAVY)
        text(d, (x0 + 40, y + 50), b, font(32), MUTED); y += 140
    text(d, (x0, 960), "Phone view, 1.6× speed", font(28), MUTED)
    base = to_bgr(base)
    first = None
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.resize(fr, (608, 1080)) if fr.shape[:2] != (1080, 608) else fr
        out = base.copy(); out[:, 80:688] = fr
        out = header_bar(out, "1", "Collect with the glove and the head camera")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 2 labels ----------
def seg_labels():
    a = cv2.aruco; p = a.DetectorParameters(); p.cornerRefinementMethod = a.CORNER_REFINE_SUBPIX
    d4 = a.ArucoDetector(a.getPredefinedDictionary(a.DICT_4X4_50), p)
    d36 = a.ArucoDetector(a.getPredefinedDictionary(a.DICT_APRILTAG_36h11), p)
    cap = cv2.VideoCapture(str(S / "seg_head_raw.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    legend = canvas((0, 0, 0)); d = ImageDraw.Draw(legend)
    # legend panel drawn once (alpha-blended later)
    first = None; src_fps = 25
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        c4, i4, _ = d4.detectMarkers(g); c36, i36, _ = d36.detectMarkers(g)
        nt = 0 if i36 is None else len(i36); nb = nta = 0
        def poly(cs, col, w=4):
            for c in cs: cv2.polylines(fr, [c.reshape(-1, 1, 2).astype(np.int32)], True, rgb2bgr(col), w, cv2.LINE_AA)
        if i36 is not None: poly(c36, TEAL)
        if i4 is not None:
            ids = i4.ravel(); ball = [c for c, k in zip(c4, ids) if 25 <= k <= 35]; tails = [c for c, k in zip(c4, ids) if 36 <= k <= 49]
            nb, nta = len(ball), len(tails); poly(ball, AMBER); poly(tails, MAG)
        # legend panel bottom-left
        ov = fr.copy(); cv2.rectangle(ov, (40, 760), (900, 1040), (255, 255, 255), -1); fr = cv2.addWeighted(ov, 0.82, fr, 0.18, 0)
        im = to_pil(fr); d = ImageDraw.Draw(im)
        rows = [(TEAL, "paper tiles", "camera pose", nt), (AMBER, "ball faces", "gripper pose", nb), (MAG, "tails on the jaws", "jaw width", nta)]
        for k, (col, src, lab, cnt) in enumerate(rows):
            y = 800 + 80 * k
            d.rectangle([70, y + 6, 106, y + 42], outline=col, width=5)
            text(d, (130, y + 24), src, font(36), INK, "lm")
            text(d, (470, y + 24), "→", font(36), MUTED, "lm")
            text(d, (530, y + 24), lab, font(36, True), NAVY, "lm")
            text(d, (870, y + 24), f"{cnt}", font(36), MUTED, "rm")
        out = header_bar(to_bgr(im), "2", "One frame carries the observation and all three labels")
        im = to_pil(out); d = ImageDraw.Draw(im)
        text(d, (W - 48, 48), "Head camera, 1× speed", font(28), (0xC9,0xD3,0xDE), "rm")
        out = to_bgr(im)
        if first is None: first = out
        # 25 fps source -> 30 fps output: repeat every 5th frame
        emit(out, 2 if i % 5 == 4 else 1)
    return first, out

# ---------- 3 deployment ----------
def seg_deploy():
    cap = cv2.VideoCapture(str(REPO / "media/head-rollout-clip.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); sfps = cap.get(cv2.CAP_PROP_FPS)
    base = canvas(PAPER); d = ImageDraw.Draw(base)
    lines = ["The same camera rides a $122 SO-101 arm beside the robot. It follows the gripper from the robot's encoders alone,",
             "placed by a linear model of where the collector's head was relative to the hand. No wrist camera."]
    for k, s in enumerate(lines): text(d, (W // 2, 905 + 44 * k), s, font(30), MUTED, "mm")
    text(d, (W // 2, 1000), "Phone view and the positioner camera, one rollout of a head-camera policy", font(26), (0x8A,0x9B,0xB0), "mm")
    base = to_bgr(base); first = None
    t_out = 0.0; i = 0; frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    dur = len(frames) / sfps; total = int(round(dur * FPS))
    for j in range(total):
        fr = frames[min(len(frames) - 1, int(j / FPS * sfps))]
        fr = cv2.resize(fr, (1920, int(round(540 * 1920 / 1364))), interpolation=cv2.INTER_AREA)
        h = fr.shape[0]; y0 = 110 + (880 - 110 - h) // 2
        out = base.copy(); out[y0:y0 + h] = fr
        out = header_bar(out, "3", "Deploy: the camera moves to a positioner beside the robot")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 4 comparison ----------
def seg_compare():
    R = {(r["policy"], r["scene"]): r for r in map(json.loads, open(G / "trials.jsonl"))}
    cols = [("mix_gHs3", "b0587b", "Head camera on the positioner", "success"),
            ("mix_gHA", "fish", "Wrist fisheye, head-camera labels", "dropped during the carry"),
            ("mix_FB", "b0587", "Fixed forward camera", "no grasp")]
    SPEED = 1.5
    vids = []
    for pol, cam, name, outc in cols:
        own = cv2.VideoCapture(glob.glob(str(G / f"{pol}_S23r0_*_{cam}.mp4"))[0]); wld = cv2.VideoCapture(glob.glob(str(G / f"{pol}_S23r0_*_world.mp4"))[0])
        el = R[(pol, 23)]["elapsed_s"]
        def grab(cap):
            fs = []
            while True:
                ok, f = cap.read()
                if not ok: return fs
                fs.append(f)
        vids.append((grab(own), grab(wld), el, name, outc))
    tmax = max(v[2] for v in vids); total = int(round(tmax / SPEED * FPS)) + FPS * 2  # hold 2 s at the end
    CW = 640; first = None
    for j in range(total):
        t = j / FPS * SPEED
        im = canvas(PAPER); d = ImageDraw.Draw(im)
        text(d, (W // 2, 140), "Placement 23: each policy's own camera (top) and a fixed third-person camera (bottom), one clock", font(30), MUTED, "mm")
        text(d, (W - 48, 48), f"t = {min(t, tmax):4.1f} s   (1.5×)", font(30), (0xC9,0xD3,0xDE), "rm")
        out = to_bgr(im)
        for c, (own, wld, el, name, outc) in enumerate(vids):
            x0 = c * CW; done = t >= el; tt = min(t, el - 1e-3)
            fo = own[min(len(own) - 1, int(tt / el * len(own)))]; fw = wld[min(len(wld) - 1, int(tt / el * len(wld)))]
            # own view -> 640x360 letterboxed
            ho, wo = fo.shape[:2]; s = min(CW / wo, 360 / ho); fo = cv2.resize(fo, (int(wo * s), int(ho * s)), interpolation=cv2.INTER_AREA)
            tile = np.zeros((360, CW, 3), np.uint8); ox = (CW - fo.shape[1]) // 2; tile[:fo.shape[0], ox:ox + fo.shape[1]] = fo
            fw = cv2.resize(fw, (CW, 360), interpolation=cv2.INTER_AREA)
            if done: tile = (tile * 0.45).astype(np.uint8); fw = (fw * 0.45).astype(np.uint8)
            out[250:610, x0:x0 + CW] = tile; out[614:974, x0:x0 + CW] = fw
        im = to_pil(out); d = ImageDraw.Draw(im)
        for c, (own, wld, el, name, outc) in enumerate(vids):
            x0 = c * CW; done = t >= el
            d.rectangle([x0 + 4, 180, x0 + CW - 4, 244], fill=NAVY)
            text(d, (x0 + CW // 2, 212), name, font(28, True), WHITE, "mm")
            if done:
                col = TEAL if outc == "success" else AMBER
                d.rectangle([x0 + 4, 980, x0 + CW - 4, 1040], fill=col)
                text(d, (x0 + CW // 2, 1010), f"{outc}  ·  {el:.0f} s", font(30, True), WHITE, "mm")
        out = header_bar(to_bgr(im), "4", "One placement, seen by each policy")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- assemble ----------
black = np.zeros((H, W, 3), np.uint8)
tc = title_card(); emit(tc, FPS * 4)
prev = tc
for seg in (seg_collection, seg_labels, seg_deploy, seg_compare):
    # we need the first frame before emitting; segments emit as they go, so fade from prev to black then start
    fade(prev, black, 0.3)
    first, last = seg(); prev = last
fade(prev, black, 0.3)
ec = end_card(); emit(ec, FPS * 6); fade(ec, black, 0.5)
ff.stdin.close(); ff.wait()
print(OUT, nframes, f"{nframes / FPS:.1f} s")
