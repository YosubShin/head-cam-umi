#!/usr/bin/env python3
"""Demo-table loop for the UMI Arena device track (YAM-UMI, with head-cam-umi as its current tracking mode).
1920x1080, 30 fps, no audio.  python media/make_demo_loop.py -> media/demo-loop.mp4

Inputs outside this repo: ../yam-umi/media (stills, the YAM-UMI paper video, the phone originals), the yam_glove6
head stream (data/raw/nas) and, for the last segment, the spring glove's yam_glove9 head stream with its detection
cache and calib files (skipped when absent). Segments: title; the glove hardware; the original labelling (fixed
camera + wrist fisheye, from the YAM-UMI video); bridge card; collection with the head camera (phone, 1.6x); the
head camera's own frame with the three label sources drawn; deployment (phone + positioner camera); one placement
seen by three policies (1.5x); next step, the passive grip-force indicator (a jug lift from the phone, then the slit readout
from the head frame); end card with
both repos' QR codes.  --short writes media/demo-loop-60s.mp4 (about 60 s) for a talk."""
import subprocess, glob, json, sys, argparse
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
S = REPO / "media/.demo_cache"; S.mkdir(exist_ok=True)
YAM = REPO.parent / "yam-umi/media"
PHONE = YAM / "originals/data-collection_iphone_20261016.mov"
JUG = YAM / "originals/spring-jug-lift_iphone_20261111.mov"  # phone: the spring glove lifting a gallon jug by its handle
HEAD = REPO / "data/raw/nas/yam_glove6_b0587b.mkv"
POSTER = REPO / "paper/poster/poster.pdf"
G = REPO / "rollout_logs/grid_20261016"
SPRING_STEM = "yam_glove9_b0587b"
SHORT = "--short" in sys.argv  # the 60 s version for a talk: same segments, shorter cuts, faster playback
OUT = REPO / ("media/demo-loop-60s.mp4" if SHORT else "media/demo-loop.mp4")
W, H, FPS = 1920, 1080, 30
NAVY, PAPER, AMBER, TEAL, INK, MUTED, WHITE = (0x16,0x32,0x4F), (0xFB,0xFA,0xF7), (0xD9,0x82,0x2B), (0x2A,0x7F,0x8F), (0x1E,0x2A,0x38), (0x5C,0x66,0x70), (255,255,255)
MAG = (0xA0, 0x3C, 0xC8); LIGHT = (0xC9,0xD3,0xDE); DIM = (0x8A,0x9B,0xB0)
FB = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"; FR = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"
def font(sz, bold=False): return ImageFont.truetype(FB if bold else FR, sz)
def rgb2bgr(c): return (c[2], c[1], c[0])
def canvas(color=PAPER): return Image.new("RGB", (W, H), color)
def to_bgr(im): return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
def to_pil(bgr): return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
def text(d, xy, s, f, fill, anchor="la"): d.text(xy, s, font=f, fill=fill, anchor=anchor)
def wrap(d, s, f, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines
def fit(im, w, h):
    """Scale to cover w x h, centre crop."""
    s = max(w / im.shape[1], h / im.shape[0]); im = cv2.resize(im, (int(round(im.shape[1] * s)), int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y0 = (im.shape[0] - h) // 2; x0 = (im.shape[1] - w) // 2; return im[y0:y0 + h, x0:x0 + w]

# ---------- pre-cuts (cached) ----------
def sh(*a): subprocess.run(a, check=True)
if not (S / "seg_coll.mp4").exists():  # phone: the four demonstrations of the take (resets cut), 1.6x, portrait 608x1080
    wins = [(2.0, 10.5), (19.5, 29.0), (41.0, 50.5), (61.0, 69.5)]
    fc = "".join(f"[0:v]trim={a}:{b},setpts=(PTS-STARTPTS)/1.6,scale=608:1080[v{k}];" for k, (a, b) in enumerate(wins)) + "".join(f"[v{k}]" for k in range(len(wins))) + f"concat=n={len(wins)}:v=1:a=0[v]"
    sh("ffmpeg", "-v", "error", "-y", "-i", str(PHONE), "-filter_complex", fc, "-map", "[v]", "-r", "30", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(S / "seg_coll.mp4"))
if not (S / "seg_head_raw.mp4").exists():  # glove6 main take 169-190 s: episodes 38 and 39 with the reset between
    sh("ffmpeg", "-v", "error", "-y", "-ss", "169", "-t", "21", "-i", str(HEAD), "-vf", "scale=1920:1080", "-r", "25", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(S / "seg_head_raw.mp4"))
if not (S / "seg_yam.mp4").exists():  # YAM-UMI paper video 23.5-34 s: one demonstration (2x) then the labels split screen
    sh("ffmpeg", "-v", "error", "-y", "-ss", "23.5", "-t", "10.5", "-i", str(YAM / "yam-umi-paper-video.mp4"), "-vf", "scale=1748:984", "-r", "30", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(S / "seg_yam.mp4"))
if JUG.exists() and not (S / "seg_jug.mp4").exists():  # second lift of the take, 7.8-14.0 s at 1x, portrait 608x1080, plus two stills for the slit crops
    sh("ffmpeg", "-v", "error", "-y", "-ss", "7.8", "-t", "6.2", "-i", str(JUG), "-vf", "scale=608:1080", "-r", "30", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(S / "seg_jug.mp4"))
    for t, name in (("9.0", "jug_rest.png"), ("11.0", "jug_up.png")): sh("ffmpeg", "-v", "error", "-y", "-ss", t, "-i", str(JUG), "-frames:v", "1", str(S / name))
if not (S / "qr_yamumi.png").exists():  # QR for github.com/YosubShin/yam-umi via the LaTeX qrcode package
    (S / "qr.tex").write_text("\\documentclass{standalone}\\usepackage{qrcode}\\begin{document}\\qrcode[height=6cm]{https://github.com/YosubShin/yam-umi}\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "qr.tex"], cwd=S, check=True, capture_output=True); sh("pdftoppm", "-r", "300", "-png", "-singlefile", str(S / "qr.pdf"), str(S / "qr_yamumi"))
if not (S / "qr_crop-1.png").exists():  # the poster's QR (github.com/YosubShin/head-cam-umi)
    sh("pdftoppm", "-r", "216", "-f", "1", "-l", "1", "-png", "-x", "8930", "-y", "80", "-W", "650", "-H", "650", str(POSTER), str(S / "qr_crop"))

# ---------- ffmpeg sink ----------
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
nframes = 0; black = np.zeros((H, W, 3), np.uint8)
def emit(bgr, n=1):
    global nframes
    for _ in range(n):
        ff.stdin.write(np.ascontiguousarray(bgr).tobytes()); nframes += 1
def fade(a, b, secs=0.4):
    n = int(secs * FPS)
    for i in range(1, n + 1):
        t = i / (n + 1); emit(cv2.addWeighted(a, 1 - t, b, t, 0))

def header_bar(im_bgr, step, title):
    im = to_pil(im_bgr); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 96], fill=NAVY)
    text(d, (48, 48), step, font(40, True), AMBER, "lm"); text(d, (140, 48), title, font(40, True), WHITE, "lm")
    return to_bgr(im)

# ---------- cards ----------
def title_card():
    im = canvas(NAVY); d = ImageDraw.Draw(im)
    text(d, (120, 300), "YAM-UMI", font(120, True), WHITE)
    text(d, (120, 450), "A fiducial-tracked glove that collects metrically labelled demonstrations for the YAM,", font(40), LIGHT)
    text(d, (120, 505), "labelled by a fixed camera and a wrist fisheye, or by one head-worn camera", font(40), LIGHT)
    text(d, (120, 620), "Yosub Shin¹,  Yuan Liu²,  Igor Molybog¹", font(40, True), WHITE)
    text(d, (120, 680), "¹University of Hawai'i at Mānoa    ²Independent", font(34), LIGHT)
    text(d, (120, 860), "The UMI Arena  ·  CoRL 2026  ·  device track", font(34), AMBER)
    return to_bgr(im)

def bridge_card():
    im = canvas(NAVY); d = ImageDraw.Draw(im)
    text(d, (W // 2, 440), "The same glove, one head-worn camera", font(72, True), WHITE, "mm")
    text(d, (W // 2, 560), "Drop the wrist camera and the fixed tracker: the camera that observes the policy also labels the glove", font(36), LIGHT, "mm")
    text(d, (W // 2, 640), "A Head Camera Is (Almost) All You Need  ·  arXiv:2610.27622  ·  poster in the paper track", font(30), AMBER, "mm")
    return to_bgr(im)

def end_card():
    im = canvas(NAVY)
    qr = cv2.imread(str(S / "qr_crop-1.png"))
    g = (qr.min(axis=2) > 240); rows = np.where(g.mean(axis=1) > 0.5)[0]; cols = np.where(g.mean(axis=0) > 0.5)[0]
    qr_head = qr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    qr_yam = cv2.imread(str(S / "qr_yamumi.png")); qr_yam = cv2.copyMakeBorder(qr_yam, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    qr_yam[(qr_yam.min(axis=2) < 128)] = rgb2bgr(NAVY)  # same ink as the poster's QR
    bgr = to_bgr(im)
    for x, q in ((1040, qr_yam), (1480, qr_head)):
        q = cv2.resize(q, (400, 400), interpolation=cv2.INTER_NEAREST); bgr[300:700, x:x + 400] = q
    im = to_pil(bgr); d = ImageDraw.Draw(im)
    text(d, (1240, 740), "yam-umi", font(34, True), WHITE, "mm"); text(d, (1240, 785), "glove, BOM, assembly", font(28), LIGHT, "mm")
    text(d, (1680, 740), "head-cam-umi", font(34, True), WHITE, "mm"); text(d, (1680, 785), "head camera, labels, policies", font(28), LIGHT, "mm")
    text(d, (120, 300), "Everything is open", font(64, True), WHITE)
    text(d, (120, 400), "github.com/YosubShin/yam-umi", font(40), AMBER)
    text(d, (120, 455), "github.com/YosubShin/head-cam-umi", font(40), AMBER)
    text(d, (120, 510), "arXiv:2610.27622", font(40), AMBER)
    for i, s in enumerate(["One printed glove, the YAM gripper's own fingers and rails", "$79 with the wrist fisheye, $22 without; 355 g / 370 g",
                           "Every label absolute, from printed fiducials, no IMU, no SLAM", "Hardware and tools: MIT licence"]):
        text(d, (120, 620 + 58 * i), "•  " + s, font(32), LIGHT)
    text(d, (120, 900), "The UMI Arena  ·  CoRL 2026  ·  device track", font(34), DIM)
    return to_bgr(im)

# ---------- 1 hardware ----------
def seg_hardware(secs=None):
    secs = secs or (5 if SHORT else 16)
    left = [fit(cv2.imread(str(YAM / "glove-on-hand.jpg")), 608, 1080), fit(cv2.imread(str(YAM / "umi-vs-yam-gripper.jpg")), 608, 1080)]
    thumbs = [(fit(cv2.imread(str(YAM / f)), 370, 278), cap) for f, cap in
              (("assembly/jaws-open.jpg", "rack-and-pinion jaws on MGN9 rails"), ("assembly/pinion-installed.jpg", "one two-plate print, 7 bought parts"), ("assembly/camera-mount-and-arc.jpg", "11-face ball, cubes, fisheye mount"))]
    base = canvas(PAPER); d = ImageDraw.Draw(base); x0 = 760
    text(d, (x0, 150), "The glove", font(56, True), NAVY)
    specs = [("Ergonomics", "355 g wrist-camera / 370 g head-camera version; UMI 780 g"),
             ("Dexterity", "the YAM gripper's own fingers and rail-and-pinion: 0–86 mm opening (narrower throw on the spring variant)"),
             ("Strength", "2 kg by a jug handle; no rail wear in a month"),
             ("Labels", "absolute 6-DoF pose and jaw width every frame from printed fiducials; squeeze force on the spring variant"),
             ("Cost", "$79 with the wrist fisheye, $22 without; $122 of shared rig")]
    y = 580
    for h, b in specs:
        text(d, (x0, y), h, font(30, True), NAVY); ls = wrap(d, b, font(30), 900)
        for k, line in enumerate(ls): text(d, (x0 + 230, y + 36 * k), line, font(30), MUTED)
        y += 36 * len(ls) + 22
    base = to_bgr(base)
    for k, (th, cap) in enumerate(thumbs):
        x = x0 + k * 385; base[240:518, x:x + 370] = th
    im = to_pil(base); d = ImageDraw.Draw(im)
    for k, (th, cap) in enumerate(thumbs): text(d, (x0 + k * 385, 526), cap, font(22), MUTED)
    base = to_bgr(im); first = None
    for j in range(secs * FPS):
        t = j / FPS; out = base.copy(); pic = left[0] if t < secs / 2 else left[1]
        if abs(t - secs / 2) < 0.4:
            a = (t - (secs / 2 - 0.4)) / 0.8; pic = cv2.addWeighted(left[0], 1 - a, left[1], a, 0)
        out[:, 80:688] = pic
        out = header_bar(out, "1", "YAM-UMI: a printed glove that replicates the YAM gripper")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 2 original labelling (from the YAM-UMI video) ----------
def seg_yam():
    cap = cv2.VideoCapture(str(S / "seg_yam.mp4")); base = canvas(PAPER); d = ImageDraw.Draw(base)
    text(d, (W // 2, 1064), "From the YAM-UMI video: a fixed 4K camera over the table reads the ball, the wrist fisheye reads the jaw tags", font(26), MUTED, "mm")
    base = to_bgr(base); first = None; i = -1
    while True:
        ok, fr = cap.read(); i += 1
        if not ok: break
        if SHORT and not 105 <= i < 240: continue  # 3.5-8.0 s: the end of the demonstration and the labels split screen
        out = base.copy(); out[100:1050, 116:1804] = cv2.resize(fr, (1688, 950), interpolation=cv2.INTER_AREA)
        out = header_bar(out, "2", "Labelled by a fixed camera and the wrist fisheye: no IMU, no SLAM")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 3 collection with the head camera ----------
def seg_collection():
    cap = cv2.VideoCapture(str(S / "seg_coll.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = canvas(PAPER); d = ImageDraw.Draw(base); x0 = 760
    text(d, (x0, 180), "Collecting demonstrations", font(56, True), NAVY)
    bullets = [("The operator wears the head camera", "a $122 4K USB camera on a $10 strap, 103 g"),
               ("The glove is printed and electronics-free", "$22, 370 g: no wrist camera, IMU or encoder (the wrist camera, its cable and the fixed camera seen here are for the ablations only)"),
               ("No protocol to follow", "what the operator sees is what the policy will see"),
               ("2–3× the demonstrations per minute of teleoperation", "6–8 s per demonstration, continuous recording")]
    y = 300
    for h, b in bullets:
        d.rectangle([x0, y + 14, x0 + 18, y + 32], fill=TEAL); text(d, (x0 + 40, y), h, font(38, True), NAVY)
        ls = wrap(d, b, font(32), 1080)
        for k, line in enumerate(ls): text(d, (x0 + 40, y + 50 + 38 * k), line, font(32), MUTED)
        y += 102 + 38 * len(ls)
    text(d, (x0, 960), "Phone view, 2.4\u00d7 speed" if SHORT else "Phone view, 1.6\u00d7 speed", font(28), MUTED)
    base = to_bgr(base); first = None
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        if SHORT and (i >= 337 or i % 3 == 2): continue  # the first two demonstrations, 2.4x
        out = base.copy(); out[:, 80:688] = fr
        out = header_bar(out, "3", "Collect with the glove and a head camera")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 4 the head camera's frame with the label sources ----------
def seg_labels():
    a = cv2.aruco; p = a.DetectorParameters(); p.cornerRefinementMethod = a.CORNER_REFINE_SUBPIX
    d4 = a.ArucoDetector(a.getPredefinedDictionary(a.DICT_4X4_50), p); d36 = a.ArucoDetector(a.getPredefinedDictionary(a.DICT_APRILTAG_36h11), p)
    cap = cv2.VideoCapture(str(S / "seg_head_raw.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); first = None
    SKIP = (int((176.8 - 169) * 25), int((182.6 - 169) * 25))  # the reset between episodes 38 (170.3-176.5 s) and 39 (182.9-189.2 s)
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        if SHORT and i < SKIP[1]: continue  # episode 39 only
        if SKIP[0] <= i < SKIP[1]:
            if i == SKIP[0]: emit(black, 6)
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); c4, i4, _ = d4.detectMarkers(g); c36, i36, _ = d36.detectMarkers(g)
        nt = 0 if i36 is None else len(i36); nb = nta = 0
        def poly(cs, col, w=4):
            for c in cs: cv2.polylines(fr, [c.reshape(-1, 1, 2).astype(np.int32)], True, rgb2bgr(col), w, cv2.LINE_AA)
        if i36 is not None: poly(c36, TEAL)
        if i4 is not None:
            ids = i4.ravel(); ball = [c for c, k in zip(c4, ids) if 25 <= k <= 35]; tails = [c for c, k in zip(c4, ids) if 36 <= k <= 49]
            nb, nta = len(ball), len(tails); poly(ball, AMBER); poly(tails, MAG)
        ov = fr.copy(); cv2.rectangle(ov, (40, 760), (900, 1040), (255, 255, 255), -1); fr = cv2.addWeighted(ov, 0.82, fr, 0.18, 0)
        im = to_pil(fr); d = ImageDraw.Draw(im)
        for k, (col, src, lab, cnt) in enumerate([(TEAL, "paper tiles", "camera pose", nt), (AMBER, "ball faces", "gripper pose", nb), (MAG, "tails on the jaws", "jaw width", nta)]):
            y = 800 + 80 * k
            d.rectangle([70, y + 6, 106, y + 42], outline=col, width=5)
            text(d, (130, y + 24), src, font(36), INK, "lm"); text(d, (470, y + 24), "→", font(36), MUTED, "lm")
            text(d, (530, y + 24), lab, font(36, True), NAVY, "lm"); text(d, (870, y + 24), f"{cnt}", font(36), MUTED, "rm")
        out = header_bar(to_bgr(im), "4", "One head-camera frame carries the observation and all three labels")
        im = to_pil(out); d = ImageDraw.Draw(im); text(d, (W - 48, 48), "Head camera, 1× speed", font(28), LIGHT, "rm"); out = to_bgr(im)
        if first is None: first = out
        emit(out, 2 if i % 5 == 4 else 1)  # 25 -> 30 fps
    return first, out

# ---------- 5 deployment ----------
def seg_deploy():
    cap = cv2.VideoCapture(str(REPO / "media/head-rollout-clip.mp4")); sfps = cap.get(cv2.CAP_PROP_FPS)
    base = canvas(PAPER); d = ImageDraw.Draw(base)
    for k, s in enumerate(["The same camera rides a $122 SO-101 arm beside the robot. It follows the gripper from the robot's encoders alone,",
                           "placed by a linear model of where the collector's head was relative to the hand. No wrist camera."]):
        text(d, (W // 2, 905 + 44 * k), s, font(30), MUTED, "mm")
    text(d, (W // 2, 1000), "Phone view and the positioner camera, one rollout of a head-camera policy" + (", 3\u00d7 speed" if SHORT else ""), font(26), DIM, "mm")
    base = to_bgr(base); first = None; frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    frames = frames[int(2.0 * sfps):]  # the arm starts moving 2 s in
    total = int(round(len(frames) / sfps * FPS))
    for j in range(0, total, 3 if SHORT else 1):
        fr = frames[min(len(frames) - 1, int(j / FPS * sfps))]
        fr = cv2.resize(fr, (1920, int(round(540 * 1920 / 1364))), interpolation=cv2.INTER_AREA)
        h = fr.shape[0]; y0 = 110 + (880 - 110 - h) // 2
        out = base.copy(); out[y0:y0 + h] = fr
        out = header_bar(out, "5", "Deploy: the head camera moves to a positioner beside the robot")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 6 one placement, three policies ----------
def seg_compare():
    R = {(r["policy"], r["scene"]): r for r in map(json.loads, open(G / "trials.jsonl"))}
    cols = [("mix_gHs3", "b0587b", "Head camera on the positioner", "success"), ("mix_gHA", "fish", "Wrist fisheye, head-camera labels", "dropped during the carry"), ("mix_FB", "b0587", "Fixed forward camera", "no grasp")]
    SPEED = 2.0 if SHORT else 1.5; vids = []
    def grab(cap):
        fs = []
        while True:
            ok, f = cap.read()
            if not ok: return fs
            fs.append(f)
    for pol, cam, name, outc in cols:
        own = grab(cv2.VideoCapture(glob.glob(str(G / f"{pol}_S23r0_*_{cam}.mp4"))[0])); wld = grab(cv2.VideoCapture(glob.glob(str(G / f"{pol}_S23r0_*_world.mp4"))[0]))
        vids.append((own, wld, R[(pol, 23)]["elapsed_s"], name, outc))
    tmax = max(v[2] for v in vids); TCAP = 16.0 if SHORT else 24.0  # the fixed camera never grasps; stop 8 s after the other two end
    total = int(round(min(tmax, TCAP) / SPEED * FPS)) + int(FPS * (0.8 if SHORT else 1.5)); CW = 640; first = None
    for j in range(total):
        t = j / FPS * SPEED
        im = canvas(PAPER); d = ImageDraw.Draw(im)
        text(d, (W // 2, 140), "Placement 23: each policy's own camera (top) and a fixed third-person camera (bottom), one clock", font(30), MUTED, "mm")
        text(d, (W - 48, 48), f"t = {min(t, TCAP, tmax):4.1f} s   ({SPEED:g}\u00d7)", font(30), LIGHT, "rm")
        out = to_bgr(im)
        for c, (own, wld, el, name, outc) in enumerate(vids):
            x0 = c * CW; done = t >= el or t >= TCAP; tt = min(t, el - 1e-3, TCAP)
            fo = own[min(len(own) - 1, int(tt / el * len(own)))]; fw = wld[min(len(wld) - 1, int(tt / el * len(wld)))]
            ho, wo = fo.shape[:2]; s = min(CW / wo, 360 / ho); fo = cv2.resize(fo, (int(wo * s), int(ho * s)), interpolation=cv2.INTER_AREA)
            tile = np.zeros((360, CW, 3), np.uint8); ox = (CW - fo.shape[1]) // 2; tile[:fo.shape[0], ox:ox + fo.shape[1]] = fo
            fw = cv2.resize(fw, (CW, 360), interpolation=cv2.INTER_AREA)
            if done: tile = (tile * 0.45).astype(np.uint8); fw = (fw * 0.45).astype(np.uint8)
            out[250:610, x0:x0 + CW] = tile; out[614:974, x0:x0 + CW] = fw
        im = to_pil(out); d = ImageDraw.Draw(im)
        for c, (own, wld, el, name, outc) in enumerate(vids):
            x0 = c * CW; d.rectangle([x0 + 4, 180, x0 + CW - 4, 244], fill=NAVY); text(d, (x0 + CW // 2, 212), name, font(28, True), WHITE, "mm")
            if t >= el or t >= TCAP:
                col = TEAL if outc == "success" else AMBER
                d.rectangle([x0 + 4, 980, x0 + CW - 4, 1040], fill=col); text(d, (x0 + CW // 2, 1010), f"{outc}  \u00b7  {el:.0f} s" + ("  (cut)" if el > TCAP else ""), font(30, True), WHITE, "mm")
        out = header_bar(to_bgr(im), "6", "One placement, seen by each policy")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 7 next: the grip-force indicator, read from the same frame ----------
def spring_available():
    return all((REPO / p).exists() for p in (f"calib/{SPRING_STEM}_dodeca_detections.npz", "calib/glove_yosub_spring_slits.json",
                                             "calib/glove_yosub_spring_dodeca_bundle.json", "calib/glove_yosub_spring_gripper_bundle_headcam.json",
                                             f"data/raw/{SPRING_STEM}.mkv", "scripts/squeeze_read.py"))

def seg_jug():
    """Lead-in from the phone: the spring glove lifts a gallon jug by its handle (second lift of the take, 1x), with the
    slit row cropped from the same clip at rest and under the load. Returns the first frame, or None without the clip."""
    if not (S / "seg_jug.mp4").exists(): return None
    cap = cv2.VideoCapture(str(S / "seg_jug.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = canvas(PAPER); d = ImageDraw.Draw(base); x0 = 760
    text(d, (x0, 150), "Lifting a gallon jug by the handle", font(52, True), NAVY)
    text(d, (x0, 225), "The spring in the handle compresses under the load and uncovers the slits", font(30), MUTED)
    rest = cv2.imread(str(S / "jug_rest.png"))[550:850, 330:530]; up = cv2.imread(str(S / "jug_up.png"))[390:690, 400:600]
    for k, (im, lab) in enumerate(((rest, "jaws closed, jug on the table"), (up, "jug in the air"))):
        cx = x0 + 500 * k
        base.paste(to_pil(cv2.resize(im, (440, 660), interpolation=cv2.INTER_CUBIC)), (cx, 310))
        d.rectangle([cx, 310, cx + 440, 970], outline=TEAL if k else LIGHT, width=4)
        text(d, (cx + 220, 1000), lab, font(28, True if k else False), NAVY if k else MUTED, "mm")
    text(d, (x0, 1046), "Phone view, 1× speed; the two stills are from the same clip, 2.2×", font(24), MUTED)
    base = to_bgr(base); first = None
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        if SHORT and i >= 141: break  # 4.7 s: the lift and the set-down
        out = base.copy(); out[:, 80:688] = fr
        out = header_bar(out, "7", "Next: grip force, still without electronics")
        if first is None: first = out
        emit(out)
    emit(out, 9 if SHORT else FPS // 2); fade(out, black, 0.3)
    return first


def seg_spring():
    """yam_glove9 frames 2300-2440 (92.0-97.6 s, one block grasp) at 1x, then 2340-2400 at 0.4x. The slit row is
    located per frame with scripts/squeeze_read.py (the reader's own geometry) and shown zoomed beside the full frame."""
    sys.path.insert(0, str(REPO / "scripts")); import squeeze_read as sq
    a = argparse.Namespace(stem=SPRING_STEM, raw_dir=str(REPO / "data/raw"), det_dir=str(REPO / "calib"), intrinsics=str(REPO / "calib/B0587b_f3_intrinsics.json"),
                           bundle=str(REPO / "calib/glove_yosub_spring_dodeca_bundle.json"), gripper_bundle=str(REPO / "calib/glove_yosub_spring_gripper_bundle_headcam.json"),
                           slits=str(REPO / "calib/glove_yosub_spring_slits.json"), episodes=str(REPO / f"data/raw/yam_glove9_glove_episodes.json"), track_hold=0.5)
    jug_first = seg_jug()
    Ss = sq.Session(a); row = Ss.row; poses = Ss.poses(); jaw = Ss.jaw_series(poses); sp = Ss.cfg["spring"]
    F0, F1, SLOW0, SLOW1 = 2300, 2440, 2340, 2400
    Wc, Hc = 320, 150; frames = {}; c_s = None
    for f in range(F0, F1 + 1):
        img = Ss.frame(f); bp = poses.get(f); width = np.nan
        if bp is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); j = jaw[f] if np.isfinite(jaw[f]) else 0.0; off = row.offset(j)
            rd = Ss.read(gray, bp, off)
            pc = cv2.projectPoints(row.corners.reshape(-1, 3) + row.AC * rd["shift"] + off + row.dir * (rd["along_mm"] / 1000), bp["rvec"], bp["tvec"], Ss.K, Ss.dist)[0].reshape(row.N, 4, 2)
            c = pc.reshape(-1, 2).mean(0); c_s = c if c_s is None else 0.7 * c_s + 0.3 * c
            if rd["valid"]: width = float(rd["width_mm"])
        if c_s is None: continue
        x0 = int(min(max(c_s[0] - Wc / 2, 0), img.shape[1] - Wc)); y0 = int(min(max(c_s[1] - Hc / 2, 0), img.shape[0] - Hc))
        zoom = cv2.resize(img[y0:y0 + Hc, x0:x0 + Wc], (1180, 553), interpolation=cv2.INTER_CUBIC)
        small = cv2.resize(img, (600, 338), interpolation=cv2.INTER_AREA)
        cv2.rectangle(small, (int(x0 * 600 / img.shape[1]), int(y0 * 338 / img.shape[0])), (int((x0 + Wc) * 600 / img.shape[1]), int((y0 + Hc) * 338 / img.shape[0])), rgb2bgr(AMBER), 2)
        frames[f] = (zoom, small, width)
    base = canvas(PAPER); d = ImageDraw.Draw(base); x0 = 1260
    bullets = [("A die spring in the handle, 2 mm of travel", "a shutter behind eight slits uncovers as it compresses"),
               ("The bar width is the squeeze", "0.5 kgf when white appears, 2.1 kgf at the stop; no electronics"),
               ("Read from the same head-camera frame", "as the pose and the jaw width: a force label for glove data, which teleop data carries as position overshoot")]
    y = 520
    for h, b in bullets:
        d.rectangle([x0, y + 12, x0 + 16, y + 28], fill=TEAL); text(d, (x0 + 36, y), h, font(30, True), NAVY)
        for k, line in enumerate(wrap(d, b, font(26), 580)): text(d, (x0 + 36, y + 42 + 32 * k), line, font(26), MUTED)
        y += 150
    text(d, (60, 1040), "Spring glove, head camera; the zoom follows the slit row beside the back-marker cube", font(24), MUTED)
    base = to_bgr(base); first = jug_first; last_w = 0.0
    def render(f, tag):
        nonlocal last_w, first
        zoom, small, width = frames[f]
        if np.isfinite(width): last_w = width
        out = base.copy(); out[130:683, 60:1240] = zoom; out[130:468, 1260:1860] = small
        im = to_pil(out); d = ImageDraw.Draw(im)
        shown = min(last_w, row.slit_mm); s = shown / row.slit_mm; kgf = sp["onset_kgf"] + sp["k_kgf_per_mm"] * shown; at_stop = last_w >= row.slit_mm - 0.05
        d.rectangle([60, 720, 1240, 790], fill=(0xEE, 0xEB, 0xE4)); d.rectangle([60, 720, 60 + int(1180 * s), 790], fill=AMBER if s > 0.5 else TEAL)
        text(d, (80, 755), f"slit bar {shown:.1f} of 2.0 mm" + ("  (at the stop)" if at_stop else ""), font(34, True), WHITE if s > 0.15 else INK, "lm")
        text(d, (1220, 755), (("\u2265 " if at_stop else "\u2248 ") + f"{kgf:.1f} kgf") if last_w > 0.05 else "at rest, below 0.5 kgf", font(34, True), WHITE if s > 0.9 else INK, "rm")
        text(d, (1240, 830), tag, font(28), MUTED, "rm")
        out = header_bar(to_bgr(im), "7", "Next: grip force, still without electronics")
        if first is None: first = out
        return out
    out = None
    for i, f in enumerate(range(F0, F1 + 1)):
        if f in frames: out = render(f, "1× speed"); emit(out, 2 if i % 5 == 4 else 1)
    for f in (() if SHORT else range(SLOW0, SLOW1 + 1)):
        if f in frames: out = render(f, "0.4× speed"); emit(out, 3)
    emit(out, FPS // 2 if SHORT else FPS)
    return first, out

# ---------- assemble ----------
tc = title_card(); emit(tc, int(FPS * (2.5 if SHORT else 4))); prev = tc
segs = [seg_hardware, seg_yam, "bridge", seg_collection, seg_labels, seg_deploy, seg_compare] + ([seg_spring] if spring_available() else [])
for seg in segs:
    fade(prev, black, 0.3)
    if seg == "bridge":
        bc = bridge_card(); emit(bc, int((1.5 if SHORT else 3.5) * FPS)); prev = bc; continue
    first, last = seg(); prev = last
fade(prev, black, 0.3)
ec = end_card(); emit(ec, int(FPS * (3 if SHORT else 6))); fade(ec, black, 0.5)
ff.stdin.close(); ff.wait()
print(OUT, nframes, f"{nframes / FPS:.1f} s", "spring segment:", spring_available())
