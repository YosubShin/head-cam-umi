#!/usr/bin/env python3
"""Demo-table loop for the UMI Arena device track (YAM-UMI, with head-cam-umi as its current tracking mode).
1920x1080, 30 fps, no audio.  python media/make_demo_loop.py -> media/demo-loop.mp4

Inputs outside this repo: ../yam-umi/media (stills, the YAM-UMI paper video, the phone originals), the yam_glove6
head stream (data/raw/nas, episodes 38 and 39 by frame index; --short also reads the HHs3 dataset's width label for them) and, for the last segment, the spring glove's yam_glove9 head stream with its detection
cache and calib files (skipped when absent). Segments: title; the glove hardware; the original labelling (fixed
camera + wrist fisheye, from the YAM-UMI video); bridge card; two glove demonstrations seen by the fixed camera and the
head camera at once, with the label sources drawn, the poses solved live, the width label and a 3D view; deployment (phone + positioner camera); one placement
seen by three policies (1.5x); next step, the passive grip-force indicator (the jug lift from the phone with the context, then one block grasp: head frame, slit zoom, live graph); end card with
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
SHORT = "--short" in sys.argv  # the 60 s version for a talk: same layout and segments, shorter cuts, faster playback
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
EP38, EP39 = (4474, 4636), (4805, 4970)  # glove6 episodes 38 and 39 as head-camera frame indices (marks against _ts.json; the stream ran at 26.3 fps, so the mkv's nominal 25 fps clock drifts)
if not (S / "seg_head_raw.mp4").exists():  # the two episodes back to back, 5 frames of margin each
    sel = f"between(n,{EP38[0] - 5},{EP38[1] + 5})+between(n,{EP39[0] - 5},{EP39[1] + 5})"
    sh("ffmpeg", "-v", "error", "-y", "-i", str(HEAD), "-vf", f"select='{sel}',setpts=N/25/TB,scale=1920:1080", "-r", "25", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(S / "seg_head_raw.mp4"))
EP2 = EP38[1] - EP38[0] + 11  # first frame of episode 39 in the cut
FX38, FX39 = (4472, 4644), (4803, 4978)  # the same two spans in the fixed camera (yam_glove6_b0587), by wall clock, margins included
FIXED = REPO / "data/raw/nas/yam_glove6_b0587.mkv"
if not (S / "seg_fixed_raw.mp4").exists():  # the fixed camera over the table, frame for frame with the head cut
    sel = f"between(n,{FX38[0]},{FX38[1]})+between(n,{FX39[0]},{FX39[1]})"
    sh("ffmpeg", "-v", "error", "-y", "-i", str(FIXED), "-vf", f"select='{sel}',setpts=N/25/TB,scale=1300:732", "-r", "25", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(S / "seg_fixed_raw.mp4"))
if not (S / "seg_yam.mp4").exists():  # YAM-UMI paper video 23.5-34 s: one demonstration (2x) then the labels split screen
    sh("ffmpeg", "-v", "error", "-y", "-ss", "23.5", "-t", "10.5", "-i", str(YAM / "yam-umi-paper-video.mp4"), "-vf", "scale=1748:984", "-r", "30", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(S / "seg_yam.mp4"))
if JUG.exists() and not (S / "seg_jug.mp4").exists():  # second lift of the take, 7.8-14.0 s, portrait 608x1080
    sh("ffmpeg", "-v", "error", "-y", "-ss", "7.8", "-t", "6.2", "-i", str(JUG), "-vf", "scale=608:1080", "-r", "30", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(S / "seg_jug.mp4"))
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
    for _ in range(n):  # in the talk layout every written frame is composed, so a side panel video can advance per frame
        ff.stdin.write(np.ascontiguousarray(compose(bgr)).tobytes()); nframes += 1
def fade(a, b, secs=0.4):
    n = int(secs * FPS)
    for i in range(1, n + 1):
        t = i / (n + 1); emit(cv2.addWeighted(a, 1 - t, b, t, 0))

def header_bar(im_bgr, step, title):
    return im_bgr  # no header bars: the panel names the section
    im = to_pil(im_bgr); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 96], fill=NAVY)
    text(d, (48, 48), "D10", font(30, True), AMBER, "lm"); text(d, (150, 48), step, font(40, True), AMBER, "lm"); text(d, (240, 48), title, font(40, True), WHITE, "lm")
    return to_bgr(im)


# ---------- talk layout (--short): a summary panel that stays, the media cycling beside it ----------
PANEL_W = 580; MW = W - PANEL_W
MEDIA = None; ACTIVE = 0; CAPTION = ""; CAP_X = 40; PENDING = False; LAST = None; PANELS = {}
BULLETS = [("The glove", "The YAM gripper's own fingers on rack-and-pinion rails; $22–79, 355–370 g, lifts 2 kg"),
           ("Labels from printed markers", "Absolute 6-DoF pose from the ball and jaw width from the tails, every frame; no IMU, no SLAM"),
           ("Or one head-worn camera", "The operator's 4K camera is both the policy's observation and the label source"),
           ("Deploy on a $122 positioner", "The same camera on an SO-101 beside the robot, placed from the encoders alone"),
           ("Rollouts, 25 placements each", "Head camera 20/25, wrist fisheye 14/25, fixed camera 12/25; one placement shown"),
           ("Next: grip force, no electronics", "A spring and slits in the handle; the bar width is the squeeze, in the same frame")]
def panel(active):
    if active in PANELS: return PANELS[active]
    im = canvas(PAPER); d = ImageDraw.Draw(im); px = MW; tx = px + 36; tw = W - tx - 36
    d.rectangle([px, 0, W, H], fill=NAVY)
    text(d, (tx, 40), "YAM-UMI", font(60, True), WHITE); text(d, (W - 36, 72), "D10", font(40, True), AMBER, "rm"); y = 120
    for line in wrap(d, "A printed glove that collects metrically labelled demonstrations for the YAM arm", font(23), tw): text(d, (tx, y), line, font(23), LIGHT); y += 30
    y += 18
    for k, (h, b) in enumerate(BULLETS):
        on = k == active; hb = wrap(d, h, font(25, True), tw); bb = wrap(d, b, font(21), tw); hh = 32 * len(hb) + 27 * len(bb) + 4
        if on: d.rectangle([px + 14, y - 8, W - 14, y + hh], fill=(0x24, 0x4A, 0x70)); d.rectangle([px + 14, y - 8, px + 20, y + hh], fill=AMBER)
        for line in hb: text(d, (tx, y), line, font(25, True), WHITE if on else DIM); y += 32
        for line in bb: text(d, (tx, y), line, font(21), LIGHT if on else (0x6E, 0x7E, 0x92)); y += 27
        y += 22
    qy, qs = 896, 140  # the two repos' QR codes, always on screen
    qr_yam = cv2.copyMakeBorder(cv2.imread(str(S / "qr_yamumi.png")), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    qr = cv2.imread(str(S / "qr_crop-1.png")); g = (qr.min(axis=2) > 240); rows = np.where(g.mean(axis=1) > 0.5)[0]; cols = np.where(g.mean(axis=0) > 0.5)[0]
    qr_head = qr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    bgr = to_bgr(im)
    for k, q in enumerate((qr_yam, qr_head)):
        x = tx + k * (qs + 16); bgr[qy:qy + qs, x:x + qs] = cv2.resize(q, (qs, qs), interpolation=cv2.INTER_AREA)
    im = to_pil(bgr); d = ImageDraw.Draw(im)
    text(d, (tx + qs // 2, 884), "yam-umi", font(16), LIGHT, "mm"); text(d, (tx + qs + 16 + qs // 2, 884), "head-cam-umi", font(16), LIGHT, "mm")
    lx = tx + 2 * qs + 34
    for k, (line, col) in enumerate((("github.com/YosubShin/", AMBER), ("yam-umi, head-cam-umi", AMBER), ("arXiv:2610.27622", DIM), ("D10 · The UMI Arena", DIM), ("CoRL 2026, device track", DIM))):
        text(d, (lx, qy + 4 + 24 * k), line, font(17), col)
    PANELS[active] = to_bgr(im); return PANELS[active]
def set_media(fn, active, caption, cap_x=40):
    global MEDIA, ACTIVE, CAPTION, CAP_X, PENDING; MEDIA, ACTIVE, CAPTION, CAP_X, PENDING = fn, active, caption, cap_x, True
def compose(full):
    global PENDING, LAST, nframes
    out = panel(ACTIVE).copy(); out[:, :MW] = MEDIA(full)
    im = to_pil(out); d = ImageDraw.Draw(im); text(d, (CAP_X, 1050), CAPTION, font(24), MUTED, "lm"); out = to_bgr(im)
    if PENDING and LAST is not None:
        n = 12
        for i in range(1, n): ff.stdin.write(cv2.addWeighted(LAST, 1 - i / n, out, i / n, 0).tobytes()); nframes += 1
    PENDING = False; LAST = out; return out
def media_canvas(): return np.full((H, MW, 3), rgb2bgr(PAPER), np.uint8)
def m_rect(x0, y0, x1, y1):
    def fn(full):
        c = full[y0:y1, x0:x1]; sc = min((MW - 80) / c.shape[1], 1000 / c.shape[0]); w, h = int(c.shape[1] * sc), int(c.shape[0] * sc)
        m = media_canvas(); y = (1010 - h) // 2; x = (MW - w) // 2; m[y:y + h, x:x + w] = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA); return m
    return fn
def m_portrait(full): m = media_canvas(); m[:, 366:974] = full[:, 80:688]; return m
def m_hardware(full):
    m = media_canvas(); m[:, 40:648] = full[:, 80:688]
    for k in range(3): x = 760 + k * 385; m[16 + k * 340:16 + k * 340 + 312, 760:1130] = full[240:552, x:x + 370]
    return m
PH = {"cap": None, "last": None, "counts": (0, 0, 0)}
LAY_SHORT = dict(W=MW, x1=16, x2=680, y0=110, vw=650, vh=366, cap_y=500, lx=16, ly=545, dy=122, v3=(680, 530, 650, 470), v3cap=(680, 1016), f3=700)
def combo(full, L):
    """The fixed camera over the table (left) and the head camera's labelled frame (right), the same moment of the same
    session, with the live camera and gripper pose in the tile world (cm, degrees), the dataset's width label and the 3D view."""
    m = np.full((H, L["W"], 3), rgb2bgr(PAPER), np.uint8); blackout = full.max() == 0; y0, vw, vh = L["y0"], L["vw"], L["vh"]
    m[y0:y0 + vh, L["x1"]:L["x1"] + vw] = 0 if blackout or PH["last"] is None else cv2.resize(PH["last"], (vw, vh), interpolation=cv2.INTER_AREA)
    m[y0:y0 + vh, L["x2"]:L["x2"] + vw] = cv2.resize(full, (vw, vh), interpolation=cv2.INTER_AREA)
    lv = PH.get("live") or {}; g = PH.get("g"); Bw = None
    if lv.get("cam") is not None and lv.get("body") is not None:
        T_wc = np.linalg.inv(lv["cam"]); Bw = T_wc[:3, :3] @ lv["body"][1] + T_wc[:3, 3]
    vx, vy, vw3, vh3 = L["v3"]
    m[vy:vy + vh3, vx:vx + vw3] = PH["V3"].render(lv.get("cam"), Bw, lv.get("R_world_body"), lv.get("tool_world"), g, PH["trail_cam"], PH["trail_tool"])
    im = to_pil(m); d = ImageDraw.Draw(im)
    text(d, L["v3cap"], "Tile world, 10 cm grid: camera (teal), gripper (amber), 3 s trails", font(20), MUTED, "lm")
    text(d, (L["x1"], L["cap_y"]), "Fixed camera over the table: the same moment, for reference", font(22), MUTED, "lm")
    text(d, (L["x2"], L["cap_y"]), "Head camera: what the policy sees, with the label sources drawn", font(22), MUTED, "lm")
    cam = lv.get("cam"); tw = lv.get("tool_world"); Rb = lv.get("R_world_body")
    if cam is not None:
        pc = np.linalg.inv(cam)[:3, 3] * 100; cam_s = f"x {pc[0]:+.1f}   y {pc[1]:+.1f}   z {pc[2]:+.1f} cm      {lv['tiles']} tiles"
    else: cam_s = "—"
    if tw is not None:
        import live_labels as LV; e = LV.euler_deg(Rb); tw = tw * 100
        grip_s = f"x {tw[0]:+.1f}   y {tw[1]:+.1f}   z {tw[2]:+.1f} cm      rot {e[0]:+.0f}° {e[1]:+.0f}° {e[2]:+.0f}°"
    else: grip_s = "—"
    for k, (col, src, lab, val) in enumerate(((TEAL, "paper tiles", "camera pose", cam_s), (AMBER, "ball faces", "gripper pose", grip_s), (MAG, "tails on the jaws", "jaw width", None))):
        y = L["ly"] + L["dy"] * k; x = L["lx"]
        d.rectangle([x, y, x + 28, y + 28], outline=col, width=5); text(d, (x + 44, y + 14), src, font(24), INK, "lm"); text(d, (x + 255, y + 14), "→", font(24), MUTED, "lm")
        text(d, (x + 295, y + 14), lab, font(24, True), NAVY, "lm")
        if val is not None: text(d, (x, y + 60), val, font(26, True), NAVY, "lm")
        elif g is not None:
            text(d, (x, y + 60), f"label {g:.2f}", font(26, True), NAVY, "lm"); f = min(max((g - 0.2) / 0.8, 0.0), 1.0)
            d.rectangle([x + 190, y + 48, x + 590, y + 72], fill=(0xEE, 0xEB, 0xE4)); d.rectangle([x + 190, y + 48, x + 190 + int(400 * f), y + 72], fill=MAG)
            text(d, (x + 190, y + 92), "closed", font(18), MUTED, "lm"); text(d, (x + 590, y + 92), "open", font(18), MUTED, "rm")
    return to_bgr(im)
def m_combo(full): return combo(full, LAY_SHORT)
def m_spring(full):
    """The spring segment lays itself out within the media width in the talk version."""
    return full[:, :MW].copy()

# ---------- 1 hardware ----------
def seg_hardware(secs=None):
    secs = secs or (7 if SHORT else 16)
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
    text(d, (W // 2, 1064), "From the YAM-UMI video; the fisheye (right) is kept for the ablations only. Pose from the ball, width from the jaw tails, both in one camera", font(26), MUTED, "mm")
    base = to_bgr(base); first = None; i = -1
    while True:
        ok, fr = cap.read(); i += 1
        if not ok: break
        if SHORT and not 124 <= i < 258: continue  # 4.1-8.6 s: the labels split screen only
        out = base.copy(); out[100:1050, 116:1804] = cv2.resize(fr, (1688, 950), interpolation=cv2.INTER_AREA)
        out[962:1050, 116:1804] = 16  # cover the video's own banner, which credits the width to the fisheye
        im = to_pil(out); d = ImageDraw.Draw(im)
        text(d, (960, 1006), "Ball \u2192 pose, jaw tails \u2192 width, one camera; the wrist fisheye is for the ablations only", font(28, True), WHITE, "mm"); out = to_bgr(im)
        out = header_bar(out, "2", "Labelled by one fixed camera: ball for pose, jaw tails for width; no IMU, no SLAM")
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
    text(d, (x0, 960), "Phone view, 1.6\u00d7 speed", font(28), MUTED)
    base = to_bgr(base); first = None
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        if SHORT and i >= 337: continue  # the first two demonstrations
        out = base.copy(); out[:, 80:688] = fr
        out = header_bar(out, "3", "Collect with the glove and a head camera")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 4 the head camera's frame with the label sources ----------
def seg_labels():
    """Episodes 38 and 39 of glove6 (39 alone in the talk version): the fixed camera and the head camera's labelled frame
    side by side, the pipeline's solvers read live (camera and gripper pose in the tile world), the dataset's width label
    for the frame, and the 3D view (media/live_labels.py)."""
    a = cv2.aruco; p = a.DetectorParameters(); p.cornerRefinementMethod = a.CORNER_REFINE_SUBPIX
    d4 = a.ArucoDetector(a.getPredefinedDictionary(a.DICT_4X4_50), p); d36 = a.ArucoDetector(a.getPredefinedDictionary(a.DICT_APRILTAG_36h11), p)
    cap = cv2.VideoCapture(str(S / "seg_head_raw.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); first = None; L = LAY_SHORT
    PH["cap"] = cv2.VideoCapture(str(S / "seg_fixed_raw.mp4"))
    sys.path.insert(0, str(REPO / "media")); import live_labels as LV
    PH["LL"] = LV.LiveLabels(); PH["V3"] = LV.View3D(PH["LL"].layout, size=L["v3"][2:], f_px=L["f3"]); PH["trail_cam"] = []; PH["trail_tool"] = []
    PH["grip"] = LV.dataset_gripper_for_cut(str(HEAD).replace(".mkv", "_ts.json"), str(HEAD.parent / "yam_glove6_glove_episodes.json"),
                                            [(EP38[0] - 5, EP38[1] + 5, 38, 72), (EP39[0] - 5, EP39[1] + 5, 39, 73)])  # HHs3 episodes 72 and 73
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        if SHORT and i < EP2: PH["cap"].read(); continue  # the talk version shows episode 39 only (the fixed camera keeps step)
        if i == EP2 and not SHORT: emit(black, 6); PH["trail_cam"].clear(); PH["trail_tool"].clear()  # the reset between the episodes is not in the cut
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); c4, i4, _ = d4.detectMarkers(g); c36, i36, _ = d36.detectMarkers(g)
        PH["live"] = PH["LL"].update(i, c36, i36, c4, i4); PH["g"] = float(PH["grip"][i]); lv = PH["live"]
        if lv["cam"] is not None: PH["trail_cam"].append(np.linalg.inv(lv["cam"])[:3, 3]); del PH["trail_cam"][:-75]
        if lv["tool_world"] is not None: PH["trail_tool"].append(lv["tool_world"]); del PH["trail_tool"][:-75]
        okf, ff_ = PH["cap"].read()
        if okf: PH["last"] = ff_
        nt = 0 if i36 is None else len(i36); nb = nta = 0
        def poly(cs, col, w=8):
            for c in cs: cv2.polylines(fr, [c.reshape(-1, 1, 2).astype(np.int32)], True, rgb2bgr(col), w, cv2.LINE_AA)
        if i36 is not None: poly(c36, TEAL)
        if i4 is not None:
            ids = i4.ravel(); ball = [c for c, k in zip(c4, ids) if 25 <= k <= 35]; tails = [c for c, k in zip(c4, ids) if 36 <= k <= 49]
            nb, nta = len(ball), len(tails); poly(ball, AMBER); poly(tails, MAG)
        PH["counts"] = (nt, nb, nta)
        out = fr
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
        out = header_bar(out, "4", "Deploy: the head camera moves to a positioner beside the robot")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 6 one placement, three policies ----------
def seg_compare():
    R = {(r["policy"], r["scene"]): r for r in map(json.loads, open(G / "trials.jsonl"))}
    cols = [("mix_gHs3", "b0587b", "Head camera on the positioner  \u00b7  20/25", "success"), ("mix_gHA", "fish", "Wrist fisheye  \u00b7  14/25", "dropped during the carry"), ("mix_FB", "b0587", "Fixed camera  \u00b7  12/25", "no grasp")]
    SPEED = 1.5; vids = []
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
        out = header_bar(to_bgr(im), "5", "Rollouts: 20/25, 14/25 and 12/25 over 25 placements; one of them seen by each policy")
        if first is None: first = out
        emit(out)
    return first, out

# ---------- 7 next: the grip-force indicator, read from the same frame ----------
def spring_available():
    return all((REPO / p).exists() for p in (f"calib/{SPRING_STEM}_dodeca_detections.npz", "calib/glove_yosub_spring_slits.json",
                                             "calib/glove_yosub_spring_dodeca_bundle.json", "calib/glove_yosub_spring_gripper_bundle_headcam.json",
                                             f"data/raw/{SPRING_STEM}.mkv", "scripts/squeeze_read.py"))

PARAS = [("Teleoperation stores the commanded width.", "On a block the operator commands 0.246 while the fingers stop at 0.335, so the policy learns to over-command and squeeze."),
         ("A glove can only store the measured width.", "A policy that reproduces it as a command merely touches the block."),
         ("The spring puts the squeeze back.", "Past contact the handle compresses up to 2 mm and a shutter uncovers the slits: the bar width is the force, 0.5 kgf at onset to 2.1 kgf at the stop, read from the same head-camera frame as the pose.")]

def spring_intro():
    """The explanation: the jug lift from the phone at 0.7x (the lift, then the set-down) beside the measured-vs-commanded
    context and the slit row tracked (cv2.TrackerMIL) and shown live. Returns the first frame, or None without the clip."""
    if not (S / "seg_jug.mp4").exists(): return None
    cap = cv2.VideoCapture(str(S / "seg_jug.mp4")); Wl = MW; xr = 680; wr = Wl - xr - 40
    base = canvas(PAPER); d = ImageDraw.Draw(base); tf = 23; hf = 30; lh = int(tf * 1.32)
    text(d, (xr, 120), "Measured ≠ commanded width", font(hf, True), NAVY); yy = 120 + hf + 14
    for h, b in PARAS:
        text(d, (xr, yy), h, font(tf, True), NAVY); yy += lh
        for line in wrap(d, b, font(tf), wr): text(d, (xr, yy), line, font(tf), INK); yy += lh
        yy += 10
    yy += 12; ch = 1000 - 36 - yy; cw = ch * 2 // 3; cx0 = xr + (wr - cw) // 2; cy0 = yy
    d.rectangle([cx0 - 3, cy0 - 3, cx0 + cw + 3, cy0 + ch + 3], outline=TEAL, width=3)
    text(d, (cx0 + cw // 2, cy0 + ch + 20), "the slit row, tracked in the clip", font(21, True), NAVY, "mm")
    base = to_bgr(base); first = None; out = None
    fr = []
    while True:
        ok, f_ = cap.read()
        if not ok: break
        fr.append(f_)
    sc = 608 / 720; bbox = (int((423 - 75) * sc), int((710 - 115) * sc), int(150 * sc), int(230 * sc))  # the slit row at 9.0 s (frame 36; phone frame 720 wide)
    SEGS = [(24, 84), (126, 170)]  # 8.6-10.6 s: approach, grasp, lift (dark to white); 12.0-13.5 s: set-down and release (white to dark)
    cen = {36: (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)}
    for order in (range(35, SEGS[0][0] - 1, -1), range(37, SEGS[-1][1] + 1)):  # backward then forward from the init frame
        tr = cv2.TrackerMIL_create(); tr.init(fr[36], bbox); b = bbox
        for f in order:
            okt, bt = tr.update(fr[f]); b = bt if okt else b; cen[f] = (b[0] + b[2] / 2, b[1] + b[3] / 2)
    c_s = None
    def frame_at(f):
        nonlocal c_s
        c = np.array(cen[f]); c_s = c if c_s is None else 0.6 * c_s + 0.4 * c
        x0_ = int(min(max(c_s[0] - 93, 0), 608 - 186)); y0_ = int(min(max(c_s[1] - 140, 0), 1080 - 279))
        o = base.copy(); o[:, 40:648] = fr[f]; o[cy0:cy0 + ch, cx0:cx0 + cw] = cv2.resize(fr[f][y0_:y0_ + 279, x0_:x0_ + 186], (cw, ch), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(o, (40 + x0_, y0_), (40 + x0_ + 186, y0_ + 279), rgb2bgr(TEAL), 2)
        return header_bar(o, "6", "Next: grip force, still without electronics")
    k = 0
    for a, b_ in SEGS:
        c_s = None; nxt = frame_at(a)
        if out is not None: fade(out, nxt, 0.25)
        for f in range(a, b_ + 1):
            out = frame_at(f); emit(out, 2 if k % 7 in (1, 3, 5) else 1); k += 1  # 0.7x
            if first is None: first = out
    emit(out, FPS // 2); fade(out, black, 0.3)
    return first

def seg_spring():
    """One block grasp of yam_glove9 (frames 2300-2440, 92.0-97.6 s) at 0.5x: the head frame and the slit row zoomed
    (located per frame with scripts/squeeze_read.py, the reader's own geometry), the context on measured vs commanded
    width, and a live graph of the jaw aperture and the slit displacement. Laid out within MW columns in the talk version."""
    sys.path.insert(0, str(REPO / "scripts")); import squeeze_read as sq
    a = argparse.Namespace(stem=SPRING_STEM, raw_dir=str(REPO / "data/raw"), det_dir=str(REPO / "calib"), intrinsics=str(REPO / "calib/B0587b_f3_intrinsics.json"),
                           bundle=str(REPO / "calib/glove_yosub_spring_dodeca_bundle.json"), gripper_bundle=str(REPO / "calib/glove_yosub_spring_gripper_bundle_headcam.json"),
                           slits=str(REPO / "calib/glove_yosub_spring_slits.json"), episodes=str(REPO / f"data/raw/yam_glove9_glove_episodes.json"), track_hold=0.5)
    intro_first = spring_intro()
    Ss = sq.Session(a); row = Ss.row; poses = Ss.poses(); jaw = Ss.jaw_series(poses); sp = Ss.cfg["spring"]
    F0, F1 = 2300, 2440; Wc, Hc = 320, 150; SRC_FPS = 25.0; N = F1 - F0 + 1
    Wl = MW; x0 = 40; wh = int(0.40 * Wl); hh = wh * 9 // 16
    xz = x0 + wh + 30; wz = min(Wl - 40 - xz, 694); hz = wz * Hc // Wc; y0 = 120; y1 = y0 + max(hh, hz) + 34
    frames = {}; c_s = None; slit = {}
    for f in range(F0, F1 + 1):
        img = Ss.frame(f); bp = poses.get(f); width = np.nan
        if bp is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); j = jaw[f] if np.isfinite(jaw[f]) else 0.0; off = row.offset(j)
            rd = Ss.read(gray, bp, off)
            pc = cv2.projectPoints(row.corners.reshape(-1, 3) + row.AC * rd["shift"] + off + row.dir * (rd["along_mm"] / 1000), bp["rvec"], bp["tvec"], Ss.K, Ss.dist)[0].reshape(row.N, 4, 2)
            c = pc.reshape(-1, 2).mean(0); c_s = c if c_s is None else 0.7 * c_s + 0.3 * c
            if rd["valid"]: width = float(rd["width_mm"])
        slit[f] = width
        if c_s is None: continue
        cx = int(min(max(c_s[0] - Wc / 2, 0), img.shape[1] - Wc)); cy = int(min(max(c_s[1] - Hc / 2, 0), img.shape[0] - Hc))
        zoom = cv2.resize(img[cy:cy + Hc, cx:cx + Wc], (wz, hz), interpolation=cv2.INTER_CUBIC)
        small = cv2.resize(img, (wh, hh), interpolation=cv2.INTER_AREA); sx = wh / img.shape[1]; sy = hh / img.shape[0]
        cv2.rectangle(small, (int(cx * sx), int(cy * sy)), (int((cx + Wc) * sx), int((cy + Hc) * sy)), rgb2bgr(AMBER), 2)
        frames[f] = (zoom, small)
    sl = np.zeros(N); last = 0.0  # slit displacement, last valid reading held, clamped to the slit
    for i, f in enumerate(range(F0, F1 + 1)):
        if np.isfinite(slit[f]): last = min(max(slit[f], 0.0), row.slit_mm)
        sl[i] = last
    jw = np.array([jaw[f] for f in range(F0, F1 + 1)], float); contact = next((i for i in range(N) if sl[i] >= 0.1), None)
    # static layer: context text and the graph's axes
    base = canvas(PAPER); d = ImageDraw.Draw(base); tf = 23; hf = 30; lh = int(tf * 1.32)
    text(d, (x0, y1), "The same reading in a pick and place", font(hf, True), NAVY); yy = y1 + hf + 14
    for h, b in (("Approach", "the jaws close on the block and the aperture falls to 0"),
                 ("Contact, then the squeeze", "the slits open: 0.5 kgf when white appears, 2.1 kgf at the stop"),
                 ("A force label for glove data", "where teleoperation data carries the overshoot as a commanded width")):
        text(d, (x0, yy), h, font(tf, True), NAVY); yy += lh
        for line in wrap(d, b, font(tf), wh): text(d, (x0, yy), line, font(tf), INK); yy += lh
        yy += 10
    text(d, (x0, y0 + hh + 6), "Head camera, 0.5× speed", font(20), MUTED); text(d, (xz, y0 + hz + 6), "The slit row, zoomed; the crop follows the row beside the back-marker cube", font(20), MUTED)
    gx0, gx1, gy0, gy1 = xz + 64, xz + wz - 64, y1 + 44, 1000 - 40; jmin, jmax = -4.0, 24.0; T = (N - 1) / SRC_FPS
    def gx(i): return gx0 + (gx1 - gx0) * i / (N - 1)
    def gyj(v): return gy1 - (gy1 - gy0) * (v - jmin) / (jmax - jmin)
    def gys(v): return gy1 - (gy1 - gy0) * v / row.slit_mm
    d.rectangle([gx0, gy0, gx1, gy1], outline=LIGHT, width=2)
    for v in (0, 5, 10, 15, 20):
        y = gyj(v); d.line([gx0, y, gx1, y], fill=(0xE6, 0xE3, 0xDC), width=1); text(d, (gx0 - 10, y), f"{v}", font(18), TEAL, "rm")
    for v in (0, 0.5, 1, 1.5, 2): text(d, (gx1 + 10, gys(v)), f"{v:g}", font(18), AMBER, "lm")
    for sec in range(int(T) + 1): x = gx(sec * SRC_FPS); d.line([x, gy1, x, gy1 + 6], fill=MUTED, width=2); text(d, (x, gy1 + 10), f"{sec} s", font(18), MUTED, "ma")
    text(d, (gx0, y1 + 8), "jaw aperture, mm (0 = at the block)", font(20, True), TEAL); text(d, (gx1, y1 + 8), "slit, mm", font(20, True), AMBER, "ra")
    base = to_bgr(base); first = intro_first
    def render(i):
        nonlocal first
        zoom, small = frames[F0 + i]; out = base.copy(); out[y0:y0 + hh, x0:x0 + wh] = small; out[y0:y0 + hz, xz:xz + wz] = zoom
        im = to_pil(out); d = ImageDraw.Draw(im)
        pts = [(gx(k), gyj(jw[k])) for k in range(i + 1) if np.isfinite(jw[k])]
        if len(pts) > 1: d.line(pts, fill=TEAL, width=3, joint="curve")
        d.line([(gx(k), gys(sl[k])) for k in range(i + 1)], fill=AMBER, width=3, joint="curve")
        x = gx(i); d.line([x, gy0, x, gy1], fill=MUTED, width=2)
        if contact is not None and i >= contact:
            xc = gx(contact)
            for yd in range(int(gy0), int(gy1), 12): d.line([xc, yd, xc, yd + 6], fill=INK, width=2)
            text(d, (xc + 8, gy0 + 6), "contact", font(20, True), INK)
        shown = sl[i]; at_stop = shown >= row.slit_mm - 0.05; kgf = sp["onset_kgf"] + sp["k_kgf_per_mm"] * shown
        d.rectangle([xz, y0 + hz - 44, xz + wz, y0 + hz], fill=(0, 0, 0))
        text(d, (xz + 16, y0 + hz - 22), f"slit bar {shown:.1f} of {row.slit_mm:.0f} mm" + ("  (at the stop)" if at_stop else ""), font(24, True), WHITE, "lm")
        text(d, (xz + wz - 16, y0 + hz - 22), (("≥ " if at_stop else "≈ ") + f"{kgf:.1f} kgf") if shown > 0.05 else "below 0.5 kgf", font(24, True), AMBER if shown > 0.05 else LIGHT, "rm")
        if np.isfinite(jw[i]): text(d, (x0 + wh - 10, y0 + 10), f"aperture {jw[i]:+.1f} mm", font(22, True), WHITE, "ra")
        out = header_bar(to_bgr(im), "6", "Next: grip force, still without electronics")
        if first is None: first = out
        return out
    out = None
    for i in range(N):
        if F0 + i in frames: out = render(i); emit(out, 3 if i % 5 in (2, 4) else 2)  # 25 fps source at 0.5x into 30 fps
    emit(out, FPS // 2 if SHORT else FPS)
    return first, out

# ---------- assemble ----------
plan = [(seg_hardware, m_hardware, 0, "The glove on the hand, then beside the YAM gripper", 700),
        (seg_yam, m_rect(116, 100, 1804, 1050), 1, "From the YAM-UMI video; pose from the ball, width from the jaw tails, in one camera. The fisheye is the ablation", 40),
        (seg_labels, m_combo, 2, "Both cameras of one glove session, " + ("one demonstration" if SHORT else "two demonstrations") + ", 1\u00d7 speed; the numbers are solved from the head frame as it plays", 16),
        (seg_deploy, m_rect(0, 115, 1920, 875), 3, "Phone view and the positioner camera" + (", 3\u00d7 speed" if SHORT else ""), 40),
        (seg_compare, m_rect(0, 180, 1920, 1040), 4, "Placement 23: each policy's own camera (top) and a third-person camera (bottom), 1.5\u00d7 speed", 40),
        (seg_spring, m_spring, 5, "Spring glove: a jug lift at 0.7\u00d7, then a block grasp at 0.5\u00d7", 680)]
plan = [e for e in plan if e[0] is not seg_spring or spring_available()]
for seg, fn, act, cap, cx in plan: set_media(fn, act, cap, cx); seg()
for _ in range(FPS): ff.stdin.write(LAST.tobytes()); nframes += 1
for i in range(1, 16): ff.stdin.write(cv2.addWeighted(LAST, 1 - i / 16, black, i / 16, 0).tobytes()); nframes += 1
ff.stdin.close(); ff.wait()
print(OUT, nframes, f"{nframes / FPS:.1f} s", "spring segment:", spring_available())
