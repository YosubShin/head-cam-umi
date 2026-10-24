# Camera Certification Gauntlet

Every camera entering the tracking fleet passes this before anything depends
on it. Each step exists because its absence once cost days (the
2026-08/09 camera campaign). Spec sheets list
controls; only behavior counts.

## Purchase checklist (before ordering)

- **px/deg ≥ ~25** at the intended resolution AND framerate ≥ 25fps
  *simultaneously* (compute px/deg = width_px / HFOV_deg; check the actual
  UVC mode table — 4K@15fps is not 4K).
- FOV fits the role: ~88–100°H for ego/head; narrower is fine for fixed
  trackers.
- Fixed focus, or lockable focus (M12 counts; AF must expose
  `focus_automatic_continuous=0` + `focus_absolute`).
- Manual exposure *listed* (will be behaviorally verified on arrival — the
  listing alone proved meaningless once).
- On-camera compression (MJPEG) for portable/Pi rigs; uncompressed (YUY2)
  acceptable only for workstation-tethered roles.
- Light budget for ~2ms exposure in the target environment (F-number +
  pixel size + sensor generation; e.g. F1.65/2µm STARVIS 2 ≈ 7× a cheap
  webcam).

## Arrival gauntlet (~1 hour)

1. **Streak test — functional manual exposure** (the "cosmetic knob" trap).
   Open `scripts/exposure_target.html` fullscreen (bar hops every 16.7ms);
   capture at low/high exposure values. Bar-ghost count ≈ exposure/16.7ms.
   Also run `scripts/exposure_sweep.py` (rapid switching outruns ISP
   brightness compensation; brightness must scale ~linearly with the value
   at fixed gain). NEVER judge exposure by settled brightness — HDR ISPs
   tone-map it flat. A spinner toy gives a continuous-motion cross-check.
   - Learn the real units (bar counts), the frame-period clamp
     (values > 1/fps silently cap), and the gain range's actual strength.
2. **Latch/reset behavior — the two control traps.**
   - cv2.VideoCapture RESETS UVC controls on open.
   - Control writes may only physically latch AFTER streaming starts
     (readback shows accepted values either way — readback is NOT proof).
   - Behavioral check: **hand-wave blur test** in `scripts/uvc_live.py` —
     crisp smear at the commanded short exposure = actually latched.
   - All pilot tools apply the settings ritual mid-stream for this reason;
     a new camera may differ — verify, don't assume.
3. **ChArUco intrinsics under verified locks** —
   `scripts/calibrate_uvc.py` (locks applied post-stream, readback printed;
   board: 10x7, 25.0mm square / 18.06mm marker measured). Pass: held-out
   mean reprojection ≲ 0.5px. Recalibrate after ANY focus change (~2% fx
   per focus step = focus breathing) — and after any doubt about whether
   focus was truly locked during calibration.
4. **Dynamics ladder** — record with `scripts/uvc_record.py` (MJPEG
   stream-copy; NEVER decode+re-encode — an inter-frame re-encode crushes
   moving regions and fakes a detection cliff) while a speed ladder
   (`e_runner.py e9` in the private experiment tree) moves an UNOCCLUDED dodeca (occlusion conflates with
   motion — deconfound first). Grade with `dodeca_dynamics.py` (same tree)
   (rigid-pair spread = built-in ground truth; measured image-velocity
   bins, immune to command-window lag).
   - Reference to beat (B0591, e=20/2ms, gain 50): det 99–100% to
     0.5 m/s, 90% to 1.0; spread 3.7mm slow → 12.5mm at 0.5–1.0 m/s.
5. **Exposure choice** — from the blur budget: smear_px = v · t · fx/Z;
   keep ≲ half a marker cell at the target speed. (B0591 @0.5m: e=20.)

## Standing reference numbers

- Rigid marker pairs (dodeca faces, same-finger markers) are a live
  noise-floor meter (~3mm on certified units).
- AprilTag corner refinement (`CORNER_REFINE_APRILTAG`) is the pipeline
  default — A/B'd at 1.6× precision and 4.5× coverage over SUBPIX on MJPEG.
- Fleet ranking and roles: e8/README.md (run 5+) in the private experiment tree.

See also: PROBLEMS_AND_FIXES.md — the full ledger of every failure mode
encountered building this system, with root causes and fix locations.
