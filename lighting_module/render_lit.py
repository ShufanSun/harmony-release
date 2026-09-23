"""Render a 'lit preview' of the final composite scene.

The HARMONY rasterizer is flat-shaded and the reference inpaints already
have natural lighting baked in. So instead of raytracing, this module
produces a *preview* image good enough for the VLM to compare against the
reference photo and refine parameters:

  base       = the final composite render (decorations placed)
  + ambient  = a flat tint scaled by ambient intensity
  + glows    = per point-light: a Gaussian halo at the projected world
               position, additively blended in the light's colour
  + sun      = a soft warm wash falling off in the directional light's
               screen-space direction (only for windows whose daylight is on)

We never re-render the scene from scratch and never modify decoration or
furniture renders on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_camera(out_dir: Path) -> dict | None:
    for cand in (out_dir / "camera_vggt.json", out_dir / "camera.json"):
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def _project(world_pt, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy):
    d = np.asarray(world_pt, dtype=np.float64) - cam_pos
    xc = float(np.dot(d, right_v))
    yc = float(np.dot(d, up_c_v))
    zc = float(np.dot(d, fwd_v))
    if zc <= 0.01:
        return None
    return (cx + fx * xc / zc, cy - fx * yc / zc, zc)


def _gaussian_disk(W: int, H: int, cx: float, cy: float,
                   radius_px: float) -> np.ndarray:
    if radius_px <= 0.5:
        return np.zeros((H, W), dtype=np.float32)
    y, x = np.ogrid[:H, :W]
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    sigma = max(radius_px / 2.0, 1.0)
    g = np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)
    return g


def _add_glow(buf: np.ndarray, mask: np.ndarray, color01: np.ndarray,
              strength: float) -> None:
    if strength <= 0:
        return
    # Local highlights need to be punchy because the rest of the frame is
    # left close to the original (already lit) inpaint composite. At
    # strength=1.0 the centre of the halo can add up to 110/255 to a white
    # pixel — typical 0.15–0.30 lamp glow → 17–33/255 bump localised to the
    # shade. The frame-wide ambient/directional contributions are kept
    # deliberately weak so this localised bump is the dominant cue.
    bump = (mask[..., None] * color01[None, None, :] * (110.0 * strength))
    out = buf.astype(np.float32) + bump
    np.clip(out, 0.0, 255.0, out=out)
    buf[...] = out.astype(np.uint8)


def _dim_base(buf: np.ndarray, factor: float) -> None:
    """Multiply the (already-lit) base composite down to an 'unlit' starting
    point so the per-light contributions added afterwards are the dominant
    visual cue. factor=0.55 gives a reasonable dim baseline; factor=1.0
    disables dimming."""
    f = float(np.clip(factor, 0.20, 1.0))
    if f >= 0.999:
        return
    out = buf.astype(np.float32) * f
    np.clip(out, 0.0, 255.0, out=out)
    buf[...] = out.astype(np.uint8)


def _ambient_lift(buf: np.ndarray, color01: np.ndarray, intensity: float) -> None:
    """Additive global ambient — fills in the dim base toward a soft base
    luminance so unlit areas don't go pitch-black. Kept weak so the scene
    still feels driven by the explicit light sources."""
    if intensity <= 0:
        return
    add = (np.asarray(color01, dtype=np.float32)
           * float(intensity) * 100.0)
    out = buf.astype(np.float32) + add[None, None, :]
    np.clip(out, 0.0, 255.0, out=out)
    buf[...] = out.astype(np.uint8)


def _scene_tone(lights: dict) -> tuple[np.ndarray, float]:
    """Derive a net 'scene tone' RGB and a total weight from active lights.

    The tone is the intensity-weighted mean colour of every emitter that's
    currently ON. Total weight conveys how much energy is in the scene
    (used to scale the grading strength: a dim scene still looks dim, even
    if the only active light is very warm).
    """
    rgb = np.zeros(3, dtype=np.float32)
    w_total = 0.0

    amb = lights.get("ambient") or {}
    w = float(amb.get("intensity", 0.0)) * 0.5  # ambient weighs half
    if w > 0:
        rgb += w * np.asarray(amb.get("color", [1, 1, 1]), dtype=np.float32)
        w_total += w

    for d in lights.get("directional") or []:
        w = float(d.get("intensity", 0.0))
        if w <= 0:
            continue
        rgb += w * np.asarray(d.get("color", [1, 1, 1]), dtype=np.float32)
        w_total += w

    for p in lights.get("point_lights") or []:
        if not p.get("on", True):
            continue
        w = float(p.get("intensity", 0.0))
        if w <= 0:
            continue
        rgb += w * np.asarray(p.get("color", [1, 1, 1]), dtype=np.float32)
        w_total += w

    for win in lights.get("windows") or []:
        if not win.get("daylight_on", True):
            continue
        w = float(win.get("intensity", 0.0))
        if w <= 0:
            continue
        rgb += w * np.asarray(win.get("color", [1, 1, 1]), dtype=np.float32)
        w_total += w

    if w_total < 1e-6:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32), 0.0
    return rgb / w_total, w_total


def _apply_color_grade(buf: np.ndarray, tone: np.ndarray,
                       strength: float) -> None:
    """Soft-shift the image's colour balance toward `tone` (RGB in [0,1]).

    Per-channel multiplier kept centred on 1 so overall luminance is roughly
    preserved — only chromaticity moves. `strength` ∈ [0, 1] is how far to
    push (0 = no change, 1 = the full tone bias). Realistic settings
    are 0.2–0.6.
    """
    if strength <= 0:
        return
    mean = float(tone.mean())
    if mean < 1e-6:
        return
    tone_norm = tone / mean                       # mean(tone_norm) = 1
    mult = 1.0 + float(strength) * (tone_norm - 1.0)
    mult = np.clip(mult, 0.6, 1.6)                # safety
    out = buf.astype(np.float32) * mult[None, None, :]
    np.clip(out, 0.0, 255.0, out=out)
    buf[...] = out.astype(np.uint8)


def _directional_wash(buf: np.ndarray, direction_world: np.ndarray,
                      cam_basis, color01: np.ndarray, intensity: float) -> None:
    if intensity <= 0:
        return
    right_v, up_c_v, _ = cam_basis
    # Project the direction into screen space; the wash brightens the side
    # the light points TOWARDS (i.e. "where the sunlit half of the room is").
    sx = float(np.dot(direction_world, right_v))
    sy = -float(np.dot(direction_world, up_c_v))
    n = (sx * sx + sy * sy) ** 0.5
    if n < 1e-6:
        sx, sy = 0.0, 1.0
    else:
        sx, sy = sx / n, sy / n

    H, W, _ = buf.shape
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    # Normalise to [-1, 1]
    nx = (xx - W / 2.0) / (W / 2.0)
    ny = (yy - H / 2.0) / (H / 2.0)
    grad = np.clip(0.5 + 0.5 * (sx * nx + sy * ny), 0.0, 1.0)

    bump = grad[..., None] * color01[None, None, :] * (255.0 * 0.30 * float(intensity))
    out = buf.astype(np.float32) + bump
    np.clip(out, 0.0, 255.0, out=out)
    buf[...] = out.astype(np.uint8)


def render(output_dir: str | Path, lights: dict, sources: list[dict],
           dst: Path, base_image: Path | None = None) -> Path | None:
    """Composite a lit preview from `lights` + `sources` and save to `dst`.

    `base_image` defaults to decorations/placements/render_decorations_placed.png
    (the final composite), falling back to the furniture render if absent.
    """
    out_dir = Path(output_dir)
    if base_image is None:
        for cand in (out_dir / "decorations" / "placements" / "render_decorations_placed.png",
                     out_dir / "decorations" / "placements" / "render_final_refine.png",
                     out_dir / "furniture" / "render_furniture_placed.png",
                     out_dir / "render_final.png"):
            if cand.exists():
                base_image = cand
                break
    if base_image is None or not Path(base_image).exists():
        print("[lighting/render] no base render found — cannot preview")
        return None

    base = Image.open(str(base_image)).convert("RGB")
    base.thumbnail((2000, 2000), Image.LANCZOS)
    buf = np.array(base, dtype=np.uint8).copy()
    H, W, _ = buf.shape

    # Camera projection
    camera = _load_camera(out_dir)
    if camera is None:
        print("[lighting/render] no camera — skipping projection of light positions")
        Image.fromarray(buf).save(str(dst))
        return dst

    cam_pos = np.array(camera["position_m"], dtype=np.float64)
    look_at = np.array(camera["look_at_m"], dtype=np.float64)
    up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    fwd = look_at - cam_pos
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.cross(fwd, up_world)
    right /= max(np.linalg.norm(right), 1e-9)
    up_c = np.cross(right, fwd)
    up_c /= max(np.linalg.norm(up_c), 1e-9)
    hfov = float(camera["hfov_deg"])
    fx = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    # ── Lights-only mode (default) ──
    # The base composite already has photographic lighting baked in; we
    # don't add a global wash, ambient lift, directional gradient, or tone
    # grade. Only the explicit light sources contribute, as localised
    # additive halos at their projected positions. This keeps the lamp /
    # window highlights as the dominant cue without the walls drifting.
    #
    # The legacy "global wash" path (dim base + ambient + directional +
    # grading) is still reachable by setting `lights_only=false` in the
    # lights dict — kept for back-compat.
    lights_only = bool(lights.get("lights_only", True))

    by_id = {s["id"]: s for s in sources if "id" in s}

    if not lights_only:
        _dim_base(buf, float(lights.get("base_dim", 0.70)))

        amb = lights.get("ambient", {})
        _ambient_lift(buf,
                      np.array(amb.get("color", [1, 1, 1]), dtype=np.float32),
                      float(amb.get("intensity", 0.0)))

        for w in lights.get("windows", []):
            if not w.get("daylight_on", True):
                continue
            s = by_id.get(w["id"])
            direction = np.array([0.0, -0.5, 1.0], dtype=np.float64)
            if s is not None and s.get("wall"):
                wall = s["wall"]
                if wall == "back":
                    direction = np.array([0.0, -0.4, 1.0])
                elif wall == "left":
                    direction = np.array([1.0, -0.4, 0.0])
                elif wall == "right":
                    direction = np.array([-1.0, -0.4, 0.0])
                elif wall == "front":
                    direction = np.array([0.0, -0.4, -1.0])
            direction /= max(np.linalg.norm(direction), 1e-9)
            _directional_wash(buf, direction, (right, up_c, fwd),
                              np.array(w.get("color", [1, 1, 1]), dtype=np.float32),
                              float(w.get("intensity", 0.0)))

        for d in lights.get("directional", []):
            direction = np.array(d.get("direction", [0.0, -1.0, 0.0]), dtype=np.float64)
            n = np.linalg.norm(direction)
            if n < 1e-6:
                continue
            direction /= n
            _directional_wash(buf, direction, (right, up_c, fwd),
                              np.array(d.get("color", [1, 1, 1]), dtype=np.float32),
                              float(d.get("intensity", 0.0)))

    # ── Localised window glows (lights-only AND legacy path) ──
    # Even in lights-only mode we want the window to "glow" where it
    # actually is on screen, not as a frame-wide wash. We project the
    # window's anchor and add a Gaussian halo sized to its physical extent.
    if lights_only:
        for w in lights.get("windows", []):
            if not w.get("daylight_on", True):
                continue
            intensity = float(w.get("intensity", 0.0))
            if intensity <= 1e-3:
                continue
            s = by_id.get(w["id"])
            if s is None:
                continue
            anchor = np.array(s["position_m"], dtype=np.float64)
            proj = _project(anchor, cam_pos, right, up_c, fwd, fx, cx, cy)
            if proj is None:
                continue
            px, py, zc = proj
            size = s.get("size_m") or {}
            size_w = float(size.get("width_m", 0.6))
            size_h = float(size.get("height_m", 0.9))
            radius_world = max(size_w, size_h) * 0.55
            radius_px = max(40.0, fx * radius_world / max(zc, 0.5))
            mask = _gaussian_disk(W, H, px, py, radius_px)
            color01 = np.array(w.get("color", [1, 1, 1]), dtype=np.float32)
            _add_glow(buf, mask, color01, intensity)

    # ── Per-lamp / per-fixture point-light glows (always on) ──
    for p in lights.get("point_lights", []):
        if not p.get("on", True) or float(p.get("intensity", 0.0)) <= 1e-3:
            continue
        s = by_id.get(p["id"])
        if s is None:
            continue
        anchor = np.array(s["position_m"], dtype=np.float64)
        offset = np.array(p.get("offset_m", [0, 0.05, 0]), dtype=np.float64)
        proj = _project(anchor + offset, cam_pos, right, up_c, fwd, fx, cx, cy)
        if proj is None:
            continue
        px, py, zc = proj
        if not (0 <= px < W and 0 <= py < H):
            continue
        size_h = (s.get("size_m") or {}).get("height_m", 0.2)
        radius_world = max(0.5 * size_h, float(p.get("radius_m", 0.05)) * 4.0)
        radius_px = max(20.0, fx * radius_world / max(zc, 0.5))
        mask = _gaussian_disk(W, H, px, py, radius_px)
        color01 = np.array(p.get("color", [1, 1, 1]), dtype=np.float32)
        _add_glow(buf, mask, color01, float(p["intensity"]))

    # Shadow pass: contact + cast shadows for every placed object.
    # Skipped only if explicitly disabled in the lights dict.
    if bool(lights.get("shadows", True)):
        from . import shadows as _shadows
        try:
            _shadows.add_shadows(buf, out_dir, lights, sources, camera)
        except Exception as e:
            print(f"[lighting/render] shadow pass failed: {e}")

    # Color grading still allowed in legacy path; lights-only skips it.
    if not lights_only:
        tone, w_total = _scene_tone(lights)
        grade_strength = float(lights.get("grading_strength", 0.30))
        grade_strength *= float(np.clip(w_total / 0.3, 0.0, 1.0))
        _apply_color_grade(buf, tone, grade_strength)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(buf).save(str(dst))
    print(f"[lighting/render] saved → {dst}")
    return dst
