"""
floorplan/pipeline.py
---------------------
Full HARMONY pipeline orchestrator.

Four stages in order:

  Stage 1 — VLM Init
    depth estimation → corner analysis → VLM scene analysis (geometry + camera
    angle) → wall mesh (walls.obj) → VLM camera placement (camera.json)

  Stage 2 — VGGT Camera Alignment
    VGGT inference on the reference image → Manhattan-world plane fitting →
    refined camera orientation (camera_vggt.json).  The VLM position is kept;
    only the viewing direction is corrected to better match real wall lines.

  Stage 3 — VLM Texture Generation
    VLM identified surface descriptions (from Stage 1) → Qwen image-edit
    pipeline generates seamless per-wall textures using the reference photo as
    a style guide → walls_metadata.json updated.

  Stage 4 — Final Render
    walls.obj + camera_vggt.json + per-wall textures → render_final.png

Usage
-----
  python -m floorplan.pipeline \\
      --image  data/indoor_images/office5.jpg \\
      --output outputs/office5_pipeline

  from floorplan.pipeline import run_pipeline
  run_pipeline("data/indoor_images/office5.jpg", "outputs/office5_pipeline")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_geometry(out: Path, camera_path: Path | str, label: str) -> None:
    """
    Render a gray geometry-only view (no textures) and save it as
    render_{label}.png inside `out`.  Skips silently if walls.obj is missing
    or the render fails.
    """
    from floorplan.wall_line.render_room import render_room

    obj_path    = out / "walls.obj"
    render_path = out / f"render_{label}.png"

    if not obj_path.exists():
        print(f"[render] walls.obj not found — skipping {label} render.")
        return
    if render_path.exists():
        print(f"[render] Reusing existing {render_path.name}")
        return
    try:
        render_room(
            mesh_path=str(obj_path),
            camera_json_path=str(camera_path),
            out_path=str(render_path),
            texture_dir=None,
            bg_color=(40, 40, 40),
        )
        print(f"[render] {label} camera render → {render_path}")
    except Exception as e:
        print(f"[render] {label} render failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 helpers: geometry + camera (no textures)
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage1(image_path: str, out: Path) -> dict:
    """
    Run the VLM scene-init stage: view check → analysis → mesh → camera.

    This is a lightweight wrapper that calls only the geometry/camera sub-steps
    of run_floorplan_stage, deferring texture generation to Stage 3.
    Returns a dict with keys: analysis, camera.
    """
    from floorplan.agent.prompt_gen import (
        analyze_floorplan, draw_floorplan_ascii, check_view_angle,
    )
    from floorplan.wall_line.wall_mesh_gen import generate_wall_mesh
    from floorplan.wall_line.camera_placement import camera_from_analysis

    print("\n" + "═" * 60)
    print(" STAGE 1 — VLM SCENE INIT")
    print("═" * 60)

    # Geometry is VGGT-only: Stage 2 supplies metric depth, normals and the
    # Manhattan room frame.  Stage 1 only needs the VLM scene analysis.

    # ── view type ─────────────────────────────────────────────────────────────
    view_info_path = out / "view_info.json"
    if view_info_path.exists():
        view_info = json.loads(view_info_path.read_text())
        print(f"[stage1] Reusing view_info.json → view_type={view_info.get('view_type')}")
    else:
        view_info = check_view_angle(image_path)
        view_info_path.write_text(json.dumps(view_info, indent=2))

    # ── VLM analysis ──────────────────────────────────────────────────────────
    analysis_cache = out / "floorplan_analysis.json"
    if analysis_cache.exists():
        print(f"[stage1] Reusing floorplan_analysis.json")
        analysis = json.loads(analysis_cache.read_text())
    else:
        analysis = analyze_floorplan(image_path)
        if analysis:
            analysis_cache.write_text(json.dumps(analysis, indent=2))
            print(f"[stage1] Analysis saved → {analysis_cache}")

    if not analysis:
        raise RuntimeError("[stage1] VLM analysis returned empty result — aborting pipeline.")

    # ── floor plan ASCII ──────────────────────────────────────────────────────
    plan_path = out / "floorplan.txt"
    if not plan_path.exists():
        plan_path.write_text(draw_floorplan_ascii(analysis) + "\n")
    print(f"[stage1] Floor plan → {plan_path}")

    # ── wall mesh ─────────────────────────────────────────────────────────────
    obj_path = out / "walls.obj"
    if obj_path.exists():
        print(f"[stage1] Reusing existing walls.obj")
    else:
        # Build the rectangular room from the VLM floor_dims.  This is only the
        # Stage-1 seed: Stage 2 (VGGT Manhattan) + geometry_refine fit the real
        # wall extents from metric depth.
        room_info = analysis.get("room", {})
        # Build floor_dims with VLM-derived fallbacks when fields are missing
        # or zero.  Without floor_dims the wall_mesh generator falls into the
        # depth-based polygonal path, which uses an inverted Z convention
        # (back wall at Z=Z_back > 0) that is incompatible with Stage-1's
        # camera placement (back wall at Z=0).  Forcing usable floor_dims
        # keeps Stage 1 on the rectangular path so the rest of the pipeline
        # operates on a consistent coordinate frame.
        _fw_raw = room_info.get("floor_width_m")
        _fd_raw = room_info.get("floor_depth_m")
        _fw = float(_fw_raw) if _fw_raw and float(_fw_raw) > 0.1 else 0.0
        _fd = float(_fd_raw) if _fd_raw and float(_fd_raw) > 0.1 else 0.0
        if _fw <= 0 or _fd <= 0:
            _cam_info = analysis.get("camera", {}) or {}
            _vfd = float(_cam_info.get("visible_floor_depth_m") or 0)
            _dist_back = float(_cam_info.get("dist_to_back_m") or 0)
            if _fw <= 0:
                _fw = 5.0
            if _fd <= 0:
                # Prefer dist_to_back_m × 1.3 (camera not flush with front wall),
                # else visible_floor_depth_m × 1.5, else default 4.0.
                _fd = (_dist_back * 1.3) if _dist_back > 0.5 \
                      else (_vfd * 1.5) if _vfd > 0.5 \
                      else 4.0
            print(f"[stage1] floor_dims fallback: VLM gave "
                  f"({_fw_raw!r}, {_fd_raw!r}) → using ({_fw:.2f}, {_fd:.2f})m  "
                  f"(forces rectangular mesh)")
        floor_dims = (_fw, _fd)
        try:
            generate_wall_mesh(
                out_path=str(obj_path),
                ceiling_h=float(room_info.get("ceiling_height_m") or 2.4),
                floor_dims=floor_dims,
            )
        except Exception as e:
            print(f"[stage1] Wall mesh generation failed: {e}")

    # ── VLM camera placement ──────────────────────────────────────────────────
    cam_path = out / "camera.json"
    if cam_path.exists():
        print(f"[stage1] Reusing existing camera.json")
        camera = json.loads(cam_path.read_text())
    else:
        camera = {}
        try:
            camera = camera_from_analysis(
                analysis=analysis,
                ref_image_path=image_path,
                out_path=str(cam_path),
            )
            print(f"[stage1] Camera saved → {cam_path}")
        except Exception as e:
            print(f"[stage1] Camera placement failed: {e}")

        # If camera_from_analysis failed (or skipped), write a fallback so
        # Stage 2 always has a valid camera.json to start from.
        if not cam_path.exists():
            room = analysis.get("room", {})
            raw_width = room.get("floor_width_m")
            raw_depth = room.get("floor_depth_m")
            raw_ceil  = room.get("ceiling_height_m")
            width_m   = float(raw_width) if raw_width else 5.0
            depth_m   = float(raw_depth) if raw_depth else 4.0
            ceiling_h = float(raw_ceil)  if raw_ceil  else 2.4
            if not raw_width or not raw_depth:
                print(f"[stage1] WARNING: fallback camera using default room dims "
                      f"({width_m:.1f}×{depth_m:.1f}m) because VLM omitted "
                      f"floor_width_m={raw_width!r}/floor_depth_m={raw_depth!r}.")
            import cv2 as _cv2
            _img = _cv2.imread(image_path)
            img_w, img_h = (_img.shape[1], _img.shape[0]) if _img is not None else (1280, 720)
            camera = {
                "position_m": [width_m / 2, ceiling_h * 0.45, depth_m - 0.05],
                "look_at_m":  [width_m / 2, ceiling_h * 0.45, 0.0],
                "up":         [0, 1, 0],
                "hfov_deg":   70.0,
                "vfov_deg":   round(70.0 * img_h / max(img_w, 1), 4),
                "width_px":   img_w,
                "height_px":  img_h,
            }
            cam_path.write_text(json.dumps(camera, indent=2))
            print(f"[stage1] Fallback camera written → {cam_path}")

    # ── VLM camera render (geometry-only, no textures) ────────────────────────
    _render_geometry(out, cam_path, label="vlm")

    print(f"[stage1] ✓ Geometry & camera complete.")
    return dict(analysis=analysis, camera=camera)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: VGGT inference + camera alignment
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage2(image_path: str, out: Path,
                vggt_subsample: int = 4,
                vggt_device: str | None = None,
                vggt_n_iter: int = 1) -> str:
    """
    Run VGGT on the reference image and align the camera via Manhattan-guided
    VLM calibration — no Stage 1 camera.json required.

    Steps:
      1. VGGT inference → depth, normals, point cloud, intrinsics.
      2. Manhattan estimation → room box anchor corners.
      3. VLM initial placement → camera positioned from Manhattan wireframe.
      4. Iterative refinement → corner-pixel translation + orbit for line angles.

    Returns the path to camera_vggt.json, or camera.json fallback if VGGT failed.
    """
    from floorplan.vggt_estimates.inference import run as run_vggt
    from floorplan.vggt_estimates.align_to_walls import (
        align_camera, align_camera_direct, save_camera,
    )

    print("\n" + "═" * 60)
    print(" STAGE 2 — VGGT CAMERA ALIGNMENT  (Manhattan + VLM loop)")
    print("═" * 60)

    vggt_dir      = out / "vggt"
    vggt_dir.mkdir(parents=True, exist_ok=True)
    cam_vggt_path = out / "camera_vggt.json"

    # ── VGGT inference ──────────────────────────────────────────────────────────
    vggt_cam_json = vggt_dir / "camera.json"
    if vggt_cam_json.exists():
        print(f"[stage2] Reusing existing geometry in {vggt_dir}")
    else:
        try:
            kw = {}
            if vggt_device:
                kw["device"] = vggt_device
            run_vggt([image_path], out_dir=str(vggt_dir), **kw)
        except Exception as e:
            print(f"[stage2] VGGT inference failed: {e}")
            fallback = out / "camera.json"
            if fallback.exists():
                print(f"[stage2] Falling back to Stage-1 camera.")
                return str(fallback)
            raise

    # ── Manhattan + VLM alignment ─────────────────────────────────────────────
    if cam_vggt_path.exists():
        print(f"[stage2] Reusing existing camera_vggt.json")
    else:
        vlm_cam_path = out / "camera.json"
        try:
            if vlm_cam_path.exists():
                # Stage 1 produced a camera — use it as starting point so the
                # VLM render_vlm.png baseline is preserved and only fine-tuned.
                print(f"[stage2] Starting from Stage-1 camera (align_camera).")
                cam_vggt = align_camera(
                    vggt_out_dir=str(vggt_dir),
                    walls_out_dir=str(out),
                    subsample=vggt_subsample,
                    n_iter=vggt_n_iter,
                )
            else:
                # No Stage 1 camera — place camera from Manhattan geometry.
                print(f"[stage2] No Stage-1 camera — using align_camera_direct.")
                cam_vggt = align_camera_direct(
                    vggt_out_dir=str(vggt_dir),
                    walls_out_dir=str(out),
                    subsample=vggt_subsample,
                    n_iter=vggt_n_iter,
                )
            save_camera(cam_vggt, str(cam_vggt_path))
        except Exception as e:
            print(f"[stage2] VGGT alignment failed: {e}")
            if vlm_cam_path.exists():
                print(f"[stage2] Falling back to Stage-1 camera.")
                return str(vlm_cam_path)
            raise

    print(f"[stage2] ✓ VGGT-refined camera → {cam_vggt_path}")
    return str(cam_vggt_path)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: VLM texture generation via Qwen image edit
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that indicate a plain, uniformly-painted surface.
# These surfaces should be rendered as a solid colour rather than a
# generative tiled texture (which would incorrectly add a pattern).
_PLAIN_SURFACE_KEYWORDS = (
    "no pattern", "uniform surface", "uniform color", "plain",
    "solid color", "solid colour", "matte paint", "painted plaster",
    "painted wall", "matte white", "matte gray", "matte grey",
    "matte beige", "matte cream", "matte off-white",
)

# Rough RGB values for common interior paint colours.
_COLOUR_MAP: dict[str, tuple[int, int, int]] = {
    "white":     (245, 243, 240),
    "off-white": (238, 235, 228),
    "cream":     (235, 228, 210),
    "beige":     (220, 210, 195),
    "gray":      (185, 183, 180),
    "grey":      (185, 183, 180),
    "light gray":(210, 208, 205),
    "light grey":(210, 208, 205),
    "dark gray": (130, 128, 126),
    "dark grey": (130, 128, 126),
    "blue":      (180, 195, 215),
    "green":     (185, 205, 185),
    "yellow":    (230, 220, 185),
}


def _is_plain_surface(surface: str) -> bool:
    """Return True when the surface description indicates a plain painted wall."""
    s = surface.lower()
    return any(kw in s for kw in _PLAIN_SURFACE_KEYWORDS)


def _sampled_wall_colour(image_path: str | Path) -> tuple[int, int, int]:
    """Representative wall colour sampled directly from the reference photo.

    Walls are textured by backprojection (texture_from_reference); the floorplan
    stage only needs a believable solid placeholder for render_final / fallback,
    so there's no need to run text-to-image generation for them. We take the
    median RGB of an upper-middle horizontal band (below the ceiling line, above
    most furniture), excluding very dark (furniture/shadow) and very bright
    (window/highlight) pixels.
    """
    from PIL import Image as _Image
    import numpy as _np
    try:
        im = _np.array(_Image.open(image_path).convert("RGB"))
    except Exception:
        return (200, 198, 195)
    H, W = im.shape[:2]
    band = im[int(0.12 * H):int(0.45 * H)].reshape(-1, 3).astype(_np.float64)
    lum = band.mean(axis=1)
    keep = (lum > 40) & (lum < 235)
    if int(keep.sum()) < 100:
        keep = _np.ones(len(band), dtype=bool)
    return tuple(int(x) for x in _np.median(band[keep], axis=0))


def _sampled_floor_colour(image_path: str | Path) -> tuple[int, int, int]:
    """Representative floor colour sampled from the photo's lower band (the
    foreground floor), excluding very dark/bright pixels. Like the wall, the
    real floor texture comes from backprojection, so the floorplan stage only
    needs a believable solid placeholder — no text-to-image needed."""
    from PIL import Image as _Image
    import numpy as _np
    try:
        im = _np.array(_Image.open(image_path).convert("RGB"))
    except Exception:
        return (190, 180, 168)
    H, W = im.shape[:2]
    band = im[int(0.72 * H):int(0.97 * H)].reshape(-1, 3).astype(_np.float64)
    lum = band.mean(axis=1)
    keep = (lum > 35) & (lum < 240)
    if int(keep.sum()) < 100:
        keep = _np.ones(len(band), dtype=bool)
    return tuple(int(x) for x in _np.median(band[keep], axis=0))


def _solid_colour_texture(surface: str, out_path: Path, size: int = 512,
                          rgb: tuple[int, int, int] | None = None) -> str:
    """
    Save a solid-colour PNG for a plain painted surface.
    Adds ±3 per-channel noise so the texture doesn't look perfectly artificial.
    An explicit `rgb` overrides the surface-description colour lookup (used to
    pass a colour sampled from the reference photo).
    """
    from PIL import Image as _Image
    import numpy as _np

    s = surface.lower()
    if rgb is None:
        rgb = (200, 198, 195)   # neutral fallback
        for name, colour in _COLOUR_MAP.items():
            if name in s:
                rgb = colour
                break

    rng = _np.random.default_rng(42)
    arr = _np.clip(
        _np.full((size, size, 3), rgb, dtype=_np.int16)
        + rng.integers(-3, 4, (size, size, 3), dtype=_np.int16),
        0, 255,
    ).astype(_np.uint8)
    _Image.fromarray(arr).save(str(out_path))
    print(f"[stage3] Solid-colour texture ({rgb}) → {out_path}")
    return str(out_path)

def _run_stage3(image_path: str, out: Path, analysis: dict) -> dict:
    """
    Generate per-wall seamless textures using Qwen image edit.

    The VLM already identified surface descriptions in Stage 1 analysis.
    Here we:
      1. Do a gray geometry-only render with the refined camera so the VLM can
         visually verify which surface is which (optional VLM check step).
      2. Call QwenImageEditPipeline for each wall surface using the reference
         photo as style input.
      3. Save walls_metadata.json with texture paths for the renderer.

    Returns a dict with walls_metadata and texture file paths.
    """
    from floorplan.agent.prompt_gen import (
        generate_texture_image,
    )

    print("\n" + "═" * 60)
    print(" STAGE 3 — VLM TEXTURE GENERATION (Qwen image edit)")
    print("═" * 60)

    def _vlm_verify_texture(texture_path: "Path | str",
                             ref_photo_path: "Path | str",
                             surface: str,
                             prompt: str) -> "dict | None":
        """Ask Qwen3-VL whether the generated texture's material/colour/pattern
        matches the corresponding surface in the reference photo.  Returns a
        dict {correct, reasoning, revised_prompt} or None on failure.
        """
        import base64, re as _re, json as _json
        try:
            import requests
        except Exception:
            return None
        try:
            tex_b64 = base64.b64encode(Path(texture_path).read_bytes()).decode("ascii")
            ref_b64 = base64.b64encode(Path(ref_photo_path).read_bytes()).decode("ascii")
        except Exception as _e:
            print(f"[stage3/verify] could not read inputs ({_e}) — skipping check")
            return None
        _ask = (
            f"You are checking whether a generated **{surface} texture** "
            f"matches the {surface} surface in a reference room photograph.\n\n"
            f"Image 1: the **generated {surface} texture** (orthographic / "
            f"flat tile view).\n"
            f"Image 2: the **reference room photograph** — look at the "
            f"{surface} portion only.\n\n"
            f"Compare them on:\n"
            f"  - dominant colour family\n"
            f"  - material (wood / tile / concrete / fabric / plaster …)\n"
            f"  - pattern direction and scale (planks, grid, herringbone, …)\n\n"
            f"Reply with strict JSON:\n"
            f"{{\"correct\": true|false, "
            f"\"reasoning\": \"<one sentence>\", "
            f"\"revised_prompt\": \"<improved prompt>\"}}\n\n"
            f"If correct is true, revised_prompt may be null.  If false, "
            f"`revised_prompt` MUST be a complete prompt suitable for an "
            f"image-generation model — same intent as the original, but "
            f"adjusted to fix the specific mismatch you observed.\n"
            f"The original prompt was: \"{prompt}\""
        )
        payload = {
            "model": "qwen3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": _ask},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{tex_b64}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            resp = _vlm_post(payload, timeout=60)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            raw = _re.sub(r"<think>[\s\S]*?</think>", "", raw,
                          flags=_re.IGNORECASE).strip()
            m = _re.search(r"\{[\s\S]*?\}", raw)
            if not m:
                print(f"[stage3/verify] no JSON in VLM reply — skipping check")
                return None
            return _json.loads(m.group())
        except Exception as _e:
            print(f"[stage3/verify] VLM check failed ({_e}) — keeping texture")
            return None

    walls_meta_path = out / "walls_metadata.json"
    if walls_meta_path.exists():
        # Reuse only if all referenced texture files actually exist on disk.
        # Otherwise fall through and regenerate the missing ones — Qwen often
        # errors on individual textures while succeeding on others, leaving
        # walls_metadata.json in a half-baked state.
        try:
            _wm_cached = json.loads(walls_meta_path.read_text())
        except Exception:
            _wm_cached = {}
        _required_files = [out / "floor_texture.png", out / "wall_texture.png"]
        for _w_orient, _w_meta in (_wm_cached or {}).items():
            _tex = _w_meta.get("texture_path") if isinstance(_w_meta, dict) else None
            if _tex:
                _required_files.append(Path(_tex))
        _missing = [str(f) for f in _required_files if not f.exists()]
        if not _missing:
            print(f"[stage3] Reusing existing walls_metadata.json (all textures present)")
            return dict(walls_metadata=_wm_cached,
                        walls_metadata_path=str(walls_meta_path))
        print(f"[stage3] walls_metadata.json exists but {len(_missing)} texture(s) "
              f"missing — regenerating: {[Path(p).name for p in _missing]}")

    # ── geometry-only preview (reuse render_vggt.png saved in Stage 2) ────────
    # (no separate render needed here; Stage 2 already wrote render_vggt.png)

    # ── texture generation ────────────────────────────────────────────────────
    global_wall_tex    = analysis.get("wall_texture", {})
    global_wall_prompt = global_wall_tex.get("synthesis_prompt", "")
    floor_tex_info     = analysis.get("floor_texture", {})
    wall_tile_size     = float(global_wall_tex.get("tile_size_m") or 2.0)

    # Floor tile size: derive from room dimensions so the texture appears 1–2 times
    # across the floor, regardless of the VLM's tile_size_m (which reflects the
    # physical material repeat, not the desired render frequency).
    room_info       = analysis.get("room", {})
    _floor_w        = float(room_info.get("floor_width_m") or 4.0)
    _floor_d        = float(room_info.get("floor_depth_m") or 4.0)
    # Stretch the floor texture across the entire floor (tile exactly once)
    # rather than repeating ~1–2 times. The renderer's UV math is `uv / tile`,
    # so setting tile = floor's longest dimension produces UV ∈ [0, 1] across.
    floor_tile_size = max(_floor_w, _floor_d)

    _prompt_to_path: dict[str, str] = {}  # cache: prompt → file path

    def _gen_texture(label: str, prompt: str, out_file: Path, surface: str) -> str | None:
        """Generate one texture and save PBR maps.  Returns file path or None."""
        if not prompt:
            print(f"[stage3] No prompt for {label} — skipping.")
            return None
        if out_file.exists():
            print(f"[stage3] Reusing {label} texture → {out_file}")
            _prompt_to_path[prompt] = str(out_file)
            return str(out_file)
        if prompt in _prompt_to_path:
            print(f"[stage3] {label} texture reused (same surface) from {_prompt_to_path[prompt]}")
            return _prompt_to_path[prompt]
        # Floors and ceilings are seen from a steep angle in the photo; passing
        # the photo as edit reference makes Qwen image-edit copy the perspective
        # crop verbatim instead of producing an orthographic tile.  Generate
        # those from text only so the synthesis_prompt's "flat top-down" wins.
        # Walls' near-front-on perspective is close enough to orthographic that
        # the reference still helps material fidelity.
        ref = None if surface in ("floor", "ceiling") else image_path
        print(f"[stage3] Generating {label} texture: \"{prompt[:70]}...\"  "
              f"(ref={'photo' if ref else 'text-only'})")
        try:
            img_bytes = generate_texture_image(
                prompt, reference_image_path=ref, surface=surface)
            # Auto-crop near-white/near-black padding the model sometimes adds
            import io
            from PIL import Image as _PILImg
            import numpy as _np
            _img = _PILImg.open(io.BytesIO(img_bytes)).convert("RGB")
            _arr = _np.array(_img)
            # Detect border rows/cols that are >90% near-white (>220) or near-black (<35)
            def _is_border_line(line):
                return bool(((line > 220).all(axis=-1).mean() > 0.90) or
                            ((line < 35).all(axis=-1).mean() > 0.90))
            top, bot = 0, _arr.shape[0] - 1
            left, right = 0, _arr.shape[1] - 1
            while top < bot and _is_border_line(_arr[top]):
                top += 1
            while bot > top and _is_border_line(_arr[bot]):
                bot -= 1
            while left < right and _is_border_line(_arr[:, left]):
                left += 1
            while right > left and _is_border_line(_arr[:, right]):
                right -= 1
            if top > 0 or bot < _arr.shape[0] - 1 or left > 0 or right < _arr.shape[1] - 1:
                _img = _img.crop((left, top, right + 1, bot + 1))
                print(f"[stage3] {label} texture cropped: "
                      f"{_arr.shape[1]}×{_arr.shape[0]} → {_img.width}×{_img.height}")
            _buf = io.BytesIO()
            _img.save(_buf, format="PNG")
            img_bytes = _buf.getvalue()
            out_file.write_bytes(img_bytes)
            _prompt_to_path[prompt] = str(out_file)
            print(f"[stage3] {label} texture saved → {out_file}")
            return str(out_file)
        except Exception as e:
            print(f"[stage3] ERROR generating {label} texture: {e}")
            return None

    def _gen_texture_with_retry(label: str, prompt: str, out_file: Path,
                                surface: str, max_retries: int = 3) -> "str | None":
        """Generate, then ask VLM to compare with the reference photo.  If the
        VLM judges the texture wrong, it returns a `revised_prompt`; we
        regenerate with that and recheck up to `max_retries` times.
        Returns the final texture path (best-effort) or None.

        Retry attempts write to a sibling .retry path so a Gemini failure on
        attempt N+1 doesn't destroy the validated result of attempt N.
        """
        cur_prompt = prompt
        last_path: "str | None" = None
        for attempt in range(1, max_retries + 1):
            if attempt == 1:
                target = out_file
            else:
                target = out_file.with_suffix(out_file.suffix + ".retry")
                try:
                    if target.exists():
                        target.unlink()
                except OSError:
                    pass
            print(f"[stage3] {label} attempt {attempt}/{max_retries}: "
                  f"\"{cur_prompt[:80]}…\"")
            gen_path = _gen_texture(label, cur_prompt, target, surface)
            if gen_path is None or not Path(gen_path).exists():
                print(f"[stage3] {label} attempt {attempt} failed — "
                      f"keeping previous result ({last_path})")
                return last_path
            if attempt > 1:
                try:
                    Path(gen_path).replace(out_file)
                    _prompt_to_path[cur_prompt] = str(out_file)
                except OSError as _e:
                    print(f"[stage3] {label} could not promote retry "
                          f"({_e}) — keeping previous result")
                    return last_path
            last_path = str(out_file)
            verdict = _vlm_verify_texture(out_file, image_path, surface, cur_prompt)
            if verdict is None:
                # VLM down or unparseable — keep the current texture
                return last_path
            ok      = bool(verdict.get("correct"))
            reason  = str(verdict.get("reasoning", ""))[:200]
            print(f"[stage3] {label} verify (attempt {attempt}): "
                  f"correct={ok} — {reason}")
            if ok:
                return last_path
            new_prompt = (verdict.get("revised_prompt") or "").strip()
            if not new_prompt or new_prompt == cur_prompt:
                # No useful refinement — stop retrying
                print(f"[stage3] {label} verify suggested no new prompt — "
                      f"keeping current texture")
                return last_path
            cur_prompt = new_prompt
        print(f"[stage3] {label}: hit max_retries={max_retries} — "
              f"keeping last attempt")
        return last_path

    # Wall + floor colours sampled from the photo — both get their real texture
    # from backprojection (texture_from_reference), so the floorplan/camera stage
    # skips text-to-image generation entirely and uses solid sampled placeholders
    # for render_final / fallback. Saves a diffusion call + VLM verify-retry loop
    # per surface, per scene.
    _wall_rgb = _sampled_wall_colour(image_path)
    _floor_rgb = _sampled_floor_colour(image_path)
    print(f"[stage3] wall texture: solid placeholder sampled from photo "
          f"{_wall_rgb} (t2i skipped — backprojection supplies real walls)")
    print(f"[stage3] floor texture: solid placeholder sampled from photo "
          f"{_floor_rgb} (t2i skipped — backprojection supplies the real floor)")

    # Floor: solid sampled placeholder (no t2i).
    floor_path = str(out / "floor_texture.png")
    if not (out / "floor_texture.png").exists():
        _solid_colour_texture("", out / "floor_texture.png", rgb=_floor_rgb)
    # Global wall fallback: solid sampled colour (no t2i).
    if not (out / "wall_texture.png").exists():
        _solid_colour_texture("", out / "wall_texture.png", rgb=_wall_rgb)

    # Per-wall textures
    walls_meta: dict = {}
    for wall in analysis.get("walls", []):
        orient  = wall.get("orientation")
        if not orient:
            continue
        surface = wall.get("surface", "")
        tex_file = out / f"wall_{orient}_texture.png"

        # Walls are textured by backprojection downstream, so skip text-to-image
        # for ALL walls (plain or patterned) and use the photo-sampled solid
        # placeholder. This is only the render_final / fallback texture.
        tex_path = (
            _solid_colour_texture(surface, tex_file, rgb=_wall_rgb)
            if not tex_file.exists()
            else str(tex_file)
        )
        walls_meta[orient] = {
            "orientation": orient,
            "length_m":    wall.get("length_m"),
            "height_m":    wall.get("height_m"),
            "surface":     surface,
            "features":    wall.get("features", []),
            "texture_path": tex_path,
            # Plain surfaces (flat painted walls etc.) have no pattern, so
            # tile boundaries would only produce subtle seams without adding
            # visual detail. Stretch the texture across the entire wall by
            # using the wall's longest dimension as the tile size.
            "tile_size_m":  (max(float(wall.get("length_m") or wall_tile_size),
                                  float(wall.get("height_m") or wall_tile_size))
                             if _is_plain_surface(surface)
                             else wall_tile_size),
        }

    # Ceiling: always plain white — never inherit the wall surface/pattern
    ceiling_surface = analysis.get("ceiling_texture", {}).get("surface", "") or \
                      "smooth matte white painted plaster, uniform surface, no pattern"
    ceiling_file = out / "ceiling_texture.png"
    if not ceiling_file.exists():
        _solid_colour_texture(ceiling_surface, ceiling_file)
    walls_meta["ceiling"] = {
        "texture_path": str(ceiling_file),
        "tile_size_m":  wall_tile_size,
    }
    walls_meta["floor"] = {
        "texture_path": floor_path,
        "tile_size_m":  floor_tile_size,
    }

    # Sanity sweep: any entry whose texture_path is null or points to a
    # nonexistent file gets a solid-colour fallback PNG. Without this,
    # render_room.py silently drops to a hardcoded RGB tuple and the
    # render looks "untextured" with no signal that anything failed.
    for _key, _meta in walls_meta.items():
        if not isinstance(_meta, dict):
            continue
        _tp = _meta.get("texture_path")
        if _tp and Path(_tp).exists():
            continue
        if _key == "floor":
            _fname = "floor_texture.png"
            _surface_for_color = floor_tex_info.get("surface", "")
        elif _key == "ceiling":
            _fname = "ceiling_texture.png"
            _surface_for_color = ceiling_surface
        else:
            _fname = f"wall_{_key}_texture.png"
            _surface_for_color = _meta.get("surface", "") or global_wall_tex.get("surface", "")
        _fp = out / _fname
        _solid_colour_texture(_surface_for_color, _fp)
        _meta["texture_path"] = str(_fp)
        print(f"[stage3] sanity sweep: filled missing {_key} texture → {_fp}")

    # Also ensure global fallbacks the renderer expects (wall_texture.png /
    # floor_texture.png) actually exist on disk; otherwise render_room
    # drops to flat RGB tuples for any orientation absent from walls_meta.
    for _gname, _gsurface in (
        ("wall_texture.png",  global_wall_tex.get("surface", "")),
        ("floor_texture.png", floor_tex_info.get("surface", "")),
    ):
        _gp = out / _gname
        if not _gp.exists():
            _solid_colour_texture(_gsurface, _gp)
            print(f"[stage3] sanity sweep: created global fallback → {_gp}")

    # Save
    walls_meta_path.write_text(json.dumps(walls_meta, indent=2))
    print(f"[stage3] ✓ walls_metadata.json → {walls_meta_path}")
    return dict(walls_metadata=walls_meta, walls_metadata_path=str(walls_meta_path))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: final render
# ─────────────────────────────────────────────────────────────────────────────

VLM_API_URL = "http://localhost:8080/v1/chat/completions"
from object_placement.vlm_backend import vlm_post as _vlm_post


def _encode_image_b64(path: str | Path) -> str:
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _vlm_assess_floor_texture(reference_path: str, render_path: str) -> dict:
    """Compare the rendered floor against the reference photo on three axes —
    rotation, plank/tile density, and colour tone — and return refinement
    signals plus VLM-authored Qwen image-edit prompts.

    Returned dict:
        rotation_deg     : int  — 0 / 90 / -90 / 180
        rotation_reason  : str
        density_ok       : bool
        density_prompt   : str  — Qwen edit prompt to densify (empty if ok)
        tone_ok          : bool
        tone_prompt      : str  — Qwen edit prompt to shift tone (empty if ok)
        all_ok           : bool — True iff nothing needs changing
    """
    import re, requests
    fallback = {
        "rotation_deg": 0, "rotation_reason": "",
        "density_ok": True, "density_prompt": "",
        "tone_ok": True, "tone_prompt": "",
        "all_ok": True,
    }
    try:
        prompt = (
            "Compare these two images, focusing ONLY on the FLOOR surface.\n\n"
            "Image 1: REFERENCE PHOTO of the room.\n"
            "Image 2: CURRENT 3D RENDER with a generated floor texture.\n\n"
            "Ignore furniture, walls, and lighting differences. Assess three things:\n\n"
            "1. ROTATION — does the floor pattern direction in Image 2 match Image 1?\n"
            "   rotation_correction_deg ∈ {0, 90, -90, 180}\n"
            "   (0 = correct, 90 = rotate render 90° CW to match, -90 = CCW, 180 = flip).\n\n"
            "2. DENSITY — does Image 2 show roughly the same number of plank/tile "
            "repeats per metre of floor as Image 1?\n"
            "   density_ok: true if matched.\n"
            "   density_prompt: if not matched, ONE short Qwen image-edit instruction "
            "telling the model how to refine the texture map "
            "(e.g. 'increase plank density to at least 15 narrow vertical planks "
            "across the image, tighter spacing'). Empty string if density is fine.\n\n"
            "3. TONE — does the colour palette of the floor in Image 2 match Image 1?\n"
            "   tone_ok: true if matched.\n"
            "   tone_prompt: if not matched, ONE short prompt describing the colour "
            "shift "
            "(e.g. 'shift palette to mixed warm brown, beige, and light grey tones, "
            "slightly desaturated, preserve the wood material'). Empty string if tone is fine.\n\n"
            "If the floor in either image is plain/textureless, set every *_ok to true and "
            "rotation_correction_deg to 0.\n\n"
            "Respond with ONLY a JSON object, no markdown:\n"
            '{"rotation_correction_deg": 0, "rotation_reason": "...",'
            ' "density_ok": true, "density_prompt": "",'
            ' "tone_ok": true, "tone_prompt": ""}'
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for p in (reference_path, render_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image_b64(p)}"},
            })
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return fallback
        data = json.loads(m.group())
        rot = int(data.get("rotation_correction_deg", 0))
        if rot not in (0, 90, -90, 180):
            rot = 0
        density_ok = bool(data.get("density_ok", True))
        tone_ok    = bool(data.get("tone_ok", True))
        density_prompt = "" if density_ok else str(data.get("density_prompt", ""))[:300]
        tone_prompt    = "" if tone_ok    else str(data.get("tone_prompt", ""))[:300]
        verdict = {
            "rotation_deg": rot,
            "rotation_reason": str(data.get("rotation_reason", ""))[:200],
            "density_ok": density_ok,
            "density_prompt": density_prompt,
            "tone_ok": tone_ok,
            "tone_prompt": tone_prompt,
            "all_ok": (rot == 0) and density_ok and tone_ok,
        }
        print(f"[stage4.5] VLM assessment: rotation={rot:+d}°  "
              f"density_ok={density_ok}  tone_ok={tone_ok}")
        if density_prompt:
            print(f"[stage4.5]   density: {density_prompt[:140]}")
        if tone_prompt:
            print(f"[stage4.5]   tone   : {tone_prompt[:140]}")
        return verdict
    except Exception as e:
        print(f"[stage4.5] VLM texture assessment failed: {e}")
        return fallback


def _refine_texture_via_edit(texture_path: Path, refinement_prompt: str,
                              cfg_scale: float = 4.5) -> bool:
    """Send the existing texture map back through Qwen image-edit with a
    refinement instruction and overwrite the file. Returns True on success.
    """
    import base64, requests
    from floorplan.agent.prompt_gen import IMG_GEN_URL
    if not texture_path.exists() or not refinement_prompt:
        return False
    framed = (
        f"{refinement_prompt} "
        "Keep this as a flat seamless tileable texture map. "
        "Pure overhead view, zero perspective, zero shadows, zero lighting, "
        "pure diffuse albedo only. Square output."
    )
    neg = (
        "3d render, perspective, depth, objects, furniture, shadows, lighting, "
        "shading, reflections, gloss, blur, distortion, artefacts"
    )
    try:
        ref_b64 = base64.b64encode(texture_path.read_bytes()).decode()
        from Qwen.image_edit_adapter import edit_image as _edit_image
        data = _edit_image({
            "prompt": framed,
            "negative_prompt": neg,
            "reference_image": ref_b64,
            "width": 1024, "height": 1024,
            "num_inference_steps": 30,
            "true_cfg_scale": cfg_scale,
            "num_images": 1,
        }, timeout=300)
        img_bytes = base64.b64decode(data["images"][0])
        texture_path.write_bytes(img_bytes)
        print(f"[refine] {texture_path.name} updated  ({len(img_bytes)//1024} KB)")
        return True
    except Exception as e:
        print(f"[refine] image-edit failed: {e}")
        return False


def _apply_floor_uv_rotation(wm_path: Path, rot_deg: int) -> bool:
    """Add rot_deg to floor.uv_rotation_deg in walls_metadata.json."""
    try:
        wm = json.loads(wm_path.read_text())
        floor_meta = wm.get("floor", {})
        prev = int(floor_meta.get("uv_rotation_deg") or 0)
        new_rot = (prev + rot_deg) % 360
        if new_rot > 180:
            new_rot -= 360
        floor_meta["uv_rotation_deg"] = new_rot
        wm["floor"] = floor_meta
        wm_path.write_text(json.dumps(wm, indent=2))
        print(f"[stage4.5] floor uv_rotation_deg: {prev}° → {new_rot}°")
        return True
    except Exception as e:
        print(f"[stage4.5] failed to apply floor rotation: {e}")
        return False


def _vlm_assess_lighting(reference_path: str, render_path: str) -> str:
    """Compare overall brightness of render vs reference photo.
    Returns one of: 'too_dark', 'too_bright', 'ok'. On any error, 'ok'
    so the loop terminates rather than thrashing.
    """
    import re, requests
    try:
        prompt = (
            "Compare the OVERALL BRIGHTNESS of these two images.\n\n"
            "Image 1: REFERENCE PHOTO of a real room.\n"
            "Image 2: 3D RENDER of the same empty room (no furniture).\n\n"
            "Ignore furniture, content, colour palette, and shadow direction. "
            "Only judge whether the render's overall light level matches the "
            "reference's overall light level.\n\n"
            "Reply with EXACTLY ONE of these tokens, no other text:\n"
            "  too_dark    — the render is noticeably darker than the reference\n"
            "  too_bright  — the render is noticeably brighter than the reference\n"
            "  ok          — the render's brightness is within ~10% of the reference\n"
        )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for p in (reference_path, render_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_image_b64(p)}"},
            })
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)
        raw = raw.strip().lower()
        for tok in ("too_dark", "too_bright", "ok"):
            if tok in raw:
                return tok
    except Exception as e:
        print(f"[stage4] VLM brightness check failed: {e}")
    return "ok"


def _run_stage4(out: Path, camera_path: str, image_path: str) -> str:
    """
    Render the room with the VGGT-aligned camera and Qwen-generated textures,
    iteratively adjusting light intensity under VLM supervision until the
    overall brightness matches the reference photo (or a budget is hit).

    Algorithm:
      - Start at intensity=0.7 (intentionally on the dark side of plausible)
      - Step +0.1 each iteration while VLM says 'too_dark'
      - On the first 'too_bright' (overshoot), halve the step and reverse
      - Continue contracting steps until VLM says 'ok' or max_iters hit
    """
    from floorplan.wall_line.render_room import render_room

    print("\n" + "═" * 60)
    print(" STAGE 4 — FINAL RENDER")
    print("═" * 60)

    obj_path    = out / "walls.obj"
    render_path = out / "render_final.png"

    if not obj_path.exists():
        print(f"[stage4] No walls.obj found — skipping render.")
        return ""

    intensity      = 0.7
    step           = 0.1
    last_verdict   = None
    max_iters      = 6
    history: list[tuple[float, str]] = []

    for it in range(1, max_iters + 1):
        try:
            render_room(
                mesh_path=str(obj_path),
                camera_json_path=camera_path,
                out_path=str(render_path),
                texture_dir=str(out),
                ref_image_path=image_path,
                ambient=0.4,
                light_intensity=intensity,
            )
        except Exception as e:
            print(f"[stage4] Render failed at intensity={intensity:.2f}: {e}")
            return ""

        verdict = _vlm_assess_lighting(image_path, str(render_path))
        history.append((round(intensity, 3), verdict))
        print(f"[stage4] iter {it}: intensity={intensity:.2f}  VLM={verdict}")

        if verdict == "ok":
            break

        if last_verdict is not None and last_verdict != verdict:
            step *= 0.5
        if verdict == "too_dark":
            intensity += step
        else:  # too_bright
            intensity -= step
        last_verdict = verdict

        intensity = max(0.2, min(2.5, intensity))
        if step < 0.01:
            break

    print(f"[stage4] ✓ Final render → {render_path}  (intensity={intensity:.2f})")
    print(f"[stage4]   lighting history: {history}")
    return str(render_path)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    output_dir: str,
    run_vggt_flag: bool = True,
    vggt_device: str | None = None,
    vggt_subsample: int = 4,
    vggt_n_iter: int = 5,
    skip_stages: set[str] | None = None,
    floor_rotation_override: int | None = None,
    texture_refine_iter: int = 2,
) -> dict:
    """
    Run the full HARMONY pipeline in four stages.

    Args:
        image_path     : Path to the reference indoor photo.
        output_dir     : Directory where all outputs are saved.
        run_vggt_flag  : If False, skip Stage 2 and use VLM camera for render.
        vggt_device    : torch device for VGGT ("cuda", "cpu", or None=auto).
        vggt_subsample : Point-cloud subsampling stride for Manhattan fitting.
        skip_stages    : Set of stage names to skip, e.g. {"stage3"}.

    Returns:
        Dict with keys: analysis, camera_path, camera_vggt_path,
                        walls_metadata_path, render_path.
    """
    skip = skip_stages or set()
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {"image_path": image_path, "output_dir": str(out)}

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if "stage1" not in skip:
        s1 = _run_stage1(image_path, out)
        results["analysis"]     = s1["analysis"]
        results["camera_path"]  = str(out / "camera.json")
    else:
        print("[pipeline] Skipping Stage 1 — loading cached analysis.")
        cache = out / "floorplan_analysis.json"
        results["analysis"] = json.loads(cache.read_text()) if cache.exists() else {}
        results["camera_path"] = str(out / "camera.json")
        s1 = {"analysis": results["analysis"]}

    analysis = results["analysis"]

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    import os as _os_s2
    if _os_s2.environ.get("SCENEWEAVE_NO_VGGT") == "1":
        run_vggt_flag = False   # ablation: VLM camera only, skip VGGT camera alignment
        print("[pipeline] SCENEWEAVE_NO_VGGT=1 — skipping Stage 2 (VGGT), using VLM camera.")
    if run_vggt_flag and "stage2" not in skip:
        cam_path = _run_stage2(
            image_path, out,
            vggt_subsample=vggt_subsample,
            vggt_device=vggt_device,
            vggt_n_iter=vggt_n_iter,
        )
    else:
        vggt_cam = out / "camera_vggt.json"
        cam_path = str(vggt_cam) if vggt_cam.exists() else str(out / "camera.json")
        print(f"[pipeline] Stage 2 (VGGT) skipped — using {Path(cam_path).name}.")
    results["camera_vggt_path"] = cam_path

    # ── Stage 2.3 — extend room depth if camera landed past front wall ────────
    # When floor_depth_m was 0 and the fallback (dist_to_back × 1.3) is too
    # shallow, VGGT calibration can place the camera past the room's front wall.
    # Rays looking sideways then see void beyond the side-wall edges.
    if "stage2.3" not in skip:
        try:
            from floorplan.post_refine_ceiling import extend_room_depth_to_camera
            extend_room_depth_to_camera(out, verbose=True)
        except Exception as _de:
            print(f"[stage2.3] depth extension skipped: {_de}")

    # ── Stage 2.5 — refine mesh ceiling height to VGGT ceiling target ────────
    # Stage 2's iterative loop locks the FLOOR anchor pixel exactly via the
    # orbit + pin steps, but the pin only enforces the floor↔ceiling SPAN.
    # If walls.obj's ceiling height differs from the H_room value used by
    # the pin, the rendered ceiling lands at the wrong pixel even though
    # the floor is correct.  This step solves analytically for the
    # ceiling_h that makes the back-wall top corner project to VGGT's
    # ceiling-anchor target pixel, then rewrites walls.obj +
    # floorplan_analysis.json + re-renders render_vggt.png + overlay.
    if "stage2.5" not in skip:
        try:
            from floorplan.post_refine_ceiling import refine_ceiling_to_vggt_target
            refine_ceiling_to_vggt_target(out, verbose=True)
        except Exception as _ce:
            print(f"[stage2.5] ceiling refinement skipped: {_ce}")

    # Sync sidecar room dimensions to the FINAL walls.obj.  Stage-2 alignment can
    # resize the mesh (especially WIDTH) without updating floorplan_analysis.json /
    # walls_metadata.json; downstream furniture placement reads floor_width_m and
    # would otherwise scale/ground against a stale size (objects float / mis-scale).
    try:
        from floorplan.post_refine_ceiling import sync_room_dims_to_mesh
        sync_room_dims_to_mesh(out, verbose=True)
    except Exception as _de:
        print(f"[dim_sync] room-dim sync skipped: {_de}")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    if "stage3" not in skip:
        s3 = _run_stage3(image_path, out, analysis)
        results["walls_metadata_path"] = s3.get("walls_metadata_path", "")
    else:
        print("[pipeline] Skipping Stage 3 — textures not regenerated.")

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    if "stage4" not in skip:
        render = _run_stage4(out, cam_path, image_path)
        results["render_path"] = render

        # ── Stage 4.5: VLM floor-texture refinement loop ────────────────────
        # Each iteration the VLM compares the render to the reference photo
        # and reports three signals at once: rotation, density, tone. Rotation
        # is applied via UV metadata; density/tone are applied by sending the
        # existing floor_texture.png back through Qwen image-edit with a
        # VLM-authored refinement prompt. Re-render after each iter, stop
        # when all checks pass or `texture_refine_iter` iterations are spent.
        # DISABLED: the floor is now a solid placeholder sampled from the photo
        # (the real floor texture comes from backprojection downstream), so there
        # is no generated floor pattern to rotate / density-match / tone-correct.
        # Skipping avoids a per-iter VLM assess + Qwen image-edit on the floor.
        if False and render and "stage4.5" not in skip:
            wm_path        = out / "walls_metadata.json"
            floor_tex_path = out / "floor_texture.png"

            if wm_path.exists() and Path(image_path).exists() and Path(render).exists():
                # Manual override: bypass VLM, just apply the rotation
                if floor_rotation_override is not None:
                    _rot = int(floor_rotation_override)
                    print(f"[stage4.5] floor rotation OVERRIDE: {_rot:+d}° "
                          f"(skipping VLM)")
                    if _rot in (90, -90, 180) and _apply_floor_uv_rotation(wm_path, _rot):
                        render = _run_stage4(out, cam_path, image_path)
                        results["render_path"] = render
                else:
                    for _i in range(max(0, int(texture_refine_iter))):
                        print(f"\n[stage4.5] ── Refinement iter {_i + 1}/"
                              f"{texture_refine_iter} ──")
                        verdict = _vlm_assess_floor_texture(image_path, render)
                        if verdict["all_ok"]:
                            print(f"[stage4.5] All checks passed — converged.")
                            break

                        changed = False
                        if verdict["rotation_deg"] in (90, -90, 180):
                            if _apply_floor_uv_rotation(wm_path, verdict["rotation_deg"]):
                                changed = True

                        # Combine density + tone into one image-edit call
                        edits = []
                        if not verdict["density_ok"] and verdict["density_prompt"]:
                            edits.append(verdict["density_prompt"])
                        if not verdict["tone_ok"] and verdict["tone_prompt"]:
                            edits.append(verdict["tone_prompt"])
                        if edits and floor_tex_path.exists():
                            if _refine_texture_via_edit(floor_tex_path, " ".join(edits)):
                                changed = True

                        if not changed:
                            print(f"[stage4.5] No applicable changes — stopping.")
                            break

                        print(f"[stage4.5] Re-rendering after refinement …")
                        render = _run_stage4(out, cam_path, image_path)
                        results["render_path"] = render
                        if not render:
                            break
    else:
        print("[pipeline] Skipping Stage 4 — render not produced.")

    print("\n" + "═" * 60)
    print(" PIPELINE COMPLETE")
    print("═" * 60)
    for k, v in results.items():
        if k not in ("analysis",):
            print(f"  {k}: {v}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Process animation
# ─────────────────────────────────────────────────────────────────────────────

def make_process_animation(
    output_dir: str | Path,
    gif_width: int = 960,
    frame_duration_ms: int = 1200,
    out_name: str = "process_animation.gif",
) -> str | None:
    """
    Collect all intermediate render PNGs from output_dir in pipeline order and
    assemble them into an animated GIF.

    Frame order:
      render_vlm → render_iter_N / render_overlay_N (sorted by N) →
      render_vggt → render_vggt_overlay →
      wall_mounted/placements/render_objects_placed →
      furniture/render_furniture_placed →
      render_final →
      manhattan_reference → manhattan_vggt (coda)

    Returns the path to the saved GIF, or None if fewer than 2 frames found.
    """
    from PIL import Image as _PIL

    out = Path(output_dir)

    # Fixed ordered frames (skip if file absent).
    # Manhattan-alignment diagnostic overlays are intentionally excluded —
    # they're rendered with a separate code path (`_draw_box_overlay`) whose
    # image-space scaling can produce inconsistent visuals vs. the actual
    # scene renders, even though the final calibrated camera is correct.
    _FIXED_LEAD = [
        "render_vlm.png",
    ]
    _FIXED_TAIL = [
        "render_vggt.png",
        "render_vggt_overlay.png",
        "wall_mounted/placements/render_objects_placed.png",
        "furniture/render_furniture_placed.png",
        "render_final.png",
    ]
    _FIXED_CODA: list[str] = [
        "manhattan_reference.png",   # clean Manhattan box on white
        "manhattan_vggt.png",        # same box, drawn over the reference photo
    ]

    # Collect iteration renders in numeric order: render_iter_0, render_overlay_0, ...
    iter_files: list[tuple[int, int, Path]] = []  # (iter_idx, sub, path)
    for p in out.glob("render_iter_*.png"):
        try:
            n = int(p.stem.split("_")[-1])
            iter_files.append((n, 0, p))
        except ValueError:
            pass
    for p in out.glob("render_overlay_*.png"):
        try:
            n = int(p.stem.split("_")[-1])
            iter_files.append((n, 1, p))
        except ValueError:
            pass
    iter_files.sort()

    frame_paths: list[Path] = []
    for name in _FIXED_LEAD:
        p = out / name
        if p.exists():
            frame_paths.append(p)
    for _, _, p in iter_files:
        frame_paths.append(p)
    for name in _FIXED_TAIL:
        p = out / name
        if p.exists():
            frame_paths.append(p)
    for name in _FIXED_CODA:
        p = out / name
        if p.exists():
            frame_paths.append(p)

    if len(frame_paths) < 2:
        print(f"[animate] Only {len(frame_paths)} frame(s) found — skipping GIF.")
        return None

    print(f"[animate] Building GIF from {len(frame_paths)} frames …")
    frames: list[_PIL.Image] = []
    for p in frame_paths:
        try:
            img = _PIL.open(p).convert("RGB")
            w, h = img.size
            new_h = int(h * gif_width / w)
            frames.append(img.resize((gif_width, new_h), _PIL.LANCZOS))
            print(f"  + {p.name}")
        except Exception as e:
            print(f"  ! {p.name}: {e}")

    if not frames:
        return None

    # All frames must be the same size — crop/pad to first frame's dimensions
    target_w, target_h = frames[0].size
    normed: list[_PIL.Image] = []
    for fr in frames:
        if fr.size != (target_w, target_h):
            canvas = _PIL.new("RGB", (target_w, target_h), (200, 200, 200))
            canvas.paste(fr.crop((0, 0, min(fr.width, target_w),
                                  min(fr.height, target_h))), (0, 0))
            normed.append(canvas)
        else:
            normed.append(fr)

    gif_path = out / out_name
    normed[0].save(
        str(gif_path),
        save_all=True,
        append_images=normed[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    print(f"[animate] Saved → {gif_path}  ({len(normed)} frames, {frame_duration_ms}ms/frame)")
    return str(gif_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="HARMONY full pipeline: VLM init → VGGT camera → textures → render")
    ap.add_argument("--image",   required=True, help="Reference indoor photo")
    ap.add_argument("--output",  required=True, help="Output directory")
    ap.add_argument("--no-vggt", action="store_true",
                    help="Skip Stage 2 (VGGT); use VLM camera directly")
    ap.add_argument("--device",  default=None,
                    help="Torch device for VGGT (e.g. 'cuda:0', 'cpu')")
    ap.add_argument("--vggt-subsample", type=int, default=4,
                    help="VGGT depth subsampling stride (default 4)")
    ap.add_argument("--vggt-n-iter", type=int, default=5,
                    help="VLM calibration iterations for Stage 2 (default 3)")
    ap.add_argument("--skip",    nargs="*", default=[],
                    help="Stage names to skip: stage1 stage2 stage3 stage4")
    ap.add_argument("--animate", action="store_true",
                    help="Assemble all intermediate renders into process_animation.gif")
    ap.add_argument("--animate-width", type=int, default=960,
                    help="GIF frame width in pixels (default 960)")
    ap.add_argument("--animate-duration", type=int, default=1200,
                    help="Milliseconds per frame in the GIF (default 1200)")
    ap.add_argument("--floor-rotation", type=int, default=None,
                    choices=[None, 0, 90, -90, 180],
                    help="Override floor UV rotation (degrees). If unset, VLM "
                         "auto-detects (often misses real rotation differences). "
                         "Use 90/-90/180 to force a rotation; 0 to force no rotation.")
    ap.add_argument("--texture-refine-iter", type=int, default=2,
                    help="Max Stage-4.5 refinement iterations (rotation + density "
                         "+ tone via VLM-authored Qwen image-edit prompts). "
                         "0 disables; default 2.")
    args = ap.parse_args()

    run_pipeline(
        image_path=args.image,
        output_dir=args.output,
        run_vggt_flag=not args.no_vggt,
        vggt_device=args.device,
        vggt_subsample=args.vggt_subsample,
        vggt_n_iter=args.vggt_n_iter,
        skip_stages=set(args.skip),
        floor_rotation_override=args.floor_rotation,
        texture_refine_iter=args.texture_refine_iter,
    )

    if args.animate:
        make_process_animation(
            output_dir=args.output,
            gif_width=args.animate_width,
            frame_duration_ms=args.animate_duration,
        )
