"""
camera_placement.py — compute a render camera for a rectangular room mesh.

Places the camera at the front of the room, aimed toward the deepest back-wall
corner.  The camera's horizontal position is solved so the corner projects to
the same pixel column as it does in the reference photograph.

Room coordinate system (matches build_room_mesh):
  X  0 → width_m    left  → right
  Y  0 → ceiling_h  floor → ceiling
  Z  0 → depth_m    back wall → front (camera side)

Output JSON:
  {
    "position_m":  [x, y, z],
    "look_at_m":   [x, y, z],
    "up":          [0, 1, 0],
    "hfov_deg":    float,
    "width_px":    int,
    "height_px":   int,
    "wall_context": { "back": {...}, "left": {...}, "right": {...}, "front": {...} }
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────

def _read_image_size(path: str) -> tuple[int, int]:
    """Return (width, height) of an image file. Falls back to (1280, 720)."""
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    return 1280, 720


def _fx(render_w: int, hfov_deg: float) -> float:
    return render_w / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))


def _floor_tilt_deg(eye_h: float, dist_to_back: float, vfov_deg: float) -> float:
    """
    Minimum downward tilt (negative degrees) so the floor appears at the bottom
    of the image.  Returns 0 if the floor is already in view with a level camera.

    Geometry:
      With a level camera the bottom ray is at -VFOV/2 below horizontal.
      Floor appears at the bottom pixel when:
          tan(VFOV/2 + |tilt|) >= eye_h / dist_to_back
      Solving for |tilt|:
          |tilt| >= arctan(eye_h / dist) - VFOV/2

    We apply 75 % of the minimum so the floor occupies roughly the bottom
    quarter of the image (natural interior framing).
    """
    min_tilt = np.degrees(np.arctan(eye_h / max(dist_to_back, 0.05))) - vfov_deg / 2
    if min_tilt > 0.5:
        # 1.3× gives a comfortable margin: floor visible at ~bottom 20% of frame.
        # Factor must be >1.0 to guarantee floor is actually in view.
        return -min_tilt * 1.3
    return 0.0


def estimate_eye_height(
    ref_image_path: str,
    Z_back: float,
    fx: float,
    render_h: int,
    ceiling_h: float,
    fallback_frac: float = 0.45,
) -> float:
    """
    Estimate camera eye height by finding the floor–wall junction row in the
    reference image, then back-projecting it to world-space metres.

    The floor meets the back wall at Y=0.  That junction projects to:
        py_floor = cy + fx * eye_h / Z_back
    → eye_h = (py_floor - cy) * Z_back / fx

    We locate py_floor by detecting strong horizontal edges in the lower half
    of the centre columns (to avoid confusion with side walls).

    Returns a clamped eye height in metres.
    """
    try:
        import cv2
        img = cv2.imread(str(ref_image_path))
        if img is None:
            raise ValueError(f"Cannot read {ref_image_path}")

        H, W = img.shape[:2]
        cy = H / 2.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Horizontal edge response (absolute vertical gradient)
        sobel = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5))

        # Focus on the centre third of columns (back wall is roughly centred)
        c_lo = W // 3
        c_hi = 2 * W // 3

        # Search in rows 35 %–90 % from top — floor junction is in lower portion
        r_lo = int(H * 0.35)
        r_hi = int(H * 0.90)

        region = sobel[r_lo:r_hi, c_lo:c_hi]
        row_strength = region.mean(axis=1)

        # Smooth to avoid noise peaks
        kernel = np.ones(max(1, H // 60)) / max(1, H // 60)
        row_strength_sm = np.convolve(row_strength, kernel, mode="same")

        peak_row = int(np.argmax(row_strength_sm)) + r_lo

        eye_h = (peak_row - cy) * Z_back / fx
        eye_h = float(np.clip(eye_h, 0.3, ceiling_h * 0.95))

        print(f"[camera] Floor junction detected at row {peak_row}/{H} "
              f"(cy={int(cy)}) → eye_h={eye_h:.3f}m  (Z_back={Z_back:.2f}m)")
        return eye_h

    except Exception as exc:
        fb = ceiling_h * fallback_frac
        print(f"[camera] Eye-height estimation failed ({exc}) "
              f"— fallback {fallback_frac:.0%} × {ceiling_h:.2f}m = {fb:.3f}m")
        return fb


# ─────────────────────────────────────────────────────────────────────────────

def compute_camera(
    width_m: float,
    depth_m: float,
    ceiling_h: float,
    ref_image_path: str | None = None,
    hfov_deg: float = 70.0,
    eye_height_m: float | None = None,
    margin_m: float = 0.05,
    cam_z_override: float | None = None,
    yaw_deg: float = 0.0,
    # ── deepest-corner pixel alignment ───────────────────────────────────────
    deepest_corner_x_m: float | None = None,
    deepest_corner_px: int | None = None,
    # ── optional VLM wall context (passed through to JSON) ───────────────────
    wall_context: dict | None = None,
) -> dict:
    """
    Return a camera dict for the rectangular room.

    Parameters
    ----------
    width_m, depth_m, ceiling_h
        Room dimensions.
    ref_image_path
        Original photograph — used to read render resolution (aspect ratio).
        Falls back to 1280×720.
    hfov_deg
        Horizontal FOV assumed during depth estimation (default 70°).
    eye_height_m
        Camera height above floor.  Defaults to 45 % of ceiling height.
    margin_m
        Pull the camera this far inside the front wall.
    deepest_corner_x_m
        World-space X of the deepest back-wall corner (0 = left, width_m = right).
        When provided together with deepest_corner_px the camera X is solved so
        that the corner projects to that pixel column.
    deepest_corner_px
        Pixel column of the deepest corner in the reference image.
    wall_context
        Dict with VLM wall metadata (features, orientation, etc.) — stored
        in the camera JSON for downstream use by the renderer.
    """
    # ── render resolution ─────────────────────────────────────────────────────
    if ref_image_path and Path(ref_image_path).exists():
        render_w, render_h = _read_image_size(ref_image_path)
        print(f"[camera] Render resolution from reference image: {render_w}×{render_h}px")
    else:
        render_w, render_h = 1280, 720
        print(f"[camera] Using fallback render resolution: {render_w}×{render_h}px")

    fx = _fx(render_w, hfov_deg)
    cx = render_w / 2.0
    fy = fx                         # square pixels assumption
    vfov = 2.0 * np.degrees(np.arctan(render_h / (2.0 * fy)))

    # ── eye height — estimated from reference image, else fallback ────────────
    if eye_height_m is None:
        # Z_back for this convention: camera is near depth_m, back wall is at Z=0.
        # Distance from camera to back wall = depth_m - margin_m.
        Z_back_est = float(depth_m) - margin_m
        if ref_image_path and Path(ref_image_path).exists() and Z_back_est > 0.1:
            eye_height_m = estimate_eye_height(
                ref_image_path=ref_image_path,
                Z_back=Z_back_est,
                fx=fx,
                render_h=render_h,
                ceiling_h=ceiling_h,
            )
        else:
            eye_height_m = ceiling_h * 0.45
    eye_height_m = float(np.clip(eye_height_m, 0.1, ceiling_h - 0.1))

    # cam_z is the camera's Z coordinate (back wall = 0, front = depth_m).
    # Use VLM scene-depth estimate (dist_to_back_m = cam_z directly) when provided;
    # otherwise fall back to placing the camera near the front wall.
    if cam_z_override is not None:
        cam_z = float(cam_z_override)
    else:
        cam_z = float(depth_m) - margin_m   # front of room, just inside

    # ── horizontal position + yaw ─────────────────────────────────────────────
    # Always centre the camera horizontally so both flanking walls are visible.
    # Then derive the required yaw from the corner_px constraint:
    #   tan(θ) = (r − k) / (1 + r*k)
    # where r = (corner_x_m − cam_x) / cam_z  (world-space direction to corner)
    #       k = (corner_px − cx) / fx          (desired NDC of corner in image)
    # This ensures the corner projects exactly to corner_px without moving cam_x.
    cam_x = width_m / 2.0

    if deepest_corner_x_m is not None and deepest_corner_px is not None:
        corner_x_used = float(deepest_corner_x_m)
        r = (corner_x_used - cam_x) / cam_z
        k = (float(deepest_corner_px) - cx) / fx
        denom = 1.0 + r * k
        if abs(denom) > 1e-6:
            yaw_rad = float(np.arctan2(r - k, denom))
        else:
            yaw_rad = np.radians(yaw_deg)

        # NOTE: a previous "if |yaw|>50° try opposite corner" guard was removed.
        # It was a band-aid for the VLM picking the wrong back_wall_side, but it
        # would silently invert correct answers on wide-angle photos and produce
        # an empty-back-wall render.  Loud failure beats silent flip; Stage 2
        # (VGGT Manhattan) re-derives the corner side from metric geometry.
        if abs(np.degrees(yaw_rad)) > 50:
            print(f"[camera] WARNING: corner solver returned extreme yaw "
                  f"{np.degrees(yaw_rad):+.1f}° at corner@{corner_x_used:.2f}m — "
                  f"VLM corner side may be wrong, or photo is unusually "
                  f"wide-angle.  Render will reflect this; inspect manhattan_vggt.png "
                  f"if available.")

        print(f"[camera] Corner-pixel yaw: corner@{corner_x_used:.2f}m "
              f"px={deepest_corner_px} cam_x={cam_x:.3f}m cam_z={cam_z:.3f}m "
              f"→ yaw={np.degrees(yaw_rad):+.1f}°")
    else:
        yaw_rad = np.radians(yaw_deg)

    cam_y = eye_height_m

    # look-at on back wall: cam_x + cam_z * tan(yaw)
    look_z = 0.0
    look_x = cam_x + cam_z * float(np.tan(yaw_rad))

    # Auto-tilt so the floor appears at the bottom of the frame.
    # Without this, the floor is often geometrically outside the FOV.
    auto_tilt = _floor_tilt_deg(eye_height_m, cam_z, vfov)
    if abs(auto_tilt) > 0.1:
        look_y = eye_height_m + cam_z * np.tan(np.radians(auto_tilt))
        print(f"[camera] Auto-tilt {auto_tilt:+.1f}° → look_at_y={look_y:.3f}m "
              f"(floor in bottom ~25% of frame)")
    else:
        look_y = eye_height_m   # floor already in view, keep level

    # ── diagnostics ───────────────────────────────────────────────────────────
    cam_dist = cam_z          # Z distance to back wall (which is at Z=0)
    visible_w = 2.0 * cam_dist * np.tan(np.radians(hfov_deg / 2.0))
    visible_h = 2.0 * cam_dist * np.tan(np.radians(vfov   / 2.0))
    print(
        f"[camera] Room      : {width_m:.2f}m W × {depth_m:.2f}m D × {ceiling_h:.2f}m H\n"
        f"[camera] Position  : ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})m\n"
        f"[camera] Look-at   : ({look_x:.3f}, {look_y:.3f}, {look_z:.3f})m\n"
        f"[camera] HFOV/VFOV : {hfov_deg:.1f}° / {vfov:.1f}°\n"
        f"[camera] Back-wall frustum @ {cam_dist:.2f}m : "
        f"{visible_w:.2f}m wide × {visible_h:.2f}m tall\n"
        f"[camera] Render    : {render_w}×{render_h}px"
    )
    if visible_w < width_m * 0.8:
        print(f"[camera] WARNING: back wall ({width_m:.1f}m) wider than frustum "
              f"({visible_w:.2f}m) — consider a wider hfov_deg")

    camera: dict = {
        "position_m":  [round(cam_x,  4), round(cam_y,  4), round(cam_z,  4)],
        "look_at_m":   [round(look_x, 4), round(look_y, 4), round(look_z, 4)],
        "up":          [0, 1, 0],
        "hfov_deg":    float(hfov_deg),
        "vfov_deg":    round(vfov, 4),
        "width_px":    int(render_w),
        "height_px":   int(render_h),
    }
    if wall_context:
        camera["wall_context"] = wall_context

    return camera


# ─────────────────────────────────────────────────────────────────────────────

def camera_from_mesh(
    mesh_path: str,
    ref_image_path: str | None = None,
    hfov_deg: float = 70.0,
    eye_height_frac: float = 0.45,
    out_path: str | None = None,
) -> dict:
    """
    Derive a camera purely from the OBJ bounding box — no VLM analysis needed.

    Works for both coordinate conventions:
      • depth-based mesh  : camera near Z=min, back wall at Z=max
      • build_room_mesh   : back wall at Z=0, front wall at Z=max
    In both cases the camera is placed just outside the minimum-Z face,
    looking toward the maximum-Z face.
    """
    try:
        import trimesh as _trimesh
    except ImportError as e:
        raise ImportError("camera_from_mesh requires 'trimesh': pip install trimesh") from e

    mesh = _trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, _trimesh.Scene):
        mesh = _trimesh.util.concatenate(mesh.dump())

    lo, hi = mesh.bounds       # (3,) min, (3,) max  — X, Y, Z
    width_m   = float(hi[0] - lo[0])
    ceiling_h = float(hi[1] - lo[1])   # Y range = floor-to-ceiling
    depth_m   = float(hi[2] - lo[2])   # Z range = room depth

    mid_x = float((lo[0] + hi[0]) / 2.0)

    print(f"[camera] Mesh bounds: X=[{lo[0]:.2f},{hi[0]:.2f}]  "
          f"Y=[{lo[1]:.2f},{hi[1]:.2f}]  Z=[{lo[2]:.2f},{hi[2]:.2f}]")
    print(f"[camera] Derived   : width={width_m:.2f}m  "
          f"ceiling={ceiling_h:.2f}m  depth={depth_m:.2f}m")

    # Read render resolution from ref image (or use the room aspect ratio)
    if ref_image_path and Path(ref_image_path).exists():
        render_w, render_h = _read_image_size(ref_image_path)
        print(f"[camera] Render resolution from reference: {render_w}×{render_h}px")
    else:
        render_w = max(1280, round(width_m * 256))
        render_h = max(720,  round(ceiling_h * 256))
        print(f"[camera] Render resolution from room dims: {render_w}×{render_h}px")

    fx   = _fx(render_w, hfov_deg)
    vfov = 2.0 * np.degrees(np.arctan(render_h / (2.0 * fx)))

    # ── camera convention + eye height ────────────────────────────────────────
    # Decide depth-based vs room-box before estimating eye height (need Z_back).
    #   depth-based: lo[2] >> 0  (walls sit at large positive Z, camera at Z=0)
    #   room-box:    lo[2] ≈ 0   (back wall at Z=0, front at Z=depth_m)
    is_depth_based = lo[2] > depth_m * 0.3

    if is_depth_based:
        Z_back_for_eye = float(hi[2])      # back wall is at max Z, camera at Z=0
    else:
        Z_back_for_eye = depth_m - 0.05   # camera at hi[2]-0.05, back wall at lo[2]≈0

    # Estimate eye height from reference image, fall back to fraction of ceiling
    if ref_image_path and Path(ref_image_path).exists():
        eye_h = estimate_eye_height(
            ref_image_path=ref_image_path,
            Z_back=Z_back_for_eye,
            fx=fx,
            render_h=render_h,
            ceiling_h=ceiling_h,
            fallback_frac=eye_height_frac,
        )
    else:
        eye_h = float(lo[1] + ceiling_h * eye_height_frac)

    if is_depth_based:
        # Original camera was at world origin
        cam_pos = [0.0, eye_h, 0.0]

        # Look-at: deepest back-left vertex
        verts_np = mesh.vertices
        far_mask  = verts_np[:, 2] >= float(hi[2]) - 0.05
        far_verts = verts_np[far_mask] if far_mask.any() else verts_np
        corner_v  = far_verts[np.argmin(far_verts[:, 0])]
        dist_back = float(corner_v[2])   # camera is at Z=0
        auto_tilt = _floor_tilt_deg(eye_h, dist_back, vfov)
        look_y = eye_h + dist_back * np.tan(np.radians(auto_tilt)) if abs(auto_tilt) > 0.1 else eye_h
        look_at = [float(corner_v[0]), look_y, float(corner_v[2])]
        print(f"[camera] Deepest back-left corner : "
              f"({corner_v[0]:.3f}, {corner_v[2]:.3f})m  "
              f"auto-tilt={auto_tilt:+.1f}°")
    else:
        # Room-box: camera inside near the front wall, looking at back wall
        cam_pos = [mid_x, eye_h, float(hi[2]) - 0.05]
        dist_back = float(hi[2]) - 0.05   # distance to back wall (at lo[2]≈0)
        auto_tilt = _floor_tilt_deg(eye_h, dist_back, vfov)
        look_y = eye_h + dist_back * np.tan(np.radians(auto_tilt)) if abs(auto_tilt) > 0.1 else eye_h
        look_at = [mid_x, look_y, float(lo[2])]
        if abs(auto_tilt) > 0.1:
            print(f"[camera] Auto-tilt {auto_tilt:+.1f}° → look_at_y={look_y:.3f}m")

    camera: dict = {
        "position_m":  [round(v, 4) for v in cam_pos],
        "look_at_m":   [round(v, 4) for v in look_at],
        "up":          [0, 1, 0],
        "hfov_deg":    float(hfov_deg),
        "vfov_deg":    round(vfov, 4),
        "width_px":    int(render_w),
        "height_px":   int(render_h),
    }

    # Show projected pixel of the deepest corner for cross-checking
    if is_depth_based:
        cx_px = render_w / 2.0
        corner_zc = float(corner_v[2]) - cam_pos[2]   # Z in camera space
        corner_xc = float(corner_v[0]) - cam_pos[0]   # X in camera space
        if corner_zc > 0:
            proj_px = round(cx_px + fx * corner_xc / corner_zc)
            print(f"[camera] Deepest corner projects to px={proj_px}  "
                  f"(image width={render_w}, centre={int(cx_px)}) "
                  f"— compare with reference image")

    print(f"[camera] Position : {cam_pos}\n"
          f"[camera] Look-at  : {look_at}\n"
          f"[camera] HFOV/VFOV: {hfov_deg:.1f}° / {vfov:.1f}°")

    if out_path:
        save_camera(camera, out_path)
    return camera


# ─────────────────────────────────────────────────────────────────────────────

def save_camera(camera: dict, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(camera, indent=2), encoding="utf-8")
    print(f"[camera] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────

def camera_from_analysis(
    analysis: dict,
    ref_image_path: str | None = None,
    hfov_deg: float = 70.0,
    out_path: str | None = None,
) -> dict:
    """
    Derive the Stage-1 camera from a VLM floorplan analysis dict.

    The VLM's corner_px / back_wall_side hint gives the deepest back-wall
    corner; its pixel column pixel-aligns the camera horizontally.  Stage 2
    (VGGT Manhattan) then refines this camera against metric geometry.

    Parameters
    ----------
    analysis
        Output of prompt_gen.analyze_floorplan.
    """
    room = analysis.get("room", {})
    raw_width = room.get("floor_width_m")
    raw_depth = room.get("floor_depth_m")
    raw_ceil  = room.get("ceiling_height_m")
    width_m   = float(raw_width)  if raw_width  else 5.0
    depth_m   = float(raw_depth)  if raw_depth  else 4.0
    ceiling_h = float(raw_ceil)   if raw_ceil   else 2.4
    if not raw_width or float(raw_width) <= 0.1:
        print(f"[camera] WARNING: VLM did not provide floor_width_m "
              f"(got {raw_width!r}) — using default 5.0m.  Render scale will "
              f"be approximate; consider re-running with VGGT to override.")
    if not raw_depth or float(raw_depth) <= 0.1:
        print(f"[camera] WARNING: VLM did not provide floor_depth_m "
              f"(got {raw_depth!r}) — using default 4.0m.  This is the most "
              f"common cause of camera-too-close-to-back-wall renders.")
    if not raw_ceil or float(raw_ceil) <= 0.1:
        print(f"[camera] WARNING: VLM did not provide ceiling_height_m "
              f"(got {raw_ceil!r}) — using default 2.4m.")

    # ── VLM camera estimates ──────────────────────────────────────────────────
    vlm_cam    = analysis.get("camera", {})
    vlm_h      = vlm_cam.get("height_m")
    vlm_tilt   = float(vlm_cam.get("tilt_deg",  0.0))
    vlm_yaw    = float(vlm_cam.get("yaw_deg",   0.0))
    vlm_dist   = vlm_cam.get("dist_to_back_m")       # metres from camera to back wall
    vlm_facing = vlm_cam.get("facing", "back")

    if vlm_h:
        vlm_h = float(np.clip(vlm_h, 0.3, ceiling_h * 0.95))
        print(f"[camera] VLM estimate — height={vlm_h:.2f}m  "
              f"tilt={vlm_tilt:+.1f}°  yaw={vlm_yaw:+.1f}°  facing={vlm_facing!r}")
    else:
        vlm_h = None

    # dist_to_back_m → camera Z coordinate (back wall is Z=0, camera near Z=depth_m)
    # Enforce a minimum: camera must be at least 75% of room depth from the back wall
    # so it sits in the front portion of the room for a realistic interior perspective.
    cam_z_min = depth_m * 0.75
    cam_z_override: float | None = None
    if vlm_dist and float(vlm_dist) > 0.4:
        raw = float(vlm_dist)
        clamped = float(np.clip(raw, cam_z_min, depth_m - 0.1))
        cam_z_override = clamped
        if clamped != raw:
            print(f"[camera] VLM dist_to_back={raw:.2f}m → cam_z={clamped:.2f}m "
                  f"(raised to 75% min {cam_z_min:.2f}m)")
        else:
            print(f"[camera] VLM dist_to_back={raw:.2f}m → cam_z={cam_z_override:.2f}m")
    else:
        # No VLM distance: default to front of room
        cam_z_override = None  # compute_camera will use depth_m - margin_m

    # ── VLM corner hint → deepest back-wall corner pixel + world X ───────────
    # The VLM's corner_px gives the pixel column of the deepest visible
    # back-wall corner and back_wall_side says which junction it is:
    #   back wall RIGHT of the corner → back-LEFT  junction → world X = 0
    #   back wall LEFT  of the corner → back-RIGHT junction → world X = width_m
    # back_wall_side is known to flip occasionally; Stage 2 (VGGT Manhattan)
    # re-derives the anchor side from metric geometry and overrides it.
    vlm_corner_px      = vlm_cam.get("corner_px")
    vlm_back_wall_side = vlm_cam.get("back_wall_side", "")   # "left" or "right"
    if vlm_back_wall_side == "right":
        vlm_corner_x_m: float | None = 0.0
    elif vlm_back_wall_side == "left":
        vlm_corner_x_m = width_m
    else:
        vlm_corner_x_m = None

    deepest_corner_x_m: float | None = None
    deepest_corner_px:  int   | None = None
    # Tracks how the corner anchor was obtained:
    #   "vlm"     → VLM corner_px + back_wall_side
    #   "facing"  → no pixel data at all, VLM facing direction (lowest confidence)
    corner_anchor_source: str = "facing"

    if vlm_corner_px is not None:
        deepest_corner_px  = int(vlm_corner_px)
        deepest_corner_x_m = vlm_corner_x_m
        if deepest_corner_x_m is None:
            deepest_corner_x_m = 0.0 if "left" in vlm_facing else width_m
        corner_anchor_source = "vlm"
        print(f"[camera] VLM corner: px={deepest_corner_px}  "
              f"world_x={deepest_corner_x_m:.2f}m  "
              f"(back_wall_side='{vlm_back_wall_side}')")

    # ── fallback: no pixel data — use VLM facing direction ──────────────────
    if deepest_corner_px is None and deepest_corner_x_m is None:
        facing_x_frac = {
            "back":        0.5,
            "back_left":   0.0,
            "back_right":  1.0,
            "left":        0.0,
            "right":       1.0,
        }.get(vlm_facing, 0.5)
        deepest_corner_x_m = width_m * facing_x_frac
        print(f"[camera] No corner pixel — VLM facing '{vlm_facing}' "
              f"→ corner_x={deepest_corner_x_m:.2f}m")

    # ── wall context from VLM walls list ─────────────────────────────────────
    wall_context: dict = {}
    for w in analysis.get("walls", []):
        orient = w.get("orientation")
        if orient:
            wall_context[orient] = {
                "length_m": w.get("length_m"),
                "height_m": w.get("height_m"),
                "surface":  w.get("surface"),
                "features": w.get("features", []),
            }

    camera = compute_camera(
        width_m=width_m,
        depth_m=depth_m,
        ceiling_h=ceiling_h,
        ref_image_path=ref_image_path,
        hfov_deg=hfov_deg,
        eye_height_m=vlm_h,           # VLM height takes priority; None → auto-estimate
        cam_z_override=cam_z_override,
        yaw_deg=vlm_yaw,              # yaw baked into cam_x + look_x simultaneously
        deepest_corner_x_m=deepest_corner_x_m,
        deepest_corner_px=deepest_corner_px,
        wall_context=wall_context or None,
    )

    # ── VLM aim target override + look_at sanity clamp ────────────────────────
    # The VLM's aim_target landmark (e.g. "centre of back-wall window") is
    # hallucination-prone (we have seen the VLM put a window on the wrong wall,
    # then aim at it), so it is only accepted after the plausibility checks
    # below.  Stage 2 (VGGT Manhattan) re-aims the camera against metric
    # geometry afterwards.
    aim_world = vlm_cam.get("aim_target_world_m")
    aim_label = str(vlm_cam.get("aim_target_landmark", "")).strip()
    aim_target_used = False
    if isinstance(aim_world, (list, tuple)) and len(aim_world) == 3:
        try:
            ax, ay, az = (float(v) for v in aim_world)
            in_room = (0.0 <= ax <= width_m + 0.05
                       and 0.0 <= ay <= ceiling_h + 0.05
                       and 0.0 <= az <= depth_m + 0.05)
            # Plausibility: a back-wall / back-corner landmark must have Z
            # closer to the BACK wall than to the camera.  If Z > cam_z * 0.6,
            # the VLM almost certainly mis-applied the convention (writing
            # Z ≈ depth_m instead of Z ≈ 0 for a back-wall feature).
            cam_z_for_check = float(camera["position_m"][2])
            z_implausible = (az > cam_z_for_check * 0.6
                             and ("back" in aim_label.lower()
                                  or "centre" in aim_label.lower()
                                  or "deep" in aim_label.lower()))
            if in_room and not z_implausible:
                camera["look_at_m"] = [round(ax, 4), round(ay, 4), round(az, 4)]
                aim_target_used = True
                print(f"[camera] VLM aim_target override → "
                      f"look_at=({ax:.2f},{ay:.2f},{az:.2f})  "
                      f"landmark='{aim_label or '(unspecified)'}'")
            elif z_implausible:
                # Auto-correct: assume the VLM swapped Z=0 ↔ Z=depth.  Mirror.
                az_flipped = float(np.clip(depth_m - az, 0.0, depth_m))
                camera["look_at_m"] = [round(ax, 4), round(ay, 4),
                                       round(az_flipped, 4)]
                aim_target_used = True
                print(f"[camera] VLM aim_target Z-flip auto-correct → "
                      f"({ax:.2f},{ay:.2f},{az:.2f}) → "
                      f"({ax:.2f},{ay:.2f},{az_flipped:.2f})  "
                      f"(landmark says back/deepest but Z was near front; "
                      f"flipped Z = depth_m − VLM_Z)")
            elif not in_room:
                print(f"[camera] VLM aim_target {aim_world} outside room "
                      f"({width_m:.1f}×{ceiling_h:.1f}×{depth_m:.1f}) — ignoring")
        except (TypeError, ValueError):
            pass

    # Final sanity clamp: even after the override, the existing yaw-based
    # `look_x = cam_x + cam_z * tan(yaw)` can extrapolate outside the room
    # for very large yaw values.  Clamp X to a small slack past the wall so
    # the look-at *direction* is preserved but the numeric coords stay sane.
    cam_pos_clamp = camera["position_m"]
    look_clamp    = camera["look_at_m"]
    _slack = 0.30   # 30 cm slack past walls — direction info preserved, no
                    # 9-metre extrapolation past a 5-metre wall.
    look_clamp[0] = float(np.clip(look_clamp[0], -_slack, width_m + _slack))
    look_clamp[2] = float(np.clip(look_clamp[2], -_slack, depth_m + _slack))
    if (look_clamp[0] != camera["look_at_m"][0]
            or look_clamp[2] != camera["look_at_m"][2]):
        print(f"[camera] look_at sanity clamp → ({look_clamp[0]:.2f},"
              f"{look_clamp[1]:.2f},{look_clamp[2]:.2f})")
    camera["look_at_m"] = [round(v, 4) for v in look_clamp]

    # ── vertical tilt: derive exact look_at_y from floor-junction fraction ──────
    # Priority:
    #   1. floor_junction_y_frac  — exact formula, most reliable
    #   2. visible_wall_height_m  — constrains top of frame
    #   3. tilt_deg               — fallback direct estimate
    pos   = np.array(camera["position_m"])
    la    = np.array(camera["look_at_m"])
    cam_x = float(pos[0])
    cam_y = float(pos[1])
    cam_z = float(pos[2])
    vfov  = float(camera.get("vfov_deg", 70.0))

    final_tilt_deg: float | None = None
    tilt_source = "none"

    # ── Path 1: floor_junction_y_frac ────────────────────────────────────────
    # The floor-wall junction at the back wall (Y=0) projects to fraction f
    # from the image top.  Exact closed-form tilt using VLM metric cam_z only:
    #   t      = (2f - 1) · tan(vfov/2)
    #   θ      = arctan2(t·cam_z − cam_y,  cam_z + t·cam_y)
    #   look_y = cam_y + cam_z · tan(θ)
    # Marigold depth is NOT used here — cam_z is the VLM perpendicular distance.
    vlm_frac = vlm_cam.get("floor_junction_y_frac")
    if vlm_frac is not None:
        f = float(np.clip(vlm_frac, 0.05, 0.95))
        t = (2.0 * f - 1.0) * np.tan(np.radians(vfov / 2.0))
        denom = cam_z + t * cam_y
        if abs(denom) > 1e-4 and cam_z > 0.01:
            theta = float(np.arctan2(t * cam_z - cam_y, denom))
            final_tilt_deg = float(np.degrees(theta))
            tilt_source = f"floor_junction_y_frac={f:.2f}"
            print(f"[camera] Junction-fraction tilt: f={f:.2f}  t={t:.3f}  "
                  f"cam_z={cam_z:.2f}m  → tilt={final_tilt_deg:+.1f}°")

    # ── Path 2: visible_wall_height_m ─────────────────────────────────────────
    if final_tilt_deg is None:
        vlm_vis_wall = vlm_cam.get("visible_wall_height_m")
        if vlm_vis_wall and float(vlm_vis_wall) > 0.1 and cam_z > 0.01:
            wall_top_y = min(float(vlm_vis_wall), ceiling_h)
            vfov_half  = np.radians(vfov / 2.0)
            angle_top  = np.arctan2(wall_top_y - cam_y, cam_z)
            tilt_frame = float(np.degrees(angle_top - vfov_half))
            tilt_frame = float(np.clip(tilt_frame, -45.0, 30.0))
            final_tilt_deg = tilt_frame
            tilt_source = f"visible_wall_height_m={vlm_vis_wall:.2f}"
            print(f"[camera] Visible-wall tilt: top_y={wall_top_y:.2f}m "
                  f"→ tilt={final_tilt_deg:+.1f}°")

    # ── Path 3: tilt_deg fallback ─────────────────────────────────────────────
    if final_tilt_deg is None and abs(vlm_tilt) > 0.5:
        final_tilt_deg = vlm_tilt
        tilt_source = "tilt_deg"

    # ── Ceiling / floor frame-coverage clamp ─────────────────────────────────
    # If the reference shows both ceiling and floor, the tilt must keep both
    # within the render's FOV.  All angles use VLM metric cam_z (no Marigold).
    if final_tilt_deg is not None and cam_z > 0.01:
        vlm_vis_wall  = vlm_cam.get("visible_wall_height_m")
        vlm_vis_floor = vlm_cam.get("visible_floor_depth_m")
        vlm_junc_frac = vlm_cam.get("floor_junction_y_frac")

        # Decide which boundaries to enforce. The VLM frequently overstates
        # visible_wall_height_m for downward-tilted shots (claims ceiling is in
        # frame when it isn't). Cross-check against floor_junction_y_frac:
        # a junction landing in the lower half (>=0.55) implies the camera is
        # tilted down enough that the ceiling is out of frame, regardless of
        # what visible_wall_height_m claims. Tightened the height threshold
        # from 0.80 to 0.95 for the same reason.
        ceiling_visible = (
            vlm_vis_wall is not None
            and float(vlm_vis_wall) >= ceiling_h * 0.95
            and (vlm_junc_frac is None or float(vlm_junc_frac) < 0.55)
        )
        floor_visible = vlm_vis_floor is not None and float(vlm_vis_floor) > 0.1

        # Minimum tilt so ceiling (at Y=ceiling_h) hits the TOP edge of the frame:
        #   top_ray angle = tilt + vfov/2  ≥  arctan((ceiling_h - cam_y)/cam_z)
        if ceiling_visible:
            ceiling_angle = float(np.degrees(np.arctan2(ceiling_h - cam_y, cam_z)))
            tilt_min = ceiling_angle - vfov / 2.0
        else:
            tilt_min = -89.0

        # Maximum tilt so floor (at Y=0) hits the BOTTOM edge of the frame:
        #   bottom_ray angle = tilt - vfov/2  ≤  arctan(-cam_y/cam_z)
        if floor_visible:
            floor_angle = float(np.degrees(np.arctan2(-cam_y, cam_z)))
            tilt_max = floor_angle + vfov / 2.0
        else:
            tilt_max = 89.0

        if tilt_min <= tilt_max:
            clamped = float(np.clip(final_tilt_deg, tilt_min, tilt_max))
            if abs(clamped - final_tilt_deg) > 0.2:
                print(f"[camera] Tilt clamped from {final_tilt_deg:+.1f}° "
                      f"to {clamped:+.1f}° "
                      f"(ceiling/floor range [{tilt_min:+.1f}°, {tilt_max:+.1f}°])")
            final_tilt_deg = clamped

    if final_tilt_deg is not None:
        tilt_rad = np.radians(final_tilt_deg)
        # look_at Y: camera height + cam_z (perpendicular to back wall) * tan(tilt)
        la[1] = float(pos[1]) + cam_z * float(np.tan(tilt_rad))
        camera["look_at_m"] = [round(float(v), 4) for v in la.tolist()]
        print(f"[camera] Tilt → look_at_y={la[1]:.3f}m  "
              f"({tilt_source}  {final_tilt_deg:+.1f}°)")

    # ── Re-enforce horizontal corner alignment after every look_at_y update ──
    # In two-point perspective the horizontal projection of a vertical world line
    # is independent of tilt (right-vector has no Y component, zc uses fwd_h).
    # We re-snap look_at_x explicitly after each tilt stage so the corner stays
    # locked to deepest_corner_px and rounding errors don't accumulate.
    #
    # SKIPPED when the aim-target override fired earlier — that override is
    # an explicit 3D landmark that takes priority over corner-pixel snapping.
    if (deepest_corner_x_m is not None and deepest_corner_px is not None
            and not aim_target_used):
        render_w = int(camera.get("width_px", 1280))
        cx_img   = render_w / 2.0
        fx_img   = _fx(render_w, hfov_deg)
        r  = (deepest_corner_x_m - cam_x) / cam_z
        k  = (float(deepest_corner_px) - cx_img) / fx_img
        denom = 1.0 + r * k
        if abs(denom) > 1e-6:
            snapped_x = cam_x + cam_z * (r - k) / denom
            if abs(snapped_x - la[0]) > 5e-4:
                print(f"[camera] Corner x re-snap: look_at_x {la[0]:.4f} → {snapped_x:.4f}")
            la[0] = snapped_x
            camera["look_at_m"] = [round(float(v), 4) for v in la.tolist()]
    elif aim_target_used:
        print(f"[camera] Corner x re-snap SKIPPED (aim_target_used=True)")

    # Final hard clamp — re-snap and aim_target overrides can place look_at outside
    # room bounds (e.g. cam_z≈0 from zero floor_depth_m blows up the re-snap formula).
    # This is the last line of defence before saving.
    _final_slack = 0.30
    _la_final = camera["look_at_m"]
    _la_x_clamped = float(np.clip(_la_final[0], -_final_slack, width_m + _final_slack))
    _la_z_clamped = float(np.clip(_la_final[2], -_final_slack, depth_m + _final_slack))
    if abs(_la_x_clamped - _la_final[0]) > 1e-3 or abs(_la_z_clamped - _la_final[2]) > 1e-3:
        print(f"[camera] Final look_at clamp: "
              f"({_la_final[0]:.3f}, {_la_final[1]:.3f}, {_la_final[2]:.3f}) → "
              f"({_la_x_clamped:.3f}, {_la_final[1]:.3f}, {_la_z_clamped:.3f})  "
              f"(room {width_m:.2f}×{depth_m:.2f}m)")
        camera["look_at_m"] = [round(_la_x_clamped, 4),
                                round(float(_la_final[1]), 4),
                                round(_la_z_clamped, 4)]

    # Yaw sanity check: if the horizontal angle between camera and look_at is
    # implausibly large (>40°) after all corrections, the corner assignment was
    # wrong (e.g. back-left corner mapped to a right-side pixel).  Fall back to
    # looking straight at the back wall center so stage-2 has a reasonable start.
    _cam_pos_xy  = camera["position_m"]
    _look_xy     = camera["look_at_m"]
    _dx = float(_look_xy[0]) - float(_cam_pos_xy[0])
    _dz = float(_look_xy[2]) - float(_cam_pos_xy[2])
    _yaw_deg = float(np.degrees(np.arctan2(abs(_dx), max(abs(_dz), 1e-6))))
    if _yaw_deg > 40.0:
        _center_x = width_m / 2.0
        print(f"[camera] Yaw sanity reset: computed yaw={_yaw_deg:.1f}° > 40° — "
              f"reverting look_at_x {_look_xy[0]:.3f} → {_center_x:.3f} (room center)")
        camera["look_at_m"][0] = round(_center_x, 4)

    if out_path:
        save_camera(camera, out_path)
    return camera


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute render camera for a rectangular room.")
    parser.add_argument("--width",      type=float, required=True,
                        help="Room width in metres")
    parser.add_argument("--depth",      type=float, required=True,
                        help="Room depth in metres")
    parser.add_argument("--ceiling",    type=float, default=2.4,
                        help="Ceiling height in metres")
    parser.add_argument("--ref",        type=str,   default=None,
                        help="Reference image path (for render resolution)")
    parser.add_argument("--hfov",       type=float, default=70.0,
                        help="Horizontal FOV in degrees")
    parser.add_argument("--eye",        type=float, default=None,
                        help="Eye height in metres (default: 45%% of ceiling)")
    parser.add_argument("--corner_px",  type=int,   default=None,
                        help="Pixel column of the deepest back-wall corner in the ref image")
    parser.add_argument("--corner_x_m", type=float, default=None,
                        help="World X of that corner (0=left wall, width_m=right wall)")
    parser.add_argument("--out",        type=str,   default="camera.json",
                        help="Output JSON path")
    args = parser.parse_args()

    cam = compute_camera(
        width_m=args.width,
        depth_m=args.depth,
        ceiling_h=args.ceiling,
        ref_image_path=args.ref,
        hfov_deg=args.hfov,
        eye_height_m=args.eye,
        deepest_corner_x_m=args.corner_x_m,
        deepest_corner_px=args.corner_px,
    )
    save_camera(cam, args.out)