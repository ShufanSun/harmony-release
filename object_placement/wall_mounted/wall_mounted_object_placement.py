"""
wall_mounted_object_placement.py — place wall-mounted GLB objects in 3D space.

For each entry in wall_mounted/wall_objects.json:
  1. Call the VLM with the reference image + object description to estimate:
       • which wall the object is on  ("back", "left", "right")
       • real-world size  [width_m, height_m, depth_m]
  2. Back-project the pixel_rect centre through the pinhole camera model onto
     the wall plane → 3D attachment point on the wall surface.
  3. Offset outward by depth/2 so the back face sits flush on the wall.
  4. Compute a rotation matrix that aligns the object's +Z front face with the
     wall's inward normal.
  5. Compute per-axis GLB scale factors: target_size / glb_bounds_size.
  6. Write wall_mounted/wall_mounted_placements.json.

Room coordinate system  (matches build_room_mesh / camera_placement.py):
  X  0 → width_m    left  → right
  Y  0 → ceiling_h  floor → ceiling
  Z  0 → depth_m    back wall → front (camera side)

Wall plane equations:
  back   Z = 0          inward normal [0,  0, +1]
  left   X = 0          inward normal [+1, 0,  0]
  right  X = width_m    inward normal [-1, 0,  0]

GLB convention assumed: object's front face points in the +Z direction.
The rotation below maps +Z → wall inward normal.

Usage:
    python -m object_placement.wall_mounted.wall_mounted_object_placement \\
        --output-dir outputs/<timestamp> \\
        --image data/indoor_images/office5.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np

VLM_API_URL = "http://localhost:8080/v1/chat/completions"

# Extra overlap added to each side of the window frame so it fully covers
# the wall opening edges with no visible gap.  5 cm per side = 10 cm total.
WINDOW_FRAME_OVERLAP_M = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers (mirrors render_room.py _camera_axes + projection)
# ─────────────────────────────────────────────────────────────────────────────

def _camera_axes(
    pos: np.ndarray, look_at: np.ndarray, up: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (right, up_c, fwd) orthonormal camera basis.

    Convention: camera looks in +fwd direction.
      right = normalize(fwd × up)
      up_c  = normalize(right × fwd)
    """
    fwd = look_at - pos
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

    right = np.cross(fwd, up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        rn = np.linalg.norm(right)
    right /= rn

    up_c = np.cross(right, fwd)
    up_c /= np.linalg.norm(up_c)
    return right, up_c, fwd


def _backproject_pixel(
    px: float,
    py: float,
    cam_pos: np.ndarray,
    right: np.ndarray,
    up_c: np.ndarray,
    fwd: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Return a unit direction vector in world space for pixel (px, py).

    Projection convention:
      px = cx + fx * xc / zc
      py = cy - fy * yc / zc   (screen Y increases downward)
    """
    xc_n = (px - cx) / fx
    yc_n = -(py - cy) / fy   # flip screen Y
    direction = xc_n * right + yc_n * up_c + fwd
    n = np.linalg.norm(direction)
    return direction / n if n > 1e-9 else direction


def _ray_wall_intersect(
    cam_pos: np.ndarray,
    direction: np.ndarray,
    wall: str,
    width_m: float,
    depth_m: float,
) -> np.ndarray | None:
    """Intersect ray (cam_pos + t*direction) with a named axis-aligned wall plane.

    Returns the 3D world-space intersection point, or None if the ray is parallel
    or points away from the wall.
    """
    eps = 1e-9
    if wall == "back":          # Z = 0
        denom = direction[2]
        if abs(denom) < eps:
            return None
        t = (0.0 - cam_pos[2]) / denom
    elif wall == "left":        # X = 0
        denom = direction[0]
        if abs(denom) < eps:
            return None
        t = (0.0 - cam_pos[0]) / denom
    elif wall == "right":       # X = width_m
        denom = direction[0]
        if abs(denom) < eps:
            return None
        t = (width_m - cam_pos[0]) / denom
    elif wall == "front":       # Z = depth_m
        denom = direction[2]
        if abs(denom) < eps:
            return None
        t = (depth_m - cam_pos[2]) / denom
    else:
        return None

    if t < 0.0:
        return None

    return cam_pos + t * direction


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_for_wall(wall: str) -> list[list[float]]:
    """
    Return a 3×3 rotation matrix (row-major list) that maps the GLB object's
    +Z axis (assumed front face) to the wall's inward normal.

    Wall inward normals:
      back   [0,  0, +1]   → +Z already aligned → identity
      left   [+1, 0,  0]   → Ry(+90°)
      right  [-1, 0,  0]   → Ry(-90°)
      front  [0,  0, -1]   → Ry(180°)
    """
    if wall == "back":
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif wall == "left":
        # Ry(+90°): maps +Z → +X
        return [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    elif wall == "right":
        # Ry(-90°): maps +Z → -X
        return [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    elif wall == "front":
        # Ry(180°): maps +Z → -Z
        return [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    else:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _wall_inward_normal(wall: str) -> np.ndarray:
    """Unit vector pointing from the wall surface into the room interior."""
    normals = {
        "back":  np.array([0.0,  0.0,  1.0]),
        "left":  np.array([1.0,  0.0,  0.0]),
        "right": np.array([-1.0, 0.0,  0.0]),
        "front": np.array([0.0,  0.0, -1.0]),
    }
    return normals.get(wall, np.array([0.0, 0.0, 1.0]))


# ─────────────────────────────────────────────────────────────────────────────
# GLB bounding-box utilities
# ─────────────────────────────────────────────────────────────────────────────

def _glb_bounds(glb_path: Path) -> np.ndarray | None:
    """Return (2, 3) array [min, max] of the GLB mesh bounds, or None on error."""
    try:
        import trimesh
        mesh = trimesh.load(str(glb_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh.bounds  # (2, 3)
    except Exception as e:
        print(f"[placement] Warning: could not load GLB {glb_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# VLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_image(path: str | Path) -> tuple[str, str]:
    """Return (base64_data, mime_type)."""
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _parse_json_response(raw: str) -> dict | list:
    text = _strip_thinking(raw)
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    json_str = match.group(1).strip() if match else text.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = json_str.rstrip().rstrip(",")
        json_str += "]" * (json_str.count("[") - json_str.count("]"))
        json_str += "}" * (json_str.count("{") - json_str.count("}"))
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[placement] Warning: could not parse JSON: {e}")
            return {}


_SIZE_AND_WALL_PROMPT = """\
You are analyzing a wall-mounted object in an interior room photograph.

ROOM DIMENSIONS
  Width  : {width_m:.2f} m   (left wall to right wall)
  Depth  : {depth_m:.2f} m   (back wall to front/camera)
  Ceiling: {ceiling_h:.2f} m

OBJECT DESCRIPTION
  Type       : {obj_type}
  Description: {description}
  Pixel rect : left={left}  top={top}  right={right}  bottom={bottom}
               (image size {img_w}×{img_h} px, origin top-left)

TASKS
-----
1. WALL IDENTIFICATION
   Which of the room's walls does this object appear to be mounted on?
   Options: "back" (the far wall, facing the camera), "left" (left side wall),
            "right" (right side wall).
   Decide based on its position in the image and visual perspective cues.

2. REAL-WORLD SIZE ESTIMATION
   Estimate the object's actual physical dimensions in metres:
     width_m  — horizontal extent (left to right when facing the wall)
     height_m — vertical extent (top to bottom)
     depth_m  — how far it protrudes from the wall surface (thickness)

   Use these reference sizes if you are unsure:
     window    : width 0.6–1.5 m, height 0.8–2.0 m, depth 0.05–0.15 m
     shelf     : width 0.4–2.0 m, height 0.02–0.05 m, depth 0.2–0.4 m
     art/photo : width 0.2–1.0 m, height 0.2–1.0 m, depth 0.02–0.05 m
     mirror    : width 0.3–1.2 m, height 0.3–1.5 m, depth 0.02–0.05 m
     tv        : width 0.5–1.8 m, height 0.3–1.1 m, depth 0.05–0.15 m
     light     : width 0.1–0.4 m, height 0.2–0.6 m, depth 0.1–0.3 m
     door      : width 0.7–1.0 m, height 1.9–2.2 m, depth 0.04–0.1 m

OUTPUT — valid JSON object only, no extra text:
{{
  "wall": "back|left|right",
  "width_m": <float>,
  "height_m": <float>,
  "depth_m": <float>
}}
"""


def estimate_size_and_wall(
    image_path: str | Path,
    obj: dict,
    room: dict,
    img_w: int,
    img_h: int,
) -> dict:
    """
    Call the VLM to estimate which wall an object is on and its real-world size.
    Returns a dict with keys: wall, width_m, height_m, depth_m.
    Falls back to sensible defaults on error.
    """
    import requests

    left, top, right, bottom = obj["pixel_rect"]
    prompt = _SIZE_AND_WALL_PROMPT.format(
        width_m=room["floor_width_m"],
        depth_m=room["floor_depth_m"],
        ceiling_h=room["ceiling_height_m"],
        obj_type=obj["type"],
        description=obj.get("description", ""),
        left=left, top=top, right=right, bottom=bottom,
        img_w=img_w, img_h=img_h,
    )

    b64, mime = _encode_image(image_path)
    payload = {
        "model": "qwen3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = requests.post(VLM_API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        result = _parse_json_response(raw)
        if isinstance(result, dict) and "wall" in result:
            # Validate and clamp
            result["wall"] = result.get("wall", "back").lower().strip()
            if result["wall"] not in ("back", "left", "right", "front"):
                result["wall"] = "back"
            result["width_m"]  = float(result.get("width_m",  0.5))
            result["height_m"] = float(result.get("height_m", 0.5))
            result["depth_m"]  = float(result.get("depth_m",  0.05))
            print(
                f"[placement] VLM → {obj['type']:12s}: "
                f"wall={result['wall']:5s}  "
                f"size={result['width_m']:.2f}×{result['height_m']:.2f}×{result['depth_m']:.3f} m"
            )
            return result
    except Exception as e:
        print(f"[placement] VLM call failed for {obj.get('type','?')}: {e}")

    # Fallback defaults per object type
    defaults = {
        "window": dict(wall="back",  width_m=1.0, height_m=1.5, depth_m=0.08),
        "door":   dict(wall="back",  width_m=0.9, height_m=2.1, depth_m=0.06),
        "shelf":  dict(wall="left",  width_m=1.0, height_m=0.03, depth_m=0.3),
        "art":    dict(wall="back",  width_m=0.5, height_m=0.5, depth_m=0.03),
        "mirror": dict(wall="back",  width_m=0.6, height_m=0.9, depth_m=0.03),
        "tv":     dict(wall="back",  width_m=1.2, height_m=0.7, depth_m=0.08),
        "light":  dict(wall="left",  width_m=0.2, height_m=0.4, depth_m=0.15),
    }
    fb = defaults.get(obj.get("type", "other"), dict(wall="back", width_m=0.4, height_m=0.4, depth_m=0.05))
    print(f"[placement] Using fallback size for {obj.get('type','?')}: {fb}")
    return fb


# ─────────────────────────────────────────────────────────────────────────────
# Main placement logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_placements(
    output_dir: Path,
    image_path: Path,
) -> list[dict]:
    """
    Compute 3D placements for all wall-mounted objects.

    Returns a list of placement dicts, one per object, with keys:
      object_index, type, glb_path,
      position_m [x, y, z],
      rotation_3x3 (3×3 rotation matrix as nested list),
      scale [sx, sy, sz],
      wall, size_m {width_m, height_m, depth_m}
    """
    wall_mounted_dir = output_dir / "wall_mounted"
    objects_dir      = wall_mounted_dir / "objects"

    # ── Load inputs ──────────────────────────────────────────────────────────
    wall_objects_path = wall_mounted_dir / "wall_objects.json"
    if not wall_objects_path.exists():
        # Fall back to segment_results.json — same data, different schema
        seg_results_path = wall_mounted_dir / "segment_results.json"
        if not seg_results_path.exists():
            raise FileNotFoundError(
                f"wall_objects.json not found: {wall_objects_path}\n"
                f"Also tried segment_results.json — neither exists."
            )
        print(f"[placement] wall_objects.json not found — deriving from segment_results.json")
        seg_data = json.loads(seg_results_path.read_text())
        wall_objects = [
            {
                "type":        seg["type"],
                "pixel_rect":  seg["box_px"],
                "output_file": str(wall_mounted_dir / seg["glb_file"].lstrip("/")),
                **{k: seg[k] for k in ("phrase", "gdino_score") if k in seg},
            }
            for seg in seg_data.get("segments", [])
            if "glb_file" in seg
        ]
        print(f"[placement] Derived {len(wall_objects)} objects from segment_results.json")
    else:
        wall_objects = json.loads(wall_objects_path.read_text())

    camera_path = output_dir / "camera.json"
    if not camera_path.exists():
        raise FileNotFoundError(f"camera.json not found: {camera_path}")
    camera: dict = json.loads(camera_path.read_text())

    floorplan_path = output_dir / "floorplan_analysis.json"
    if not floorplan_path.exists():
        raise FileNotFoundError(f"floorplan_analysis.json not found: {floorplan_path}")
    floorplan: dict = json.loads(floorplan_path.read_text())
    room: dict = floorplan["room"]

    width_m    = float(room["floor_width_m"])
    depth_m    = float(room["floor_depth_m"])
    ceiling_h  = float(room["ceiling_height_m"])
    img_w: int = int(camera["width_px"])
    img_h: int = int(camera["height_px"])
    hfov_deg   = float(camera["hfov_deg"])

    # ── Camera parameters ────────────────────────────────────────────────────
    cam_pos  = np.array(camera["position_m"],  dtype=np.float64)
    look_at  = np.array(camera["look_at_m"],   dtype=np.float64)
    up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)

    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)

    fx = img_w / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))
    fy = fx       # square pixels
    cx = img_w / 2.0
    cy = img_h / 2.0

    # ── Load structural openings (ground-truth window positions) ─────────────
    openings: list[dict] = []
    openings_path = output_dir / "openings" / "openings.json"
    if openings_path.exists():
        openings = json.loads(openings_path.read_text())
        print(f"[placement] Loaded {len(openings)} structural openings from openings.json")

    placements: list[dict] = []

    for idx, obj in enumerate(wall_objects):
        obj_type = obj.get("type", "unknown")
        if obj_type != "window":
            print(f"[placement] Skipping [{idx}] {obj_type} (only windows are placed)")
            continue

        glb_name = Path(obj.get("output_file", "")).stem + ".glb"
        glb_path = objects_dir / glb_name
        if not glb_path.exists():
            candidates = sorted(objects_dir.glob(f"object_{idx:02d}_*.glb"))
            if candidates:
                glb_path = candidates[0]
            else:
                print(f"[placement] GLB not found for {obj_type} (index {idx}) — skipping")
                continue

        print(f"\n[placement] Processing [{idx}] {obj_type}  pixel_rect={obj['pixel_rect']}")

        # ── 1. Try structural opening first ─────────────────────────────────
        # openings.json gives the exact hole geometry — use it directly for
        # windows so the frame is embedded precisely in the carved opening.
        window_opening = next(
            (o for o in openings if o.get("type") == "window"), None
        )

        if window_opening is not None:
            wall          = window_opening.get("wall", "back")
            opening_w     = float(window_opening["width_m"])
            opening_h     = float(window_opening["height_m"])
            depth_obj     = 0.1   # frame thickness — frame straddles the wall surface

            # Scale the frame slightly larger than the hole so its edges overlap
            # the surrounding wall, leaving no visible gap.
            width_obj  = opening_w  + 2.0 * WINDOW_FRAME_OVERLAP_M
            height_obj = opening_h + 2.0 * WINDOW_FRAME_OVERLAP_M

            # Centre of the opening on the wall surface (frame centre = hole centre)
            cx_wall = float(window_opening["offset_from_left_m"]) + opening_w / 2.0
            cy_wall = float(window_opening.get("sill_height_m", 0.0)) + opening_h / 2.0

            # Map wall-surface (cx_wall, cy_wall) to world XYZ
            if wall == "back":
                wall_pt = np.array([cx_wall, cy_wall, 0.0])
            elif wall == "left":
                wall_pt = np.array([0.0, cy_wall, cx_wall])
            elif wall == "right":
                wall_pt = np.array([width_m, cy_wall, depth_m - cx_wall])
            else:   # front
                wall_pt = np.array([cx_wall, cy_wall, depth_m])

            # Place frame centred on the hole; straddle the wall (Z offset = 0)
            inward_normal = _wall_inward_normal(wall)
            position = wall_pt   # centre of frame coincides with centre of hole
            print(
                f"[placement]   Structural opening: {opening_w:.2f}×{opening_h:.2f}m  "
                f"frame (with {WINDOW_FRAME_OVERLAP_M*100:.0f}cm overlap): "
                f"{width_obj:.2f}×{height_obj:.2f}m  wall={wall}  centre={position}"
            )
        else:
            # ── Fallback: VLM estimate + back-projection ─────────────────────
            print("[placement]   No structural opening found — falling back to VLM")
            vlm_result = estimate_size_and_wall(image_path, obj, room, img_w, img_h)
            wall       = vlm_result["wall"]
            width_obj  = vlm_result["width_m"]
            height_obj = vlm_result["height_m"]
            depth_obj  = vlm_result["depth_m"]

            left, top, right_px, bottom = obj["pixel_rect"]
            cx_px = (left + right_px) / 2.0
            cy_px = (top  + bottom)   / 2.0

            ray_dir = _backproject_pixel(
                cx_px, cy_px,
                cam_pos, right_v, up_c_v, fwd_v,
                fx, fy, cx, cy,
            )
            world_pt = _ray_wall_intersect(cam_pos, ray_dir, wall, width_m, depth_m)
            if world_pt is None:
                t_fallback = np.linalg.norm(look_at - cam_pos)
                world_pt = cam_pos + ray_dir * t_fallback

            world_pt[1] = float(np.clip(world_pt[1], 0.0, ceiling_h))
            if wall == "back":
                world_pt[0] = float(np.clip(world_pt[0], 0.0, width_m))
                world_pt[2] = 0.0
            elif wall == "left":
                world_pt[2] = float(np.clip(world_pt[2], 0.0, depth_m))
                world_pt[0] = 0.0
            elif wall == "right":
                world_pt[2] = float(np.clip(world_pt[2], 0.0, depth_m))
                world_pt[0] = width_m

            inward_normal = _wall_inward_normal(wall)
            position = world_pt + inward_normal * (depth_obj / 2.0)

        # ── 2. Rotation aligned to wall ───────────────────────────────────────
        rotation = _rotation_for_wall(wall)

        # ── 3. Scale GLB to match opening size ───────────────────────────────
        bounds = _glb_bounds(glb_path)
        if bounds is not None:
            glb_size = bounds[1] - bounds[0]
            glb_size = np.maximum(glb_size, 1e-6)
            target_size = np.array([width_obj, height_obj, depth_obj])
            scale = (target_size / glb_size).tolist()
        else:
            scale = [width_obj, height_obj, depth_obj]

        print(f"[placement]   Scale {scale}")

        placements.append({
            "object_index": idx,
            "type":         obj_type,
            "description":  obj.get("description", ""),
            "glb_path":     str(glb_path),
            "wall":         wall,
            "size_m": {
                "width_m":  width_obj,
                "height_m": height_obj,
                "depth_m":  depth_obj,
            },
            "position_m":   position.tolist(),
            "rotation_3x3": rotation,
            "scale":        scale,
        })

    return placements


# ─────────────────────────────────────────────────────────────────────────────
# Renderer: composite placed objects onto the base room render
# ─────────────────────────────────────────────────────────────────────────────

NEAR_CLIP = 0.05   # metres


def _rasterize_tri(
    buf: np.ndarray,
    pts2d: np.ndarray,    # (3, 2)
    rgb: np.ndarray,      # (3,) float [0, 255]
    light: float,
) -> None:
    """Fill one triangle into buf using barycentric rasterization."""
    H, W = buf.shape[:2]
    xmin = max(0,   int(np.floor(pts2d[:, 0].min())))
    xmax = min(W-1, int(np.ceil (pts2d[:, 0].max())))
    ymin = max(0,   int(np.floor(pts2d[:, 1].min())))
    ymax = min(H-1, int(np.ceil (pts2d[:, 1].max())))
    if xmin > xmax or ymin > ymax:
        return

    p0, p1, p2 = pts2d[0], pts2d[1], pts2d[2]
    v0 = p1 - p0
    v1 = p2 - p0
    denom = float(v0[0] * v1[1] - v0[1] * v1[0])
    if abs(denom) < 0.5:
        return

    ys, xs = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    qx = xs.ravel().astype(np.float32) - p0[0]
    qy = ys.ravel().astype(np.float32) - p0[1]
    s = (qx * v1[1] - qy * v1[0]) / denom
    t = (qy * v0[0] - qx * v0[1]) / denom
    mask = (s >= 0) & (t >= 0) & (s + t <= 1.0)
    if not mask.any():
        return

    color = np.clip(rgb * light, 0, 255).astype(np.uint8)
    pxi = xs.ravel()[mask].astype(np.int32)
    pyi = ys.ravel()[mask].astype(np.int32)
    buf[pyi, pxi] = color


def _project_vertex(
    v: np.ndarray,   # (3,) world space
    cam_pos: np.ndarray,
    right_v: np.ndarray,
    up_c_v: np.ndarray,
    fwd_v: np.ndarray,
    fx: float, cx: float, cy: float,
) -> tuple[float, float, float]:
    """Return (px, py, zc) for one world-space vertex."""
    d  = v - cam_pos
    xc = float(np.dot(d, right_v))
    yc = float(np.dot(d, up_c_v))
    zc = float(np.dot(d, fwd_v))
    if zc <= NEAR_CLIP:
        return float("nan"), float("nan"), zc
    px = cx + fx * xc / zc
    py = cy - fx * yc / zc
    return px, py, zc


def _mesh_face_color(mesh, face_idx: int) -> np.ndarray:
    """Return an (3,) float RGB array [0,255] for a face.

    Tries vertex colours → visual material base_color → grey fallback.
    """
    try:
        import trimesh
        # Vertex colours
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            vc = mesh.visual.vertex_colors
            if vc.shape[0] == len(mesh.vertices):
                tri = mesh.faces[face_idx]
                avg = vc[tri, :3].mean(axis=0).astype(np.float32)
                return avg
        # Material base colour
        if hasattr(mesh.visual, "material"):
            mat = mesh.visual.material
            bc = getattr(mat, "baseColorFactor", None) or getattr(mat, "diffuse", None)
            if bc is not None:
                bc = np.array(bc[:3], dtype=np.float32)
                # Values in [0,1] → scale to [0,255]
                if bc.max() <= 1.0:
                    bc = bc * 255.0
                return bc
    except Exception:
        pass
    return np.array([180.0, 180.0, 180.0])   # neutral grey fallback


def _collect_placement_faces(
    p: dict,
    cam_pos: np.ndarray,
    right_v: np.ndarray,
    up_c_v: np.ndarray,
    fwd_v: np.ndarray,
    fx: float, cx: float, cy: float,
    light_dir: np.ndarray,
    ambient: float,
) -> list[tuple[float, np.ndarray, np.ndarray, float]]:
    """Load, transform, project, and shade one GLB placement.  Returns face data list."""
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("trimesh required") from e

    glb_path = Path(p["glb_path"])
    if not glb_path.exists():
        print(f"[render_objects] GLB not found: {glb_path} — skipping")
        return []
    try:
        mesh = trimesh.load(str(glb_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
    except Exception as e:
        print(f"[render_objects] Failed to load {glb_path}: {e}")
        return []

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    verts  = mesh.vertices.astype(np.float64) - center[np.newaxis, :]
    verts  = verts * np.array(p["scale"], dtype=np.float64)[np.newaxis, :]
    verts  = verts @ np.array(p["rotation_3x3"], dtype=np.float64).T
    verts  = verts + np.array(p["position_m"], dtype=np.float64)[np.newaxis, :]

    faces = mesh.faces
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = np.where(norms > 1e-12, cross / norms, cross)

    print(f"[render_objects] {p['type']:12s}: {len(faces)} faces  pos={np.round(verts.mean(0),3).tolist()}")

    face_data: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for fi in range(len(faces)):
        tri_verts = verts[faces[fi]]
        fn = face_normals[fi]

        to_cam_n = cam_pos - tri_verts.mean(axis=0)
        to_cam_n /= np.linalg.norm(to_cam_n) + 1e-9
        if np.dot(fn, to_cam_n) < 0.0:
            continue

        pts2d, zcs, valid = [], [], True
        for v in tri_verts:
            px, py, zc = _project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
            if zc <= NEAR_CLIP:
                valid = False; break
            pts2d.append([px, py]); zcs.append(zc)
        if not valid:
            continue

        light_factor = float(ambient + (1.0 - ambient) * max(0.0, np.dot(fn, light_dir)))
        face_data.append((float(np.mean(zcs)), np.array(pts2d, dtype=np.float32),
                          _mesh_face_color(mesh, fi), light_factor))
    return face_data


def render_placements(
    output_dir: Path,
    placements: list[dict],
    camera: dict,
    animate: bool = False,
    animate_gif_width: int = 960,
    animate_frame_ms: int = 1000,
) -> Path:
    """
    Composite all placed objects onto the best available base render.

    Base render priority:
      1. openings/render_with_glass.png
      2. openings/render_openings.png
      3. render.png

    Saves result to wall_mounted/render_with_objects.png and returns that path.

    If animate=True, also saves wall_mounted/placement_animation.gif showing
    objects added one at a time to the scene.
    """
    try:
        import trimesh
        from PIL import Image
    except ImportError as e:
        raise ImportError("render_placements requires trimesh and Pillow") from e

    # ── Pick base render ──────────────────────────────────────────────────────
    candidates = [
        output_dir / "openings" / "render_with_glass.png",
        output_dir / "openings" / "render_openings.png",
        output_dir / "render_final.png",
        output_dir / "render.png",
    ]
    base_path = next((p for p in candidates if p.exists()), None)
    if base_path is None:
        raise FileNotFoundError(
            f"No base render found in {output_dir} — "
            "expected render_with_glass.png, render_openings.png, or render.png"
        )
    print(f"[render_objects] Base render: {base_path}")

    base_img = Image.open(base_path).convert("RGB")
    base_np  = np.array(base_img, dtype=np.uint8)
    H, W = base_np.shape[:2]

    # ── Camera ────────────────────────────────────────────────────────────────
    cam_pos  = np.array(camera["position_m"],  dtype=np.float64)
    look_at  = np.array(camera["look_at_m"],   dtype=np.float64)
    up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)

    hfov_deg = float(camera["hfov_deg"])
    fx = W / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    _light_dir = np.array([0.4, 0.8, 0.4], dtype=np.float64)
    _light_dir /= np.linalg.norm(_light_dir)
    AMBIENT = 0.45

    # ── Collect face data per placement ──────────────────────────────────────
    per_placement: list[list[tuple]] = []
    for p in placements:
        faces = _collect_placement_faces(
            p, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy, _light_dir, AMBIENT)
        per_placement.append(faces)

    # ── Final composite (all objects) ────────────────────────────────────────
    all_face_data = [fd for group in per_placement for fd in group]
    all_face_data.sort(key=lambda x: -x[0])
    print(f"[render_objects] Rasterizing {len(all_face_data)} visible faces…")

    buf = base_np.copy()
    for _, pts2d_arr, face_rgb, light_factor in all_face_data:
        _rasterize_tri(buf, pts2d_arr, face_rgb, light_factor)

    out_path = output_dir / "wall_mounted" / "render_with_objects.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(buf).save(str(out_path))
    print(f"[render_objects] Saved → {out_path}")

    # ── Animation: one frame per added object ────────────────────────────────
    if animate:
        gif_frames: list[Image.Image] = []
        gif_w = animate_gif_width
        gif_h = int(H * gif_w / W)

        # Frame 0: empty room
        gif_frames.append(Image.fromarray(base_np).resize((gif_w, gif_h), Image.LANCZOS))
        print(f"[animate] Frame 0: base render")

        accumulated: list[tuple] = []
        for i, group in enumerate(per_placement):
            if not group:
                continue
            accumulated.extend(group)
            sorted_acc = sorted(accumulated, key=lambda x: -x[0])
            frame_buf = base_np.copy()
            for _, pts2d_arr, face_rgb, light_factor in sorted_acc:
                _rasterize_tri(frame_buf, pts2d_arr, face_rgb, light_factor)
            ptype = placements[i].get("type", f"obj{i}")
            print(f"[animate] Frame {i+1}: +{ptype}")
            gif_frames.append(Image.fromarray(frame_buf).resize((gif_w, gif_h), Image.LANCZOS))

        if len(gif_frames) >= 2:
            gif_path = output_dir / "wall_mounted" / "placement_animation.gif"
            gif_frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=gif_frames[1:],
                duration=animate_frame_ms,
                loop=0,
            )
            print(f"[animate] Saved → {gif_path}  ({len(gif_frames)} frames)")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# OBJ exporter: merge wall mesh + placed objects into one combined OBJ
# ─────────────────────────────────────────────────────────────────────────────

def export_combined_obj(output_dir: Path, placements: list[dict]) -> Path:
    """
    Load the best available wall mesh, append transformed placement meshes,
    and save the combined geometry as wall_mounted/walls_with_objects.obj.

    Wall mesh priority:
      1. openings/walls_with_openings.obj  (has window/door holes)
      2. walls.obj

    Returns the output path.
    """
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("export_combined_obj requires trimesh") from e

    # ── Pick wall mesh ────────────────────────────────────────────────────────
    wall_candidates = [
        output_dir / "openings" / "walls_with_openings.obj",
        output_dir / "walls.obj",
    ]
    wall_path = next((p for p in wall_candidates if p.exists()), None)
    if wall_path is None:
        raise FileNotFoundError(f"No wall mesh found in {output_dir}")
    print(f"[export_obj] Wall mesh: {wall_path}")

    wall_mesh = trimesh.load(str(wall_path), force="mesh")
    if isinstance(wall_mesh, trimesh.Scene):
        wall_mesh = trimesh.util.concatenate(wall_mesh.dump())

    meshes = [wall_mesh]

    for p in placements:
        glb_path = Path(p["glb_path"])
        if not glb_path.exists():
            print(f"[export_obj] GLB not found: {glb_path} — skipping")
            continue

        try:
            obj_mesh = trimesh.load(str(glb_path), force="mesh")
            if isinstance(obj_mesh, trimesh.Scene):
                obj_mesh = trimesh.util.concatenate(obj_mesh.dump())
        except Exception as e:
            print(f"[export_obj] Failed to load {glb_path}: {e}")
            continue

        # Center, scale, rotate, translate  (same as render_placements)
        bounds = obj_mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        verts = obj_mesh.vertices.astype(np.float64) - center[np.newaxis, :]

        scale = np.array(p["scale"], dtype=np.float64)
        verts = verts * scale[np.newaxis, :]

        R = np.array(p["rotation_3x3"], dtype=np.float64)
        verts = verts @ R.T

        pos = np.array(p["position_m"], dtype=np.float64)
        verts = verts + pos[np.newaxis, :]

        placed = trimesh.Trimesh(
            vertices=verts.astype(np.float32),
            faces=obj_mesh.faces,
            process=False,
        )
        meshes.append(placed)
        print(f"[export_obj] Added {p['type']:12s} ({len(obj_mesh.faces)} faces)")

    combined = trimesh.util.concatenate(meshes)
    out_path = output_dir / "wall_mounted" / "walls_with_objects.obj"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out_path))
    print(f"[export_obj] Combined OBJ → {out_path}  "
          f"({len(combined.vertices)} verts, {len(combined.faces)} faces)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place wall-mounted GLB objects in 3D space using VLM size estimation."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Pipeline output directory (contains camera.json, floorplan_analysis.json, wall_mounted/)",
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the reference room photograph (used for VLM size estimation)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip the final composite render (only write wall_mounted_placements.json)",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Save placement_animation.gif showing objects added one at a time",
    )
    parser.add_argument(
        "--animate-width", type=int, default=960,
        help="GIF frame width in pixels (default 960)",
    )
    parser.add_argument(
        "--animate-duration", type=int, default=1000,
        help="Milliseconds per frame in the GIF (default 1000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_path = Path(args.image)

    if not image_path.exists():
        parser.error(f"Reference image not found: {image_path}")

    placements = compute_placements(output_dir, image_path)

    out_path = output_dir / "wall_mounted" / "wall_mounted_placements.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(placements, indent=2))
    print(f"\n[placement] Wrote {len(placements)} placements → {out_path}")

    if placements:
        export_combined_obj(output_dir, placements)

    if not args.no_render and placements:
        camera_path = output_dir / "camera_vggt.json"
        if not camera_path.exists():
            camera_path = output_dir / "camera.json"
        camera: dict = json.loads(camera_path.read_text())
        render_placements(
            output_dir, placements, camera,
            animate=args.animate,
            animate_gif_width=args.animate_width,
            animate_frame_ms=args.animate_duration,
        )


if __name__ == "__main__":
    main()
