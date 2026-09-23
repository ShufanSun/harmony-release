"""
place_objects.py — place wall-mounted non-window/door objects (art, mirrors, etc.)
                   using VLM size estimation and image back-projection.

Pipeline per object:
  1. Back-project the bounding-box centre pixel through the architectural camera
     to find which wall the object sits on and its world-space position.
  2. Ask the VLM to estimate the real-world width × height × depth.
  3. Load the generated GLBs, scale it to the estimated dimensions (p5/p95 robust
     scaling to handle stray generated reconstructions vertices), orient it to face
     the room interior, translate it to the wall position.
  4. Append the placed geometry to the scene mesh (walls_with_windows.obj →
     walls_with_objects.obj) and do a two-pass render.

Usage:
    python -m object_placement.wall_mounted.placements.place_objects \\
        --output-dir  outputs/20260331_031530 \\
        --image       data/indoor_images/living_room9.jpg \\
        --types       art          # comma-separated, default: art
        --render-only              # re-render existing OBJ without re-placing
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── reuse helpers from fill_openings ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root
from object_placement.wall_mounted.placements.fill_openings import (
    _room_dims_from_obj,
    _get_vertex_colors,
    _rot_y,
    _rasterize_vc_tri,
    _make_projector,
    render_with_glbs,
    NEAR_CLIP,
    EPS,
)

VLM_API_URL = os.environ.get("VLM_API_URL", "http://localhost:8080/v1/chat/completions")

# Route VLM chat through the shared backend router (Qwen local / gpt-5.5 NVIDIA).
from object_placement.vlm_backend import vlm_post as _vlm_post

# Multiplicative pad applied to the silhouette extents when snapping a
# window plane.  The window inpaint silhouette is often shrunk by occluders
# (curtains, plants, furniture) in the reference photo; scaling the snap
# target up keeps the rendered window plane closer to the true window size.
# Doors are not scaled — they're rarely occluded enough to need padding.
_WINDOW_SNAP_SCALE = float(os.environ.get("WINDOW_SNAP_SCALE", "1.3"))


# ── type defaults (used when VLM is unavailable) ──────────────────────────────
_TYPE_DEFAULTS: dict[str, dict] = {
    "art":      dict(width_m=0.60, height_m=0.50, depth_m=0.03),
    "mirror":   dict(width_m=0.60, height_m=0.90, depth_m=0.05),
    "tv":       dict(width_m=1.20, height_m=0.70, depth_m=0.08),
    "shelf":    dict(width_m=1.00, height_m=0.04, depth_m=0.30),
    "cabinet":  dict(width_m=0.80, height_m=1.20, depth_m=0.35),
    "window":   dict(width_m=1.20, height_m=1.00, depth_m=0.05),
    "door":     dict(width_m=0.90, height_m=2.10, depth_m=0.05),
    "clock":    dict(width_m=0.30, height_m=0.30, depth_m=0.05),
    "light":    dict(width_m=0.20, height_m=0.40, depth_m=0.20),
    "radiator": dict(width_m=0.80, height_m=0.60, depth_m=0.10),
    "curtain":  dict(width_m=1.00, height_m=2.20, depth_m=0.05),
    "other":    dict(width_m=0.40, height_m=0.40, depth_m=0.05),
}

_VLM_SIZE_PROMPT = """\
You are analyzing a wall-mounted object in an interior photograph.

ROOM DIMENSIONS
  Width  : {width_m:.2f} m  (left wall to right wall)
  Depth  : {depth_m:.2f} m  (back wall to front/camera)
  Ceiling: {ceiling_h:.2f} m

OBJECT
  Type       : {obj_type}
  Pixel bbox : left={left}  top={top}  right={right}  bottom={bottom}
               (image {img_w}×{img_h} px, origin top-left)

TASKS
1. Identify which wall this object is mounted on:
   "back"  (far wall, faces camera), "left", "right", or "front" (behind camera — rare).
2. Estimate the object's real-world dimensions in metres:
     width_m  — horizontal extent
     height_m — vertical extent
     depth_m  — protrusion from the wall surface

Reference sizes:
  art/photo : 0.2–1.0 m wide, 0.2–1.0 m tall, 0.02–0.05 m deep
  mirror    : 0.3–1.2 m wide, 0.3–1.5 m tall, 0.02–0.05 m deep
  tv        : 0.5–1.8 m wide, 0.3–1.1 m tall, 0.05–0.15 m deep
  shelf     : 0.4–2.0 m wide, 0.03–0.06 m tall, 0.2–0.4 m deep
  clock     : 0.2–0.5 m wide, 0.2–0.5 m tall, 0.05–0.10 m deep
  light     : 0.1–0.4 m wide, 0.2–0.6 m tall, 0.1–0.3 m deep

Reply with JSON only — no explanation:
{{
  "wall":     "back|left|right|front",
  "width_m":  <float>,
  "height_m": <float>,
  "depth_m":  <float>
}}"""

def _wall_placement_order(
    targets:      list[dict],
    csys:         dict,
    pos:          np.ndarray,
    right:        np.ndarray,
    fwd_h:        np.ndarray,
    tilt_tan:     float,
    fx:           float,
    cx_px:        float,
    cy_px:        float,
    W_m:          float,
    D_m:          float,
    ceil:         float,
) -> list[dict]:
    """Order targets for collision-safe placement.

    Strategy (matches user mental model):
      1. Group segments by their likely wall (quick bbox-centre ray cast).
      2. Order walls: corner walls (wall1, wall2 from csys) first,
         then other walls.  Corner walls are nearer the camera's deepest
         recessed corner, so placing them first means back-wall objects
         can slide around already-placed corner-wall objects.
      3. Within each wall sort by free-axis distance from corner_world
         (closest to corner first), so inner objects anchor the wall
         before outer objects try to fit.

    Falls back to original order on any error.
    """
    corner_world = csys["corner_world"]
    wall1        = csys.get("wall1", "")
    wall2        = csys.get("wall2", "")

    # Quick wall detection: back-project bbox centre, find which wall it hits
    def _quick_wall(seg: dict) -> str:
        x1, y1, x2, y2 = seg["box_px"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        ray = _backproject(cx, cy, pos, right, fwd_h, tilt_tan, fx, cx_px, cy_px)
        result = _find_wall(pos, ray, W_m, D_m, ceil)
        return result[0] if result is not None else "unknown"

    # Assign wall + compute free-axis distance from corner per segment
    def _sort_key(seg: dict) -> tuple[int, float]:
        w = _quick_wall(seg)
        # Wall priority: corner walls first (priority 0), others later (priority 1)
        wall_priority = 0 if w in (wall1, wall2) else 1
        # Free-axis distance from corner_world (smaller = closer to corner = placed first)
        free_ax = 0 if w in ("back", "front") else 2
        seg_fa  = (seg["box_px"][0] + seg["box_px"][2]) / 2.0  # pixel centre (proxy)
        # Convert pixel to rough world free-axis via projection of corner and wall ends
        # Simpler: use image-space distance from projected corner, scoped per wall
        proj = _project_world_to_px(corner_world, pos, right, fwd_h, tilt_tan, fx, cx_px, cy_px)
        if proj is not None:
            dist = math.hypot(
                (seg["box_px"][0] + seg["box_px"][2]) / 2 - proj[0],
                (seg["box_px"][1] + seg["box_px"][3]) / 2 - proj[1],
            )
        else:
            dist = 0.0
        return (wall_priority, dist)

    try:
        ordered = sorted(targets, key=_sort_key)
        # Log the order
        lines = []
        for seg in ordered:
            w = _quick_wall(seg)
            lines.append(f"{seg.get('type','?')}[{seg['index']}]@{w}")
        print("[placement_order] " + " → ".join(lines))
        return ordered
    except Exception as e:
        print(f"[placement_order] failed ({e}) — using original order")
        return targets


_VLM_WALL_MOUNTED_PROMPT = """\
You are analyzing a segmented object in an interior photograph.

The highlighted region (bounding box: left={left}, top={top}, right={right}, bottom={bottom} \
in a {img_w}×{img_h} image) shows a "{obj_type}".

Determine whether this object is WALL-MOUNTED (physically attached to or hanging on a wall) \
or FREESTANDING (resting on the floor, a desk, a shelf, a table, or another horizontal surface).

Examples:
  Wall-mounted: wall sconce, wall bracket lamp, picture light, mounted shelf, mounted TV, \
art/mirror/clock hung on wall, window, curtain rod.
  Freestanding: floor lamp, table lamp, desk lamp, bedside lamp, standalone appliance, \
floor-standing shelving unit, anything sitting on top of furniture.

Key test for lights: look at how the lamp connects to its support.
  WALL-MOUNTED (wall sconce) — the lamp's arm/bracket extends from a flat wall surface; there \
is NO base or stand below it; the attachment point is at mid- or upper-wall height and no \
furniture appears directly beneath the attachment.
  FREESTANDING — the lamp sits on or above furniture (desk, table, nightstand, shelf), or has a \
visible base/stand resting on the floor.  If the lamp's bottom overlaps a desk, table, or \
cabinet surface, it is a DESK/TABLE lamp — NOT wall-mounted — even if it looks sleek.

When in doubt, favour FREESTANDING.  A wrong "wall_mounted=true" places a desk lamp on an empty \
wall in the 3D scene, which is worse than omitting it.

Reply with JSON only — no explanation:
{{
  "wall_mounted": true,
  "reason": "<one short phrase>"
}}
or
{{
  "wall_mounted": false,
  "reason": "<one short phrase>"
}}"""


def _is_wall_mounted_vlm(
    image_path: Path,
    seg: dict,
    img_w: int,
    img_h: int,
) -> bool:
    """Ask VLM whether the segment is genuinely wall-mounted. Returns True if yes or if VLM fails."""
    import requests

    x1, y1, x2, y2 = seg["box_px"]
    obj_type = seg.get("type", "other")

    # Types that are inherently wall-mounted — skip the check to save time
    ALWAYS_WALL = {"window", "door", "art", "photo", "mirror", "clock", "tv",
                   "curtain", "blind", "shutter"}
    # NOTE: "light" is intentionally NOT in ALWAYS_WALL — wall sconces are
    # wall-mounted but floor lamps / table lamps are not.  Let the VLM decide.
    if obj_type in ALWAYS_WALL:
        return True

    prompt = _VLM_WALL_MOUNTED_PROMPT.format(
        left=x1, top=y1, right=x2, bottom=y2,
        img_w=img_w, img_h=img_h, obj_type=obj_type,
    )
    b64, mime = _encode_image(image_path)
    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        result = _parse_json(raw)
        if isinstance(result, dict) and "wall_mounted" in result:
            is_mounted = bool(result["wall_mounted"])
            reason     = result.get("reason", "")
            print(f"  [wall_check] {obj_type}: wall_mounted={is_mounted}  ({reason})")
            return is_mounted
    except Exception as e:
        print(f"  [wall_check] VLM call failed: {e}")

    # Fail-policy depends on object type.  For "light" the ambiguity between
    # a wall sconce and a desk/table lamp is high, and a wrong "yes" produces
    # a lamp floating on an empty wall — much worse than dropping it.  For
    # other ambiguous types (shelf, cabinet, radiator) the same logic holds.
    # Only types in ALWAYS_WALL above short-circuit to True before reaching
    # this VLM call.
    _FAIL_CLOSED = {"light", "lamp", "sconce", "shelf", "cabinet",
                    "radiator", "other"}
    if obj_type in _FAIL_CLOSED:
        print(f"  [wall_check] {obj_type}: VLM unavailable — fail-closed "
              f"(rejecting as wall-mounted)")
        return False
    return True   # conservative fail-open only for types with a strong wall prior


# ─────────────────────────────────────────────────────────────────────────────
# Camera / projection helpers (architectural two-point perspective)
# ─────────────────────────────────────────────────────────────────────────────

def _camera_setup(cam: dict):
    """Return (pos, right, up_world, fwd_h, tilt_tan, fx, cx, cy, W, H)."""
    pos     = np.array(cam["position_m"],      dtype=float)
    look_at = np.array(cam["look_at_m"],       dtype=float)
    up_w    = np.array(cam.get("up", [0,1,0]), dtype=float)
    W, H    = int(cam["width_px"]), int(cam["height_px"])
    hfov    = float(cam["hfov_deg"])
    fx      = W / (2.0 * math.tan(math.radians(hfov / 2.0)))
    cx, cy  = W / 2.0, H / 2.0

    fwd = look_at - pos
    fwd /= np.linalg.norm(fwd)

    right_raw = np.cross(fwd, np.array([0., 1., 0.]))
    rn = np.linalg.norm(right_raw)
    right = right_raw / rn if rn > 1e-9 else np.array([1., 0., 0.])

    fwd_h   = np.array([fwd[0], 0., fwd[2]], dtype=float)
    fwd_h_n = np.linalg.norm(fwd_h)
    if fwd_h_n > 1e-9:
        fwd_h    /= fwd_h_n
        tilt_tan  = float(fwd[1]) / fwd_h_n
    else:
        fwd_h    = fwd.copy()
        tilt_tan = 0.0

    return pos, right, up_w, fwd_h, tilt_tan, fx, cx, cy, W, H


def _backproject(px_px: float, py_px: float, pos, right, fwd_h, tilt_tan, fx, cx, cy) -> np.ndarray:
    """Return unit ray direction in world space for screen pixel (px_px, py_px)."""
    xc_n =  (px_px - cx) / fx
    yc_n = -(py_px - cy) / fx
    # yc = d[1] - tilt*zc  →  d[1] = (yc_n + tilt_tan) * zc
    direction = xc_n * right + (yc_n + tilt_tan) * np.array([0., 1., 0.]) + fwd_h
    n = np.linalg.norm(direction)
    return direction / n if n > 1e-9 else direction


def _project_world_to_px(
    pt: np.ndarray,
    pos: np.ndarray,
    right: np.ndarray,
    fwd_h: np.ndarray,
    tilt_tan: float,
    fx: float,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    """Project a 3-D world point to image pixel coordinates (px_col, px_row).

    Returns None if the point is behind the camera.
    """
    d   = pt - pos
    zc  = float(np.dot(d, fwd_h))   # depth along horizontal look axis
    if zc < 0.01:
        return None
    xc  = float(np.dot(d, right))   # horizontal camera coordinate
    yc  = float(d[1]) - tilt_tan * zc  # vertical (world-y corrected for tilt)
    px  = cx + fx * xc / zc
    py  = cy - fx * yc / zc
    return float(px), float(py)


def _ray_hit_wall(pos: np.ndarray, ray: np.ndarray, wall: str,
                  W: float, D: float, ceil: float,
                  tol: float = 0.0) -> np.ndarray | None:
    """Intersect ray with axis-aligned wall plane; return hit point or None.

    tol: bounds tolerance in metres.  When tol > 0 the hit is accepted if the
    point is within tol of the wall boundary, and then clamped to [0, W/D].
    """
    if wall == "back":
        t = -pos[2] / (ray[2] + 1e-12)
    elif wall == "front":
        t = (D - pos[2]) / (ray[2] + 1e-12)
    elif wall == "left":
        t = -pos[0] / (ray[0] + 1e-12)
    elif wall == "right":
        t = (W - pos[0]) / (ray[0] + 1e-12)
    else:
        return None
    if t < 0.01:
        return None
    pt = pos + t * ray
    # check within wall bounds (with optional tolerance)
    if wall in ("back", "front"):
        if not (-tol <= pt[0] <= W + tol and 0 <= pt[1] <= ceil):
            return None
        pt[0] = float(np.clip(pt[0], 0.0, W))
    else:
        if not (-tol <= pt[2] <= D + tol and 0 <= pt[1] <= ceil):
            return None
        pt[2] = float(np.clip(pt[2], 0.0, D))
    return pt


def _camera_inside(pos: np.ndarray, wall: str, W: float, D: float) -> bool:
    """True if the camera is on the interior side of the given wall.

    Uses a small tolerance (0.05 m) only for floating-point slop — cameras that
    are genuinely outside a wall (common with VGGT calibration) are excluded.
    """
    TOL = 0.05
    if wall == "back":   return pos[2] >= -TOL
    if wall == "front":  return pos[2] <= D + TOL
    if wall == "left":   return pos[0] >= -TOL
    if wall == "right":  return pos[0] <= W + TOL
    return False


# Maximum out-of-bounds distance (m) still accepted for a wall hit.
# Needed because VGGT places the camera slightly outside the room, causing
# rays to the correct wall to land just beyond the room boundary.
_WALL_HIT_TOL = 1.0


def _find_wall(pos, ray, W, D, ceil) -> tuple[str, np.ndarray] | None:
    """Return (wall_name, hit_point) for the nearest valid wall intersection.

    Pass 1: strict in-bounds hits only (no tolerance).
    Pass 2: if pass 1 yields no result, accept hits within _WALL_HIT_TOL of the
    boundary (clamped to room bounds) — handles cameras placed slightly outside
    the room by VGGT calibration.
    The nearest hit (by ray distance) among valid candidates is returned.
    """
    for tol in (0.0, _WALL_HIT_TOL):
        best_t, best_wall, best_pt = float("inf"), None, None
        for wall in ("back", "left", "right", "front"):
            if not _camera_inside(pos, wall, W, D):
                continue
            pt = _ray_hit_wall(pos, ray, wall, W, D, ceil, tol=tol)
            if pt is None:
                continue
            d_vec = pt - pos
            t = float(np.dot(d_vec, d_vec))
            if t < best_t:
                best_t, best_wall, best_pt = t, wall, pt
        if best_wall:
            if tol > 0:
                print(f"  [place] wall={best_wall} hit via tolerance ({tol}m) — "
                      f"camera may be outside room")
            return best_wall, best_pt
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Near-seam wall adjudication (VLM)
# ─────────────────────────────────────────────────────────────────────────────
# Geometry back-projection locks each object to whichever wall plane its mask
# centroid ray hits.  For an object mounted on a SIDE wall but whose centroid
# projects just past the side/back corner in screen space (e.g. a cabinet on the
# left wall near the corner), the ray hits the BACK wall and the object is placed
# on the wrong wall.  We detect that ambiguity (hit lands near a corner seam) and,
# for those cases ONLY, let the VLM adjudicate between the two candidate walls —
# keeping geometry authoritative everywhere else (no regression on clear cases).

_WALL_DESC = {
    "back":  "the BACK wall — the wall that faces the camera, farthest from you",
    "front": "the FRONT wall — the wall behind the camera",
    "left":  "the LEFT wall — the side wall that recedes into the distance on the LEFT",
    "right": "the RIGHT wall — the side wall that recedes into the distance on the RIGHT",
}

_ADJUDICATE_WALL_PROMPT = """\
The thick RED rectangle marks a "{obj_type}" that is mounted on a wall in this room.
It sits near a corner where two walls meet, so decide which of these two walls it is
physically mounted ON (its back surface touches that wall):

  A = {desc_a}
  B = {desc_b}

Cue: an object mounted on a SIDE wall faces ACROSS the room — its front face is roughly
perpendicular to the back wall and you see it at an angle / edge-on. An object on the
BACK wall faces the camera head-on. Judge by the wall surface directly behind the object
and the direction its face points.

Reply with JSON only: {{"wall": "A"}} or {{"wall": "B"}}."""


def _seam_candidate(wall: str, world_pt: np.ndarray, W: float, D: float,
                    margin_frac: float = 0.15, margin_min: float = 0.4) -> str | None:
    """If a back/front-wall hit lands near a left/right corner (or a left/right-wall
    hit lands near a back/front corner), return the ADJACENT wall the object might
    really be on. Else None (unambiguous — trust geometry)."""
    if wall in ("back", "front"):
        m = max(margin_min, margin_frac * W)
        if world_pt[0] <= m:
            return "left"
        if world_pt[0] >= W - m:
            return "right"
    elif wall in ("left", "right"):
        m = max(margin_min, margin_frac * D)
        if world_pt[2] <= m:
            return "back"
        if world_pt[2] >= D - m:
            return "front"
    return None


def _adjudicate_wall_vlm(image_path: Path, seg: dict, wall_a: str, wall_b: str,
                         img_w: int, img_h: int) -> str:
    """Ask the VLM which of two candidate walls a near-corner object is mounted on.
    Returns the chosen wall; falls back to wall_a (geometry's pick) on any failure."""
    import io as _io
    import base64 as _b64
    obj_type = seg.get("type", "other")
    x1, y1, x2, y2 = seg["box_px"]
    try:
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        im = _Image.open(image_path).convert("RGB")
        dr = _ImageDraw.Draw(im)
        dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0),
                     width=max(4, im.width // 250))
        buf = _io.BytesIO(); im.save(buf, format="PNG")
        b64, mime = _b64.b64encode(buf.getvalue()).decode(), "image/png"
    except Exception:
        b64, mime = _encode_image(image_path)
    prompt = _ADJUDICATE_WALL_PROMPT.format(
        obj_type=obj_type, desc_a=_WALL_DESC[wall_a], desc_b=_WALL_DESC[wall_b])
    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0.1,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        result = _parse_json(raw)
        pick = str(result.get("wall", "")).strip().upper() if isinstance(result, dict) else ""
        chosen = wall_a if pick == "A" else wall_b if pick == "B" else wall_a
        print(f"  [wall_adjudicate] {obj_type}: A={wall_a} B={wall_b} → VLM picked "
              f"{pick or '?'} ({chosen})")
        return chosen
    except Exception as e:
        print(f"  [wall_adjudicate] {obj_type}: VLM failed ({e}) — keeping geometry wall {wall_a}")
        return wall_a


# ─────────────────────────────────────────────────────────────────────────────
# VLM size estimation
# ─────────────────────────────────────────────────────────────────────────────

def _mask_path_for_seg(seg: dict, wm_dir: Path) -> Path:
    """Return the correct mask path for a segment.

    The mask filename is derived from the GLB file (e.g. "inpaint_01_shelf.glb")
    rather than from the reindexed segment index in segment_results.json.
    The GLB encodes the original segmentation index and type, which is what the
    mask files were saved under (e.g. "segment_01_shelf_mask.png").

    Falls back to the legacy "<index>_<type>_mask.png" naming if no GLB match.
    """
    # 1) Most authoritative: the mask_file recorded in segment_results.json.
    #    This is what segmentation actually wrote and survives any index-vs-
    #    filename mismatch (e.g. segment_05_window_mask.png written for
    #    seg.index=4 because earlier reindexing renamed the entries but not
    #    the files).
    explicit = seg.get("mask_file")
    if explicit:
        candidate = wm_dir / "segmented" / explicit
        if candidate.exists():
            return candidate

    glb_file = seg.get("glb_file") or seg.get("glb_path") or ""
    glb_stem = Path(glb_file).stem   # e.g. "inpaint_01_shelf"

    # 2) Pattern: inpaint_NN_TYPE  →  segment_NN_TYPE_mask.png
    m = re.match(r"inpaint_(\d+)_(\w+)", glb_stem)
    if m:
        orig_idx  = m.group(1)          # e.g. "01"
        orig_type = m.group(2)          # e.g. "shelf"
        candidate = wm_dir / "segmented" / f"segment_{orig_idx}_{orig_type}_mask.png"
        if candidate.exists():
            return candidate

    # 3) Fallback: legacy index + type
    seg_idx  = seg.get("index", 0)
    obj_type = seg.get("type", "other")
    return wm_dir / "segmented" / f"segment_{seg_idx:02d}_{obj_type}_mask.png"


def _encode_image(path: Path) -> tuple[str, str]:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return base64.b64encode(path.read_bytes()).decode(), mime


def _parse_json(raw: str) -> dict:
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    s = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(s)
    except Exception:
        return {}


def estimate_size_from_mask(
    seg: dict,
    wm_dir: Path,
    wall: str,
    world_pt: np.ndarray,
    pos: np.ndarray,
    fwd_h: np.ndarray,
    fx: float,
    right: np.ndarray | None = None,
    cy: float | None = None,
    tilt_tan: float = 0.0,
    ceiling_h: float = 2.7,
) -> dict | None:
    """Compute real-world size from the SAM mask pixel extent and wall depth.

    Physical height for most objects:
        physical_h = pixel_h / fx * Z_c

    For curtains (and doors), the bottom is often occluded by furniture so the
    mask height is truncated.  Instead we infer the full height by:
      1. Finding the world-y of the curtain TOP from the mask's topmost pixel:
             top_world_y = pos_y + (cy − top_pixel_y) / fx * Z_c  (+tilt correction)
      2. Setting height = top_world_y  (curtain extends from floor y=0 to top)
      3. Setting center_y = top_world_y / 2  (vertical midpoint for placement)

    Physical width accounts for foreshortening (wall is rarely face-on):
        physical_w = pixel_w / (|dot(wall_w_axis, right)| * fx) * Z_c
    """
    mask_path = _mask_path_for_seg(seg, wm_dir)

    if not mask_path.exists():
        return None

    try:
        mask_arr = np.array(Image.open(mask_path))
    except Exception:
        return None

    # Foreground pixels (non-zero in any channel)
    if mask_arr.ndim == 3:
        fg = mask_arr.sum(axis=2) > 0
    else:
        fg = mask_arr > 0

    ys, xs = np.where(fg)
    if len(xs) == 0:
        return None

    obj_type = seg.get("type", "other")

    # Pixel extent of the object
    pixel_w = float(xs.max() - xs.min())
    pixel_h = float(ys.max() - ys.min())

    # Camera-space depth: project world_pt onto fwd_h
    d_vec = world_pt - pos
    Z_c   = float(np.dot(d_vec, fwd_h))
    if Z_c < 0.1:
        return None

    # Width correction for wall orientation (foreshortening).
    # For left/right walls the object width runs along z; for back/front along x.
    if right is not None:
        if wall in ("left", "right"):
            w_comp = abs(float(right[2]))   # z-axis component of camera right
        else:                               # back, front
            w_comp = abs(float(right[0]))   # x-axis component of camera right
        w_comp = max(w_comp, 0.1)           # avoid divide-by-zero for face-on walls
    else:
        w_comp = 1.0                        # fallback: assume face-on

    physical_w = pixel_w / (w_comp * fx) * Z_c
    depth_def  = _TYPE_DEFAULTS.get(obj_type, _TYPE_DEFAULTS["other"])["depth_m"]

    result = dict(width_m=round(physical_w, 3), depth_m=depth_def)

    # ── Height: full floor-to-top for curtains (bottom often occluded) ─────────
    if obj_type in ("curtain", "door") and cy is not None:
        top_pixel_y  = float(ys.min())
        # World y of curtain top using perspective + tilt
        top_world_y  = pos[1] + (cy - top_pixel_y) / fx * Z_c + tilt_tan * Z_c
        top_world_y  = float(np.clip(top_world_y, 0.3, ceiling_h))
        physical_h   = top_world_y          # extends from floor (y=0) to top
        result["height_m"]  = round(physical_h, 3)
        result["center_y"]  = round(top_world_y / 2.0, 3)
        print(f"  [mask-size] {obj_type}: pixel={pixel_w:.0f}×{pixel_h:.0f}px  "
              f"Z_c={Z_c:.2f}m  w_comp={w_comp:.3f}  top_pixel={top_pixel_y:.0f}  "
              f"top_world={top_world_y:.2f}m  → {physical_w:.2f}×{physical_h:.2f}m (floor-to-top)")
    else:
        physical_h = pixel_h / fx * Z_c
        # ── Sanity clamp for grazing-angle walls ──────────────────────────────
        # On a wall viewed near-parallel to the view direction (e.g. a mirror on
        # the right wall while the camera looks down the room), Z_c collapses and
        # the pixel→world projection yields an implausibly tiny / aspect-flipped
        # size (mask taller-than-wide but result wider-than-tall). Detect that and
        # rebuild from the mask's aspect ratio anchored to the type default so a
        # mirror stays mirror-sized. Width uses the same rebuild (both dims are
        # unreliable at grazing angles).
        _defs     = _TYPE_DEFAULTS.get(obj_type, _TYPE_DEFAULTS["other"])
        _mask_asp = pixel_h / max(pixel_w, 1.0)                 # h/w of the mask
        _phys_asp = physical_h / max(physical_w, 1e-3)
        _too_small = max(physical_w, physical_h) < 0.30
        _asp_flip  = (_mask_asp >= 1.0) != (_phys_asp >= 1.0)
        _asp_bad   = abs(np.log(_phys_asp / max(_mask_asp, 1e-3))) > np.log(2.2)
        if _too_small or _asp_flip or _asp_bad:
            if _mask_asp >= 1.0:                # taller than wide
                physical_h = _defs["height_m"]
                physical_w = physical_h / _mask_asp
            else:                              # wider than tall
                physical_w = _defs["width_m"]
                physical_h = physical_w * _mask_asp
            result["width_m"] = round(physical_w, 3)
            print(f"  [mask-size] {obj_type}: grazing-wall clamp "
                  f"(mask {pixel_w:.0f}×{pixel_h:.0f}px asp={_mask_asp:.2f}, "
                  f"raw {_phys_asp:.2f}) → {physical_w:.2f}×{physical_h:.2f}m")
        result["height_m"] = round(physical_h, 3)
        # Derive center_y from mask vertical centroid so art/lights etc.
        # are placed at the correct height matching their segmentation mask.
        if cy is not None:
            center_pixel_y = float(ys.mean())
            center_world_y = pos[1] + (cy - center_pixel_y) / fx * Z_c + tilt_tan * Z_c
            center_world_y = float(np.clip(center_world_y, 0.05, ceiling_h - 0.05))
            result["center_y"] = round(center_world_y, 3)
            print(f"  [mask-size] {obj_type}: pixel={pixel_w:.0f}×{pixel_h:.0f}px  "
                  f"Z_c={Z_c:.2f}m  w_comp={w_comp:.3f}  center_py={center_pixel_y:.0f}  "
                  f"center_wy={center_world_y:.2f}m  → {physical_w:.2f}×{physical_h:.2f}m")
        else:
            print(f"  [mask-size] {obj_type}: pixel={pixel_w:.0f}×{pixel_h:.0f}px  "
                  f"Z_c={Z_c:.2f}m  w_comp={w_comp:.3f}  → {physical_w:.2f}×{physical_h:.2f}m")

    return result


def estimate_size_vlm(
    image_path: Path,
    seg: dict,
    room: dict,
    img_w: int,
    img_h: int,
) -> dict:
    """Call VLM to estimate real-world size and wall. Falls back to type defaults."""
    import requests

    x1, y1, x2, y2 = seg["box_px"]
    obj_type = seg.get("type", "other")
    prompt = _VLM_SIZE_PROMPT.format(
        width_m=room["floor_width_m"],
        depth_m=room["floor_depth_m"],
        ceiling_h=room["ceiling_height_m"],
        obj_type=obj_type,
        left=x1, top=y1, right=x2, bottom=y2,
        img_w=img_w, img_h=img_h,
    )
    b64, mime = _encode_image(image_path)
    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        result = _parse_json(raw)
        if isinstance(result, dict) and "wall" in result:
            result["wall"]     = result.get("wall", "back").lower().strip()
            result["width_m"]  = float(result.get("width_m",  0.5))
            result["height_m"] = float(result.get("height_m", 0.5))
            result["depth_m"]  = float(result.get("depth_m",  0.04))
            print(f"  [vlm] {obj_type}: wall={result['wall']}  "
                  f"{result['width_m']:.2f}×{result['height_m']:.2f}×{result['depth_m']:.3f} m")
            return result
    except Exception as e:
        print(f"  [vlm] call failed: {e}")

    fb = dict(_TYPE_DEFAULTS.get(obj_type, _TYPE_DEFAULTS["other"]))
    print(f"  [vlm] fallback for {obj_type}: {fb}")
    return fb


# ─────────────────────────────────────────────────────────────────────────────
# Front-face detection: ask VLM which face should face the room
# ─────────────────────────────────────────────────────────────────────────────

_VLM_FRONT_FACE_PROMPT = """\
You are determining which face of a 3D wall-mounted object should face into the room.

Image 1 — reference photo: the real {obj_type} as it looks in the room.
Images 2–5 — the 3D model rendered from 4 different sides, labelled A/B/C/D.
  A = facing along +Z axis
  B = facing along -Z axis
  C = facing along +X axis
  D = facing along -X axis

Which single view (A, B, C, or D) shows the face that should be visible to the room — \
the decorative / functional front side (e.g. shelf surface, lamp shade, mirror face, \
cabinet door, screen, canvas)?

Reply ONLY with JSON: {{"front": "A"}}, {{"front": "B"}}, {{"front": "C"}}, or {{"front": "D"}}"""

# Mapping from VLM answer to (flip_z, yaw_offset_deg).
# yaw_offset_deg is added to glb_front_deg so the chosen face ends up
# pointing toward the room after the wall-placement rotation.
# _rot_y convention: _rot_y(-deg) applied to verts, so:
#   glb_front_deg = -90 → _rot_y(+90°) → +Z → -X ... (see fill_openings._rot_y)
# After algebra:
#   Z+ front → no extra yaw, no flip
#   Z- front → flip_z=True (or yaw+180)
#   X+ front → glb_front_deg offset = -90  (+X rotates to +Z = room-facing)
#   X- front → glb_front_deg offset = +90  (-X rotates to +Z = room-facing)
_FRONT_FACE_MAP = {
    "A": {"flip_z": False, "yaw_offset": 0},
    "B": {"flip_z": True,  "yaw_offset": 0},
    "C": {"flip_z": False, "yaw_offset": -90},
    "D": {"flip_z": False, "yaw_offset":  90},
}


def _vlm_front_face_check(
    glb_path:   Path,
    ref_path:   Path,
    obj_type:   str,
    panel_size: int = 220,
) -> dict | None:
    """Render the GLB from all 4 horizontal directions and ask the VLM which face
    should face the room.

    Returns {"flip_z": bool, "yaw_offset": int} or None on failure.
    yaw_offset is added to glb_front_deg to orient the chosen face toward the room.
    """
    import requests

    try:
        import trimesh
        scene = trimesh.load(str(glb_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            mesh  = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        else:
            mesh = scene
        if len(mesh.vertices) == 0:
            return None
    except Exception as e:
        print(f"  [front_face] GLB load failed: {e}")
        return None

    bb_center = ((mesh.bounds[0] + mesh.bounds[1]) / 2).astype(np.float64)
    verts_raw = mesh.vertices.astype(np.float64) - bb_center
    vc        = _get_vertex_colors(mesh)

    def _render_from_dir(yaw_deg: float) -> np.ndarray:
        """Render the face that is at the given yaw direction.

        yaw_deg=0   → show +Z face  (rotate 0°, then negate Z for zbuf)
        yaw_deg=180 → show -Z face  (rotate 180°)
        yaw_deg=-90 → show +X face  (rotate -90°, +X maps to +Z)
        yaw_deg=90  → show -X face  (rotate +90°, -X maps to +Z)
        """
        v = verts_raw.copy()
        v[:, 1] = -v[:, 1]                          # generator flip_y
        # Rotate around Y so the desired face points along +Z
        if abs(yaw_deg) > 0.5:
            R = _rot_y(math.radians(yaw_deg))
            v = v @ R.T
        v[:, 2] = -v[:, 2]                          # zbuf shows min-Z → negate
        p05 = np.percentile(v, 5,  axis=0)
        p95 = np.percentile(v, 95, axis=0)
        scale = panel_size * 0.82 / max(float((p95 - p05)[:2].max()), 1e-6)
        buf  = np.full((panel_size, panel_size, 3), 200, dtype=np.uint8)
        zbuf = np.full((panel_size, panel_size), np.inf, dtype=np.float32)
        hx, hy = panel_size / 2.0, panel_size / 2.0
        proj = [(hx + scale * vx, hy - scale * vy, vz) for vx, vy, vz in v]
        for tri in mesh.faces:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            pts = np.array([proj[i0], proj[i1], proj[i2]], dtype=np.float32)
            col = np.array([vc[i0],   vc[i1],   vc[i2]],  dtype=np.uint8)
            _rasterize_vc_tri(buf, zbuf, pts, col)
        return buf

    def _enc(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    try:
        view_a = _render_from_dir(0)     # +Z face
        view_b = _render_from_dir(180)   # -Z face
        view_c = _render_from_dir(-90)   # +X face
        view_d = _render_from_dir(90)    # -X face
        ref_arr = np.array(
            Image.open(ref_path).convert("RGB").resize(
                (panel_size, panel_size), Image.LANCZOS))
    except Exception as e:
        print(f"  [front_face] render failed: {e}")
        return None

    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_enc(ref_arr)}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_enc(view_a)}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_enc(view_b)}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_enc(view_c)}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_enc(view_d)}"}},
                {"type": "text",
                 "text": _VLM_FRONT_FACE_PROMPT.format(obj_type=obj_type)},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=30)
        resp.raise_for_status()
        raw    = resp.json()["choices"][0]["message"]["content"]
        parsed = _parse_json(raw)
        if isinstance(parsed, dict) and "front" in parsed:
            choice = str(parsed["front"]).strip().upper()
            if choice in _FRONT_FACE_MAP:
                result = _FRONT_FACE_MAP[choice]
                print(f"  [front_face] VLM chose {choice} → "
                      f"flip_z={result['flip_z']}  yaw_offset={result['yaw_offset']}°")
                return result
    except Exception as e:
        print(f"  [front_face] VLM call failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GLB orientation check via VLM
# ─────────────────────────────────────────────────────────────────────────────

_VLM_ORIENT_GRID_PROMPT = """\
Image 1 — reference photo: the {obj_type} as it appears in the real room \
(correct orientation, correct face visible).

Image 2 — a 2×2 grid of the SAME 3D model rendered 4 ways, each showing the \
ROOM-FACING side of the object under different orientation corrections:
  Panel A (top-left):     no correction
  Panel B (top-right):    vertical flip only
  Panel C (bottom-left):  face swap only  (shows the other side of the object)
  Panel D (bottom-right): vertical flip + face swap

Look at the reference image and decide which panel shows the {obj_type} in the \
CORRECT orientation — right-side-up, with the proper room-facing face visible.

Key cues:
  • text, patterns, or asymmetric details should read the same way as in the reference
  • the heavier / base part of the object should be at the bottom
  • the decorative face (not a blank back) should be visible

Reply with JSON only — no explanation:
{{"correct_panel": "A|B|C|D"}}"""

# Panel letter → (flip_y, flip_z)
# flip_y: True = flip Y axis (corrects the 3D generator image-Y-down convention)
# flip_z: True = flip Z axis (brings near/visible face to the room-facing side)
_ORIENT_PANELS = {
    "A": (False, False),
    "B": (True,  False),
    "C": (False, True),
    "D": (True,  True),
}


def _render_glb_one_panel(
    verts_raw: np.ndarray,  # centered raw GLB vertices (N,3)
    faces: np.ndarray,
    vc: np.ndarray,
    flip_y: bool,
    flip_z: bool,
    size: int = 200,
) -> np.ndarray:
    """Render one orientation panel showing the ROOM-FACING face.

    After placement, _transform_glb_to_wall snaps the min-Z face to the wall
    and the MAX-Z face faces the room.  The zbuf test (depth < zbuf) shows the
    min-Z face by default.  We therefore negate Z after all orientation flips
    so that the room-facing (max-Z) face becomes the min-Z face in the render
    and is correctly shown to the VLM for comparison.

    Panel correspondence:
      A (flip_y=F, flip_z=F): shows raw back face, upside-down
      B (flip_y=T, flip_z=F): shows raw back face, right-side-up
      C (flip_y=F, flip_z=T): shows raw front face, upside-down
      D (flip_y=T, flip_z=T): shows raw front face, right-side-up  ← correct for the 3D generator
    """
    v = verts_raw.copy()
    if flip_y:
        v[:, 1] = -v[:, 1]
    if flip_z:
        v[:, 2] = -v[:, 2]

    # Negate Z so the room-facing (max-Z) face is what the zbuf shows.
    v[:, 2] = -v[:, 2]

    p05 = np.percentile(v, 5,  axis=0)
    p95 = np.percentile(v, 95, axis=0)
    span_x = max(float(p95[0] - p05[0]), 1e-6)
    span_y = max(float(p95[1] - p05[1]), 1e-6)
    scale = size * 0.82 / max(span_x, span_y)

    buf  = np.full((size, size, 3), 200, dtype=np.uint8)
    zbuf = np.full((size, size), np.inf, dtype=np.float32)
    hx, hy = size / 2.0, size / 2.0

    proj = [(hx + scale * vx, hy - scale * vy, vz)
            for vx, vy, vz in v]

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        pts = np.array([proj[i0], proj[i1], proj[i2]], dtype=np.float32)
        col = np.array([vc[i0], vc[i1], vc[i2]], dtype=np.uint8)
        _rasterize_vc_tri(buf, zbuf, pts, col)

    return buf


def _render_glb_orientation_grid(
    glb_path: Path,
    panel_size: int = 200,
) -> np.ndarray | None:
    """Render GLB in 4 flip_y×flip_z combinations; return a 2×2 labelled grid (RGB)."""
    try:
        import trimesh
        from PIL import ImageDraw, ImageFont
        scene = trimesh.load(str(glb_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if not geoms:
                return None
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        else:
            mesh = scene
        if len(mesh.vertices) == 0:
            return None
    except Exception as e:
        print(f"  [orient] GLB load failed: {e}")
        return None

    bb_center = ((mesh.bounds[0] + mesh.bounds[1]) / 2).astype(np.float64)
    verts_raw = mesh.vertices.astype(np.float64) - bb_center
    vc = _get_vertex_colors(mesh)

    pad = 4   # pixels between panels
    label_h = 22
    grid_w = panel_size * 2 + pad * 3
    grid_h = (panel_size + label_h) * 2 + pad * 3
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    panel_order = [("A", 0, 0), ("B", 0, 1), ("C", 1, 0), ("D", 1, 1)]
    for label, row, col in panel_order:
        flip_y, flip_z = _ORIENT_PANELS[label]
        panel = _render_glb_one_panel(verts_raw, mesh.faces, vc, flip_y, flip_z, panel_size)

        x0 = pad + col * (panel_size + pad)
        y0 = pad + row * (panel_size + label_h + pad) + label_h
        grid[y0:y0 + panel_size, x0:x0 + panel_size] = panel

        # Draw label above each panel
        label_img = Image.fromarray(grid)
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x0 + panel_size // 2 - 6, y0 - label_h + 3), label,
                  fill=(30, 30, 30), font=font)
        grid = np.array(label_img)

    return grid


_PAINTING_MIRROR_PROMPT = """\
You are checking whether a 3D-placed PAINTING / ART / FRAME has been
LEFT-RIGHT MIRRORED relative to its reference photo.

Image 1: REFERENCE — the painting/art as it appears in the original room
photo.  This is the ground truth.

Image 2: 3D RENDER — the same painting placed on the wall in our render.

Compare the two for LEFT-RIGHT asymmetric features ONLY:
  - Text or signatures (e.g. artist signature on the LEFT vs RIGHT)
  - Asymmetric figures, buildings, faces (which way they face)
  - Patterns or compositional elements (e.g. tree on the LEFT in the
    reference but on the RIGHT in the render)

IMPORTANT:
  - Only flag mirroring if asymmetric features are clearly reversed.
  - Do NOT flag for slight perspective, lighting, or colour differences.
  - For purely abstract / symmetric / repeating patterns, respond
    "mirrored": false.

Reply ONLY with JSON:
{"mirrored": true|false, "feature": "<which asymmetric feature>", "reasoning": "<one short sentence>"}
"""


def _vlm_orientation_check(
    glb_path: Path,
    canvas_path: Path,
    obj_type: str,
) -> dict | None:
    """Ask VLM which of 4 flip_y×flip_z orientations matches the reference.

    Renders a 2×2 grid (panels A/B/C/D) and compares to the reference canvas.
    Returns {"flip_y": bool, "flip_x": bool, "flip_z": bool} on success, or None.
    """
    import requests

    grid_arr = _render_glb_orientation_grid(glb_path)
    if grid_arr is None:
        print(f"  [orient] grid render failed")
        return None

    def _enc(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    try:
        ref_img  = Image.open(canvas_path).convert("RGB")
        grid_img = Image.fromarray(grid_arr)
        b64_ref  = _enc(ref_img)
        b64_grid = _enc(grid_img)
    except Exception as e:
        print(f"  [orient] image encode failed: {e}")
        return None

    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_ref}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_grid}"}},
                {"type": "text",
                 "text": _VLM_ORIENT_GRID_PROMPT.format(obj_type=obj_type)},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=30)
        resp.raise_for_status()
        raw    = resp.json()["choices"][0]["message"]["content"]
        result = _parse_json(raw)
        if isinstance(result, dict) and "correct_panel" in result:
            panel  = str(result["correct_panel"]).strip().upper()
            if panel not in _ORIENT_PANELS:
                print(f"  [orient] VLM returned unknown panel {panel!r} — using default D")
                panel = "D"
            flip_y, flip_z = _ORIENT_PANELS[panel]
            print(f"  [orient] VLM → panel={panel}  flip_y={flip_y}  flip_z={flip_z}")
            return {"flip_y": flip_y, "flip_x": False, "flip_z": flip_z}
    except Exception as e:
        print(f"  [orient] VLM call failed: {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Geometric bbox-alignment fallback
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_align_yaw(
    verts0:  np.ndarray,   # centered, flip_y-corrected GLB vertices (N,3)
    wall:    str,
    cam_pos: np.ndarray,   # scene camera position, for 180° disambiguation
    world_pt: np.ndarray,  # approximate object position in world space
) -> int:
    """Return the Y-rotation angle (degrees, 0–359) that aligns the object's
    tightest horizontal dimension with the wall normal.

    For a wall-mounted cabinet this means: wall-depth (shortest XZ dimension)
    becomes the wall-normal axis, wall-width (longest XZ dimension) becomes the
    axis along the wall.  The 180° ambiguity (front vs back) is resolved by
    orienting the "camera-facing" half toward the camera.
    """
    # Wall-normal axis index: Z for back/front, X for left/right
    depth_col = 2 if wall in ("back", "front") else 0

    # Find the angle (0–179°) that minimises span along wall-normal axis.
    # 180° symmetry means we only need to search half the circle.
    best_angle, best_span = 0, float("inf")
    for a in range(0, 180, 2):
        rv = verts0 @ _rot_y(math.radians(a)).T
        span = float(rv[:, depth_col].max() - rv[:, depth_col].min())
        if span < best_span:
            best_span = span
            best_angle = a

    # 180° disambiguation: choose the half whose centroid faces toward the camera.
    # The camera-facing side of the object is the front (room-facing) side.
    # Compute the dot product of (object_centroid – cam_pos) with the wall-outward
    # direction; the correct orientation has the front pointing away from the wall
    # (i.e., toward the camera).
    cam_dir = np.array(world_pt, dtype=float) - np.array(cam_pos, dtype=float)
    cam_dir_norm = cam_dir / (np.linalg.norm(cam_dir) + 1e-9)

    rv0   = verts0 @ _rot_y(math.radians(best_angle)).T
    rv180 = verts0 @ _rot_y(math.radians(best_angle + 180)).T

    # For each candidate, the "front half" is the half of vertices furthest from the wall.
    # For back wall that means max-Z half; for right wall max-X half; etc.
    # Use the face whose mean position has a higher dot product toward the camera.
    wall_outward = {"back": np.array([0,0,1]), "front": np.array([0,0,-1]),
                    "left": np.array([1,0,0]), "right": np.array([-1,0,0])}
    out_dir = wall_outward.get(wall, np.array([0,0,1]))

    front_half_0   = rv0[rv0[:,   depth_col] > rv0[:,   depth_col].mean()]
    front_half_180 = rv180[rv180[:, depth_col] > rv180[:, depth_col].mean()]

    score_0   = float(np.mean(front_half_0   @ out_dir)) if len(front_half_0)   else 0
    score_180 = float(np.mean(front_half_180 @ out_dir)) if len(front_half_180) else 0

    # Correct orientation: front half is on the out_dir side (score > 0)
    # Pick whichever candidate has the front half more aligned with out_dir
    if score_180 > score_0:
        best_angle = (best_angle + 180) % 360

    return best_angle


# ─────────────────────────────────────────────────────────────────────────────
# VLM front-face detection — which Y-rotation shows the front of the GLB?
# ─────────────────────────────────────────────────────────────────────────────

_FRONT_VIEWS = {
    "+Z":   0,
    "+X+Z": 45,
    "+X":   90,
    "+X-Z": 135,
    "-Z":   180,
    "-X-Z": 225,
    "-X":   270,
    "-X+Z": 315,
}

_VLM_FRONT_VIEW_PROMPT = """\
Image 1 is a reference photo of a {obj_type} cropped from a room scene.
Images 2–9 are 8 rendered views of its 3D model, each rotated around the vertical axis.
The view labels in order are: {labels}.

Which view label shows the FRONT FACE of the object — the side that faces into \
the room and looks most like the reference photo?

Reply with JSON only: {{"front_view": "<label>"}}"""


def _render_glb_8views(
    glb_path:   Path,
    size_m:     dict,
    project_fn,           # (x,y,z) → (px,py)|None — scene camera projector
    cam_pos:    np.ndarray,
    world_pt:   np.ndarray,
    img_w:      int,
    img_h:      int,
    thumb_size: int = 256,
) -> dict | None:
    """Render 8 Y-rotations of the GLB using the scene camera at world_pt.

    The GLB is scaled by VLM height, placed at world_pt, and projected through the
    real scene camera — so VLM sees exactly what the camera would see for each
    rotation, matching the perspective of the reference canvas.

    Returns dict label → HxWx3 uint8 thumbnail, or None on failure.
    """
    try:
        import trimesh
        scene = trimesh.load(str(glb_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if not geoms:
                return None
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        else:
            mesh = scene
        if len(mesh.vertices) == 0:
            return None
    except Exception as e:
        print(f"  [front_view] load failed: {e}")
        return None

    bb_center = ((mesh.bounds[0] + mesh.bounds[1]) / 2).astype(np.float64)
    verts0 = mesh.vertices.astype(np.float64) - bb_center
    verts0[:, 1] = -verts0[:, 1]   # the 3D generator Y-flip

    p05 = np.percentile(verts0, 5, axis=0)
    p95 = np.percentile(verts0, 95, axis=0)
    robust = p95 - p05
    s = float(size_m["height_m"]) / max(float(robust[1]), 1e-6)
    verts0 *= s

    vc = _get_vertex_colors(mesh)

    result = {}
    for label, angle_deg in _FRONT_VIEWS.items():
        # Place rotated GLB at world_pt
        rv = (verts0 @ _rot_y(math.radians(angle_deg)).T) + world_pt

        # Project through the calibrated scene camera.
        # project_fn (from _make_projector) returns (px, py, cam_z) or None.
        proj = []
        px_list, py_list = [], []
        for v in rv:
            p = project_fn(v)
            if p is not None:                   # cam_z > NEAR_CLIP already filtered
                proj.append((float(p[0]), float(p[1]), float(p[2])))
                px_list.append(float(p[0]))
                py_list.append(float(p[1]))
            else:
                proj.append(None)

        if len(px_list) < 3:
            result[label] = np.full((thumb_size, thumb_size, 3), 185, dtype=np.uint8)
            continue

        # Crop bounds with padding
        min_px, max_px = min(px_list), max(px_list)
        min_py, max_py = min(py_list), max(py_list)
        pad_x = max((max_px - min_px) * 0.25, 10)
        pad_y = max((max_py - min_py) * 0.25, 10)
        cx0 = max(0, int(min_px - pad_x))
        cx1 = min(img_w - 1, int(max_px + pad_x))
        cy0 = max(0, int(min_py - pad_y))
        cy1 = min(img_h - 1, int(max_py + pad_y))
        cw, ch = cx1 - cx0 + 1, cy1 - cy0 + 1

        if cw <= 0 or ch <= 0:
            result[label] = np.full((thumb_size, thumb_size, 3), 185, dtype=np.uint8)
            continue

        # Render into crop-sized buffer (offset projections by crop origin)
        buf  = np.full((ch, cw, 3), 185, dtype=np.uint8)
        zbuf = np.full((ch, cw), np.inf, dtype=np.float32)
        for tri in mesh.faces:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            if proj[i0] is None or proj[i1] is None or proj[i2] is None:
                continue
            pts = np.array([
                [proj[i0][0] - cx0, proj[i0][1] - cy0, proj[i0][2]],
                [proj[i1][0] - cx0, proj[i1][1] - cy0, proj[i1][2]],
                [proj[i2][0] - cx0, proj[i2][1] - cy0, proj[i2][2]],
            ], dtype=np.float32)
            col = np.array([vc[i0], vc[i1], vc[i2]], dtype=np.uint8)
            _rasterize_vc_tri(buf, zbuf, pts, col)

        result[label] = np.array(
            Image.fromarray(buf).resize((thumb_size, thumb_size), Image.LANCZOS))

    return result


def _vlm_front_view(
    glb_path:   Path,
    canvas_path: Path,
    obj_type:   str,
    size_m:     dict,
    project_fn,
    cam_pos:    np.ndarray,
    world_pt:   np.ndarray,
    img_w:      int,
    img_h:      int,
) -> int | None:
    """Ask VLM which of 8 scene-camera renders of the GLB matches the canvas.

    Each render places the GLB at world_pt with a different Y-rotation and
    projects it through the real scene camera — same perspective as the canvas.

    Returns the Y-rotation angle (0/45/…/315) or None on failure.
    """
    import requests

    views = _render_glb_8views(
        glb_path, size_m, project_fn, cam_pos, world_pt, img_w, img_h)
    if views is None:
        return None

    def _enc(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    try:
        ref_img = Image.open(canvas_path).convert("RGB")
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_enc(ref_img)}"}},
        ]
        for arr in views.values():
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{_enc(Image.fromarray(arr))}"}})
        content.append({"type": "text", "text": _VLM_FRONT_VIEW_PROMPT.format(
            obj_type=obj_type, labels=", ".join(views.keys()))})
    except Exception as e:
        print(f"  [front_view] encode failed: {e}")
        return None

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw  = resp.json()["choices"][0]["message"]["content"]
        res  = _parse_json(raw)
        if isinstance(res, dict) and "front_view" in res:
            label = str(res["front_view"]).strip()
            if label in _FRONT_VIEWS:
                angle = _FRONT_VIEWS[label]
                print(f"  [front_view] VLM → front={label} ({angle}°)")
                return angle
        print(f"  [front_view] VLM unparseable: {raw!r}")
    except Exception as e:
        print(f"  [front_view] VLM call failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Wall-lamp facing-direction check (post-placement)
# ─────────────────────────────────────────────────────────────────────────────

_VLM_LAMP_FACING_PROMPT = """\
You are checking whether a wall-mounted lamp (sconce) is facing the correct \
direction after 3D placement.

Image 1 — reference: the lamp as seen in the original room photo{ref_note}.
Image 2 — rendered 3D model: how the lamp looks now, projected through the \
real scene camera at its wall position.

The lamp is mounted on the {wall_desc}.

Look at the horizontal facing direction of the lamp:
  • Which side does the ARM, SHADE, or BODY lean/extend toward along the wall?
  • Use image edges as reference: "left" = toward the left edge of the image.

Step 1 — Reference (Image 1): does the arm/shade point LEFT or RIGHT?
Step 2 — Render    (Image 2): does the arm/shade point LEFT or RIGHT?
Step 3 — Are they the SAME direction?

Reply with JSON only — no explanation:
{{"ref_direction": "left|right|unclear", \
"render_direction": "left|right|unclear", \
"same_direction": true|false}}"""


def _render_placed_verts_crop(
    verts: np.ndarray,   # (N,3) world-space — already placed
    faces: np.ndarray,   # (F,3) int indices
    vc: np.ndarray,      # (N,3) uint8 vertex colors
    project_fn,          # scene-camera projector: v→(px,py,zc)|None
    bbox_px: list,       # [x1,y1,x2,y2] reference bbox for crop size hint
    img_w: int,
    img_h: int,
    thumb_size: int = 256,
) -> np.ndarray | None:
    """Render already-placed world-space verts through the scene camera.

    Returns a thumb_size×thumb_size uint8 RGB crop around the visible object
    pixels, or None if fewer than 3 vertices project into the image.
    """
    proj = []
    px_list, py_list = [], []
    for v in verts:
        p = project_fn(v)
        if p is not None:
            proj.append((float(p[0]), float(p[1]), float(p[2])))
            px_list.append(float(p[0]))
            py_list.append(float(p[1]))
        else:
            proj.append(None)

    if len(px_list) < 3:
        return None

    # Crop bounds: union of projected verts + reference bbox, with padding
    x1, y1, x2, y2 = bbox_px
    min_px = min(min(px_list), x1)
    max_px = max(max(px_list), x2)
    min_py = min(min(py_list), y1)
    max_py = max(max(py_list), y2)
    pad_x  = max((max_px - min_px) * 0.2, 10)
    pad_y  = max((max_py - min_py) * 0.2, 10)
    cx0 = max(0, int(min_px - pad_x))
    cx1 = min(img_w - 1, int(max_px + pad_x))
    cy0 = max(0, int(min_py - pad_y))
    cy1 = min(img_h - 1, int(max_py + pad_y))
    cw, ch = cx1 - cx0 + 1, cy1 - cy0 + 1
    if cw <= 0 or ch <= 0:
        return None

    buf  = np.full((ch, cw, 3), 185, dtype=np.uint8)
    zbuf = np.full((ch, cw), np.inf, dtype=np.float32)

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if proj[i0] is None or proj[i1] is None or proj[i2] is None:
            continue
        pts = np.array([
            [proj[i0][0] - cx0, proj[i0][1] - cy0, proj[i0][2]],
            [proj[i1][0] - cx0, proj[i1][1] - cy0, proj[i1][2]],
            [proj[i2][0] - cx0, proj[i2][1] - cy0, proj[i2][2]],
        ], dtype=np.float32)
        col = np.array([vc[i0], vc[i1], vc[i2]], dtype=np.uint8)
        _rasterize_vc_tri(buf, zbuf, pts, col)

    return np.array(
        Image.fromarray(buf).resize((thumb_size, thumb_size), Image.LANCZOS))


def _vlm_lamp_facing_check(
    ref_path: Path,         # reference image (canvas crop or full room image)
    bbox_px: list,          # [x1,y1,x2,y2] of the lamp in the reference image
    verts: np.ndarray,      # placed world-space vertices (N,3)
    faces: np.ndarray,
    vc: np.ndarray,
    project_fn,
    img_w: int,
    img_h: int,
    wall: str = "back",
    is_canvas: bool = False,  # True → ref_path is already a canvas crop (no bbox crop needed)
) -> bool:
    """Return True if the placed lamp needs a tangential-axis mirror to match the reference.

    Renders the already-placed lamp through the scene camera, shows the
    reference alongside, and asks VLM whether they face the same horizontal
    direction.  Returns False on any failure (safe default = no flip).
    """
    import requests

    render_arr = _render_placed_verts_crop(
        verts, faces, vc, project_fn, bbox_px, img_w, img_h)
    if render_arr is None:
        print("  [lamp_facing] render crop failed — skipping check")
        return False

    def _enc(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    try:
        if is_canvas:
            ref_img = Image.open(ref_path).convert("RGB")
        else:
            x1, y1, x2, y2 = bbox_px
            ref_img = Image.open(ref_path).convert("RGB").crop((x1, y1, x2, y2))
        render_img = Image.fromarray(render_arr)

        _wall_desc = {
            "back": "back wall (facing toward the camera)",
            "front": "front wall (behind the camera)",
            "left": "left side wall",
            "right": "right side wall",
        }.get(wall, f"{wall} wall")
        _ref_note = " (isolated object crop)" if is_canvas else " (cropped to object bbox)"

        prompt = _VLM_LAMP_FACING_PROMPT.format(
            wall_desc=_wall_desc, ref_note=_ref_note)

        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_enc(ref_img)}"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_enc(render_img)}"}},
            {"type": "text", "text": prompt},
        ]
    except Exception as e:
        print(f"  [lamp_facing] image encode failed: {e}")
        return False

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    print("  [lamp_facing] calling VLM …")
    try:
        resp = _vlm_post(payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        print(f"  [lamp_facing] raw VLM response: {raw[:200]}")
        res = _parse_json(raw)
        if isinstance(res, dict) and "same_direction" in res:
            same = bool(res["same_direction"])
            ref_dir    = res.get("ref_direction", "?")
            render_dir = res.get("render_direction", "?")
            print(f"  [lamp_facing] VLM → ref={ref_dir}  render={render_dir}  "
                  f"same={same} → {'no flip' if same else 'FLIP'}")
            # If either direction is unclear, don't flip (safe default)
            if ref_dir == "unclear" or render_dir == "unclear":
                print("  [lamp_facing] direction unclear — skipping flip")
                return False
            return not same
        else:
            print(f"  [lamp_facing] VLM response missing 'same_direction' — skipping flip (res={res!r})")
    except Exception as e:
        print(f"  [lamp_facing] VLM call failed: {e}")

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive VLM placement initialisation
# ─────────────────────────────────────────────────────────────────────────────

_VLM_INIT_PROMPT = """\
You are placing a {obj_type} into a 3D room scene.

Room: width={W:.2f}m (X-axis, left wall X=0 → right wall X={W:.2f}), \
depth={D:.2f}m (Z-axis, back wall Z=0 → front/near wall Z={D:.2f}), \
ceiling height={ceil:.2f}m.

Image 1: the original room photo — the {obj_type} is highlighted with a red box.
Images 2–9: 8 renders of the {obj_type}'s 3D model, each rotated a different \
amount around its vertical axis, projected through the real scene camera at the \
object's detected wall position. Rotation labels in order: {labels}.

Determine ALL placement parameters:

wall          — which wall the object is mounted on:
                "back" (far wall, Z=0), "front" (near wall, Z={D:.2f}),
                "left" (X=0), "right" (X={W:.2f})

wall_pos_frac — 0.0–1.0 fraction of the wall's span where the object's centre sits.
                back/front walls → fraction of room width (0=left side, 1=right side).
                left/right walls → fraction of room depth (0=back-wall side, 1=front-wall side).

center_height_m — height of object centre from floor (metres)

object_height_m — real-world height of the object (metres)

front_view    — which render label shows the FRONT face of the object \
(the side that faces into the room and matches the reference photo)

Reply with JSON only — no explanation:
{{"wall":"...", "wall_pos_frac":<0-1>, "center_height_m":<m>, \
"object_height_m":<m>, "front_view":"<label>"}}"""


def _vlm_init_placement(
    image_path:     Path,
    seg:            dict,
    glb_path:       Path,
    rough_world_pt: np.ndarray,   # from back-projection — used only for rendering
    room:           dict,
    project_fn,                   # scene camera projector
    cam_pos:        np.ndarray,
    img_w:          int,
    img_h:          int,
) -> dict | None:
    """Single VLM call that returns rotation, wall, position, and height.

    Renders 8 Y-rotations at rough_world_pt via the scene camera, annotates
    the room image with the object's bounding box, and asks VLM for all
    placement parameters at once.

    Returns dict with keys:
        wall            str
        world_pt        np.ndarray (3,)  — world position of object centre
        object_height_m float
        front_view_deg  int              — Y-rotation of front face (0/45/…/315)
    or None on failure.
    """
    import requests

    W    = float(room["floor_width_m"])
    D    = float(room["floor_depth_m"])
    ceil = float(room.get("ceiling_height_m", 2.7))
    obj_type = seg.get("type", "object")
    x1, y1, x2, y2 = seg["box_px"]

    # Rough height for 8-view renders (≈40% of ceiling, corrected by VLM output)
    rough_size = {"height_m": ceil * 0.4, "width_m": 0.5, "depth_m": 0.3}

    views = _render_glb_8views(
        glb_path, rough_size, project_fn, cam_pos,
        rough_world_pt, img_w, img_h)
    if views is None:
        return None

    def _enc(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    try:
        from PIL import ImageDraw
        room_img = Image.open(image_path).convert("RGB").copy()
        draw = ImageDraw.Draw(room_img)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_enc(room_img)}"}},
        ]
        for arr in views.values():
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{_enc(Image.fromarray(arr))}"}})
        content.append({"type": "text", "text": _VLM_INIT_PROMPT.format(
            obj_type=obj_type, W=W, D=D, ceil=ceil,
            labels=", ".join(views.keys()))})
    except Exception as e:
        print(f"  [vlm_init] encode failed: {e}")
        return None

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        res = _parse_json(raw)
        if not isinstance(res, dict):
            print(f"  [vlm_init] unparseable: {raw!r}")
            return None

        wall = str(res.get("wall", "back")).lower().strip()
        if wall not in ("back", "front", "left", "right"):
            wall = "back"

        frac     = float(max(0.0, min(1.0, res.get("wall_pos_frac", 0.5))))
        center_h = float(res.get("center_height_m", ceil * 0.5))
        obj_h    = float(res.get("object_height_m", 0.6))
        label    = str(res.get("front_view", "+Z")).strip()
        front_deg = _FRONT_VIEWS.get(label, 0)

        # Build world_pt from wall + frac + height
        if wall == "back":
            world_pt = np.array([frac * W, center_h, 0.0])
        elif wall == "front":
            world_pt = np.array([frac * W, center_h, D])
        elif wall == "left":
            world_pt = np.array([0.0, center_h, frac * D])
        else:  # right
            world_pt = np.array([W, center_h, frac * D])

        print(f"  [vlm_init] wall={wall}  pos_frac={frac:.2f}  "
              f"center_h={center_h:.2f}m  obj_h={obj_h:.2f}m  "
              f"front={label}({front_deg}°)")
        return {
            "wall":            wall,
            "world_pt":        world_pt,
            "object_height_m": obj_h,
            "front_view_deg":  front_deg,
        }
    except Exception as e:
        print(f"  [vlm_init] VLM call failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GLB transform — scale + orient + translate to wall
# ─────────────────────────────────────────────────────────────────────────────

def _place_at_center(
    glb_path:     Path,
    size:         dict,
    room:         dict,
    cam_pos:      np.ndarray,
    glb_front_deg: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Scale GLB, orient front face toward camera, place at room centre.

    glb_front_deg: VLM-determined angle of the front face in the flip_y-corrected
                   GLB space (0/45/90/…/315).  Pre-rotates by -glb_front_deg to
                   bring the front to canonical +Z, then swings +Z toward the camera.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("place_objects requires trimesh: pip install trimesh")

    if not glb_path.exists():
        print(f"  [center] GLB not found: {glb_path}")
        return None
    try:
        scene = trimesh.load(str(glb_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if not geoms:
                return None
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        else:
            mesh = scene
    except Exception as e:
        print(f"  [center] Failed to load {glb_path.name}: {e}")
        return None

    if len(mesh.vertices) == 0:
        return None

    bb_center = ((mesh.bounds[0] + mesh.bounds[1]) / 2).astype(np.float64)
    verts = mesh.vertices.astype(np.float64) - bb_center

    # the 3D generator: image-Y increases downward → flip Y to world-up
    verts[:, 1] = -verts[:, 1]

    p05 = np.percentile(verts, 5,  axis=0)
    p95 = np.percentile(verts, 95, axis=0)
    robust = p95 - p05

    target_h = float(size["height_m"])
    s = target_h / max(robust[1], 1e-6)
    verts *= s

    W = float(room["floor_width_m"])
    D = float(room["floor_depth_m"])
    cx_room = W / 2.0
    cz_room = D / 2.0

    # 1. Pre-rotate by -glb_front_deg to bring the VLM-identified front face to +Z.
    if glb_front_deg != 0:
        verts = verts @ _rot_y(-math.radians(glb_front_deg)).T

    # 2. Rotate +Z toward the scene camera so we see the front face in the render.
    dx = float(cam_pos[0]) - cx_room
    dz = float(cam_pos[2]) - cz_room
    cam_angle = math.atan2(dx, dz)
    verts = verts @ _rot_y(cam_angle).T
    print(f"  [center] front_deg={glb_front_deg}  cam_angle={math.degrees(cam_angle):.1f}°")

    # Place centre at room centre, bottom on floor
    verts[:, 0] += cx_room
    verts[:, 2] += cz_room
    verts[:, 1] -= verts[:, 1].min()

    vc = _get_vertex_colors(mesh)
    print(f"  [center] {glb_path.name}: scale={s:.3f}  "
          f"size≈{(verts[:,0].max()-verts[:,0].min()):.2f}w "
          f"x {(verts[:,1].max()-verts[:,1].min()):.2f}h "
          f"x {(verts[:,2].max()-verts[:,2].min()):.2f}d m")
    return verts, mesh.faces, vc


def _transform_glb_to_wall(
    glb_path: Path,
    wall:     str,
    world_pt: np.ndarray,   # centre of object on the wall surface
    size:     dict,         # width_m, height_m, depth_m
    room:     dict,
    cam_pos:       np.ndarray | None = None,  # for bbox-align fallback
    flip_y:        bool = True,  # flip Y axis (the 3D generator image-Y convention)
    flip_x:        bool = False, # flip X axis (left-right mirror)
    flip_z:        bool = False, # flip Z axis (front/back face swap)
    glb_front_deg:    int | None = None,  # VLM-determined Y-rotation; None = VLM didn't answer
    front_yaw_offset: int = 0,           # yaw_offset from _vlm_front_face_check (±90 when front at ±X)
    embed:            bool = False, # True → embed mode: width scaling + outward snap
    outward:          bool = False, # True → snap room-facing face to wall, object extends outside room
    detilt:           bool = True,  # False → skip PCA de-tilt (use for windows/doors)
    skip_obb_swap:    bool = False, # True → skip 90° wall-plane safety-swap (use for window/door)
    force_upright:    bool = False, # True → force height axis exactly to world-Y (use for window/door)
    force_floor:      bool = False, # True → always snap bottom to y=0 (use for door: never hover)
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load GLB, scale to target size, orient to wall, place at world_pt.

    Returns (verts_world, faces, vertex_colors) or None.
    Uses p5/p95 robust scaling so stray the 3D generator vertices don't distort the scale.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("place_objects requires trimesh: pip install trimesh")

    if not glb_path.exists():
        print(f"  [place] GLB not found: {glb_path}")
        return None
    try:
        scene = trimesh.load(str(glb_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if not geoms:
                return None
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        else:
            mesh = scene
    except Exception as e:
        print(f"  [place] Failed to load {glb_path.name}: {e}")
        return None

    if len(mesh.vertices) == 0:
        return None

    bb_center = ((mesh.bounds[0] + mesh.bounds[1]) / 2).astype(np.float64)
    verts = mesh.vertices.astype(np.float64) - bb_center

    # Robust p5/p95 scale factors (computed on raw centered verts before any flips)
    p05 = np.percentile(verts, 5,  axis=0)
    p95 = np.percentile(verts, 95, axis=0)
    robust = p95 - p05
    bb_z   = float(mesh.bounds[1][2] - mesh.bounds[0][2])   # depth axis in GLB

    # Orientation corrections for generated GLBs convention:
    #   flip_y: the 3D generator image Y increases downward → world Y should increase upward
    #   flip_x: left-right mirror correction if VLM detected the texture is mirrored
    #   flip_z: front/back face swap — applied when VLM determined the -Z face is the
    #           decorative/functional side (e.g., painting canvas, lamp front)
    # Applied before scaling so robust[] magnitudes remain valid.
    if flip_y:
        verts[:, 1] = -verts[:, 1]
    if flip_x:
        verts[:, 0] = -verts[:, 0]
    # flip_z is NOT applied here as a raw GLB-Z flip.
    # Both flat and 3D OBB blocks below apply flip_z AFTER OBB alignment
    # as a wall-normal-axis flip, which is coordinate-independent.

    # ── Flat-object OBB placement (detilt=False) ───────────────────────────────
    # For thin flat objects (window, curtain, art, painting, frame, mirror):
    #   1. SVD OBB: thinnest axis → depth (wall-normal), most-Y-aligned → height,
    #      remaining → width (wall-free).  Single rotation aligns all three axes.
    #   2. flip_z applied after OBB as a wall-normal-axis flip.
    #   3. Non-uniform scale to exact (depth_m × height_m × width_m).
    #   4. Translate, snap back face to wall, clamp, early return.
    # This replaces the old SVD plane-normal de-tilt + post-hoc flatten/90°/
    # fractional-upright pipeline which left large residual tilts.
    if not detilt:
        _na = 2 if wall in ("back", "front") else 0   # wall-normal axis index
        _fa = 0 if wall in ("back", "front") else 2   # wall-free   axis index
        _wall_n = np.zeros(3); _wall_n[_na] = 1.0
        _wall_f = np.zeros(3); _wall_f[_fa] = 1.0

        try:
            _c  = verts.mean(axis=0)
            _cv = verts - _c
            _, _sv, _Vt = np.linalg.svd(_cv, full_matrices=False)
            # numpy svd: singular values are descending → Vt[2] = smallest SV = thinnest axis
            _ax_d = _Vt[2].copy()          # depth / thinnest → wall normal
            _ax_a = _Vt[0].copy()          # largest spread
            _ax_b = _Vt[1].copy()          # medium spread
            # Assign height vs width.
            if force_upright:
                # force_upright (windows/doors): ignore SVD axes for height/width.
                # Force height axis exactly to world-Y projected onto the
                # depth-perpendicular plane.  This guarantees a perfectly rectangular
                # frame in the wall plane regardless of perspective shear baked into
                # the GLB (SVs often nearly equal for square-ish windows, making
                # SVD axis assignment numerically unstable and the frame appear tilted).
                _wy   = np.array([0., 1., 0.])
                _ax_h = _wy - np.dot(_wy, _ax_d) * _ax_d   # project out depth
                _ah_n = np.linalg.norm(_ax_h)
                if _ah_n > 1e-6:
                    _ax_h /= _ah_n
                else:
                    _ax_h = _ax_a if abs(_ax_a[1]) >= abs(_ax_b[1]) else _ax_b
                _ax_w = np.cross(_ax_h, _ax_d)
                _aw_n = np.linalg.norm(_ax_w)
                if _aw_n > 1e-6:
                    _ax_w /= _aw_n
            else:
                # Primary: Y-alignment (the 3D generator objects are captured in Y-up world space,
                # so the height axis of the object aligns with Y in the GLB).
                # Override with dimension-based ONLY when SVs are clearly different
                # (ratio > 1.3) AND target dimensions are clearly different — in that
                # case the SV magnitudes reliably reflect the aspect ratio.
                # When SVs are nearly equal (e.g. window: 87.98 vs 85.74, ratio=1.03),
                # the larger SV does NOT reliably indicate which axis is vertical;
                # Y-alignment is the correct signal.
                _fh = float(size.get("height_m", 1.0))
                _fw = float(size.get("width_m",  _fh))
                _fat = 0.05 * max(_fh, _fw)
                _sv_ratio_hw = _sv[0] / max(_sv[1], 1e-6)   # SV[0]/SV[1], always ≥ 1
                _use_dim_based = (abs(_fh - _fw) > _fat) and (_sv_ratio_hw > 1.3)
                if _use_dim_based:
                    # SVs clearly reflect aspect → larger SV maps to larger target dim
                    if _fh >= _fw:
                        _ax_h, _ax_w = _ax_a, _ax_b   # height is larger
                    else:
                        _ax_h, _ax_w = _ax_b, _ax_a   # width is larger
                else:
                    # Y-alignment: height axis is the one more aligned with world-up
                    if abs(_ax_a[1]) >= abs(_ax_b[1]):
                        _ax_h, _ax_w = _ax_a, _ax_b
                    else:
                        _ax_h, _ax_w = _ax_b, _ax_a
            # Canonicalize signs: depth → +Z (the 3D generator back face at +Z), height → +Y
            # Using _wall_n for depth sign is wrong for left/right walls (wall_n=X,
            # depth=Z → dot≈0, arbitrary sign).  +Z is deterministic for all walls.
            _glb_z_f = np.array([0., 0., 1.])
            if np.dot(_ax_d, _glb_z_f) < 0: _ax_d = -_ax_d
            if _ax_h[1]                < 0:  _ax_h = -_ax_h
            if np.dot(_ax_w, _wall_f)  < 0:  _ax_w = -_ax_w
            # Build rotation: OBB frame → wall frame
            _src = np.column_stack([_ax_d, _ax_h, _ax_w])   # src cols = OBB axes
            _dst = np.column_stack([_wall_n, [0., 1., 0.], _wall_f])  # dst cols
            _R   = _dst @ _src.T
            verts = _cv @ _R.T     # re-centered and aligned to wall frame
            print(f"  [place] OBB de-tilt: SV={np.round(_sv, 3).tolist()}")
        except Exception:
            pass   # fallback: no rotation (object placed as-is)

        # flip_z: swap front vs back face by negating the wall-normal axis.
        # After OBB (depth canonicalized toward +Z → +wall_n): front (the 3D generator min-Z)
        # ends up at min-_na.  For back/left walls snap puts min at wall, so
        # flip_z=True moves front to max-_na (room).  For right/front walls snap
        # puts MAX at wall, so the same logic is inverted: use XOR formula.
        if flip_z ^ (wall in ("right", "front")):
            verts[:, _na] = -verts[:, _na]

        # Non-uniform scale to exact target dimensions
        _td  = float(size.get("depth_m",  0.05))
        _th  = float(size.get("height_m", 1.0))
        _tw  = float(size.get("width_m",  _th))
        _ext_na = max(float(verts[:, _na].max() - verts[:, _na].min()), 1e-6)
        _ext_h  = max(float(verts[:,  1].max() - verts[:,  1].min()), 1e-6)
        _ext_fa = max(float(verts[:, _fa].max() - verts[:, _fa].min()), 1e-6)
        # Safety: if OBB mapped height/width to wrong axes, detect via aspect ratio
        # and swap with a 90° wall-plane rotation.
        # Skip for window/door (skip_obb_swap=True): frame-snap already corrects
        # geometry to exact mask extents, so the swap only corrupts vertex-color
        # appearance without any geometric benefit.
        _want_portrait = _th >= _tw
        _is_portrait   = _ext_h >= _ext_fa
        if _want_portrait != _is_portrait and not skip_obb_swap:
            _tmp          = verts[:, _fa].copy()
            verts[:, _fa] = verts[:, 1]
            verts[:, 1]   = -_tmp
            _ext_h, _ext_fa = _ext_fa, _ext_h
            print(f"  [place] OBB 90° wall-plane safety-swap: "
                  f"{'→portrait' if _want_portrait else '→landscape'}")
        verts[:, _na] *= _td / _ext_na
        if force_upright:
            # For rectified window/door planes, the plane's H/W ratio is the TRUE
            # object aspect (from perspective-unwarped SAM crop).  Preserve it by
            # applying a uniform in-plane scale factor — if we scaled h and w
            # independently to match the VLM's size_m estimate, we'd stretch the
            # rectified texture back into the distortion we just removed.
            # Use the smaller of (target_h/plane_h, target_w/plane_w) so the
            # scaled frame fits within both target dimensions.
            # SCENEWEAVE_WINDOW_FULL: when the window silhouette is unreliable
            # (clipped at the image edge and/or occluded by curtains) the
            # mask-derived target height undershoots the true floor-to-ceiling
            # window.  Opt-in override (value = fraction of ceiling height, e.g.
            # 0.8) forces the window to a large floor-to-ceiling size; width
            # follows via the uniform in-plane scale below.  force_upright is
            # only True for window/door planes.
            _wf = os.environ.get("SCENEWEAVE_WINDOW_FULL")
            if _wf and "window" in Path(glb_path).stem.lower():
                _full_h = float(room.get("ceiling_height_m", 2.7)) * float(_wf)
                if _full_h > _th:
                    print(f"  [place] WINDOW_FULL override: target height "
                          f"{_th:.3f}→{_full_h:.3f}m (floor-to-ceiling)")
                    _th = _full_h
            _sh = _th / _ext_h
            _sw = _tw / _ext_fa
            # For rectified window/door planes: prefer fitting to HEIGHT
            # (silhouette top→bottom is reliable; the bbox width is often
            # foreshortened by photo-edge clipping or partial occlusion).
            # Using `_sh` as the uniform in-plane scale fills the silhouette
            # height and lets the resulting width follow the rectified GLB's
            # intrinsic aspect — which is what you want when the rectified
            # texture is already correctly proportioned.
            _s_inplane = _sh
            verts[:,  1]  *= _s_inplane
            verts[:, _fa] *= _s_inplane
            print(f"  [place] OBB scale (rectified, uniform, fit-to-height): "
                  f"depth {_ext_na:.3f}→{_td:.3f}m  "
                  f"inplane ×{_s_inplane:.3f} "
                  f"(h: {_ext_h*_s_inplane:.3f}/{_th:.3f}, "
                  f"w: {_ext_fa*_s_inplane:.3f}/{_tw:.3f}  "
                  f"[vs sw={_sw:.3f}])")
        else:
            verts[:,  1]  *= _th / _ext_h
            verts[:, _fa] *= _tw / _ext_fa
            print(f"  [place] OBB scale: depth {_ext_na:.3f}→{_td:.3f}m  "
                  f"h {_ext_h:.3f}→{_th:.3f}m  w {_ext_fa:.3f}→{_tw:.3f}m")

        # Translate to placement centre
        _W   = float(room["floor_width_m"])
        _D   = float(room["floor_depth_m"])
        _ceil = float(room.get("ceiling_height_m", 2.7))
        verts += world_pt

        # Snap back face flush to wall surface (outward=False → protrudes into room)
        _snap_outward = outward or embed
        if _snap_outward:
            if   wall == "back":  verts[:, 2] -= verts[:, 2].max()
            elif wall == "front": verts[:, 2] += _D - verts[:, 2].min()
            elif wall == "left":  verts[:, 0] -= verts[:, 0].max()
            elif wall == "right": verts[:, 0] += _W - verts[:, 0].min()
        else:
            if   wall == "back":  verts[:, 2] -= verts[:, 2].min()
            elif wall == "front": verts[:, 2] += _D - verts[:, 2].max()
            elif wall == "left":  verts[:, 0] -= verts[:, 0].min()
            elif wall == "right": verts[:, 0] += _W - verts[:, 0].max()
            # For window/door planes (force_upright) nudge 3 mm into the room so the
            # back face never exactly coincides with the wall polygon (z-fighting).
            if force_upright:
                _nudge = 0.003
                if   wall == "back":  verts[:, 2] += _nudge
                elif wall == "front": verts[:, 2] -= _nudge
                elif wall == "left":  verts[:, 0] += _nudge
                elif wall == "right": verts[:, 0] -= _nudge

        # Ceiling / floor clamp
        _overshoot = verts[:, 1].max() - _ceil
        if _overshoot > 0: verts[:, 1] -= _overshoot
        _floor_gap = -float(verts[:, 1].min())
        if _floor_gap > 0: verts[:, 1] += _floor_gap
        # Doors always reach the floor — mask/OBB sizing can leave the placed
        # centre floating mid-wall (bottom above y=0) when the door's true
        # floor contact is occluded by furniture; snap it down unconditionally
        # rather than only correcting the below-floor case above.
        if force_floor:
            verts[:, 1] -= float(verts[:, 1].min())

        # Room-bounds clamp along free axis (never distort individual vertices)
        if not embed:
            if wall in ("back", "front"):
                _lo = float(verts[:, 0].min())
                if _lo < 0.0:  verts[:, 0] -= _lo
                _hi = float(verts[:, 0].max())
                if _hi > _W:   verts[:, 0] -= (_hi - _W)
            else:
                _lo = float(verts[:, 2].min())
                if _lo < 0.0:  verts[:, 2] -= _lo
                _hi = float(verts[:, 2].max())
                if _hi > _D:   verts[:, 2] -= (_hi - _D)

        vc  = _get_vertex_colors(mesh, glb_path)
        _s  = _th / max(float(robust[1]), 1e-6)
        print(f"  [place] {glb_path.name}: centre={np.round(world_pt,3).tolist()}  scale={_s:.3f}")
        return verts, mesh.faces, vc

    # ── 3D-object OBB placement (detilt=True) ─────────────────────────────────
    # For 3D wall-mounted objects (cabinet, shelf, light, …):
    #   Unlike flat objects where thinnest SVD axis = depth, 3D objects can be
    #   shorter than they are deep (e.g. a wide shelf 0.4m tall × 0.3m deep).
    #   We instead pick the SVD axis MOST ALIGNED WITH THE WALL NORMAL as depth,
    #   then the most-Y-aligned remaining axis as height, and rotate once to align
    #   all three to the wall frame.  flip_z swaps front/back.  Uniform scale by
    #   height preserves all proportions.  Early return, same as flat objects.
    if detilt:
        if wall not in ("back", "front", "left", "right"):
            return None
        _na = 2 if wall in ("back", "front") else 0
        _fa = 0 if wall in ("back", "front") else 2
        _wall_n = np.zeros(3); _wall_n[_na] = 1.0
        _wall_f = np.zeros(3); _wall_f[_fa] = 1.0

        # NOTE: front_yaw_offset is NOT pre-applied here.  The OBB "most-Z-aligned"
        # heuristic uses the original GLB Z axis as the depth reference (the 3D generator
        # convention: depth = GLB-Z).  Pre-rotating by ±90° would swap the depth
        # and width axes before SVD, causing the shelf/cabinet to orient 90° wrong.
        # The front_yaw_offset is already folded into glb_front_deg (applied as a
        # world-space yaw rotation after OBB de-tilt below).

        try:
            _c  = verts.mean(axis=0)
            _cv = verts - _c
            _, _sv, _Vt = np.linalg.svd(_cv, full_matrices=False)
            # Depth axis = thinnest SVD axis (smallest singular value).
            # Wall-mounted objects always have depth << width and depth << height,
            # so the thinnest physical dimension reliably identifies depth.
            # This is GLB-axis-convention-independent (works whether depth is along
            # X, Y, or Z in the GLB), unlike "most-Z-aligned" which breaks when
            # the GLB has depth along X (e.g. cabinet GLBs from the 3D generator inpainting).
            _idx_d  = int(np.argmin(_sv))
            _remain = [i for i in range(3) if i != _idx_d]
            # Assign height vs width among the remaining two axes.
            # Primary: Y-alignment (generated GLBs captured in Y-up world space,
            # so the height axis of the object aligns with Y in the GLB).
            # Override with dimension-based ONLY when SVs are clearly different
            # (ratio > 1.3) AND target dimensions are clearly different.
            # For nearly-equal SVs (e.g. 54.188 vs 53.176, ratio=1.019) the
            # larger SV is an unreliable height indicator; Y-alignment is better.
            _height_m_t = float(size.get("height_m", 1.0))
            _width_m_t  = float(size.get("width_m",  _height_m_t))
            _sv_rem     = [float(_sv[i]) for i in _remain]
            _aspect_tol = 0.05 * max(_height_m_t, _width_m_t)
            _sv_rem_hi  = max(_sv_rem); _sv_rem_lo = min(_sv_rem)
            _sv_ratio_3d = _sv_rem_hi / max(_sv_rem_lo, 1e-6)
            _use_dim_3d  = (abs(_height_m_t - _width_m_t) > _aspect_tol) and (_sv_ratio_3d > 1.3)
            if _use_dim_3d:
                if _height_m_t >= _width_m_t:
                    _idx_h = _remain[int(np.argmax(_sv_rem))]
                else:
                    _idx_h = _remain[int(np.argmin(_sv_rem))]
            else:
                _dots_y = [abs(float(_Vt[i][1])) for i in _remain]
                _idx_h  = _remain[int(np.argmax(_dots_y))]
            _idx_w  = [i for i in _remain if i != _idx_h][0]
            _ax_d = _Vt[_idx_d].copy()
            _ax_h = _Vt[_idx_h].copy()
            _ax_w = _Vt[_idx_w].copy()
            # Canonicalize depth sign toward +Z (generated GLBs depth = Z convention).
            # Using _wall_n is WRONG for left/right walls: wall_n=X but depth≈Z,
            # so dot≈0 → arbitrary sign → 50% front/back reversal.
            _glb_z = np.array([0., 0., 1.])
            if np.dot(_ax_d, _glb_z)  < 0: _ax_d = -_ax_d
            if _ax_h[1]                < 0: _ax_h = -_ax_h
            if np.dot(_ax_w, _wall_f)  < 0: _ax_w = -_ax_w
            _src = np.column_stack([_ax_d, _ax_h, _ax_w])
            _dst = np.column_stack([_wall_n, [0., 1., 0.], _wall_f])
            _R   = _dst @ _src.T
            verts = _cv @ _R.T
            print(f"  [place] OBB de-tilt (3D): "
                  f"thin-sv={_sv[_idx_d]:.3f}  SV={np.round(_sv, 3).tolist()}  "
                  f"axes=(d={_idx_d},h={_idx_h},w={_idx_w})  "
                  f"yaw_off={front_yaw_offset}  flip_z={flip_z}")
        except Exception:
            pass

        # flip_z: negate wall-normal axis to swap front vs back face.
        # After OBB (depth canonicalized toward +Z → maps to +wall_n):
        #   - the 3D generator front face (min-Z) → min-_na
        #   - Snap: back/left walls put MIN-_na at wall surface
        #           front/right walls put MAX-_na at wall surface
        # For back/left: flip_z=True (VLM panel B, front at -Z) moves front from
        #   min-_na to max-_na (room side), leaving back at min-_na (wall). ✓
        # For right/front: snap puts MAX at wall, so WITHOUT flip we already have
        #   front (min-_na) in room and back (max-_na) at wall. flip_z must be
        #   INVERTED — effective flip = VLM flip_z XOR (wall in right/front).
        _flip_effective = flip_z ^ (wall in ("right", "front"))
        if _flip_effective:
            verts[:, _na] = -verts[:, _na]

        # Non-uniform scale to the estimated size_m (silhouette-authoritative),
        # mirroring the flat-object path. Scaling by HEIGHT ONLY inherited the
        # Hunyuan mesh's native aspect ratio, which is often wrong (e.g. a cabinet
        # meshed at ~4:1 W/H ballooned to 3.7m wide when fit to a 0.92m height,
        # far exceeding its mask). Axes are already wall-aligned here:
        # _na = wall-normal (depth), 1 = height, _fa = free axis (width).
        _th = float(size["height_m"])
        _tw = float(size.get("width_m", _th))
        _td = float(size.get("depth_m", 0.35))
        _ext_na = max(float(verts[:, _na].max() - verts[:, _na].min()), 1e-6)
        _ext_h  = max(float(verts[:, 1].max()  - verts[:, 1].min()),  1e-6)
        _ext_fa = max(float(verts[:, _fa].max() - verts[:, _fa].min()), 1e-6)
        verts[:, _na] *= _td / _ext_na
        verts[:, 1]   *= _th / _ext_h
        verts[:, _fa] *= _tw / _ext_fa
        _s = _th / _ext_h   # representative (height) scale, for the log line below

        # Translate to placement centre
        _W   = float(room["floor_width_m"])
        _D   = float(room["floor_depth_m"])
        _ceil = float(room.get("ceiling_height_m", 2.7))
        verts += world_pt

        # Snap back face to wall surface
        _snap_outward = outward or embed
        if _snap_outward:
            if   wall == "back":  verts[:, 2] -= verts[:, 2].max()
            elif wall == "front": verts[:, 2] += _D - verts[:, 2].min()
            elif wall == "left":  verts[:, 0] -= verts[:, 0].max()
            elif wall == "right": verts[:, 0] += _W - verts[:, 0].min()
        else:
            if   wall == "back":  verts[:, 2] -= verts[:, 2].min()
            elif wall == "front": verts[:, 2] += _D - verts[:, 2].max()
            elif wall == "left":  verts[:, 0] -= verts[:, 0].min()
            elif wall == "right": verts[:, 0] += _W - verts[:, 0].max()

        # Ceiling / floor clamp
        _overshoot = verts[:, 1].max() - _ceil
        if _overshoot > 0: verts[:, 1] -= _overshoot
        _floor_gap = -float(verts[:, 1].min())
        if _floor_gap > 0: verts[:, 1] += _floor_gap

        # Room-bounds clamp (free axis)
        if not embed:
            if wall in ("back", "front"):
                _lo = float(verts[:, 0].min())
                if _lo < 0.0:  verts[:, 0] -= _lo
                _hi = float(verts[:, 0].max())
                if _hi > _W:   verts[:, 0] -= (_hi - _W)
            else:
                _lo = float(verts[:, 2].min())
                if _lo < 0.0:  verts[:, 2] -= _lo
                _hi = float(verts[:, 2].max())
                if _hi > _D:   verts[:, 2] -= (_hi - _D)

        vc  = _get_vertex_colors(mesh, glb_path)
        print(f"  [place] {glb_path.name}: centre={np.round(world_pt,3).tolist()}  scale={_s:.3f}")
        return verts, mesh.faces, vc

    target_h = float(size["height_m"])

    # Scale uniformly by height — preserves all proportions including depth.
    s = target_h / max(robust[1], 1e-6)
    verts *= s

    # Apply rotation.
    # Priority 1: VLM-determined non-zero angle — direct world-space yaw.
    # Priority 2: bbox-align (or wall-heuristic) + VLM disambiguation for 180°.
    # Priority 3: wall-normal heuristic (last resort).
    _wall_heuristic = {"back": 0, "front": 180, "left": 90, "right": 270}.get(wall, 0)
    _wall_out = {"back": np.array([0,0,1.]), "front": np.array([0,0,-1.]),
                 "left": np.array([1,0,0.]),  "right": np.array([-1,0,0.])}.get(wall, np.array([0,0,1.]))
    if glb_front_deg is not None and glb_front_deg != 0 and detilt:
        # VLM direct yaw — skip for windows/curtains which need 90° snap
        yaw = glb_front_deg
        print(f"  [orient] VLM yaw={yaw}°")
    elif cam_pos is not None:
        raw_yaw = _bbox_align_yaw(verts, wall, cam_pos, world_pt)
        # All wall-mounted objects must be axis-aligned (face parallel to wall surface).
        # Snap the continuous bbox-align angle to the nearest 90°-multiple for all objects,
        # then resolve 180° front-vs-back ambiguity using:
        #   detilt=False (flat): face mean vs wall-outward
        #   detilt=True  (3D):   VLM glb_front_deg if available, else face mean
        depth_col = 2 if wall in ("back", "front") else 0
        candidates = [0, 90, 180, 270]
        best_snap, best_span = 0, float("inf")
        for c in candidates:
            rv = verts @ _rot_y(math.radians(c)).T
            span = float(rv[:, depth_col].max() - rv[:, depth_col].min())
            if span < best_span:
                best_span = span
                best_snap = c
        if detilt and glb_front_deg is not None:
            # 3D objects: use VLM glb_front_deg to resolve 180° ambiguity.
            fr = math.radians(glb_front_deg)
            front_glb = np.array([math.sin(fr), 0., math.cos(fr)])
            sc0   = float(_rot_y(math.radians(best_snap))         @ front_glb @ _wall_out)
            sc180 = float(_rot_y(math.radians((best_snap+180)%360)) @ front_glb @ _wall_out)
            yaw = (best_snap + 180) % 360 if sc180 > sc0 else best_snap
        else:
            # Flat objects or no VLM: pick between best_snap and best_snap+180° using
            # which rotation puts more of the "front half" toward the room.
            rv0   = verts @ _rot_y(math.radians(best_snap)).T
            rv180 = verts @ _rot_y(math.radians((best_snap + 180) % 360)).T
            fh0   = rv0[rv0[:, depth_col]   > rv0[:,   depth_col].mean()]
            fh180 = rv180[rv180[:, depth_col] > rv180[:, depth_col].mean()]
            s0   = float(np.mean(fh0   @ _wall_out)) if len(fh0)   else 0.0
            s180 = float(np.mean(fh180 @ _wall_out)) if len(fh180) else 0.0
            yaw  = (best_snap + 180) % 360 if s180 > s0 else best_snap
        print(f"  [orient] bbox-align-snapped yaw={yaw}° (raw={raw_yaw}°)")
    else:
        yaw = _wall_heuristic
        print(f"  [orient] wall-heuristic yaw={yaw}°")

    verts = verts @ _rot_y(math.radians(yaw)).T

    # NOTE: flat objects (detilt=False) returned early from the OBB block above.
    # Everything below runs only for 3D objects (detilt=True).

    if wall not in ("back", "front", "left", "right"):
        return None

    W     = float(room["floor_width_m"])
    D     = float(room["floor_depth_m"])
    ceil  = float(room.get("ceiling_height_m", 2.7))

    # Non-uniform width scaling for embed objects (windows/doors):
    # After yaw rotation the world axes are:
    #   back/front walls → width=X, height=Y, depth=Z
    #   left/right walls → width=Z, height=Y, depth=X
    # Scale the width axis to exactly match size["width_m"] so the frame
    # fills the opening precisely.
    if embed:
        target_w = float(size.get("width_m", 0))
        if target_w > 0:
            if wall in ("back", "front"):
                cur_span = max(float(verts[:, 0].max() - verts[:, 0].min()), 1e-6)
                verts[:, 0] *= target_w / cur_span
            else:  # left / right
                cur_span = max(float(verts[:, 2].max() - verts[:, 2].min()), 1e-6)
                verts[:, 2] *= target_w / cur_span

    center = world_pt.copy()
    verts += center

    # Position along wall-normal axis.
    # outward=True (or embed=True) → snap room-facing (max) face to wall surface;
    #   object extends outward through the wall to the exterior.
    # outward=False               → snap back (min) face to wall surface;
    #   object protrudes into the room (art, curtain, light).
    snap_outward = outward or embed
    if snap_outward:
        if wall == "back":
            verts[:, 2] -= verts[:, 2].max()   # room-facing (max Z) at Z=0, depth toward -Z
        elif wall == "front":
            verts[:, 2] += D - verts[:, 2].min()  # room-facing (min Z) at Z=D, depth toward +Z
        elif wall == "left":
            verts[:, 0] -= verts[:, 0].max()   # room-facing (max X) at X=0, depth toward -X
        elif wall == "right":
            verts[:, 0] += W - verts[:, 0].min()  # room-facing (min X) at X=W, depth toward +X
    else:
        # Snap back face flush to wall — object protrudes into room
        if wall == "back":
            verts[:, 2] -= verts[:, 2].min()
        elif wall == "front":
            verts[:, 2] += D - verts[:, 2].max()
        elif wall == "left":
            verts[:, 0] -= verts[:, 0].min()
        elif wall == "right":
            verts[:, 0] += W - verts[:, 0].max()

    # Shift down if above ceiling
    overshoot = verts[:, 1].max() - ceil
    if overshoot > 0:
        verts[:, 1] -= overshoot

    # Slide the object as a rigid body to stay within room bounds.
    # Never clamp individual vertices — that distorts the mesh geometry.
    if not embed:
        if wall in ("back", "front"):  # free axis is X
            lo = float(verts[:, 0].min())
            if lo < 0.0:
                verts[:, 0] -= lo          # slide right until min X = 0
            hi = float(verts[:, 0].max())
            if hi > W:
                verts[:, 0] -= (hi - W)   # slide left until max X = W
        else:  # left / right — free axis is Z
            lo = float(verts[:, 2].min())
            if lo < 0.0:
                verts[:, 2] -= lo
            hi = float(verts[:, 2].max())
            if hi > D:
                verts[:, 2] -= (hi - D)
    # Lift off floor if any vertex is below Y=0
    floor_gap = -float(verts[:, 1].min())
    if floor_gap > 0:
        verts[:, 1] += floor_gap

    vc = _get_vertex_colors(mesh)
    print(f"  [place] {glb_path.name}: centre={np.round(center,3).tolist()}  scale={s:.3f}")
    return verts, mesh.faces, vc


# ─────────────────────────────────────────────────────────────────────────────
# Main placement pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    output_dir:  str | Path,
    image_path:  str | Path,
    target_types: list[str] | None = None,
    limit: int | None = None,
    center_only: bool = False,
    use_vlm: bool = True,
) -> None:
    out_dir      = Path(output_dir)
    openings_dir = out_dir / "openings"
    wm_dir       = out_dir / "wall_mounted"
    obj_dir      = wm_dir / "objects"
    place_dir    = wm_dir / "placements"
    place_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)

    # ── Camera ────────────────────────────────────────────────────────────────
    for _cp in [out_dir / "camera_vggt.json",
                out_dir / "camera.json",
                openings_dir / "camera.json"]:
        if _cp.exists():
            cam_path = _cp
            break
    else:
        raise FileNotFoundError(f"No camera.json found in {out_dir}")
    cam = json.loads(cam_path.read_text())

    # ── Room dimensions (from actual walls OBJ, not floorplan_analysis) ───────
    analysis = {}
    anal_path = out_dir / "floorplan_analysis.json"
    if anal_path.exists():
        analysis = json.loads(anal_path.read_text())
    room = dict(analysis.get("room", {}))
    base_obj = (openings_dir / "walls_with_openings.obj"
                if (openings_dir / "walls_with_openings.obj").exists()
                else out_dir / "walls.obj")
    obj_dims = _room_dims_from_obj(base_obj)
    if obj_dims:
        room.update(obj_dims)

    W_m   = float(room["floor_width_m"])
    D_m   = float(room["floor_depth_m"])
    ceil  = float(room.get("ceiling_height_m", 2.7))

    # ── Segments ──────────────────────────────────────────────────────────────
    seg_results = json.loads((wm_dir / "segment_results.json").read_text())
    segments    = seg_results.get("segments", [])
    targets     = [
        s for s in segments
        if (target_types is None or s.get("type") in target_types) and s.get("glb_file")
        and (obj_dir / Path(s["glb_file"]).name).exists()
    ]

    # Load existing placements for VLM-offline orientation fallback.
    # When VLM is unavailable, preserve previously VLM-corrected orientations
    # (flip_y / flip_z) so they're not overwritten by the raw default.
    _existing_placements: dict[int, dict] = {}
    _existing_json = place_dir / "object_placements.json"
    if _existing_json.exists():
        try:
            for _p in json.loads(_existing_json.read_text()):
                _existing_placements[int(_p["segment_index"])] = _p
        except Exception:
            pass
    print(f"[place_objects] {len(targets)} segment(s) of type(s) {target_types} with GLB")
    if limit is not None:
        targets = targets[:limit]
        print(f"[place_objects] --limit {limit}: placing first {len(targets)} segment(s)")

    if not targets:
        print("[place_objects] Nothing to place — exiting.")
        return

    # ── Camera setup ──────────────────────────────────────────────────────────
    pos, right, up_w, fwd_h, tilt_tan, fx, cx, cy, W_px, H_px = _camera_setup(cam)
    _project = _make_projector(pos, np.array(cam["look_at_m"]), up_w,
                               float(cam["hfov_deg"]), W_px, H_px)

    # ── Determine base OBJ (windows may already be placed) ────────────────────
    windows_obj  = place_dir / "walls_with_windows.obj"
    plain_obj    = openings_dir / "walls_with_openings.obj"
    fallback_obj = out_dir / "walls.obj"
    if windows_obj.exists():
        obj_in_path = windows_obj
    elif plain_obj.exists():
        obj_in_path = plain_obj
    else:
        obj_in_path = fallback_obj
    obj_out_path = place_dir / "walls_with_objects.obj"

    orig_lines = Path(obj_in_path).read_text().splitlines(keepends=True)
    n_verts    = sum(1 for ln in orig_lines if ln.strip().startswith("v "))

    new_verts: list[str] = []
    new_faces: list[str] = ["\n# ── Wall-mounted object placements ───────────────\n"]

    placements: list[dict] = []

    # ── Collision state: track placed objects for slide-from-collision ────────
    wall_state: list[dict] = []
    state_path = place_dir / "wall_placements.txt"

    for seg in targets:
        idx      = seg["index"]
        obj_type = seg.get("type", "other")
        glb_path = obj_dir / Path(seg["glb_file"]).name
        x1, y1, x2, y2 = seg["box_px"]

        print(f"\n[place_objects] Segment {idx:02d} ({obj_type}), bbox=[{x1},{y1},{x2},{y2}]")

        # ── 0. Wall-mounted check ───────────────────────────────────────────
        if not _is_wall_mounted_vlm(image_path, seg, W_px, H_px):
            print(f"  [wall_check] not wall-mounted — skipping")
            continue

        # ── 1. Back-project mask centroid → wall + world point ───────────────
        # Use mask centroid if available (more accurate than bbox centre);
        # the mask lives in the same coordinate space as the original image.
        mask_path = _mask_path_for_seg(seg, wm_dir)
        cx_px, cy_px = (x1 + x2) / 2.0, (y1 + y2) / 2.0   # default: bbox centre
        if mask_path.exists():
            try:
                mask_arr = np.array(Image.open(mask_path))
                fg = mask_arr.sum(axis=2) > 0 if mask_arr.ndim == 3 else mask_arr > 0
                ys_m, xs_m = np.where(fg)
                if len(xs_m) > 0:
                    cx_px, cy_px = float(xs_m.mean()), float(ys_m.mean())
                    print(f"  [mask] centroid=({cx_px:.1f},{cy_px:.1f})  bbox centre=({(x1+x2)/2:.1f},{(y1+y2)/2:.1f})")
            except Exception:
                pass

        ray = _backproject(cx_px, cy_px, pos, right, fwd_h, tilt_tan, fx, cx, cy)
        hit = _find_wall(pos, ray, W_m, D_m, ceil)

        if hit is None:
            print("  [place] ray does not hit any wall — skipping")
            continue
        wall, world_pt = hit
        print(f"  [place] wall={wall}  world_pt={np.round(world_pt,3).tolist()}")

        # ── 1b. Near-seam wall adjudication (VLM) ────────────────────────────
        # Geometry locks the wall, but a mask centroid that lands near a corner
        # seam is ambiguous (a left-wall cabinet near the corner projects just
        # past the left/back seam → geometry says back). For those cases ONLY,
        # let the VLM pick between the two candidate walls.
        seam_alt = _seam_candidate(wall, world_pt, W_m, D_m)
        if seam_alt is not None and use_vlm:
            adj_wall = _adjudicate_wall_vlm(image_path, seg, wall, seam_alt, W_px, H_px)
            if adj_wall != wall:
                new_hit = _ray_hit_wall(pos, ray, adj_wall, W_m, D_m, ceil, tol=_WALL_HIT_TOL)
                if new_hit is not None:
                    print(f"  [wall_adjudicate] near seam: geometry={wall} → VLM={adj_wall}  "
                          f"world_pt {np.round(world_pt,3).tolist()} → {np.round(new_hit,3).tolist()}")
                    wall, world_pt = adj_wall, new_hit
                else:
                    print(f"  [wall_adjudicate] VLM chose {adj_wall} but ray misses it — keeping {wall}")

        # ── 2 + 3. VLM determines wall, position, height, rotation in one call ──
        # Renders 8 Y-rotations at rough world_pt through the scene camera and
        # asks VLM for all placement parameters simultaneously.
        glb_front_deg = 0
        init = _vlm_init_placement(
            image_path, seg, glb_path, world_pt, room,
            project_fn=_project, cam_pos=pos, img_w=W_px, img_h=H_px,
        ) if use_vlm else None
        if init is not None:
            # Wall locked to geometry — VLM sees a perspective image and cannot
            # reliably map screen-space regions to world-space wall names.
            geo_wall      = wall
            glb_front_deg = init["front_view_deg"]
            # Use mask back-projection for size+center — more accurate than VLM estimates.
            mask_size = estimate_size_from_mask(
                seg, wm_dir, geo_wall, world_pt, pos, fwd_h, fx, right,
                cy=cy, tilt_tan=tilt_tan, ceiling_h=ceil,
            )
            if mask_size is not None:
                size = mask_size
                # Always override depth for flat objects
                if obj_type in ("window", "door", "curtain"):
                    size["depth_m"] = _TYPE_DEFAULTS.get(obj_type, {}).get("depth_m", 0.05)
            else:
                size = {
                    "height_m": init["object_height_m"],
                    "width_m":  init["object_height_m"] * 1.2,
                    "depth_m":  _TYPE_DEFAULTS.get(obj_type, {}).get("depth_m", 0.05)
                              if obj_type in ("window", "door", "curtain")
                              else init["object_height_m"] * 0.4,
                }
            # Apply mask-derived center_y; keep geometry lateral position
            center_y = float(size.pop("center_y")) if "center_y" in size else float(init["world_pt"][1])
            world_pt = np.array([world_pt[0], center_y, world_pt[2]])
            print(f"  [vlm_init] wall={geo_wall} (geo-locked)  "
                  f"world_pt={np.round(world_pt,3).tolist()}  size={size}")
        else:
            # Fallback: mask geometry / VLM size estimate
            print("  [vlm_init] failed — falling back to mask/VLM size estimation")
            size = estimate_size_from_mask(
                seg, wm_dir, wall, world_pt, pos, fwd_h, fx, right,
                cy=cy, tilt_tan=tilt_tan, ceiling_h=ceil,
            )
            if size is None:
                size = estimate_size_vlm(image_path, seg, room, W_px, H_px)
            if "center_y" in size:
                world_pt = world_pt.copy()
                world_pt[1] = float(size.pop("center_y"))
            if "wall" in size:
                vlm_wall = size["wall"]
                if vlm_wall != wall:
                    new_hit = _ray_hit_wall(pos, ray, vlm_wall, W_m, D_m, ceil)
                    if new_hit is not None:
                        wall, world_pt = vlm_wall, new_hit

        # ── Flip corrections (upside-down / left-right mirror / front-back) ───
        # the 3D generator convention:
        #   flip_y=True  — image Y increases downward; must flip for world-up.
        #   flip_z=True  — the near/visible face is at min Z; the snap in
        #                  _transform_glb_to_wall puts min-Z on the wall, so we
        #                  must flip Z to bring the visible face to face the room.
        #   flip_x=False — no horizontal mirror by default.
        # VLM overrides these when a canvas reference is available.
        orient = {
            "flip_y": True,
            "flip_x": False,
            "flip_z": True,
        }
        # Determine reference image for orientation check.
        # Priority: inpainted crop → canvas crop → bbox crop of the room image.
        # Always run the orientation check so VLM can correct upside-down/face issues.
        canvas_path = None
        canvas_file = seg.get("inpaint_file") or seg.get("canvas_file")
        if canvas_file:
            cp = (wm_dir / "inpainted" / seg["inpaint_file"] if seg.get("inpaint_file")
                  else wm_dir / "segmented" / seg["canvas_file"])
            if cp.exists():
                canvas_path = cp

        ref_for_orient = canvas_path
        if ref_for_orient is None:
            # Fall back to a bbox crop of the original room image saved as a temp file
            try:
                _crop_path = place_dir / f"_orient_ref_{idx:02d}.png"
                _cx1, _cy1, _cx2, _cy2 = x1, y1, x2, y2
                Image.open(image_path).convert("RGB").crop(
                    (_cx1, _cy1, _cx2, _cy2)).save(str(_crop_path))
                ref_for_orient = _crop_path
                print(f"  [orient] no canvas file — using bbox crop as reference")
            except Exception as _e:
                print(f"  [orient] bbox crop failed: {_e}")

        front_yaw_offset = 0
        if ref_for_orient is not None:
            # For flat objects (detilt=False: art, curtain, mirror, etc.) flip_z
            # correctness depends on the yaw that bbox-align will choose later, which
            # in turn depends on flip_z — a circular dependency.  The safe default is
            # flip_z=True (the 3D generator: near/visible face is min-Z; snap puts min-Z at wall,
            # so flip_z=True brings the visible face to face the room).  Only use the
            # VLM to refine flip_y and flip_x; leave flip_z at its default for flat types.
            _is_flat = obj_type in ("window", "door", "curtain",
                                    "art", "frame", "painting")

            # ── Per-segment manual orient override ──────────────────────
            # When the VLM picks the wrong panel for a particular GLB, set
            # `"orient_override": {"flip_y": bool, "flip_x": bool, "flip_z":
            # bool, "yaw_offset_deg": int}` on the segment_results entry to
            # bypass both VLM calls.  Any unset key keeps the default.
            _orient_override = seg.get("orient_override") or {}
            front_face_flip_z = None
            front_yaw_offset  = 0
            if _orient_override:
                if "flip_y" in _orient_override:
                    orient["flip_y"] = bool(_orient_override["flip_y"])
                if "flip_x" in _orient_override:
                    orient["flip_x"] = bool(_orient_override["flip_x"])
                if "flip_z" in _orient_override:
                    orient["flip_z"] = bool(_orient_override["flip_z"])
                    front_face_flip_z = orient["flip_z"]
                if "yaw_offset_deg" in _orient_override:
                    front_yaw_offset = int(_orient_override["yaw_offset_deg"])
                print(f"  [orient_override] {dict(orient)} "
                      f"yaw_off={front_yaw_offset}° (skipping VLM)")
            elif use_vlm:
                # Step A: 4-direction front-face check — authoritative flip_z for 3D objects.
                if not _is_flat:
                    ff = _vlm_front_face_check(glb_path, ref_for_orient, obj_type)
                    if ff is not None:
                        orient["flip_z"]  = ff["flip_z"]
                        front_face_flip_z = ff["flip_z"]
                        front_yaw_offset  = int(ff.get("yaw_offset", 0))

                # Step B: 4-panel grid check — always refines flip_y; refines flip_z only
                # for 3D objects (when front-face check did not already pin it).
                vlm_orient = _vlm_orientation_check(glb_path, ref_for_orient, obj_type)
                if vlm_orient is not None:
                    orient["flip_y"] = vlm_orient["flip_y"]
                    orient["flip_x"] = vlm_orient.get("flip_x", False)
                    if not _is_flat and front_face_flip_z is None:
                        orient["flip_z"] = vlm_orient["flip_z"]
                    # For flat objects: keep default flip_z=True (the 3D generator convention)
                elif idx in _existing_placements:
                    # VLM offline — restore previously VLM-corrected orientation
                    prev_orient = _existing_placements[idx].get("orientation")
                    if prev_orient is not None:
                        orient = prev_orient
                        print(f"  [orient] VLM offline — restored previous orientation: {prev_orient}")

            # ── Wall-consistency guard on the front-face yaw offset ────────────
            # `_vlm_front_face_check` maps its 4-way answer to a yaw_offset of
            # 0 / 180 (front at ±Z) or ±90 (front at ±X).  Which of these is
            # geometrically valid depends on the wall the object sits on:
            #   • back / front walls  → room-facing normal is ±Z → the front
            #     must end up facing ±Z, so a ±90 offset (front rotated onto the
            #     wall-TANGENT ±X axis) turns a flat panel edge-on and renders it
            #     as a tilted parallelogram jutting into the room.  Only 0/180
            #     are admissible.
            #   • left / right walls  → room-facing normal is ±X → ±90 is exactly
            #     what's needed, and 0/180 would be edge-on instead.
            # The VLM (especially when its render is ambiguous or the call times
            # out and falls back) sometimes returns the wrong family for the
            # wall.  Reject an offset whose axis-family contradicts the wall so a
            # mis-pick can never produce the floating-tilted-TV artifact.  This
            # is geometry — not a heuristic — so it's safe for every object type.
            _wall_normal_is_z = wall in ("back", "front")
            _offset_on_tangent = abs(int(front_yaw_offset) % 180) == 90  # ±90/±270
            if front_yaw_offset != 0 and _offset_on_tangent == _wall_normal_is_z:
                # Family mismatch: drop the tangential/normal rotation.  Preserve
                # only the 180° front/back ambiguity component (valid on any wall).
                _kept = 180 if (int(front_yaw_offset) % 360) in (180, 270) else 0
                print(f"  [front_face] yaw_offset {front_yaw_offset:+d}° is "
                      f"incompatible with a {wall}-wall normal "
                      f"({'±Z' if _wall_normal_is_z else '±X'}) — "
                      f"clamping to {_kept}° to keep the panel flush")
                front_yaw_offset = _kept

            if front_yaw_offset != 0:
                glb_front_deg = (glb_front_deg + front_yaw_offset) % 360
                print(f"  [front_face] applied yaw_offset {front_yaw_offset:+d}° → "
                      f"glb_front_deg={glb_front_deg}")

            # Mirror-check for art/painting/frame is now done POST-render
            # via correlation between the placed-painting crop and the
            # inpaint texture (see `_vlm_check_placed_painting_mirror`
            # called after `_render_objects`).  The previous in-loop
            # VLM-based check on the GLB-isolated render was unreliable
            # (it kept saying "abstract, no asymmetric features" or
            # "consistent" for clearly-flipped paintings) and setting
            # flip_x=True via orient didn't visibly mirror the rendered
            # plane either.  The post-render correlation check + GLB
            # mutation is more direct and robust.

        # Window/door plane GLBs are Y-up — override VLM flip_y/flip_x and save back.
        if obj_type in ("window", "door"):
            orient = dict(orient)
            orient["flip_y"] = False
            orient["flip_x"] = False

        # ── 3b. Collision avoidance — slide along wall to avoid overlap ─────
        slid = _slide_from_collision(wall, world_pt, size, room, wall_state, idx)
        if slid is not None:
            world_pt = slid

        # ── 4. Transform GLB ─────────────────────────────────────────────────
        if center_only:
            result = _place_at_center(glb_path, size, room,
                                      cam_pos=pos, glb_front_deg=glb_front_deg)
        else:
            result = _transform_glb_to_wall(
                glb_path, wall, world_pt, size, room,
                cam_pos=pos,
                flip_y=orient["flip_y"], flip_x=orient["flip_x"],
                flip_z=orient.get("flip_z", True),
                glb_front_deg=glb_front_deg,
                front_yaw_offset=front_yaw_offset,
                outward=False,   # window/door are planes flush on wall
                # Flat wall-hung types use SVD+flatten path (detilt=False):
                # windows, doors, curtains, art, frames, paintings.
                # Mirrors have framed depth → use 3D OBB path (detilt=True).
                detilt=(obj_type not in ("window", "door", "curtain",
                                         "art", "frame", "painting")),
                skip_obb_swap=(obj_type in ("window", "door")),
                force_upright=(obj_type in ("window", "door")),
                force_floor=(obj_type == "door"),
            )
        if result is None:
            print("  [place] transform failed — skipping")
            continue
        verts, faces, vc = result

        # ── 4b. Wall-lamp facing-direction check ──────────────────────────────
        # For light objects: render the placed lamp from the scene camera and
        # ask VLM if it faces the same horizontal direction as the reference.
        # If not, mirror the already-placed vertices along the wall-tangential
        # axis (X for back/front walls, Z for left/right walls).
        # NOTE: we do NOT re-run _transform_glb_to_wall because glb_front_deg==0
        # falls through to bbox_align inside that function, so "+180°" would not
        # give a correct 180° flip of the actual placement.  Direct vertex
        # mirroring is exact regardless of the original rotation pipeline.
        if obj_type == "light" and not center_only:
            ref_for_facing = canvas_path or image_path
            needs_flip = _vlm_lamp_facing_check(
                ref_for_facing, [x1, y1, x2, y2],
                verts, faces, vc, _project, W_px, H_px,
                wall=wall,
                is_canvas=(canvas_path is not None),
            )
            if needs_flip:
                print(f"  [lamp_facing] mirroring along wall-tangential axis "
                      f"(wall={wall})")
                flipped = verts.copy()
                if wall in ("back", "front"):
                    # tangential axis = X; mirror around lamp centre X
                    flipped[:, 0] = 2.0 * world_pt[0] - verts[:, 0]
                else:  # left / right
                    # tangential axis = Z; mirror around lamp centre Z
                    flipped[:, 2] = 2.0 * world_pt[2] - verts[:, 2]
                verts = flipped
                orient = dict(orient)
                orient["lamp_facing_flipped"] = True
                orient["lamp_facing_wall"] = wall
                orient["lamp_facing_world_pt"] = world_pt.tolist()

        # ── 4c. Wall-normal lean correction ──────────────────────────────────
        # the 3D generator objects often lean: the bottom touches the wall but the top
        # floats away (or vice versa). Rotate around the bottom pivot to flush
        # the object end-to-end against the wall — same as run_corner_based().
        # 3D objects (lights, cabinets, shelves) naturally protrude from the
        # wall; that protrusion is NOT tilt — cap correction to 3° for them.
        _flat_types = ("window", "door", "curtain", "art", "frame", "painting")
        _tilt_cap   = None if obj_type in _flat_types else 3.0
        verts, tilt_theta_deg = _apply_tilt_correction(
            verts, wall, obj_type, W_m, D_m, max_theta_deg=_tilt_cap)

        # ── 5. Append to OBJ ─────────────────────────────────────────────────
        base = n_verts + 1
        new_faces.append(f"# segment {idx:02d} ({obj_type})\n")
        for x, y, z in verts:
            new_verts.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for tri in faces:
            i0, i1, i2 = int(tri[0]) + base, int(tri[1]) + base, int(tri[2]) + base
            new_faces.append(f"f {i0} {i1} {i2}\n")
        n_verts += len(verts)

        # Project centre for JSON record
        centre_pt = _project(world_pt)
        placements.append({
            "segment_index": idx,
            "type":          obj_type,
            "wall":          wall,
            "world_pt":      world_pt.tolist(),
            "projected_px":  [centre_pt[0], centre_pt[1]] if centre_pt else None,
            "bbox_px":       [x1, y1, x2, y2],
            "size_m":        {k: size[k] for k in ("width_m", "height_m", "depth_m")},
            "glb_file":      str(Path(seg["glb_file"]).name),
            "orientation":   orient,
            "glb_front_deg": glb_front_deg,
            "tilt_theta_deg": tilt_theta_deg,
        })
        print(f"  [place] {len(verts)} verts, {len(faces)} faces appended")

        # Record in collision state
        f_min, f_max, z_min, z_max = _wall_bbox(wall, world_pt, size)
        wall_state.append({
            "seg_idx": idx, "wall": wall,
            "free_min": f_min, "free_max": f_max,
            "z_min": z_min, "z_max": z_max,
        })

    # ── Save collision state ─────────────────────────────────────────────────
    _save_wall_state(wall_state, state_path)

    # ── Write combined OBJ ────────────────────────────────────────────────────
    with open(obj_out_path, "w") as f:
        f.writelines(orig_lines)
        f.writelines(new_verts)
        f.writelines(new_faces)
    print(f"\n[place_objects] Scene mesh → {obj_out_path}")

    # ── Save placements JSON ──────────────────────────────────────────────────
    placements_path = place_dir / "object_placements.json"
    with open(placements_path, "w") as f:
        json.dump(placements, f, indent=2)
    print(f"[place_objects] Placements → {placements_path}")

    # ── Render ────────────────────────────────────────────────────────────────
    render_path = _render_objects(
        str(obj_out_path), cam_path, place_dir, out_dir, placements, room)

    # ── Post-render mirror check ─────────────────────────────────────────────
    # The pre-placement GLB-isolated mirror check is unreliable because it
    # lacks room context.  After the full room is rendered, compare each
    # placed art/painting/frame against the reference photo IN CONTEXT, and
    # if mirrored, set flip_x=True on its orientation, re-save the JSON,
    # and re-render.
    _art_segs = [(_i, _p) for _i, _p in enumerate(placements)
                 if _p.get("type") in ("art", "painting", "frame")]
    if _art_segs and render_path and Path(render_path).exists():
        _any_flipped = False
        for _i, _p in _art_segs:
            # Placement dict uses bbox_px / segment_index / glb_file.
            _bbox = _p.get("bbox_px") or _p.get("bbox")
            _seg_idx = _p.get("segment_index", _p.get("seg_idx", _i))
            # Derive inpaint PNG path from the glb_file name (they share
            # the inpaint_<idx>_<type> stem).
            _glb_name = _p.get("glb_file", "")
            _inpaint_path = None
            if _glb_name.endswith(".glb"):
                _candidate = (Path(out_dir) / "wall_mounted" / "inpainted"
                              / (_glb_name[:-4] + ".png"))
                if _candidate.exists():
                    _inpaint_path = _candidate
            try:
                if _vlm_check_placed_painting_mirror(
                        ref_image_path=image_path,
                        render_path=render_path,
                        bbox=_bbox,
                        obj_type=_p.get("type", "art"),
                        seg_idx=_seg_idx,
                        inpaint_path=_inpaint_path):
                    # Mirror the GLB on disk (flat plane: flipping local-X
                    # vertices left↔right effectively mirrors the displayed
                    # texture, since UVs are unchanged).  Mirroring the GLB
                    # itself avoids relying on the orient[flip_x] code path
                    # downstream, mirroring the painting permanently in the
                    # GLB the way decorations' mirror_check fixes the the 3D generator
                    # left-right flip on a computer monitor.
                    _glb_full = (Path(out_dir) / "wall_mounted" / "objects"
                                 / _glb_name)
                    if _glb_full.exists() and _flip_glb_horizontally(_glb_full):
                        print(f"  [post-mirror] seg {_seg_idx}: mirrored "
                              f"{_glb_name} in place (left↔right)")
                        _any_flipped = True
                    else:
                        print(f"  [post-mirror] seg {_seg_idx}: GLB not "
                              f"flipped ({_glb_full})")
            except Exception as _pe:
                print(f"  [post-mirror] seg {_seg_idx}: skipped ({_pe})")
        if _any_flipped:
            # GLB was mutated on disk — re-render to pick up the new texture
            # orientation.  Placements JSON unchanged (orient stays the same).
            print(f"[post-mirror] re-rendering with mirrored GLB(s) …")
            # Note: the OBJ file's geometry won't change for flat planes —
            # the overlay pass redraws the textured face on top, and the
            # overlay reads the (now updated) flip_x from `placements`.
            _render_objects(
                str(obj_out_path), cam_path, place_dir, out_dir, placements, room)


def _flip_glb_horizontally(glb_path: Path) -> bool:
    """Mirror the GLB mesh across its local X axis (left↔right) and re-save
    in place.  For flat plane GLBs (window/door/art/painting/frame) this is
    equivalent to mirroring the displayed texture left-right, since UVs are
    unchanged but vertex positions move to the mirrored side.

    Inverts face winding after the reflection so normals stay outward.
    """
    try:
        import trimesh
        scene = trimesh.load(str(glb_path), force="scene", process=False)
        if isinstance(scene, trimesh.Trimesh):
            scene = trimesh.Scene([scene])
        # scale_matrix(-1, axis=[1,0,0]) reflects across the YZ plane —
        # i.e. flips the X coord of every vertex.
        M = trimesh.transformations.scale_matrix(
            -1.0, origin=[0, 0, 0], direction=[1.0, 0.0, 0.0])
        for _name, g in list(scene.geometry.items()):
            if isinstance(g, trimesh.Trimesh):
                g.apply_transform(M)
                g.invert()   # re-invert winding so normals point outward
        scene.export(str(glb_path))
        return True
    except Exception as e:
        print(f"  [mirror_fix] GLB flip failed for {glb_path.name}: {e}")
        return False


def _vlm_check_placed_painting_mirror(
    ref_image_path: str | Path,
    render_path: str | Path,
    bbox: list | None,
    obj_type: str,
    seg_idx: int,
    inpaint_path: "str | Path | None" = None,
) -> bool:
    """Detect whether the placed painting in the render is LEFT-RIGHT
    mirrored relative to the reference texture.

    Programmatic — uses normalized cross-correlation between the cropped
    painting region of the render and (a) the inpaint texture, (b) the
    horizontally-flipped inpaint texture.  If (b) correlates higher, the
    painting is mirrored on the wall.

    The VLM-based version was unreliable: even with the FULL reference +
    render side-by-side, it kept saying "no asymmetric features" or
    "consistent" for a clearly-flipped painting (orange-left in reference
    but orange-right in render).  Pixel correlation reliably catches this.

    Falls back to the reference-photo crop when no inpaint is available.
    """
    try:
        from PIL import Image as _PI
        if bbox is None or len(bbox) != 4:
            print(f"  [post-mirror] seg {seg_idx} ({obj_type}): "
                  f"no bbox available — skipping")
            return False
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 - x1 < 8 or y2 - y1 < 8:
            print(f"  [post-mirror] seg {seg_idx} ({obj_type}): "
                  f"bbox too small ({x2-x1}×{y2-y1}px) — skipping")
            return False

        render_img = _PI.open(render_path).convert("RGB")
        rW, rH = render_img.size
        x1c = max(0, min(rW - 1, x1))
        y1c = max(0, min(rH - 1, y1))
        x2c = max(x1c + 1, min(rW, x2))
        y2c = max(y1c + 1, min(rH, y2))
        placed_crop = render_img.crop((x1c, y1c, x2c, y2c))

        # Reference: prefer the inpaint PNG (clean texture, baked-in
        # orientation), fall back to the bbox crop of the original photo.
        if inpaint_path is not None and Path(inpaint_path).exists():
            ref_crop = _PI.open(inpaint_path).convert("RGB")
            ref_label = f"inpaint {Path(inpaint_path).name}"
        else:
            ref_img = _PI.open(ref_image_path).convert("RGB")
            refW, refH = ref_img.size
            sx, sy = refW / rW, refH / rH
            ref_crop = ref_img.crop((int(x1c*sx), int(y1c*sy),
                                      int(x2c*sx), int(y2c*sy)))
            ref_label = f"ref bbox {bbox}"

        target_sz = 128
        placed = np.array(placed_crop.resize((target_sz, target_sz)),
                          dtype=np.float64)
        ref_n  = np.array(ref_crop.resize((target_sz, target_sz)),
                          dtype=np.float64)
        ref_f  = np.array(ref_crop.transpose(_PI.FLIP_LEFT_RIGHT)
                          .resize((target_sz, target_sz)),
                          dtype=np.float64)

        def _ncc(a: np.ndarray, b: np.ndarray) -> float:
            a = a.flatten() - a.mean()
            b = b.flatten() - b.mean()
            denom = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-9
            return float((a * b).sum() / denom)

        c_normal  = _ncc(placed, ref_n)
        c_flipped = _ncc(placed, ref_f)
        # Require flipped correlation to be at least 5% higher than normal
        # before flagging — guards against noise on near-symmetric paintings.
        _MIRROR_MARGIN = 0.05
        is_mirrored = (c_flipped - c_normal) > (_MIRROR_MARGIN * abs(c_normal))
        print(f"  [post-mirror] seg {seg_idx} ({obj_type}): "
              f"corr(placed, {ref_label})={c_normal:+.3f}  "
              f"corr(placed, flipped)={c_flipped:+.3f} → "
              f"{'MIRRORED' if is_mirrored else 'OK'}")
        return is_mirrored
    except Exception as e:
        print(f"  [post-mirror] seg {seg_idx} ({obj_type}): check failed ({e})")
        return False


def _render_objects(
    obj_path: str,
    cam_path: Path,
    place_dir: Path,
    out_dir:   Path,
    placements: list[dict],
    room:       dict,
    proj_tex_image_path: "str | Path | None" = None,
) -> str:
    """Two-pass render: render plain walls as base, then overlay all placed GLBs.

    Pass 1 base image:
      • If render_openings_filled.png exists AND no windows/doors are being placed
        here (i.e. fill_openings already painted the window textures), use it so
        that existing fill_openings results are preserved.
      • Otherwise render the plain walls mesh (walls_with_windows.obj or walls.obj)
        — NOT walls_with_objects.obj, which includes raw untextured window-plane
        geometry that would show as grey smears on the wall.
    Pass 2 overlays all placed GLBs (including window planes) with vertex colors.
    """
    out_path = str(place_dir / "render_objects_placed.png")

    placing_openings = any(p.get("type") in ("window", "door") for p in placements)
    openings_render  = place_dir / "render_openings_filled.png"

    if not placing_openings and openings_render.exists():
        base_path = str(openings_render)
        print(f"[place_objects] Pass 1 — using existing {openings_render.name} as base")
    elif proj_tex_image_path is not None:
        # Use the provided image directly as the background — no 3D render.
        import shutil as _shutil
        base_path = str(place_dir / "_objects_render_tmp.png")
        _shutil.copy2(str(proj_tex_image_path), base_path)
        print(f"[place_objects] Pass 1 — using {Path(proj_tex_image_path).name} directly as base")
    else:
        from floorplan.wall_line.render_room import render_room
        base_path = str(place_dir / "_objects_render_tmp.png")
        tex_dir   = str(out_dir)
        # When placing window/door planes, render from plain walls mesh — NOT from
        # walls_with_objects.obj which has the window-plane geometry appended as raw
        # OBJ faces (no material/texture), which renders as grey opaque patches that
        # make the overlay window appear invisible.  The overlay pass will draw the
        # textured window planes on top of the plain wall render.
        if placing_openings:
            walls_mesh = place_dir / "walls_with_windows.obj"
            if not walls_mesh.exists():
                walls_mesh = out_dir / "walls.obj"
            mesh_for_base = str(walls_mesh) if walls_mesh.exists() else obj_path
            print(f"[place_objects] Pass 1 — rendering plain walls for window-plane base …")
        else:
            mesh_for_base = obj_path
            print(f"[place_objects] Pass 1 — no base render found, rendering from OBJ …")
        render_room(
            mesh_path=mesh_for_base, camera_json_path=str(cam_path),
            out_path=base_path, texture_dir=str(out_dir), no_legend=True,
        )

    # Pass 2 — overlay each placed GLB with vertex colors
    print(f"[place_objects] Pass 2 — overlaying GLB vertex colors …")
    _overlay_glbs(base_path, out_path, placements, room, cam_path, out_dir)

    print(f"[place_objects] Render → {out_path}")
    return out_path


def _apply_tilt_correction(
    verts: np.ndarray,
    wall: str,
    obj_type: str,
    W_m: float,
    D_m: float,
    forced_theta_deg: float | None = None,
    max_theta_deg: float | None = None,
) -> tuple:
    """Rotate the mesh in the wall-normal/height plane so the wall-facing face
    is flush end-to-end.

    the 3D generator objects often lean: the bottom touches the wall but the top floats
    away.  This rotates the mesh around the bottom-wall pivot until the top
    also touches.

    For regular objects (outward=False): min-na side touches wall.
    For windows/doors (outward=True):   max-na side (room-facing) touches wall.

    If forced_theta_deg is provided, that angle is used directly instead of
    being measured from the mesh (used by OBJ write and render overlay so they
    always apply the same angle as the placement loop, regardless of scaling).

    Returns (corrected_verts, theta_deg_applied).
    """
    na        = 2 if wall in ("back", "front") else 0
    ya        = 1
    is_window = obj_type in ("window", "door")

    if wall in ("back", "left"):
        wall_plane   = 0.0
        extreme_fn   = np.max if is_window else np.min
    else:
        wall_plane   = D_m if wall == "front" else W_m
        extreme_fn   = np.min if is_window else np.max

    if forced_theta_deg is not None:
        theta = np.radians(forced_theta_deg)
        print(f"  [tilt-correct] {obj_type} wall={wall}: using saved θ={forced_theta_deg:.2f}°")
    else:
        # Measure wall-nearest point in top vs bottom Y-quartile
        p75 = float(np.percentile(verts[:, ya], 75))
        p25 = float(np.percentile(verts[:, ya], 25))
        top_m = verts[:, ya] >= p75
        bot_m = verts[:, ya] <= p25
        if top_m.sum() < 3 or bot_m.sum() < 3:
            return verts, 0.0

        na_top = float(extreme_fn(verts[top_m, na]))
        na_bot = float(extreme_fn(verts[bot_m, na]))
        y_top  = float(np.mean(verts[top_m, ya]))
        y_bot  = float(np.mean(verts[bot_m, ya]))
        dy     = y_top - y_bot
        d_na   = na_top - na_bot   # how much the top drifts from the bottom along na

        if abs(dy) < 0.01 or abs(d_na) < 0.005:
            return verts, 0.0   # no significant tilt

        theta = -np.arctan2(d_na, dy)   # negative: swing top toward wall
        if max_theta_deg is not None:
            max_r = np.radians(max_theta_deg)
            if abs(theta) > max_r:
                print(f"  [tilt-correct] {obj_type} wall={wall}: "
                      f"d_na={d_na:.3f}m dy={dy:.3f}m θ={np.degrees(theta):.2f}° "
                      f"→ capped to ±{max_theta_deg:.1f}°")
                theta = np.sign(theta) * max_r
            else:
                print(f"  [tilt-correct] {obj_type} wall={wall}: "
                      f"d_na={d_na:.3f}m dy={dy:.3f}m θ={np.degrees(theta):.2f}°")
        else:
            print(f"  [tilt-correct] {obj_type} wall={wall}: "
                  f"d_na={d_na:.3f}m dy={dy:.3f}m θ={np.degrees(theta):.2f}°")

    if theta == 0.0:
        return verts, 0.0

    # Rotation angle: pivot at (y_min, wall_plane)
    y_pivot  = float(verts[:, ya].min())
    na_pivot = wall_plane

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dy_v  = verts[:, ya] - y_pivot
    dna_v = verts[:, na] - na_pivot

    verts = verts.copy()
    verts[:, ya] = y_pivot  + cos_t * dy_v - sin_t * dna_v
    verts[:, na] = na_pivot + sin_t * dy_v + cos_t * dna_v

    # Re-snap so the back face (wall-touching side, outward=False) stays at wall_plane.
    # Windows use outward=False — back face is at min-na (back/left) or max-na (front/right),
    # same as non-window objects.  The old is_window branches used the front-face convention
    # from the deprecated outward=True window path and would push windows behind the wall.
    if wall in ("back", "left"):
        verts[:, na] -= verts[:, na].min()
    else:
        verts[:, na] += wall_plane - verts[:, na].max()

    return verts, float(np.degrees(theta))


def _overlay_glbs(
    base_png:   str,
    out_png:    str,
    placements: list[dict],
    room:       dict,
    cam_path:   Path,
    out_dir:    Path,
) -> None:
    """Software-rasterize GLB vertex colors on top of base_png."""
    wm_dir  = out_dir / "wall_mounted"
    obj_dir = wm_dir / "objects"
    cam     = json.loads(cam_path.read_text())

    pos     = np.array(cam["position_m"],      dtype=float)
    look_at = np.array(cam["look_at_m"],       dtype=float)
    up      = np.array(cam.get("up", [0,1,0]), dtype=float)
    W_px, H_px = int(cam["width_px"]), int(cam["height_px"])
    _project   = _make_projector(pos, look_at, up, float(cam["hfov_deg"]), W_px, H_px)

    _base_img = Image.open(base_png).convert("RGB")
    if _base_img.size != (W_px, H_px):
        _base_img = _base_img.resize((W_px, H_px), Image.LANCZOS)
    base = np.array(_base_img, dtype=np.uint8)
    zbuf = np.full((H_px, W_px), np.inf, dtype=np.float32)

    # Render order: windows/doors first (flush on wall) so overlapping objects
    # (curtains, art) are drawn on top and correctly occlude them via z-buffer.
    _PLANE_TYPES = {"window", "door"}
    placements = (
        [p for p in placements if p.get("type") in _PLANE_TYPES] +
        [p for p in placements if p.get("type") not in _PLANE_TYPES]
    )

    for p in placements:
        glb_path = obj_dir / p["glb_file"]
        world_pt = np.array(p["world_pt"])
        size     = p["size_m"]
        wall     = p["wall"]
        orient   = p.get("orientation", {"flip_y": True, "flip_x": False, "flip_z": True})

        _gfd = p.get("glb_front_deg")
        glb_front_deg = int(_gfd) if _gfd is not None else None
        obj_type_p    = str(p.get("type", ""))
        _PLANE_OBJ_TYPES = ("window", "door", "art", "painting", "frame")

        # Flat plane GLBs (window/door/art/…) need flip_y=True: generate_window_plane
        # maps image-row-0 to world +Y, but OBB+placement inverts Y vs image convention.
        # For window/door, flip_x must always be False (plain panes/doors
        # have no asymmetric features).  For art/painting/frame, preserve
        # the flip_x set by the mirror check earlier.
        if obj_type_p in _PLANE_OBJ_TYPES:
            orient = dict(orient)
            orient["flip_y"] = True
            if obj_type_p in ("window", "door"):
                orient["flip_x"] = False
            # else: keep orient["flip_x"] from the mirror check

        result = _transform_glb_to_wall(
            glb_path, wall, world_pt, size, room,
            cam_pos=pos,
            flip_y=orient.get("flip_y", True),
            flip_x=orient.get("flip_x", False),
            flip_z=orient.get("flip_z", True),
            glb_front_deg=glb_front_deg,
            embed=False,
            outward=False,   # window/door are planes flush on wall
            detilt=(obj_type_p not in ("window", "door", "curtain",
                                       "art", "frame", "painting")),
            skip_obb_swap=(obj_type_p in _PLANE_OBJ_TYPES),
            force_upright=(obj_type_p in _PLANE_OBJ_TYPES),
            force_floor=(obj_type_p == "door"),
        )
        if result is None:
            continue
        verts, faces, vc = result

        # Art/painting/frame/mirror GLBs are now generated as flat planes with
        # rectified + SAM-segmented vertex colors baked in.  No need to re-sample
        # from the raw (non-rectified) inpaint PNG — use the GLB vertex colors as-is.

        # Re-apply lamp facing-direction flip if it was applied during placement
        if orient.get("lamp_facing_flipped"):
            flip_wall = orient.get("lamp_facing_wall", wall)
            flip_pt   = np.array(orient.get("lamp_facing_world_pt", world_pt.tolist()),
                                 dtype=float)
            flipped = verts.copy()
            if flip_wall in ("back", "front"):
                flipped[:, 0] = 2.0 * flip_pt[0] - verts[:, 0]
            else:
                flipped[:, 2] = 2.0 * flip_pt[2] - verts[:, 2]
            verts = flipped

        W_r = float(room.get("floor_width_m", 4.0))
        D_r = float(room.get("floor_depth_m", 4.0))
        saved_theta = p.get("tilt_theta_deg")
        verts, _ = _apply_tilt_correction(
            verts, wall, obj_type_p, W_r, D_r,
            forced_theta_deg=saved_theta,
        )

        # Curtain: pin top to the shared wall rod height recorded at placement time.
        # The re-derived transform may land slightly differently from placed_raw,
        # so we force the exact top here for visual consistency.
        if obj_type_p == "curtain" and "final_top_y" in p:
            _target_top = float(p["final_top_y"])
            _cur_top    = float(verts[:, 1].max())
            if abs(_target_top - _cur_top) > 1e-4:
                verts = verts.copy()
                verts[:, 1] += _target_top - _cur_top

        proj = [_project(v) for v in verts]

        # Flat window/door/art planes: the 257×257 pixel-grid mesh needs BOTH
        # triangle rasterisation and vertex splatting.
        #   - At high render resolution (e.g. 2500×1667), triangles cover many
        #     pixels → rasterisation fills the bulk; pure splatting would leave
        #     gaps (1 vertex → 1 pixel, producing a semi-transparent ghost).
        #   - At low resolution, triangles shrink below the 0.5-denominator
        #     threshold in _rasterize_vc_tri and get silently dropped → splat
        #     fills them in.
        # Splat runs second and z-tests each pixel, so it only fills holes.
        if obj_type_p in ("window", "door", "art", "painting", "frame"):
            n_tris = 0
            for tri in faces:
                i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
                p0, p1, p2 = proj[i0], proj[i1], proj[i2]
                if p0 is None or p1 is None or p2 is None:
                    continue
                pts = np.array([p0, p1, p2], dtype=np.float32)
                col = np.array([vc[i0], vc[i1], vc[i2]], dtype=np.uint8)
                _rasterize_vc_tri(base, zbuf, pts, col)
                n_tris += 1
            n_splat = 0
            for i, pv in enumerate(proj):
                if pv is None:
                    continue
                xi = int(pv[0]);  yi = int(pv[1])
                if 0 <= xi < W_px and 0 <= yi < H_px and pv[2] < zbuf[yi, xi]:
                    base[yi, xi] = vc[i]
                    zbuf[yi, xi] = pv[2]
                    n_splat += 1
            print(f"  [overlay] seg {p['segment_index']:02d} ({p['type']}): "
                  f"{n_tris}/{len(faces)} tris + {n_splat} splat")
        else:
            n_drawn = 0
            for tri in faces:
                i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
                p0, p1, p2 = proj[i0], proj[i1], proj[i2]
                if p0 is None or p1 is None or p2 is None:
                    continue
                pts = np.array([p0, p1, p2], dtype=np.float32)
                col = np.array([vc[i0], vc[i1], vc[i2]], dtype=np.uint8)
                _rasterize_vc_tri(base, zbuf, pts, col)
                n_drawn += 1
            print(f"  [overlay] seg {p['segment_index']:02d} ({p['type']}): {n_drawn}/{len(faces)} faces")

    Image.fromarray(base).save(out_png)


# ─────────────────────────────────────────────────────────────────────────────
# Corner-based coordinate system and placement pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _corner_coord_system(room: dict, cam_pos: np.ndarray) -> dict:
    """Establish the deepest-corner coordinate system for VLM reasoning.

    The deepest floor corner (where back wall meets the visible side wall at
    the floor) becomes the origin (cX=0, cY=0, cZ=0).

      cX-axis : runs along the back wall   (world X, length = floor_width_m)
      cY-axis : runs along the side wall   (world Z, length = floor_depth_m)
      cZ-axis : vertical height            (world Y, 0 → ceiling_height_m)

    Which side wall is "visible" is inferred from the camera position:
      cam_x > W/2  →  left wall visible   →  corner = world (0, 0, 0)
      cam_x ≤ W/2  →  right wall visible  →  corner = world (W, 0, 0)

    Wall placement constraints (the axis locked to 0):
      wall1 = back wall   →  objects have cY = 0  (world Z = 0)
      wall2 = side wall   →  objects have cX = 0  (world X = 0 or W)

    Returns a dict used to build VLM prompts and convert corner↔world coords.
    """
    W    = float(room["floor_width_m"])
    D    = float(room["floor_depth_m"])
    ceil = float(room.get("ceiling_height_m", 2.7))

    if float(cam_pos[0]) > W / 2.0:
        # Camera on the right — left wall visible, corner at world (0, 0, 0)
        side_wall     = "left"
        corner_world  = np.array([0.0, 0.0, 0.0])
        corner_x_flip = False   # cX = world X (no flip)
    else:
        # Camera on the left — right wall visible, corner at world (W, 0, 0)
        side_wall     = "right"
        corner_world  = np.array([W, 0.0, 0.0])
        corner_x_flip = True    # cX = W - world_X (measured from right corner)

    return {
        "corner_world":  corner_world,
        "wall1":         "back",      # bottom edge along cX; constraint: cY = 0
        "wall2":         side_wall,   # bottom edge along cY; constraint: cX = 0
        "corner_x_flip": corner_x_flip,
        "X_len":         W,
        "Y_len":         D,
        "Z_len":         ceil,
    }


# ── Height estimation from mask ───────────────────────────────────────────────

def _estimate_height_from_mask(
    mask_path: Path,
    world_pt:  np.ndarray,
    pos:       np.ndarray,
    fwd_h:     np.ndarray,
    fx:        float,
    cy_px:     float,
    tilt_tan:  float,
    ceil:      float,
) -> tuple[float, float] | None:
    """Numerically compute the object's world-Y bottom and top from mask pixels.

    Projects the topmost and bottommost foreground pixels through the
    architectural camera to recover world heights.

    Returns (bottom_world_y, top_world_y) in metres, or None on failure.
    """
    if not mask_path.exists():
        return None
    try:
        mask_arr = np.array(Image.open(mask_path))
    except Exception:
        return None
    fg = mask_arr.sum(axis=2) > 0 if mask_arr.ndim == 3 else mask_arr > 0
    ys, _ = np.where(fg)
    if len(ys) == 0:
        return None

    Z_c = float(np.dot(world_pt - pos, fwd_h))
    if Z_c < 0.1:
        return None

    # top pixel (smallest image-Y) → highest world Y
    top_py    = float(ys.min())
    top_wy    = pos[1] + (cy_px - top_py) / fx * Z_c + tilt_tan * Z_c
    top_wy    = float(np.clip(top_wy,    0.0, ceil))

    # bottom pixel (largest image-Y) → lowest world Y
    bot_py    = float(ys.max())
    bot_wy    = pos[1] + (cy_px - bot_py) / fx * Z_c + tilt_tan * Z_c
    bot_wy    = float(np.clip(bot_wy, 0.0, ceil))

    if bot_wy > top_wy:
        bot_wy, top_wy = top_wy, bot_wy
    height = top_wy - bot_wy
    if height < 0.05:
        return None

    print(f"  [height-mask] top_px={top_py:.0f}→{top_wy:.2f}m  "
          f"bot_px={bot_py:.0f}→{bot_wy:.2f}m  "
          f"height={height:.2f}m  Z_c={Z_c:.2f}m")
    return bot_wy, top_wy


# ── Full mask-to-wall projection (all pixels → wall-plane coords) ─────────────

def _project_mask_to_scene(
    mask_path:   Path,
    pos:         np.ndarray,
    right:       np.ndarray,
    fwd_h:       np.ndarray,
    tilt_tan:    float,
    fx:          float,
    cx_px:       float,
    cy_px:       float,
    W_m:         float,
    D_m:         float,
    ceil:        float,
    sample_step: int = 4,
) -> dict | None:
    """Back-project every (sampled) foreground mask pixel through the calibrated
    VGGT camera and let each ray find its own wall hit.

    Because the mask and the render share the same pixel space (same camera,
    same resolution), this directly maps the 2D mask footprint to 3D wall
    coordinates without needing a pre-specified wall.

    Each pixel votes for a wall.  The dominant wall is the identified surface.
    Positions are aggregated per-wall with robust percentiles.

    Returns:
        {
          "wall":       str,    # dominant wall name
          "free_med":   float,  # median free-axis on dominant wall
          "free_lo":    float,  # p5  free-axis
          "free_hi":    float,  # p95 free-axis
          "height_lo":  float,  # p5  world-Y (floor of object)
          "height_hi":  float,  # p95 world-Y (top of object)
          "wall_votes": dict,   # {wall: count} for debugging
        }
    or None if fewer than 3 pixels hit any wall.
    """
    if not mask_path.exists():
        return None
    try:
        mask_arr = np.array(Image.open(mask_path))
    except Exception:
        return None
    fg = mask_arr.sum(axis=2) > 0 if mask_arr.ndim == 3 else mask_arr > 0
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return None

    # Subsample
    idx = np.arange(0, len(xs), sample_step)

    # Accumulate hits per wall
    wall_hits: dict[str, dict[str, list[float]]] = {}

    for i in idx:
        ray = _backproject(float(xs[i]), float(ys[i]),
                           pos, right, fwd_h, tilt_tan, fx, cx_px, cy_px)
        result = _find_wall(pos, ray, W_m, D_m, ceil)
        if result is None:
            continue
        w, hit = result
        if w not in wall_hits:
            wall_hits[w] = {"free": [], "height": []}
        free = float(hit[0]) if w in ("back", "front") else float(hit[2])
        wall_hits[w]["free"].append(free)
        wall_hits[w]["height"].append(float(hit[1]))

    if not wall_hits:
        return None

    # Dominant wall = most pixel hits
    wall_votes = {w: len(d["free"]) for w, d in wall_hits.items()}
    dominant   = max(wall_votes, key=wall_votes.get)

    fv = np.array(wall_hits[dominant]["free"])
    hv = np.array(wall_hits[dominant]["height"])

    if len(fv) < 3:
        return None

    result = dict(
        wall       = dominant,
        free_med   = float(np.median(fv)),
        free_lo    = float(np.percentile(fv,  5)),
        free_hi    = float(np.percentile(fv, 95)),
        height_lo  = float(np.percentile(hv,  5)),
        height_hi  = float(np.percentile(hv, 95)),
        wall_votes = wall_votes,
        n_pixels   = int(len(fv)),
    )
    print(f"  [mask→scene] votes={wall_votes}  dominant={dominant}  "
          f"free=[{result['free_lo']:.3f}…{result['free_hi']:.3f}]m (med {result['free_med']:.3f})  "
          f"height=[{result['height_lo']:.3f}…{result['height_hi']:.3f}]m")
    return result


# ── Compute projected screen bbox of a placed GLB ─────────────────────────────

def _rendered_bbox_px(
    glb_path:   Path,
    wall:       str,
    world_pt:   np.ndarray,
    size:       dict,
    room:       dict,
    project_fn,
    flip_y:          bool = True,
    flip_x:          bool = False,
    flip_z:          bool = True,
    glb_front_deg:   int  = 0,
    front_yaw_offset: int = 0,
    detilt:          bool = True,
) -> tuple[float, float, float, float] | None:
    """Return the (x1, y1, x2, y2) screen bbox of the placed GLB, or None."""
    result = _transform_glb_to_wall(
        glb_path, wall, world_pt, size, room,
        flip_y=flip_y, flip_x=flip_x, flip_z=flip_z, glb_front_deg=glb_front_deg,
        front_yaw_offset=front_yaw_offset, detilt=detilt,
    )
    if result is None:
        return None
    verts, _, _ = result
    proj = [project_fn(v) for v in verts]
    valid = [p for p in proj if p is not None]
    if len(valid) < 3:
        return None
    pxs = [p[0] for p in valid]
    pys = [p[1] for p in valid]
    return min(pxs), min(pys), max(pxs), max(pys)


# ── Iterative free-axis refinement ────────────────────────────────────────────

def _refine_free_axis(
    glb_path:         Path,
    wall:             str,
    world_pt:         np.ndarray,    # mutable starting position
    size:             dict,
    room:             dict,
    pos:              np.ndarray,
    fwd_h:            np.ndarray,
    fx:               float,
    project_fn,
    mask_path:        Path,
    flip_y:           bool = True,
    flip_x:           bool = False,
    flip_z:           bool = True,
    glb_front_deg:    int  = 0,
    front_yaw_offset: int  = 0,
    detilt:           bool = True,
    max_iters:        int  = 6,
) -> np.ndarray:
    """Shift world_pt along the wall's free axis until the rendered centroid
    pixel-aligns with the mask centroid.  Z (world Y / height) is fixed.

    free axis:  world X  for back/front walls
                world Z  for left/right walls

    Returns refined world_pt (a new array; input is not modified).
    """
    if not mask_path.exists():
        return world_pt.copy()
    try:
        mask_arr = np.array(Image.open(mask_path))
    except Exception:
        return world_pt.copy()
    fg = mask_arr.sum(axis=2) > 0 if mask_arr.ndim == 3 else mask_arr > 0
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return world_pt.copy()

    tgt_cx = float(xs.mean())   # target screen X centroid (from mask)
    free_ax = 0 if wall in ("back", "front") else 2   # world X or world Z

    W = float(room["floor_width_m"])
    D = float(room["floor_depth_m"])
    hw = float(size.get("width_m", 0.0)) / 2.0   # half-width margin

    pt = world_pt.copy()
    for i in range(max_iters):
        bbox = _rendered_bbox_px(glb_path, wall, pt, size, room, project_fn,
                                  flip_y=flip_y, flip_x=flip_x, flip_z=flip_z,
                                  glb_front_deg=glb_front_deg,
                                  front_yaw_offset=front_yaw_offset, detilt=detilt)
        if bbox is None:
            break
        curr_cx = (bbox[0] + bbox[2]) / 2.0
        px_err  = tgt_cx - curr_cx

        if abs(px_err) < 1.5:   # ~1 pixel convergence
            print(f"  [refine] iter {i}: converged (px_err={px_err:.2f})")
            break

        # Convert pixel error to world metres using camera depth
        Z_c       = max(float(np.dot(pt - pos, fwd_h)), 0.1)
        world_err = px_err / fx * Z_c

        # Determine the sign: does moving the free axis in the + direction
        # increase (+) or decrease (-) the screen X column?
        # This equals dot(camera_right, free_axis_unit_vector).
        # camera_right = cross(fwd_h, up)
        right_dir = np.cross(fwd_h, np.array([0., 1., 0.]))
        rn = float(np.linalg.norm(right_dir))
        if rn > 1e-9:
            right_dir /= rn
        if wall in ("left", "right"):
            # free axis = world Z; dot with (0,0,1)
            axis_sign = float(right_dir[2])
        else:
            # back/front walls; free axis = world X; dot with (1,0,0)
            axis_sign = float(right_dir[0])
        if axis_sign < 0:
            world_err = -world_err

        room_end = W if free_ax == 0 else D
        step = float(np.clip(world_err, -0.3, 0.3))
        # Clamp with half-width margin so the full object stays inside the room
        pt[free_ax] = float(np.clip(pt[free_ax] + step, hw, room_end - hw))
        print(f"  [refine] iter {i}: px_err={px_err:+.1f}  "
              f"Zc={Z_c:.2f}m  step={step:+.3f}m  "
              f"pt[{free_ax}]→{pt[free_ax]:.3f}")

    return pt


# ── Wall placement state (collision tracking) ─────────────────────────────────


def _wall_bbox(wall: str, world_pt: np.ndarray, size: dict) -> tuple[float, float, float, float]:
    """Return (free_min, free_max, z_min, z_max) for the object on its wall.

    free axis = world X for back/front walls, world Z for left/right walls.
    z axis    = world Y (height) for all walls.
    """
    hw = float(size["width_m"])  / 2.0
    hh = float(size["height_m"]) / 2.0
    if wall in ("back", "front"):
        free_min = float(world_pt[0]) - hw
        free_max = float(world_pt[0]) + hw
    else:
        free_min = float(world_pt[2]) - hw
        free_max = float(world_pt[2]) + hw
    z_min = float(world_pt[1]) - hh
    z_max = float(world_pt[1]) + hh
    return free_min, free_max, z_min, z_max


def _overlaps_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return a_min < b_max and b_min < a_max


def _slide_from_collision(
    wall:         str,
    world_pt:     np.ndarray,
    size:         dict,
    room:         dict,
    state:        list[dict],
    seg_idx:      int,
    gap:          float = 0.02,   # 2 cm clearance
    max_slide_m:  float = 0.15,   # max slide — keep tight to preserve mask alignment
    min_z_frac:   float = 0.15,   # minimum height-overlap fraction to count as blocking
) -> np.ndarray | None:
    """Find a non-overlapping position for the object along the wall's free axis.

    Uses a gap-sweep approach: collects all blocked intervals from height-overlapping
    same-wall objects, merges them, then finds the closest clear position to the
    original world_pt within max_slide_m.  If no clear gap exists within that
    distance, returns the original world_pt (overlap accepted) rather than placing
    the object far from its natural position.  Returns None only when the object
    cannot fit on the wall at all.
    """
    free_ax  = 0 if wall in ("back", "front") else 2
    room_end = float(room["floor_width_m"]) if free_ax == 0 else float(room["floor_depth_m"])
    hw = float(size["width_m"])  / 2.0
    hh = float(size["height_m"]) / 2.0

    obj_z_min = float(world_pt[1]) - hh
    obj_z_max = float(world_pt[1]) + hh
    desired   = float(world_pt[free_ax])

    # Collect blocked intervals from same-wall objects that overlap in height.
    # Require a minimum height-overlap fraction to avoid near-tangent overlaps
    # from blocking large swathes of the wall.
    blocked: list[tuple[float, float]] = []
    obj_h = obj_z_max - obj_z_min
    for rec in state:
        if rec["wall"] != wall or rec["seg_idx"] == seg_idx:
            continue
        if not _overlaps_1d(obj_z_min, obj_z_max, rec["z_min"], rec["z_max"]):
            continue
        # Check minimum height-overlap fraction
        overlap_h = min(obj_z_max, rec["z_max"]) - max(obj_z_min, rec["z_min"])
        min_h = min(obj_h, rec["z_max"] - rec["z_min"])
        if min_h > 0 and overlap_h / min_h < min_z_frac:
            continue
        # Expand by hw + gap on each side to get the forbidden centre range
        blocked.append((rec["free_min"] - hw - gap, rec["free_max"] + hw + gap))

    if not blocked:
        return world_pt.copy()

    # Merge overlapping blocked intervals
    blocked.sort()
    merged: list[tuple[float, float]] = [blocked[0]]
    for lo, hi in blocked[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    # Check if desired position is already clear
    room_lo, room_hi = hw, room_end - hw
    if all(not _overlaps_1d(desired - 1e-6, desired + 1e-6, lo, hi) for lo, hi in merged):
        return world_pt.copy()

    # Build list of candidate clear positions (edges of blocked intervals and room bounds)
    candidates: list[float] = [room_lo, room_hi]
    for lo, hi in merged:
        candidates.append(hi)   # just past this block's right edge
        candidates.append(lo - 2 * hw)  # just before this block's left edge (centre)

    # Determine "same side" preference: which side of the merged-blocks centroid
    # the desired position is on.  We prefer slide candidates on the SAME side so
    # that the object's image-relative ordering (camera-side vs corner-side) is
    # preserved.  Crossing to the opposite side typically reverses the visual
    # left/right order against the photo, which the user perceives as wrong.
    blocks_lo = min(lo for lo, _ in merged)
    blocks_hi = max(hi for _, hi in merged)
    blocks_mid = 0.5 * (blocks_lo + blocks_hi)
    desired_side = 1 if desired >= blocks_mid else -1   # +1 = above blocks, -1 = below

    # First pass: only consider candidates on the same side as desired
    best: float | None = None
    best_dist = float("inf")
    for cand in candidates:
        cand = float(np.clip(cand, room_lo, room_hi))
        if any(_overlaps_1d(cand - 1e-6, cand + 1e-6, lo, hi) for lo, hi in merged):
            continue
        cand_side = 1 if cand >= blocks_mid else -1
        if cand_side != desired_side:
            continue
        d = abs(cand - desired)
        if d < best_dist:
            best_dist = d
            best = cand

    # Second pass: if no same-side candidate, fall back to any side
    crossed_side = False
    if best is None:
        for cand in candidates:
            cand = float(np.clip(cand, room_lo, room_hi))
            if any(_overlaps_1d(cand - 1e-6, cand + 1e-6, lo, hi) for lo, hi in merged):
                continue
            d = abs(cand - desired)
            if d < best_dist:
                best_dist = d
                best = cand
        crossed_side = best is not None

    pt = world_pt.copy()
    # When the only valid candidate is on the OPPOSITE side of the wall blocks,
    # use a tighter cap to avoid swapping image-relative ordering.  Same-side
    # slides may use the full max_slide_m budget.
    effective_cap = min(max_slide_m, 0.6) if crossed_side else max_slide_m
    if best is not None and best_dist <= effective_cap:
        if abs(best - desired) > 1e-4:
            tag = " (crossed-side)" if crossed_side else ""
            print(f"  [collision] slid ax{free_ax} {desired:.3f} → {best:.3f} "
                  f"(Δ={best-desired:+.3f}m){tag} to clear {len(merged)} block(s)")
        pt[free_ax] = best
    else:
        # No nearby gap on the preferred side — keep the original (possibly
        # overlapping) position rather than re-ordering relative to the photo.
        if best is not None:
            tag = " on opposite side" if crossed_side else ""
            print(f"  [collision] nearest gap {best:.3f}m is {best_dist:.2f}m away{tag} "
                  f"(>{effective_cap:.2f}m limit) — keeping original position {desired:.3f}m")
        else:
            print(f"  [collision] no clear position on {wall} wall — keeping original position")
    return pt


def _save_wall_state(state: list[dict], path: Path) -> None:
    """Write placement state to a human-readable txt file."""
    lines = [
        "# wall_placements.txt — one placed object per line\n",
        "# seg_idx  wall  free_min  free_max  z_min  z_max\n",
        "#   free = world X for back/front walls, world Z for left/right walls\n",
        "#   z    = world Y (height)\n",
    ]
    for r in state:
        lines.append(
            f"{r['seg_idx']:3d}  {r['wall']:<6s}  "
            f"{r['free_min']:7.4f}  {r['free_max']:7.4f}  "
            f"{r['z_min']:7.4f}  {r['z_max']:7.4f}\n"
        )
    path.write_text("".join(lines))


# ── Wall branch classifier (which world wall appears left/right of the corner) ──

def _classify_wall_branches(csys: dict, right: np.ndarray) -> dict:
    """Map wall1 / wall2 to left-branch / right-branch as seen in the image.

    The "left branch" is the wall whose floor-edge direction from the corner
    has a negative dot-product with the camera's right vector (goes left in
    the image).  The "right branch" is the other one.

    Also computes the world extension direction of each branch, so callers
    can convert VLM branch answers back to world coordinates.

    Returns a dict with keys:
        left_wall   : world wall name ("back"/"left"/"right")
        right_wall  : world wall name
        left_len    : length of that wall (metres)
        right_len   : length of that wall (metres)
        left_dir    : unit vec in world space pointing along left branch
        right_dir   : unit vec in world space pointing along right branch
    """
    corner  = csys["corner_world"]
    wall1   = csys["wall1"]   # "back"
    wall2   = csys["wall2"]   # "left" or "right"
    W       = csys["X_len"]
    D       = csys["Y_len"]

    # Direction each wall extends from the corner in world space
    # Back wall: always extends along ±X from the corner
    if csys["corner_x_flip"]:
        # corner at (W,0,0) — back wall extends in -X direction
        wall1_dir = np.array([-1.0, 0.0, 0.0])
        wall1_len = W
    else:
        # corner at (0,0,0) — back wall extends in +X direction
        wall1_dir = np.array([1.0, 0.0, 0.0])
        wall1_len = W

    # Side wall: always extends in +Z direction from corner (depth axis)
    wall2_dir = np.array([0.0, 0.0, 1.0])
    wall2_len = D

    # Dot product with camera right → positive means image-right, negative means image-left
    d1 = float(np.dot(wall1_dir, right))
    d2 = float(np.dot(wall2_dir, right))

    if d1 <= d2:
        # wall1 is more to the image-left
        return dict(left_wall=wall1, left_len=wall1_len, left_dir=wall1_dir,
                    right_wall=wall2, right_len=wall2_len, right_dir=wall2_dir)
    else:
        return dict(left_wall=wall2, left_len=wall2_len, left_dir=wall2_dir,
                    right_wall=wall1, right_len=wall1_len, right_dir=wall1_dir)


def _wall_features_str(wall_context: dict, wall: str) -> str:
    """Return a human-readable description of wall openings, or 'no openings'."""
    ctx   = wall_context.get(wall, {})
    feats = ctx.get("features", [])
    if not feats:
        return "no openings (plain wall)"
    parts = []
    for f in feats:
        ftype = f.get("type", "opening")
        w_m   = f.get("width_m", "?")
        h_m   = f.get("height_m", "?")
        off_m = f.get("offset_from_left_m", "?")
        parts.append(f"{ftype} {w_m}×{h_m}m at +{off_m}m from corner")
    return ";  ".join(parts)


# ── VLM prompt for corner-based wall identification ───────────────────────────

_VLM_CORNER_PROMPT = """\
You are identifying which wall a {obj_type} is mounted on in a room photo.

DEEPEST CORNER COORDINATE SYSTEM
  Look at the image and find the DEEPEST VISIBLE FLOOR CORNER — the V-shaped
  point where two walls meet at the floor, farthest from the camera.
  That corner is the ORIGIN (X=0, Y=0, Z=0).

  X-axis : runs along the wall to the LEFT  of the corner  (0 → {X_len:.2f} m)
  Y-axis : runs along the wall to the RIGHT of the corner  (0 → {Y_len:.2f} m)
  Z-axis : vertical height  (0 = floor  →  {Z_len:.2f} m = ceiling)

  If a third wall is visible:
    • Closes the LEFT  branch at its far end → it is at X = {X_len:.2f} m
    • Closes the RIGHT branch at its far end → it is at Y = {Y_len:.2f} m

WALL FEATURES (for identification)
  Left-branch wall  ({X_len:.2f} m long): {left_feat}
  Right-branch wall ({Y_len:.2f} m long): {right_feat}

PLACEMENT CONSTRAINT
  Object on the LEFT-branch wall  →  Y = 0
  Object on the RIGHT-branch wall →  X = 0

GEOMETRY HINT
  Camera ray through the object centre hits the "{hint_branch}" wall.
  Override only if the image clearly contradicts this.

Image 1 : Room photo — {obj_type} outlined with a red box.
Image 2 : Same photo with segmented mask overlaid in semi-transparent white.

TASK
  1. Which wall is the {obj_type} on?  "left_branch"  or  "right_branch"
  2. Position of the object centre along that wall's free axis:
       • left_branch  → X value  (0 = corner,  {X_len:.2f} m = far end)
       • right_branch → Y value  (0 = corner,  {Y_len:.2f} m = far end)

Reply with JSON only — no explanation:
{{"branch": "left_branch|right_branch", "free_pos_m": <float>}}"""


_VLM_WALL4_PROMPT = """\
You are identifying which wall a {obj_type} is mounted on in a room photo.

THE DEEPEST CORNER
  The farthest corner of the room (where two walls meet at the back) is visible in
  the image at approximately pixel column {corner_col} (image is {img_w} px wide).
  Two walls extend from this corner:
    • LEFT  wall  (extends to the LEFT  of the corner in the image) = "{left_wall_name}"  {left_feats}
    • RIGHT wall  (extends to the RIGHT of the corner in the image) = "{right_wall_name}"  {right_feats}

OBJECT BOUNDING BOX
  The {obj_type} is outlined with a red box: columns {obj_col_lo}–{obj_col_hi}, rows {obj_row_lo}–{obj_row_hi}.
  The deepest corner is at column {corner_col}.

GEOMETRY HINT
  Camera-ray analysis places the object on the "{hint_wall}" wall.
  Trust this unless the image clearly shows otherwise.

HOW TO DECIDE
  Compare the object's horizontal position to the corner column:
    • Object mostly LEFT  of column {corner_col} → it is on the LEFT  wall ("{left_wall_name}")
    • Object mostly RIGHT of column {corner_col} → it is on the RIGHT wall ("{right_wall_name}")
  Look at which flat surface the object's BACK is resting against to confirm.

Image 1 : Room photo with red bounding box around the {obj_type}.
Image 2 : Same photo with the segmented mask highlighted.

Reply with JSON only — no explanation, use the exact wall name:
{{"wall": "{left_wall_name}|{right_wall_name}"}}"""


# ── Post-calibration wall-context re-check ────────────────────────────────────

_VLM_WALL_CONTEXT_CHECK_PROMPT = """\
You are verifying that the wall labels in a rendered interior room are correct.

CURRENT WALL-FEATURE ASSIGNMENTS (from prior scene analysis):
  "back"  wall (should face the camera directly): {back_feats}
  "left"  wall (should be to the camera's left):  {left_feats}
  "right" wall (should be to the camera's right): {right_feats}

Look at the rendered room image. The room has up to three visible walls:
  • The BACK wall is the main flat surface most directly facing the camera
    (typically the largest, centred wall you see ahead of you).
  • The LEFT wall extends to the LEFT  of the deepest visible floor corner.
  • The RIGHT wall extends to the RIGHT of the deepest visible floor corner.

TASK
  Check whether each label's described features actually match the wall you
  see in that position.  Common errors are a left/right swap (the features
  for "left" and "right" belong to the opposite walls), or a back/side swap
  (features labelled "back" are actually on a side wall and vice versa).

Reply with JSON only — no extra text:
{{
  "back_actually_is":  "back|left|right",
  "left_actually_is":  "back|left|right",
  "right_actually_is": "back|left|right",
  "notes": "<one sentence explanation, or empty string>"
}}

If all labels are already correct, all three fields must equal their own name
(e.g. "back_actually_is": "back").
If any features say "no openings (plain wall)" for all walls, or you cannot
tell, return all fields as their own name (no change)."""


def _recheck_wall_context(
    cam:         dict,
    render_path: Path | None,
    W_m:         float,
    D_m:         float,
) -> dict:
    """Re-verify wall_context label correspondence using the rendered room image.

    After VGGT camera calibration the rendering may show different wall
    relationships than Stage 1 assumed.  A quick VLM pass on the render
    detects left/right or back/side swaps and remaps wall_context in memory.
    Returns an updated cam dict (original is not mutated).
    """
    import requests

    wctx = cam.get("wall_context")
    if not wctx:
        return cam
    if render_path is None or not Path(render_path).exists():
        return cam

    back_feats  = _wall_features_str(wctx, "back")
    left_feats  = _wall_features_str(wctx, "left")
    right_feats = _wall_features_str(wctx, "right")

    # Skip if all three walls have the same description — VLM can't distinguish
    if back_feats == left_feats == right_feats:
        return cam

    try:
        b64 = base64.b64encode(Path(render_path).read_bytes()).decode()
    except Exception:
        return cam

    prompt = _VLM_WALL_CONTEXT_CHECK_PROMPT.format(
        back_feats=back_feats,
        left_feats=left_feats,
        right_feats=right_feats,
    )

    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw    = resp.json()["choices"][0]["message"]["content"]
        result = _parse_json(raw)
    except Exception as e:
        print(f"[wall_context] re-check VLM call failed ({e}) — keeping existing labels")
        return cam

    if not isinstance(result, dict):
        return cam

    mapping = {
        "back":  str(result.get("back_actually_is",  "back")).strip(),
        "left":  str(result.get("left_actually_is",  "left")).strip(),
        "right": str(result.get("right_actually_is", "right")).strip(),
    }
    # Validate: values must be a permutation of back/left/right
    valid = {"back", "left", "right"}
    if set(mapping.values()) != valid:
        print(f"[wall_context] re-check returned invalid mapping {mapping} — ignoring")
        return cam

    if all(mapping[k] == k for k in valid):
        notes = result.get("notes", "")
        print(f"[wall_context] re-check: labels confirmed correct.{' '+notes if notes else ''}")
        return cam

    # Apply remap using inverse permutation.
    # mapping[label] = pos means label's features actually belong at position pos.
    # So position pos should receive label's old features → invert the mapping.
    old_ctx = dict(wctx)
    inv     = {v: k for k, v in mapping.items()}
    new_ctx = {}
    for new_label in ("back", "left", "right"):
        src = inv.get(new_label, new_label)   # which old label had these features
        new_ctx[new_label] = old_ctx.get(src, {})
    # Preserve unrelated keys (e.g. "front")
    for k, v in old_ctx.items():
        if k not in new_ctx:
            new_ctx[k] = v

    notes = result.get("notes", "")
    print(f"[wall_context] re-check corrected labels: {mapping}.{' '+notes if notes else ''}")
    cam = dict(cam)
    cam["wall_context"] = new_ctx
    return cam


def _camera_visible_walls(
    cam_pos:  np.ndarray,
    look_at:  np.ndarray,
    hfov_deg: float,
    W_m:      float,
    D_m:      float,
) -> list[str]:
    """Return ordered list of walls visible from the camera.

    A wall is visible when the angle between the camera look direction and the
    direction toward the wall's center is less than 90°.  Walls whose centers
    lie in the rear hemisphere of the camera (angle ≥ 90°) cannot be the
    primary surface the camera is looking at and are excluded.

    Using 90° (not 90°+hfov/2) avoids including side walls that appear only
    at extreme image edges or project off-screen — a common problem when the
    camera is angled diagonally toward a corner.

    "front" wall is always excluded (it faces into the camera).
    """
    look = np.array(look_at, dtype=float) - np.array(cam_pos, dtype=float)
    look /= max(float(np.linalg.norm(look)), 1e-9)

    # Strict 90° threshold: only walls whose center is in the forward hemisphere.
    max_angle = math.radians(90.0)

    # Wall center positions (at camera height, mid-depth/width)
    cam = np.array(cam_pos, dtype=float)
    walls = {
        "back":  np.array([W_m / 2, cam[1], 0.0]),
        "front": np.array([W_m / 2, cam[1], D_m]),
        "left":  np.array([0.0,     cam[1], D_m / 2]),
        "right": np.array([W_m,     cam[1], D_m / 2]),
    }

    result = []
    for name, wall_center in walls.items():
        if name == "front":
            continue   # front wall always excluded
        toward = wall_center - cam
        d = float(np.linalg.norm(toward))
        if d < 1e-6:
            continue
        toward /= d
        angle = float(np.arccos(np.clip(np.dot(look, toward), -1.0, 1.0)))
        if angle < max_angle:
            result.append((name, angle))

    # Sort: smallest angle = most directly facing
    result.sort(key=lambda x: x[1])
    return [name for name, _ in result]


def _vlm_wall4(
    image_path:     Path,
    mask_path:      Path,
    seg:            dict,
    hint_wall:      str        = "back",
    wall_context:   dict | None = None,
    visible_walls:  list[str] | None = None,
    branches:       dict | None = None,  # from _classify_wall_branches
    corner_world:   np.ndarray | None = None,  # 3-D world position of deepest corner
    # Camera params
    cam_pos:        np.ndarray | None = None,
    cam_right:      np.ndarray | None = None,
    cam_fwd_h:      np.ndarray | None = None,
    cam_tilt_tan:   float = 0.0,
    cam_fx:         float = 100.0,
    cam_cx:         float = 0.0,
    cam_cy:         float = 0.0,
    cam_W_px:       int = 183,
    cam_H_px:       int = 275,
    room_W_m:       float = 3.5,
    room_D_m:       float = 3.5,
    room_ceil:      float = 2.7,
) -> str | None:
    """Ask VLM which visible wall an object is mounted on.

    Uses corner-relative descriptions (left/right of the deepest corner as seen
    in the image) so VLM reasons about what it actually sees, not abstract 3-D
    world-coordinate labels.

    Returns a world wall name ("back", "left", or "right"), or None on failure.
    """
    import requests

    wctx = wall_context or {}
    allowed = [w for w in (visible_walls or ["back", "left", "right"]) if w != "front"]
    if not allowed:
        allowed = ["back", "left", "right"]

    safe_hint = hint_wall if hint_wall in allowed else allowed[0]

    def _wall_feat_str(w: str) -> str:
        ctx   = wctx.get(w, {})
        feats = ctx.get("features", [])
        if not feats:
            return "(plain wall, no openings)"
        parts = []
        for f in feats:
            ftype = f.get("type", "opening")
            w_m   = f.get("width_m", "?")
            h_m   = f.get("height_m", "?")
            off_m = f.get("offset_from_left_m", "?")
            parts.append(f"{ftype} {w_m}×{h_m}m")
        return "(" + "; ".join(parts) + ")"

    # ── Project deepest corner to pixel column ────────────────────────────────
    corner_col = cam_W_px // 2   # fallback: image centre
    if (corner_world is not None and cam_pos is not None
            and cam_right is not None and cam_fwd_h is not None):
        proj = _project_world_to_px(
            corner_world, cam_pos, cam_right, cam_fwd_h,
            cam_tilt_tan, cam_fx, cam_cx, cam_cy,
        )
        if proj is not None:
            corner_col = int(round(proj[0]))

    # ── Determine left/right branch world names ───────────────────────────────
    if branches is not None:
        left_wall_name  = branches["left_wall"]
        right_wall_name = branches["right_wall"]
    else:
        # Fallback without branch info: pick from allowed
        left_wall_name  = allowed[0]
        right_wall_name = allowed[1] if len(allowed) > 1 else allowed[0]

    left_feats  = _wall_feat_str(left_wall_name)
    right_feats = _wall_feat_str(right_wall_name)

    x1, y1, x2, y2 = seg["box_px"]

    prompt = _VLM_WALL4_PROMPT.format(
        obj_type        = seg.get("type", "object"),
        img_w           = cam_W_px,
        img_h           = cam_H_px,
        corner_col      = corner_col,
        left_wall_name  = left_wall_name,
        right_wall_name = right_wall_name,
        left_feats      = left_feats,
        right_feats     = right_feats,
        obj_col_lo      = x1,
        obj_col_hi      = x2,
        obj_row_lo      = y1,
        obj_row_hi      = y2,
        hint_wall       = safe_hint,
    )

    try:
        from PIL import ImageDraw
        room_img = Image.open(image_path).convert("RGB").copy()
        draw = ImageDraw.Draw(room_img)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        # Draw a vertical line at corner_col so VLM can see the split point
        draw.line([(corner_col, 0), (corner_col, cam_H_px)], fill=(0, 200, 0), width=2)

        overlay = room_img.copy()
        if mask_path.exists():
            mask_arr = np.array(Image.open(mask_path).convert("L"))
            mask_bin = (mask_arr > 0).astype(np.uint8) * 180
            mask_rgba = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
            mask_rgba[:, :, 0] = 255
            mask_rgba[:, :, 3] = mask_bin
            mask_img = Image.fromarray(mask_rgba, "RGBA")
            overlay  = overlay.convert("RGBA")
            overlay.alpha_composite(mask_img)
            overlay  = overlay.convert("RGB")

        def _enc_img(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        b64_room    = _enc_img(room_img)
        b64_overlay = _enc_img(overlay)
    except Exception as e:
        print(f"  [vlm_wall4] image prepare failed: {e}")
        return None

    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_room}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_overlay}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw  = resp.json()["choices"][0]["message"]["content"]
        res  = _parse_json(raw)
        if isinstance(res, dict) and "wall" in res:
            w = str(res["wall"]).lower().strip()
            if w in allowed:
                print(f"  [vlm_wall4] → {w}  (hint={safe_hint}, corner_col={corner_col}, "
                      f"left={left_wall_name}, right={right_wall_name})")
                return w
            print(f"  [vlm_wall4] VLM returned '{w}' not in allowed={allowed}, ignoring")
    except Exception as e:
        print(f"  [vlm_wall4] failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Corner-based placement main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_corner_based(
    output_dir:         str | Path,
    image_path:         str | Path,
    target_types:       list[str] | None = None,
    limit:              int | None = None,
    refine_iters:       int  = 5,
    use_vlm:            bool = True,
    proj_tex_image_path: str | Path | None = None,
) -> None:
    """Placement pipeline using the deepest-corner coordinate system.

    Pipeline per object:
      1. Determine which corner (back-left or back-right) is deepest from camera.
      2. Ask VLM which wall (back or side) using the corner coordinate frame.
      3. Numerically estimate the object height by projecting the mask top/bottom
         pixels through the camera to world Y coordinates.
      4. Set the wall-normal axis to 0 (Y=0 for back wall, X=0 for side wall).
         Estimate the free-axis initial position by back-projecting the mask centroid.
      5. Scale and place the GLB at the computed position.
      6. Iteratively shift the free axis so the rendered centroid aligns with the
         mask centroid (Z / height stays fixed throughout).
      7. Check for bounding-box collisions with previously placed wall objects.
         Slide minimally along the free axis if needed.
      8. Append to OBJ, render, and save the updated wall-placement state to a txt file.
    """
    out_dir      = Path(output_dir)
    openings_dir = out_dir / "openings"
    wm_dir       = out_dir / "wall_mounted"
    obj_dir      = wm_dir / "objects"
    place_dir    = wm_dir / "placements"
    place_dir.mkdir(parents=True, exist_ok=True)
    image_path   = Path(image_path)

    # ── Camera ────────────────────────────────────────────────────────────────
    for _cp in [out_dir / "camera_vggt.json",
                out_dir / "camera.json",
                openings_dir / "camera.json"]:
        if _cp.exists():
            cam_path = _cp
            break
    else:
        raise FileNotFoundError(f"No camera.json found in {out_dir}")
    cam = json.loads(cam_path.read_text())

    # ── Room dimensions ───────────────────────────────────────────────────────
    analysis = {}
    anal_path = out_dir / "floorplan_analysis.json"
    if anal_path.exists():
        analysis = json.loads(anal_path.read_text())
    room = dict(analysis.get("room", {}))
    base_obj = (openings_dir / "walls_with_openings.obj"
                if (openings_dir / "walls_with_openings.obj").exists()
                else out_dir / "walls.obj")
    obj_dims = _room_dims_from_obj(base_obj)
    if obj_dims:
        room.update(obj_dims)

    W_m  = float(room["floor_width_m"])
    D_m  = float(room["floor_depth_m"])
    ceil = float(room.get("ceiling_height_m", 2.7))

    # ── Re-verify wall_context vs rendered walls ──────────────────────────────
    # After VGGT calibration the camera may have shifted enough that Stage 1's
    # VLM wall labels ("back"/"left"/"right") no longer match the rendered view.
    # Ask the VLM to compare the render against the feature descriptions and
    # correct any left/right or back/side swap before placement begins.
    render_vggt = out_dir / "render_vggt.png"
    cam = _recheck_wall_context(cam, render_vggt, W_m, D_m)

    # ── Camera setup ──────────────────────────────────────────────────────────
    pos, right, up_w, fwd_h, tilt_tan, fx, cx_px, cy_px, W_px, H_px = _camera_setup(cam)
    _project = _make_projector(pos, np.array(cam["look_at_m"]), up_w,
                                float(cam["hfov_deg"]), W_px, H_px)

    # ── Visible walls from camera geometry ────────────────────────────────────
    vis_walls = _camera_visible_walls(
        pos, np.array(cam["look_at_m"]),
        float(cam["hfov_deg"]), W_m, D_m,
    )
    print(f"[corner] visible walls (camera geometry): {vis_walls}")

    # ── Corner coordinate system ──────────────────────────────────────────────
    csys     = _corner_coord_system(room, pos)
    branches = _classify_wall_branches(csys, right)
    print(f"[corner] deepest corner = {csys['corner_world'].tolist()}  "
          f"wall1={csys['wall1']}  wall2={csys['wall2']}")
    print(f"[corner] image-left branch = {branches['left_wall']} ({branches['left_len']:.2f}m)  "
          f"image-right branch = {branches['right_wall']} ({branches['right_len']:.2f}m)")

    # ── Segments ──────────────────────────────────────────────────────────────
    seg_results = json.loads((wm_dir / "segment_results.json").read_text())
    segments    = seg_results.get("segments", [])
    targets     = [
        s for s in segments
        if (target_types is None or s.get("type") in target_types)
        and s.get("glb_file")
        and (obj_dir / Path(s["glb_file"]).name).exists()
    ]
    if limit is not None:
        targets = targets[:limit]
    # Process window/door planes first so other objects can collide-avoid them
    _PLANE_TYPES = {"window", "door"}
    targets = (
        [s for s in targets if s.get("type") in _PLANE_TYPES] +
        [s for s in targets if s.get("type") not in _PLANE_TYPES]
    )
    print(f"[corner] {len(targets)} segment(s) of type(s) {target_types}")
    if not targets:
        return

    # ── OBJ scaffold ─────────────────────────────────────────────────────────
    windows_obj  = place_dir / "walls_with_windows.obj"
    plain_obj    = openings_dir / "walls_with_openings.obj"
    fallback_obj = out_dir / "walls.obj"
    # Window/door are plane GLBs placed flush on the wall.
    # Never use the holes-only mesh — always use filled or plain walls.
    placing_openings = False
    obj_in_path = (windows_obj  if windows_obj.exists()
                   else fallback_obj)
    obj_out_path = place_dir / "walls_with_objects.obj"

    orig_lines = Path(obj_in_path).read_text().splitlines(keepends=True)

    # ── Fresh wall state for this run (collision tracking within this run only) ─
    # Previous runs' placements are not in the new OBJ, so don't load stale state.
    state_path = place_dir / "wall_placements.txt"
    wall_state: list[dict] = []

    # Pre-seed collision state with window/door openings so wall-mounted objects
    # are slid away from windows rather than placed on top of them.
    opening_json = place_dir / "opening_placements.json"
    if opening_json.exists():
        try:
            openings = json.loads(opening_json.read_text())
            for op in openings:
                op_wall = str(op.get("wall", "")).lower()
                if op_wall not in ("back", "front", "left", "right"):
                    continue
                pos_w  = np.array(op["position"], dtype=float)
                ow     = float(op["opening_width_m"])
                oh     = float(op["opening_height_m"])
                sill   = float(op.get("sill_height_m", 0.0))
                hw = ow / 2.0
                if op_wall in ("back", "front"):
                    free_min = pos_w[0] - hw
                    free_max = pos_w[0] + hw
                else:
                    free_min = pos_w[2] - hw
                    free_max = pos_w[2] + hw
                wall_state.append({
                    "seg_idx":  -1,          # sentinel: not a placed object
                    "wall":     op_wall,
                    "free_min": free_min,
                    "free_max": free_max,
                    "z_min":    sill,
                    "z_max":    sill + oh,
                })
            if wall_state:
                print(f"[corner] Pre-seeded {len(wall_state)} window/door bbox(es) into collision state")
        except Exception as e:
            print(f"[corner] Warning: could not load opening_placements.json: {e}")

    # Build segment→opening lookup for windows/doors: exact position + dimensions.
    opening_by_seg: dict[int, dict] = {}
    if opening_json.exists():
        try:
            for op in json.loads(opening_json.read_text()):
                si = op.get("matched_segment_index")
                if si is not None:
                    opening_by_seg[int(si)] = op
        except Exception:
            pass

    # Load previously saved placements and carry over types not being re-placed.
    placements_path = place_dir / "object_placements.json"
    carryover_placements: list[dict] = []
    if placements_path.exists() and target_types:
        try:
            existing = json.loads(placements_path.read_text())
            carryover_placements = [p for p in existing
                                    if p.get("type") not in target_types]
            if carryover_placements:
                print(f"[corner] Carrying over {len(carryover_placements)} existing "
                      f"placements for non-targeted types.")
        except Exception as e:
            print(f"[corner] Warning: could not load existing placements ({e})")

    # ── Order targets: corner walls first, then back wall; within each wall
    # closest-to-corner segment first.  This means collision sliding always
    # moves objects away from already-placed inner ones, never the reverse.
    targets = _wall_placement_order(
        targets, csys, pos, right, fwd_h, tilt_tan, fx, cx_px, cy_px,
        W_m, D_m, ceil,
    )

    # ── Seed carryover placements into collision state ─────────────────────────
    # Placements from previous runs must be visible to _slide_from_collision so
    # newly placed objects don't land on top of already-placed ones.
    for _cp in carryover_placements:
        _cp_wall = str(_cp.get("wall", "")).lower()
        if _cp_wall not in ("back", "front", "left", "right"):
            continue
        _cp_pos  = np.array(_cp.get("world_pt", _cp.get("position", [0, 0, 0])), dtype=float)
        _cp_sz   = _cp.get("size_m", _cp.get("size", {}))
        _cp_w    = float(_cp_sz.get("width_m",  0.5))
        _cp_h    = float(_cp_sz.get("height_m", 0.5))
        _cp_hw   = _cp_w / 2.0
        if _cp_wall in ("back", "front"):
            _cp_fmin, _cp_fmax = _cp_pos[0] - _cp_hw, _cp_pos[0] + _cp_hw
        else:
            _cp_fmin, _cp_fmax = _cp_pos[2] - _cp_hw, _cp_pos[2] + _cp_hw
        wall_state.append({
            "seg_idx":  _cp.get("segment_index", _cp.get("seg_idx", -99)),
            "wall":     _cp_wall,
            "free_min": _cp_fmin,
            "free_max": _cp_fmax,
            "z_min":    float(_cp_pos[1]) - _cp_h / 2.0,
            "z_max":    float(_cp_pos[1]) + _cp_h / 2.0,
        })
    if carryover_placements:
        print(f"[corner] Seeded {len(carryover_placements)} carryover placement(s) into collision state")

    placements:        list[dict]  = []
    placed_raw:        list[tuple] = []   # (comment, verts_array, faces_array) for deferred OBJ write
    window_footprints: list[dict]  = []   # footprints of placed windows/doors (for hole rebuild)

    for seg in targets:
        idx      = seg["index"]
        obj_type = seg.get("type", "other")
        glb_path = obj_dir / Path(seg["glb_file"]).name
        x1, y1, x2, y2 = seg["box_px"]
        print(f"\n[corner] Segment {idx:02d} ({obj_type})  bbox=[{x1},{y1},{x2},{y2}]")

        if use_vlm and not _is_wall_mounted_vlm(image_path, seg, W_px, H_px):
            print(f"  [wall_check] not wall-mounted — skipping")
            continue

        mask_path = _mask_path_for_seg(seg, wm_dir)

        # ── Step 1: rough back-projection to find which wall ──────────────────
        cx_m = (x1 + x2) / 2.0
        cy_m = (y1 + y2) / 2.0
        if mask_path.exists():
            try:
                ma   = np.array(Image.open(mask_path))
                fg   = ma.sum(axis=2) > 0 if ma.ndim == 3 else ma > 0
                ys_m, xs_m = np.where(fg)
                if len(xs_m):
                    cx_m, cy_m = float(xs_m.mean()), float(ys_m.mean())
            except Exception:
                pass

        # ── Step 2a: Identify wall + free-axis position from mask pixel votes ────
        # Every mask pixel is back-projected; each votes for whichever wall it
        # hits naturally.  Dominant wall wins.  Median free-axis = position.
        # The mask and render share pixel space (same VGGT camera + resolution).
        mask_proj = _project_mask_to_scene(
            mask_path, pos, right, fwd_h, tilt_tan,
            fx, cx_px, cy_px, W_m, D_m, ceil)

        # ── Detect bbox edge-clipping early (before wall logic uses it) ────────
        # Python loops don't reset locals between iterations, so without
        # computing these per-iteration we'd carry over the previous segment's
        # clip flags and route the next segment through the wrong branch.
        _EDGE_TOL_PX = 5
        clipped_left   = int(x1) <= _EDGE_TOL_PX
        clipped_right  = int(x2) >= int(W_px) - _EDGE_TOL_PX
        clipped_top    = int(y1) <= _EDGE_TOL_PX
        clipped_bottom = int(y2) >= int(H_px) - _EDGE_TOL_PX
        x_clipped = clipped_left or clipped_right

        # Per-type default width:height ratio (used when bbox-derived width
        # is unreliable — i.e. X-clipped at the image border).
        _DEFAULT_ASPECT = {
            "art": 0.75, "painting": 0.75, "frame": 0.75, "poster": 0.75,
            "mirror": 0.65,
            "window": 0.60, "door": 0.45,
            "tv": 1.78, "shelf": 5.0, "clock": 1.0,
        }
        _aspect = _DEFAULT_ASPECT.get(obj_type, 0.75)

        if mask_proj is not None:
            wall      = mask_proj["wall"]
            # `free_med` (median of back-projected mask pixel Z's) collapses
            # near wall corners — all rays converge to the corner and the
            # median snaps to the corner coord, putting paintings at the
            # wrong end of the wall.  Use the bbox-CENTER back-projection
            # instead: a single ray through the silhouette's centre is
            # geometry-stable and matches what the photo's bbox-x order
            # implies.  Fall back to free_med only when the centre ray
            # misses the chosen wall.
            free_init = mask_proj["free_med"]
            try:
                _fa_idx = 0 if wall in ("back", "front") else 2

                # When the bbox is X-clipped at an image edge, the bbox
                # CENTRE is biased toward the image — back-projecting it
                # gives a center-Z that's too far inward.  Instead anchor
                # on the NON-clipped edge: back-project that edge pixel
                # and place the painting CENTER half-width from it,
                # extending toward the clipped side.
                _bbox_w_px_local = max(1, int(x2 - x1))
                _bbox_h_px_local = max(1, int(y2 - y1))
                # Width estimate for the half-width anchor offset: when
                # X-clipped, the bbox aspect undercounts → use default
                # aspect × estimated height (rough = 1m for art) as a
                # reasonable proxy until the actual height is known.
                _est_h = 1.0   # rough — refined later by mask height
                if x_clipped:
                    _half_w_world = (_est_h * _aspect) / 2.0
                else:
                    _half_w_world = (_est_h *
                                     (_bbox_w_px_local / _bbox_h_px_local)) / 2.0
                _anchor_px = None
                _anchor_dir_to_centre = 0  # +1 means painting centre is at higher world fa than anchor
                if clipped_left and not clipped_right:
                    # Left side cut off → anchor on right bbox edge
                    _anchor_px = (float(x2), cy_m)
                    # Map "image-x larger" to wall fa direction:
                    #   left wall (fa=Z): image-x↑ → world Z↓
                    #   right wall (fa=Z): image-x↑ → world Z↑
                    #   back wall (fa=X): image-x↑ → world X↑
                    #   front wall (fa=X): image-x↑ → world X↓
                    _anchor_dir_to_centre = {
                        "left":  +1, "right": -1,
                        "back":  -1, "front": +1,
                    }.get(wall, 0)
                elif clipped_right and not clipped_left:
                    # Right side cut off → anchor on left bbox edge
                    _anchor_px = (float(x1), cy_m)
                    _anchor_dir_to_centre = {
                        "left":  -1, "right": +1,
                        "back":  +1, "front": -1,
                    }.get(wall, 0)

                if _anchor_px is not None:
                    _ray = _backproject(_anchor_px[0], _anchor_px[1],
                                         pos, right, fwd_h, tilt_tan,
                                         fx, cx_px, cy_px)
                    _hit = _ray_hit_wall(pos, _ray, wall, W_m, D_m, ceil,
                                          tol=_WALL_HIT_TOL)
                    if _hit is not None:
                        _fa_anchor = float(_hit[_fa_idx])
                        _fa_centre = _fa_anchor + _anchor_dir_to_centre * _half_w_world
                        room_len = W_m if _fa_idx == 0 else D_m
                        if 0.0 <= _fa_centre <= room_len:
                            print(f"  [corner] X-clipped → anchor on "
                                  f"non-clipped edge px={_anchor_px[0]:.0f}: "
                                  f"fa_anchor={_fa_anchor:.3f}m  "
                                  f"fa_centre={_fa_centre:.3f}m "
                                  f"(was median={free_init:.3f}m)")
                            free_init = _fa_centre
                else:
                    # Neither side clipped — use bbox-centre back-projection
                    _bbox_ray = _backproject(cx_m, cy_m, pos, right, fwd_h,
                                             tilt_tan, fx, cx_px, cy_px)
                    _bbox_hit = _ray_hit_wall(pos, _bbox_ray, wall,
                                               W_m, D_m, ceil,
                                               tol=_WALL_HIT_TOL)
                    if _bbox_hit is not None:
                        _fa_bbox = float(_bbox_hit[_fa_idx])
                        if 0.0 <= _fa_bbox <= (W_m if _fa_idx == 0 else D_m):
                            if abs(_fa_bbox - free_init) > 0.20:
                                print(f"  [corner] free_init: "
                                      f"median={free_init:.3f}m → "
                                      f"bbox-centre back-proj={_fa_bbox:.3f}m  "
                                      f"(median collapsed near corner)")
                            free_init = _fa_bbox
            except Exception as _e:
                print(f"  [corner] anchor adjustment skipped: {_e}")
        else:
            # Fallback: single centroid ray for wall + position
            ray = _backproject(cx_m, cy_m, pos, right, fwd_h, tilt_tan, fx, cx_px, cy_px)
            hit = _find_wall(pos, ray, W_m, D_m, ceil)
            if hit is None:
                print("  [corner] ray miss — skipping")
                continue
            rough_wall, rough_world_pt = hit
            wall      = rough_wall
            free_init = float(rough_world_pt[0]) if wall in ("back","front") else float(rough_world_pt[2])
            print(f"  [corner] mask proj failed — centroid fallback wall={wall}")

        # ── Step 2b: Height from mask top/bottom pixels (exact, not percentile) ─
        # Project the topmost and bottommost mask pixels through the camera.
        # This gives the true world-Y extent of the object without p5/p95 clipping.
        # A reference depth point on the identified wall is needed for the camera-
        # space depth used in the back-projection formula.
        ref_ray = _backproject(cx_m, cy_m, pos, right, fwd_h, tilt_tan, fx, cx_px, cy_px)
        ref_hit = _ray_hit_wall(pos, ref_ray, wall, W_m, D_m, ceil, tol=_WALL_HIT_TOL)
        ref_pt  = ref_hit if ref_hit is not None else (pos + ref_ray * 3.0)
        height_result = _estimate_height_from_mask(
            mask_path, ref_pt, pos, fwd_h, fx, cy_px, tilt_tan, ceil)
        if height_result is not None:
            bot_wy, top_wy = height_result
            if obj_type == "curtain":
                # Distinguish a floor-length DRAPE from a short VALANCE/topper.
                # A valance has a wide, short silhouette (width >> height); a
                # full drape is tall.  Only a real drape should be pinned to the
                # floor — a valance must keep its mask-derived height, otherwise
                # a 53px-tall topper gets stretched floor-to-ceiling.
                _is_valance = False
                try:
                    import cv2 as _cv2
                    _mk = _cv2.imread(str(mask_path), 0)
                    _ys, _xs = np.where(_mk > 127)
                    if len(_xs) and (_ys.max() - _ys.min()) > 0:
                        if (_xs.max() - _xs.min()) > 2.5 * (_ys.max() - _ys.min()):
                            _is_valance = True
                except Exception:
                    pass
                if _is_valance:
                    object_h = float(np.clip(top_wy - bot_wy, 0.05, ceil))
                    center_y = (bot_wy + top_wy) / 2.0
                    print(f"  [place] curtain is a VALANCE (wide-short silhouette) "
                          f"— mask height {object_h:.2f}m, NOT floor-pinned")
                else:
                    # Curtains hang from the rod (silhouette top) to the floor
                    # regardless of how much cloth the segmentation mask captured.
                    # The absolute mask-top pixel is unreliable as the rod height
                    # (sheer fabric leaks pixels above the rod; camera projection
                    # clamps them at the ceiling), so use mask_proj["height_hi"]
                    # (p95 of the projected mask) when available.  Bottom pinned to
                    # the floor so generated meshes that don't fill their bbox still
                    # drape the full rod-to-floor span uniformly.
                    bot_wy = 0.0
                    if mask_proj is not None and "height_hi" in mask_proj:
                        top_wy = float(np.clip(mask_proj["height_hi"], 0.05, ceil))
                    else:
                        top_wy = float(np.clip(top_wy, 0.05, ceil))
                    object_h = top_wy
                    center_y = top_wy / 2.0
            else:
                object_h = float(np.clip(top_wy - bot_wy, 0.05, ceil))
                center_y = (bot_wy + top_wy) / 2.0
        else:
            object_h = ceil * 0.30
            center_y = ceil * 0.55
            print(f"  [corner] height fallback: h={object_h:.2f}m")

        # ── Step 3: VLM decides the wall; mask aligns the exact position ─────────
        # Wall assignment: geometry is ground truth when decisive; VLM only
        # tiebreaks when the mask votes are genuinely split across walls.
        # VLM cannot reliably map 2D image regions to 3D world-wall names —
        # it sees a perspective image and doesn't know the room's coordinate
        # frame.  Geometry (back-projection) does know exactly which wall each
        # pixel lands on.
        geo_hint = wall if wall in vis_walls else (vis_walls[0] if vis_walls else "back")
        wall_votes     = (mask_proj or {}).get("wall_votes", {})
        total_votes    = sum(wall_votes.values()) or 1
        dominant_share = wall_votes.get(geo_hint, 0) / total_votes

        # Only invoke VLM if geometry is ambiguous (dominant wall < 70% of votes)
        vlm_wall: str | None = None
        if use_vlm and dominant_share < 0.70:
            print(f"  [corner] geometry ambiguous ({dominant_share:.0%} on {geo_hint}) — asking VLM")
            vlm_wall = _vlm_wall4(
                image_path, mask_path, seg,
                hint_wall     = geo_hint,
                wall_context  = cam.get("wall_context", {}),
                visible_walls = vis_walls,
                branches      = branches,
                corner_world  = csys["corner_world"],
                cam_pos       = pos,
                cam_right     = right,
                cam_fwd_h     = fwd_h,
                cam_tilt_tan  = tilt_tan,
                cam_fx        = fx,
                cam_cx        = cx_px,
                cam_cy        = cy_px,
                cam_W_px      = W_px,
                cam_H_px      = H_px,
                room_W_m      = W_m,
                room_D_m      = D_m,
                room_ceil     = ceil,
            )
        else:
            print(f"  [corner] geometry decisive ({dominant_share:.0%} on {geo_hint}) — skipping VLM")

        if vlm_wall is not None and vlm_wall != wall:
            print(f"  [corner] VLM wall4: {wall} → {vlm_wall}")
            wall = vlm_wall

            # Re-derive free-axis position for the VLM-chosen wall.
            # _project_mask_to_scene finds the nearest-hit wall, which may differ
            # from the VLM wall when the camera is angled (e.g. a left-wall object's
            # pixels mostly hit the back wall by ray distance).
            # Instead: shoot each mask pixel ray at the VLM wall specifically, collect
            # free-axis hits, and take the median.
            fa_vlm  = 0 if wall in ("back", "front") else 2
            fa_hits: list[float] = []
            try:
                ma_arr = np.array(Image.open(mask_path))
                fg_arr = ma_arr.sum(axis=2) > 0 if ma_arr.ndim == 3 else ma_arr > 0
                ys_m2, xs_m2 = np.where(fg_arr)
                step = max(1, len(xs_m2) // 400)   # sample up to ~400 pixels
                for i in range(0, len(xs_m2), step):
                    r = _backproject(float(xs_m2[i]), float(ys_m2[i]),
                                     pos, right, fwd_h, tilt_tan, fx, cx_px, cy_px)
                    pt = _ray_hit_wall(pos, r, wall, W_m, D_m, ceil, tol=_WALL_HIT_TOL)
                    if pt is not None:
                        fa_hits.append(float(pt[fa_vlm]))
            except Exception as _e:
                print(f"  [corner] wall re-project error: {_e}")

            if len(fa_hits) >= 3:
                free_init = float(np.median(fa_hits))
                # Also update mask_width from p5/p95 span on the VLM wall
                fa_arr = np.array(fa_hits)
                vlm_free_lo = float(np.percentile(fa_arr,  5))
                vlm_free_hi = float(np.percentile(fa_arr, 95))
                # Temporarily patch mask_proj width fields so downstream uses correct wall
                if mask_proj is not None:
                    mask_proj = dict(mask_proj,
                                     free_med=free_init,
                                     free_lo=vlm_free_lo,
                                     free_hi=vlm_free_hi)
                print(f"  [corner] VLM-wall re-project: {len(fa_hits)} hits → "
                      f"free_med={free_init:.3f}m  span=[{vlm_free_lo:.3f},{vlm_free_hi:.3f}]m")
            else:
                # Not enough rays hit the VLM wall — use centroid ray fallback
                cray = _backproject(cx_m, cy_m, pos, right, fwd_h, tilt_tan, fx, cx_px, cy_px)
                cpt  = _ray_hit_wall(pos, cray, wall, W_m, D_m, ceil, tol=_WALL_HIT_TOL)
                if cpt is not None:
                    free_init = float(cpt[fa_vlm])
                    print(f"  [corner] VLM-wall centroid fallback: free={free_init:.3f}m")
                else:
                    # Last resort: place at wall midpoint
                    free_init = (W_m if fa_vlm == 0 else D_m) / 2.0
                    print(f"  [corner] VLM-wall midpoint fallback: free={free_init:.3f}m")
        else:
            print(f"  [corner] VLM wall4 confirms: {wall}")

        # Width from silhouette bbox aspect × already-computed height.
        # `clipped_left/right`, `x_clipped`, and `_aspect` are computed at the
        # top of this iteration (hoisted so wall-positioning code can use
        # them too).  Here we just consume them.
        bbox_w_px = max(1, int(x2 - x1))
        bbox_h_px = max(1, int(y2 - y1))

        if x_clipped:
            # Bbox width is unreliable — derive width from height × default aspect.
            mask_width_aspect = object_h * _aspect
            print(f"  [corner] bbox X-clipped at image border "
                  f"(left={clipped_left}, right={clipped_right}) — "
                  f"using default aspect {_aspect:.2f} → "
                  f"width={mask_width_aspect:.3f}m (was bbox-aspect "
                  f"{object_h * (bbox_w_px / bbox_h_px):.3f}m)")
        else:
            mask_width_aspect = object_h * (bbox_w_px / bbox_h_px)

        mask_width_freespan = (float(mask_proj["free_hi"] - mask_proj["free_lo"])
                               if mask_proj is not None else 0.0)
        mask_width = max(mask_width_aspect, mask_width_freespan)
        mask_width = float(np.clip(mask_width, 0.05, max(W_m, D_m)))
        if not x_clipped:
            print(f"  [corner] mask_width: aspect={mask_width_aspect:.3f}m "
                  f"freespan={mask_width_freespan:.3f}m → using {mask_width:.3f}m")
        _default_depth = _TYPE_DEFAULTS.get(obj_type, _TYPE_DEFAULTS["other"])["depth_m"]
        size = {
            "height_m": object_h,
            "width_m":  mask_width,
            "depth_m":  _default_depth,
        }

        # ── Grazing-angle sanity clamp ────────────────────────────────────────
        # On a wall the camera views near edge-on (e.g. a mirror on the right
        # wall while looking down the room), the mask→scene projection collapses
        # object_h / mask_width and can flip the aspect (mask taller-than-wide but
        # result wider-than-tall), yielding a 0.27×0.09 "mirror". Detect the
        # implausible / aspect-flipped case from the segmentation bbox and rebuild
        # the size from the bbox aspect anchored to the type default. Update
        # object_h / mask_width too so the downstream mesh scaling follows.
        if obj_type not in ("window", "door", "curtain"):
            _bw = max(float(x2 - x1), 1.0); _bh = max(float(y2 - y1), 1.0)
            _bbox_asp = _bh / _bw
            _sz_asp   = size["height_m"] / max(size["width_m"], 1e-3)
            _defs     = _TYPE_DEFAULTS.get(obj_type, _TYPE_DEFAULTS["other"])
            if (max(size["width_m"], size["height_m"]) < 0.30
                    or (_bbox_asp >= 1.0) != (_sz_asp >= 1.0)
                    or abs(np.log(_sz_asp / max(_bbox_asp, 1e-3))) > np.log(2.2)):
                if _bbox_asp >= 1.0:                # taller than wide
                    size["height_m"] = _defs["height_m"]
                    size["width_m"]  = _defs["height_m"] / _bbox_asp
                else:
                    size["width_m"]  = _defs["width_m"]
                    size["height_m"] = _defs["width_m"] * _bbox_asp
                object_h   = size["height_m"]
                mask_width = size["width_m"]
                print(f"  [corner] grazing-wall size clamp ({obj_type}): "
                      f"bbox {_bw:.0f}×{_bh:.0f}px asp={_bbox_asp:.2f}, raw asp={_sz_asp:.2f} "
                      f"→ {size['width_m']:.2f}×{size['height_m']:.2f}m")

        # ── Build world_pt with wall constraint + free axis ───────────────────

        # Slide along the mounted wall surface (free axis only) so bbox stays
        # inside the room.  Never moves perpendicular to the wall.
        free_ax  = 0 if wall in ("back", "front") else 2
        room_len = W_m if free_ax == 0 else D_m
        hw_obj   = size["width_m"] / 2.0
        free_init_clamped = float(np.clip(free_init, hw_obj, room_len - hw_obj))
        if abs(free_init_clamped - free_init) > 1e-4:
            print(f"  [corner] bbox OOB: free_pos {free_init:.3f}m → {free_init_clamped:.3f}m "
                  f"(slid along ax{free_ax} to fit in [0,{room_len:.2f}]m)")
        free_init = free_init_clamped

        # ── Painting-below-window constraint ─────────────────────────────────
        # Art/photo on the same wall as a window and horizontally overlapping
        # it should hang BELOW the sill, not at curtain/window height.
        _ART_BELOW_WIN_TYPES = {"art", "photo", "painting", "frame"}
        if obj_type in _ART_BELOW_WIN_TYPES:
            _half_w = size["width_m"] / 2.0
            _pf_lo  = free_init - _half_w
            _pf_hi  = free_init + _half_w
            for _ws in wall_state:
                if _ws.get("wall") != wall or _ws.get("seg_idx", 0) != -1:
                    continue
                _sill = float(_ws.get("z_min", 0.0))
                if _sill < 0.15:  # skip floor-level openings (doors)
                    continue
                if _pf_hi > float(_ws["free_min"]) and _pf_lo < float(_ws["free_max"]):
                    _max_top = _sill
                    if center_y + object_h / 2.0 > _max_top:
                        center_y = float(np.clip(
                            _max_top - object_h / 2.0,
                            object_h / 2.0, ceil - object_h / 2.0,
                        ))
                        print(f"  [corner] painting-below-window: sill={_sill:.2f}m "
                              f"→ center_y clamped to {center_y:.2f}m")
                    break

        if wall == "back":
            world_pt = np.array([free_init, center_y, 0.0])
        elif wall == "front":
            world_pt = np.array([free_init, center_y, D_m])
        elif wall == "left":
            world_pt = np.array([0.0, center_y, free_init])
        else:  # right
            world_pt = np.array([W_m, center_y, free_init])

        print(f"  [corner] wall={wall}  world_pt={np.round(world_pt,3).tolist()}  "
              f"h={object_h:.2f}m  cy={center_y:.2f}m")

        # ── Step 5: Orientation ───────────────────────────────────────────────
        # the 3D generator convention:
        #   flip_y=True  — image Y increases downward; must flip for world-up.
        #   flip_z=True  — the near/visible face is at min Z; the snap in
        #                  _transform_glb_to_wall puts min-Z on the wall, so we
        #                  must flip Z to bring the visible face to face the room.
        #   flip_x=False — no horizontal mirror by default.
        # VLM grid check overrides these when a reference image is available.
        orient = {
            "flip_y": True,
            "flip_x": False,
            "flip_z": True,
        }
        canvas_file = seg.get("inpaint_file") or seg.get("canvas_file")

        # Determine reference image for VLM orientation check.
        canvas_path_c = None
        if canvas_file:
            cp = (wm_dir / "inpainted" / seg["inpaint_file"] if seg.get("inpaint_file")
                  else wm_dir / "segmented" / seg["canvas_file"])
            if cp.exists():
                canvas_path_c = cp

        ref_for_orient = canvas_path_c
        if ref_for_orient is None:
            try:
                _crop_path = place_dir / f"_orient_ref_{idx:02d}.png"
                Image.open(image_path).convert("RGB").crop(
                    (x1, y1, x2, y2)).save(str(_crop_path))
                ref_for_orient = _crop_path
                print(f"  [orient] no canvas file — using bbox crop as reference")
            except Exception as _e:
                print(f"  [orient] bbox crop failed: {_e}")

        # For windows/doors/curtains: VLM flip_z is unreliable (thin flat GLBs with
        # ambiguous face directions; windows use frame-snap which overrides geometry).
        # For art/mirror/frame/painting: the VLM can clearly identify front vs back,
        # so trust its result.
        _skip_vlm_flip_z = obj_type in ("window", "door", "curtain")
        front_yaw_offset  = 0      # extra yaw from front-face detection (X-facing objects)
        front_face_flip_z = None   # authoritative flip_z from front-face check

        # ── Per-segment manual orient override ──────────────────────────────
        # When the VLM picks the wrong panel for a particular GLB, set
        # `"orient_override": {"flip_y": bool, "flip_x": bool, "flip_z":
        # bool, "yaw_offset_deg": int}` on the segment_results entry to
        # bypass both VLM calls.  Any unset key keeps the default.
        _orient_override = seg.get("orient_override") or {}
        if _orient_override:
            if "flip_y" in _orient_override:
                orient["flip_y"] = bool(_orient_override["flip_y"])
            if "flip_x" in _orient_override:
                orient["flip_x"] = bool(_orient_override["flip_x"])
            if "flip_z" in _orient_override:
                orient["flip_z"] = bool(_orient_override["flip_z"])
                front_face_flip_z = orient["flip_z"]
            if "yaw_offset_deg" in _orient_override:
                front_yaw_offset = int(_orient_override["yaw_offset_deg"])
            print(f"  [orient_override] {dict(orient)} "
                  f"yaw_off={front_yaw_offset}° (skipping VLM)")
        elif use_vlm and ref_for_orient is not None:
            # Step A: 4-direction front-face check — authoritative for 3D objects and
            # art/mirror/frame/painting.  Skipped for window/door/curtain (frame-snap
            # corrects geometry; face detection unreliable for glazed/patterned surfaces).
            if not _skip_vlm_flip_z:
                ff = _vlm_front_face_check(glb_path, ref_for_orient, obj_type)
                if ff is not None:
                    orient["flip_z"]  = ff["flip_z"]
                    front_face_flip_z = ff["flip_z"]
                    front_yaw_offset  = int(ff.get("yaw_offset", 0))

            # Step B: full 4-panel grid check — refines flip_y; also refines flip_z
            # for art/mirror/frame/painting when front-face check didn't pin it.
            vlm_orient = _vlm_orientation_check(glb_path, ref_for_orient, obj_type)
            if vlm_orient is not None:
                orient["flip_y"] = vlm_orient["flip_y"]
                orient["flip_x"] = vlm_orient.get("flip_x", False)
                if not _skip_vlm_flip_z and front_face_flip_z is None:
                    # No dedicated front-face check — trust the grid result for flip_z too
                    orient["flip_z"] = vlm_orient["flip_z"]

        # Flat plane GLBs (window/door/art/…): flip_y=True.
        # generate_window_plane creates the mesh with image-row-0 at world +Y,
        # but the OBB + placement pipeline inverts Y relative to image convention,
        # so flip_y=True is required to display the texture right-side up.
        _PLANE_OBJ_TYPES_PLACE = ("window", "door", "art", "painting", "frame")
        if obj_type in _PLANE_OBJ_TYPES_PLACE:
            flip_y = True
            flip_x = False
            orient = dict(orient)   # copy — don't mutate the original seg dict
            orient["flip_y"] = True
            orient["flip_x"] = False
        else:
            flip_y = orient["flip_y"]
            flip_x = orient["flip_x"]
        flip_z = orient.get("flip_z", True)

        # ── Step 5 / 6: VLM front-view + iterative free-axis refinement ─────────
        # All object types (including windows) use the same pipeline here.
        # Windows are placed like art/curtains: back face flush to wall, bbox_align
        # yaw + VLM disambiguation. A wall hole is cut from their footprint afterward.
        glb_front_deg = 0
        rough_size    = {"height_m": ceil * 0.4, "width_m": 0.5, "depth_m": 0.3}
        if use_vlm:
            front_deg_vlm = _vlm_front_view(
                glb_path, image_path if not canvas_file else
                (wm_dir / "inpainted" / seg["inpaint_file"]
                 if seg.get("inpaint_file") and
                 (wm_dir / "inpainted" / seg["inpaint_file"]).exists()
                 else image_path),
                obj_type, rough_size, _project, pos, world_pt, W_px, H_px,
            )
            if front_deg_vlm is not None:
                glb_front_deg = front_deg_vlm
        # ── Wall-consistency guard on the front-face yaw offset ────────────────
        # `_vlm_front_face_check` maps its 4-way answer to a yaw_offset of
        # 0 / 180 (front at ±Z) or ±90 (front at ±X).  Which family is
        # geometrically valid depends on the wall the object sits on:
        #   • back / front walls → room-facing normal is ±Z, so the front must
        #     end up facing ±Z.  A ±90 offset rotates the front onto the wall-
        #     TANGENT (±X) axis, turning a flat panel edge-on — it renders as a
        #     tilted parallelogram jutting into the room (the floating-TV bug).
        #     Only 0/180 are admissible.
        #   • left / right walls → room-facing normal is ±X, so ±90 is exactly
        #     right and 0/180 would be edge-on instead.
        # The front-face VLM (especially when its render is ambiguous or the
        # call falls back after a timeout) sometimes returns the wrong family
        # for the wall.  Reject an offset whose axis-family contradicts the
        # wall so a mis-pick can never produce the tilted/floating artifact.
        # This is geometry, not a heuristic, so it is safe for every type.
        _wall_normal_is_z = wall in ("back", "front")
        _offset_on_tangent = abs(int(front_yaw_offset) % 180) == 90  # ±90 / ±270
        if front_yaw_offset != 0 and _offset_on_tangent == _wall_normal_is_z:
            # Family mismatch — drop the tangential/normal rotation but keep the
            # 180° front/back ambiguity component (valid on any wall).
            _kept = 180 if (int(front_yaw_offset) % 360) in (180, 270) else 0
            print(f"  [front_face] yaw_offset {front_yaw_offset:+d}° is "
                  f"incompatible with a {wall}-wall normal "
                  f"({'±Z' if _wall_normal_is_z else '±X'}) — clamping to "
                  f"{_kept}° to keep the panel flush against the wall")
            front_yaw_offset = _kept

        # Add yaw offset from front-face detection (non-zero when front is along X axis).
        # Only apply for flat objects (windows/curtains/art etc.) where glb_front_deg
        # is a simple world-space yaw. For 3D objects (shelves, cabinets) the OBB
        # "most-Z-aligned" heuristic already selects the correct depth axis from the
        # original GLB orientation; adding yaw_offset afterward would rotate the
        # already-correct shelf and point the thin side at the wall instead of the back.
        _is_detilt_obj = obj_type not in (
            "window", "door", "curtain", "art", "frame", "painting"
        )
        if front_yaw_offset != 0 and not _is_detilt_obj:
            glb_front_deg = (glb_front_deg + front_yaw_offset) % 360
            print(f"  [front_face] applied yaw_offset {front_yaw_offset:+d}° → "
                  f"glb_front_deg={glb_front_deg}")

        # Windows/doors: skip iterative centroid refinement — handles or
        # non-symmetric parts shift the rendered centroid, causing frame drift.
        if obj_type not in ("window", "door"):
            world_pt = _refine_free_axis(
                glb_path, wall, world_pt, size, room,
                pos, fwd_h, fx, _project, mask_path,
                flip_y=flip_y, flip_x=flip_x, flip_z=flip_z,
                glb_front_deg=glb_front_deg,
                front_yaw_offset=front_yaw_offset,
                detilt=(obj_type not in ("window", "door", "curtain",
                                         "art", "frame", "painting")),
                max_iters=refine_iters,
            )
            print(f"  [corner] after refinement: world_pt={np.round(world_pt,3).tolist()}")

        # ── Step 7: Collision detection — slide if overlapping ─────────────────
        # Skip sliding for: windows/doors (fixed by mask), and curtains/blinds
        # (they intentionally overlap a window opening — conflict with the window
        # placement is expected and correct).
        _SKIP_COLLISION = {"window", "door", "curtain", "blind", "shutter"}
        if obj_type not in _SKIP_COLLISION:
            # Frame-shaped objects (paintings, mirrors, clocks) can need a
            # large slide when their initial back-projected position is at
            # a wall corner where rays converge.  The default 0.15 m cap
            # is too tight (clear gap is often 1-2 m away).  Allow up to
            # half the wall length for these flat types.
            _flat_frames = {"art", "painting", "frame", "mirror", "clock"}
            if obj_type in _flat_frames:
                _wall_axis_len = (float(room.get("floor_width_m", 4.0))
                                  if wall in ("back", "front")
                                  else float(room.get("floor_depth_m", 4.0)))
                _max_slide = _wall_axis_len / 2.0
                slid = _slide_from_collision(
                    wall, world_pt, size, room, wall_state, idx,
                    max_slide_m=_max_slide,
                )
            else:
                slid = _slide_from_collision(wall, world_pt, size, room, wall_state, idx)
            if slid is None:
                print(f"  [corner] no clear wall position for seg {idx} — skipping")
                continue
            world_pt = slid

        # ── Step 8: Final GLB transform ───────────────────────────────────────
        result = _transform_glb_to_wall(
            glb_path, wall, world_pt, size, room,
            cam_pos=pos, flip_y=flip_y, flip_x=flip_x, flip_z=flip_z,
            glb_front_deg=glb_front_deg,
            front_yaw_offset=front_yaw_offset,
            embed=False,
            outward=False,   # plane GLB: back face flush to wall, textured face into room
            detilt=(obj_type not in ("window", "door", "curtain",
                                       "art", "frame", "painting")),
            skip_obb_swap=(obj_type in ("window", "door")),
            force_upright=(obj_type in ("window", "door")),
            force_floor=(obj_type == "door"),
        )
        if result is None:
            print("  [corner] transform failed — skipping")
            continue
        verts, faces, vc = result

        # ── Window/door plane: snap to mask footprint (no wall hole) ─────────────
        # Scale the plane GLB so it exactly covers the 2D mask-projected region.
        # Both horizontal and vertical extents come directly from mask_proj so
        # the plane matches the segmented mask, not any VLM-estimated size.
        if obj_type in ("window", "door"):
            fa = 0 if wall in ("back", "front") else 2

            frm_fa_lo = float(np.percentile(verts[:, fa], 2))
            frm_fa_hi = float(np.percentile(verts[:, fa], 98))
            frm_y_lo  = float(np.percentile(verts[:, 1],  2))
            frm_y_hi  = float(np.percentile(verts[:, 1],  98))

            if mask_proj is not None:
                tgt_fa_lo = float(mask_proj["free_lo"])
                tgt_fa_hi = float(mask_proj["free_hi"])
                # Use mask height bounds directly (p5/p95 of projected mask pixels)
                tgt_y_lo  = float(np.clip(mask_proj["height_lo"], 0.0, ceil))
                tgt_y_hi  = float(np.clip(mask_proj["height_hi"], 0.0, ceil))
            else:
                tgt_fa_lo = frm_fa_lo
                tgt_fa_hi = frm_fa_hi
                tgt_y_lo  = float(np.clip(center_y - object_h / 2.0, 0.0, ceil))
                tgt_y_hi  = float(np.clip(center_y + object_h / 2.0, 0.0, ceil))

            # Doors always reach the floor. mask_proj["height_lo"] reflects the
            # lowest VISIBLE mask pixel, which understates the true floor contact
            # whenever furniture occludes the door's base (the common case) —
            # ground it unconditionally instead of trusting the mask bottom.
            if obj_type == "door":
                tgt_y_lo = 0.0

            # Window-only: scale the snap target up to compensate for
            # occluded silhouettes (curtains, plants, etc. in the reference
            # photo shrink the visible window mask).  The top edge is
            # ANCHORED at the silhouette top — only the bottom and sides
            # extend outward — because the top of a window is rarely the
            # occluded edge (curtains drape down, plants sit below sill,
            # furniture covers the lower portion).  Doors are excluded.
            if obj_type == "window" and _WINDOW_SNAP_SCALE != 1.0:
                fa_room_max = float(W_m if wall in ("back", "front") else D_m)
                # Horizontal: scale around centre
                ctr_h  = (tgt_fa_lo + tgt_fa_hi) / 2.0
                half_h = (tgt_fa_hi - tgt_fa_lo) / 2.0 * _WINDOW_SNAP_SCALE
                tgt_fa_lo = float(np.clip(ctr_h - half_h, 0.0, fa_room_max))
                tgt_fa_hi = float(np.clip(ctr_h + half_h, 0.0, fa_room_max))
                # Vertical: keep tgt_y_hi (silhouette top) fixed; extend only downward
                silhouette_v = tgt_y_hi - tgt_y_lo
                tgt_y_lo = float(np.clip(tgt_y_hi - silhouette_v * _WINDOW_SNAP_SCALE,
                                          0.0, ceil))
                print(f"  [corner] window scale-up ×{_WINDOW_SNAP_SCALE:.2f} "
                      f"(top-anchored): h=[{tgt_fa_lo:.3f},{tgt_fa_hi:.3f}]m  "
                      f"v=[{tgt_y_lo:.3f},{tgt_y_hi:.3f}]m")

            for axis, cur_lo, cur_hi, tgt_lo, tgt_hi, label in [
                (fa, frm_fa_lo, frm_fa_hi, tgt_fa_lo, tgt_fa_hi, "h"),
                (1,  frm_y_lo,  frm_y_hi,  tgt_y_lo,  tgt_y_hi,  "v"),
            ]:
                cur_w = max(cur_hi - cur_lo, 1e-6)
                tgt_w = max(tgt_hi - tgt_lo, 0.01)
                verts[:, axis] = (
                    (tgt_lo + tgt_hi) / 2.0
                    + (verts[:, axis] - (cur_lo + cur_hi) / 2.0) * (tgt_w / cur_w)
                )
                print(f"  [corner] window/door {label}-snap (mask): "
                      f"[{cur_lo:.3f},{cur_hi:.3f}] → [{tgt_lo:.3f},{tgt_hi:.3f}]m")

        # ── Top-pin: slide object up so its top vertex aligns with top_wy ────────
        # The GLB centroid is rarely at the geometric bbox centre (wider bottoms,
        # off-centre handles, asymmetric mass).  Centering at center_y therefore
        # leaves the object floating with a gap above it.  Instead, pin the TOP
        # of the actual mesh to top_wy — the object hangs down from its attachment
        # point (rod, nail, hook) rather than floating at its centroid.
        # Windows/doors are already fully snapped by their own block above.
        if obj_type not in ("window", "door") and height_result is not None:
            mesh_top   = float(verts[:, 1].max())
            shift_y    = top_wy - mesh_top
            if abs(shift_y) > 1e-4:
                verts[:, 1] += shift_y
                world_pt = world_pt.copy()
                world_pt[1] += shift_y
                print(f"  [corner] top-pin: mesh_top={mesh_top:.3f}m → top_wy={top_wy:.3f}m "
                      f"(shift={shift_y:+.3f}m)")

        # ── Curtain: non-uniform free-axis scale to match mask width ─────────────
        # Curtains don't need uniform aspect-ratio preservation — they're flat fabric
        # panels whose width on the wall comes directly from the mask, not GLB proportions.
        # Scale the free (horizontal wall) axis so the curtain fills exactly the
        # mask-projected horizontal extent.  Y is already handled by top-pin above
        # and curtain bottom-alignment after the loop.
        if obj_type == "curtain" and mask_proj is not None:
            fa       = 0 if wall in ("back", "front") else 2
            tgt_lo   = float(mask_proj["free_lo"])
            tgt_hi   = float(mask_proj["free_hi"])
            cur_lo   = float(verts[:, fa].min())
            cur_hi   = float(verts[:, fa].max())
            cur_w    = max(cur_hi - cur_lo, 1e-6)
            tgt_w    = max(tgt_hi - tgt_lo, 0.01)
            tgt_ctr  = (tgt_lo + tgt_hi) / 2.0
            cur_ctr  = (cur_lo + cur_hi) / 2.0
            verts[:, fa] = tgt_ctr + (verts[:, fa] - cur_ctr) * (tgt_w / cur_w)
            # Update world_pt free axis to match
            world_pt = world_pt.copy()
            world_pt[fa] = tgt_ctr
            print(f"  [corner] curtain w-snap: [{cur_lo:.3f},{cur_hi:.3f}] → [{tgt_lo:.3f},{tgt_hi:.3f}]m")

        # ── Wall-lamp facing-direction check ──────────────────────────────────
        # Same logic as run(): render the placed lamp from scene camera, ask VLM
        # if it faces the same horizontal direction as the reference. If not,
        # mirror vertices along wall-tangential axis.
        print(f"  [corner] obj_type={obj_type!r} — checking lamp facing")
        if obj_type == "light":
            ref_for_facing = canvas_path_c or image_path
            needs_flip = _vlm_lamp_facing_check(
                ref_for_facing, [x1, y1, x2, y2],
                verts, faces, vc, _project, W_px, H_px,
                wall=wall,
                is_canvas=(canvas_path_c is not None),
            )
            if needs_flip:
                flipped = verts.copy()
                if wall in ("back", "front"):
                    flipped[:, 0] = 2.0 * world_pt[0] - verts[:, 0]
                else:
                    flipped[:, 2] = 2.0 * world_pt[2] - verts[:, 2]
                verts = flipped
                orient["lamp_facing_flipped"] = True
                orient["lamp_facing_wall"]     = wall
                orient["lamp_facing_world_pt"] = world_pt.tolist()
                flip_y = orient["flip_y"]
                flip_x = orient["flip_x"]
                flip_z = orient.get("flip_z", True)
                print(f"  [lamp_facing] mirroring along wall-tangential axis "
                      f"(wall={wall}  pivot={world_pt.round(3).tolist()})")

        # ── Back-face tilt correction ─────────────────────────────────────────
        _flat_types = ("window", "door", "curtain", "art", "frame", "painting")
        _tilt_cap   = None if obj_type in _flat_types else 3.0
        verts, tilt_theta_deg = _apply_tilt_correction(
            verts, wall, obj_type, W_m, D_m, max_theta_deg=_tilt_cap)

        # ── Store raw verts/faces for deferred OBJ write ──────────────────────
        placed_raw.append((f"# segment {idx:02d} ({obj_type})\n", verts, faces))

        # Record placement — sync size_m from actual verts so JSON matches OBJ
        if obj_type == "curtain" and mask_proj is not None:
            fa = 0 if wall in ("back", "front") else 2
            size["width_m"] = float(verts[:, fa].max() - verts[:, fa].min())
        if obj_type in ("window", "door"):
            fa = 0 if wall in ("back", "front") else 2
            size["width_m"]  = float(verts[:, fa].max() - verts[:, fa].min())
            size["height_m"] = float(verts[:, 1].max()  - verts[:, 1].min())
            world_pt = world_pt.copy()
            world_pt[1] = (verts[:, 1].max() + verts[:, 1].min()) / 2.0
            world_pt[fa] = (verts[:, fa].max() + verts[:, fa].min()) / 2.0
        centre_px = _project(world_pt)
        placement = {
            "segment_index": idx,
            "type":          obj_type,
            "wall":          wall,
            "world_pt":      world_pt.tolist(),
            "projected_px":  [centre_px[0], centre_px[1]] if centre_px else None,
            "bbox_px":       [x1, y1, x2, y2],
            "size_m":        {k: size[k] for k in ("width_m", "height_m", "depth_m")},
            "glb_file":      str(Path(seg["glb_file"]).name),
            "orientation":   orient,
            "glb_front_deg": glb_front_deg,
            "tilt_theta_deg": tilt_theta_deg,
        }
        placements.append(placement)

        # Update wall state using actual placed vertex bounds (more accurate than
        # size estimates, which may differ after frame-snap / tilt correction).
        _fa = 0 if wall in ("back", "front") else 2
        wall_state.append({
            "seg_idx":  idx,
            "wall":     wall,
            "free_min": float(verts[:, _fa].min()),
            "free_max": float(verts[:, _fa].max()),
            "z_min":    float(verts[:, 1].min()),
            "z_max":    float(verts[:, 1].max()),
        })
        _save_wall_state(wall_state, state_path)
        print(f"  [corner] placed: {len(verts)} verts  state saved → {state_path.name}")

    # ── Curtain bottom alignment ──────────────────────────────────────────────
    # Compute the lowest curtain-bottom per wall from placed_raw verts.
    # Store the target y_min back into placements["size_m"] so the OBJ write
    # and the render overlay both use the same geometry.
    # ── Curtain shared top alignment ─────────────────────────────────────────
    # All curtains on the same wall should hang from the same rod height (the
    # maximum top-y across curtains on that wall).  Shift any shorter curtain
    # up so its top matches the tallest curtain on the same wall.
    # This must run BEFORE bottom-alignment so the bottom-stretch uses the
    # already-top-aligned mesh.
    curtain_wall_ymax: dict[str, float] = {}
    for pi, (_, p_verts, _f) in enumerate(placed_raw):
        if placements[pi]["type"] == "curtain":
            w_key = placements[pi]["wall"]
            ymax  = float(p_verts[:, 1].max())
            if w_key not in curtain_wall_ymax or ymax > curtain_wall_ymax[w_key]:
                curtain_wall_ymax[w_key] = ymax

    for pi, (comment, p_verts, p_faces) in enumerate(placed_raw):
        if placements[pi]["type"] != "curtain":
            continue
        w_key      = placements[pi]["wall"]
        target_top = curtain_wall_ymax.get(w_key)
        if target_top is None:
            continue
        cur_top = float(p_verts[:, 1].max())
        shift   = target_top - cur_top
        if abs(shift) < 1e-4:
            continue
        shifted = p_verts.copy()
        shifted[:, 1] += shift
        placed_raw[pi] = (comment, shifted, p_faces)
        placements[pi]["world_pt"][1] += shift
        placements[pi]["final_top_y"]  = float(target_top)
        print(f"  [corner] curtain {pi:02d} top {cur_top:.3f}m → {target_top:.3f}m "
              f"(wall={w_key}  shift={shift:+.3f}m)")

    # Curtain bottom alignment is handled at placement time now: object_h
    # is set to top_wy (rod-to-floor span) for every curtain so the OBB
    # scale produces a mesh that already spans bbox=[0, top_wy] in world.
    # No post-stretch is needed — pass 1 (placed_raw) and pass 2 (overlay)
    # both derive their verts from the same uniform OBB scale, keeping
    # the OBJ file and the rendered PNG visually identical.

    # ── Rebuild wall holes to match placed window/door footprints ────────────
    # After all windows are transformed, we know their exact world-space extents.
    # Rebuild walls_with_openings.obj so the hole is sized to the GLB frame,
    # not the other way around.
    if window_footprints and placing_openings:
        try:
            from floorplan.openings.wall_openings import build_walls_with_openings_obj

            W_r   = float(room.get("floor_width_m",    W_m))
            D_r   = float(room.get("floor_depth_m",    D_m))
            ceil_ = float(room.get("ceiling_height_m", ceil))

            rebuilt_openings: list[dict] = []
            for fp in window_footprints:
                fp_wall  = fp["wall"]
                fp_lo    = float(fp["free_min"])
                fp_hi    = float(fp["free_max"])
                fp_ymin  = float(fp["y_min"])
                fp_ymax  = float(fp["y_max"])
                fp_w     = max(0.01, fp_hi - fp_lo)
                fp_h     = max(0.01, fp_ymax - fp_ymin)
                # Convert free-axis extents to offset_from_left_m (wall_openings convention):
                #   back  wall: left = lower X  → offset = free_lo
                #   front wall: left = higher X → offset = W - free_hi
                #   left  wall: left = near Z   → offset = D - free_hi  (Z=D is near/left)
                #   right wall: left = far Z    → offset = free_lo      (Z=0 is far/left)
                if fp_wall == "back":
                    offset_l = fp_lo
                elif fp_wall == "front":
                    offset_l = W_r - fp_hi
                elif fp_wall == "left":
                    offset_l = D_r - fp_hi
                else:  # right
                    offset_l = fp_lo
                rebuilt_openings.append({
                    "wall":             fp_wall,
                    "type":             "window",
                    "offset_from_left_m": round(max(0.0, offset_l), 4),
                    "width_m":          round(fp_w,  4),
                    "sill_height_m":    round(max(0.0, fp_ymin), 4),
                    "height_m":         round(fp_h,  4),
                })
                print(f"  [corner] hole rebuild {fp_wall}: offset={offset_l:.3f}m "
                      f"w={fp_w:.3f}m sill={fp_ymin:.3f}m h={fp_h:.3f}m")

            # Load analysis for room + visible walls
            analysis_r = {}
            if anal_path.exists():
                analysis_r = json.loads(anal_path.read_text())
            build_walls_with_openings_obj(
                {**analysis_r, "room": room},
                rebuilt_openings,
                plain_obj.parent if plain_obj.exists() else fallback_obj.parent,
                plain_obj.parent if plain_obj.exists() else fallback_obj.parent,
            )
            print("[corner] walls_with_openings.obj rebuilt to match window footprints")
            # Re-read the base mesh lines with the corrected holes
            orig_lines = Path(obj_in_path).read_text().splitlines(keepends=True)
        except Exception as e:
            print(f"[corner] Warning: hole rebuild failed ({e}) — using original mesh")

    # ── Write combined OBJ ────────────────────────────────────────────────────
    # Use placed_raw directly — these verts are fully processed (OBB de-tilt,
    # flip, scale, frame-snap for windows, tilt correction) and match the
    # window_footprints used for the hole rebuild.  Re-running _transform_glb_to_wall
    # here would skip frame-snap and could apply the wrong OBB swap.
    base_n_verts = sum(1 for ln in orig_lines if ln.strip().startswith("v "))
    with open(obj_out_path, "w") as f:
        f.writelines(orig_lines)
        f.write("\n# ── Wall-mounted object placements (corner-based) ─\n")
        running_n = base_n_verts
        _PLANE_TYPES_OBJ = {"window", "door", "art", "painting", "frame"}
        for (comment, w_verts, w_faces), p in zip(placed_raw, placements):
            seg_idx    = p["segment_index"]
            obj_type_w = str(p.get("type", ""))

            # Flat-plane GLBs have front+back face layers (for OBB detection).
            # Keep only the wall-side layer to avoid a visible double mesh.
            if obj_type_w in _PLANE_TYPES_OBJ and len(w_verts) % 2 == 0:
                half = len(w_verts) // 2
                wall_w = p.get("wall", "")
                # Wall-normal axis: Z for back/front walls, X for left/right
                na = 2 if wall_w in ("back", "front") else 0
                front_center = float(w_verts[:half, na].mean())
                back_center  = float(w_verts[half:, na].mean())
                # Wall surface is at the boundary (Z=0 for back, X=0 for left, etc.)
                # Keep whichever half is closer to the wall
                if abs(front_center) <= abs(back_center) if wall_w in ("back", "left") else front_center >= back_center:
                    keep_verts = w_verts[:half]
                    keep_faces = np.array([f for f in w_faces if f.max() < half])
                else:
                    keep_verts = w_verts[half:]
                    keep_faces = np.array([f - half for f in w_faces if f.min() >= half])
                w_verts = keep_verts
                w_faces = keep_faces

            f.write(f"# segment {seg_idx:02d} ({obj_type_w})\n")
            for v in w_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in w_faces:
                f.write(f"f {face[0]+running_n+1} {face[1]+running_n+1} {face[2]+running_n+1}\n")
            running_n += len(w_verts)
    print(f"\n[corner] Scene mesh → {obj_out_path}")

    # ── Merge carryover placements and write OBJ entries for them too ──────────
    if carryover_placements:
        with open(obj_out_path, "a") as f:
            f.write("\n# ── Carried-over placements (non-targeted types) ─\n")
            for p in carryover_placements:
                seg_idx    = p["segment_index"]
                obj_type_w = str(p.get("type", ""))
                glb_path_w = obj_dir / p["glb_file"]
                wall_w     = p["wall"]
                world_pt_w = np.array(p["world_pt"], dtype=float)
                size_w     = p["size_m"]
                orient_w   = p.get("orientation", {"flip_y": wall_w in ("left","right"), "flip_x": wall_w in ("right","front")})
                _gfd_w     = p.get("glb_front_deg")
                gfd_w      = int(_gfd_w) if _gfd_w is not None else None
                res = _transform_glb_to_wall(
                    glb_path_w, wall_w, world_pt_w, size_w, room,
                    cam_pos=pos,
                    flip_y=orient_w.get("flip_y", True),
                    flip_x=orient_w.get("flip_x", False),
                    flip_z=orient_w.get("flip_z", True),
                    glb_front_deg=gfd_w,
                    embed=False,
                    outward=False,   # window/door are planes flush on wall (no outward protrusion)
                    detilt=(obj_type_w not in ("window", "door", "curtain",
                                               "mirror", "art", "frame", "painting")),
                    force_floor=(obj_type_w == "door"),
                )
                if res is None:
                    continue
                w_verts, w_faces, _ = res
                saved_theta_w = p.get("tilt_theta_deg")
                w_verts, _ = _apply_tilt_correction(
                    w_verts, wall_w, obj_type_w, W_m, D_m,
                    forced_theta_deg=saved_theta_w,
                )
                f.write(f"# segment {seg_idx:02d} ({obj_type_w})\n")
                for v in w_verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in w_faces:
                    f.write(f"f {face[0]+running_n+1} {face[1]+running_n+1} {face[2]+running_n+1}\n")
                running_n += len(w_verts)
        placements = placements + carryover_placements

    # ── Save placements JSON ──────────────────────────────────────────────────
    with open(placements_path, "w") as f:
        json.dump(placements, f, indent=2)
    print(f"[corner] Placements → {placements_path}")

    # ── Render ────────────────────────────────────────────────────────────────
    render_path = _render_objects(
        str(obj_out_path), cam_path, place_dir, out_dir, placements, room,
        proj_tex_image_path=proj_tex_image_path)

    # ── Post-render mirror check (same logic as in `run`) ───────────────────
    _art_segs = [(_i, _p) for _i, _p in enumerate(placements)
                 if _p.get("type") in ("art", "painting", "frame")]
    if _art_segs and render_path and Path(render_path).exists():
        _any_flipped = False
        for _i, _p in _art_segs:
            # Placement dict uses bbox_px / segment_index / glb_file.
            _bbox = _p.get("bbox_px") or _p.get("bbox")
            _seg_idx = _p.get("segment_index", _p.get("seg_idx", _i))
            # Derive inpaint PNG path from the glb_file name (they share
            # the inpaint_<idx>_<type> stem).
            _glb_name = _p.get("glb_file", "")
            _inpaint_path = None
            if _glb_name.endswith(".glb"):
                _candidate = (Path(out_dir) / "wall_mounted" / "inpainted"
                              / (_glb_name[:-4] + ".png"))
                if _candidate.exists():
                    _inpaint_path = _candidate
            try:
                if _vlm_check_placed_painting_mirror(
                        ref_image_path=image_path,
                        render_path=render_path,
                        bbox=_bbox,
                        obj_type=_p.get("type", "art"),
                        seg_idx=_seg_idx,
                        inpaint_path=_inpaint_path):
                    # Mirror the GLB on disk (flat plane: flipping local-X
                    # vertices left↔right effectively mirrors the displayed
                    # texture, since UVs are unchanged).  Mirroring the GLB
                    # itself avoids relying on the orient[flip_x] code path
                    # downstream, mirroring the painting permanently in the
                    # GLB the way decorations' mirror_check fixes the the 3D generator
                    # left-right flip on a computer monitor.
                    _glb_full = (Path(out_dir) / "wall_mounted" / "objects"
                                 / _glb_name)
                    if _glb_full.exists() and _flip_glb_horizontally(_glb_full):
                        print(f"  [post-mirror] seg {_seg_idx}: mirrored "
                              f"{_glb_name} in place (left↔right)")
                        _any_flipped = True
                    else:
                        print(f"  [post-mirror] seg {_seg_idx}: GLB not "
                              f"flipped ({_glb_full})")
            except Exception as _pe:
                print(f"  [post-mirror] seg {_seg_idx}: skipped ({_pe})")
        if _any_flipped:
            with open(placements_path, "w") as f:
                json.dump(placements, f, indent=2)
            print(f"[post-mirror] re-rendering with flip_x corrections …")
            _render_objects(
                str(obj_out_path), cam_path, place_dir, out_dir, placements, room,
                proj_tex_image_path=proj_tex_image_path)


# ─────────────────────────────────────────────────────────────────────────────
# Placement animation
# ─────────────────────────────────────────────────────────────────────────────

def make_placement_animation(
    out_dir:          Path,
    placements:       list[dict],
    room:             dict,
    cam_path:         Path,
    gif_width:        int  = 960,
    frame_duration_ms: int = 1000,
    out_name:         str  = "placement_animation.gif",
) -> str | None:
    """
    Build a GIF showing wall-mounted objects placed one at a time.

    Each object contributes TWO frames so tilt-correction is visible:
      Frame A  : object placed at raw position (pre-tilt)
      Frame B  : object after tilt correction / curtain pin (final position)

    Objects are added incrementally — each new frame builds on the previous
    one (O(N) renders total, not O(N²)).

    Returns the path to the saved GIF, or None on failure.
    """
    import os, tempfile
    from PIL import Image as _PIL

    place_dir = out_dir / "wall_mounted" / "placements"

    # ── Same base-render selection as _render_objects ────────────────────────
    placing_openings = any(p.get("type") in ("window", "door") for p in placements)
    openings_render  = place_dir / "render_openings_filled.png"

    if not placing_openings and openings_render.exists():
        base_png = str(openings_render)
    else:
        base_png = str(place_dir / "_objects_render_tmp.png")
        if not Path(base_png).exists():
            from floorplan.wall_line.render_room import render_room
            walls_mesh = place_dir / "walls_with_windows.obj"
            if not walls_mesh.exists():
                walls_mesh = out_dir / "walls.obj"
            mesh_path = str(walls_mesh) if walls_mesh.exists() else None
            if mesh_path is None:
                print("[animate] No wall mesh found — cannot render base")
                return None
            render_room(mesh_path=mesh_path, camera_json_path=str(cam_path),
                        out_path=base_png, texture_dir=str(out_dir))

    if not Path(base_png).exists():
        print(f"[animate] Base render not found: {base_png}")
        return None

    base_arr = np.array(_PIL.open(base_png).convert("RGB"), dtype=np.uint8)
    H, W = base_arr.shape[:2]
    gif_h = int(H * gif_width / W)

    # ── Window-first render order (same as _overlay_glbs) ────────────────────
    _PLANE_TYPES = {"window", "door"}
    ordered = (
        [p for p in placements if p.get("type") in _PLANE_TYPES] +
        [p for p in placements if p.get("type") not in _PLANE_TYPES]
    )

    print(f"[animate] Building placement GIF: {len(ordered)} objects, incremental …")
    frames: list[_PIL.Image] = []

    def _arr_to_frame(arr: np.ndarray) -> _PIL.Image:
        return _PIL.fromarray(arr).resize((gif_width, gif_h), _PIL.LANCZOS)

    def _tmp_png() -> str:
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        return f.name

    # Frame 0: empty base
    frames.append(_arr_to_frame(base_arr))
    print(f"[animate] Frame 0: base render")

    # prev_png tracks the accumulated render so each object is overlaid ONCE
    prev_png = base_png

    for i, p in enumerate(ordered):
        ptype = p.get("type", f"obj{i}")
        seg   = p.get("segment_index", i)

        # ── Frame A: object placed WITHOUT tilt correction ────────────────────
        # Temporarily zero out tilt so _overlay_glbs renders the raw position
        p_notilt = dict(p, tilt_theta_deg=0.0)
        tmp_a = _tmp_png()
        _overlay_glbs(prev_png, tmp_a, [p_notilt], room, cam_path, out_dir)
        frames.append(_arr_to_frame(np.array(_PIL.open(tmp_a).convert("RGB"), dtype=np.uint8)))
        print(f"[animate] Frame {len(frames)-1}: +{ptype} (seg {seg}) [pre-tilt]")

        # ── Frame B: object with full tilt + pin (final position) ─────────────
        tmp_b = _tmp_png()
        _overlay_glbs(prev_png, tmp_b, [p], room, cam_path, out_dir)
        frames.append(_arr_to_frame(np.array(_PIL.open(tmp_b).convert("RGB"), dtype=np.uint8)))
        print(f"[animate] Frame {len(frames)-1}: +{ptype} (seg {seg}) [corrected]")

        os.unlink(tmp_a)
        # keep tmp_b as the new accumulated base for the next object
        prev_png = tmp_b

    # Clean up the last accumulated temp file (it's not base_png)
    if prev_png != base_png and Path(prev_png).exists():
        os.unlink(prev_png)

    if len(frames) < 2:
        print("[animate] Not enough frames — skipping GIF")
        return None

    gif_path = place_dir / out_name
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    print(f"[animate] Saved → {gif_path}  ({len(frames)} frames, {frame_duration_ms}ms/frame)")
    return str(gif_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Place wall-mounted objects (art, mirrors, etc.) in the scene."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output directory (contains wall_mounted/, openings/, …)")
    ap.add_argument("--image", required=True,
                    help="Original room photograph (for VLM size estimation)")
    ap.add_argument("--types", default=None,
                    help="Comma-separated object types to place (default: all). "
                         "E.g. art,curtain,light,window")
    ap.add_argument("--render-only", action="store_true",
                    help="Re-render existing walls_with_objects.obj without re-placing")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only place the first N objects (for debugging)")
    ap.add_argument("--center-only", action="store_true",
                    help="Place objects at room centre (no wall snap) for visual inspection")
    ap.add_argument("--corner", action="store_true",
                    help="Use corner-based coordinate system pipeline (new approach)")
    ap.add_argument("--refine-iters", type=int, default=5,
                    help="Iterative free-axis refinement steps (--corner mode, default: 5)")
    ap.add_argument("--animate", action="store_true",
                    help="Save placement_animation.gif showing objects added one at a time")
    ap.add_argument("--animate-width", type=int, default=960,
                    help="GIF frame width in pixels (default 960)")
    ap.add_argument("--animate-duration", type=int, default=1000,
                    help="Milliseconds per frame in the GIF (default 1000)")
    args = ap.parse_args()

    out_dir      = Path(args.output_dir)
    place_dir    = out_dir / "wall_mounted" / "placements"
    target_types = [t.strip() for t in args.types.split(",")] if args.types else None

    # ── Shared camera + room loading ─────────────────────────────────────────
    def _load_cam_and_room():
        for _cp in [out_dir / "camera_vggt.json",
                    out_dir / "camera.json",
                    out_dir / "openings" / "camera.json"]:
            if _cp.exists():
                _cam = _cp
                break
        else:
            raise FileNotFoundError(f"No camera_vggt.json / camera.json in {out_dir}")
        _analysis = {}
        if (out_dir / "floorplan_analysis.json").exists():
            _analysis = json.loads((out_dir / "floorplan_analysis.json").read_text())
        _room = dict(_analysis.get("room", {}))
        _dims = _room_dims_from_obj(out_dir / "openings" / "walls_with_openings.obj")
        if _dims:
            _room.update(_dims)
        return _cam, _room

    if args.render_only:
        cam_path, room = _load_cam_and_room()
        obj_path = str(place_dir / "walls_with_objects.obj")
        placements_path = place_dir / "object_placements.json"
        placements = json.loads(placements_path.read_text()) if placements_path.exists() else []
        # Composite onto the re-projected wall texture (wood paneling etc.) when it
        # exists — matches the full-placement path. Otherwise --render-only falls
        # back to the raw OBJ render with the old/plain wall texture.
        _ref_tex = Path(out_dir) / "render_ref_texture.png"
        _render_objects(obj_path, cam_path, place_dir, out_dir, placements, room,
                        proj_tex_image_path=str(_ref_tex) if _ref_tex.exists() else None)
        if args.animate and placements:
            make_placement_animation(
                out_dir, placements, room, cam_path,
                gif_width=args.animate_width,
                frame_duration_ms=args.animate_duration,
            )
    elif args.corner:
        run_corner_based(
            output_dir=args.output_dir,
            image_path=args.image,
            target_types=target_types,
            limit=args.limit,
            refine_iters=args.refine_iters,
        )
        if args.animate:
            cam_path, room = _load_cam_and_room()
            placements_path = place_dir / "object_placements.json"
            placements = json.loads(placements_path.read_text()) if placements_path.exists() else []
            if placements:
                make_placement_animation(
                    out_dir, placements, room, cam_path,
                    gif_width=args.animate_width,
                    frame_duration_ms=args.animate_duration,
                )
    else:
        run(output_dir=args.output_dir,
            image_path=args.image,
            target_types=target_types,
            limit=args.limit,
            center_only=args.center_only)
        if args.animate:
            cam_path, room = _load_cam_and_room()
            placements_path = place_dir / "object_placements.json"
            placements = json.loads(placements_path.read_text()) if placements_path.exists() else []
            if placements:
                make_placement_animation(
                    out_dir, placements, room, cam_path,
                    gif_width=args.animate_width,
                    frame_duration_ms=args.animate_duration,
                )


if __name__ == "__main__":
    main()
