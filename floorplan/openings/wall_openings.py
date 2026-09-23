"""
wall_openings.py — detect door/window openings with VLM and punch holes in the wall mesh.

Pipeline:
  1. Load reference image + existing floorplan_analysis.json (room dims + features)
  2. Call VLM with a focused opening-detection prompt to get accurate sill heights
     (or fall back to features already in floorplan_analysis.json)
  3. For each wall with openings: tessellate the wall quad into a grid of sub-quads,
     omitting the cells that correspond to each opening
  4. Write walls_with_openings.obj (reuses the same .mtl as walls.obj)

Usage:
    python -m floorplan.openings.wall_openings --output-dir outputs/20260325_XXXXXX --image data/indoor_images/office5.jpg
    python -m floorplan.openings.wall_openings --output-dir outputs/20260325_XXXXXX --no-vlm

Outputs are written to outputs/20260325_XXXXXX/openings/:
    openings.json               — detected opening parameters
    walls_with_openings.obj     — room mesh with holes punched
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
from pathlib import Path

import requests

VLM_API_URL = "http://localhost:8080/v1/chat/completions"
from object_placement.vlm_backend import vlm_post as _vlm_post

# ── VLM helpers ──────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type)."""
    path = Path(image_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _parse_json_response(raw: str) -> list | dict:
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
            print(f"[wall_openings] Warning: could not parse JSON: {e}")
            return []


# ── Opening detection VLM prompts ────────────────────────────────────────────

_MEASUREMENT_GUIDE = """\
MEASUREMENT GUIDE
-----------------
offset_from_left_m  = distance from the LEFT END of the wall to the LEFT EDGE of the opening.
                      "Left end" is defined per wall as seen from INSIDE the room:
                        back  wall — left = image-left end  (lower x end)
                        right wall — left = back-right corner (far end, z=0 side)
                        left  wall — left = front-left corner (near end, z=D side, image-left edge)
                        front wall — left = back-right-ish corner (higher x end)

width_m    = opening width  (door ≈ 0.9m, window ≈ 0.8–1.5m)
height_m   = opening height (door ≈ 2.1m, window ≈ 0.9–1.2m)
sill_height_m = height of the BOTTOM edge of the opening above the floor
                  door → 0.0  (opening starts at floor)
                  window → estimate, typically 0.85–1.1m

Scale anchors:
  • Standard interior door: 0.9m wide, 2.1m tall, sill at 0.0m
  • Standard window: 1.0–1.3m wide, 1.0–1.2m tall, sill at 0.9m
  • Use the ceiling height ({{ceiling_h:.1f}}m) and any visible door/window frames to calibrate.

DOOR IDENTIFICATION — STRICT RULES
  A door MUST satisfy ALL of the following:
    ✓ You can see INTO a different space through it (another room, hallway, exterior)
    ✓ It has a visible door frame (rectangular surround) OR an open gap at floor level
    ✓ Bottom edge is at the floor (sill = 0.0 m)
  DO NOT mark as door:
    ✗ Dark/differently-painted wall panels or sections
    ✗ Wardrobes, cabinets, or any furniture (even tall ones with handles)
    ✗ Wall niches, alcoves, or recesses that do not go through the wall
    ✗ Artwork, mirrors, or decorative panels

VALIDATION RULES
  • offset_from_left_m + width_m  ≤  wall length
  • sill_height_m + height_m      ≤  ceiling_h ({{ceiling_h:.1f}}m)
  • Only include openings you are CERTAIN are structural holes in the wall
  • When in doubt, OMIT — a missed opening is better than a false one

OUTPUT — valid JSON array only, no markdown, no explanation:
[
  {{"wall": "back|left|right|front", "type": "door|window",
    "width_m": 0.0, "height_m": 0.0, "offset_from_left_m": 0.0, "sill_height_m": 0.0}},
  ...
]
Return [] if no openings are visible.
"""

_OPENING_PROMPT_TMPL = """\
OPENING DETECTION TASK
======================
Identify all visible doors and windows in this interior image.

ROOM CONTEXT (from prior analysis):
  Room dimensions: {width_m:.1f}m wide × {depth_m:.1f}m deep, ceiling {ceiling_h:.1f}m
  Visible walls: {visible_walls}

For each visible opening (door or window) output one entry in the JSON array below.

""" + _MEASUREMENT_GUIDE.replace("{{ceiling_h:.1f}}", "{ceiling_h:.1f}")

_COMPARISON_PROMPT_TMPL = """\
OPENING DETECTION — RENDER vs REFERENCE COMPARISON
===================================================
You are given TWO images (in this order):
  IMAGE 1 (REFERENCE): the real room photograph
  IMAGE 2 (RENDER):    a 3D reconstruction of the same room with SOLID WALLS (no holes yet)

Both images show the SAME camera angle and room geometry.

WALL LABELS (how to identify each wall in IMAGE 2)
---------------------------------------------------
  back  — the wall directly facing the camera (fills the centre of the image)
  left  — side wall on the LEFT side of the image
  right — side wall on the RIGHT side of the image
  front — wall behind the camera (usually not visible or at extreme edges)

YOUR TASK
---------
Find every door and window that appears in IMAGE 1 but is MISSING in IMAGE 2
(i.e. it is blocked by a solid wall in the render).

HOW TO IDENTIFY — TYPE RULES
-----------------------------
  WINDOW
    • Bright rectangular region or visible window frame / glass in IMAGE 1
      where IMAGE 2 shows solid wall.
    • Bottom edge is ABOVE the floor (sill typically 0.8–1.1 m up).
    • Does NOT extend all the way to the floor.

  DOOR
    • A structural hole through the wall — you can see INTO another room, hallway,
      or exterior space through it (even if partially closed, a gap or frame is visible).
    • Has a physical door frame (rectangular surround, typically wood or metal).
    • Bottom edge is AT the floor level (sill = 0 m).
    • Extends from floor to near-ceiling height (typically 2.0–2.2 m tall).
    • The area seen through it looks like a DIFFERENT SPACE (different wall, corridor, outside).

  NOT A DOOR — do NOT mark these:
    • Dark or differently-coloured wall sections / wall treatments / painted panels
    • Wardrobes, cabinets, closets (furniture — even with tall panels or handles)
    • Wall niches, alcoves, or recesses that do NOT penetrate through the wall
    • Large artwork, posters, mirrors, or framed pieces on the wall
    • Tall windows that reach the floor (classify as window with sill=0 instead)
    • Any feature where you cannot see through to a different space

  IGNORE: furniture, art, shelves, decorations, reflections.
  RULE: When in doubt, do NOT include. Only mark openings you are CERTAIN are
        structural holes in the wall that lead to another space.

FOR EACH OPENING
----------------
1. Identify which WALL the opening belongs to (back / left / right / front).
2. Locate the SAME opening in IMAGE 2 (the render).  The render has solid walls,
   so you must find the matching wall surface and estimate where the hole would be.
3. Mark the pixel bounding box of the opening IN IMAGE 2 coordinates (NOT IMAGE 1).
   — Align the box to the SAME relative position on the wall as seen in IMAGE 1:
     same fraction from the left/right edges of that wall, same fraction from floor/ceiling.
   — Make sure the box stays entirely within the wall surface in IMAGE 2.
   — Do NOT copy pixel coordinates from IMAGE 1 directly; perspectives may differ slightly.

OUTPUT — valid JSON array only, no markdown, no extra text:
[
  {{"type": "door|window", "wall": "back|left|right|front",
    "pixel_rect": [left_px, top_px, right_px, bottom_px]}},
  ...
]
pixel_rect coordinates are in IMAGE 2 pixel space (0,0 = top-left corner).
Return [] if no openings are visible.
"""

_WALL_OPENING_PROMPT_TMPL = """\
OPENING DETECTION — WALL-LEVEL SEMANTIC ANALYSIS
=================================================
You are given TWO images (in this order):
  IMAGE 1 (REFERENCE): the real room photograph
  IMAGE 2 (RENDER):    a 3D render of the same room — solid walls, same camera angle

{wall_labels_context}

ROOM DIMENSIONS (use to estimate metres):
  back/front wall width  = {{floor_width:.2f}} m
  left/right wall depth  = {{floor_depth:.2f}} m
  ceiling height          = {{ceiling_h:.2f}} m

YOUR TASK
---------
For every structural opening (window or door) visible in IMAGE 1,
output ONE entry per opening.

HOW TO MEASURE — pixel-relative fractions
------------------------------------------
Each wall has a known pixel bounding box in IMAGE 2 (provided above in the wall labels).
Measure the opening's position as FRACTIONS of that wall's pixel bounding box:

  left_gap_frac  = (opening_left_px  − wall_x_min) / (wall_x_max − wall_x_min)
                   fraction of wall pixel width to the left of the opening
  width_frac     = (opening_right_px − opening_left_px) / (wall_x_max − wall_x_min)
                   fraction of wall pixel width covered by the opening
  sill_frac      = (wall_y_max − opening_bottom_px) / (wall_y_max − wall_y_min)
                   fraction of wall pixel height from the floor up to the sill
                   (floor = bottom of wall = wall_y_max;  0.0 = opening starts at floor)
  height_frac    = (opening_bottom_px − opening_top_px) / (wall_y_max − wall_y_min)
                   fraction of wall pixel height spanned by the opening

IMPORTANT: measure directly from what you SEE in IMAGE 2 as the wall edge pixels.
  • Use the wall's pixel bbox edges as rulers — the opening's pixel distances from
    those edges give you the fractions above.
  • For walls that extend beyond the image border, clip the measurement to the
    visible portion (but the code will handle the off-screen part automatically).

VERTICAL ESTIMATE (cross-check with metres)
  • Ceiling height = {{ceiling_h:.2f}} m.  Use it to sanity-check sill_frac and height_frac.
  • Typical window: sill ≈ 0.3–0.5 × ceiling height from floor; height ≈ 0.4–0.7 × ceiling height.
  • Floor-to-ceiling glass: sill_frac ≈ 0.0, height_frac ≈ 0.9–1.0.
  • Door: sill_frac = 0.0, height_frac ≈ 0.75–0.85 of ceiling height.

RULES
  • Do NOT count furniture, mirrors, artwork, or wall panels.
  • Only include openings you are CERTAIN penetrate through the wall.

OUTPUT — valid JSON array only, no markdown, no extra text:
[
  {{"wall": "back|left|right|front", "type": "window|door",
    "left_gap_frac": 0.0, "width_frac": 0.0,
    "sill_frac": 0.0, "height_frac": 0.0}},
  ...
]
Return [] if no structural openings are visible.
"""

_REFINEMENT_PROMPT_TMPL = """\
OPENING REFINEMENT — ADJUST CURRENT PLACEMENT
==============================================
You are given TWO images (in this order):
  IMAGE 1 (REFERENCE): the real room photograph
  IMAGE 2 (RENDER):    a 3D reconstruction of the same room where each door/window
                       is shown as a SKY-BLUE rectangle (the hole punched through the wall)

Both images show the SAME camera angle.

CURRENT OPENINGS IN IMAGE 2
---------------------------
{opening_context}

YOUR TASK
---------
For each sky-blue rectangle in IMAGE 2, find the MATCHING opening in IMAGE 1 and
correct the rectangle so it visually occupies the SAME position and size on that wall.

Focus on:
  • Horizontal position — is the opening centred/left/right relative to the wall edges?
  • Vertical position   — where does the sill (bottom) and soffit (top) fall on the wall?
  • Width and height    — what fraction of the wall does the opening cover?
Compare those fractions between IMAGE 1 and IMAGE 2, and move/resize the rectangle to match.

WINDOW-SIZE PRIORITY
--------------------
For windows and doors, reason from DISTANCE TO VISIBLE WALL EDGES in IMAGE 1:
  • compare the opening's left, right, top, and bottom gaps to the visible wall boundaries around it
  • place the box so those four edge distances are relatively consistent with the reference image
  • if the opening appears nearly equal-distance from multiple visible wall edges, preserve that balanced relationship even if the reconstructed wall size is imperfect
  • correct width if the blue opening occupies too little or too much of the wall width
  • correct height if the blue opening occupies too little or too much of the wall height
  • do NOT rely on raw image-pixel alignment across different wall planes or perspective views
  • use the provided wall_width_fraction / wall_height_fraction / left_gap_fraction / right_gap_fraction / top_gap_fraction / bottom_gap_fraction only as the CURRENT guess; change them if the reference image indicates a different ratio
  • do not preserve the old box size just because it is close
  • if needed, resize aggressively so the sky-blue hole matches the visible frame/opening

RULES
-----
  • Keep the same NUMBER of openings — do not add or remove any.
  • Preserve the SAME ids and SAME order as listed above.
  • The corrected pixel_rect must stay on the same wall / same local image region.
  • Judge placement and size relative to that wall plane and its visible edges, not by matching absolute screen pixels.
  • Focus on: horizontal position, vertical position, width, and height.
  • If an opening already matches well, output its current bounding box unchanged.

OUTPUT — valid JSON array only, no markdown, no extra text:
[
  {{"id": 1, "type": "door|window",
    "pixel_rect": [left_px, top_px, right_px, bottom_px],
    "rationale": "short evidence summary",
    "edge_assessment": {{
      "left_gap": "smaller|similar|larger",
      "right_gap": "smaller|similar|larger",
      "top_gap": "smaller|similar|larger",
      "bottom_gap": "smaller|similar|larger"
    }},
    "suggested_adjustment": "move_right|move_left|move_up|move_down|widen|narrow|taller|shorter|keep"
  }},
  ...
]
Use `rationale` as a brief observable summary, not hidden reasoning.
pixel_rect coordinates are in IMAGE 2 pixel space (0,0 = top-left corner).
Return [] if there are no sky-blue openings in IMAGE 2.
"""


_REFINEMENT_FRACS_PROMPT_TMPL = """\
OPENING REFINEMENT — RELATIVE FRACTION MEASUREMENT
====================================================
You are given TWO images (in this order):
  IMAGE 1 (REFERENCE): the real room photograph
  IMAGE 2 (RENDER):    a 3D reconstruction of the same room where each door/window
                       is shown as a SKY-BLUE rectangle

Both images share the EXACT SAME camera viewpoint and room geometry.

CURRENT OPENINGS
----------------
{opening_context}

For each opening the context above shows:
  • wall_pixel_rect   — the pixel boundary of the WALL that contains this opening
                        (left_px, top_px, right_px, bottom_px in IMAGE 2 coordinates;
                         same coordinates apply to IMAGE 1 since camera is identical)
  • current fractions — left_gap_fraction, width_fraction, sill_fraction, height_fraction
                        describing the CURRENT (possibly incorrect) placement

YOUR TASK
---------
For each sky-blue rectangle in IMAGE 2, find the SAME physical opening in IMAGE 1.
Using the wall_pixel_rect as the wall boundary in IMAGE 1, measure how the opening
sits RELATIVE TO ITS WALL:

  left_gap_frac  = (opening_left_px  − wall_left_px)  / (wall_right_px − wall_left_px)
  width_frac     = (opening_right_px − opening_left_px) / (wall_right_px − wall_left_px)
  sill_frac      = (wall_bottom_px   − opening_bottom_px) / (wall_bottom_px − wall_top_px)
  height_frac    = (opening_bottom_px − opening_top_px)  / (wall_bottom_px − wall_top_px)

IMPORTANT GUIDELINES
--------------------
  • sill_frac is measured from the FLOOR UP, so a window that nearly touches the floor
    has sill_frac ≈ 0.0; a high window has larger sill_frac.
  • A window/door that nearly fills the full wall height has height_frac ≈ 0.9–1.0.
  • If an opening already matches IMAGE 1 well, keep the current fractions.
  • Do NOT apply perspective distortion corrections — measure apparent fractions in the
    image, not corrected 3D dimensions.
  • Keep the same NUMBER, ORDER, and id values as listed above.

OUTPUT — valid JSON array only, no markdown, no extra text:
[
  {{"id": 1,
    "left_gap_frac": 0.10,
    "width_frac": 0.60,
    "sill_frac": 0.05,
    "height_frac": 0.88,
    "rationale": "window nearly fills wall height, small floor gap, roughly centered"
  }},
  ...
]
"""


# ── Pixel → wall projection ───────────────────────────────────────────────────

def _build_ray(px: float, py: float, cam: dict):
    """
    Return the unnormalised ray direction (d_x, d_y, d_z) for a given image
    pixel using the architectural (two-point) projection identical to render_room.
    The ray satisfies  ray @ fwd_h = 1.
    Also returns (pos, fwd_h, tilt_tan) needed for plane intersections.
    Returns None if the camera matrix is degenerate.
    """
    import numpy as np

    pos      = np.array(cam["position_m"], dtype=float)
    look_at  = np.array(cam["look_at_m"],  dtype=float)
    hfov_deg = float(cam["hfov_deg"])
    img_W    = int(cam["width_px"])
    img_H    = int(cam["height_px"])

    fwd = look_at - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0., 1., 0.]))
    right /= np.linalg.norm(right)

    fwd_h = np.array([fwd[0], 0.0, fwd[2]])
    fwd_h_n = float(np.linalg.norm(fwd_h))
    if fwd_h_n < 1e-9:
        return None
    fwd_h    /= fwd_h_n
    tilt_tan  = float(fwd[1]) / fwd_h_n

    fx = img_W / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    cx, cy = img_W / 2.0, img_H / 2.0

    # Invert architectural projection (ray @ fwd_h = 1):
    #   right[0]*d_x + right[2]*d_z = (px-cx)/fx
    #   fwd_h[0]*d_x + fwd_h[2]*d_z = 1
    screen_xc = (px - cx) / fx
    d_y       = (cy - py) / fx + tilt_tan

    r, f = right, fwd_h
    det = float(r[0] * f[2] - r[2] * f[0])
    if abs(det) < 1e-9:
        return None

    d_x = float((screen_xc * f[2] - r[2]) / det)
    d_z = float((r[0] - screen_xc * f[0]) / det)
    return pos, (d_x, float(d_y), d_z), fwd_h, tilt_tan


def _pixel_to_nearest_wall(
    px: float, py: float, cam: dict,
    W: float, D: float, ceiling_h: float,
    visible_orients: set[str] | None = None,
) -> tuple[str, float, float] | None:
    """
    Ray-cast from pixel (px, py) through the architectural camera and return
    (orient, u, y) for the NEAREST wall plane that the ray hits within its
    physical extent  [0, wall_length] × [0, ceiling_h].

    orient  : "back" | "left" | "right" | "front"
    u       : offset from the wall's left end (matching offset_from_left_m convention)
    y       : height above floor

    visible_orients: if given, only those walls are candidates.
    """
    res = _build_ray(px, py, cam)
    if res is None:
        return None
    pos, ray, _fwd_h, _tilt = res
    px0, py0, pz0 = float(pos[0]), float(pos[1]), float(pos[2])
    d_x, d_y, d_z = ray

    candidates: list[tuple[float, str, float, float]] = []  # (t, orient, u, y)

    # Back wall  z = 0
    if abs(d_z) > 1e-9:
        t = -pz0 / d_z
        if t > 1e-4:
            hx = px0 + t * d_x
            hy = py0 + t * d_y
            if 0 <= hx <= W and 0 <= hy <= ceiling_h:
                candidates.append((t, "back", hx, hy))

    # Front wall  z = D
    if abs(d_z) > 1e-9:
        t = (D - pz0) / d_z
        if t > 1e-4:
            hx = px0 + t * d_x
            hy = py0 + t * d_y
            if 0 <= hx <= W and 0 <= hy <= ceiling_h:
                u = W - hx   # left-end = x=W for front wall
                candidates.append((t, "front", u, hy))

    # Left wall  x = 0
    if abs(d_x) > 1e-9:
        t = -px0 / d_x
        if t > 1e-4:
            hz = pz0 + t * d_z
            hy = py0 + t * d_y
            if 0 <= hz <= D and 0 <= hy <= ceiling_h:
                u = D - hz   # left-end = z=D (front corner), increases toward back
                candidates.append((t, "left", u, hy))

    # Right wall  x = W
    if abs(d_x) > 1e-9:
        t = (W - px0) / d_x
        if t > 1e-4:
            hz = pz0 + t * d_z
            hy = py0 + t * d_y
            if 0 <= hz <= D and 0 <= hy <= ceiling_h:
                candidates.append((t, "right", hz, hy))  # left-end = z=0 (back corner)

    if visible_orients:
        candidates = [c for c in candidates if c[1] in visible_orients]

    if not candidates:
        return None

    # Pick the closest (smallest t) wall plane
    best = min(candidates, key=lambda c: c[0])
    return best[1], best[2], best[3]   # orient, u, y


def _pixel_rects_to_openings(
    raw_openings: list[dict],
    cam: dict,
    room: dict,
    wall_hints: list[str | None] | None = None,
) -> list[dict]:
    """
    Convert VLM pixel_rect detections to physical opening dicts.
    Wall assignment is determined automatically by nearest-wall ray casting
    unless a per-opening wall hint is supplied. Entries without pixel_rect are
    passed through unchanged.
    """
    W         = float(room.get("floor_width_m",    5.0) or 5.0)
    D         = float(room.get("floor_depth_m",    4.0) or 4.0)
    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)
    visible   = set(room.get("_visible_walls", []) or []) or None

    result = []
    for idx, raw in enumerate(raw_openings):
        opening_type = raw.get("type", "window")
        wall_hint = None
        if wall_hints and idx < len(wall_hints):
            wall_hint = wall_hints[idx]

        if "pixel_rect" not in raw:
            result.append(raw)
            continue

        rect = raw["pixel_rect"]
        if len(rect) < 4:
            print(f"[wall_openings] SKIP {opening_type}: bad pixel_rect {rect}")
            continue
        x1, y1, x2, y2 = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Determine wall from centre pixel, optionally forcing the current wall
        centre_visible = {wall_hint} if wall_hint else visible
        centre_hit = _pixel_to_nearest_wall(mid_x, mid_y, cam, W, D, ceiling_h, centre_visible)
        if centre_hit is None:
            print(f"[wall_openings] SKIP {opening_type}: centre pixel doesn't hit any wall")
            continue
        wall = wall_hint or centre_hit[0]

        # Project the full bbox footprint onto the wall plane.
        # Using all four corners plus edge midpoints better preserves apparent
        # opening size under perspective than relying on the bbox centre lines.
        forced = {wall}
        sample_pts = [
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),
            (mid_x, y1), (mid_x, y2), (x1, mid_y), (x2, mid_y),
        ]
        hits = [
            _pixel_to_nearest_wall(px, py, cam, W, D, ceiling_h, forced)
            for px, py in sample_pts
        ]
        valid_hits = [h for h in hits if h is not None]
        if len(valid_hits) < 4:
            print(f"[wall_openings] SKIP {wall} {opening_type}: bbox footprint misses wall")
            continue

        wall_len = W if wall in ("back", "front") else D
        us = [h[1] for h in valid_hits]
        ys = [h[2] for h in valid_hits]
        u_lo  = max(0.0, min(us))
        u_hi  = min(wall_len, max(us))
        sill  = max(0.0, min(ys))
        top   = min(ceiling_h, max(ys))

        # Type-specific vertical snapping
        if opening_type == "door":
            # Doors start at the floor — the bottom pixel rarely hits exactly y=0 due
            # to perspective foreshortening, so we force sill to 0.
            sill = 0.0
            top  = min(ceiling_h, max(ys))  # re-clamp in case top was unreasonably high
        elif opening_type == "window" and sill < _WIN_MIN_SILL_M:
            sill = _WIN_MIN_SILL_M   # windows must clear the floor

        if u_hi - u_lo < 0.05 or top - sill < 0.05:
            print(f"[wall_openings] SKIP {wall} {opening_type}: degenerate projection")
            continue

        print(f"[wall_openings] {wall} {opening_type}: "
              f"pixel [{x1:.0f},{y1:.0f}→{x2:.0f},{y2:.0f}] → "
              f"offset={u_lo:.2f}m w={u_hi-u_lo:.2f}m "
              f"sill={sill:.2f}m h={top-sill:.2f}m")
        result.append({
            "wall":               wall,
            "type":               opening_type,
            "offset_from_left_m": round(u_lo, 3),
            "width_m":            round(u_hi - u_lo, 3),
            "sill_height_m":      round(sill, 3),
            "height_m":           round(top - sill, 3),
        })

    return result


# ── Opening visibility filter ────────────────────────────────────────────────

def _project_wall_pixel_bbox(wall: str, cam: dict, room: dict) -> tuple[float, float, float, float] | None:
    """Project an entire wall plane onto the image and return its pixel bbox."""
    W = float(room.get("floor_width_m", 5.0) or 5.0)
    D = float(room.get("floor_depth_m", 4.0) or 4.0)
    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)

    op = {
        "wall": wall,
        "offset_from_left_m": 0.0,
        "width_m": W if wall in ("back", "front") else D,
        "sill_height_m": 0.0,
        "height_m": ceiling_h,
    }
    return _project_opening_pixel_bbox(op, cam, W, D)


def _project_opening_pixel_bbox(opening: dict, cam: dict, W: float, D: float) -> tuple[float, float, float, float] | None:
    """Project an opening's 3D corners onto the image and return its pixel bbox."""
    res = _build_ray(0, 0, cam)   # just to get pos / camera axes
    if res is None:
        return None
    import numpy as np
    pos = res[0]

    wall      = opening.get("wall", "")
    u_lo      = float(opening.get("offset_from_left_m", 0))
    u_hi      = u_lo + float(opening.get("width_m", 0))
    y_lo      = float(opening.get("sill_height_m", 0))
    y_hi      = y_lo + float(opening.get("height_m", 0))

    def _world(u: float, y: float) -> np.ndarray:
        """Wall-local (u, y) → world 3-D."""
        if wall == "back":   return np.array([u,       y, 0.0])
        if wall == "right":  return np.array([W,       y, u  ])
        if wall == "left":   return np.array([0.0,     y, D-u])
        if wall == "front":  return np.array([W-u,     y, D  ])
        return np.array([u, y, 0.0])

    corners = [_world(u, y) for u in (u_lo, u_hi) for y in (y_lo, y_hi)]

    look_at  = np.array(cam["look_at_m"], dtype=float)
    hfov_deg = float(cam["hfov_deg"])
    img_W    = int(cam["width_px"])
    fwd  = look_at - pos;  fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.,1.,0.]); right /= np.linalg.norm(right)
    fwd_h = np.array([fwd[0],0.,fwd[2]]); fwd_h /= np.linalg.norm(fwd_h)
    fx = img_W / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    cx = img_W / 2.0

    pys = []
    pxs = []
    cy = int(cam["height_px"]) / 2.0
    tilt_tan = float(fwd[1]) / max(np.linalg.norm(np.array([fwd[0], 0.0, fwd[2]])), 1e-9)
    for c in corners:
        d  = c - pos
        xc = float(d @ right)
        zc = float(d @ fwd_h)
        if zc < 0.01:
            continue
        yc = float(d[1]) - tilt_tan * zc
        pxs.append(cx + fx * xc / zc)
        pys.append(cy - fx * yc / zc)

    if not pxs or not pys:
        return None
    return min(pxs), min(pys), max(pxs), max(pys)


def _opening_pixel_rect(opening: dict, cam: dict, W: float, D: float) -> tuple | None:
    """Return only the horizontal pixel span of an opening for visibility checks."""
    bbox = _project_opening_pixel_bbox(opening, cam, W, D)
    if bbox is None:
        return None
    return bbox[0], bbox[2]


def _refinement_opening_context(current_openings: list[dict], cam: dict, room: dict) -> str:
    """Build a concise per-opening context block for the refinement prompt."""
    W = float(room.get("floor_width_m", 5.0) or 5.0)
    D = float(room.get("floor_depth_m", 4.0) or 4.0)
    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)
    lines: list[str] = []
    for idx, op in enumerate(current_openings, start=1):
        bbox = _project_opening_pixel_bbox(op, cam, W, D)
        if bbox is None:
            bbox_str = "unknown"
            left_gap_frac = right_gap_frac = top_gap_frac = bottom_gap_frac = -1.0
        else:
            bbox_str = f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
            wall_bbox = _project_wall_pixel_bbox(op.get("wall", ""), cam, room)
            if wall_bbox is None:
                left_gap_frac = right_gap_frac = top_gap_frac = bottom_gap_frac = -1.0
            else:
                wall_w = max(wall_bbox[2] - wall_bbox[0], 1e-9)
                wall_h = max(wall_bbox[3] - wall_bbox[1], 1e-9)
                left_gap_frac = max(0.0, (bbox[0] - wall_bbox[0]) / wall_w)
                right_gap_frac = max(0.0, (wall_bbox[2] - bbox[2]) / wall_w)
                top_gap_frac = max(0.0, (bbox[1] - wall_bbox[1]) / wall_h)
                bottom_gap_frac = max(0.0, (wall_bbox[3] - bbox[3]) / wall_h)
        wall = op.get("wall", "?")
        wall_bbox = _project_wall_pixel_bbox(wall, cam, room)
        wall_bbox_str = (
            "unknown" if wall_bbox is None
            else f"[{wall_bbox[0]:.0f}, {wall_bbox[1]:.0f}, {wall_bbox[2]:.0f}, {wall_bbox[3]:.0f}]"
        )
        wall_len = W if wall in ("back", "front") else D
        width_frac = float(op.get("width_m", 0.0)) / max(wall_len, 1e-9)
        height_frac = float(op.get("height_m", 0.0)) / max(ceiling_h, 1e-9)
        sill_frac = float(op.get("sill_height_m", 0.0)) / max(ceiling_h, 1e-9)
        horiz_balance = abs(left_gap_frac - right_gap_frac) if left_gap_frac >= 0 and right_gap_frac >= 0 else -1.0
        vert_balance = abs(top_gap_frac - bottom_gap_frac) if top_gap_frac >= 0 and bottom_gap_frac >= 0 else -1.0
        lines.append(
            f"  id={idx} type={op.get('type', 'window')} wall={wall} wall_pixel_rect={wall_bbox_str} current_pixel_rect={bbox_str} "
            f"wall_width_fraction={width_frac:.3f} wall_height_fraction={height_frac:.3f} sill_fraction={sill_frac:.3f} "
            f"left_gap_fraction={left_gap_frac:.3f} right_gap_fraction={right_gap_frac:.3f} "
            f"top_gap_fraction={top_gap_frac:.3f} bottom_gap_fraction={bottom_gap_frac:.3f} "
            f"horizontal_balance_delta={horiz_balance:.3f} vertical_balance_delta={vert_balance:.3f}"
        )
    return "\n".join(lines)


def _mirror_opening(op: dict, W: float, D: float) -> dict:
    """Return the opening with offset measured from the other end of the wall."""
    wall      = op.get("wall", "")
    wall_len  = W if wall in ("back", "front") else D
    new_offset = wall_len - float(op["offset_from_left_m"]) - float(op["width_m"])
    return {**op, "offset_from_left_m": max(0.0, new_offset)}


def _filter_visible_openings(openings: list[dict], cam: dict, room: dict) -> list[dict]:
    """
    For each opening:
      1. If it projects within the image → keep as-is.
      2. If it's off-screen, try the mirrored offset (measured from the other wall end).
         If the mirror IS on-screen → keep the mirrored version (the analysis VLM
         may have measured from the near rather than far end of the wall).
      3. If still off-screen → skip with a warning.
    """
    if not cam:
        return openings
    W         = float(room.get("floor_width_m",    5.0) or 5.0)
    D         = float(room.get("floor_depth_m",    4.0) or 4.0)
    img_W     = int(cam.get("width_px", 4000))

    # Allow 40% overflow: rasterization clips triangles at the viewport edge, so a
    # window whose projected edges fall slightly outside the image is still rendered.
    _overflow = img_W * 0.4

    result = []
    for op in openings:
        rect = _opening_pixel_rect(op, cam, W, D)
        visible = rect is not None and rect[0] <= img_W + _overflow and rect[1] >= -_overflow

        if visible:
            print(f"[wall_openings] KEEP {op.get('wall')} {op.get('type')}: "
                  f"screen px {rect[0]:.0f}–{rect[1]:.0f}")
            result.append(op)
            continue

        # Try mirrored offset
        mirrored = _mirror_opening(op, W, D)
        rect2 = _opening_pixel_rect(mirrored, cam, W, D)
        visible2 = rect2 is not None and rect2[0] <= img_W + _overflow and rect2[1] >= -_overflow

        if visible2:
            print(f"[wall_openings] KEEP {op.get('wall')} {op.get('type')} "
                  f"(mirrored offset {op['offset_from_left_m']:.2f}→{mirrored['offset_from_left_m']:.2f}m): "
                  f"screen px {rect2[0]:.0f}–{rect2[1]:.0f}")
            result.append(mirrored)
        else:
            px_str = f"px {rect[0]:.0f}–{rect[1]:.0f}" if rect else "behind camera"
            print(f"[wall_openings] SKIP {op.get('wall')} {op.get('type')}: "
                  f"off-screen ({px_str}, image 0–{img_W})")

    return result


# ─────────────────────────────────────────────────────────────────────────────

def _call_vlm(prompt: str, images: list[str], max_tokens: int = 600) -> list[dict]:
    """Send prompt + one or more images to the VLM; parse and return opening list."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img_path in images:
        b64, mime = _encode_image(img_path)
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}})
    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    resp = _vlm_post(payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    result = _parse_json_response(raw)
    if isinstance(result, list):
        return result
    print(f"[wall_openings] VLM returned non-list; result={result}")
    return []


def _room_prompt_args(room: dict) -> dict:
    return dict(
        width_m   = float(room.get("floor_width_m",   5.0) or 5.0),
        depth_m   = float(room.get("floor_depth_m",   4.0) or 4.0),
        ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4),
        visible_walls = ", ".join(room.get("_visible_walls", ["back", "left", "right"])),
    )


def detect_openings_vlm(image_path: str, room: dict) -> list[dict]:
    """Call the VLM with the reference image only to detect openings."""
    prompt = _OPENING_PROMPT_TMPL.format(**_room_prompt_args(room))
    print("[wall_openings] Calling VLM (single-image) for opening detection...")
    try:
        result = _call_vlm(prompt, [image_path])
        print(f"[wall_openings] VLM detected {len(result)} opening(s).")
        return result
    except Exception as e:
        print(f"[wall_openings] VLM call failed: {e}")
        return []


def detect_openings_brightness_diff(
    image_path: str,
    render_path: str,
    cam: dict,
    room: dict,
    abs_threshold: float = 165.0,
    diff_threshold: float = 30.0,
    min_wall_frac: float = 0.07,
    min_width_m: float = 0.35,
    min_height_m: float = 0.35,
    gap_m: float = 0.12,
) -> list[dict]:
    """
    Detect window openings purely from image brightness.

    Since the render is pixel-aligned with the reference photo (VGGT camera),
    pixels that are much brighter in the reference than in the solid-wall render
    are exactly where outdoor light (windows/doors) enters the room.

    Algorithm:
      1. Compute a per-pixel luminance mask: reference_lum >= abs_threshold
         AND (reference_lum - render_lum) >= diff_threshold
      2. Project each flagged pixel to a wall plane via _pixel_to_nearest_wall
      3. For each wall, histogram the u-coordinates and split at gaps >= gap_m
         to find separate window clusters
      4. Each cluster yields one opening dict with physical dimensions

    Parameters
    ----------
    abs_threshold  : minimum luminance (0-255) in reference to count as bright
    diff_threshold : minimum (reference - render) luminance difference
    min_wall_frac  : minimum fraction of wall length a cluster must span to be kept
    gap_m          : minimum gap (metres) that separates two distinct windows
    """
    import numpy as np
    from PIL import Image

    render_img = Image.open(render_path).convert("RGB")
    W_px, H_px = render_img.size
    ref_img = Image.open(image_path).convert("RGB").resize((W_px, H_px), Image.LANCZOS)

    ref = np.array(ref_img, dtype=np.float32)
    rnd = np.array(render_img, dtype=np.float32)

    ref_lum = ref.mean(axis=2)
    rnd_lum = rnd.mean(axis=2)

    # Bright-window mask: bright in reference AND much brighter than solid render
    mask = (ref_lum >= abs_threshold) & ((ref_lum - rnd_lum) >= diff_threshold)

    W      = float(room.get("floor_width_m",    5.0) or 5.0)
    D      = float(room.get("floor_depth_m",    4.0) or 4.0)
    ceil_h = float(room.get("ceiling_height_m", 2.4) or 2.4)
    visible = set(room.get("_visible_walls", []) or []) or None

    # Camera z position — used to clip side-wall holes that extend past the camera.
    # When the camera is outside the front wall (cam_z > D), even a small offset
    # near the front corner creates a large void in the render.  Use an aggressive
    # margin proportional to how far outside the room the camera sits.
    try:
        _cam_z = float(cam["position_m"][2])
    except Exception:
        _cam_z = D
    try:
        _cam_x = float(cam["position_m"][0])
    except Exception:
        _cam_x = 0.0
    if _cam_z > D:
        # Camera past front wall: keep at least (overshoot + 0.5m) of solid wall
        # before any left-wall opening (otherwise the opening near the front corner
        # shows void through the wall when rendered from this angle).
        _overshoot    = _cam_z - D
        _left_u_min  = _overshoot + 0.5
    else:
        _left_u_min = max(0.0, D - _cam_z + 0.5)

    if _cam_x >= W:
        # Camera is outside the right wall: looking through a right-wall opening
        # shows the room interior (not void), so no proximity clipping needed.
        _right_u_max = D
    elif _cam_z > D:
        _right_u_max = D - ((_cam_z - D) + 0.5)
    else:
        _right_u_max = max(0.0, _cam_z - 0.5)

    ys_px, xs_px = np.where(mask)
    n_px = len(xs_px)
    if n_px == 0:
        print("[wall_openings] brightness-diff: no bright pixels found")
        return []
    step = max(1, n_px // 15000)
    print(f"[wall_openings] brightness-diff: {n_px} bright pixels, step={step}")

    wall_pts: dict[str, list[tuple[float, float]]] = {}
    for xi, yi in zip(xs_px[::step].tolist(), ys_px[::step].tolist()):
        hit = _pixel_to_nearest_wall(float(xi), float(yi), cam, W, D, ceil_h, visible)
        if hit is None:
            continue
        orient, u, y = hit
        wall_pts.setdefault(orient, []).append((u, y))

    openings: list[dict] = []

    for orient, pts in wall_pts.items():
        wall_len = W if orient in ("back", "front") else D
        if len(pts) < 10:
            continue

        us = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        # Build fine-grained histogram along wall axis; find clusters separated by gaps
        res = 0.02  # 2 cm resolution
        bins = max(10, int(wall_len / res))
        hist, edges = np.histogram(us, bins=bins, range=(0.0, wall_len))

        # Dilate histogram slightly (fill 1-bin gaps) then find segments
        hist_d = np.convolve(hist, np.ones(max(1, int(gap_m / res / 2)), dtype=int),
                             mode="same") > 0
        clusters: list[tuple[float, float]] = []
        in_seg, seg_start = False, 0
        for i, active in enumerate(hist_d):
            if active and not in_seg:
                in_seg, seg_start = True, i
            elif not active and in_seg:
                clusters.append((edges[seg_start], edges[i]))
                in_seg = False
        if in_seg:
            clusters.append((edges[seg_start], edges[len(hist)]))

        for u_lo, u_hi in clusters:
            w_m = u_hi - u_lo
            if w_m < min_wall_frac * wall_len or w_m < min_width_m:
                continue
            in_cluster = (us >= u_lo) & (us <= u_hi)
            ys_c = ys[in_cluster]
            if len(ys_c) < 5:
                continue
            # Use percentiles to trim stray points
            y_lo = float(np.percentile(ys_c, 3))
            y_hi = float(np.percentile(ys_c, 97))
            y_lo = max(0.0, y_lo)
            y_hi = min(ceil_h, y_hi)

            if y_hi - y_lo < min_height_m:
                continue

            # Large windows (>1.2m wide, >30% of wall) are likely floor-to-ceiling
            # glass panels where furniture or low light blocks the bottom portion.
            # Snap to full height so the mesh opening matches the physical glass.
            is_large_panel = w_m >= 1.2 and w_m >= 0.3 * wall_len
            if is_large_panel:
                y_lo = 0.0
                y_hi = ceil_h

            # Classify as door only when naturally at floor level (not snapped)
            opening_type = "door" if (not is_large_panel and y_lo < 0.1) else "window"
            if opening_type == "window" and y_lo < _WIN_MIN_SILL_M:
                y_lo = _WIN_MIN_SILL_M
            if y_hi - y_lo < 0.1:
                continue

            openings.append({
                "wall":               orient,
                "type":               opening_type,
                "offset_from_left_m": round(float(u_lo), 3),
                "width_m":            round(float(u_hi - u_lo), 3),
                "sill_height_m":      round(y_lo, 3),
                "height_m":           round(float(y_hi - y_lo), 3),
            })
            print(f"[wall_openings] brightness-diff {orient} {opening_type}: "
                  f"offset={u_lo:.2f}m w={u_hi-u_lo:.2f}m "
                  f"sill={y_lo:.2f}m h={y_hi-y_lo:.2f}m")

    # ── Post-process side walls ───────────────────────────────────────────────
    # 1. For each side wall keep only the LARGEST window (camera geometry often
    #    produces small duplicate/noise clusters on left & right walls).
    for orient in ("left", "right"):
        wall_ops = [o for o in openings if o["wall"] == orient]
        if len(wall_ops) > 1:
            largest = max(wall_ops, key=lambda o: o["width_m"])
            removed = len(wall_ops) - 1
            openings = [o for o in openings if o["wall"] != orient]
            openings.append(largest)
            print(f"[wall_openings] brightness-diff: dropped {removed} noise cluster(s) on {orient} wall")

    # 2. Mirror heuristic: camera outside a side wall can only see the near-front
    #    portion of that wall.  If left/right windows exist and one is much narrower,
    #    extend the smaller one using the mirror of the larger.
    left_ops  = [o for o in openings if o["wall"] == "left"]
    right_ops = [o for o in openings if o["wall"] == "right"]
    if len(left_ops) == 1 and len(right_ops) == 1:
        lw = left_ops[0];  rw = right_ops[0]
        l_lo = lw["offset_from_left_m"];  l_hi = l_lo + lw["width_m"]
        r_lo = rw["offset_from_left_m"];  r_hi = r_lo + rw["width_m"]
        # Mirror: right_u = D - left_u (symmetry about mid-depth)
        mir_r_lo = D - l_hi;  mir_r_hi = D - l_lo
        mir_l_lo = D - r_hi;  mir_l_hi = D - r_lo
        # Extend right window if near-camera truncation is likely
        if (rw["width_m"] < lw["width_m"] * 0.65
                and abs(r_hi - mir_r_hi) < 0.6):
            rw["offset_from_left_m"] = round(max(0.0, mir_r_lo), 3)
            rw["width_m"] = round(min(D, mir_r_hi) - rw["offset_from_left_m"], 3)
            # Large panel: snap to floor-to-ceiling
            if rw["width_m"] >= 1.2 and rw["width_m"] >= 0.3 * D:
                rw["sill_height_m"] = 0.0
                rw["height_m"] = round(ceil_h, 3)
            print(f"[wall_openings] Mirror heuristic → right window "
                  f"offset={rw['offset_from_left_m']:.2f}m w={rw['width_m']:.2f}m")
        # Symmetric case: extend left window
        elif (lw["width_m"] < rw["width_m"] * 0.65
              and abs(l_hi - mir_l_hi) < 0.6):
            lw["offset_from_left_m"] = round(max(0.0, mir_l_lo), 3)
            lw["width_m"] = round(min(D, mir_l_hi) - lw["offset_from_left_m"], 3)
            if lw["width_m"] >= 1.2 and lw["width_m"] >= 0.3 * D:
                lw["sill_height_m"] = 0.0
                lw["height_m"] = round(ceil_h, 3)
            print(f"[wall_openings] Mirror heuristic → left window "
                  f"offset={lw['offset_from_left_m']:.2f}m w={lw['width_m']:.2f}m")

    # 3. Camera-proximity clipping (applied AFTER mirror so mirror uses raw positions)
    wall_len_side = D
    clipped_out: list[dict] = []
    for op in openings:
        orient = op["wall"]
        if orient not in ("left", "right"):
            continue
        u_lo = op["offset_from_left_m"]
        u_hi = u_lo + op["width_m"]
        if orient == "left":
            u_lo = max(u_lo, _left_u_min)
        else:
            u_hi = min(u_hi, _right_u_max)
        w = u_hi - u_lo
        if w < min_width_m or w < min_wall_frac * wall_len_side:
            clipped_out.append(op)
            print(f"[wall_openings] brightness-diff: {orient} window removed after clip (w={w:.2f}m)")
        else:
            op["offset_from_left_m"] = round(u_lo, 3)
            op["width_m"] = round(w, 3)
            # Re-check large-panel floor-to-ceiling snap after clip
            if w >= 1.2 and w >= 0.3 * wall_len_side:
                op["sill_height_m"] = 0.0
                op["height_m"] = round(ceil_h, 3)
    for op in clipped_out:
        openings.remove(op)

    print(f"[wall_openings] brightness-diff: {len(openings)} opening(s) detected")
    return openings


# ── Post-alignment wall correspondence ───────────────────────────────────────

def compute_wall_correspondence(cam: dict, room: dict) -> dict:
    """
    After VGGT/Manhattan alignment, compute which 3D room walls are visible from the
    calibrated camera and where each wall appears in the rendered image (left/center/right).

    This creates a persistent understanding of the 2D→3D wall mapping so that
    downstream VLM calls (opening detection, refinement) receive prompts with correct
    room-coordinate labels rather than generic image-space labels.

    Returns a dict saved as wall_correspondence.json:
      visible_walls       : room-coord wall names visible from this camera
      non_visible_walls   : walls behind or exterior to the camera
      image_order         : visible walls sorted left→right by pixel position
      image_space_to_room : e.g. {"left": "left", "right": "back"}
                            maps VLM image-space label → room coordinate label
      wall_pixel_centers  : pixel-x of each visible wall's centre in the render
      wall_pixel_ranges   : [x_min, x_max] in image pixels for each visible wall
    """
    W         = float(room.get("floor_width_m",    5.0) or 5.0)
    D         = float(room.get("floor_depth_m",    4.0) or 4.0)
    img_W     = int(cam.get("width_px", 4000))
    img_H     = int(cam.get("height_px", 2250))

    try:
        cam_pos = cam["position_m"]
        cam_x, cam_z = float(cam_pos[0]), float(cam_pos[2])
        look_at = cam["look_at_m"]
        look_dir = [look_at[i] - cam_pos[i] for i in range(3)]
    except Exception:
        return {}

    # Interior-facing + aimed-at check (same as in run())
    def _aimed(orient: str) -> bool:
        if orient == "back":   return cam_z > -0.1 and look_dir[2] < 0
        if orient == "front":  return cam_z < D + 0.1 and look_dir[2] > 0
        if orient == "left":   return cam_x > -0.1 and look_dir[0] < 0
        if orient == "right":  return cam_x < W + 0.1 and look_dir[0] > 0
        return False

    all_walls = ["back", "left", "right", "front"]
    analysis_visible = set(room.get("_visible_walls", all_walls))

    visible: list[str] = []
    non_visible: list[str] = []
    for w in all_walls:
        if w in analysis_visible and _aimed(w):
            visible.append(w)
        else:
            non_visible.append(w)

    # Project each visible wall and measure its full pixel footprint (x and y)
    wall_px_center: dict[str, float] = {}
    wall_px_range:  dict[str, list]  = {}
    wall_px_bbox:   dict[str, list]  = {}   # [x_min, y_min, x_max, y_max]
    for wall in visible:
        bbox = _project_wall_pixel_bbox(wall, cam, room)
        if bbox is not None:
            x_lo, y_lo, x_hi, y_hi = bbox
            wall_px_center[wall] = (x_lo + x_hi) / 2.0
            wall_px_range[wall]  = [round(x_lo, 1), round(x_hi, 1)]
            wall_px_bbox[wall]   = [round(x_lo, 1), round(y_lo, 1),
                                     round(x_hi, 1), round(y_hi, 1)]

    # Sort visible walls by pixel position (left → right in image)
    image_order = sorted(wall_px_center.keys(), key=lambda w: wall_px_center[w])
    n = len(image_order)

    # Build image-space label → room-coord remap.
    # The VLM prompt defines:
    #   "left"  = side wall on the LEFT side of the image
    #   "right" = side wall on the RIGHT side of the image
    #   "back"  = wall facing the camera (visible in centre/RIGHT of the render)
    # With n≥2 sorted left→right by pixel position:
    #   leftmost  → "left"
    #   rightmost → "right" AND "back" (both map to the same dominant facing wall)
    image_space_to_room: dict[str, str] = {}
    if n >= 1:
        image_space_to_room["left"]  = image_order[0]
        image_space_to_room["right"] = image_order[-1]
        # "back" = the rightmost/most-central facing wall; for n≥2 use rightmost
        # so VLM saying "back" or "right" both land on the same room wall.
        image_space_to_room["back"]  = image_order[-1] if n == 2 else image_order[n // 2]

    changed = {k: v for k, v in image_space_to_room.items() if k != v}
    if changed:
        print(f"[wall_openings] wall_correspondence: image-space→room remap: {changed}")

    correspondence = {
        "visible_walls":       visible,
        "non_visible_walls":   non_visible,
        "image_order":         image_order,
        "image_space_to_room": image_space_to_room,
        "wall_pixel_centers":  {w: round(wall_px_center[w], 1) for w in image_order},
        "wall_pixel_ranges":   wall_px_range,
        "wall_pixel_bboxes":   wall_px_bbox,
        "image_width_px":      img_W,
        "image_height_px":     img_H,
    }
    return correspondence


def _build_wall_labels_context(cam: dict, room: dict) -> tuple[str, dict[str, str]]:
    """
    Build a dynamic WALL LABELS prompt section and a VLM-label→room-label remap dict.

    Reads from room["_wall_correspondence"] if already computed by
    compute_wall_correspondence(); otherwise falls back to projecting walls on the fly.

    Returns:
      wall_labels_context : str  — prompt section telling the VLM which room-coord
                                   label to use for each visible wall in the render.
      remap               : dict — maps image-space VLM label ("left"/"right"/"back")
                                   to the correct room-coordinate label.
    """
    # Prefer the pre-computed correspondence (written to wall_correspondence.json in run())
    corr = room.get("_wall_correspondence") or {}
    image_order        = corr.get("image_order", [])
    wall_px_centers    = corr.get("wall_pixel_centers", {})
    remap              = dict(corr.get("image_space_to_room", {}))
    visible            = corr.get("visible_walls", room.get("_visible_walls", []) or [])
    img_W              = corr.get("image_width_px", int(cam.get("width_px", 4000)))

    # If correspondence not yet available, compute it now (e.g. called stand-alone)
    if not image_order:
        live_corr  = compute_wall_correspondence(cam, room)
        image_order     = live_corr.get("image_order", [])
        wall_px_centers = live_corr.get("wall_pixel_centers", {})
        remap           = dict(live_corr.get("image_space_to_room", {}))
        visible         = live_corr.get("visible_walls", visible)
        img_W           = live_corr.get("image_width_px", img_W)

    if not image_order:
        generic = (
            "WALL LABELS (identify from IMAGE 2):\n"
            "  back  — wall facing the camera (visible in centre/right of the render)\n"
            "  left  — side wall on the LEFT side of the image\n"
            "  right — side wall on the RIGHT side of the image\n"
            "  front — wall behind the camera (rarely visible)"
        )
        return generic, {}

    def _pos_label(px: float) -> str:
        frac = px / max(img_W, 1)
        if frac < 0.35:  return "LEFT side"
        if frac > 0.65:  return "RIGHT side"
        return "CENTER"

    # Invert remap to find the image-space label for each room wall
    room_to_img = {v: k for k, v in remap.items()}

    lines = [
        "WALL LABELS — IMPORTANT: use EXACTLY these room-coordinate labels in your output.",
        "  After 3D alignment the visible walls appear at these positions in IMAGE 2:",
    ]
    for wall in image_order:
        px  = wall_px_centers.get(wall, img_W / 2)
        pos = _pos_label(px)
        img_lbl = room_to_img.get(wall, wall)
        # Include the pixel bbox so VLM can measure openings relative to wall edges
        wall_px_bbox = corr.get("wall_pixel_bboxes", {})
        bb = wall_px_bbox.get(wall)
        bbox_str = (f"  pixel bbox in IMAGE 2: x=[{bb[0]:.0f}, {bb[2]:.0f}]  y=[{bb[1]:.0f}, {bb[3]:.0f}]"
                    if bb else "")
        lines.append(
            f'  Use label "{wall}" — visible on the {pos} of the render'
            + (f' (instinctively called "{img_lbl}" — use "{wall}" instead)' if img_lbl != wall else "")
        )
        if bbox_str:
            lines.append(bbox_str)
    invisible = [w for w in ("back", "left", "right", "front") if w not in visible]
    if invisible:
        lines.append(
            f"  NOT visible / do NOT output labels: {', '.join(invisible)}"
        )

    changed = {k: v for k, v in remap.items() if k != v}
    if changed:
        print(f"[wall_openings] wall-label remap (image-space → room): {changed}")

    return "\n".join(lines), remap


def detect_openings_hybrid(
    image_path: str,
    render_path: str,
    cam: dict,
    room: dict,
) -> list[dict]:
    """
    Hybrid detection combining VLM semantic analysis with brightness-diff precision.

    Phase 1 — VLM (reference + render):
        Returns per-opening estimates of wall, type, sill_height_m, height_m,
        left_frac (horizontal position as fraction of wall), and width_frac.
        VLM is reliable for vertical dims (sill / height) and wall assignment,
        less reliable for exact horizontal pixel positions.

    Phase 2 — Brightness-diff refinement (horizontal only):
        For each VLM-estimated opening, look for a brightness cluster on the
        same wall whose u-range overlaps the VLM's fractional estimate.
        If a matching cluster exists, use its precise u_lo/u_hi.
        Otherwise keep the VLM's fractional estimate converted to metres.
    """
    W      = float(room.get("floor_width_m",    5.0) or 5.0)
    D      = float(room.get("floor_depth_m",    4.0) or 4.0)
    ceil_h = float(room.get("ceiling_height_m", 2.4) or 2.4)

    # Camera proximity limits — prevent side-wall openings extending past camera
    try:
        _cam_z = float(cam["position_m"][2])
    except Exception:
        _cam_z = D
    try:
        _cam_x = float(cam["position_m"][0])
    except Exception:
        _cam_x = 0.0
    if _cam_z > D:
        _overshoot   = _cam_z - D
        _left_u_min  = _overshoot + 0.5
    else:
        _left_u_min = max(0.0, D - _cam_z + 0.5)
    if _cam_x >= W:
        _right_u_max = D
    elif _cam_z > D:
        _right_u_max = D - ((_cam_z - D) + 0.5)
    else:
        _right_u_max = max(0.0, _cam_z - 0.5)

    # ── Phase 1: VLM wall-level analysis ─────────────────────────────────────
    # Build a camera-aware wall-labels section so the VLM uses the correct room
    # coordinate labels after VGGT/Manhattan alignment (the render may show the
    # "back" wall on the right side of the image, etc.).
    wall_labels_ctx, vlm_wall_remap = _build_wall_labels_context(cam, room)
    prompt = _WALL_OPENING_PROMPT_TMPL.format(
        wall_labels_context=wall_labels_ctx,
        floor_width=W, floor_depth=D, ceiling_h=ceil_h
    )
    print("[wall_openings] Calling VLM (wall-level semantic analysis)...")
    try:
        vlm_raw = _call_vlm(prompt, [image_path, render_path], max_tokens=600)
    except Exception as e:
        print(f"[wall_openings] VLM wall analysis failed: {e}")
        vlm_raw = []

    if not vlm_raw:
        print("[wall_openings] hybrid: VLM returned nothing — skipping")
        return []

    # Remap any image-space VLM labels to room-coordinate labels in case the VLM
    # still used generic positional labels ("right" for the wall on the right side of
    # the image) rather than the corrected labels from the prompt.
    if vlm_wall_remap:
        remapped_count = 0
        for item in vlm_raw:
            orig = item.get("wall", "")
            mapped = vlm_wall_remap.get(orig, orig)
            if mapped != orig:
                item["wall"] = mapped
                remapped_count += 1
        if remapped_count:
            print(f"[wall_openings] Remapped {remapped_count} VLM wall label(s) to room coords")

    # Drop any VLM result whose wall is not in the updated visible walls list.
    visible_set = set(room.get("_visible_walls", []) or [])
    if visible_set:
        before = len(vlm_raw)
        vlm_raw = [item for item in vlm_raw if item.get("wall", "") in visible_set]
        dropped = before - len(vlm_raw)
        if dropped:
            print(f"[wall_openings] Dropped {dropped} VLM opening(s) on non-visible walls")

    print(f"[wall_openings] hybrid: VLM returned {len(vlm_raw)} opening estimate(s)")

    # ── Phase 2: brightness-diff clusters per wall ───────────────────────────
    import numpy as np
    from PIL import Image

    render_img = Image.open(render_path).convert("RGB")
    W_px, H_px = render_img.size
    ref_img = Image.open(image_path).convert("RGB").resize((W_px, H_px), Image.LANCZOS)

    ref = np.array(ref_img, dtype=np.float32)
    rnd = np.array(render_img, dtype=np.float32)
    ref_lum = ref.mean(axis=2)
    rnd_lum = rnd.mean(axis=2)
    mask = (ref_lum >= 185.0) & ((ref_lum - rnd_lum) >= 50.0)

    visible = set(room.get("_visible_walls", []) or []) or None

    # Project bright pixels to walls → collect (u, y) per wall
    ys_px, xs_px = np.where(mask)
    n_px = len(xs_px)
    step = max(1, n_px // 15000)
    wall_pts: dict[str, list[tuple[float, float]]] = {}  # orient → [(u, y), ...]
    for xi, yi in zip(xs_px[::step].tolist(), ys_px[::step].tolist()):
        hit = _pixel_to_nearest_wall(float(xi), float(yi), cam, W, D, ceil_h, visible)
        if hit is None:
            continue
        orient, u, y = hit
        wall_pts.setdefault(orient, []).append((u, y))

    def _bright_clusters(orient: str, gap_m: float = 0.10) -> list[tuple[float, float]]:
        """Return (u_lo, u_hi) bright clusters for a wall, sorted by u."""
        pts = wall_pts.get(orient, [])
        if len(pts) < 8:
            return []
        wall_len = W if orient in ("back", "front") else D
        us = np.array([p[0] for p in pts])
        res = 0.02
        bins = max(10, int(wall_len / res))
        hist, edges = np.histogram(us, bins=bins, range=(0.0, wall_len))
        fill = max(1, int(gap_m / res / 2))
        hist_d = np.convolve(hist, np.ones(fill, dtype=int), mode="same") > 0
        clusters: list[tuple[float, float]] = []
        in_seg, seg_start = False, 0
        for i, active in enumerate(hist_d):
            if active and not in_seg:
                in_seg, seg_start = True, i
            elif not active and in_seg:
                clusters.append((float(edges[seg_start]), float(edges[i])))
                in_seg = False
        if in_seg:
            clusters.append((float(edges[seg_start]), float(edges[len(hist)])))
        return clusters

    def _bright_cluster_y(orient: str, u_lo: float, u_hi: float) -> tuple[float, float]:
        """Return (y_lo, y_hi) for the bright pixels within [u_lo, u_hi] on a wall."""
        pts = wall_pts.get(orient, [])
        ys_in = [p[1] for p in pts if u_lo - 0.05 <= p[0] <= u_hi + 0.05]
        if len(ys_in) < 4:
            return 0.0, ceil_h
        ys_arr = np.array(ys_in)
        return float(np.percentile(ys_arr, 3)), float(np.percentile(ys_arr, 97))

    # ── Wall pixel bboxes for pixel-fraction → world conversion ─────────────
    # Precompute per-wall pixel bboxes so we can ray-cast VLM pixel fractions
    # back to world coordinates (handles perspective correctly).
    corr_bboxes: dict[str, list] = (
        (room.get("_wall_correspondence") or {}).get("wall_pixel_bboxes", {})
    )

    def _vlm_fracs_to_world(
        wall: str,
        left_gap_frac: float, width_frac: float,
        sill_frac: float,     height_frac: float,
    ) -> tuple[float, float, float, float]:
        """
        Convert VLM pixel-relative fractions to world-space (u_lo, u_hi, sill, h).
        Fractions are measured relative to the wall's pixel bbox in the render:
          left_gap_frac: gap from wall's left pixel edge to opening's left pixel edge
          width_frac:    opening pixel width / wall pixel width
          sill_frac:     gap from floor up (wall_y_max − opening_bottom) / wall_height
          height_frac:   opening pixel height / wall pixel height
        Ray-casting each edge pixel back to the wall plane gives accurate world coords.
        """
        bb = corr_bboxes.get(wall)
        wall_len = W if wall in ("back", "front") else D
        if bb is None:
            # No pixel bbox — fall back to simple fraction × physical length
            u_lo = max(0.0, left_gap_frac * wall_len)
            u_hi = min(wall_len, u_lo + width_frac * wall_len)
            sill = max(0.0, sill_frac * ceil_h)
            h    = min(ceil_h - sill, height_frac * ceil_h)
            return u_lo, u_hi, sill, h

        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        px_w = max(x_max - x_min, 1.0)
        px_h = max(y_max - y_min, 1.0)

        # Pixel x of left and right opening edges
        open_x_lo = x_min + left_gap_frac * px_w
        open_x_hi = open_x_lo + width_frac * px_w
        # Pixel y of bottom (sill) and top of opening
        # sill_frac is from the floor up: floor = y_max, ceiling = y_min
        open_y_bot = y_max - sill_frac * px_h           # pixel y of sill
        open_y_top = open_y_bot - height_frac * px_h    # pixel y of opening top

        # Vertical midpoint for horizontal ray-cast
        mid_y = (open_y_bot + open_y_top) / 2.0
        # Horizontal midpoint for vertical ray-cast
        mid_x = (open_x_lo + open_x_hi) / 2.0

        forced = {wall}
        hit_lo = _pixel_to_nearest_wall(open_x_lo, mid_y, cam, W, D, ceil_h, forced)
        hit_hi = _pixel_to_nearest_wall(open_x_hi, mid_y, cam, W, D, ceil_h, forced)
        hit_bot = _pixel_to_nearest_wall(mid_x, open_y_bot, cam, W, D, ceil_h, forced)
        hit_top = _pixel_to_nearest_wall(mid_x, open_y_top, cam, W, D, ceil_h, forced)

        # Horizontal: use ray-cast if both edges hit; else fall back to fractions
        if hit_lo and hit_hi:
            u_lo = max(0.0, min(hit_lo[1], hit_hi[1]))
            u_hi = min(wall_len, max(hit_lo[1], hit_hi[1]))
        else:
            u_lo = max(0.0, left_gap_frac * wall_len)
            u_hi = min(wall_len, u_lo + width_frac * wall_len)

        # Vertical: use ray-cast heights if both edges hit
        if hit_bot and hit_top:
            sill = max(0.0, min(hit_bot[2], hit_top[2]))
            h    = min(ceil_h - sill, max(hit_bot[2], hit_top[2]) - sill)
        else:
            sill = max(0.0, sill_frac * ceil_h)
            h    = min(ceil_h - sill, height_frac * ceil_h)

        return u_lo, u_hi, max(0.0, sill), max(0.05, h)

    # ── Combine VLM estimates with brightness-diff horizontal precision ───────
    openings: list[dict] = []

    for item in vlm_raw:
        wall = item.get("wall", "")
        if wall not in ("back", "left", "right", "front"):
            continue
        opening_type = item.get("type", "window")
        wall_len = W if wall in ("back", "front") else D

        # VLM pixel-relative fractions (new format) or legacy physical fractions
        has_px_fracs = "left_gap_frac" in item or "sill_frac" in item
        if has_px_fracs:
            lgf = float(item.get("left_gap_frac", 0.0) or 0.0)
            wf  = float(item.get("width_frac",    0.5) or 0.5)
            sf  = float(item.get("sill_frac",     0.0) or 0.0)
            hf  = float(item.get("height_frac",   0.5) or 0.5)
            vlm_u_lo, vlm_u_hi, sill, h = _vlm_fracs_to_world(wall, lgf, wf, sf, hf)
        else:
            # Legacy: left_frac / width_frac as physical-length fractions
            lf = float(item.get("left_frac",     0.0) or 0.0)
            wf = float(item.get("width_frac",    0.2) or 0.2)
            sill = max(0.0, float(item.get("sill_height_m", 0.5) or 0.5))
            h    = max(0.1, float(item.get("height_m",      1.2) or 1.2))
            sill = min(sill, ceil_h - 0.1)
            h    = min(h, ceil_h - sill)
            vlm_u_lo = max(0.0, lf * wall_len)
            vlm_u_hi = min(wall_len, (lf + wf) * wall_len)

        if vlm_u_hi - vlm_u_lo < 0.05:
            print(f"[wall_openings] hybrid: skip {wall} {opening_type} — VLM width too small")
            continue

        # Find the closest brightness-diff cluster on this wall
        clusters = _bright_clusters(wall)
        best_cluster: tuple[float, float] | None = None
        best_overlap = 0.0
        for c_lo, c_hi in clusters:
            overlap = max(0.0, min(vlm_u_hi, c_hi) - max(vlm_u_lo, c_lo))
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = (c_lo, c_hi)

        if best_cluster is not None and best_overlap > 0.05:
            # Use brightness-diff horizontal, VLM vertical
            u_lo, u_hi = best_cluster
            # Clip side-wall openings that extend past the camera
            if wall == "left":
                u_lo = max(u_lo, _left_u_min)
            elif wall == "right":
                u_hi = min(u_hi, _right_u_max)
            src = "bright-diff"
        else:
            # No bright signal — trust VLM fractions
            u_lo, u_hi = vlm_u_lo, vlm_u_hi
            # Clip side-wall openings that extend past the camera
            if wall == "left":
                u_lo = max(u_lo, _left_u_min)
            elif wall == "right":
                u_hi = min(u_hi, _right_u_max)
            src = "vlm-frac"

        # Snap door sill
        if opening_type == "door":
            sill = 0.0
        elif sill < _WIN_MIN_SILL_M:
            sill = _WIN_MIN_SILL_M

        openings.append({
            "wall":               wall,
            "type":               opening_type,
            "offset_from_left_m": round(u_lo, 3),
            "width_m":            round(u_hi - u_lo, 3),
            "sill_height_m":      round(sill, 3),
            "height_m":           round(h, 3),
        })
        print(f"[wall_openings] hybrid [{src}] {wall} {opening_type}: "
              f"offset={u_lo:.2f}m w={u_hi-u_lo:.2f}m sill={sill:.2f}m h={h:.2f}m")

    print(f"[wall_openings] hybrid: {len(openings)} opening(s)")
    return openings


def refine_openings_vlm_fracs(
    image_path: str, render_path: str, room: dict, cam: dict, current_openings: list[dict],
) -> list[dict]:
    """
    Refine openings by asking the VLM to measure relative gap fractions from the
    reference image.  Fractions are converted directly to physical dimensions so
    there is no pixel-rect → wall-plane projection step and no scale guessing.
    """
    if not current_openings:
        return []

    W         = float(room.get("floor_width_m",    5.0) or 5.0)
    D         = float(room.get("floor_depth_m",    4.0) or 4.0)
    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)

    prompt = _REFINEMENT_FRACS_PROMPT_TMPL.format(
        opening_context=_refinement_opening_context(current_openings, cam, room)
    )
    print("[wall_openings] Calling VLM (fraction-based refinement)...")
    try:
        raw = _call_vlm(prompt, [image_path, render_path], max_tokens=600)
        print(f"[wall_openings] VLM returned {len(raw)} fraction estimate(s).")
        if len(raw) != len(current_openings):
            print("[wall_openings] Fraction refinement count mismatch; keeping current.")
            return []

        accepted: list[dict] = []
        for idx, (prev, item) in enumerate(zip(current_openings, raw)):
            wall     = prev.get("wall", "")
            wall_len = W if wall in ("back", "front") else D

            left_gap_frac = max(0.0, float(item.get("left_gap_frac", 0.0)))
            width_frac    = max(0.01, float(item.get("width_frac",    0.5)))
            sill_frac     = max(0.0,  float(item.get("sill_frac",     0.0)))
            height_frac   = max(0.01, float(item.get("height_frac",   0.5)))

            nxt = {
                **prev,
                "offset_from_left_m": round(left_gap_frac * wall_len,  3),
                "width_m":            round(width_frac    * wall_len,  3),
                "sill_height_m":      round(sill_frac     * ceiling_h, 3),
                "height_m":           round(height_frac   * ceiling_h, 3),
            }
            rationale = str(item.get("rationale", "")).strip()
            print(
                f"[wall_openings] frac-refine #{idx+1}: {wall} {prev.get('type')} "
                f"offset={nxt['offset_from_left_m']:.2f}m w={nxt['width_m']:.2f}m "
                f"sill={nxt['sill_height_m']:.2f}m h={nxt['height_m']:.2f}m"
                + (f" — {rationale}" if rationale else "")
            )
            nxt = _accept_refined_opening(prev, nxt, room)
            accepted.append(nxt)

        return accepted
    except Exception as e:
        print(f"[wall_openings] VLM fraction refinement failed: {e}")
        return []


# ── Fallback: read openings from floorplan_analysis.json ─────────────────────

def openings_from_analysis(analysis: dict) -> list[dict]:
    """
    Extract door/window openings from floorplan_analysis.json.
    Adds sill_height_m = 0.0 for doors, 0.9 for windows (not in original schema).
    """
    openings = []
    for wall in analysis.get("walls", []):
        orient = wall.get("orientation", "")
        for feat in wall.get("features", []):
            if feat.get("type") not in ("door", "window"):
                continue
            sill = 0.0 if feat["type"] == "door" else float(feat.get("sill_height_m", 0.9))
            openings.append({
                "wall":              orient,
                "type":              feat["type"],
                "width_m":           float(feat.get("width_m", 0.9)),
                "height_m":          float(feat.get("height_m", 2.1 if feat["type"] == "door" else 1.0)),
                "offset_from_left_m": float(feat.get("offset_from_left_m", 0.0)),
                "sill_height_m":     sill,
            })
    return openings


# ── Opening validation ────────────────────────────────────────────────────────

# Minimum gap between a window edge and the floor / ceiling (metres).
# A window that the VLM places too close to the floor or ceiling is pulled back
# so the wall mesh always has a visible sill and a visible soffit band.
_WIN_MIN_SILL_M   = 0.15   # minimum gap between window bottom and floor
_WIN_MIN_SOFFIT_M = 0.10   # minimum gap between window top  and ceiling
_DOOR_MIN_HEIGHT_M = 1.8   # reject refinements that shrink a door below this
_DOOR_MAX_SILL_M   = 0.15  # sill above this → declared door is retyped as window
_REFINE_MAX_ITERS = 3
_REFINE_EPS_M     = 0.03


def _merge_framed_openings(
    openings: list[dict],
    frame_gap_m: float = 0.40,
    sill_tol_m: float  = 0.40,
    height_tol_m: float = 0.60,
) -> list[dict]:
    """
    Merge horizontally adjacent openings on the same wall into one larger opening.

    A multi-section window (e.g. three panes separated by vertical frames) is
    detected by the VLM as several separate openings.  Since window frames are
    generated in a later pass, we collapse such clusters into a single bounding
    rectangle here so only one hole is punched in the wall mesh.

    Merging criteria (all must hold):
      • Same wall and same type
      • Horizontal gap between consecutive openings ≤ frame_gap_m
      • Sill heights within sill_tol_m of each other
      • Heights within height_tol_m of each other

    The merged opening spans from the leftmost offset to the rightmost edge.
    Sill and height are the envelope (min sill, max top).
    """
    if not openings:
        return openings

    # Group by (wall, type)
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for op in openings:
        groups[(op.get("wall", ""), op.get("type", "window"))].append(op)

    merged: list[dict] = []
    for (wall, typ), ops in groups.items():
        # Sort by horizontal offset
        ops = sorted(ops, key=lambda o: float(o["offset_from_left_m"]))

        cluster: list[dict] = [ops[0]]
        for op in ops[1:]:
            prev = cluster[-1]
            prev_right = float(prev["offset_from_left_m"]) + float(prev["width_m"])
            gap = float(op["offset_from_left_m"]) - prev_right
            sill_diff   = abs(float(op["sill_height_m"])  - float(prev["sill_height_m"]))
            height_diff = abs(float(op["height_m"])       - float(prev["height_m"]))

            # For windows: only gap and sill proximity matter — height can vary
            # between panes (frames will be added later to fill the difference).
            # For doors: also check height since mixing door/non-door is wrong.
            height_ok = (typ != "door") or (height_diff <= height_tol_m)
            if gap <= frame_gap_m and sill_diff <= sill_tol_m and height_ok:
                cluster.append(op)
            else:
                merged.append(_collapse_cluster(cluster))
                cluster = [op]
        merged.append(_collapse_cluster(cluster))

    if len(merged) < len(openings):
        print(f"[wall_openings] Merged {len(openings)} opening(s) → {len(merged)} "
              f"(frame-divided windows collapsed)")
    return merged


def _collapse_cluster(ops: list[dict]) -> dict:
    """Collapse a list of same-wall openings into one bounding-box opening."""
    if len(ops) == 1:
        return ops[0]
    u_lo  = min(float(o["offset_from_left_m"]) for o in ops)
    u_hi  = max(float(o["offset_from_left_m"]) + float(o["width_m"]) for o in ops)
    sill  = min(float(o["sill_height_m"]) for o in ops)
    top   = max(float(o["sill_height_m"]) + float(o["height_m"]) for o in ops)
    return {
        **ops[0],
        "offset_from_left_m": round(u_lo, 3),
        "width_m":            round(u_hi - u_lo, 3),
        "sill_height_m":      round(sill, 3),
        "height_m":           round(top - sill, 3),
    }


def _enforce_opening_type_shape(openings: list[dict], ceiling_h: float) -> list[dict]:
    """
    Apply type-specific physical constraints after any detection or refinement pass.

    door
      • sill must be 0.0 (doors start at the floor)
      • height must be >= _DOOR_MIN_HEIGHT_M
      • if sill > _DOOR_MAX_SILL_M the opening is retyped to "window" (VLM misclassified)

    window
      • sill >= _WIN_MIN_SILL_M   (windows clear the floor)
      • top  <= ceiling_h − _WIN_MIN_SOFFIT_M   (soffit band visible above window)
      • Windows whose height drops to ≤ 0 after clamping are dropped.
    """
    result = []
    for op in openings:
        op   = dict(op)
        typ  = op.get("type", "window")
        sill = float(op.get("sill_height_m", 0.0))
        h    = float(op.get("height_m", 1.0))

        if typ == "door":
            if sill > _DOOR_MAX_SILL_M:
                # Sill far above floor — VLM misclassified; reclassify as window
                print(f"[wall_openings] RETYPE door→window on {op.get('wall')}: "
                      f"sill={sill:.2f}m > {_DOOR_MAX_SILL_M}m floor threshold")
                op["type"] = "window"
                # fall through to window handling below
            else:
                if sill != 0.0:
                    print(f"[wall_openings] SNAP door sill→0 on {op.get('wall')}: "
                          f"{sill:.2f}m → 0.0m")
                op["sill_height_m"] = 0.0
                if h < _DOOR_MIN_HEIGHT_M:
                    new_h = min(2.1, ceiling_h - 0.05)
                    print(f"[wall_openings] RAISE door height on {op.get('wall')}: "
                          f"{h:.2f}m → {new_h:.2f}m")
                    op["height_m"] = new_h
                result.append(op)
                continue

        # Window shape enforcement (also handles doors retyped above)
        sill    = float(op.get("sill_height_m", 0.0))
        h       = float(op.get("height_m", 1.0))
        top     = sill + h
        new_sill = max(sill, _WIN_MIN_SILL_M)
        new_top  = min(top,  ceiling_h - _WIN_MIN_SOFFIT_M)
        new_h    = new_top - new_sill

        if new_h <= 0:
            print(f"[wall_openings] DROP window on {op.get('wall')}: "
                  f"height {h:.2f}m → {new_h:.2f}m after gap enforcement (out of range).")
            continue

        if new_sill != sill or new_top != top:
            print(f"[wall_openings] ADJUST window on {op.get('wall')}: "
                  f"sill {sill:.2f}→{new_sill:.2f}m, "
                  f"top {top:.2f}→{new_top:.2f}m (gap enforcement)")
        result.append({**op, "sill_height_m": new_sill, "height_m": new_h})

    return result


# keep old name as alias so any external callers still work
_enforce_window_gaps = _enforce_opening_type_shape


def _opening_wall_fractions(op: dict, room: dict) -> dict[str, float]:
    """Return wall-relative size and margin fractions for an opening."""
    W = float(room.get("floor_width_m", 5.0) or 5.0)
    D = float(room.get("floor_depth_m", 4.0) or 4.0)
    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)

    wall = op.get("wall", "")
    wall_len = W if wall in ("back", "front") else D
    wall_len = max(wall_len, 1e-9)
    ceiling_h = max(ceiling_h, 1e-9)

    width = float(op.get("width_m", 0.0))
    height = float(op.get("height_m", 0.0))
    left = float(op.get("offset_from_left_m", 0.0))
    sill = float(op.get("sill_height_m", 0.0))
    right = max(0.0, wall_len - left - width)
    top_gap = max(0.0, ceiling_h - sill - height)

    return {
        "width_frac": width / wall_len,
        "height_frac": height / ceiling_h,
        "left_gap_frac": left / wall_len,
        "right_gap_frac": right / wall_len,
        "sill_frac": sill / ceiling_h,
        "top_gap_frac": top_gap / ceiling_h,
    }


def _accept_refined_opening(prev: dict, nxt: dict, room: dict) -> dict:
    """Reject implausible refinement jumps that obviously worsen the opening."""
    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)
    reasons: list[str] = []

    if prev.get("type") == "door":
        # Doors must stay at the floor and must not shrink too much
        if float(nxt.get("sill_height_m", 0.0)) > _DOOR_MAX_SILL_M:
            reasons.append(
                f"door sill drifted to {nxt.get('sill_height_m'):.2f}m > {_DOOR_MAX_SILL_M}m"
            )
        if float(nxt.get("height_m", 2.1)) < float(prev.get("height_m", 2.1)) * 0.60:
            reasons.append(
                f"door height shrank {prev.get('height_m', 2.1):.2f}→{nxt.get('height_m', 2.1):.2f}m"
            )
        if reasons:
            print(
                f"[wall_openings] REJECT refined door on {prev.get('wall')}: "
                + "; ".join(reasons)
                + ". Keeping previous placement."
            )
            return prev
        # Snap sill to 0 regardless
        nxt = dict(nxt)
        nxt["sill_height_m"] = 0.0
        nxt["height_m"]      = max(float(nxt.get("height_m", 2.1)), _DOOR_MIN_HEIGHT_M)
        nxt["height_m"]      = min(float(nxt["height_m"]), ceiling_h - 0.05)
        return nxt

    # Windows: accept the VLM's refined placement without fraction-based guards.
    # Scale, sill, and gap adjustments can legitimately be very large when the
    # initial estimate (from scene analysis) is far off.  Geometric bounds
    # (sill >= _WIN_MIN_SILL_M, top <= ceiling - _WIN_MIN_SOFFIT_M, width > 0)
    # are enforced downstream by _enforce_opening_type_shape.
    next_frac = _opening_wall_fractions(nxt, room)
    print(
        f"[wall_openings] ACCEPT refined window on {prev.get('wall')}: "
        f"w={nxt.get('width_m', 0):.2f}m h={nxt.get('height_m', 0):.2f}m "
        f"sill={nxt.get('sill_height_m', 0):.2f}m offset={nxt.get('offset_from_left_m', 0):.2f}m "
        f"(wall_w={next_frac['width_frac']:.2f} wall_h={next_frac['height_frac']:.2f})"
    )
    return nxt


def _max_opening_delta_m(prev_ops: list[dict], next_ops: list[dict]) -> float:
    """Return the largest absolute parameter change between two opening lists."""
    if len(prev_ops) != len(next_ops):
        return float("inf")

    max_delta = 0.0
    for prev, nxt in zip(prev_ops, next_ops):
        for key in ("offset_from_left_m", "width_m", "sill_height_m", "height_m"):
            max_delta = max(max_delta, abs(float(prev.get(key, 0.0)) - float(nxt.get(key, 0.0))))
    return max_delta


# ── Room geometry helpers ─────────────────────────────────────────────────────

def wall_floor_endpoints(orient: str, W: float, D: float) -> tuple[float, float, float, float]:
    """
    Return (x0, z0, x1, z1) for each wall's floor-level endpoints, following the
    'interior-facing left' convention used by the VLM prompt:
      back  — left=x=0 side,       right=x=W side       (winding: normal +z toward interior)
      right — left=z=0 (back) side, right=z=D (front)   (winding: normal -x toward interior)
      left  — left=z=D (front),     right=z=0 (back)    (winding: normal +x toward interior)
      front — left=x=W,             right=x=0            (winding: normal -z toward interior)
    These endpoints match the winding used by build_room_mesh() in wall_mesh_gen.py.
    """
    if orient == "back":   return (0, 0, W, 0)
    if orient == "right":  return (W, 0, W, D)
    if orient == "left":   return (0, D, 0, 0)
    if orient == "front":  return (W, D, 0, D)
    raise ValueError(f"Unknown orientation: {orient!r}")


def wall_length(x0, z0, x1, z1) -> float:
    return math.sqrt((x1 - x0) ** 2 + (z1 - z0) ** 2)


# ── Mesh tessellation with holes ─────────────────────────────────────────────

def tessellate_wall_with_openings(
    x0: float, z0: float, x1: float, z1: float,
    ceiling_h: float,
    openings: list[dict],
) -> tuple[list, list]:
    """
    Build a wall mesh (vertices + triangle faces) for a wall with rectangular holes.

    The wall extends from (x0, 0, z0) to (x1, 0, z1) at floor level and up to ceiling_h.
    openings: list of dicts with keys offset_from_left_m, width_m, sill_height_m, height_m.

    The wall is tessellated into a grid of sub-quads aligned to opening boundaries.
    Each sub-quad corresponding to an opening is omitted.

    Winding follows the same convention as build_room_mesh (interior-facing normal).
    Returns (vertices, faces) using 1-based indices (for OBJ).
    """
    L = wall_length(x0, z0, x1, z1)
    if L < 1e-6:
        return [], []

    def world_pt(u: float, y: float) -> tuple[float, float, float]:
        t = u / L
        return (x0 + t * (x1 - x0), y, z0 + t * (z1 - z0))

    # Collect u-splits (along wall) and y-splits (vertical)
    u_vals: set[float] = {0.0, L}
    y_vals: set[float] = {0.0, ceiling_h}
    for op in openings:
        u_lo = float(op["offset_from_left_m"])
        u_hi = u_lo + float(op["width_m"])
        y_lo = float(op["sill_height_m"])
        y_hi = y_lo + float(op["height_m"])
        u_vals.update({max(0.0, u_lo), min(L, u_hi)})
        y_vals.update({max(0.0, y_lo), min(ceiling_h, y_hi)})

    u_splits = sorted(u_vals)
    y_splits = sorted(y_vals)

    # Build the set of (ui, yi) grid cells that are holes
    hole_cells: set[tuple[int, int]] = set()
    for op in openings:
        u_lo = float(op["offset_from_left_m"])
        u_hi = u_lo + float(op["width_m"])
        y_lo = float(op["sill_height_m"])
        y_hi = y_lo + float(op["height_m"])
        for ui in range(len(u_splits) - 1):
            uc = (u_splits[ui] + u_splits[ui + 1]) / 2
            if u_lo - 1e-6 <= uc <= u_hi + 1e-6:
                for yi in range(len(y_splits) - 1):
                    yc = (y_splits[yi] + y_splits[yi + 1]) / 2
                    if y_lo - 1e-6 <= yc <= y_hi + 1e-6:
                        hole_cells.add((ui, yi))

    vertices: list[tuple] = []
    faces:    list[tuple] = []

    for ui in range(len(u_splits) - 1):
        for yi in range(len(y_splits) - 1):
            if (ui, yi) in hole_cells:
                continue
            u0c, u1c = u_splits[ui], u_splits[ui + 1]
            y0c, y1c = y_splits[yi], y_splits[yi + 1]

            # Quad winding: BL → BR → TR → TL (interior-facing, CCW from inside)
            v_bl = world_pt(u0c, y0c)
            v_br = world_pt(u1c, y0c)
            v_tr = world_pt(u1c, y1c)
            v_tl = world_pt(u0c, y1c)

            base = len(vertices) + 1      # 1-indexed for OBJ
            vertices += [v_bl, v_br, v_tr, v_tl]
            faces.append((base, base + 1, base + 2))   # BL→BR→TR
            faces.append((base, base + 2, base + 3))   # BL→TR→TL

    return vertices, faces


# ── OBJ builder ──────────────────────────────────────────────────────────────

def build_walls_with_openings_obj(
    analysis: dict,
    openings: list[dict],
    input_dir: Path,
    out_dir: Path,
) -> Path:
    """
    Rebuild the room mesh with rectangular holes where doors/windows are.
    Writes <out_dir>/walls_with_openings.obj and returns its path.
    MTL is referenced via a relative path pointing back to <input_dir>/walls.mtl.
    """
    room      = analysis.get("room", {})
    W         = float(room.get("floor_width_m",   5.0) or 5.0)
    D         = float(room.get("floor_depth_m",   4.0) or 4.0)
    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)

    # Walls that are actually visible (detected by VLM in earlier pipeline stage).
    # Openings are only punched in these walls; non-visible walls stay solid.
    visible_wall_orients: set[str] = {
        w["orientation"] for w in analysis.get("walls", [])
        if "orientation" in w
    }
    if not visible_wall_orients:
        visible_wall_orients = {"back", "left", "right", "front"}

    # Group openings by wall orientation
    openings_by_wall: dict[str, list] = {}
    for op in openings:
        w = op.get("wall", "")
        openings_by_wall.setdefault(w, []).append(op)

    all_vertices: list[tuple] = []
    all_faces:    list[tuple] = []

    for orient in ("back", "right", "front", "left"):
        x0, z0, x1, z1 = wall_floor_endpoints(orient, W, D)
        # Only punch holes in walls that were visible in the scene analysis
        if orient not in visible_wall_orients:
            wall_ops = []
        else:
            wall_ops = openings_by_wall.get(orient, [])
        L = wall_length(x0, z0, x1, z1)
        clamped_ops = []
        for op in wall_ops:
            u_lo = float(op["offset_from_left_m"])
            u_hi = u_lo + float(op["width_m"])
            y_lo = float(op["sill_height_m"])
            y_hi = y_lo + float(op["height_m"])
            if u_lo >= L or u_hi <= 0 or y_lo >= ceiling_h or y_hi <= 0:
                print(f"[wall_openings] SKIP {orient} {op['type']}: out of wall bounds")
                continue
            clamped_ops.append({
                "offset_from_left_m": max(0.0, u_lo),
                "width_m":            min(L,   u_hi) - max(0.0, u_lo),
                "sill_height_m":      max(0.0, y_lo),
                "height_m":           min(ceiling_h, y_hi) - max(0.0, y_lo),
            })
        verts, faces = tessellate_wall_with_openings(x0, z0, x1, z1, ceiling_h, clamped_ops)
        offset = len(all_vertices)
        all_vertices.extend(verts)
        all_faces.extend((f[0] + offset, f[1] + offset, f[2] + offset) for f in faces)
        print(f"[wall_openings] {orient:5s}: {len(verts)//4} sub-quads, {len(clamped_ops)} opening(s)")

    # Floor cap
    xlo, xhi = 0.0, W
    zlo, zhi = 0.0, D
    base = len(all_vertices) + 1
    all_vertices += [
        (xlo, 0.0, zlo), (xlo, 0.0, zhi), (xhi, 0.0, zhi), (xhi, 0.0, zlo),
    ]
    all_faces += [(base, base+1, base+2), (base, base+2, base+3)]

    # Ceiling cap
    base = len(all_vertices) + 1
    all_vertices += [
        (xlo, ceiling_h, zlo), (xhi, ceiling_h, zlo),
        (xhi, ceiling_h, zhi), (xlo, ceiling_h, zhi),
    ]
    all_faces += [(base, base+1, base+2), (base, base+2, base+3)]

    # MTL lives in input_dir; reference it relative to out_dir
    mtl_src = input_dir / "walls.mtl"
    try:
        mtl_ref = str(mtl_src.relative_to(out_dir))
    except ValueError:
        # out_dir is a subdirectory of input_dir — use ../walls.mtl
        mtl_ref = "../walls.mtl"

    out_path = out_dir / "walls_with_openings.obj"
    with open(out_path, "w") as f:
        f.write("# Wall mesh with openings — wall_openings.py\n")
        f.write(f"# {len(all_vertices)} vertices, {len(all_faces)} faces\n")
        if mtl_src.exists():
            f.write(f"mtllib {mtl_ref}\n")
        f.write("\n")

        for x, y, z in all_vertices:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        f.write("\n")

        for tri in all_faces:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")

    print(f"[wall_openings] Saved → {out_path}  "
          f"({len(all_vertices)} verts, {len(all_faces)} faces)")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def _find_render(input_dir: Path) -> str | None:
    """Return the best available render PNG from the output dir.
    Prefers the VGGT-aligned render when present."""
    for name in ("render_vggt.png", "render_balanced.png", "render_yaw_floor.png",
                 "render_yaw_corner.png", "render.png"):
        p = input_dir / name
        if p.exists():
            return str(p)
    return None


def _render_openings_preview(
    analysis: dict,
    openings: list[dict],
    input_dir: Path,
    out_dir: Path,
    camera_json: Path,
    image_path: str | None,
    render_name: str = "render_openings.png",
) -> tuple[Path, Path | None]:
    """Build the punched-wall mesh and optionally render a preview image."""
    obj_path = build_walls_with_openings_obj(analysis, openings, input_dir, out_dir)

    if not camera_json.exists():
        print("[wall_openings] No camera.json found — skipping render.")
        return obj_path, None

    try:
        from floorplan.wall_line.render_room import render_room

        render_out = out_dir / render_name
        render_room(
            mesh_path=str(obj_path),
            camera_json_path=str(camera_json),
            out_path=str(render_out),
            ref_image_path=image_path,
            bg_color=(210, 230, 255),  # bright sky-like colour visible through openings
        )
        print(f"[wall_openings] Render saved → {render_out}")
        return obj_path, render_out
    except Exception as e:
        print(f"[wall_openings] Render failed: {e}")
        return obj_path, None


def _refine_openings_to_match_reference(
    image_path: str | None,
    analysis: dict,
    room: dict,
    openings: list[dict],
    input_dir: Path,
    out_dir: Path,
    camera_json: Path,
    cam_dict: dict,
) -> list[dict]:
    """Iteratively refine opening placement until the punched-hole render stabilizes."""
    if not image_path or not cam_dict or not openings:
        return openings

    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)
    current = [{**op} for op in openings]

    for iter_idx in range(1, _REFINE_MAX_ITERS + 1):
        print(f"[wall_openings] Refinement iteration {iter_idx}/{_REFINE_MAX_ITERS}...")
        _obj_path, preview_path = _render_openings_preview(
            analysis=analysis,
            openings=current,
            input_dir=input_dir,
            out_dir=out_dir,
            camera_json=camera_json,
            image_path=image_path,
            render_name=f"render_openings_refine_{iter_idx}.png",
        )
        if preview_path is None:
            break

        candidate = refine_openings_vlm_fracs(
            image_path=image_path,
            render_path=str(preview_path),
            room=room,
            cam=cam_dict,
            current_openings=current,
        )
        if not candidate:
            break

        candidate = _filter_visible_openings(candidate, cam_dict, room)
        candidate = _enforce_window_gaps(candidate, ceiling_h)
        if len(candidate) != len(current):
            print(
                "[wall_openings] Refinement produced an inconsistent opening set; "
                "keeping previous placement."
            )
            break

        delta_m = _max_opening_delta_m(current, candidate)
        print(f"[wall_openings] Refinement delta: {delta_m:.3f}m")
        current = candidate
        if delta_m <= _REFINE_EPS_M:
            print(
                f"[wall_openings] Refinement converged (<= {_REFINE_EPS_M:.2f}m change); "
                "stopping."
            )
            break

    return current


def _parse_obj_depth(obj_path: Path) -> float | None:
    """Return the max Z coordinate in an OBJ file (= actual room depth used in the mesh)."""
    try:
        z_max = None
        with open(obj_path) as fh:
            for line in fh:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        z = float(parts[3])
                        if z_max is None or z > z_max:
                            z_max = z
        return z_max
    except Exception:
        return None


def run(output_dir: str | Path, image_path: str | None = None,
        render_path: str | None = None, use_vlm: bool = True) -> Path:
    """
    Main entry point.  Returns the path to walls_with_openings.obj.

    output_dir  : existing pipeline output directory containing floorplan_analysis.json.
                  All outputs are written to <output_dir>/openings/.
    image_path  : reference image path (required for VLM call)
    render_path : path to a rendered scene PNG for comparison-based detection.
                  If None and output_dir contains a render, it is used automatically.
    use_vlm     : if True, call VLM for refined opening detection;
                  if False, use features already in floorplan_analysis.json
    """
    input_dir = Path(output_dir)
    analysis_path = input_dir / "floorplan_analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"floorplan_analysis.json not found in {input_dir}")

    out_dir = input_dir / "openings"
    out_dir.mkdir(exist_ok=True)

    # Copy walls_metadata.json into the subfolder so render_room can find textures
    # when rendering walls_with_openings.obj from this directory.
    # Texture paths inside are absolute, so the copy works without path fixup.
    import shutil as _shutil
    src_meta = input_dir / "walls_metadata.json"
    if src_meta.exists():
        _shutil.copy2(src_meta, out_dir / "walls_metadata.json")

    with open(analysis_path) as f:
        analysis = json.load(f)

    room = analysis.get("room", {})

    # Override floor_depth_m with the actual mesh depth from walls.obj.
    # The VLM analysis can underestimate room depth; the VGGT-calibrated mesh
    # uses the real depth (D_mesh). Using analysis depth leaves the front section
    # of all walls absent in walls_with_openings.obj.
    walls_obj_path = input_dir / "walls.obj"
    D_mesh = _parse_obj_depth(walls_obj_path)
    D_analysis = float(room.get("floor_depth_m", 0.0) or 0.0)
    if D_mesh and D_mesh > D_analysis:
        print(f"[wall_openings] Overriding floor_depth_m: {D_analysis:.4f}m → {D_mesh:.4f}m (from walls.obj)")
        room["floor_depth_m"] = D_mesh

    visible_walls = [w["orientation"] for w in analysis.get("walls", [])
                     if "orientation" in w]
    room["_visible_walls"] = visible_walls

    # Load camera parameters for pixel-to-wall projection.
    # Prefer the VGGT-calibrated camera (camera_vggt.json) when available;
    # fall back to the Stage-1 VLM camera (camera.json).
    cam_dict: dict = {}
    camera_json = (
        input_dir / "camera_vggt.json"
        if (input_dir / "camera_vggt.json").exists()
        else input_dir / "camera.json"
    )
    if camera_json.exists():
        cam_dict = json.loads(camera_json.read_text())
        print(f"[wall_openings] Using camera: {camera_json.name}")

    # ── Re-derive visible walls from the calibrated camera ───────────────────
    # The VLM analysis labels visible walls based on the un-aligned perspective.
    # After VGGT/Manhattan alignment the room coordinate frame can rotate so that
    # different walls face the camera.  Re-filter _visible_walls so only walls
    # whose interior face points toward the camera are kept.
    if cam_dict:
        W_cam = float(room.get("floor_width_m", 5.0) or 5.0)
        D_cam = float(room.get("floor_depth_m", 4.0) or 4.0)
        cam_pos = cam_dict.get("position_m", [0, 0, 0])
        cam_x, cam_z = float(cam_pos[0]), float(cam_pos[2])
        look_at = cam_dict.get("look_at_m", [0, 0, 0])
        look_dir = [look_at[i] - cam_pos[i] for i in range(3)]

        # Interior-facing criterion: camera is on the normal side of the wall AND
        # the look direction has a component pointing toward the wall.
        def _wall_interior_and_aimed(orient: str) -> bool:
            if orient == "back":   # z=0, normal=+z
                return cam_z > -0.1 and look_dir[2] < 0
            if orient == "front":  # z=D, normal=-z
                return cam_z < D_cam + 0.1 and look_dir[2] > 0
            if orient == "left":   # x=0, normal=+x
                return cam_x > -0.1 and look_dir[0] < 0
            if orient == "right":  # x=W, normal=-x
                return cam_x < W_cam + 0.1 and look_dir[0] > 0
            return True

        updated_visible = [w for w in visible_walls if _wall_interior_and_aimed(w)]
        if updated_visible != visible_walls:
            print(f"[wall_openings] Visible walls updated by camera geometry: "
                  f"{visible_walls} → {updated_visible}")
        room["_visible_walls"] = updated_visible
        visible_walls = updated_visible

    # ── Compute and persist wall correspondence ──────────────────────────────
    # After alignment the camera may see different walls than the pre-alignment
    # VLM analysis expected.  Compute the 2D→3D wall mapping from the calibrated
    # camera geometry and save it so every downstream VLM call knows which
    # room-coordinate label maps to which position in the rendered image.
    if cam_dict:
        correspondence = compute_wall_correspondence(cam_dict, room)
        if correspondence:
            corr_path = out_dir / "wall_correspondence.json"
            with open(corr_path, "w") as f:
                json.dump(correspondence, f, indent=2)
            print(f"[wall_openings] Wall correspondence saved → {corr_path}")
            # Also attach to room so in-process calls can use it without re-reading
            room["_wall_correspondence"] = correspondence

    # ── Opening detection ────────────────────────────────────────────────────
    analysis_openings = openings_from_analysis(analysis)

    openings: list[dict] = []

    if image_path:
        effective_render = render_path or _find_render(input_dir)
        if effective_render and cam_dict:
            if use_vlm:
                # Primary: VLM semantic analysis + brightness-diff horizontal precision
                print(f"[wall_openings] Hybrid detection (VLM + brightness-diff): {effective_render}")
                openings = detect_openings_hybrid(
                    image_path, effective_render, cam_dict, room
                )
            if not openings:
                # Fallback: brightness-diff only (no VLM server or returned nothing)
                print(f"[wall_openings] Brightness-diff only detection: {effective_render}")
                openings = detect_openings_brightness_diff(
                    image_path, effective_render, cam_dict, room
                )
        elif use_vlm:
            openings = detect_openings_vlm(image_path, room)

    if not openings:
        print("[wall_openings] Falling back to features in floorplan_analysis.json")
        openings = analysis_openings or openings_from_analysis(analysis)

    # Merge frame-divided openings: a window split into multiple panes by vertical
    # frames should be one large opening in the mesh (frames are added later).
    # Only openings separated by a gap smaller than a wall section are merged.
    openings = _merge_framed_openings(openings)

    # Drop openings whose projected position is entirely outside the camera view.
    # This catches wrong offsets from the analysis fallback (e.g. measured from
    # the wrong end of the wall) before they produce invisible holes in the mesh.
    if cam_dict:
        openings = _filter_visible_openings(openings, cam_dict, room)

    if not openings:
        print("[wall_openings] No openings detected — writing mesh without holes.")

    # Enforce minimum floor/ceiling gaps for windows
    ceiling_h = float(room.get("ceiling_height_m", 2.4) or 2.4)
    openings = _enforce_window_gaps(openings, ceiling_h)

    if use_vlm and image_path and cam_dict and openings:
        openings = _refine_openings_to_match_reference(
            image_path=image_path,
            analysis=analysis,
            room=room,
            openings=openings,
            input_dir=input_dir,
            out_dir=out_dir,
            camera_json=camera_json,
            cam_dict=cam_dict,
        )

    # Save openings data into the subfolder
    openings_path = out_dir / "openings.json"
    with open(openings_path, "w") as f:
        json.dump(openings, f, indent=2)
    print(f"[wall_openings] Openings saved → {openings_path}")

    if openings:
        print("\n[wall_openings] Openings to punch:")
        for op in openings:
            print(f"  {op['wall']:5s} {op['type']:6s}  "
                  f"w={op['width_m']:.2f}m h={op['height_m']:.2f}m  "
                  f"offset={op['offset_from_left_m']:.2f}m  "
                  f"sill={op['sill_height_m']:.2f}m")
        print()

    obj_path, _render_out = _render_openings_preview(
        analysis=analysis,
        openings=openings,
        input_dir=input_dir,
        out_dir=out_dir,
        camera_json=camera_json,
        image_path=image_path,
        render_name="render_openings.png",
    )

    return obj_path


def main():
    ap = argparse.ArgumentParser(
        description="Detect wall openings (doors/windows) via VLM and punch holes in wall mesh.")
    ap.add_argument("--output-dir", required=True,
                    help="Existing pipeline output directory (contains floorplan_analysis.json); "
                         "results are written to <output-dir>/openings/")
    ap.add_argument("--image",  default=None,
                    help="Reference image path (required for VLM call)")
    ap.add_argument("--render", default=None,
                    help="Rendered scene PNG for comparison-based detection "
                         "(auto-detected from output-dir if omitted)")
    ap.add_argument("--no-vlm", action="store_true",
                    help="Skip VLM call; use features already in floorplan_analysis.json")
    args = ap.parse_args()

    obj_path = run(
        output_dir=args.output_dir,
        image_path=args.image,
        render_path=args.render,
        use_vlm=not args.no_vlm,
    )
    print(f"\nDone → {obj_path}")


if __name__ == "__main__":
    main()
