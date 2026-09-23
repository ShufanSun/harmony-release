"""
object_placement.py — analyse scene depth and produce a back-to-front
placement order for furniture objects using the render camera and a VLM.

The script:
  1. Loads segment_results.json (furniture bboxes + types) and camera.json
     (camera position / FOV / room size).
  2. Loads openings (windows/doors) and wall-mounted object placements so the
     VLM is aware of the full already-placed context.
  3. Filters out carpet / rug — those are placed independently beforehand.
  4. Annotates the render image with numbered bounding boxes (furniture) and
     ghost outlines (openings + wall-mounted objects) so the VLM can reference
     the complete floorplan.
  5. Sends the annotated image + structured scene context to the VLM and asks
     it to reason about depth order (farthest-from-camera first) and spatial
     relationships (near left / right / back wall, centre-room, etc.).
  6. Saves the VLM's structured analysis to
     <output_dir>/furniture/placement_analysis.json.

Usage:
    python -m object_placement.furniture.object_placement \\
        --output-dir outputs/20260331_031530
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# ── Config ─────────────────────────────────────────────────────────────────────

VLM_API_URL = "http://localhost:8080/v1/chat/completions"

# Route VLM chat through the shared backend router (Qwen local / gpt-5.5 NVIDIA).
from object_placement.vlm_backend import vlm_post as _vlm_post

_SKIP_TYPES = {"carpet", "rug"}

# ── Render-resolution cap ───────────────────────────────────────────────────────
# A native camera (e.g. 6067×3467 = 21 MP) OOMs the offscreen renderer (it
# allocates a width_px×height_px framebuffer per placement iteration).  All
# render/placement functions derive their intrinsics purely from resolution
# (fx = W / (2·tan(hfov/2)), fy from height/vfov), so scaling width_px+height_px
# while keeping hfov/vfov/position/look_at auto-scales every projection and
# leaves the 3-D WORLD placement UNCHANGED (it is resolution-independent).
_PLACE_MAX_PX_DEFAULT = 8_000_000


def cap_camera_resolution(cam: dict) -> dict:
    """Return a COPY of *cam* whose render resolution is capped at
    SCENEWEAVE_PLACE_MAX_PX pixels (default 8 MP).

    Only width_px / height_px change (scaled by s = sqrt(MAX_PX/(W·H))); every
    other key — hfov_deg, vfov_deg, position_m, look_at_m, up, wall_context and
    any vggt '_'-prefixed keys — is preserved unchanged.  Because intrinsics are
    derived from width_px/height_px downstream, this scales the projection only;
    no 3-D / world-coordinate math is affected.  When already within budget the
    camera is returned unchanged.
    """
    try:
        max_px = int(os.environ.get("SCENEWEAVE_PLACE_MAX_PX", _PLACE_MAX_PX_DEFAULT))
    except (TypeError, ValueError):
        max_px = _PLACE_MAX_PX_DEFAULT
    try:
        W = int(cam["width_px"])
        H = int(cam["height_px"])
    except (KeyError, TypeError, ValueError):
        return cam
    if max_px <= 0 or W * H <= max_px:
        return cam
    s = (max_px / float(W * H)) ** 0.5
    new_cam = dict(cam)            # shallow copy preserves ALL other keys
    new_cam["width_px"]  = int(round(W * s))
    new_cam["height_px"] = int(round(H * s))
    print(f"[place] render-res cap: {W}x{H} -> "
          f"{new_cam['width_px']}x{new_cam['height_px']} "
          f"(s={s:.2f}, MAX_PX={max_px})")
    return new_cam

# Colours used to draw labelled bboxes on the annotated image.
_PALETTE = [
    (255,  80,  80),  # red
    ( 80, 180, 255),  # blue
    ( 80, 230,  80),  # green
    (255, 200,  50),  # yellow
    (200,  80, 255),  # purple
    (255, 140,  40),  # orange
    ( 60, 220, 200),  # teal
    (255, 100, 180),  # pink
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _encode_image(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _load_font(size: int = 16):
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate_render(
    render_path: Path,
    segments: list[dict],
    out_path: Path,
) -> Image.Image:
    """Draw labelled, colour-coded bounding boxes for furniture segments over
    whatever render is provided (should already include wall-mounted objects)."""
    img  = Image.open(render_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(15)

    for i, seg in enumerate(segments):
        colour = _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = seg["box_px"]
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                           outline=colour)
        label = f"{seg['index']}:{seg['type']}"
        bbox  = draw.textbbox((0, 0), label, font=font)
        tw    = bbox[2] - bbox[0]
        th    = bbox[3] - bbox[1]
        lx    = x1
        ly    = max(0, y1 - th - 4)
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=colour)
        draw.text((lx + 3, ly + 2), label, fill=(255, 255, 255), font=font)

    img.save(out_path)
    print(f"[placement] Annotated render → {out_path}")
    return img


# ── Depth heuristic ────────────────────────────────────────────────────────────

def _image_depth_order(segments: list[dict]) -> list[dict]:
    """Sort by bbox centre-y ascending (top of image = farther from camera)."""
    def _cy(seg: dict) -> float:
        x1, y1, x2, y2 = seg["box_px"]
        return (y1 + y2) / 2.0
    return sorted(segments, key=_cy)


# ── Camera-space depth computation ────────────────────────────────────────────

def _compute_camera_depth(seg: dict, cam: dict) -> float:
    """Approximate camera-space depth for a segment by back-projecting its
    bbox bottom-centre onto the floor plane.

    Returns the signed distance from the camera along its forward axis.
    Larger = farther from camera.  Falls back to a rough image-y proxy when
    the ray doesn't hit the floor.
    """
    import sys
    from pathlib import Path as _P
    _ROOT = _P(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from object_placement.wall_mounted.wall_mounted_object_placement import (
        _camera_axes, _backproject_pixel,
    )

    cam_pos  = np.array(cam["position_m"], dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],  dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    W    = int(cam["width_px"])
    H    = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    bx1, by1, bx2, by2 = seg["box_px"]
    px_img = (float(bx1) + float(bx2)) / 2.0
    py_img = float(by2)
    ray = _backproject_pixel(px_img, py_img, cam_pos, right_v, up_c_v, fwd_v,
                             fx, fx, cx, cy)
    # Intersect ray with Y=0 floor
    denom = ray[1]
    if abs(denom) < 1e-9 or denom >= 0.0:
        # Parallel / upward — fall back to inverse image-y proxy
        return 1e6 - (by1 + by2) / 2.0
    t = -cam_pos[1] / denom
    if t < 0.0:
        return 1e6 - (by1 + by2) / 2.0
    floor_pt = cam_pos + t * ray
    return float(np.dot(floor_pt - cam_pos, fwd_v))


def _reorder_placement(
    entries: list[dict],
    segments: list[dict],
    cam: dict,
    room_w: float,
    room_d: float,
) -> list[dict]:
    """Re-sort VLM placement entries by the deterministic rules:

      A. Deepest-corner object first.
      B. Wall-adjacent objects before centre objects.
      C. Farther-from-camera first (camera-space depth descending).
      D. Back wall beats left/right walls on ties.
      E. Dependencies (on_top_of / in_front_of) must precede their dependants.

    Classifications (depth / wall_affinity / dependencies) come from the VLM;
    this function only rewrites the ORDER.
    """
    seg_by_idx = {s["index"]: s for s in segments}

    # Compute per-entry depth from the segment bbox (VLM entries lack box_px
    # until we enrich them).  We also identify the "deepest corner" object:
    # the back-wall object whose camera-space depth is maximal.
    depth_map: dict[int, float] = {}
    for e in entries:
        idx = e.get("index")
        seg = seg_by_idx.get(idx)
        if seg is None:
            depth_map[idx] = -1.0
            continue
        depth_map[idx] = _compute_camera_depth(seg, cam)

    # Identify the deepest corner: the entry with the maximum camera-space
    # depth whose wall_affinity is a wall (back/left/right).  Preference for
    # "back" when ties fall within 5 % of the overall camera depth range.
    wall_entries = [e for e in entries
                    if e.get("wall_affinity") in ("back", "left", "right")]
    deepest_idx = None
    if wall_entries:
        d_sorted = sorted(wall_entries, key=lambda e: -depth_map.get(e["index"], 0))
        deepest_idx = d_sorted[0]["index"]
        top_depth  = depth_map[deepest_idx]
        for e in d_sorted[1:]:
            if top_depth - depth_map[e["index"]] < 0.05 * max(room_w, room_d):
                if e.get("wall_affinity") == "back" and d_sorted[0].get("wall_affinity") != "back":
                    deepest_idx = e["index"]
                    break
            else:
                break

    def _wall_rank(e: dict) -> int:
        wa = e.get("wall_affinity")
        if wa == "back":           return 0
        if wa in ("left", "right"): return 1
        return 2   # centre

    def _sort_key(e: dict) -> tuple:
        idx = e["index"]
        is_deepest   = 0 if idx == deepest_idx else 1   # deepest corner first
        pass_idx     = 0 if _wall_rank(e) < 2 else 1    # walls before centre
        neg_depth    = -depth_map.get(idx, 0.0)         # farther first
        wall_rank    = _wall_rank(e)                    # back > sides on ties
        return (is_deepest, pass_idx, neg_depth, wall_rank, idx)

    sorted_entries = sorted(entries, key=_sort_key)

    # ── Ablation: wo_depth_first_traversal — shuffle the placement order ──────
    import os as _abl_os
    if _abl_os.environ.get("SCENEWEAVE_ABLATE_SHUFFLE_ORDER") == "1":
        import random as _abl_rand
        _abl_rand.seed(int(_abl_os.environ.get("SCENEWEAVE_ABLATE_SEED", "0")))
        sorted_entries = list(entries)
        _abl_rand.shuffle(sorted_entries)
        print("[ablation] SHUFFLE_ORDER: furniture placed in random order (no depth-first traversal)")

    # Topological fix-up for dependencies — preserve the primary sort order
    # but defer any entry whose anchor hasn't been placed yet.
    pending = list(sorted_entries)
    placed: set[int] = set()
    final: list[dict] = []
    safety = len(pending) * 2 + 1
    while pending and safety > 0:
        safety -= 1
        progressed = False
        for i, e in enumerate(pending):
            deps: list[int] = []
            for key in ("on_top_of", "in_front_of", "behind", "facing_toward"):
                v = e.get(key)
                if v is not None:
                    try:
                        deps.append(int(v))
                    except (TypeError, ValueError):
                        pass
            if all(d in placed for d in deps if d in depth_map):
                final.append(e)
                placed.add(int(e["index"]))
                pending.pop(i)
                progressed = True
                break
        if not progressed:
            # Cycle / dangling dep — flush the remainder in primary-sort order.
            print(f"[placement] dependency cycle detected — flushing {len(pending)} entries in sort order")
            final.extend(pending)
            break

    return final


# ── Context builders ──────────────────────────────────────────────────────────

def _build_openings_context(openings: list[dict]) -> str:
    """Describe window/door openings in plain text for the VLM prompt."""
    if not openings:
        return "  (none detected)"
    lines = []
    for op in openings:
        wall   = op.get("wall", "?")
        otype  = op.get("type", "opening")
        offset = op.get("offset_from_left_m")
        width  = op.get("width_m")
        sill   = op.get("sill_height_m")
        height = op.get("height_m")
        desc = f"  {wall} wall — {otype}"
        if offset is not None and width is not None:
            desc += f"  at {offset:.2f}m from left corner, width={width:.2f}m"
        if sill is not None and height is not None:
            desc += f", sill={sill:.2f}m, top={sill + height:.2f}m"
        lines.append(desc)
    return "\n".join(lines)


def _build_wall_objects_context(wall_objects: list[dict]) -> str:
    """Describe already-placed wall-mounted objects for the VLM prompt."""
    if not wall_objects:
        return "  (none placed)"
    lines = []
    for wo in wall_objects:
        wall  = wo.get("wall", "?")
        otype = wo.get("type", "object")
        pos   = wo.get("world_pt", [])
        size  = wo.get("size_m", {})
        desc  = f"  {wall} wall — {otype}"
        if len(pos) == 3:
            desc += f"  world=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m"
        if size:
            w = size.get("width_m")
            h = size.get("height_m")
            if w and h:
                desc += f", size={w:.2f}×{h:.2f}m"
        lines.append(desc)
    return "\n".join(lines)


# ── VLM prompt ─────────────────────────────────────────────────────────────────

_ANALYSIS_PROMPT_TEMPLATE = """\
You are analysing the furniture layout of a {room_type} from a single photograph.

ROOM DIMENSIONS
  Width  (left-wall → right-wall): {room_w:.2f} m
  Depth  (camera → back-wall):     {room_d:.2f} m
  Ceiling height:                  {room_h:.2f} m

CAMERA
  Position : x={cam_x:.2f} m, y={cam_y:.2f} m (height), z={cam_z:.2f} m
  Look-at  : x={lat_x:.2f} m, y={lat_y:.2f} m,          z={lat_z:.2f} m
  The camera faces the BACK wall (z = 0, farthest).
  LEFT wall  is at x = 0.
  RIGHT wall is at x = {room_w:.2f} m.
  FRONT (near camera) is at z = {room_d:.2f} m.

{wall_image_positions}
  CRITICAL: wall_affinity is which wall the object's BACK PHYSICALLY TOUCHES — NOT
  which wall has art, windows, or other features nearby.  Use the projected wall
  positions above together with which direction the object's backrest faces to decide.

ALREADY PLACED — OPENINGS (windows / doors)
(shown as white outlines on the image; furniture must not block doors and should
respect clearance in front of windows)
{openings_context}

ALREADY PLACED — WALL-MOUNTED OBJECTS
(shown as cyan outlines on the image; furniture is often grouped in relation to
these — e.g. sofa centred under artwork, chair near a reading lamp)
{wall_objects_context}

FURNITURE OBJECTS TO PLACE (bounding boxes are drawn and labelled on the image as idx:type)
{object_list}

NOTE: Each object has a **geometry_wall_hint** computed by back-projecting its
bbox centre through the camera onto the nearest wall plane.  This is the
geometrically correct wall — use it as the PRIMARY signal for wall_affinity
(and the "wall" field for wall-mounted objects).  Only override the hint if
you are CERTAIN the image contradicts it.

IMPORTANT — OBJECT VERIFICATION
Before placing each object, verify FIVE things:

  A. **Is it actually an INDOOR object?**  Some detected segments may be
     OUTDOOR SCENERY visible through a window (trees, sky, buildings, landscape),
     or part of the room structure itself (wall, floor, ceiling, window frame,
     curtain rod).  These are NOT furniture and must be EXCLUDED.
     Set "exclude": true and "exclude_reason": "outdoor scenery" (or similar).

  B. **Is it actually FLOOR furniture?**  Some detected objects may actually be
     WALL-MOUNTED items (e.g. a macramé wall hanging detected as "bookcase",
     a fabric art piece, a wall shelf, a wall clock).  If an object is clearly
     HANGING ON A WALL and not standing on the floor, set its wall_affinity to
     "wall_mounted" — it will be excluded from floor placement.

  C. **Is the type label correct?**  The auto-detection may mislabel objects
     (e.g. a desk labelled as "coffee_table", a side table labelled as "sofa").
     Override the "type" field with the correct type based on what you actually
     see in the image.  Common corrections:
       • A table with a chair in front of it against a wall → "desk" not "coffee_table"
       • A wall hanging / tapestry / macramé → "wall_art" not "bookcase"
       • A bench → "bench" not "sofa"
       • Trees / foliage through a window → EXCLUDE (not a real object)
       • Part of the curtain / window frame → EXCLUDE

  D. **Is it a small DECORATIVE object sitting ON TOP OF furniture?**
     Small items resting on a surface — potted plants on a shelf, vases on a
     table, books on a cabinet, decorative objects on a console — are NOT
     independent floor furniture.  They are part of the supporting furniture's
     visual texture and will be included in the inpainting of that surface.
     EXCLUDE these with "exclude": true and "exclude_reason": "decorative
     object on surface" (or similar).
     Only include a plant/object if it is LARGE AND STANDING ON THE FLOOR
     independently (e.g. a tall floor plant, a floor lamp).

  E. **Is this a FRAGMENT of a larger object?**  The detector sometimes splits
     one piece of furniture into multiple overlapping or adjacent bounding boxes
     (e.g. a long cabinet → 4 separate "cabinet" detections at different
     horizontal positions).  If multiple same-type segments at similar vertical
     positions clearly depict parts of the SAME PHYSICAL OBJECT, keep ONLY the
     one with the largest bounding box and EXCLUDE the rest with "exclude": true
     and "exclude_reason": "fragment of idx=<kept_index>" (or similar).

GEOMETRIC DEPTH PRIOR
Objects with a smaller image-y for their bbox centre appear higher in the
image and are therefore farther from the camera (perspective projection).
Rough order, farthest first:
{depth_prior}

TASK — describe each object relationally, then order them for placement.

For each object, use RELATIONAL vocabulary: say what the object is **facing
towards**, what its **back is against**, what it sits **next to**, and what
it is **in front of**.  Think of the scene the way an interior designer would:

  1. **depth**          — "far" (near back wall), "mid" (middle of room),
                          or "close" (near camera).
  2. **wall_affinity**  — "back", "left", "right", or "centre".  A wall
                          affinity means the object's BACK IS PHYSICALLY
                          TOUCHING that wall (or flush along it).
                          Determine this by the object's HORIZONTAL POSITION
                          in the image and which direction its backrest/back
                          panel faces — NOT by what decorative features are
                          on the wall nearby.
                          • Object skewed to left side of image, backrest
                            facing left → wall_affinity = "left"
                          • Object skewed to right side of image, backrest
                            facing right → wall_affinity = "right"
                          • Object centered, facing camera directly →
                            wall_affinity = "back"
                          VISIBLE-FACE RULE (strongest cue — overrides position):
                          An object at the far LEFT or RIGHT edge of the image
                          can still be against the BACK wall (sitting at the
                          left/right END of it).  Decide by which face you see:
                            • You see its FRONT straight-on (cabinet shelves,
                              doors, drawers, or a sofa's seat cushions face the
                              CAMERA) → it is against the BACK wall →
                              wall_affinity = "back", even if it sits at the
                              image edge.
                            • You see its SIDE / profile edge-on (shelves face
                              left or right, you look ALONG the object's depth)
                              → it is against a SIDE wall → "left"/"right".
                          Use the geometry_wall_hint (from bbox position) as a
                          secondary check; the visible-face rule wins when they
                          disagree.
  3. **opening_relation** — if the object sits against a wall with a
                          window/door, describe its position FACING that
                          opening, e.g.:
                            • "centred under the back-wall window, facing the
                               camera"
                            • "back against the left wall, next to the
                               left-wall window"
                            • "in the back-right corner, back against the
                               right wall, facing into the room"
                          Use null if no opening is nearby.
  4. **group**          — a short label if the object belongs to a grouping
                          (e.g. "coffee table in front of sofa", "side chair
                          next to fireplace").

     FUNCTIONAL GROUPINGS AND FACING DIRECTION:
     Objects that form a functional set should be described together.
     The **notes** field MUST state which direction each object faces.

     DESK + CHAIR rule:
       • The DESK is the anchor — it sits flush against a wall (wall_affinity
         = the wall it's against).  Place the desk FIRST.
       • The CHAIR faces TOWARD the desk:
           - chair's in_front_of = desk index
           - chair's facing_toward = desk index
           - chair's wall_affinity = "centre" (it is NOT against a wall)
       • NEVER set the desk as in_front_of the chair — the desk is the
         fixed furniture, the chair is pulled up to it.

     Other common groupings:
       • A coffee table in front of a sofa → faces same direction as the sofa
       • Dining chairs around a table → each faces TOWARD the table centre
         (facing_toward = table index)
       • Two sofas facing each other → note the facing relationship
       • A TV stand → faces the seating area
     The downstream placement uses facing_toward to orient the 3D model,
     so "facing_toward": <desk_idx> is critical information.

  5. **on_top_of**      — index of the object this rests ON TOP OF (plant on
                          a coffee table, lamp on a side table).  null if on
                          the floor.
  6. **in_front_of**    — index of the object this sits IN FRONT OF (a coffee
                          table in front of a sofa, a foot-stool in front of
                          an armchair).  The anchor object must be placed
                          first.  null if not in front of anything.
                          IMPORTANT: "in front of" means BETWEEN the anchor
                          object and the CAMERA (closer to the viewer).
                          If a small table's bbox centre is ABOVE (higher in
                          the image / farther from camera) a sofa's bbox
                          centre, that table is BEHIND the sofa (between the
                          sofa and the wall), NOT in front of it.  Use
                          "behind" in the notes and set in_front_of to null.
  7. **behind**         — index of the object this sits BEHIND (between the
                          anchor and the wall behind it).  A side table whose
                          image-y bbox centre is smaller (higher/farther)
                          than the sofa's is behind the sofa.  null if not
                          behind anything.
  8. **notes**          — one sentence using relational vocabulary ("facing
                          towards…", "back against…", "next to…", "in front
                          of…", "behind…") to describe the object's role.
  9. **count**          — the TOTAL number of clearly-IDENTICAL instances of
                          this object visible in the scene that form one set
                          (e.g. a set of 4 matching dining chairs around a
                          table → count = 4 on the chair entry).  Segmentation
                          often captures only ONE of several identical items
                          (the rest are occluded behind a table, merged, or
                          missed); report the TRUE total you can see so the
                          missing copies can be filled in by reusing this
                          object's 3-D model.  Count ONLY items of the SAME
                          design (a matching set), never just the same
                          category — two different armchairs are count = 1
                          each.  Use 1 for a unique single object.  Put the
                          full set count on EVERY segmented member of the set
                          (if two of the four chairs were segmented, both get
                          count = 4).

PLACEMENT ORDER — the ORDER decides which object is laid down first so that
subsequent objects can be snapped against already-placed geometry without
depth ambiguities from occlusion.

  Rule A — DEEPEST CORNER FIRST.  The single object that sits in the
           deepest wall CORNER of the room (the corner farthest from the
           camera) is placed before anything else.

  Rule B — WALL OBJECTS before room objects.  Every object whose
           wall_affinity is "back", "left", or "right" is placed before ANY
           object whose wall_affinity is "centre".

  Rule C — FARTHER FIRST.  Within each group, sort by distance from the
           camera — the object whose back is against the farther end of
           its wall goes first.  For the left/right walls this means the
           back end (near the back wall) is placed first, then progress
           toward the camera.  For the back wall, the corner closer to the
           deepest scene corner is placed first.

  Rule D — BACK WALL beats SIDE walls.  If two wall objects are at
           comparable camera distances, the one against the back wall is
           placed first.

  Rule E — DEPENDENCIES.  Any object with "on_top_of", "in_front_of",
           or "behind" MUST appear AFTER the object it depends on.  This
           overrides every other rule when they conflict.

  Rule F — INCLUDE ALL OBJECTS.  Every segment index listed above MUST
           appear exactly once in placement_order.  Do NOT drop any object.
           If two segments look like duplicates of the same physical object,
           still include both — deduplication is handled downstream.

OUTPUT — respond with ONLY this JSON object (no markdown fences, no extra text):
{{
  "placement_order": [
    {{
      "index": <segment index int>,
      "type": "<type string>",
      "exclude": false,
      "exclude_reason": null,
      "depth": "far" | "mid" | "close",
      "wall_affinity": "back" | "left" | "right" | "centre" | "wall_mounted",
      "wall": "<which wall this object is on: 'back', 'left', or 'right'.  REQUIRED when wall_affinity is 'wall_mounted'.  null for centre/floor objects>",
      "on_top_of": <index of the object this rests on, or null>,
      "in_front_of": <index of the object this is placed in front of, or null>,
      "facing_toward": <index of the object this faces toward, or null — e.g. chair facing desk, dining chair facing table>,
      "behind": <index of the object this sits behind (between it and wall), or null>,
      "opening_relation": "<relational position phrase, or null>",
      "group": "<brief group label or null>",
      "notes": "<one sentence using 'facing towards / back against / next to / in front of / behind'>",
      "count": <int total of identical instances in this object's matching set (1 if unique); same value on every segmented member of the set>
    }}
  ],
  "scene_summary": "<2-3 sentences describing the layout using relational vocabulary>"
}}
"""


def _project_x_frac(pt: np.ndarray, cam: dict) -> float | None:
    """Project a 3D point to image-x fraction [0,1]. Returns None if behind camera."""
    from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes
    cam_pos  = np.array(cam["position_m"], dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],  dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, _, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    W     = int(cam["width_px"])
    hfov  = float(cam["hfov_deg"])
    fx    = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    d     = np.array(pt, dtype=np.float64) - cam_pos
    depth = float(np.dot(d, fwd_v))
    if depth < 0.1:
        return None
    x_img = fx * float(np.dot(d, right_v)) / depth + W / 2.0
    return x_img / W


def _x_frac_to_side(f: float | None) -> str:
    if f is None:
        return "not visible from camera"
    if f < 0.33:
        return "LEFT side of the image"
    if f > 0.67:
        return "RIGHT side of the image"
    return "CENTER of the image"


def _describe_wall_positions(cam: dict, room_w: float, room_d: float,
                              room_h: float = 2.7) -> str:
    """Return a camera-accurate description of where each wall appears in the image."""
    mid_h = room_h / 2.0
    back_f  = _project_x_frac(np.array([room_w / 2.0, mid_h, 0.0]),     cam)
    left_f  = _project_x_frac(np.array([0.0,          mid_h, room_d / 2.0]), cam)
    right_f = _project_x_frac(np.array([room_w,       mid_h, room_d / 2.0]), cam)

    back_side  = _x_frac_to_side(back_f)
    left_side  = _x_frac_to_side(left_f)
    right_side = _x_frac_to_side(right_f)

    lines = [
        "HOW TO IDENTIFY EACH WALL FROM THE PHOTOGRAPH",
        f"  • LEFT wall  (x=0)        → its centre projects to the {left_side}.",
        f"                               Objects against it have their back on the left (x=0) side, facing +X into the room.",
        f"  • RIGHT wall (x={room_w:.1f}m) → its centre projects to the {right_side}.",
        f"                               Objects against it have their back on the right (x={room_w:.1f}m) side, facing -X into the room.",
        f"  • BACK wall  (z=0)        → its centre projects to the {back_side}.",
        f"                               Objects against it face the camera (+Z).",
        "  CRITICAL: use the projected wall positions above — NOT raw image-left/right",
        "  heuristics — to assign wall_affinity.  A sofa appearing on the right side of",
        f"  the image may be against the BACK wall if the back wall projects to the {back_side}.",
        "  Use the object's bbox horizontal centre relative to each wall's projected",
        "  position, plus which direction its backrest faces, to decide.",
    ]
    return "\n".join(lines)


def _footprint_spans(seg: dict, cam: dict) -> tuple[float, float, float] | None:
    """Back-project the bbox's two bottom corners to the floor (y=0) and return
    (span_x, span_z, mean_z) of the resulting footprint segment.

    span_x ≫ span_z  → the object's base runs parallel to the back/front wall.
    span_z ≫ span_x  → the base runs along a side wall.
    Returns None if either ray fails to hit the floor in front of the camera or
    the depth lands implausibly far outside the room.
    """
    from object_placement.wall_mounted.wall_mounted_object_placement import (
        _camera_axes, _backproject_pixel,
    )
    bx1, by1, bx2, by2 = seg["box_px"]
    W = int(cam["width_px"]); H = int(cam["height_px"])
    cam_pos = np.array(cam["position_m"], dtype=np.float64)
    look_at = np.array(cam["look_at_m"], dtype=np.float64)
    up_w    = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_w)
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx_px, cy_px = W / 2.0, H / 2.0
    pts = []
    for px in (float(bx1), float(bx2)):
        ray = _backproject_pixel(px, float(by2), cam_pos,
                                 right_v, up_c_v, fwd_v, fx, fx, cx_px, cy_px)
        if abs(ray[1]) < 1e-6:
            return None
        t = (0.0 - cam_pos[1]) / ray[1]
        if t <= 0:
            return None
        pts.append(cam_pos + t * ray)
    span_x = abs(float(pts[0][0] - pts[1][0]))
    span_z = abs(float(pts[0][2] - pts[1][2]))
    mean_z = (float(pts[0][2]) + float(pts[1][2])) / 2.0
    return span_x, span_z, mean_z


def _footprint_strongly_parallel_to_back(
    seg: dict, cam: dict, room_d: float,
) -> bool:
    """True when the object's floor base runs clearly PARALLEL to the back wall
    (span_x dominates span_z by a wide margin) AND sits in the back half of the
    room.  This is a high-confidence "against the back wall" signal that is safe
    to use to override the VLM's left/right corner guess — a genuine side-wall
    object would have its base running along Z (span_z dominant) instead.
    """
    spans = _footprint_spans(seg, cam)
    if spans is None:
        return False
    span_x, span_z, mean_z = spans
    # Strong parallel-to-back: span_x at least 3× span_z (base nearly constant
    # depth), and the footprint lands in the back ~60% of the room.
    parallel = span_x > max(span_z * 3.0, 0.15)
    in_back  = mean_z <= max(room_d * 0.6, 1.0)
    return parallel and in_back


_SEATING_TYPES_SIDE = {"sofa", "couch", "loveseat", "sectional", "armchair",
                       "chair", "chaise_lounge", "bench"}


def _seating_side_wall_from_edge(seg: dict, cam: dict, obj_type: str):
    """Detect a SIDE-wall (left/right) seating piece under a near-head-on camera.

    When the camera faces the back wall, a sofa/chair placed AGAINST A SIDE WALL
    (perpendicular, facing into the room) appears at the far left/right image
    edge as a TALL-NARROW box — the camera sees its END/arm, not its long front.
    A back-wall sofa instead appears WIDE.  Both the VLM and the geometric
    centre-ray mistake the side-wall piece for a back-wall one (the back wall
    fills the frame), so use this bbox aspect+edge cue.  Returns 'left'/'right'
    or None.
    """
    if obj_type.lower().replace("-", "_") not in _SEATING_TYPES_SIDE:
        return None
    box = seg.get("box_px")
    W = float(cam.get("width_px", 0) or 0)
    if not box or W <= 0:
        return None
    x1, _y1, x2, _y2 = box
    bw = max(1.0, float(x2) - float(x1))
    bh = max(1.0, float(_y2) - float(_y1))
    aspect = bw / bh                     # <1.3 ⇒ end-on (not a wide front view)
    tol = 0.02 * W
    touch_left  = x1 <= tol
    touch_right = x2 >= W - tol
    if aspect < 1.3:
        if touch_right and not touch_left:
            return "right"
        if touch_left and not touch_right:
            return "left"
    return None


def _wall_from_center_ray(
    seg: dict,
    cam: dict,
    room_w: float,
    room_d: float,
    room_h: float,
) -> str | None:
    """Identify which geometric wall an object's BACK is against by casting a ray
    through its bbox CENTRE and taking the nearest wall plane it strikes inside
    the room footprint.

    This is more robust than the floor-footprint heuristic for large objects in
    diagonal corner views — the footprint back-projection of a big sofa lands
    mid-room and reads as "centre", but the centre ray passes through the object
    and cleanly hits the wall behind it.

    Walls the camera sits OUTSIDE of (and therefore looks *through* — e.g. the
    front wall when the camera is just outside the doorway) are excluded, since
    the ray would spuriously strike them first.  Returns "back"/"left"/"right",
    or None when no confident in-room wall is hit.
    """
    from object_placement.wall_mounted.wall_mounted_object_placement import (
        _camera_axes, _backproject_pixel,
    )
    bx1, by1, bx2, by2 = seg["box_px"]
    W = int(cam["width_px"]); H = int(cam["height_px"])
    cam_pos = np.array(cam["position_m"], dtype=np.float64)
    look_at = np.array(cam["look_at_m"], dtype=np.float64)
    up_w    = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_w)
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx_px, cy_px = W / 2.0, H / 2.0

    pxc = (float(bx1) + float(bx2)) / 2.0
    pyc = (float(by1) + float(by2)) / 2.0
    ray = _backproject_pixel(pxc, pyc, cam_pos, right_v, up_c_v, fwd_v,
                             fx, fx, cx_px, cy_px)
    M = 0.4  # in-room tolerance on the non-plane axes

    # (name, axis, plane-value).  Skip a wall if the camera is OUTSIDE the room
    # on that wall's side — it is being looked through, not at.
    candidates = [("back", 2, 0.0), ("left", 0, 0.0), ("right", 0, room_w)]
    if cam_pos[2] < 0.0:            candidates = [c for c in candidates if c[0] != "back"]
    if cam_pos[0] < 0.0:            candidates = [c for c in candidates if c[0] != "left"]
    if cam_pos[0] > room_w:         candidates = [c for c in candidates if c[0] != "right"]

    hits: list[tuple[float, str]] = []
    for name, axis, val in candidates:
        d = ray[axis]
        if abs(d) < 1e-9:
            continue
        t = (val - cam_pos[axis]) / d
        if t <= 0:
            continue
        p = cam_pos + t * ray
        if not (-M <= p[1] <= room_h + M):
            continue
        other = 0 if axis == 2 else 2
        lim = room_w if other == 0 else room_d
        if not (-M <= p[other] <= lim + M):
            continue
        hits.append((t, name))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    return hits[0][1]


def _compute_wall_hint(
    seg: dict,
    cam: dict,
    room_w: float,
    room_d: float,
) -> str:
    """Estimate which wall the object's back is against.

    Primary signal: bbox horizontal edge position in the image.
    - An object whose bbox extends into the far RIGHT of the image (right edge >
      75 % of image width, centre > 45 %) likely has its back against the RIGHT wall.
    - An object whose bbox extends into the far LEFT of the image (left edge <
      25 % of image width, centre < 55 %) likely has its back against the LEFT wall.
    - Otherwise: back-project bbox centre ray to back/left/right planes and pick
      the nearest hit (almost always the back wall for centred objects).

    Returns "back", "left", "right", or "centre".
    """
    from object_placement.wall_mounted.wall_mounted_object_placement import (
        _camera_axes, _backproject_pixel,
    )

    bx1, by1, bx2, by2 = seg["box_px"]
    W = int(cam["width_px"])
    H = int(cam["height_px"])

    bx2_frac = float(bx2) / W
    bx1_frac = float(bx1) / W

    # Horizontal edge heuristic — catches side-wall objects that the ray-cast misses
    # because a large object's bbox centre may project onto the back wall even when
    # the object is clearly hugging a side wall.
    # Use PROJECTED wall x-fractions instead of hardcoded 0.25/0.75 so the threshold
    # is correct for cameras that are not facing straight at the back wall.
    cam_pos  = np.array(cam["position_m"], dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],  dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx_px, cy_px = W / 2.0, H / 2.0

    # ── Gap gate: is the object actually NEAR a wall, or free-standing? ──────────
    # The ray-cast and edge heuristics below ALWAYS resolve to some wall plane —
    # every center ray into a closed box eventually strikes one — so a foreground
    # piece (an accent chair in a conversation group, a centred coffee table) gets
    # falsely "wall-bound". Fix: back-project the bbox BOTTOM-centre (the object's
    # floor contact) onto the floor plane y=0 to recover its real (x,z) footprint,
    # then measure clearance to each wall. If it floats off the back wall AND off
    # both side walls, it is free-standing → return "centre" (no wall snap). Only a
    # footprint that genuinely hugs a wall falls through to the wall-picking logic.
    BACK_GAP_M = 1.2      # back-wall furniture: floor-front sits within ~its depth
    SIDE_GAP_M = 0.6      # side-wall furniture: footprint centre within ~half-depth
    px_b = (float(bx1) + float(bx2)) / 2.0
    py_b = float(by2)                                  # bottom edge = floor contact
    ray_b = _backproject_pixel(px_b, py_b, cam_pos, right_v, up_c_v, fwd_v,
                               fx, fx, cx_px, cy_px)
    if abs(ray_b[1]) > 1e-6:
        t_floor = (0.0 - cam_pos[1]) / ray_b[1]        # intersect floor plane y=0
        if t_floor > 0:
            fp = cam_pos + t_floor * ray_b
            x0, z0 = float(fp[0]), float(fp[2])
            # Only trust the footprint if it lands inside the room envelope.
            if -1.0 <= x0 <= room_w + 1.0 and 0.0 <= z0 <= room_d + 1.0:
                g_back, g_left, g_right = abs(z0), abs(x0), abs(room_w - x0)
                if g_back > BACK_GAP_M and g_left > SIDE_GAP_M and g_right > SIDE_GAP_M:
                    print(f"    [wall_hint] idx={seg.get('index')} footprint "
                          f"(x={x0:.2f},z={z0:.2f}) floats off all walls "
                          f"(gaps back={g_back:.2f} L={g_left:.2f} R={g_right:.2f}) "
                          f"→ centre")
                    return "centre"

    # Project left/right wall centres to get camera-correct thresholds.
    def _proj_x(pt3: list) -> float | None:
        d = np.array(pt3, dtype=np.float64) - cam_pos
        depth = float(np.dot(d, fwd_v))
        if depth < 0.1:
            return None
        return (fx * float(np.dot(d, right_v)) / depth + cx_px) / W

    mid_h = 1.35
    left_wall_xf  = _proj_x([0.0,    mid_h, room_d / 2.0])
    right_wall_xf = _proj_x([room_w, mid_h, room_d / 2.0])

    # ── Footprint-orientation discriminator ────────────────────────────────────
    # The edge heuristic below can't tell a true side-wall object from a back-wall
    # object sitting at the LEFT/RIGHT END of the back wall — both project to the
    # image edge.  Resolve it geometrically: back-project the two bottom corners
    # of the bbox to the floor plane and compare how the footprint runs.
    #   • runs along X (width) → it lies flat against the back (or front) wall → "back"
    #   • runs along Z (depth) → it lies along a side wall → keep side classification
    # Returns "back" when the footprint clearly runs along X, else None (undecided).
    def _footprint_runs_along_x() -> bool | None:
        ray_l = _backproject_pixel(float(bx1), float(by2), cam_pos,
                                   right_v, up_c_v, fwd_v, fx, fx, cx_px, cy_px)
        ray_r = _backproject_pixel(float(bx2), float(by2), cam_pos,
                                   right_v, up_c_v, fwd_v, fx, fx, cx_px, cy_px)
        pts = []
        for ray in (ray_l, ray_r):
            if abs(ray[1]) < 1e-6:
                return None
            t = (0.0 - cam_pos[1]) / ray[1]
            if t <= 0:
                return None
            p = cam_pos + t * ray
            # Only the DEPTH (z) needs to land near the room — a back-wall object
            # at the left/right END can project slightly beyond the room's X
            # extent when room_w / camera yaw is imperfect, so don't reject on X.
            if not (-1.5 <= p[2] <= room_d + 1.5):
                return None
            pts.append(p)
        span_x = abs(float(pts[0][0] - pts[1][0]))
        span_z = abs(float(pts[0][2] - pts[1][2]))
        # The discriminating signal is whether the base runs parallel to the
        # back wall (constant depth, span_z≈0) or along a side wall (span_x≈0).
        if span_x > span_z * 1.3:
            return True   # base runs along X → back/front wall
        if span_z > span_x * 1.3:
            return False  # base runs along Z → side wall
        return None  # roughly diagonal / ambiguous — defer to edge + ray-cast

    if left_wall_xf is not None and right_wall_xf is not None:
        # Threshold at midpoints between each wall projection and the image edge.
        right_thresh_lo = (right_wall_xf + 1.0) / 2.0   # halfway between right-wall and right edge
        right_thresh_hi = right_wall_xf - 0.05           # just inside right wall projection
        left_thresh_hi  = (left_wall_xf + 0.0) / 2.0    # halfway between left edge and left-wall
        left_thresh_lo  = left_wall_xf + 0.05

        if bx2_frac > right_thresh_lo and bx1_frac > right_thresh_hi:
            if _footprint_runs_along_x() is True:
                print(f"    [wall_hint] idx={seg.get('index')} near right edge but "
                      f"footprint runs along X → back wall (end of back wall)")
                return "back"
            return "right"
        if bx1_frac < left_thresh_hi and bx2_frac < left_thresh_lo:
            if _footprint_runs_along_x() is True:
                print(f"    [wall_hint] idx={seg.get('index')} near left edge but "
                      f"footprint runs along X → back wall (end of back wall)")
                return "back"
            return "left"
    else:
        # Fallback to fixed thresholds when projection fails
        if bx2_frac > 0.75 and bx1_frac > 0.50:
            if _footprint_runs_along_x() is True:
                return "back"
            return "right"
        if bx1_frac < 0.25 and bx2_frac < 0.50:
            if _footprint_runs_along_x() is True:
                return "back"
            return "left"

    px_c = (float(bx1) + float(bx2)) / 2.0
    py_c = (float(by1) + float(by2)) / 2.0
    ray = _backproject_pixel(px_c, py_c, cam_pos, right_v, up_c_v, fwd_v,
                             fx, fx, cx_px, cy_px)

    hits: list[tuple[str, float]] = []
    # Back wall: Z = 0
    if abs(ray[2]) > 1e-9:
        t = (0.0 - cam_pos[2]) / ray[2]
        if t > 0:
            pt = cam_pos + t * ray
            if -0.5 <= pt[0] <= room_w + 0.5 and -0.5 <= pt[1] <= 4.0:
                hits.append(("back", t))
    # Left wall: X = 0
    if abs(ray[0]) > 1e-9:
        t = (0.0 - cam_pos[0]) / ray[0]
        if t > 0:
            pt = cam_pos + t * ray
            if -0.5 <= pt[2] <= room_d + 0.5 and -0.5 <= pt[1] <= 4.0:
                hits.append(("left", t))
        # Right wall: X = room_w
        t = (room_w - cam_pos[0]) / ray[0]
        if t > 0:
            pt = cam_pos + t * ray
            if -0.5 <= pt[2] <= room_d + 0.5 and -0.5 <= pt[1] <= 4.0:
                hits.append(("right", t))

    if not hits:
        return "centre"
    hits.sort(key=lambda h: h[1])
    return hits[0][0]


def _build_object_list(segments: list[dict], cam: dict | None = None,
                       room_w: float = 0, room_d: float = 0) -> str:
    lines = []
    for seg in segments:
        x1, y1, x2, y2 = seg["box_px"]
        hint = ""
        if cam is not None and room_w > 0 and room_d > 0:
            wall = _compute_wall_hint(seg, cam, room_w, room_d)
            hint = f"  geometry_wall_hint={wall}"
        lines.append(
            f"  idx={seg['index']}  type={seg['type']:<15}  "
            f"box=[{x1},{y1},{x2},{y2}]  gdino_score={seg.get('gdino_score', '?')}"
            f"{hint}"
        )
    return "\n".join(lines)


def _build_depth_prior(ordered: list[dict]) -> str:
    lines = []
    for rank, seg in enumerate(ordered, 1):
        x1, y1, x2, y2 = seg["box_px"]
        cy = (y1 + y2) / 2.0
        lines.append(
            f"  {rank}. idx={seg['index']} {seg['type']}  "
            f"bbox_centre_y={cy:.0f}px"
        )
    return "\n".join(lines)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _vlm_back_wall(photo_img, crop_img, obj_type: str, idx) -> "str | None":
    """Ask the VLM which wall an object's back/headboard is flush against, read
    directly from the photo. Authoritative for beds (headboard) and cabinet-like
    storage (back), where geometry + front-detect can't reliably tell (corner
    beds, partial/occluded silhouettes). Returns 'back'|'left'|'right'|'centre'|None."""
    role = ("headboard (the tall padded end)" if "bed" in obj_type
            else "back (the flat side, opposite the doors/drawers/shelves)")
    prompt = (
        f"Look at the {obj_type} in this room photograph"
        + (" (a close-up crop is also provided)." if crop_img is not None else ".")
        + f"\nWhich wall is its {role} placed flush against?\n"
        "- back   = the far wall that faces the camera\n"
        "- left   = the wall on the left side of the room\n"
        "- right  = the wall on the right side of the room\n"
        "- centre = free-standing in the middle, not against any wall\n"
        "Reason from where the piece physically sits. Answer with ONLY one word: "
        "back, left, right, or centre."
    )
    try:
        pb64, pmime = _encode_image(photo_img)
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": f"data:{pmime};base64,{pb64}"}}]
        if crop_img is not None:
            cb64, cmime = _encode_image(crop_img)
            content.append({"type": "text", "text": f"Close-up of the {obj_type}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:{cmime};base64,{cb64}"}})
        payload = {"model": "qwen3",
                   "messages": [{"role": "user", "content": content}],
                   "max_tokens": 300,
                   "chat_template_kwargs": {"enable_thinking": True}}
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"]).strip().lower()
        found = None
        for tok in raw.replace(",", " ").replace(".", " ").replace("'", " ").split():
            if tok in ("back", "left", "right", "centre", "center"):
                found = "centre" if tok == "center" else tok
        return found
    except Exception as e:
        print(f"  [back_wall_vlm] idx={idx} {obj_type}: failed ({e})")
        return None


def run(output_dir: str | Path, no_wall_reasoning: bool = False) -> Path:
    out_root    = Path(output_dir)
    furn_dir    = out_root / "furniture"
    results_p   = furn_dir / "segment_results.json"
    camera_p    = (out_root / "camera_vggt.json"
                   if (out_root / "camera_vggt.json").exists()
                   else out_root / "camera.json")
    floorplan_p = out_root / "floorplan_analysis.json"
    openings_p  = out_root / "openings" / "openings.json"
    wm_placements_p = out_root / "wall_mounted" / "placements" / "object_placements.json"
    wm_openings_p   = out_root / "wall_mounted" / "placements" / "opening_placements.json"

    # Use the render that already has wall-mounted objects placed; fall back to
    # the plain final render if the wall-mounted pipeline hasn't been run yet.
    wm_render_p = out_root / "wall_mounted" / "placements" / "render_objects_placed.png"
    render_p    = wm_render_p if wm_render_p.exists() else out_root / "render_final.png"

    for p in (results_p, camera_p, render_p):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print(f"[placement] Using render: {render_p}")

    # ── Load data ─────────────────────────────────────────────────────────
    with open(results_p) as f:
        seg_data = json.load(f)
    with open(camera_p) as f:
        cam = json.load(f)

    # Openings: prefer the detailed opening_placements.json (has bbox_px for annotation);
    # fall back to the raw openings.json (no bbox_px but has wall positions).
    openings: list[dict] = []
    if wm_openings_p.exists():
        with open(wm_openings_p) as f:
            openings = json.load(f)
        print(f"[placement] Loaded {len(openings)} opening(s) from wall_mounted placements.")
    elif openings_p.exists():
        with open(openings_p) as f:
            openings = json.load(f)
        print(f"[placement] Loaded {len(openings)} opening(s) from openings.json.")
    else:
        print("[placement] No openings file found — skipping opening context.")

    # Wall-mounted objects
    wall_objects: list[dict] = []
    if wm_placements_p.exists():
        with open(wm_placements_p) as f:
            wall_objects = json.load(f)
        print(f"[placement] Loaded {len(wall_objects)} wall-mounted object(s).")
    else:
        print("[placement] No wall-mounted placements file found — skipping.")

    room_w = room_d = room_h = None
    room_type = "living room"
    # Prefer the ACTUAL geometry box (walls.obj) for room dimensions: the wall-hint
    # back-projection casts onto these exact planes (x=0..room_w, z=0..room_d), so
    # they MUST match the placed box. The floorplan VLM estimate is often wrong or
    # incomplete (e.g. floor_width_m=4.0, floor_depth_m=0.0 while the real box is
    # 6.7×6.3), which mislabels corner pieces' wall_affinity.
    _walls_obj = out_root / "walls.obj"
    if _walls_obj.exists():
        try:
            _v = np.array([[float(x) for x in _l.split()[1:4]]
                           for _l in open(_walls_obj) if _l.startswith("v ")])
            room_w = float(_v[:, 0].max() - _v[:, 0].min())
            room_d = float(_v[:, 2].max() - _v[:, 2].min())
            print(f"[placement] room dims from walls.obj box: "
                  f"{room_w:.2f}×{room_d:.2f}m")
        except Exception as _we:
            print(f"[placement] walls.obj parse failed ({_we}) — using floorplan dims")
            room_w = room_d = None
    if floorplan_p.exists():
        with open(floorplan_p) as f:
            fp = json.load(f)
        room    = fp.get("room", {})
        if room_w is None:
            room_w = float(room.get("floor_width_m",  4.5))
        if room_d is None:
            room_d = float(room.get("floor_depth_m",  4.0))
        room_h  = float(room.get("ceiling_height_m", 2.7))
        room_type = room.get("room_type", "living_room").replace("_", " ")
    else:
        wc = cam.get("wall_context", {})
        if room_w is None:
            room_w = float(wc.get("back", {}).get("length_m", 4.5))
        if room_d is None:
            room_d = float(wc.get("left", {}).get("length_m", 4.0))
    if room_h is None:
        room_h = 2.7

    # room_d = 0 breaks geometry_wall_hint ray-casting and wall projection.
    # Fall back to camera z-position (camera is roughly at the front wall).
    if room_d <= 0.1:
        cam_z_pos = float(cam["position_m"][2])
        room_d = max(cam_z_pos * 1.25, 2.5)
        print(f"[placement] WARNING: room_d was {room_d:.2f}m or invalid — "
              f"estimated from camera z={cam_z_pos:.2f}m → room_d={room_d:.2f}m")

    all_segments: list[dict] = seg_data.get("segments", [])
    segments = [
        s for s in all_segments
        if s.get("type") not in _SKIP_TYPES and not s.get("sam_failed")
    ]

    if not segments:
        print("[placement] No furniture segments to analyse.")
        return furn_dir

    pos     = cam["position_m"]
    look_at = cam["look_at_m"]
    cam_x, cam_y, cam_z = pos
    lat_x, lat_y, lat_z = look_at

    print(f"[placement] {len(segments)} object(s) to order (carpet/rug excluded).")

    # ── Annotate render ───────────────────────────────────────────────────
    ann_path = furn_dir / "placement_annotated.png"
    annotate_render(render_p, segments, ann_path)

    # ── Geometric depth prior ─────────────────────────────────────────────
    depth_ordered = _image_depth_order(segments)

    # ── Build VLM prompt ──────────────────────────────────────────────────
    if no_wall_reasoning:
        wall_pos_text = (
            "HOW TO IDENTIFY EACH WALL FROM THE PHOTOGRAPH\n"
            "  (No wall-identification guidance provided for this ablation.)\n"
            "  Assign wall_affinity based solely on your visual judgement of the image."
        )
    else:
        wall_pos_text = _describe_wall_positions(cam, room_w, room_d, room_h)

    prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
        room_type            = room_type,
        room_w               = room_w,
        room_d               = room_d,
        room_h               = room_h,
        cam_x                = cam_x, cam_y = cam_y, cam_z = cam_z,
        lat_x                = lat_x, lat_y = lat_y, lat_z = lat_z,
        wall_image_positions = wall_pos_text,
        openings_context     = _build_openings_context(openings),
        wall_objects_context = _build_wall_objects_context(wall_objects),
        object_list          = _build_object_list(segments, cam, room_w, room_d),
        depth_prior          = _build_depth_prior(depth_ordered),
    )

    # ── Call VLM ──────────────────────────────────────────────────────────
    ann_b64, ann_mime = _encode_image(Image.open(ann_path))

    # Build a grid of individual object crops so VLM can identify each object.
    # This helps correct mislabelled types (e.g. macramé vs bookcase, desk vs table).
    crop_images: list[dict] = []
    seg_dir = furn_dir / "segmented"
    for seg in segments:
        crop_file = seg.get("crop_file")
        if crop_file and (seg_dir / crop_file).exists():
            try:
                crop_img = Image.open(seg_dir / crop_file).convert("RGB")
                # Resize to thumbnail for token efficiency
                crop_img.thumbnail((256, 256))
                cb64, cmime = _encode_image(crop_img)
                crop_images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{cmime};base64,{cb64}"},
                })
                crop_images.append({
                    "type": "text",
                    "text": f"↑ Close-up crop of idx={seg['index']} (detected as: {seg['type']})",
                })
            except Exception:
                pass

    vlm_content: list[dict] = [
        {"type": "text",      "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{ann_mime};base64,{ann_b64}"}},
    ]

    # Original photograph — needed for the `count` field: the annotated render is
    # the EMPTY room, so the VLM can't see how many identical instances exist
    # (segmentation misses occluded duplicates, e.g. dining chairs behind a table).
    photo_p = next((p for p in sorted(out_root.glob(f"{out_root.name}.*"))
                    if p.suffix.lower() in (".jpeg", ".jpg", ".png", ".webp", ".avif")), None)
    if photo_p is None:
        photo_p = next((p for p in sorted(out_root.glob("rgb_*.*"))
                        if p.suffix.lower() in (".jpeg", ".jpg") and "texture" not in p.name), None)
    if photo_p is not None:
        try:
            pb64, pmime = _encode_image(Image.open(photo_p).convert("RGB"))
            vlm_content.append({"type": "text", "text":
                "\nORIGINAL PHOTOGRAPH (the real room with all furniture visible). Use "
                "THIS image to set each object's `count`: how many identical instances "
                "are present, INCLUDING ones not separately outlined (e.g. duplicate "
                "dining chairs partly hidden behind a table)."})
            vlm_content.append({"type": "image_url",
                                "image_url": {"url": f"data:{pmime};base64,{pb64}"}})
        except Exception:
            pass

    if crop_images:
        vlm_content.append({"type": "text", "text": "\nINDIVIDUAL OBJECT CROPS (use these to verify object types):"})
        vlm_content.extend(crop_images)

    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": vlm_content,
        }],
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    print("[placement] Calling VLM …")
    resp = _vlm_post(payload, timeout=180)
    resp.raise_for_status()
    raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])

    # ── Parse VLM response ────────────────────────────────────────────────
    m = re.search(r"\{[\s\S]*\}", raw)
    analysis: dict = {}
    if m:
        try:
            analysis = json.loads(m.group())
        except json.JSONDecodeError as e:
            print(f"[placement] WARNING: JSON parse failed: {e}")
            analysis = {"raw_response": raw}
    else:
        print("[placement] WARNING: VLM returned no JSON block.")
        analysis = {"raw_response": raw}

    # Enrich placement entries with glb_file + box_px from segment data
    seg_by_idx = {s["index"]: s for s in segments}
    for entry in analysis.get("placement_order", []):
        idx = entry.get("index")
        if idx is not None and idx in seg_by_idx:
            seg = seg_by_idx[idx]
            stem = f"inpaint_{idx:02d}_{seg['type']}"
            entry["glb_file"] = seg.get("glb_file", f"objects/{stem}.glb")
            entry["box_px"]   = seg["box_px"]

    # Override wall_affinity / wall with geometric wall hint.
    # The VLM often confuses left/right with oblique cameras — the
    # back-projected wall hit is always correct for wall-adjacent objects.
    for entry in analysis.get("placement_order", []):
        idx = entry.get("index")
        seg = seg_by_idx.get(idx)
        if seg is None:
            continue
        geo_wall = _compute_wall_hint(seg, cam, room_w, room_d)
        ray_wall = _wall_from_center_ray(seg, cam, room_w, room_d, room_h)
        vlm_wall = entry.get("wall_affinity", "centre")
        _SIDES = ("left", "right")
        _WALLS = ("back", "left", "right")
        # Reconcile the VLM's wall label with the box geometry:
        #   1. Free-standing: footprint floats off every wall (geo=centre) but the
        #      VLM bound it to a side wall — a foreground accent chair near, not
        #      against, the wall. Un-bind it.
        #   2. Centre-ray reconciliation: when the VLM says the object IS against a
        #      wall, snap it to whichever geometric wall (z=0 / x=0 / x=room_w) its
        #      bbox-centre ray actually strikes.  This is the authoritative
        #      which-wall signal — it fixes diagonal-corner views where the VLM's
        #      "back wall I'm facing" is really the geometric side/right wall, and
        #      it works even for large objects whose floor footprint reads "centre".
        #   3/4. Footprint fallbacks (left/right swap, parallel-to-back) for the
        #      rare case where the centre ray misses every wall.
        # Side-wall seating under a head-on camera: an end-on sofa/chair hugging
        # the L/R image edge is against that side wall, even though the VLM and
        # the centre-ray both read "back" (the back wall fills the frame).  This
        # takes priority — it's the only cue that distinguishes a perpendicular
        # side-wall sofa from a back-wall one in a head-on view.
        _edge_side = _seating_side_wall_from_edge(seg, cam, entry.get("type", ""))
        if _edge_side is not None and vlm_wall != _edge_side:
            print(f"[placement] idx={idx}: VLM wall_affinity={vlm_wall} "
                  f"→ geometry={_edge_side} (end-on seating at {_edge_side} image edge)")
            entry["wall_affinity"] = _edge_side
        elif geo_wall == "centre" and vlm_wall in _SIDES:
            # Un-bind a free-standing piece the VLM wrongly pinned to a side wall
            # (a foreground accent chair near, not against, the wall) — but NOT a
            # SOFA/large seating: a sofa the VLM places against a side wall almost
            # always IS against it; the footprint only reads "centre" because the
            # back-projected depth drifted off the wall (the mask still wants it).
            # Un-binding leaves the sofa floating in the room (004818).
            #
            # The same drift hits anything WIDE, and for types that are
            # structurally wall-bound the un-bind is never right: a fireplace,
            # bookcase or wardrobe cannot stand free in a room.  `elegant` is the
            # worked example — the VLM correctly answered wall_affinity="left"
            # for a 2.46 m fireplace, geometry read "centre" because the
            # back-projected footprint of something that wide drifts off the
            # wall, and the un-bind dropped it to "centre", leaving it rendered
            # diagonally across the middle of the room.  Geometry is the weaker
            # signal here, not the stronger one, so the semantic answer stands.
            _t = str(entry.get("type", "")).lower().replace("-", "_")
            _WALL_BOUND = ("sofa", "couch", "sectional", "loveseat",
                           "fireplace", "hearth", "mantel", "mantelpiece",
                           "bookcase", "bookshelf", "shelf", "shelving",
                           "cabinet", "cupboard", "wardrobe", "armoire",
                           "dresser", "sideboard", "credenza", "buffet",
                           "radiator", "built_in", "console")
            if any(k in _t for k in _WALL_BOUND):
                print(f"[placement] idx={idx}: VLM wall_affinity={vlm_wall} KEPT "
                      f"({_t} is wall-bound — geometry reads centre only because "
                      f"a wide footprint back-projects off the wall)")
            else:
                print(f"[placement] idx={idx}: VLM wall_affinity={vlm_wall} "
                      f"→ geometry=centre (free-standing, off all walls)")
                entry["wall_affinity"] = "centre"
        elif vlm_wall in _WALLS and ray_wall is not None and ray_wall != vlm_wall:
            print(f"[placement] idx={idx}: VLM wall_affinity={vlm_wall} "
                  f"→ geometry={ray_wall} (centre-ray wall reconciliation)")
            entry["wall_affinity"] = ray_wall
        elif vlm_wall in _SIDES and geo_wall in _SIDES and vlm_wall != geo_wall:
            print(f"[placement] idx={idx}: VLM wall_affinity={vlm_wall} "
                  f"→ geometry={geo_wall} (left/right fix)")
            entry["wall_affinity"] = geo_wall
        elif (vlm_wall in _SIDES and geo_wall == "back"
              and _footprint_strongly_parallel_to_back(seg, cam, room_d)):
            print(f"[placement] idx={idx}: VLM wall_affinity={vlm_wall} "
                  f"→ geometry=back (footprint runs parallel to back wall)")
            entry["wall_affinity"] = "back"
        # else: keep the VLM's wall_affinity.
        entry["wall"] = entry.get("wall_affinity", vlm_wall)

    # ── Authoritative VLM back-against-wall check (beds & storage) ─────────────
    # Geometry + front-detect can't reliably tell which wall a headboard / cabinet
    # back sits against (corner beds, partial silhouettes). For these types the
    # semantic answer is the authority: ask the VLM directly from the photo and
    # override wall_affinity. The downstream bed_guard / wall-flush rotation then
    # turns the headboard/back flush against that wall automatically.
    _BACK_WALL_TYPES = ("bed", "cabinet", "dresser", "sideboard", "console",
                        "bookcase", "wardrobe", "chest", "tv_stand", "credenza",
                        "buffet", "nightstand")
    if not no_wall_reasoning and photo_p is not None:
        try:
            _photo_img = Image.open(photo_p).convert("RGB")
        except Exception:
            _photo_img = None
        for entry in (analysis.get("placement_order", []) if _photo_img is not None else []):
            t = str(entry.get("type", "")).lower().replace("-", "_")
            if not any(k in t for k in _BACK_WALL_TYPES):
                continue
            seg = seg_by_idx.get(entry.get("index"))
            crop_img = None
            if seg and seg.get("crop_file") and (seg_dir / seg["crop_file"]).exists():
                try:
                    crop_img = Image.open(seg_dir / seg["crop_file"]).convert("RGB")
                    crop_img.thumbnail((320, 320))
                except Exception:
                    crop_img = None
            vw = _vlm_back_wall(_photo_img, crop_img, t, entry.get("index"))
            if vw in ("back", "left", "right") and entry.get("wall_affinity") != vw:
                print(f"[placement] idx={entry.get('index')} {t}: back-wall VLM → "
                      f"{vw} (was {entry.get('wall_affinity')})")
                entry["wall_affinity"] = vw
                entry["wall"] = vw

    # Deterministic re-sort: enforce deepest-corner-first + wall-before-centre
    # + camera-depth descending + dependency topology over whatever order the
    # VLM returned.  The VLM's classification (depth / wall_affinity /
    # dependencies) is preserved — only the ORDER changes.
    if analysis.get("placement_order"):
        try:
            reordered = _reorder_placement(
                analysis["placement_order"], segments, cam, room_w, room_d,
            )
            analysis["placement_order"] = reordered
            print(f"[placement] Re-sorted {len(reordered)} entries by "
                  f"deepest-corner / wall-first / camera-depth rules.")
        except Exception as e:
            print(f"[placement] WARNING: programmatic re-sort failed: {e}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = furn_dir / "placement_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[placement] Analysis saved → {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    if "scene_summary" in analysis:
        print(f"\nScene summary:\n  {analysis['scene_summary']}\n")

    if "placement_order" in analysis:
        print("Placement order (back-wall first → camera last):")
        for entry in analysis["placement_order"]:
            deps = []
            if entry.get("on_top_of") is not None:
                deps.append(f"on_top_of={entry['on_top_of']}")
            if entry.get("in_front_of") is not None:
                deps.append(f"in_front_of={entry['in_front_of']}")
            if entry.get("behind") is not None:
                deps.append(f"behind={entry['behind']}")
            dep_str = f"  [{', '.join(deps)}]" if deps else ""
            print(f"  [{entry.get('index'):>2}] {entry.get('type'):<15}  "
                  f"depth={entry.get('depth'):<6}  wall={entry.get('wall_affinity'):<8}  "
                  f"group={entry.get('group') or '-'}{dep_str}")
            if entry.get("opening_relation"):
                print(f"       opening: {entry['opening_relation']}")
            if entry.get("notes"):
                print(f"       → {entry['notes']}")

    return furn_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyse furniture depth ordering and spatial relationships via VLM."
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Pipeline output directory (contains camera.json, render_final.png, furniture/)",
    )
    ap.add_argument(
        "--no-wall-reasoning", dest="no_wall_reasoning", action="store_true", default=False,
        help="Ablation: omit wall-identification guidance from the VLM prompt",
    )
    args = ap.parse_args()
    run(output_dir=args.output_dir, no_wall_reasoning=args.no_wall_reasoning)


if __name__ == "__main__":
    main()
