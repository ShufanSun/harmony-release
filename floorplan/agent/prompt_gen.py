from __future__ import annotations
import base64
import json
import re
import requests
from pathlib import Path


VLM_API_URL = "http://localhost:8080/v1/chat/completions"
from object_placement.vlm_backend import vlm_post as _vlm_post
IMG_GEN_URL = "http://localhost:8000/generate"


FLOORPLAN_ANALYSIS_PROMPT = """You are reconstructing only what is VISIBLE in a single photograph.
Do NOT infer, guess, or add anything not directly visible. Work through every step in order.

━━━ STEP 1 — TRACE THE BASEBOARD ━━━
The baseboard (skirting board) is the narrow trim strip where every WALL meets the FLOOR.
Furniture does NOT have a baseboard — only walls do.

Scan along the floor level from the LEFT edge of the image to the RIGHT edge.
For each continuous baseboard segment write one line:

  "Segment [letter]: [image position, e.g. left-third] — runs [diagonally-away / horizontally / diagonally-toward] — wall label: [left_wall / back_wall / right_wall]"

Direction guide:
  • Baseboard runs DIAGONALLY AWAY from viewer (converging toward a vanishing point) → SIDE WALL
  • Baseboard runs roughly HORIZONTALLY across the image                             → BACK WALL
  • Baseboard HIDDEN behind furniture → write "hidden behind [object]", but the wall is still present

At each point where the baseboard CHANGES DIRECTION = a real ROOM CORNER.
  Diagonal → Horizontal transition = near corner where side wall meets back wall
  Horizontal → Diagonal transition = far corner where back wall meets the other side wall

⚠ BASEBOARD CORNER TEST (apply at every direction change):
  Is the baseboard CONTINUOUS through the direction change, bending at the corner?
    YES → real room corner (baseboard bends = wall bends).
    NO  → something is blocking the view (furniture). Trace past it; the walls are still flat.
  Does the edge at the corner show painted plaster on BOTH sides?
    NO (wood / laminate / metal on either face) → furniture edge, NOT a wall corner.
    Ignore it and continue tracing the baseboard past the furniture.

After tracing, write a one-line summary:
  "Visible walls: [left_wall / back_wall / right_wall — whichever you traced a baseboard for]"

━━━ STEP 2 — COUNT DOORS AND WINDOWS EXACTLY ━━━
Carefully examine each visible wall. Count only what you can DIRECTLY see.

What counts as a DOOR:
  ✓ A door frame — vertical jambs + horizontal head, typically ~0.9m wide, ~2.0–2.1m tall
  ✓ An open doorway — clear rectangular opening in the wall surface
  ✗ Shelves, pegboards, cabinets, wardrobes, whiteboards → NOT doors
  ✗ Dark patches or shadows that aren't openings → NOT doors

What counts as a WINDOW:
  ✓ A glazed pane, frosted glass section, curtain covering an actual window opening
  ✗ Mirrors, TV screens, pictures → NOT windows

COUNTING PROCEDURE — do this for EVERY visible wall:
  a) Scan the wall surface left-to-right from floor to ceiling.
  b) For each candidate feature, ask: "Is this a structural opening in the wall?"
     If YES → it counts. If NO (furniture/object) → skip.
  c) Write the exact count. A count of "1" means exactly one feature of that type.
  d) If a wall is NOT visible in Step 1 → write "not visible".

FEATURE TABLE (fill in exactly, do not skip rows):
  Format: "WALL | doors (n) | windows (n) | notes"
  Example:
    LEFT  | doors: 1  | windows: 0  | door frame visible at left edge, ~0.9m wide
    BACK  | doors: 0  | windows: 1  | window upper right ~1.2m wide
    RIGHT | not visible → no features
    FRONT | not visible → no features

RE-EXAMINATION: After filling the table, look at the image one more time.
  • Did you miss any door/window that was partially hidden behind furniture?
  • Did you accidentally count a piece of furniture as a structural opening?
  Correct the table if needed, then proceed.

⚠ The features[] array in the final JSON MUST match the counts in this table exactly.
   If the table says "doors: 1" for a wall, features[] must contain exactly 1 door entry.
   If the table says "doors: 0", features[] must be empty for doors on that wall.

━━━ STEP 3 — IDENTIFY ROOM TYPE AND SET DIMENSION PRIORS ━━━
From the visible furniture, fixtures, and layout, identify the most likely room type.
Then use standard dimensions for that type as your starting estimate, refined by
what you can see in the image.

  Room type → typical floor (W × D) and ceiling height:
    home_office   : 3–5m × 3–5m,  ceiling 2.4–2.7m
    bedroom       : 3–5m × 3–5m,  ceiling 2.4–2.7m
    living_room   : 4–7m × 4–6m,  ceiling 2.4–3.0m
    kitchen       : 3–4m × 3–4m,  ceiling 2.4–2.7m
    meeting_room  : 4–8m × 3–6m,  ceiling 2.7–3.0m
    corridor      : 1.5–2m × 4–10m, ceiling 2.4–2.7m
    bathroom      : 1.5–3m × 1.5–3m, ceiling 2.4–2.7m
    other         : estimate from visible evidence

  Scale anchors visible in the image (use these to refine the prior):
    • Door: width ≈ 0.9m, height ≈ 2.1m
    • Desk: depth ≈ 0.7m; desk+chair+aisle ≈ 2.6m
    • Standard ceiling: 2.4–2.7m

  Output: room_type string, floor_width_m, floor_depth_m, ceiling_height_m.

━━━ STEP 4 — ESTIMATE VISIBLE WALL DIMENSIONS ━━━
Use the dimension priors from Step 3 and visible scale anchors.

For each visible wall estimate:
  • length_m (horizontal extent of the wall surface)
  • height_m = ceiling_height_m from Step 3
  • For each door/window: width_m, height_m, offset_from_left_m
    offset_from_left_m = distance from the LEFT END of that wall to the left edge of the feature.
    Verify: offset_from_left_m + width_m ≤ length_m.

━━━ STEP 5 — OUTPUT CHECKS ━━━
Before writing JSON, go through this checklist line by line:
  1) Re-read your feature TABLE from Step 2. For EACH wall:
       • Count the entries in features[] you are about to write.
       • The count must match the table exactly. If it doesn't, fix it NOW.
  2) For every feature entry: offset_from_left_m + width_m ≤ wall length_m.
  3) floor_width_m and floor_depth_m are consistent with the visible wall lengths.
  4) No wall that was "not visible" in Step 1 has any features[] entries.

━━━ STEP 6 — CAMERA PLACEMENT ━━━
You are reconstructing the camera that took this photograph.

HEIGHT:
  Look at recognisable objects whose real-world height you know:
    • Door frame top ≈ 2.1 m above floor
    • Standard desk surface ≈ 0.75 m above floor
    • Seated person's eye level ≈ 1.1–1.2 m
    • Standing person's eye level ≈ 1.5–1.7 m
    • Tripod / mounted camera ≈ 1.0–1.4 m
  The horizon (eye level) is the row where horizontal perspective lines converge.
  If the back wall's top edge appears ABOVE image centre → camera is below mid-wall height.
  If the back wall's top edge appears BELOW image centre → camera is above mid-wall height.
  Give a single number: camera_height_m above the floor.

FLOOR-WALL JUNCTION POSITION (primary tilt input):
  Find the floor-wall junction line at the DEEPEST visible corner — the horizontal line at
  the base of the back wall where it meets the floor.
  Estimate at what fraction of the image HEIGHT this line appears:
    0.0 = at the very TOP of the image   (camera looking sharply downward)
    0.5 = at the exact VERTICAL CENTRE   (camera level, horizon at mid-height)
    1.0 = at the very BOTTOM of the image (camera looking upward, no floor visible)
  Examples:
    • Junction at 30% from top (most of image is wall, little floor) → floor_junction_y_frac ≈ 0.30
    • Junction at 70% from top (lots of floor visible)              → floor_junction_y_frac ≈ 0.70
    • Junction at 55% from top (roughly level, slight floor)        → floor_junction_y_frac ≈ 0.55
  Give floor_junction_y_frac as a float 0.0–1.0.

TILT (vertical):
  This is a secondary confirmation only — the floor_junction_y_frac above is used for precision.
  Give tilt_deg: positive = tilted UP, negative = tilted DOWN.  Typical indoor: −5° to −15°.

YAW (horizontal rotation):
  How much is the camera rotated left/right from looking straight at the back wall?
  Method: find the single-point perspective vanishing point — where all horizontal lines
  converge. Its horizontal pixel position relative to image centre determines yaw.
    • Vanishing point AT image centre → yaw_deg = 0 (looking straight at back wall)
    • Vanishing point to the LEFT of centre (px < cx)  → yaw_deg negative (camera rotated left)
    • Vanishing point to the RIGHT of centre (px > cx) → yaw_deg positive (camera rotated right)
  Formula: yaw_deg ≈ atan2(vanishing_point_px − image_cx, focal_length_px)
  For a 70° HFOV image: focal_length ≈ image_width / (2 * tan 35°) ≈ 0.714 * image_width.
  Typical range: −45° to +45°. Give yaw_deg as a float.

AIM TARGET (where in the room is the camera pointed?):
  In addition to yaw_deg above, give a CONCRETE 3D TARGET POINT inside the room
  that the optical axis lands on.  This is much more reliable than yaw_deg
  alone for non-rectangular rooms (vaulted ceilings, glass corners, L-shapes)
  where vanishing-point detection is unstable.

  ★ COORDINATE SYSTEM — READ CAREFULLY, DO NOT GUESS ★
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • The CAMERA sits NEAR Z = floor_depth_m  (the FRONT of the room).
    • The CAMERA looks TOWARD Z = 0           (the BACK wall).
    • So the BACK WALL is at Z = 0, not Z = floor_depth_m.
    • X = 0 is the LEFT wall;  X = floor_width_m is the RIGHT wall.
    • Y = 0 is the floor;  Y = ceiling_height_m is the ceiling.

  Therefore for these typical landmarks, expect Z VALUES NEAR ZERO:
    • Centre of back-wall window  → Z ≈ 0.0,  X ≈ window_offset + window_width/2
    • Back-right corner crease    → Z ≈ 0.0,  X ≈ floor_width_m
    • Back-left corner crease     → Z ≈ 0.0,  X ≈ 0.0
  Side walls receding from camera have Z varying from 0 (deep) to floor_depth_m
  (near camera).  An aim target on the SIDE wall has X = 0 or X = floor_width_m
  and Z somewhere between 0 and ~floor_depth_m / 2 (rarely deeper than mid-room).

  WORKED EXAMPLES:
    Room 5m W × 4m D × 2.7m H, camera at front-centre looking at back wall:
      "centre of back-wall picture frame at chest height"  → [2.5, 1.2, 0.0]
      "back-right vertical wall crease, eye level"         → [5.0, 1.4, 0.0]
      "ceiling fixture above coffee table"                 → [2.5, 2.7, 1.5]

  Pick the most prominent visible LANDMARK the camera appears centred on
  (back-wall window, vertical wall corner, ceiling fixture, the deepest
  visible wall crease) and write its world coords as aim_target_world_m.

  HARD CONSTRAINTS:
    • aim_target_world_m[0] (X) ∈ [0, floor_width_m]
    • aim_target_world_m[1] (Y) ∈ [0, ceiling_height_m]  (typically ~1.0–1.5m)
    • aim_target_world_m[2] (Z) ∈ [0, floor_depth_m] — for a back-wall feature
                                  Z MUST be ≈ 0.  For a side-wall feature midway
                                  along the wall, Z ≈ floor_depth_m / 2.  IT IS
                                  AN ERROR to put Z near floor_depth_m for a
                                  back-wall landmark.
  ALSO give aim_target_landmark: a one-line description of the chosen feature
  (e.g. "centre of back-wall window", "vertical crease at back-right wall corner").
  This field overrides yaw_deg if both look correct — a directly-estimated 3D
  point cannot extrapolate outside the room the way a (yaw, dist) pair can.

DISTANCE FROM BACK WALL:
  Estimate how far the camera is from the back wall by reasoning about what is physically
  between the camera and the back wall:
    • If a desk is visible near the back wall: desk depth (~0.6 m) + chair (~0.5 m) + aisle (~0.8 m) → ~1.9 m
    • Desk + chair closer: ~1.4–1.6 m
    • Open floor with no furniture blocking: ~1.0–1.5 m
    • Camera is right at the entrance / doorway: close to floor_depth_m
  Give dist_to_back_m: the estimated distance in metres from the camera to the back wall.
  Must be between 0.5 and floor_depth_m − 0.1.

FACING:
  Which wall or corner is the camera primarily aimed at?
  Choose one: "back", "back_left", "back_right", "left", "right"
  • "back"       — camera looks straight at the back wall, roughly centred
  • "back_left"  — aimed toward the back-left corner
  • "back_right" — aimed toward the back-right corner
  • "left"       — mostly facing the left wall
  • "right"      — mostly facing the right wall

DEEPEST CORNER LINE:
  Find the vertical wall line that appears DEEPEST (farthest from camera) in the image.
  This is the vertical crease where two walls meet — it runs from floor to ceiling and is
  the point of maximum perspective convergence.

  Step 1 — locate it: scan left-to-right across the image. Find the vertical line where
    the receding floor/ceiling edges converge. This is usually a sharp vertical edge
    where two differently-coloured or differently-lit wall surfaces meet.

  Step 2 — estimate its pixel column: give corner_px as an integer (0 = left image edge,
    image_width = right edge). Use the pixel column of the vertical crease itself.

  Step 3 — look left and right of that line in the image:
    • What is immediately to the LEFT of the line? (e.g. "left wall going away from camera",
      "back wall", "window on left wall")
    • What is immediately to the RIGHT? (e.g. "back wall extending right", "right wall")

  Step 4 — determine back_wall_side: is the BACK WALL to the "left" or "right" of the corner line?
    If the back wall occupies the portion of the image to the RIGHT of the corner → back_wall_side = "right"
    If the back wall occupies the portion of the image to the LEFT of the corner  → back_wall_side = "left"

  Example reasoning:
    "I see a vertical crease at roughly pixel 340. To the left of it the left wall recedes away.
     To the right the back wall extends across most of the image.
     → corner_px = 340, back_wall_side = 'right'"

━━━ STEP 7 — TEXTURES AND TILE SIZE ━━━
Describe floor/wall surfaces so a seamless texture image of that MATERIAL can be generated.
  • ONLY describe the material surface itself: grain direction, pattern, color, finish.
  • NO lighting, shadows, perspective, depth, or objects.
  • DO NOT use the words "tile", "tiled", "tiles", or "tileable" in the prompt unless the
    actual surface visibly consists of ceramic/porcelain/stone tiles (e.g. bathroom floor).
    For painted walls, wood floors, carpet, concrete — do NOT use those words.
  • Good (painted wall):  "smooth matte white painted plaster, uniform surface, no pattern, no shadows"
  • Good (oak floor):     "light grey herringbone oak parquet, flat overhead view, no shadows, no depth"
  • Good (ceramic floor): "30 cm square white ceramic floor tiles, grout lines, flat overhead view"
  • Bad:  "warm ambient lighting, soft shadows, tileable texture pattern, polished"

TILE SIZE — estimate how large ONE physical repeat of the pattern is, in metres:
  Floor examples:
    • 30 cm square ceramic tiles       → tile_size_m = 0.30
    • 20 cm wide herringbone oak strip → tile_size_m = 0.20
    • Large 60 cm porcelain slab       → tile_size_m = 0.60
    • Plain carpet / featureless vinyl → tile_size_m = 1.00 (no visible repeat)
  Wall examples:
    • Painted plaster, no pattern      → tile_size_m = 2.00 (treat as very large tile)
    • 10 cm subway tile                → tile_size_m = 0.10
    • Brick (~24 cm long)              → tile_size_m = 0.24
  Method: look at how many repeats of the pattern fit across a known dimension
    (e.g. a 0.9 m door width), then divide: tile_size_m = 0.9 / count.
  Give tile_size_m as a float in metres for both floor and wall.

━━━ STEP 8 — VISIBLE FRAME ━━━
Estimate what portion of the room the camera actually captures vertically.
This determines how the render should be cropped — it should show the SAME amount
of wall and floor as the reference image, not the full room height.

  visible_wall_height_m:
    Look at the tallest continuous wall surface visible. How many metres of vertical
    wall height is visible from floor to the top of the visible wall?
    • Full wall visible (floor to ceiling)     → visible_wall_height_m = ceiling_height_m
    • Only bottom two-thirds of wall visible   → visible_wall_height_m ≈ ceiling_height_m × 0.67
    • Half a door visible (bottom cut off)     → use the visible door height as reference:
      if the door is 2.1 m and only the top 1.0 m is shown → visible_wall_height_m ≈ 1.0 m
    Give visible_wall_height_m in metres.

  visible_floor_depth_m:
    Estimate how much floor depth (front-to-back) is visible in the image.
    • Floor visible all the way to the back wall → visible_floor_depth_m ≈ dist_to_back_m
    • Only a shallow strip of floor visible      → give the estimated depth in metres
    Give visible_floor_depth_m in metres.

  FRAMING ADEQUACY (zoom in/out reasoning):
    The empty-geometry render should reveal enough usable floor/ceiling area to match the reference framing.
    • Desk/workspace heuristic: if the reference image shows a desk/work area, the visible floor region
      should plausibly fit a ~2.0m × 1.0m desk footprint near a visible wall (avoid a paper-thin floor strip).
    • Ceiling-light heuristic: if a ceiling light fixture is visible in the reference image, do NOT crop the
      ceiling to zero in the render — ensure some ceiling region is visible around/above that fixture.

━━━ OUTPUT — VALID JSON ONLY, NO MARKDOWN ━━━
{
  "room": {
    "room_type": "home_office|bedroom|living_room|kitchen|meeting_room|corridor|bathroom|other",
    "floor_width_m": 0.0,
    "floor_depth_m": 0.0,
    "ceiling_height_m": 0.0
  },
  "walls": [
    {
      "orientation": "back|left|right|front",
      "length_m": 0.0,
      "height_m": 0.0,
      "features": [
        {"type": "window|door", "width_m": 0.0, "height_m": 0.0, "offset_from_left_m": 0.0}
      ],
      "surface": "string"
    }
  ],
  "camera": {
    "height_m": 0.0,
    "tilt_deg": 0.0,
    "floor_junction_y_frac": 0.0,
    "yaw_deg": 0.0,
    "dist_to_back_m": 0.0,
    "facing": "back|back_left|back_right|left|right",
    "corner_px": 0,
    "back_wall_side": "left|right",
    "aim_target_world_m": [0.0, 0.0, 0.0],
    "aim_target_landmark": "string",
    "visible_wall_height_m": 0.0,
    "visible_floor_depth_m": 0.0
  },
  "floor_texture": {
    "material": "string",
    "color": "string",
    "pattern": "string",
    "tile_size_m": 0.0,
    "synthesis_prompt": "string"
  },
  "wall_texture": {
    "material": "string",
    "color": "string",
    "finish": "string",
    "tile_size_m": 0.0,
    "synthesis_prompt": "string"
  }
}"""


# ── helpers ──────────────────────────────────────────────────────────────────


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _parse_json_response(raw: str) -> dict:
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
            print(f"[prompt_gen] Warning: could not parse JSON: {e}")
            print(f"[prompt_gen] Raw (first 500 chars):\n{raw[:500]}")
            return {}


# ── Top-down floor plan renderer ─────────────────────────────────────────────

def _travel_direction(p1, p2):
    """Cardinal direction of travel from grid point p1 to p2."""
    dc, dr = p2[0] - p1[0], p2[1] - p1[1]
    if abs(dc) >= abs(dr):
        return "right" if dc >= 0 else "left"
    return "down" if dr >= 0 else "up"


def _corner_char(in_dir: str, out_dir: str) -> str:
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    extensions = frozenset([opposite[in_dir], out_dir])
    return {
        frozenset(["right", "down"]): "┌",
        frozenset(["left",  "down"]): "┐",
        frozenset(["right", "up"  ]): "└",
        frozenset(["left",  "up"  ]): "┘",
    }.get(extensions, "+")


def draw_floorplan_ascii(analysis: dict) -> str:
    """
    Render a top-down ASCII floor plan.

    Top-down orientation:
      y=0 (top row)    = back of room  (furthest from camera)
      y=max (bot row)  = front of room (camera side)
      x=0 (left col)  = left wall
      x=max (right col)= right wall
    """
    room   = analysis.get("room", {})

    walls_list = []
    for w in analysis.get("walls", []):
        clean = dict(w)
        clean["features"] = [
            f for f in w.get("features", [])
            if f.get("type") in ("door", "window")
        ]
        walls_list.append(clean)

    pw = float(room.get("floor_width_m") or room.get("dimensions", {}).get("width_m", 5.0) or 5.0)
    pd = float(room.get("floor_depth_m") or room.get("dimensions", {}).get("depth_m", 4.0) or 4.0)
    polygon_m = [[0, 0], [pw, 0], [pw, pd], [0, pd]]

    polygon_m = [[float(x), float(y)] for x, y in polygon_m]
    xs = [p[0] for p in polygon_m]
    ys = [p[1] for p in polygon_m]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = x_max - x_min or 1.0
    span_y = y_max - y_min or 1.0

    ceiling_h = room.get("ceiling_height_m", room.get("dimensions", {}).get("height_m", "?"))

    scale = max(4, round(60 / span_x))
    cols = round(span_x * scale) + 1
    rows = round(span_y * scale) + 1

    def to_col(x): return round((x - x_min) * scale)
    def to_row(y): return round((y - y_min) * scale)

    # Wall orientations that are actually visible in the image
    visible_orients = {w["orientation"] for w in walls_list if "orientation" in w}

    grid_verts = [(to_col(x), to_row(y)) for x, y in polygon_m]
    n = len(grid_verts)
    grid = [[" "] * cols for _ in range(rows)]

    # Map each polygon edge index → its wall orientation (or "inner")
    min_row, max_row = 0, rows - 1
    min_col, max_col = 0, cols - 1

    def _edge_orient(c1, r1, c2, r2) -> str:
        if r1 == r2:   # horizontal edge
            return "back" if r1 == min_row else ("front" if r1 == max_row else "inner")
        if c1 == c2:   # vertical edge
            return "left" if c1 == min_col else ("right" if c1 == max_col else "inner")
        return "inner"

    edge_drawn = []   # parallel to polygon edges; True if this edge is rendered
    for i in range(n):
        c1, r1 = grid_verts[i]
        c2, r2 = grid_verts[(i + 1) % n]
        orient = _edge_orient(c1, r1, c2, r2)
        draw = (orient == "inner") or (orient in visible_orients)
        edge_drawn.append(draw)
        if not draw:
            continue
        if r1 == r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if 0 <= r1 < rows and 0 <= c < cols:
                    grid[r1][c] = "─"
        elif c1 == c2:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                if 0 <= r < rows and 0 <= c1 < cols:
                    grid[r][c1] = "│"

    # Draw corner chars only where both meeting edges are drawn
    for i in range(n):
        in_drawn  = edge_drawn[(i - 1) % n]
        out_drawn = edge_drawn[i]
        if not (in_drawn or out_drawn):
            continue
        c, r = grid_verts[i]
        pc, pr = grid_verts[(i - 1) % n]
        nc, nr = grid_verts[(i + 1) % n]
        in_d  = _travel_direction((pc, pr), (c, r))
        out_d = _travel_direction((c, r), (nc, nr))
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = _corner_char(in_d, out_d)

    horiz: list[tuple] = []
    vert:  list[tuple] = []
    for i in range(n):
        c1, r1 = grid_verts[i]
        c2, r2 = grid_verts[(i + 1) % n]
        if r1 == r2:
            horiz.append((r1, min(c1, c2), max(c1, c2)))
        elif c1 == c2:
            vert.append((c1, min(r1, r2), max(r1, r2)))

    edge_for = {
        "back":  min(horiz, key=lambda e: e[0], default=None),
        "front": max(horiz, key=lambda e: e[0], default=None),
        "left":  min(vert,  key=lambda e: e[0], default=None),
        "right": max(vert,  key=lambda e: e[0], default=None),
    }
    wall_map = {w["orientation"]: w for w in walls_list}

    for orient, edge in edge_for.items():
        wall = wall_map.get(orient)
        if not wall or not edge:
            continue
        for feat in wall.get("features", []):
            char = "W" if feat.get("type") == "window" else "D"
            off  = feat.get("offset_from_left_m", 0.0)
            fw   = feat.get("width_m", 0.8)
            if orient in ("back", "front"):
                row, c_lo, c_hi = edge
                cs = c_lo + round(off * scale)
                ce = c_lo + round((off + fw) * scale)
                for c in range(max(c_lo + 1, cs), min(c_hi, ce)):
                    if 0 <= c < cols:
                        grid[row][c] = char
            else:
                col, r_lo, r_hi = edge
                rs = r_lo + round(off * scale)
                re_ = r_lo + round((off + fw) * scale)
                for r in range(max(r_lo + 1, rs), min(r_hi, re_)):
                    if 0 <= r < rows:
                        grid[r][col] = char

    grid_lines = ["".join(row) for row in grid]

    pad = "  "
    lines: list[str] = []

    room_type = room.get("room_type", room.get("type", "room"))
    lines.append(f"FLOOR PLAN — {room_type.upper()}")
    lines.append(
        f"Floor: {span_x:.1f}m × {span_y:.1f}m  |  Ceiling: {ceiling_h}m"
    )
    lines.append("")

    inner = cols - 2
    lines.append(pad + " ←" + "─" * (inner // 2 - 2) + f" {span_x:.1f}m " + "─" * max(0, inner - inner // 2 - 3) + "→")
    lines.append(pad + f"{'y=0  (back of room)':^{cols}}")

    mid = rows // 2
    for r, row_str in enumerate(grid_lines):
        suffix = {mid - 1: "  ↑", mid: f"  {span_y:.1f}m", mid + 1: "  ↓"}.get(r, "")
        lines.append(pad + row_str + suffix)

    lines.append(pad + f"{'y=max (front of room)':^{cols}}")
    lines.append("")

    lines.append("Walls:")
    for orient in ["back", "front", "left", "right"]:
        wall = wall_map.get(orient)
        if not wall:
            continue
        feats = wall.get("features", [])
        feat_str = ""
        if feats:
            parts = [
                f"{f['type']} {f.get('width_m','?')}×{f.get('height_m','?')}m"
                f" @{f.get('offset_from_left_m','?')}m"
                for f in feats
            ]
            feat_str = "  →  " + ", ".join(parts)
        lines.append(
            f"  {orient:5s}  {wall.get('length_m','?'):.1f}m × {wall.get('height_m','?'):.1f}m"
            f"  [{wall.get('surface','?')}]{feat_str}"
        )
    lines.append("")
    lines.append("Legend:  W = window   D = door")
    lines.append(f"Scale:   1 char ≈ {1/scale:.3f}m")

    return "\n".join(lines)


# ── Stage A: yaw — deepest back-wall corner alignment ────────────────────────

def _yaw_corner_prompt(corner_px: int | None, img_width: int | None) -> str:
    if corner_px is not None and img_width and img_width > 0:
        hint = (f"The deepest back-wall corner vertical line is expected near pixel column "
                f"{corner_px} (~{corner_px/img_width*100:.0f}% from left) in the render.")
    else:
        hint = "Identify the deepest visible vertical back-wall corner crease in both images."
    return f"""\
You are given:
  IMAGE 1 — REFERENCE photo of a real room.
  IMAGE 2 — SOFTWARE RENDER of the same room's empty geometry.

{hint}

━━━ TASK — ALIGN THE DEEPEST BACK-WALL CORNER ━━━
Find the vertical line where the deepest concave back-wall corner appears in each image.
Measure its X position as a fraction of IMAGE WIDTH from the left edge (0.0=left, 1.0=right).

  corner_x_ref    = X fraction in IMAGE 1
  corner_x_render = X fraction in IMAGE 2

  yaw_delta_deg = rotation needed to align render corner to reference corner position.
    Positive = rotate camera RIGHT (corner moves left in image).
    Negative = rotate camera LEFT  (corner moves right in image).
    Approximate formula: yaw_delta_deg ≈ (corner_x_ref − corner_x_render) × hfov
    Keep within ±20°.  Write 0 if already well-matched (< 2% difference).

Output ONLY valid JSON, no markdown:
{{
  "corner_x_ref":    <float 0–1>,
  "corner_x_render": <float 0–1>,
  "yaw_delta_deg":   <float>,
  "reasoning":       "<one sentence>"
}}"""


def refine_yaw_corner(
    ref_image_path: str,
    render_path: str,
    camera: dict,
    corner_px: int | None = None,
    img_width: int | None = None,
) -> tuple[dict, None]:
    """Stage A: align the deepest back-wall corner vertical line via yaw rotation."""
    import numpy as np
    hfov = float(camera.get("hfov_deg", 70.0))
    prompt = _yaw_corner_prompt(corner_px, img_width)
    ref_b64, render_b64 = encode_image(ref_image_path), encode_image(render_path)
    def _mime(p): return "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{_mime(ref_image_path)};base64,{ref_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:{_mime(render_path)};base64,{render_b64}"}},
        ]}],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    print("[prompt_gen] Stage A — yaw corner alignment...")
    try:
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        data = _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[prompt_gen] Stage A VLM failed ({e}) — skipping.")
        return camera, None

    yaw_delta = float(data.get("yaw_delta_deg", 0.0))
    cx_ref    = data.get("corner_x_ref")
    cx_render = data.get("corner_x_render")
    print(f"[prompt_gen] Stage A: corner_x ref={cx_ref} render={cx_render}  yaw_delta={yaw_delta:+.1f}°")

    if abs(yaw_delta) < 0.5:
        return camera, None

    # Clamp yaw delta: never rotate more than hfov/2 in a single step to
    # prevent look_at flying outside the room on bad VLM measurements.
    max_step = hfov / 2.0
    if abs(yaw_delta) > max_step:
        print(f"[prompt_gen] Stage A: yaw_delta={yaw_delta:+.1f}° exceeds max {max_step:.1f}° — clamping.")
        yaw_delta = float(np.clip(yaw_delta, -max_step, max_step))

    pos  = np.array(camera["position_m"], dtype=float)
    look = np.array(camera["look_at_m"],  dtype=float)
    fwd  = look - pos
    dist = float(np.linalg.norm(fwd))
    yr = np.radians(yaw_delta)
    cos_y, sin_y = float(np.cos(yr)), float(np.sin(yr))
    dx = fwd[0] * cos_y - fwd[2] * sin_y
    dz = fwd[0] * sin_y + fwd[2] * cos_y
    fwd[0], fwd[2] = dx, dz
    new_look = pos + fwd / np.linalg.norm(fwd) * dist
    camera["look_at_m"] = [round(float(v), 4) for v in new_look]
    print(f"[prompt_gen] Stage A: yaw {yaw_delta:+.1f}° applied → look_at={camera['look_at_m']}")
    return camera, None


# ── Stage B: yaw — floor/ceiling junction line direction ─────────────────────

def _yaw_floor_vp_prompt(hfov: float) -> str:
    return f"""\
You are given:
  IMAGE 1 — REFERENCE photo of a real room.
  IMAGE 2 — SOFTWARE RENDER of the same room's empty geometry.

━━━ TASK — ALIGN FLOOR/CEILING PERSPECTIVE LINES ━━━
Look at the diagonal lines on the FLOOR where the LEFT and RIGHT side walls meet the floor.
These lines converge toward a vanishing point on the horizon.

Also look at the CEILING edges (where ceiling meets the side walls) — these converge to the
same vanishing point.

Estimate the vanishing point X position as a fraction of image width from the left edge:
  0.0 = far left,  0.5 = image centre,  1.0 = far right

  floor_vp_x_ref    = vanishing-point X fraction in IMAGE 1 (reference)
  floor_vp_x_render = vanishing-point X fraction in IMAGE 2 (render)

  yaw_delta_deg = (floor_vp_x_ref − floor_vp_x_render) × {hfov:.1f}
    Positive = rotate camera RIGHT.  Negative = rotate camera LEFT.
    Write 0 if lines are not clearly visible or already well-matched (< 2% X difference).

Output ONLY valid JSON, no markdown:
{{
  "floor_vp_x_ref":    <float 0–1 or null>,
  "floor_vp_x_render": <float 0–1 or null>,
  "yaw_delta_deg":     <float>,
  "reasoning":         "<one sentence>"
}}"""


def refine_yaw_floor_vp(
    ref_image_path: str,
    render_path: str,
    camera: dict,
) -> tuple[dict, None]:
    """Stage B: refine yaw from floor/ceiling vanishing-point direction."""
    import numpy as np
    hfov = float(camera.get("hfov_deg", 70.0))
    prompt = _yaw_floor_vp_prompt(hfov)
    ref_b64, render_b64 = encode_image(ref_image_path), encode_image(render_path)
    def _mime(p): return "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{_mime(ref_image_path)};base64,{ref_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:{_mime(render_path)};base64,{render_b64}"}},
        ]}],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    print("[prompt_gen] Stage B — yaw floor/ceiling VP alignment...")
    try:
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        data = _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[prompt_gen] Stage B VLM failed ({e}) — skipping.")
        return camera, None

    yaw_delta  = float(data.get("yaw_delta_deg", 0.0))
    vp_ref     = data.get("floor_vp_x_ref")
    vp_render  = data.get("floor_vp_x_render")
    print(f"[prompt_gen] Stage B: floor_vp ref={vp_ref} render={vp_render}  yaw_delta={yaw_delta:+.1f}°")

    if abs(yaw_delta) < 0.5:
        return camera, None

    pos  = np.array(camera["position_m"], dtype=float)
    look = np.array(camera["look_at_m"],  dtype=float)
    fwd  = look - pos
    dist = float(np.linalg.norm(fwd))
    yr = np.radians(float(np.clip(yaw_delta, -20.0, 20.0)))
    cos_y, sin_y = float(np.cos(yr)), float(np.sin(yr))
    dx = fwd[0] * cos_y - fwd[2] * sin_y
    dz = fwd[0] * sin_y + fwd[2] * cos_y
    fwd[0], fwd[2] = dx, dz
    new_look = pos + fwd / np.linalg.norm(fwd) * dist
    camera["look_at_m"] = [round(float(v), 4) for v in new_look]
    print(f"[prompt_gen] Stage B: yaw {yaw_delta:+.1f}° applied → look_at={camera['look_at_m']}")
    return camera, None


# ── Stage X: arc — lateral camera position around deepest corner ─────────────

def _arc_position_prompt(cam_y: float, current_angle_deg: float) -> str:
    return f"""\
You are given:
  IMAGE 1 — REFERENCE photo of a real room.
  IMAGE 2 — RENDERED version of the same room.

TASK — measure the angle of the SIDE WALL floor junction near the deepest back corner.

The side wall floor junction is the line running from the deepest corner (where the back wall
meets a side wall) TOWARD the camera, along the floor-wall boundary of that side wall.
In the image this line starts at the corner and converges toward the bottom-centre.

MEASUREMENT:
  Find that side wall junction line in IMAGE 1 (reference) and IMAGE 2 (render).
  Measure the angle it makes with the HORIZONTAL at the corner:
    90°  = perfectly vertical (camera directly in front of corner, no lateral offset)
    45°  = line runs diagonally (moderate lateral offset)
    0°   = perfectly horizontal (camera very far to the side)
  Typical range: 20° – 80°.

  Note: the current render was computed with camera_y = {cam_y:.3f} m, giving a
  geometric side-wall angle of {current_angle_deg:.1f}°.

Output ONLY this JSON (no markdown, no extra text):
{{
  "side_wall_angle_ref": <float, degrees, 0-90>,
  "side_wall_angle_render": <float, degrees, 0-90>,
  "corner_side": "<left|right>  — which side of the image the corner is on",
  "reasoning": "<one sentence>"
}}
"""


def refine_arc_position(
    ref_image_path: str,
    render_path: str,
    camera: dict,
    floor_width_m: float | None = None,
) -> tuple[dict, None]:
    """Stage X: slide camera laterally along an arc around the deepest corner
    so the side-wall floor-junction angle matches the reference.

    The side wall floor junction makes angle θ = arctan(cam_y / lateral_offset)
    with horizontal, independent of cam_z and tilt.  Moving on a constant-radius
    arc around the corner adjusts the lateral offset without changing zoom.
    """
    import numpy as np

    cam_y  = float(camera["position_m"][1])
    cam_x  = float(camera["position_m"][0])
    cam_z  = float(camera["position_m"][2])
    look   = [float(v) for v in camera["look_at_m"]]
    look_x, look_z = look[0], look[2]

    # Determine which back corner the camera is looking at.
    # Corner is at (0, 0, 0) (left) or (floor_width_m, 0, 0) (right).
    if floor_width_m and floor_width_m > 0:
        dist_left  = abs(look_x - 0.0)
        dist_right = abs(look_x - floor_width_m)
        corner_3d_x = 0.0 if dist_left <= dist_right else floor_width_m
    else:
        # Fallback: guess from look_at horizontal position vs camera x
        corner_3d_x = 0.0 if look_x < cam_x else (floor_width_m or cam_x * 2)

    lateral_offset = abs(cam_x - corner_3d_x)
    if lateral_offset < 1e-3:
        lateral_offset = 1e-3  # avoid division by zero
    current_angle_deg = float(np.degrees(np.arctan2(cam_y, lateral_offset)))

    prompt = _arc_position_prompt(cam_y, current_angle_deg)
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _img_to_data_url(ref_image_path)}},
            {"type": "image_url", "image_url": {"url": _img_to_data_url(render_path)}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": 512,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    print("[prompt_gen] Stage X — arc lateral position...")
    try:
        resp = _vlm_post(payload, timeout=60)
        data = _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[prompt_gen] Stage X VLM failed ({e}) — skipping.")
        return camera, None

    angle_ref = data.get("side_wall_angle_ref")
    angle_ren = data.get("side_wall_angle_render")
    print(f"[prompt_gen] Stage X: side_wall_angle ref={angle_ref}° render={angle_ren}° "
          f"(current analytic={current_angle_deg:.1f}°)")

    if angle_ref is None or float(angle_ref) < 5.0:
        print("[prompt_gen] Stage X: no reliable angle measurement — skipping.")
        return camera, None

    target_angle_rad = float(np.radians(float(angle_ref)))
    if np.tan(target_angle_rad) < 1e-6:
        print("[prompt_gen] Stage X: target angle too small — skipping.")
        return camera, None

    # Solve for new lateral offset from target angle
    # θ = arctan(cam_y / lateral) → lateral = cam_y / tan(θ)
    target_lateral = cam_y / float(np.tan(target_angle_rad))
    target_lateral = max(0.1, target_lateral)

    # Arc radius in horizontal plane (x-z), centred on corner
    r = float(np.sqrt((cam_x - corner_3d_x) ** 2 + cam_z ** 2))
    if target_lateral >= r:
        # Can't satisfy without going behind the back wall — clamp
        target_lateral = r * 0.95
    new_cam_z = float(np.sqrt(max(0.01, r ** 2 - target_lateral ** 2)))

    # Preserve same side (left/right of corner)
    sign = 1.0 if cam_x >= corner_3d_x else -1.0
    new_cam_x = corner_3d_x + sign * target_lateral

    if abs(new_cam_x - cam_x) < 0.05 and abs(new_cam_z - cam_z) < 0.05:
        print(f"[prompt_gen] Stage X: position already correct — no change.")
        return camera, None

    camera["position_m"] = [round(new_cam_x, 4), round(cam_y, 4), round(new_cam_z, 4)]
    # look_at stays at same world point — camera naturally re-angles toward it
    print(f"[prompt_gen] Stage X: arc move cam=({cam_x:.2f},{cam_z:.2f})→"
          f"({new_cam_x:.2f},{new_cam_z:.2f})  lateral {lateral_offset:.2f}→{target_lateral:.2f}m  "
          f"angle {current_angle_deg:.1f}°→{float(angle_ref):.1f}°")
    return camera, None


# ── Stage D: tilt — ceiling/floor proportion ─────────────────────────────────

def _tilt_proportion_prompt(cam_z: float, cam_y: float, ceiling_h: float, vfov: float) -> str:
    return f"""\
You are given:
  IMAGE 1 — REFERENCE photo of a real room.
  IMAGE 2 — SOFTWARE RENDER of the same room's empty geometry.

RENDER GEOMETRY (ground truth):
  cam_z         = {cam_z:.2f} m
  camera height = {cam_y:.2f} m
  ceiling height= {ceiling_h:.2f} m
  vertical FOV  = {vfov:.1f}°

━━━ TASK — MEASURE CEILING / FLOOR PROPORTION ━━━
In IMAGE 1 (reference), measure the Y positions of two junction lines as fractions of image
height from the TOP (0.0 = very top, 1.0 = very bottom):

  floor_frac_ref   = Y fraction where the floor meets the back wall (floor-wall junction)
  ceiling_frac_ref = Y fraction where the ceiling meets the back wall (ceiling-wall junction)
                     Write null if the ceiling is not visible in IMAGE 1.
  ceiling_visible  = true if ceiling is clearly visible in IMAGE 1

Then in IMAGE 2 (render), measure the same:
  floor_frac_render   = floor-wall junction Y fraction
  ceiling_frac_render = ceiling-wall junction Y fraction (null if not visible)
  ceiling_visible_render = true if ceiling is clearly visible in IMAGE 2

These fractions will be used to analytically compute the correct camera tilt.

Output ONLY valid JSON, no markdown:
{{
  "floor_frac_ref":          <float 0–1>,
  "ceiling_frac_ref":        <float 0–1 or null>,
  "ceiling_visible":         <bool>,
  "floor_frac_render":       <float 0–1>,
  "ceiling_frac_render":     <float 0–1 or null>,
  "ceiling_visible_render":  <bool>,
  "reasoning": "<one sentence: describe ceiling/floor proportion difference>"
}}"""


# ── Stage CD: jointly balance zoom and tilt from reference proportions ────────

def balance_zoom_tilt(
    ref_image_path: str,
    render_path: str,
    camera: dict,
    ceiling_h: float = 2.4,
    room_depth_m: float | None = None,
) -> tuple[dict, float | None]:
    """
    Jointly solve cam_z and tilt so the rendered ceiling/floor proportions match
    the reference — without blindly zooming out to show ceiling when the reference
    itself doesn't show ceiling.

    When ceiling IS visible in reference:
      • Solve cam_z analytically so the ceiling-to-floor angular span matches.
        span = arctan((ceiling_h-cam_y)/z) + arctan(cam_y/z)
             = (floor_frac_ref - ceiling_frac_ref) × vfov
        (Binary search; monotonically decreasing in z.)
      • Compute tilt from floor_frac_ref at the solved cam_z.

    When ceiling NOT visible in reference:
      • Keep current cam_z (do NOT zoom out — reference intentionally hides ceiling).
      • Compute tilt from floor_frac_ref, clamped so ceiling stays off-screen.

    Returns (updated_camera, expanded_room_depth_m_or_None).
    """
    import numpy as np

    pos    = np.array(camera["position_m"], dtype=float)
    look   = np.array(camera["look_at_m"],  dtype=float)
    cam_z  = float(pos[2])
    cam_y  = float(pos[1])
    look_z = float(look[2])
    vfov   = float(camera.get("vfov_deg", 43.0))
    vfov_rad = float(np.radians(vfov))

    # Reuse the same tilt-proportion prompt (measures floor/ceiling fractions)
    prompt = _tilt_proportion_prompt(cam_z, cam_y, ceiling_h, vfov)
    ref_b64    = encode_image(ref_image_path)
    render_b64 = encode_image(render_path)
    def _mime(p): return "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{_mime(ref_image_path)};base64,{ref_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:{_mime(render_path)};base64,{render_b64}"}},
        ]}],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    print("[prompt_gen] Stage CD — balance zoom+tilt...")
    try:
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        data = _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[prompt_gen] Stage CD VLM failed ({e}) — skipping.")
        return camera, None, False

    floor_frac_ref   = data.get("floor_frac_ref")
    ceiling_frac_ref = data.get("ceiling_frac_ref")
    # ceiling_visible is True when the VLM found a measurable ceiling fraction in the reference.
    # The response schema uses ceiling_frac_ref=null when ceiling is off-screen,
    # so derive the flag from that rather than a nonexistent "ceiling_visible" key.
    ceiling_visible  = ceiling_frac_ref is not None
    print(f"[prompt_gen] Stage CD: floor_frac_ref={floor_frac_ref}  "
          f"ceiling_frac_ref={ceiling_frac_ref}  ceiling_visible={ceiling_visible}")

    if floor_frac_ref is None:
        print("[prompt_gen] Stage CD: no floor fraction measured — skipping.")
        return camera, None, False

    f_f = float(np.clip(float(floor_frac_ref), 0.05, 0.95))
    new_cam_z     = None
    expanded_depth = None

    # ── Case 1: ceiling visible → solve cam_z from angular span ──────────────
    if ceiling_visible and ceiling_frac_ref is not None:
        f_c = float(np.clip(float(ceiling_frac_ref), 0.0, f_f - 0.05))
        target_span_rad = (f_f - f_c) * vfov_rad

        if target_span_rad > float(np.radians(3.0)):
            # Binary search: span(z) is monotonically decreasing in z
            lo_z, hi_z = 0.3, 40.0
            for _ in range(60):
                mid_z = (lo_z + hi_z) / 2.0
                span = (float(np.arctan((ceiling_h - cam_y) / mid_z))
                        + float(np.arctan(cam_y / mid_z)))
                if span > target_span_rad:
                    lo_z = mid_z
                else:
                    hi_z = mid_z
            solved_z = (lo_z + hi_z) / 2.0
            print(f"[prompt_gen] Stage CD: target_span={float(np.degrees(target_span_rad)):.1f}°  "
                  f"solved cam_z={solved_z:.2f}m (current={cam_z:.2f}m)")

            if abs(solved_z - cam_z) / max(cam_z, 0.1) > 0.05:
                new_cam_z = round(float(solved_z), 3)
                if room_depth_m is not None and new_cam_z > room_depth_m - 0.1:
                    expanded_depth = round(new_cam_z + 0.3, 2)

    # ── Compute tilt at the final cam_z ──────────────────────────────────────
    final_cam_z = new_cam_z if new_cam_z is not None else cam_z

    try:
        t_f   = (2.0 * f_f - 1.0) * float(np.tan(np.radians(vfov / 2.0)))
        denom = final_cam_z + t_f * cam_y
        if abs(denom) < 1e-4:
            return camera, None, ceiling_visible
        target_tilt_deg = float(np.degrees(np.arctan2(t_f * final_cam_z - cam_y, denom)))
    except (TypeError, ValueError):
        return camera, None, ceiling_visible

    # ── Clamp tilt based on ceiling visibility intent ─────────────────────────
    ceiling_angle_deg = float(np.degrees(np.arctan2(ceiling_h - cam_y, final_cam_z)))
    floor_angle_deg   = float(np.degrees(np.arctan2(-cam_y, final_cam_z)))
    tilt_up_limit     = ceiling_angle_deg - vfov / 2.0   # minimum tilt: just shows ceiling
    tilt_down_limit   = floor_angle_deg   + vfov / 2.0   # maximum tilt: just shows floor

    if ceiling_visible:
        if tilt_up_limit <= tilt_down_limit:
            target_tilt_deg = float(np.clip(target_tilt_deg, tilt_up_limit, tilt_down_limit))
        else:
            # Still too close for the given span even after solving — prioritize ceiling
            print(f"[prompt_gen] Stage CD: tilt conflict at z={final_cam_z:.2f}m "
                  f"— prioritizing ceiling ({tilt_up_limit:+.1f}°)")
            target_tilt_deg = tilt_up_limit
    else:
        # Reference has no ceiling: ensure ceiling stays hidden
        target_tilt_deg = min(target_tilt_deg, tilt_up_limit - 0.5)

    # ── Apply position and look_at ────────────────────────────────────────────
    dz = cam_z - look_z
    current_tilt_deg = float(np.degrees(np.arctan2(float(look[1]) - cam_y, dz))) if abs(dz) > 1e-4 else 0.0

    pos_changed  = new_cam_z is not None
    tilt_changed = abs(target_tilt_deg - current_tilt_deg) > 0.5

    if not pos_changed and not tilt_changed:
        print(f"[prompt_gen] Stage CD: already balanced — no change.")
        return camera, None, ceiling_visible

    if pos_changed:
        camera["position_m"] = [round(float(pos[0]), 4), round(cam_y, 4), round(float(new_cam_z), 4)]

    final_z = float(camera["position_m"][2])
    new_look_y = cam_y + (final_z - look_z) * float(np.tan(np.radians(target_tilt_deg)))
    camera["look_at_m"] = [round(float(look[0]), 4), round(new_look_y, 4), round(float(look[2]), 4)]

    print(f"[prompt_gen] Stage CD: cam_z {cam_z:.2f}→{final_z:.2f}m  "
          f"tilt {current_tilt_deg:+.1f}°→{target_tilt_deg:+.1f}°")
    return camera, expanded_depth, ceiling_visible


_VIEW_CHECK_PROMPT = """\
Look at this indoor image and answer two questions:

1. VIEW TYPE — which best describes the camera angle?
   "eye_level"  : camera at roughly standing / seated eye height, walls clearly visible
   "oblique"    : camera angled significantly downward or upward but walls still visible
   "top_down"   : camera pointing mostly straight down, floor dominates, walls barely visible

2. WALLS VISIBLE — are the room walls (vertical surfaces) and the floor-wall junction lines
   clearly visible and suitable for perspective alignment?
   true  : yes, at least one wall face and its floor-wall edge are clearly visible
   false : no — floor dominates, or walls are cut off / obscured

Respond with ONLY valid JSON:
{
  "view_type": "eye_level|oblique|top_down",
  "walls_visible": true|false,
  "reasoning": "<one short sentence>"
}
"""


def check_view_angle(image_path: str) -> dict:
    """
    Quick VLM call to determine if the image has clearly visible walls and what
    the viewing angle is.  Returns a dict with keys:
      view_type     : "eye_level" | "oblique" | "top_down"
      walls_visible : bool
      reasoning     : str
    Defaults to {"view_type": "eye_level", "walls_visible": True} on failure.
    """
    try:
        import requests
        image_b64 = encode_image(image_path)
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        payload = {
            "model": "qwen3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",      "text": _VIEW_CHECK_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }],
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        print("[prompt_gen] Checking view angle...")
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw  = resp.json()["choices"][0]["message"]["content"]
        data = _parse_json_response(raw)
        view_type     = str(data.get("view_type",     "eye_level")).lower()
        walls_visible = bool(data.get("walls_visible", True))
        reasoning     = data.get("reasoning", "")
        print(f"[prompt_gen] View check: type={view_type!r}  "
              f"walls_visible={walls_visible}  reason: {reasoning}")
        return {"view_type": view_type, "walls_visible": walls_visible,
                "reasoning": reasoning}
    except Exception as e:
        print(f"[prompt_gen] View angle check failed ({e}) — assuming eye_level with walls.")
        return {"view_type": "eye_level", "walls_visible": True, "reasoning": ""}


def analyze_floorplan(image_path: str) -> dict:
    """Send image to Qwen VLM and extract structured floor plan data."""
    print(f"[prompt_gen] Encoding image: {image_path}")
    image_b64 = encode_image(image_path)

    prompt_text = FLOORPLAN_ANALYSIS_PROMPT

    payload = {
        "model": "qwen3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    print("[prompt_gen] Calling VLM for floor plan analysis...")
    import requests
    response = _vlm_post(payload, timeout=180)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return _parse_json_response(raw)


_VLM_REFINE_TEXTURE_PROMPT = """\
You are preparing a SEAMLESS TILEABLE ALBEDO TEXTURE map for a 3D scene.

I am giving you ONE input: the original room photograph.

Your task is to look at the photograph and write a single descriptive prompt
for a text-to-image model that will produce a flat, repeatable texture of the
{surface_label} material visible in the scene.

The texture must show ONLY the bare material — no objects, no shadows, no
lighting, no perspective, no panels or grout (unless the surface really IS
tiled, like ceramic bathroom flooring).

Inspect the photo and identify, FOR THE {surface_label_upper} ONLY:
  1. The base color  (be specific: warm beige, cool grey, light oak, etc. —
     describe the AVERAGE colour, ignoring shadows and highlights)
  2. The finish      (matte, satin, gloss, painted, plastered, raw, polished)
  3. The pattern     ("none" if uniform paint / smooth concrete / plain
                      carpet; otherwise describe: herringbone, chevron,
                      6×6 ceramic tile, brick, etc.)
  4. The tile size in metres (one repeat of the visible pattern; for
                              uniform paint use 2.0 as a stand-in for "no repeat")

Then synthesize ALL of the above into a single self-contained prompt for the
image generator.  The prompt MUST:
  • lead with the material description (color + finish + pattern)
  • include "{view_phrase}"
  • include explicit anti-light language: "no shadows, no lighting, no shading,
    no ambient occlusion, no highlights, no darkening, pure diffuse albedo"
  • include explicit anti-grid language UNLESS the actual material is tiled:
    "no panel divisions, no tile lines, no grout lines, no seams, no joints,
     completely uniform color"
  • NOT mention furniture, perspective, depth, room layout, or 3D geometry

If the previous synthesis_prompt was {hint!r}, you may borrow vocabulary
from it but improve specificity (e.g. "warm cream-beige" beats "beige";
"light grey-toned oak with soft brown grain" beats "wood").

Return ONLY valid JSON:
{{
  "color": "...",
  "finish": "...",
  "pattern": "none" | "...",
  "tile_size_m": 0.0,
  "synthesis_prompt": "the full prompt to feed the image generator"
}}
"""


def _vlm_refine_texture_prompt(reference_image_path: str | Path,
                               surface: str,
                               hint_prompt: str = "") -> dict | None:
    """Call the VLM to look at the reference photo and produce an improved
    synthesis prompt for the {surface} texture.  Returns None on failure so
    callers can fall back to the original analysis prompt.
    """
    surface_label = surface.replace("_", " ")
    if surface in ("floor", "ceiling"):
        view_phrase = "perfectly flat orthographic top-down view, zero perspective"
    else:
        view_phrase = ("perfectly flat front-on orthographic view, "
                       "zero perspective, all features axis-aligned")
    text = _VLM_REFINE_TEXTURE_PROMPT.format(
        surface_label=surface_label,
        surface_label_upper=surface_label.upper(),
        view_phrase=view_phrase,
        hint=hint_prompt or "(no hint)",
    )

    try:
        ref_b64 = encode_image(reference_image_path)
    except Exception as e:
        print(f"[prompt_gen] [tex_refine] failed to encode ref image: {e}")
        return None

    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
            ],
        }],
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            print(f"[prompt_gen] [tex_refine] no JSON in response: {raw[:200]}")
            return None
        parsed = json.loads(m.group())
        sp = str(parsed.get("synthesis_prompt", "")).strip()
        if len(sp) < 30:
            print(f"[prompt_gen] [tex_refine] prompt too short, skipping refinement")
            return None
        return parsed
    except Exception as e:
        print(f"[prompt_gen] [tex_refine] VLM call failed: {e}")
        return None


def _texture_edit_prompt(surface: str, synthesis_prompt: str) -> str:
    """
    Wrap a raw synthesis_prompt with texture-extraction instructions matching
    the Qwen/texture_refine.py style, so QwenImageEditPipeline knows to produce
    a seamless flat texture map rather than a scene-composition image.
    """
    if surface in ("floor", "ceiling"):
        view_hint = (
            "perfectly flat top-down orthographic view, "
            "all planks/tiles/lines must be exactly horizontal or vertical, "
            "zero perspective, zero foreshortening, "
            "no reflections, no specular sheen, no wet look, pure diffuse albedo"
        )
    else:
        view_hint = (
            "perfectly flat front-on orthographic view, "
            "all horizontal features (grout lines, brick rows, wood grain) "
            "must be exactly horizontal — zero diagonal lines, "
            "zero perspective, zero foreshortening"
        )
    surface_label = surface.replace("_", " ")
    return (
        f"Extract the {surface_label} texture from this room photo and convert it "
        f"into a seamless tileable texture map. "
        f"Material: {synthesis_prompt}. "
        f"Requirements: {view_hint}. "
        f"Pure diffuse albedo only — absolutely no lighting, no shading, no shadows, "
        f"no ambient occlusion, no baked light, no highlights, no darkening at edges. "
        f"No furniture, no depth cues, no perspective distortion. "
        f"Pure repeating surface pattern only, square image."
    )


def generate_pbr_maps(
    albedo_path: str,
    width: int = 1024,
    height: int = 1024,
) -> dict[str, bytes]:
    """Call /generate_pbr to produce normal, roughness, and metallic maps
    aligned to the given albedo texture.  Returns {map_type: png_bytes}."""
    payload = {
        "albedo_image": encode_image(albedo_path),
        "width": width,
        "height": height,
        "num_inference_steps": 20,   # 3 passes × 20 steps ≈ 4 min; 30 steps timed out
        "true_cfg_scale": 4.0,
    }
    response = requests.post(IMG_GEN_URL.replace("/generate", "/generate_pbr"),
                             json=payload, timeout=900)   # 15 min: 3 passes × ~3 min each
    response.raise_for_status()
    data = response.json()
    return {k: base64.b64decode(v) for k, v in data.items()}


def _save_pbr_maps(albedo_path: str, label: str) -> None:
    """Generate and save PBR maps next to the albedo file.
    Files: <stem>_normal.png, <stem>_roughness.png, <stem>_metallic.png"""
    try:
        maps = generate_pbr_maps(albedo_path)
        stem = Path(albedo_path).with_suffix("").as_posix()
        for map_type, png_bytes in maps.items():
            out = f"{stem}_{map_type}.png"
            with open(out, "wb") as f:
                f.write(png_bytes)
            print(f"[prompt_gen] {label} {map_type} map saved → {out}")
    except Exception as e:
        print(f"[prompt_gen] PBR map generation failed for {label}: {e}")


def generate_texture_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    reference_image_path: str | None = None,
    surface: str = "wall",
) -> bytes:
    """Request a texture image from the img_server (QwenImageEditPipeline).

    The prompt is wrapped with texture-extraction framing before sending so the
    model produces a seamless flat texture map rather than a scene image.
    If reference_image_path is provided the room photo is passed as the edit
    reference so the model can visually sample the actual surface appearance.
    """
    edit_prompt = _texture_edit_prompt(surface, prompt)
    payload: dict = {
        "prompt": edit_prompt,
        "negative_prompt": (
            "3d render, perspective, depth, objects, furniture, low quality, "
            "blurry, distorted, artifacts, shadows, lighting, shading, "
            "ambient occlusion, baked lighting, highlights, darkening, "
            "directional light, specular, gloss, reflections, wet look, "
            "mirror effect, glossy sheen"
        ),
        "num_images": 1,
        "width": width,
        "height": height,
        "num_inference_steps": 30,
        "true_cfg_scale": 4.0,
    }
    if reference_image_path:
        payload["reference_image"] = encode_image(reference_image_path)

    # Routed through adapter — backend selected by SCENEWEAVE_IMG_EDIT env var
    from Qwen.image_edit_adapter import edit_image as _edit_image
    data = _edit_image(payload, timeout=300)
    return base64.b64decode(data["images"][0])


# ── pipeline ──────────────────────────────────────────────────────────────────

def run_floorplan_stage(image_path: str, output_dir: Path) -> dict:
    """
    Full floor plan pipeline:
      1. Quick VLM view-angle check (top-down vs perspective)
      2. Analyze image with Qwen VLM → structured floor plan + camera + texture data
      3. Build the rectangular wall mesh from the VLM floor dims (VGGT refines it)
      4. Draw ASCII floor plan and save to floorplan.txt
      5. Generate seamless floor texture via img_server
      6. Generate seamless wall texture via img_server
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: quick view-angle check ──────────────────────────────────────
    # Determines whether corner-pixel alignment is appropriate for this image.
    # Top-down or no-wall views skip the deepest-corner logic entirely.
    view_info_path = output_dir / "view_info.json"
    if view_info_path.exists():
        view_info = json.loads(view_info_path.read_text(encoding="utf-8"))
        print(f"[prompt_gen] Reusing existing view_info.json")
    else:
        view_info = check_view_angle(image_path)
        view_info_path.write_text(json.dumps(view_info, indent=2), encoding="utf-8")
    # Only disable corner alignment for confirmed top-down views.
    # walls_visible=False alone is NOT enough to disable — the VLM may misjudge
    # oblique or partial-floor views as wall-less when walls are clearly present.
    use_corner_px = view_info["view_type"] != "top_down"
    if not use_corner_px:
        print(f"[prompt_gen] Corner-pixel alignment DISABLED — confirmed top-down view")
    elif not view_info["walls_visible"]:
        print(f"[prompt_gen] walls_visible=False reported but keeping corner alignment "
              f"(view_type={view_info['view_type']!r} — not top_down)")

    # ── Step 2: VLM analysis ──────────────────────────────────────────────────
    # If floorplan_analysis.json already exists (e.g. rerunning the pipeline to
    # tweak camera/render), load it and skip the Qwen VLM call entirely.
    analysis_cache = output_dir / "floorplan_analysis.json"
    if analysis_cache.exists():
        print(f"[prompt_gen] Reusing existing floorplan analysis → {analysis_cache}")
        with open(analysis_cache) as _f:
            analysis = json.load(_f)
    else:
        analysis = analyze_floorplan(image_path)

    # ── Step 3: generate the rectangular wall mesh from VLM dimensions ────────
    # Seed box only — VGGT (Stage 2 / geometry_refine) fits the real extents.
    if analysis:
        try:
            from floorplan.wall_line.wall_mesh_gen import generate_wall_mesh
            room_info = (analysis or {}).get("room", {})
            vlm_ceiling = room_info.get("ceiling_height_m") or 2.4
            vlm_floor_w = room_info.get("floor_width_m")
            vlm_floor_d = room_info.get("floor_depth_m")
            room_type   = room_info.get("room_type", "unknown")
            print(f"[prompt_gen] Room type: {room_type}  "
                  f"ceiling={vlm_ceiling}m  "
                  f"floor={vlm_floor_w}×{vlm_floor_d}m")
            if vlm_floor_w and vlm_floor_d:
                generate_wall_mesh(
                    out_path=str(output_dir / "walls.obj"),
                    ceiling_h=float(vlm_ceiling),
                    floor_dims=(float(vlm_floor_w), float(vlm_floor_d)),
                )
            else:
                print("[prompt_gen] VLM omitted floor dims — skipping wall mesh")
        except Exception as e:
            print(f"[prompt_gen] Wall mesh generation failed: {e}")

    # ── Step 4: camera placement ───────────────────────────────────────────────
    if analysis:
        try:
            from floorplan.wall_line.camera_placement import camera_from_analysis, save_camera
            camera = camera_from_analysis(
                analysis=analysis,
                ref_image_path=image_path,
                out_path=str(output_dir / "camera.json"),
            )

            # ── Top-down override ────────────────────────────────────────────
            # When the reference image is a top-down/overhead view the normal
            # perspective camera (looking at the back wall) is wrong.  Override
            # to a camera positioned above the room centre, looking straight down.
            if view_info["view_type"] == "top_down":
                room_info   = (analysis or {}).get("room", {})
                fw  = float(room_info.get("floor_width_m")   or 4.0)
                fd  = float(room_info.get("floor_depth_m")   or 4.0)
                ch  = float(room_info.get("ceiling_height_m") or 2.4)
                # Place camera above room centre at ceiling + 1 m, looking down
                cam_h   = ch + max(fw, fd) * 0.5   # high enough to see the floor
                mid_x   = fw / 2.0
                mid_z   = fd / 2.0
                img_W_ref, img_H_ref = camera.get("width_px", 1280), camera.get("height_px", 720)
                camera["position_m"]  = [round(mid_x, 4), round(cam_h, 4), round(mid_z, 4)]
                camera["look_at_m"]   = [round(mid_x, 4), 0.0, round(mid_z, 4)]
                camera["up"]          = [0, 0, -1]   # north = -Z when looking down
                camera["width_px"]    = img_W_ref
                camera["height_px"]   = img_H_ref
                save_camera(camera, str(output_dir / "camera.json"))
                print(f"[prompt_gen] Top-down camera override: "
                      f"pos=({mid_x:.2f},{cam_h:.2f},{mid_z:.2f}) looking down")

        except Exception as e:
            print(f"[prompt_gen] Camera placement failed: {e}")

    # Step 5: render is deferred — run AFTER texture generation so
    # walls_metadata.json exists when the renderer loads it (see below).

    if not analysis:
        print("[prompt_gen] Floor plan analysis returned empty result.")
        return {}

    # Save JSON
    analysis_path = output_dir / "floorplan_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[prompt_gen] Analysis saved → {analysis_path}")

    # Save + print ASCII plan
    floorplan_txt = draw_floorplan_ascii(analysis)
    plan_path = output_dir / "floorplan.txt"
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(floorplan_txt + "\n")
    print(f"[prompt_gen] Floor plan saved → {plan_path}")
    print("\n" + floorplan_txt + "\n")

    # Summary
    room   = analysis.get("room", {})
    walls  = analysis.get("walls", [])
    print(
        f"[prompt_gen] Room  : {room.get('room_type', room.get('type','?'))}  "
        f"{room.get('floor_width_m','?')}×{room.get('floor_depth_m','?')}m  "
        f"ceiling={room.get('ceiling_height_m','?')}m\n"
        f"[prompt_gen] Walls : {len(walls)}"
    )
    for wall in walls:
        feats = ", ".join(
            f"{f['type']} {f.get('width_m','?')}×{f.get('height_m','?')}m"
            for f in wall.get("features", [])
        )
        print(
            f"[prompt_gen]   {wall.get('orientation','?'):5s} "
            f"{wall.get('length_m','?')}m × {wall.get('height_m','?')}m"
            + (f"  [{feats}]" if feats else "")
        )

    camera_json = output_dir / "camera.json"
    results = {
        "analysis": analysis,
        "analysis_path": str(analysis_path),
        "floorplan_txt_path": str(plan_path),
        "camera_path": str(camera_json) if camera_json.exists() else None,
    }

    # ── Generate textures ─────────────────────────────────────────────────────
    # Global floor + fallback wall texture
    for tex_key, label, out_name in [
        ("floor_texture", "floor", "floor_texture.png"),
        ("wall_texture",  "wall",  "wall_texture.png"),
    ]:
        out_path = output_dir / out_name
        if out_path.exists():
            print(f"[prompt_gen] Reusing existing {label} texture → {out_path}")
            results[f"{label}_texture_path"] = str(out_path)
            continue
        tex = analysis.get(tex_key, {})
        hint_prompt = tex.get("synthesis_prompt", "")

        # Per-scene prompt refinement: ask the VLM to inspect the photo and
        # produce a high-specificity, anti-tile / anti-shadow prompt for THIS
        # particular {label} surface.  Falls back to the bulk-analysis hint.
        refined = _vlm_refine_texture_prompt(image_path, label,
                                             hint_prompt=hint_prompt)
        if refined and refined.get("synthesis_prompt"):
            prompt = refined["synthesis_prompt"]
            tex.update({
                "color":          refined.get("color", tex.get("color", "")),
                "finish":         refined.get("finish", tex.get("finish", "")),
                "pattern":        refined.get("pattern", tex.get("pattern", "")),
                "tile_size_m":    refined.get("tile_size_m",
                                              tex.get("tile_size_m", 1.0)),
                "synthesis_prompt": prompt,
            })
            analysis[tex_key] = tex
            # Persist the refined prompt back to floorplan_analysis.json so
            # future re-runs see the better description.
            try:
                analysis_path.write_text(json.dumps(analysis, indent=2))
            except Exception:
                pass
            print(f"[prompt_gen] [{label}] VLM-refined prompt: \"{prompt[:120]}...\"")
        else:
            prompt = hint_prompt
            if prompt:
                print(f"[prompt_gen] [{label}] using analysis prompt "
                      f"(refine unavailable): \"{prompt[:80]}...\"")

        if not prompt:
            print(f"[prompt_gen] No synthesis_prompt for {label}, skipping.")
            continue
        print(f"[prompt_gen] Generating {label} texture …")
        try:
            img_bytes = generate_texture_image(prompt, reference_image_path=image_path, surface=label)
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            print(f"[prompt_gen] {label.capitalize()} texture saved → {out_path}")
            results[f"{label}_texture_path"] = str(out_path)
            _save_pbr_maps(str(out_path), label)
        except Exception as e:
            print(f"[prompt_gen] ERROR generating {label} texture: {e}")

    # ── Per-wall textures + walls_metadata.json ───────────────────────────────
    walls_meta: dict = {}
    global_wall_tex = analysis.get("wall_texture", {})
    global_wall_prompt = global_wall_tex.get("synthesis_prompt", "")

    # Cache: prompt string → generated file path.
    # Pre-seed with the already-generated global wall texture so walls whose
    # surface description matches it reuse that file without a new generation.
    # Also pre-seed with any per-wall texture files that already exist on disk.
    _prompt_to_path: dict[str, str] = {}
    global_wall_file = output_dir / "wall_texture.png"
    if global_wall_prompt and global_wall_file.exists():
        _prompt_to_path[global_wall_prompt] = str(global_wall_file)

    for wall in analysis.get("walls", []):
        orient = wall.get("orientation")
        if not orient:
            continue

        surface = wall.get("surface", "")
        features = wall.get("features", [])

        # Build a synthesis prompt for this specific wall surface.
        # Avoid "tile/tileable" unless the surface is literally ceramic tiles —
        # those words bias image generators toward square-tile patterns.
        if surface and surface not in ("string", "?"):
            wall_prompt = (
                f"{surface}, seamless repeating surface material, "
                "flat even lighting, no perspective, no shadows, photorealistic"
            )
        else:
            wall_prompt = global_wall_prompt

        # Per-wall VLM refinement: produces a more specific anti-tile /
        # anti-shadow prompt than the per-orientation "surface" hint above.
        # Cached per-orientation so we don't pay the VLM call N times when
        # a wall is reused via _prompt_to_path.
        if wall_prompt and orient not in _prompt_to_path:
            _refined_wall = _vlm_refine_texture_prompt(
                image_path, f"{orient}_wall", hint_prompt=wall_prompt)
            if _refined_wall and _refined_wall.get("synthesis_prompt"):
                wall_prompt = _refined_wall["synthesis_prompt"]
                print(f"[prompt_gen] [{orient}_wall] VLM-refined: "
                      f"\"{wall_prompt[:120]}...\"")

        tex_path_str = None
        if wall_prompt:
            existing_file = output_dir / f"wall_{orient}_texture.png"
            if wall_prompt in _prompt_to_path:
                # Same visual appearance as a previously generated texture — reuse it.
                tex_path_str = _prompt_to_path[wall_prompt]
                print(f"[prompt_gen] {orient} wall texture reused from {tex_path_str} "
                      f"(same surface)")
            elif existing_file.exists():
                # File already on disk from a previous run — reuse without regenerating.
                tex_path_str = str(existing_file)
                _prompt_to_path[wall_prompt] = tex_path_str
                print(f"[prompt_gen] {orient} wall texture reused (existing file) → {existing_file}")
            else:
                out_name = f"wall_{orient}_texture.png"
                print(f"[prompt_gen] Generating {orient} wall texture: \"{wall_prompt[:70]}...\"")
                try:
                    img_bytes = generate_texture_image(wall_prompt, reference_image_path=image_path, surface=f"{orient}_wall")
                    tex_file = output_dir / out_name
                    with open(tex_file, "wb") as f:
                        f.write(img_bytes)
                    print(f"[prompt_gen] {orient} wall texture saved → {tex_file}")
                    tex_path_str = str(tex_file)
                    _prompt_to_path[wall_prompt] = tex_path_str
                    _save_pbr_maps(str(tex_file), f"{orient} wall")
                except Exception as e:
                    print(f"[prompt_gen] ERROR generating {orient} wall texture: {e}")

        walls_meta[orient] = {
            "orientation": orient,
            "length_m":    wall.get("length_m"),
            "height_m":    wall.get("height_m"),
            "surface":     surface,
            "features":    features,
            "texture_path": tex_path_str,
            "tile_size_m":  global_wall_tex.get("tile_size_m") or 2.0,
        }

    # Floor entry with tile size
    floor_tex_info = analysis.get("floor_texture", {})
    floor_tile_size = float(floor_tex_info.get("tile_size_m") or 0.5)
    floor_tex_path  = results.get("floor_texture_path")
    walls_meta["floor"] = {
        "texture_path": floor_tex_path,
        "tile_size_m":  floor_tile_size,
    }

    # Ceiling uses the wall texture synthesis prompt (or a lighter version).
    # Reuse from cache (file on disk or prompt already generated) to avoid an
    # extra image-gen call when ceiling uses the same surface as the walls.
    ceiling_prompt = global_wall_tex.get("synthesis_prompt", "")
    if ceiling_prompt:
        ceiling_file = output_dir / "ceiling_texture.png"
        if ceiling_file.exists():
            print(f"[prompt_gen] Reusing existing ceiling texture → {ceiling_file}")
            walls_meta["ceiling"] = {
                "texture_path": str(ceiling_file),
                "tile_size_m":  global_wall_tex.get("tile_size_m") or 2.0,
            }
        elif ceiling_prompt in _prompt_to_path:
            # Same prompt as an already-generated texture — symlink via path reuse.
            walls_meta["ceiling"] = {
                "texture_path": _prompt_to_path[ceiling_prompt],
                "tile_size_m":  global_wall_tex.get("tile_size_m") or 2.0,
            }
            print(f"[prompt_gen] Ceiling texture reused from "
                  f"{_prompt_to_path[ceiling_prompt]} (same surface)")
        else:
            try:
                img_bytes = generate_texture_image(ceiling_prompt, reference_image_path=image_path, surface="ceiling")
                with open(ceiling_file, "wb") as f:
                    f.write(img_bytes)
                _save_pbr_maps(str(ceiling_file), "ceiling")
                walls_meta["ceiling"] = {
                    "texture_path": str(ceiling_file),
                    "tile_size_m":  global_wall_tex.get("tile_size_m") or 2.0,
                }
                _prompt_to_path[ceiling_prompt] = str(ceiling_file)
                print(f"[prompt_gen] Ceiling texture saved → {ceiling_file}")
            except Exception as e:
                print(f"[prompt_gen] ERROR generating ceiling texture: {e}")

    if walls_meta:
        meta_path = output_dir / "walls_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(walls_meta, f, indent=2)
        print(f"[prompt_gen] Walls metadata saved → {meta_path}")
        results["walls_metadata_path"] = str(meta_path)

        # ── Human-readable wall features summary ─────────────────────────────
        lines = ["WALL FEATURES SUMMARY", "=" * 40]
        for orient, meta in walls_meta.items():
            if orient == "ceiling":
                continue
            length = meta.get("length_m")
            height = meta.get("height_m")
            surface = meta.get("surface", "—")
            dims = f"{length}m × {height}m" if length and height else ""
            lines.append(f"\n{orient.upper()} wall{('  ' + dims) if dims else ''}")
            lines.append(f"  Surface : {surface}")
            features = meta.get("features") or []
            if features:
                for feat in features:
                    ftype  = feat.get("type", "?")
                    fw     = feat.get("width_m",  "?")
                    fh     = feat.get("height_m", "?")
                    offset = feat.get("offset_from_left_m")
                    pos_str = f" @ {offset}m from left" if offset is not None else ""
                    lines.append(f"  {ftype.capitalize()}: {fw}m wide × {fh}m tall{pos_str}")
            else:
                lines.append("  (no doors or windows)")
        summary_path = output_dir / "wall_features.txt"
        summary_path.write_text("\n".join(lines) + "\n")
        print(f"[prompt_gen] Wall features summary saved → {summary_path}")
        results["wall_features_path"] = str(summary_path)

        # ── Re-write OBJ with texture MTL now that walls_meta is complete ────
        obj_path = output_dir / "walls.obj"
        if obj_path.exists():
            try:
                import trimesh as _tm
                from floorplan.wall_line.wall_mesh_gen import write_obj as _write_obj
                _mesh = _tm.load(str(obj_path), force="mesh")
                if isinstance(_mesh, _tm.Scene):
                    _mesh = _tm.util.concatenate(_mesh.dump())
                _verts = [tuple(float(c) for c in v) for v in _mesh.vertices]
                # trimesh faces are 0-indexed; OBJ needs 1-indexed
                _faces = [tuple(int(i) + 1 for i in f) for f in _mesh.faces]
                _write_obj(str(obj_path), _verts, _faces, walls_meta=walls_meta)
            except Exception as e:
                print(f"[prompt_gen] OBJ texture attachment failed: {e}")

    # ── Step 7: render room mesh (after textures + metadata are ready) ─────────
    mesh_obj    = output_dir / "walls.obj"
    camera_json = output_dir / "camera.json"
    render_path = output_dir / "render.png"
    if mesh_obj.exists():
        try:
            from floorplan.wall_line.render_room import render_room
            render_room(
                mesh_path=str(mesh_obj),
                camera_json_path=str(camera_json) if camera_json.exists() else None,
                out_path=str(render_path),
                ref_image_path=image_path,
            )
            results["render_path"] = str(render_path)
        except Exception as e:
            print(f"[prompt_gen] Room render failed: {e}")

    # ── Derive shared inputs used across refinement stages ───────────────────
    # Stage A (corner-pixel yaw alignment) anchors on the VLM's corner_px hint
    # for the deepest visible back-wall corner; VGGT re-derives the anchor from
    # metric geometry downstream.
    _refine_corner_px: int | None = None
    _has_concave_corner: bool = False
    _vlm_corner_px = ((analysis or {}).get("camera", {}) or {}).get("corner_px")
    if use_corner_px and _vlm_corner_px is not None:
        try:
            _refine_corner_px = int(_vlm_corner_px)
            _has_concave_corner = True
        except (TypeError, ValueError):
            _refine_corner_px = None
    if _refine_corner_px is None:
        print("[prompt_gen] No corner pixel available — skipping Stage A yaw alignment "
              "(Stage B floor-VP will handle yaw).")

    def _room_meta():
        _r = analysis.get("room", {})
        return (
            float(_r.get("floor_depth_m") or 0) or None,
            float(_r.get("ceiling_height_m") or 2.4),
        )

    def _expand_mesh(new_depth: float, ceiling_h: float):
        """Expand the floor plan and rebuild the mesh in-place."""
        old_depth, _ = _room_meta()
        print(f"[prompt_gen] Expanding floor plan: {old_depth}m → {new_depth}m")
        analysis["room"]["floor_depth_m"] = new_depth
        with open(output_dir / "floorplan_analysis.json", "w") as _f:
            json.dump(analysis, _f, indent=2)
        try:
            from floorplan.wall_line.wall_mesh_gen import generate_wall_mesh
            _ri = analysis.get("room", {})
            generate_wall_mesh(
                out_path=str(mesh_obj), ceiling_h=ceiling_h,
                floor_dims=(float(_ri.get("floor_width_m") or 4.0), float(new_depth)),
            )
            print(f"[prompt_gen] Mesh rebuilt depth={new_depth}m → {mesh_obj}")
        except Exception as _me:
            print(f"[prompt_gen] Mesh expansion failed: {_me}")

    def _do_rerender(out_name: str) -> str:
        from floorplan.wall_line.camera_placement import save_camera
        from floorplan.wall_line.render_room import render_room
        save_camera(camera, str(camera_json))
        out_path = output_dir / out_name
        render_room(mesh_path=str(mesh_obj), camera_json_path=str(camera_json),
                    out_path=str(out_path), ref_image_path=image_path)
        results["render_path"] = str(out_path)
        print(f"[prompt_gen] Re-render saved → {out_path}")
        return str(out_path)

    # ── Stage A: yaw — deepest back-wall corner alignment ────────────────────
    # Skipped when no concave corner was detected (all-convex scene) to prevent
    # massive yaw overcorrection from aligning to furniture/object corners.
    if _has_concave_corner and render_path.exists() and camera_json.exists() and mesh_obj.exists():
        try:
            import json as _json
            camera = _json.loads(camera_json.read_text())
            orig = (list(camera["look_at_m"]), list(camera["position_m"]))
            camera, _ = refine_yaw_corner(
                image_path, str(render_path), camera,
                corner_px=_refine_corner_px, img_width=img_W or None,
            )
            if camera["look_at_m"] != orig[0]:
                _do_rerender("render_yaw_corner.png")
            else:
                print("[prompt_gen] Stage A: no yaw change.")
        except Exception as e:
            print(f"[prompt_gen] Stage A failed: {e}")

    # ── Stage B: yaw — floor/ceiling perspective line direction ──────────────
    # Skipped when no concave corner was detected: a single-flat-wall view has
    # no converging floor VP to measure — running B would produce spurious yaw.
    _cur_render = results.get("render_path") or str(render_path)
    if _has_concave_corner and Path(_cur_render).exists() and camera_json.exists():
        try:
            import json as _json
            camera = _json.loads(camera_json.read_text())
            orig_look = list(camera["look_at_m"])
            camera, _ = refine_yaw_floor_vp(image_path, _cur_render, camera)
            if camera["look_at_m"] != orig_look:
                _do_rerender("render_yaw_floor.png")
            else:
                print("[prompt_gen] Stage B: no yaw change.")
        except Exception as e:
            print(f"[prompt_gen] Stage B failed: {e}")

    # ── Stage X: arc — lateral camera position around deepest corner ─────────
    # After yaw aligns the corner horizontally, move camera along an arc so the
    # side-wall floor-junction angle matches the reference perspective.
    # Skipped when no concave corner detected: there is no visible corner to
    # arc around — camera is centred on a single flat wall.
    _cur_render = results.get("render_path") or str(render_path)
    if _has_concave_corner and Path(_cur_render).exists() and camera_json.exists():
        try:
            import json as _json
            camera = _json.loads(camera_json.read_text())
            orig_pos = list(camera["position_m"])
            _r = analysis.get("room", {})
            _floor_width = float(_r.get("floor_width_m") or 0) or None
            camera, _ = refine_arc_position(
                image_path, _cur_render, camera,
                floor_width_m=_floor_width,
            )
            if camera["position_m"] != orig_pos:
                _do_rerender("render_arc.png")
            else:
                print("[prompt_gen] Stage X: no arc change.")
        except Exception as e:
            print(f"[prompt_gen] Stage X failed: {e}")

    # ── Stage A (verify): re-align corner after arc move ─────────────────────
    # Stage X changes cam_x/cam_z, which shifts where the corner appears in the
    # image.  Re-run corner yaw alignment so the corner is pixel-aligned before
    # tilt and zoom begin.  Skipped when no concave corner was detected.
    _cur_render = results.get("render_path") or str(render_path)
    if _has_concave_corner and Path(_cur_render).exists() and camera_json.exists():
        try:
            import json as _json
            camera = _json.loads(camera_json.read_text())
            orig_look = list(camera["look_at_m"])
            camera, _ = refine_yaw_corner(
                image_path, _cur_render, camera,
                corner_px=_refine_corner_px, img_width=img_W or None,
            )
            if camera["look_at_m"] != orig_look:
                _do_rerender("render_yaw_verify.png")
            else:
                print("[prompt_gen] Stage A (verify): corner already pixel-aligned.")
        except Exception as e:
            print(f"[prompt_gen] Stage A (verify) failed: {e}")

    # ── Stage CD: jointly balance zoom + tilt from reference proportions ────────
    # Replaces the old separate D(pre) + C + D(final) pipeline.
    # Solves cam_z and tilt together so rendered proportions match the reference
    # WITHOUT over-zooming to show ceiling when the reference intentionally hides it.
    _ceiling_visible_in_ref: bool = True   # default; updated by Stage CD
    _cur_render = results.get("render_path") or str(render_path)
    if Path(_cur_render).exists() and camera_json.exists():
        try:
            import json as _json
            camera = _json.loads(camera_json.read_text())
            orig_pos  = list(camera["position_m"])
            orig_look = list(camera["look_at_m"])
            _room_depth, _ceiling_h = _room_meta()
            camera, expanded_depth, _ceiling_visible_in_ref = balance_zoom_tilt(
                ref_image_path=image_path, render_path=_cur_render,
                camera=camera, ceiling_h=_ceiling_h, room_depth_m=_room_depth,
            )
            if expanded_depth is not None:
                _expand_mesh(expanded_depth, _ceiling_h)
            if camera["position_m"] != orig_pos or camera["look_at_m"] != orig_look:
                _do_rerender("render_balanced.png")
            else:
                print("[prompt_gen] Stage CD: no change.")
        except Exception as e:
            print(f"[prompt_gen] Stage CD failed: {e}")

    # ── Stage CD-hide: enforce no-ceiling when reference has none ────────────
    # After zoom, the camera might still show ceiling because the tilt-clamp in
    # Stage CD wasn't tight enough.  Geometrically check and tilt down until the
    # ceiling is just out of frame — keeping cam position (corner alignment) fixed.
    if not _ceiling_visible_in_ref and camera_json.exists():
        try:
            import math as _math, json as _json
            camera = _json.loads(camera_json.read_text())
            _, _ceiling_h = _room_meta()
            _pos  = camera["position_m"]
            _look = camera["look_at_m"]
            _cam_y, _cam_z = float(_pos[1]), float(_pos[2])
            _look_z = float(_look[2])
            _vfov_h = float(camera.get("vfov_deg", 43.0))
            # Current tilt
            _dz = _cam_z - _look_z
            _cur_tilt = _math.degrees(_math.atan2(float(_look[1]) - _cam_y, _dz)) if abs(_dz) > 1e-4 else 0.0
            # Angle to ceiling from camera horizontal
            _ceil_angle = _math.degrees(_math.atan2(_ceiling_h - _cam_y, _cam_z))
            # Tilt at which ceiling just enters top of frame
            _tilt_ceiling_limit = _ceil_angle - _vfov_h / 2.0
            if _cur_tilt > _tilt_ceiling_limit - 0.3:
                # Ceiling is visible (or right at the edge) — tilt down to hide it
                _hide_tilt = _tilt_ceiling_limit - 0.5   # 0.5° margin below ceiling
                _new_look_y = _cam_y + _dz * _math.tan(_math.radians(_hide_tilt))
                camera["look_at_m"] = [round(float(_look[0]), 4),
                                       round(_new_look_y, 4),
                                       round(float(_look[2]), 4)]
                print(f"[prompt_gen] Stage CD-hide: ref has no ceiling — tilt "
                      f"{_cur_tilt:+.2f}°→{_hide_tilt:+.2f}° to hide ceiling")
                _do_rerender("render_no_ceiling.png")
            else:
                print(f"[prompt_gen] Stage CD-hide: ceiling already out of frame "
                      f"(tilt={_cur_tilt:+.2f}°, limit={_tilt_ceiling_limit:+.2f}°) — no change.")
        except Exception as e:
            print(f"[prompt_gen] Stage CD-hide failed: {e}")

    return results
