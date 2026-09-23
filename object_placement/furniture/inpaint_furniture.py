"""
inpaint_furniture.py — upscale & complete segmented furniture objects
                       using Qwen image-edit.

Takes the canvas images produced by segment_furniture.py (object on grey
background, possibly occluded or low-res) and asks Qwen image-edit to:
  • generate a sharp high-resolution, fully complete version of the object
  • use the canvas as a reference for shape, colour, material, and style
  • remove decorative clutter (pillows, books, lamps) left on top
  • render the object in isolation on a solid grey background

Pipeline:
  1. Load segment_results.json from <output_dir>/furniture/
  2. For each segment:
       a. Load its canvas PNG
       b. Build a type-specific Qwen image-edit prompt
       c. Send canvas + prompt to the image-edit server
       d. Save output as inpaint_<idx>_<type>.png
  3. Update segment_results.json with inpaint_file fields.

Usage:
    python -m object_placement.furniture.inpaint_furniture \\
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
from PIL import Image

from Qwen.image_edit_adapter import edit_image as _edit_image

EDIT_API_URL = "http://localhost:8000/generate"   # legacy; routed via adapter
VLM_API_URL  = "http://localhost:8080/v1/chat/completions"

# Route VLM chat through the shared backend router (Qwen local / gpt-5.5 NVIDIA).
from object_placement.vlm_backend import vlm_post as _vlm_post
OBJ_IMG_SIZE = 512    # output resolution
EDIT_STEPS   = 20
MAX_RETRIES  = 2      # max quality-check retries before accepting

# Types whose output is reliably anchored by a good canvas — skip VLM quality
# check entirely (shape/texture are straightforward to recover from reference).
_ALWAYS_SKIP_QC: set[str] = set()  # nothing skipped unconditionally

# Types that skip the quality check only when the canvas is large / not heavily
# occluded (fg ≥ _GOOD_CANVAS_FG).  With a clear reference the model just needs
# to enhance details, not be judged for shape correctness.
_SKIP_QC_IF_GOOD = {"sofa", "bed", "bookcase", "cabinet", "armchair"}
_GOOD_CANVAS_FG  = 0.35   # ≥35% foreground pixels = "large and not heavily occluded"

# Large pieces whose 3D-gen breaks when the inpaint reproduces a partial/occluded
# crop.  When SAM fails and the canvas is a raw bbox RECTANGLE (mask_fallback==
# "bbox_rect") — which for a foreground bed grabs the WHOLE bed wall (headboard +
# nightstands + lamps + the bed cut off at the frame edge) — the faithful path
# clones that wide strip and Hunyuan makes a stripe mesh.  For these types we
# instead RECONSTRUCT a single complete standalone object, text-driven.
_RECONSTRUCT_TYPES = {
    "bed", "sofa", "sectional", "couch", "loveseat", "daybed",
    "dining_table", "desk", "cabinet", "wardrobe", "dresser", "sideboard",
}
_RECONSTRUCT_DESCRIPTOR = {
    "bed":          "a headboard, a full rectangular mattress with bedding, and the bed frame/base, normal bed proportions",
    "sofa":         "a full seat, back cushions, both armrests, and a visible base/legs",
    "dining_table": "a full rectangular top with all four legs",
    "desk":         "a full flat rectangular top with normal legs/support",
    "cabinet":      "a full rectangular body with doors/drawers and a flat top",
    "wardrobe":     "a full tall rectangular cabinet with doors",
    "dresser":      "a full rectangular chest of drawers",
    "sideboard":    "a full low rectangular cabinet with doors/drawers",
}

# ── Per-type generation descriptors ──────────────────────────────────────────

# Completion instructions per type: what to extend/fill in and what to remove.
# The model sees a canvas where the object sits on a plain grey background —
# grey areas are "missing" regions that must be filled to complete the object.
_TYPE_DESCRIPTOR: dict[str, str] = {
    "sofa": (
        "isolate the sofa from this scene — remove the room background, wall, floor, and any other furniture; "
        "complete any occluded parts of the sofa (backrest, armrests, seat, legs) by inferring from visible regions; "
        "KEEP the sofa's overall silhouette EXACTLY as shown — if it is a STRAIGHT sofa, keep it straight; "
        "do NOT turn it into an L-shaped, sectional, or corner sofa, and do not add extra seats or a chaise; "
        "KEEP the sofa's throw pillows, cushions, and any blanket or throw draped on it EXACTLY as shown "
        "(same colours, count, and placement) — do NOT remove them, they are part of the sofa's look; "
        "remove only unrelated clutter that is NOT soft furnishing (remote control, books, magazines, cups, mugs); "
        "result: a single complete sofa with its pillows and blanket on a plain grey background"
    ),
    "chair": (
        "isolate the chair from this scene — remove the room background and all other furniture; "
        "complete any hidden parts (seat, backrest, legs) by inferring from what is visible; "
        "clear the seat of any objects; "
        "result: a single complete chair on a plain grey background"
    ),
    "armchair": (
        "isolate the armchair from this scene — remove background and surrounding furniture; "
        "complete any occluded parts (seat, backrest, armrests, legs) from visible context; "
        "it is a SINGLE-SEAT armchair with exactly ONE seat cushion — do NOT widen or extend "
        "it into a two-seat loveseat or a multi-seat sofa; keep it one single chair; "
        "remove pillows and throws; "
        "result: a single complete single-seat armchair on a plain grey background"
    ),
    "coffee_table": (
        "isolate the table from this scene — remove background, floor, and ALL other furniture "
        "(chairs, sofas, stools, lamps — nothing else should remain); "
        "complete any occluded surface areas (e.g. tabletop hidden behind a sofa) — "
        "if the table is round, the top must be a full unbroken circle; "
        "if legs are not visible or hidden by the floor/carpet, add appropriate thin legs "
        "consistent with the table style; "
        "preserve the exact shape, material, and colour of the visible parts; "
        "result: ONLY a single complete table with legs visible on a plain grey background, no other objects"
    ),
    "dining_table": (
        "isolate the dining table from this scene — remove background and surrounding objects; "
        "complete any hidden tabletop area; "
        "if legs are not visible or hidden, add appropriate legs consistent with the table style; "
        "remove all dishes, objects, and settings from the surface; "
        "result: a single complete dining table with all legs visible on a plain grey background"
    ),
    "desk": (
        "isolate the desk from this scene — remove background, wall, and surrounding objects; "
        "complete any hidden parts of the surface; "
        "if legs or base are not visible, add appropriate legs consistent with the desk style; "
        "remove all items from the desktop; "
        "result: a single complete desk with legs visible on a plain grey background"
    ),
    "bookcase": (
        "isolate the bookcase from this scene — remove background, wall, and surrounding objects; "
        "complete any hidden portions of the frame and shelves; "
        "shelves should be empty — remove books and decorations; "
        "result: a single complete bookcase on a plain grey background"
    ),
    "cabinet": (
        "isolate the cabinet from this scene — remove background and surrounding furniture; "
        "complete any hidden panels, doors, or hardware; "
        "clean top surface — no objects; "
        "result: a single complete cabinet on a plain grey background"
    ),
    "bed": (
        "isolate the bed from this scene — remove background and surrounding furniture; "
        "complete any hidden portions of the frame, headboard, and mattress; "
        "KEEP the bed's pillows, cushions, bedding, blanket and throws EXACTLY as shown "
        "(same colours, count, and placement) — do NOT remove them, they are part of the "
        "bed's look; "
        "result: a single complete made bed with its pillows on a plain grey background"
    ),
    "plant": (
        "isolate the plant or tree from this scene — remove background, wall, floor, and surrounding furniture; "
        "complete any hidden portions (pot base, upper leaves, full stem height) from what is visible; "
        "result: a single complete plant on a plain grey background"
    ),
    "carpet": (
        "redraw this carpet as a flat rectangular pattern viewed from "
        "DIRECTLY ABOVE — as if the camera is on the ceiling looking "
        "straight down at the floor. The four edges of the carpet must be "
        "perfectly horizontal and vertical, parallel to the image borders. "
        "All four corners must be 90° right angles. NO perspective, NO "
        "foreshortening, NO depth, NO tilt — every part of the carpet is "
        "the same distance from the camera. "
        "Use the visible pattern, weave, texture, fringe, and colour to "
        "reconstruct the full carpet (fill in any region currently occluded "
        "by furniture or shadows by extrapolating the existing pattern). "
        "Output: a single complete axis-aligned rectangular carpet centred "
        "on a plain grey (185,185,185) background. The carpet should fill "
        "roughly 75–85 % of the image area with a clear grey border on all "
        "four sides — the carpet does NOT touch any edge of the image."
    ),
    "rug": (
        "redraw this rug as a flat rectangular pattern viewed from "
        "DIRECTLY ABOVE — as if the camera is on the ceiling looking "
        "straight down at the floor. The four edges must be perfectly "
        "horizontal and vertical, parallel to the image borders. All four "
        "corners must be 90° right angles. NO perspective, NO "
        "foreshortening, NO depth, NO tilt. "
        "Use the visible pattern, colour, and fringe to reconstruct the "
        "full rug (fill in any region currently occluded by furniture or "
        "shadows by extrapolating the existing pattern). "
        "Output: a single complete axis-aligned rectangular rug centred on "
        "a plain grey (185,185,185) background, filling roughly 75–85 % of "
        "the image area with a clear grey border on all four sides."
    ),
    "other": (
        "isolate this furniture piece from the scene — remove the room background and surrounding objects; "
        "complete any occluded parts from visible context; "
        "result: a single complete object on a plain grey background"
    ),
}

# ── Per-type negative prompts ─────────────────────────────────────────────────

_GLOBAL_NEG = (
    "blurry, out of focus, low resolution, low quality, pixelated, jpeg artefacts, "
    "grainy, noisy, soft focus, "
    "room interior, wall, floor, ceiling, window, door, curtain, "
    "other furniture around it, background objects, second object, two objects, multiple objects, "
    "chair next to it, stool, extra furniture, "
    "people, person, hands, border, watermark, "
    "cropped edges, cut-off, incomplete shape, missing parts, "
    "distorted, warped, deformed"
)

_TYPE_NEG: dict[str, str] = {
    "sofa": (
        "blurry, low resolution, low quality, "
        "remote control, book, magazine, tray, cup, mug, vase, lamp, "
        "people, person sitting, room, wall, floor, background, "
        "cropped, cut-off, incomplete, distorted"
    ),
    "chair": (
        "blurry, low resolution, low quality, "
        "throw pillow, decorative pillow, cushion on top, pillow on seat, blanket, throw, "
        "people, person sitting, room, wall, floor, background, "
        "cropped, cut-off, incomplete, distorted"
    ),
    "armchair": (
        "blurry, low resolution, low quality, "
        "throw pillow, decorative pillow, cushion on top, blanket, throw, "
        "people, person sitting, room, wall, floor, background, "
        "cropped, cut-off, incomplete, distorted"
    ),
    "coffee_table": (
        "blurry, low resolution, low quality, "
        "hole in surface, gap in tabletop, missing chunk, void in middle, incomplete disc, "
        "chair, office chair, stool, seat, sofa, couch, other furniture, second object, "
        "books, magazines, remote control, tray, bowl, vase, candle, cup, mug, "
        "people, room, wall, floor, background, "
        "cropped, cut-off, incomplete, distorted"
    ),
    "dining_table": (
        "blurry, low resolution, low quality, "
        "plates, glasses, cutlery, food, bowl, vase, candle, tablecloth, "
        "people, room, wall, floor, background, "
        "cropped, cut-off, incomplete, distorted"
    ),
    "desk": (
        "blurry, low resolution, low quality, "
        "monitor, keyboard, mouse, books, lamp, pen, cup, paper, notebook, phone, "
        "people, room, wall, floor, background, "
        "cropped, cut-off, incomplete, distorted"
    ),
    "bookcase": (
        "blurry, low resolution, low quality, "
        "books, decorations, vase, picture frame, plant, objects on shelves, "
        "people, room, wall, floor, background, "
        "cropped, cut-off, incomplete, distorted"
    ),
    "bed": (
        "blurry, low resolution, low quality, "
        "people, person lying, room, wall, floor, background, "
        "cropped, cut-off, incomplete, distorted"
    ),
    "carpet": (
        "blurry, low resolution, low quality, "
        "two carpets, two rugs, multiple rugs, layered rugs, small rug on top, rug on rug, "
        "second carpet, extra rug, stacked rugs, area rug on top of carpet, "
        "perspective view, angled view, side view, oblique view, three-quarter view, "
        "looking from above at an angle, isometric, axonometric, "
        "3D depth, foreshortening, vanishing point, depth cues, "
        "trapezoid shape, parallelogram, diamond shape, rhombus, kite shape, "
        "tilted rectangle, rotated rectangle, tapered edges, narrowing edges, "
        "non-parallel edges, edges converging, "
        "rounded corners, soft corners, oval shape, circular shape, "
        "different pattern, wrong colour, generic texture, invented design, "
        "irregular shape, wavy edges, "
        "off-centre, edge of image, touching the image edge, partial carpet, half carpet, "
        "carpet only filling small area of frame, mostly grey background, "
        "furniture on top, shadows, people, room, wall, floor visible beside it, "
        "cropped, cut-off, incomplete"
    ),
    "rug": (
        "blurry, low resolution, low quality, "
        "two rugs, multiple rugs, layered rugs, small rug on top, rug on rug, second rug, "
        "rounded corners, soft corners, different pattern, wrong colour, generic texture, "
        "irregular shape, wavy edges, trapezoid, parallelogram, diamond shape, kite shape, "
        "tilted rectangle, rotated rectangle, "
        "perspective view, angled view, oblique view, foreshortening, vanishing point, "
        "off-centre, partial rug, rug only filling small area of frame, mostly grey background, "
        "furniture on top, shadows, people, room, wall, floor visible beside it, "
        "cropped, cut-off, incomplete"
    ),
}

# ── Per-type CFG scale ────────────────────────────────────────────────────────

_TYPE_CFG: dict[str, float] = {
    "carpet":       5.5,   # high: must follow reference pattern and corner geometry exactly
    "rug":          5.5,
    "sofa":         4.0,   # medium: preserve shape/colour, text cleans up surface
    "armchair":     4.0,
    "plant":        4.0,
    "bookcase":     5.0,   # higher: need to fill empty shelves convincingly
    "bed":          4.5,
}
_DEFAULT_CFG = 5.0

# ── Helpers ───────────────────────────────────────────────────────────────────

_BG_GREY    = (185, 185, 185)   # background colour used in canvas images
_CARPET_PAD = 0.05              # padding fraction around carpet/rug when building canvas

# Named colours: (R, G, B, label)
_COLOUR_TABLE = [
    (245, 245, 220, "beige"),
    (255, 255, 255, "white"),
    (240, 230, 210, "cream"),
    (210, 180, 140, "tan"),
    (188, 143, 143, "rose"),
    (160, 120,  80, "brown"),
    (101,  67,  33, "dark brown"),
    (139,  90,  43, "walnut"),
    (205, 133,  63, "light wood"),
    (255, 165,   0, "orange"),
    (220,  80,  60, "red"),
    (180,  30,  30, "dark red"),
    (100, 150,  80, "olive green"),
    ( 50, 130,  70, "green"),
    ( 30,  80, 160, "blue"),
    ( 70, 130, 180, "steel blue"),
    (100,  80, 160, "purple"),
    ( 80,  80,  80, "dark grey"),
    (105, 105, 105, "charcoal"),
    (128, 128, 128, "grey"),
    (169, 169, 169, "light grey"),
    ( 30,  30,  30, "black"),
    (210, 180, 160, "light beige"),
    (255, 250, 240, "off-white"),
]


def _canvas_fg_fraction(canvas: Image.Image) -> float:
    """Return the fraction of pixels that are not background grey (0.0–1.0)."""
    arr = np.array(canvas.convert("RGB"), dtype=np.float32)
    bg  = np.array(_BG_GREY, dtype=np.float32)
    diff = np.abs(arr - bg).max(axis=2)
    return float((diff > 25).mean())


def _dominant_colour(canvas: Image.Image) -> str:
    """Return a colour name describing the dominant non-background colour in the canvas."""
    arr = np.array(canvas.convert("RGB"), dtype=np.float32)
    bg  = np.array(_BG_GREY, dtype=np.float32)

    # Mask out background pixels (within 25 units of the grey background)
    diff = np.abs(arr - bg).max(axis=2)
    fg   = arr[diff > 25]

    if len(fg) < 100:
        return ""   # not enough foreground pixels to determine colour

    def _nearest(px) -> str:
        best_name, best_dist = "", float("inf")
        for r, g, b, name in _COLOUR_TABLE:
            d = float(np.sqrt((px[0]-r)**2 + (px[1]-g)**2 + (px[2]-b)**2))
            if d < best_dist:
                best_dist, best_name = d, name
        return best_name

    median = np.median(fg, axis=0)   # (R, G, B)

    # A single median destroys anything patterned: a bold black-and-white rug
    # medians to mid-grey, and the prompt that reaches the editor becomes
    # "grey woven fabric" — which is what flattened pexels_2343465's Moroccan
    # carpet into a plain pale slab.  The median is only a fair summary when
    # the object is roughly one colour.  Detect the two-tone case from the
    # luminance spread and name BOTH ends instead of averaging them away.
    lum = fg @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lo_px = np.median(fg[lum <= np.percentile(lum, 20)], axis=0)
    hi_px = np.median(fg[lum >= np.percentile(lum, 80)], axis=0)
    spread = float(np.linalg.norm(hi_px - lo_px))
    _PATTERN_MIN = float(os.environ.get("SCENEWEAVE_PATTERN_SPREAD", "110"))
    if spread >= _PATTERN_MIN:
        lo_name, hi_name = _nearest(lo_px), _nearest(hi_px)
        if lo_name and hi_name and lo_name != hi_name:
            return f"patterned {lo_name} and {hi_name}"

    return _nearest(median)


def _enforce_grey_border(
    img: Image.Image,
    margin_frac: float = 0.10,
    bg_tol: int = 18,
) -> Image.Image:
    """Guarantee a grey border around the inpainted object.

    Image-edit models routinely ignore "leave margin" instructions and let the
    object run right to the edge of the canvas, leaving only a sliver of white
    or grey at some edges.  That's the worst possible input for the downstream
    SAM segmentation pass — too little grey for SAM to recognise background,
    too much to treat as carpet.

    This post-processing step:
      1. Detects the tight bbox of non-grey content.
      2. Centres it on a larger grey canvas with `margin_frac` of the object's
         long edge as background on every side.
      3. Resizes back to the original image dimensions.

    The result is guaranteed to have at least ~`margin_frac * 100` % grey
    border on every side regardless of what the model produced.
    """
    arr   = np.array(img.convert("RGB"), dtype=np.float32)
    bg    = np.array(_BG_GREY, dtype=np.float32)
    is_fg = np.abs(arr - bg).max(axis=2) > bg_tol

    rows = np.any(is_fg, axis=1)
    cols = np.any(is_fg, axis=0)
    if not rows.any() or not cols.any():
        return img

    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))

    crop = arr[r0:r1 + 1, c0:c1 + 1].astype(np.uint8)
    ch, cw = crop.shape[:2]

    margin = int(round(max(cw, ch) * margin_frac))
    side   = max(cw, ch) + 2 * margin
    canvas = np.full((side, side, 3), _BG_GREY, dtype=np.uint8)
    y0 = (side - ch) // 2
    x0 = (side - cw) // 2
    canvas[y0:y0 + ch, x0:x0 + cw] = crop

    out = Image.fromarray(canvas).resize(img.size, Image.LANCZOS)
    return out


# Material signatures: (hue_range_deg, saturation_range, value_range, label)
# Checked in order; first match wins.
_MATERIAL_TABLE: list[tuple[str, str]] = [
    # type  →  material keyword
    ("sofa",         "upholstered"),
    ("armchair",     "upholstered"),
    ("chair",        "upholstered"),
    ("bed",          "upholstered"),
    ("coffee_table", "wood"),
    ("dining_table", "wood"),
    ("desk",         "wood"),
    ("bookcase",     "wood"),
    ("cabinet",      "wood"),
    ("carpet",       "woven fabric"),
    ("rug",          "woven fabric"),
    ("plant",        "organic"),
]

# Per-type material refinement based on dominant colour (heuristic)
_LEATHER_COLOURS = {"black", "dark brown", "brown", "walnut", "charcoal", "dark grey"}
_FABRIC_TYPES    = {"sofa", "armchair", "chair", "bed"}


def _detect_material(obj_type: str, dominant_colour: str) -> str:
    """Return a material keyword to prepend to the generation prompt."""
    if obj_type in _FABRIC_TYPES:
        if dominant_colour in _LEATHER_COLOURS:
            return "leather"
        return "fabric"
    for t, mat in _MATERIAL_TABLE:
        if t == obj_type:
            return mat
    return ""


def _encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_VLM_CANVAS_DESCRIBE_PROMPT = """\
You are helping an image inpainting system isolate and complete ONE specific object.

Image 1 (CROP): a raw crop from the original room photo. It may contain MULTIPLE objects — furniture, plants, decorations overlapping each other.
Image 2 (CANVAS): the segmentation mask result — it shows ONLY the target object isolated on a grey background. Grey areas are regions where the object is occluded or cut off.

IMPORTANT: The crop may show several objects, but ONLY the object highlighted in the canvas is the target. Identify which object in the crop corresponds to the canvas shape, and describe ONLY that object. Ignore all other objects in the crop.

The object was auto-detected as "{obj_type}" but this label may be WRONG — trust what you see.

Describe in 3-4 sentences:
1. What the TARGET object ACTUALLY is (match it between crop and canvas by shape/position), e.g. "a woven bench with light wood legs" or "a brass bowl-shaped side table".
2. Which parts of THIS object are missing, occluded, or CUT OFF (grey areas in the canvas, OR clipped at the edge of the crop) and must be reconstructed so the object becomes COMPLETE and free-standing. Explicitly call out any legs, base, pedestal, or stem that must continue all the way DOWN to the floor, and any portion that extends beyond the frame — the finished object must be whole with NOTHING cropped or floating.
3. If this object normally carries SOFT FURNISHINGS resting on it (throw pillows, cushions, a blanket or throw on a sofa / bed / armchair), treat those as PART of the object: describe their colour, count and placement so they are KEPT exactly as shown. Do NOT treat pillows or blankets as occlusions to remove — only genuine clutter (remotes, cups, books, plants in front) should be removed.
4. Key material/colour/texture details to preserve.

Do NOT describe other objects visible in the crop. Output ONLY the description, no JSON, no markdown.
"""

_VLM_CANVAS_DESCRIBE_PROMPT_SINGLE = """\
You are helping an image inpainting system complete a partially-visible object.

This image shows a SINGLE segmented object on a grey background. Grey areas are missing/occluded regions that need to be filled.
The object was auto-detected as "{obj_type}" but this label may be WRONG — trust what you see.

Describe in 3-4 sentences:
1. What the object ACTUALLY is (shape, function).
2. Which parts are missing or CUT OFF (grey areas, or clipped at the edge) that need reconstruction so the object becomes COMPLETE and free-standing — explicitly include any legs, base, pedestal or stem that must reach all the way DOWN to the floor; the finished object must be whole with NOTHING cropped or floating.
3. If this object normally carries SOFT FURNISHINGS (throw pillows, cushions, a blanket/throw on a sofa/bed/armchair), treat them as PART of the object and KEEP them as shown — do NOT remove pillows or blankets.
4. Key material/colour/texture details to preserve.
Output ONLY the description, no JSON, no markdown.
"""


def _vlm_describe_canvas(canvas: Image.Image, obj_type: str,
                         crop: Image.Image | None = None) -> str | None:
    """Ask VLM to describe what's actually in the canvas and what needs completing.

    If crop (raw bbox from original image) is available, sends both images so VLM
    can see the object in its room context — much more informative than the
    isolated canvas alone.
    """
    try:
        if crop is not None:
            prompt_text = _VLM_CANVAS_DESCRIBE_PROMPT.format(
                obj_type=obj_type.replace("_", " "))
            content: list[dict] = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_encode_pil(crop)}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_encode_pil(canvas)}"}},
            ]
        else:
            prompt_text = _VLM_CANVAS_DESCRIBE_PROMPT_SINGLE.format(
                obj_type=obj_type.replace("_", " "))
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_encode_pil(canvas)}"}},
            ]
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 250,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _vlm_post(payload, timeout=30)
        resp.raise_for_status()
        desc = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"  [canvas-describe] {desc}")
        return desc
    except Exception as e:
        print(f"  [canvas-describe] VLM unavailable: {e}")
        return None


_VLM_QUALITY_PROMPT = """\
You are evaluating an AI-generated furniture image against a reference canvas.

The FIRST image is the reference canvas (segmented from a real room photo).
The SECOND image is the AI-generated result.

The result should show a SINGLE, COMPLETE {obj_type} isolated on a plain grey background,
and its overall shape and structure must be consistent with the reference.

Rate the quality on these criteria:
1. Is the result clearly recognisable as a {obj_type}?
2. Is the object complete (no missing limbs, legs, or major surfaces)?
3. Is it isolated — no other furniture, no room background, no people?
4. Is the shape coherent and consistent with the reference? (e.g. if the reference shows a round table, the result must also be round — not square or rectangular)
5. Is there only ONE {obj_type} visible?
6. Is the surface fully solid with no holes, gaps, voids, or missing patches? (a round tabletop must be a complete disc — any missing chunk in the middle is a failure)

Return a JSON object with:
  "ok": true if ALL six criteria pass, false otherwise
  "score": integer 1–5  (5 = perfect, 1 = completely wrong)
  "reason": one short sentence explaining any failure

Output ONLY the JSON, no markdown, no explanation.
"""


def _vlm_quality_check(result: Image.Image, obj_type: str,
                       canvas: Image.Image | None = None) -> dict:
    """Ask the VLM to rate the generated image quality against the reference canvas.
    Returns {"ok": bool, "score": int, "reason": str} or {"ok": True} on failure."""
    try:
        content: list[dict] = [
            {"type": "text",
             "text": _VLM_QUALITY_PROMPT.format(obj_type=obj_type.replace("_", " "))},
        ]
        if canvas is not None:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_encode_pil(canvas)}"}})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_encode_pil(result)}"}})

        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"  [quality-check] VLM unavailable: {e}")
    return {"ok": True}   # can't check → accept


# Seating that should be stripped of decorative throw pillows (so the pillows are
# placed separately by the decoration stage).  Beds intentionally KEEP their
# pillows/bedding.
_STRIP_PILLOW_TYPES = {"sofa", "couch", "sectional", "loveseat", "daybed"}


def _vlm_has_pillows(result: Image.Image, obj_type: str) -> bool:
    """Ask the VLM whether the generated seating still has any loose/decorative
    THROW pillows resting on it (vs the fixed seat/back cushions that are part of
    the frame). Returns True if throw pillows remain (→ retry removal), False if
    clean or on any failure (so we never loop forever)."""
    try:
        prompt = (
            f"This image shows a {obj_type.replace('_', ' ')} on a plain background. "
            "Are there any LOOSE / DECORATIVE THROW PILLOWS resting on the seat or "
            "back — the kind used as accent decoration, NOT the fixed attached seat "
            "and back cushions that are part of the sofa frame? "
            "Answer with ONLY one word: YES (throw pillows present) or NO (none)."
        )
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_encode_pil(result)}"}},
            ]}],
            "max_tokens": 60,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].upper()
        toks = re.findall(r"\b(YES|NO)\b", raw)
        return bool(toks) and toks[-1] == "YES"
    except Exception as e:
        print(f"  [pillow-check] VLM unavailable: {e}")
        return False


def _strip_pillows(result: Image.Image, obj_type: str,
                   colour: str = "", material: str = "", max_iter: int = 3) -> Image.Image:
    """Iteratively remove decorative throw pillows from a seating image via focused
    image-edits, verifying each pass with the VLM. The edit reuses the current
    result as the reference and pushes a strong 'no throw pillows' prompt at high
    CFG, so the sofa's shape/colour/fabric stay identical while the loose pillows
    are replaced by the sofa's own upholstery. Returns the cleaned image (or the
    last attempt if it can't fully clear them)."""
    pre = (f"{colour}, " if colour else "") + (f"{material} " if material else "")
    name = obj_type.replace("_", " ")
    neg = ("throw pillow, decorative pillow, accent pillow, orange pillow, loose "
           "pillow, cushion placed on top, stacked cushions, blanket, throw, "
           "blurry, low quality, room, wall, floor, background")
    for it in range(max_iter):
        if not _vlm_has_pillows(result, obj_type):
            if it:
                print(f"  [pillow-strip] {name} clean after {it} pass(es)")
            return result
        print(f"  [pillow-strip {it + 1}] throw pillows present — removing …")
        prompt = (
            f"{pre}a {name} with NO decorative throw pillows. "
            "Remove EVERY loose accent/throw pillow resting on the seat or back, "
            "and fill those areas with the sofa's own plain attached back cushions "
            "and matching upholstery. Keep the exact same overall shape, colour, "
            "fabric texture, arms, legs, and everything else identical. A single "
            f"{name} on a plain grey background, no throw pillows, nothing stacked "
            "on the seat."
        )
        payload = {
            "prompt": prompt, "negative_prompt": neg,
            "width": OBJ_IMG_SIZE, "height": OBJ_IMG_SIZE,
            "num_inference_steps": EDIT_STEPS, "true_cfg_scale": 6.0,
            "num_images": 1, "reference_image": _encode_pil(result),
        }
        try:
            data = _edit_image(payload, timeout=300)
            result = Image.open(io.BytesIO(base64.b64decode(data["images"][0]))).convert("RGB")
        except Exception as e:
            print(f"  [pillow-strip] edit failed: {e}")
            break
    if _vlm_has_pillows(result, obj_type):
        print(f"  [pillow-strip] {name} still shows pillows after {max_iter} passes — keeping best effort")
    return result


def _build_prompt(obj_type: str, phrase: str,
                  colour: str = "", material: str = "",
                  heavily_occluded: bool = False,
                  vlm_description: str | None = None,
                  prompt_override: str | None = None,
                  preserve_appearance: bool = False,
                  reconstruct: bool = False) -> str:
    """Build the generation prompt for a furniture piece.

    If prompt_override is provided, return it verbatim (still prefixed with
    colour/material hints if available) — used for one-off requests like
    "remove only the pillows from the bed, keep the rest".

    If vlm_description is provided, use it to build a canvas-aware prompt
    instead of the generic type descriptor.  This ensures the inpainting
    completes *this specific object* rather than hallucinating from the type name.
    """
    # Colour + material prefix
    if colour:
        style_prefix = f"{colour}, "
    else:
        colour_m = re.match(
            r"^((?:light |dark |bright |deep )?\w+(?:\s+\w+)?)\s+" + re.escape(obj_type.replace("_", " ")),
            phrase, re.IGNORECASE,
        )
        style_prefix = f"{colour_m.group(1).strip()}, " if colour_m else ""
    if material:
        style_prefix = f"{material}, {style_prefix}"

    if prompt_override:
        return f"{style_prefix}{prompt_override} Render at maximum sharpness and detail, with a MATTE finish under flat, even, diffuse lighting — NO glossy reflections, NO specular highlights, NO glare or shiny glaze on any surface (a glossy sheen makes the 3D texture come out mis-coloured). Keep the EXACT colour and wood/material tone shown in the reference — do NOT warm, cool, lighten or darken it, and do NOT shift the hue; match the reference's exact shade."

    if reconstruct:
        # The reference is a PARTIAL, occluded bbox crop that also contains other
        # furniture and is cut off at the frame edges (SAM failed).  Do NOT clone it —
        # reconstruct one complete standalone object from text.
        _t = obj_type.replace("_", " ")
        _desc = _RECONSTRUCT_DESCRIPTOR.get(obj_type, f"a complete, standard-form {_t}")
        return (
            f"{style_prefix}The reference image is a PARTIAL, occluded crop that is cut off "
            f"at the edges and also contains OTHER furniture. RECONSTRUCT a single COMPLETE, "
            f"standalone {_t} — {_desc}. IGNORE the occlusion gaps, the cut-off edges, and ALL "
            f"surrounding furniture, nightstands, lamps, framed art, walls and floor. Output "
            f"ONLY that ONE complete {_t} in a natural upright orientation, with normal real-world "
            f"proportions (NOT stretched or squashed), centered on a plain solid grey background "
            f"with nothing else. MATTE finish, flat even diffuse lighting, NO glossy reflections or "
            f"specular highlights. Keep the material and colour consistent with the reference."
        )

    if vlm_description:
        # VLM-driven prompt: describe what's there and what to complete.
        # Beds are special: their bedding (mattress, fitted sheet, blanket/quilt,
        # duvet, sheets) is PART of the bed, not clutter on top — stripping
        # "blankets" leaves only the frame. Only loose decorative pillows should
        # be removed.
        if obj_type == "bed" and os.environ.get("SCENEWEAVE_KEEP_BED_PILLOWS"):
            # Flag to keep the bed's pillows (no removal) — reproduce the bed
            # exactly as shown, pillows included.
            removal_clause = (
                "KEEP the bed EXACTLY as shown — the mattress, fitted sheet, "
                "duvet/comforter, blanket, quilt, sheets, headboard, footboard, "
                "bed frame, AND all the pillows currently on it, each in its "
                "exact original colour, shape and position. Do NOT remove, add, "
                "or recolour any pillows. Only clear unrelated clutter (books, "
                "trays, cups, clothing). The bed should look made and complete."
            )
        elif obj_type == "bed":
            removal_clause = (
                "KEEP all of the bed's pillows, bedding, duvet/comforter, blanket, "
                "quilt, sheets, headboard, footboard, and frame exactly as shown — "
                "do NOT remove the pillows or any bedding. Only remove the room "
                "background and surrounding furniture. The bed should look made and "
                "complete WITH its pillows."
            )
        elif obj_type in ("sofa", "sectional", "couch", "loveseat", "armchair", "chair"):
            # Reproduce the seating EXACTLY as shown — keep its own seat/back
            # cushions and any pillows that are ALREADY present, in their exact
            # original colours (accent pillows are often a different colour; do
            # not recolour or remove them).  Critically, do NOT INVENT pillows,
            # headrests, or cushions that aren't in the reference — a plain sofa
            # must stay plain.  Only clear unrelated clutter.
            removal_clause = (
                "Reproduce this seating EXACTLY as shown — same number of "
                "cushions and pillows, no more, no fewer. KEEP its existing seat "
                "cushions, back cushions, and any pillows that are ALREADY there, "
                "each in its EXACT original colour and material (accent pillows "
                "are often a DIFFERENT colour from the sofa body — preserve them, "
                "do not recolour or blend them away). Do NOT ADD any new throw "
                "pillows, decorative cushions, or headrests that are not present "
                "in the reference; if the sofa is plain, keep it plain. Remove "
                "ONLY unrelated clutter resting on it (books, magazines, trays, "
                "cups, remotes, blankets draped over it)."
            )
        else:
            removal_clause = (
                "Remove ALL items resting on top of or placed on the object "
                "(books, magazines, trays, cups, remotes, decorations, pillows, "
                "blankets) — the surface must be completely clear."
            )
        # When the object is clearly and largely visible, the reference IS the
        # target — the model must reproduce it faithfully, NOT restyle it. This
        # is the standing "preserve original look, only complete the occluded
        # region" rule. A clear sofa should come back as the same sofa.
        if preserve_appearance:
            preserve_clause = (
                "The object is FULLY VISIBLE and clear in this reference image — "
                "this IS the target. Reproduce its EXACT appearance: identical "
                "geometry, proportions, silhouette, colour, material, and texture. "
                "Do NOT restyle, redesign, recolour, smooth, simplify, or "
                "beautify it; do not change cushion count, leg style, or shape. "
                "Only fill in regions that are genuinely missing or occluded — "
                "leave everything already visible exactly as it is. "
            )
        else:
            preserve_clause = (
                "Complete the missing regions by continuing the visible surfaces, "
                "material, and style naturally. "
            )
        return (
            f"{style_prefix}"
            f"This image shows: {vlm_description} "
            f"{preserve_clause}"
            f"{removal_clause} "
            "Do NOT add any other objects — output ONLY this single bare object on a plain grey background. "
            "Render at maximum sharpness and detail, with a MATTE finish under flat, even, diffuse lighting — NO glossy reflections, NO specular highlights, NO glare or shiny glaze on any surface (a glossy sheen makes the 3D texture come out mis-coloured). Keep the EXACT colour and wood/material tone shown in the reference — do NOT warm, cool, lighten or darken it, and do NOT shift the hue; match the reference's exact shade."
        )

    # Fallback: type-based descriptor
    descriptor = _TYPE_DESCRIPTOR.get(obj_type, _TYPE_DESCRIPTOR["other"])

    occlusion_note = (
        "The object is partially hidden by other furniture — infer the full shape from "
        "the visible portions and reconstruct the occluded parts plausibly. "
        if heavily_occluded else
        "Complete any partially hidden portions by continuing the visible surfaces naturally. "
    )

    return (
        f"{style_prefix}{descriptor} "
        f"{occlusion_note}"
        "Render at maximum sharpness and detail, with a MATTE finish under flat, even, diffuse lighting — NO glossy reflections, NO specular highlights, NO glare or shiny glaze on any surface (a glossy sheen makes the 3D texture come out mis-coloured). Keep the EXACT colour and wood/material tone shown in the reference — do NOT warm, cool, lighten or darken it, and do NOT shift the hue; match the reference's exact shade."
    )


def _call_edit(canvas: Image.Image | None, obj_type: str, phrase: str,
               crop: Image.Image | None = None,
               masked_canvas: Image.Image | None = None,
               prompt_override: str | None = None,
               reconstruct_mode: bool = False) -> Image.Image | None:
    """Generate a complete furniture image, with VLM quality-check retries.

    The canvas reference is ALWAYS used — never dropped — because even a sparse
    canvas carries material/colour/texture information that text alone cannot.

    crop: raw bbox crop from original image — passed to VLM for context.
    masked_canvas: segmented object on grey (shows missing regions) — for VLM.

    Retry strategy (up to MAX_RETRIES):
      attempt 0 — normal CFG, heavily_occluded based on fg fraction
      retry if score >= 3 (recognizable but incomplete) — reduce CFG so the
        model leans harder on the reference to fill missing regions
      retry if score <= 2 (unrecognizable / wrong shape) — boost CFG so the
        text prompt drives the correct structure
    """
    neg = _TYPE_NEG.get(obj_type, _GLOBAL_NEG)
    # Suppress glossy reflections/glaze — a specular sheen on the inpaint makes the
    # 3D-gen texture come out mis-coloured (e.g. a wood coffee table turning purple).
    neg += (", glossy, glossy finish, specular highlight, glare, reflection, "
            "shiny, sheen, glaze, lacquer, mirror finish, blown-out highlight, "
            "reflective surface, wet look")

    # canvas is always required; if somehow None (SAM-failed), still pass a white canvas
    # so the API call succeeds and the prompt drives the output
    if canvas is None:
        canvas = Image.new("RGB", (OBJ_IMG_SIZE, OBJ_IMG_SIZE), (255, 255, 255))
        fg = 0.0
    else:
        fg = _canvas_fg_fraction(canvas)

    colour   = _dominant_colour(canvas)
    material = _detect_material(obj_type, colour)
    if colour or material:
        print(f"  detected colour: {colour}  material: {material}")

    # SHORT-PROMPT mode (SCENEWEAVE_INPAINT_SHORT=1): the A/B tester showed the
    # long per-type descriptor + heavy negatives + VLM-description drives Gemini
    # to redesign/hallucinate (e.g. a "cabinet" label on a floating shelf).  A
    # short, image-led prompt on the (already real-pixel) hybrid reference scores
    # markedly higher.  Skip the VLM describe call and lighten the negatives.
    _short = bool(os.environ.get("SCENEWEAVE_INPAINT_SHORT"))
    if _short:
        vlm_desc = None
        neg = ("blurry, low quality, cropped, cut-off, incomplete, "
               "other furniture, room background, people, text, watermark")
    else:
        # Ask VLM to describe what's actually in the canvas — avoids type-label
        # hallucinations (e.g. "bookcase" label on a macramé wall hanging, or
        # "coffee_table" producing a desk-with-chair).
        # Send the masked canvas (shows missing regions) + crop (shows room context).
        vlm_canvas = masked_canvas if masked_canvas is not None else canvas
        vlm_desc = _vlm_describe_canvas(vlm_canvas, obj_type, crop=crop)

    base_cfg = _TYPE_CFG.get(obj_type, _DEFAULT_CFG)
    # Boost CFG when canvas is sparse so text guidance dominates over the noisy reference
    if fg < 0.20:
        boost    = 1.0 + (0.20 - fg) / 0.20
        # Capped at 6.0 rather than the old 9.0 so a sparse canvas doesn't hand
        # the edit entirely to the text prompt.
        #
        # NOTE: this only bites on the QWEN backend.  The Gemini adapter drops
        # true_cfg_scale outright, so on a gemini run neither the cap nor the
        # boost reaches the model.  office8's desk came back as a generic
        # castored twin-pedestal piece from a 7%-foreground canvas on a GEMINI
        # run, so whatever caused that, it was not this number — the fix there
        # is a denser canvas (tighter crop) and a more specific prompt, both of
        # which this backend actually reads.
        base_cfg = min(base_cfg * (1.0 + boost * 0.5), 6.0)

    # Clear, well-exposed reference: the object is the target, so lean HARD on it
    # (low CFG → reference dominates) and tell the prompt to preserve, not restyle.
    # Prevents the editor from redesigning a clearly-visible sofa/chair/bed.
    clear_ref = fg >= _GOOD_CANVAS_FG
    if clear_ref:
        base_cfg = min(base_cfg, 2.5)

    # Reconstruct mode: the canvas is an unreliable bbox-rect (SAM failed) of a large
    # piece — e.g. a foreground bed's whole-wall strip. Don't preserve/clone it; drive a
    # COMPLETE standalone object from text, and never skip the quality check below.
    if reconstruct_mode:
        clear_ref = False
        base_cfg  = max(base_cfg, 7.0)
        print(f"  [reconstruct] {obj_type}: SAM-failed bbox-rect canvas — "
              f"reconstructing a complete standalone object (text-driven, cfg={base_cfg:.1f})")

    # Occlusion level: heavily occluded when fg < 25%
    base_occluded = fg < 0.25

    best_result: Image.Image | None = None
    best_score  = 0
    prev_score  = None   # VLM score from the previous attempt
    # Whether true_cfg_scale reaches the model at all.  The Gemini adapter
    # documents that it silently drops the qwen-only fields, so on that backend
    # every cfg number computed above is dead — say so once, plainly, instead
    # of printing confident values nobody can act on.
    _backend      = (os.environ.get("SCENEWEAVE_IMG_EDIT") or "qwen").lower()
    _cfg_is_live  = _backend not in ("gemini", "gemini-3", "google")
    _retry_clause = ""
    if not _cfg_is_live:
        print(f"  [inpaint] backend={_backend}: true_cfg_scale is NOT sent "
              f"(adapter drops qwen-only fields) — cfg values below are "
              f"diagnostic only; retries vary the prompt instead")

    for attempt in range(MAX_RETRIES + 1):
        if attempt == 0:
            cur_cfg      = base_cfg
            cur_occluded = base_occluded
        else:
            # Retry direction depends on WHY the previous attempt failed:
            #
            # score >= 3 — object is recognisable and mostly correct but has a
            #   missing region or incomplete surface.  Reduce CFG so the model
            #   leans harder on the reference to fill the gap.
            #
            # score <= 2 — the object is substantially wrong or incomplete.
            #   Boost CFG so the text prompt drives the correct structure,
            #   BUT keep heavily_occluded=False so the reference still anchors
            #   shape identity (prevents hallucinating a completely different shape).
            #
            recognizable = prev_score is not None and prev_score >= 3
            if recognizable:
                cur_cfg      = max(base_cfg * 0.6, 1.5)   # lower CFG → more ref influence
                cur_occluded = False
                print(f"  [retry {attempt}] object recognizable (score={prev_score}) — "
                      f"reducing CFG to {cur_cfg:.1f} to lean on reference")
            else:
                cur_cfg      = min(base_cfg * 1.8, 9.0)   # higher CFG → more text influence
                cur_occluded = base_occluded   # never force heavily-occluded: keep shape anchor
                print(f"  [retry {attempt}] object incomplete (score={prev_score}) — "
                      f"boosting CFG to {cur_cfg:.1f}, keeping reference as shape anchor")
            # The CFG move above is a no-op on the Gemini backend — the adapter
            # drops steps/cfg/w/h/num_images ("qwen-only fields are silently
            # dropped").  With SCENEWEAVE_INPAINT_SHORT=1 the prompt branch below
            # also ignores cur_occluded, so on a gemini+short run every retry
            # re-sent a byte-identical request and differed only by sampling
            # noise — three QC attempts spending three edits to ask the same
            # question.  Carry the retry intent in the PROMPT, which is one of
            # the two things this backend actually reads.
            if not _cfg_is_live:
                _retry_clause = (
                    " The previous attempt left part of the object missing or "
                    "unfinished: complete every surface and edge, and keep the "
                    "shape, colour and proportions of the reference exactly."
                    if recognizable else
                    " The previous attempt produced the wrong object. Look "
                    "again at the reference and reproduce THAT specific piece — "
                    "its real silhouette, proportions and materials — not a "
                    "generic example of its category.")

        if _short and not prompt_override and not reconstruct_mode:
            t = obj_type.replace("_", " ")
            pre = f"{colour} {material} ".strip() + " " if (colour or material) else ""
            short_txt = (
                f"Extract the {t} shown and amodally complete it into a single, whole "
                f"{t} on a plain grey background. Keep its exact shape, colour, material "
                f"and proportions — do not redesign it. Remove everything else (other "
                f"furniture, room, people). Fill any occluded or cut-off parts, including "
                f"legs/base down to the floor, so nothing is cropped or floating.")
            prompt = _build_prompt(obj_type, phrase, colour=colour, material=material,
                                   prompt_override=(pre + short_txt + _retry_clause))
        else:
            prompt = _build_prompt(obj_type, phrase, colour=colour, material=material,
                                   heavily_occluded=cur_occluded,
                                   vlm_description=vlm_desc,
                                   prompt_override=((prompt_override + _retry_clause)
                                                    if prompt_override else prompt_override),
                                   preserve_appearance=(clear_ref or
                                                        (crop is not None and not reconstruct_mode)),
                                   reconstruct=reconstruct_mode)
            if _retry_clause and not prompt_override:
                prompt = prompt + _retry_clause
        mode   = f"{'occluded' if cur_occluded else 'clear'}-ref fg={fg:.0%}"
        _cfg_note = f"cfg={cur_cfg:.1f}" + ("" if _cfg_is_live else " [ignored]")
        print(f"  attempt {attempt} ({mode}, {_cfg_note}): {prompt[:100]}")

        payload: dict = {
            "prompt":              prompt,
            "negative_prompt":     neg,
            "width":               OBJ_IMG_SIZE,
            "height":              OBJ_IMG_SIZE,
            "num_inference_steps": EDIT_STEPS,
            "true_cfg_scale":      cur_cfg,
            "num_images":          1,
        }
        payload["reference_image"] = _encode_pil(canvas)

        try:
            data = _edit_image(payload, timeout=300)
            img_bytes = base64.b64decode(data["images"][0])
            result    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"  generation failed: {e}")
            continue

        # ── Seating: iteratively strip throw pillows ──────────────────────────
        # The canvas-describe prompt keeps the right upholstery but gemini also
        # keeps the decorative throw pillows.  Run focused removal edits on the
        # RESULT (strong "no throw pillows" prompt, high CFG), verifying with the
        # VLM until the seat is pillow-free.  Beds are excluded — they keep pillows.
        if obj_type in _STRIP_PILLOW_TYPES and not reconstruct_mode:
            return _strip_pillows(result, obj_type, colour=colour, material=material)

        # VLM quality check — skip for types whose output is reliably
        # anchored by a clear canvas reference.
        skip_qc = (not reconstruct_mode) and (
            obj_type in _ALWAYS_SKIP_QC
            or (obj_type in _SKIP_QC_IF_GOOD and fg >= _GOOD_CANVAS_FG)
        )
        if skip_qc:
            print(f"  quality check skipped (type={obj_type}, fg={fg:.0%})")
            return result

        qc = _vlm_quality_check(result, obj_type, canvas=canvas)
        score = qc.get("score", 3)
        ok    = qc.get("ok", True)
        print(f"  quality: ok={ok} score={score} — {qc.get('reason', '')}")
        prev_score = score

        if score > best_score:
            best_score  = score
            best_result = result

        if ok:
            return result   # good enough — use immediately

        if attempt < MAX_RETRIES:
            direction = "reducing CFG (more reference)" if score >= 3 else "boosting CFG (more text)"
            print(f"  quality too low — retrying ({direction}) …")

    if best_result is not None:
        print(f"  using best result (score={best_score}) after {MAX_RETRIES + 1} attempts")
    return best_result


_VLM_INDOOR_CHECK_PROMPT = """\
Look at this cropped region from a room photo. It was detected as a "{obj_type}".

Is this an actual physical object INSIDE the room (e.g. a potted plant on the floor, \
furniture, a decoration on a shelf)?

Or is it something OUTSIDE the room visible through a window (e.g. trees, sky, \
buildings seen through glass), a reflection, or part of the room structure itself \
(wall, floor, ceiling, window frame)?

Answer with ONLY a JSON object:
  {{"indoor": true}} if it is a real indoor object that could be picked up and moved
  {{"indoor": false, "reason": "..."}} if it is outdoor scenery, a reflection, or room structure
"""


def _vlm_is_indoor_object(crop_path: Path, obj_type: str) -> bool:
    """Ask VLM whether the detected object is actually inside the room."""
    try:
        img = Image.open(crop_path).convert("RGB")
        content = [
            {"type": "text",
             "text": _VLM_INDOOR_CHECK_PROMPT.format(obj_type=obj_type.replace("_", " "))},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_encode_pil(img)}"}},
        ]
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 100,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _vlm_post(payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group())
            indoor = result.get("indoor", True)
            reason = result.get("reason", "")
            print(f"  [indoor-check] indoor={indoor}"
                  + (f" — {reason}" if reason else ""))
            return bool(indoor)
    except Exception as e:
        print(f"  [indoor-check] VLM unavailable: {e} — assuming indoor")
    return True  # fail-open: assume indoor if VLM is unavailable


def _build_hybrid_ref(seg: dict, seg_dir: Path,
                      crop_img: Image.Image | None,
                      masked_canvas: Image.Image | None) -> Image.Image | None:
    """Build a hybrid reference image for the edit model.

    Takes the raw crop and greys out pixels far from the SAM mask, keeping the
    object's full shape (including items sitting on it) while removing other
    furniture that would be reproduced by the edit model.

    Falls back to masked_canvas or crop if mask isn't available.
    """
    mask_file = seg.get("mask_file")
    box_px = seg.get("box_px")

    # Try to build hybrid from crop + mask
    if (crop_img is not None and mask_file and box_px
            and (seg_dir / mask_file).exists()):
        try:
            from scipy.ndimage import binary_dilation, label
            full_mask = np.array(Image.open(seg_dir / mask_file).convert("L"))
            x1, y1, x2, y2 = box_px
            # Crop mask to same bbox as the crop image
            mask_crop = full_mask[y1:y2, x1:x2]
            # Resize mask to match crop_img size (crop may have been resized)
            cw, ch = crop_img.size
            if mask_crop.shape[:2] != (ch, cw):
                mask_crop = np.array(
                    Image.fromarray(mask_crop).resize((cw, ch), Image.NEAREST))
            # Dilate mask to include items sitting on/near the object
            struct = np.ones((31, 31), dtype=bool)
            dilated = binary_dilation(mask_crop > 127, structure=struct)

            # ── Coherence guard ──────────────────────────────────────────
            # The hybrid is only a useful reference if it still LOOKS like one
            # object.  When the SAM mask comes back fragmentary, intersecting
            # it with the crop yields a scatter of disconnected shards, and
            # dilation then drags in whatever sits nearby — in office8 the
            # desk's reference was 13 blobs covering 7.4% of the frame, the
            # largest only 25.7% of the foreground, and two of the shards were
            # the OFFICE CHAIR'S CHROME CASTERS.  The edit model reproduced
            # exactly that: a walnut desk on chrome casters.  It was being
            # faithful to a corrupted reference, which is why neither lowering
            # CFG nor simplifying the prompt changed the result — feeding the
            # plain masked canvas instead produces the correct desk.
            # Previously the fallback chain below was reachable ONLY via an
            # exception, so a degenerate hybrid was always returned.
            _fill = float(dilated.mean())
            _lab, _n = label(dilated)
            _sizes = np.bincount(_lab.ravel())[1:]
            _largest = float(_sizes.max() / _sizes.sum()) if _sizes.size else 0.0
            _MIN_FILL = float(os.environ.get("SCENEWEAVE_HYBRID_MIN_FILL", "0.15"))
            _MIN_BLOB = float(os.environ.get("SCENEWEAVE_HYBRID_MIN_BLOB", "0.50"))
            if _fill < _MIN_FILL or _largest < _MIN_BLOB:
                # Name the criterion that actually failed — either alone is
                # enough to reject, and reporting both as if both failed makes
                # the log misleading (a 24% fill was being printed as "(<15%)").
                _why = []
                if _fill < _MIN_FILL:
                    _why.append(f"fill {_fill:.1%} < {_MIN_FILL:.0%}")
                if _largest < _MIN_BLOB:
                    _why.append(f"largest blob {_largest:.1%} < {_MIN_BLOB:.0%}")
                print(f"  [hybrid] REJECTED — {' and '.join(_why)} "
                      f"({int(_n)} components, fill={_fill:.1%}). "
                      f"Falling back — a shredded reference makes the edit model "
                      f"invent an object out of the fragments.")
            else:
                # Composite: object pixels from crop, grey elsewhere
                crop_arr = np.array(crop_img)
                grey = np.full_like(crop_arr, 185)
                hybrid = np.where(dilated[:, :, np.newaxis], crop_arr, grey)
                print(f"  using hybrid reference (crop + dilated mask) "
                      f"[fill={_fill:.0%} blob={_largest:.0%}]")
                return Image.fromarray(hybrid)
        except Exception as e:
            print(f"  [hybrid] failed: {e} — falling back")

    # Fallback chain
    if masked_canvas is not None:
        print(f"  using masked canvas as reference")
        return masked_canvas
    if crop_img is not None:
        print(f"  using crop as reference (no canvas/mask)")
        return crop_img
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(output_dir: str | Path,
        indices: "list[int] | None" = None,
        force: bool = False) -> Path:
    """Inpaint each furniture segment.

    Args:
        output_dir: pipeline output directory (must contain `furniture/`).
        indices:   if given, only re-inpaint segments whose `index` is in this
                   list.  Other segments are left untouched.
        force:     if True, delete any existing inpaint output for the
                   targeted segments before regenerating (otherwise the
                   existing file is reused).

    Per-segment customisation: a segment may include
        "inpaint_prompt_override": "<custom prompt>"
    in `segment_results.json`.  When present, it is used verbatim by
    `_call_edit` instead of the default type-based prompt.  Useful when
    you want to remove only a sub-region (e.g. "remove only the pillows
    from the bed; keep the bedding/blanket/headboard intact") without
    touching the rest of the object.
    """
    out_dir     = Path(output_dir) / "furniture"
    seg_dir     = out_dir / "segmented"
    inpaint_dir = out_dir / "inpainted"
    results_p   = out_dir / "segment_results.json"

    if not results_p.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run segment_furniture first.\n"
            f"Expected: {results_p}"
        )

    inpaint_dir.mkdir(parents=True, exist_ok=True)

    with open(results_p) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[inpaint_furniture] No segments in segment_results.json — nothing to do.")
        return inpaint_dir

    if indices is not None:
        _wanted = set(int(i) for i in indices)
        print(f"[inpaint_furniture] --indices filter: targeting "
              f"{sorted(_wanted)} (other segments left untouched)")
    else:
        _wanted = None

    for seg in segments:
        if _wanted is not None and int(seg.get("index", -1)) not in _wanted:
            continue
        # When targeting specific indices with --force, drop the existing
        # inpaint so it actually regenerates rather than reusing.
        if force:
            _idx = int(seg.get("index", -1))
            _ot  = seg.get("type", "other")
            _existing = inpaint_dir / f"inpaint_{_idx:02d}_{_ot}.png"
            if _existing.exists():
                _existing.unlink()
                print(f"  --force: removed existing {_existing.name}")
        idx         = seg["index"]
        obj_type    = seg.get("type", "other")
        phrase      = seg.get("phrase", obj_type)
        canvas_file = seg.get("canvas_file")

        sam_failed = seg.get("sam_failed", False)
        # When SAM failed but the segmenter saved a rectangular bbox-mask as
        # fallback (`mask_fallback == "bbox_rect"`), the canvas/crop already
        # contain the actual photo pixels — treat as a normal segment so
        # inpainting preserves the object's identity instead of regenerating
        # it from text.  Without this, the cabinet/bookshelf etc. comes out
        # as a generic invented object.
        has_bbox_fallback = sam_failed and seg.get("mask_fallback") == "bbox_rect"
        if has_bbox_fallback:
            sam_failed = False   # treat like a normal segment for ref handling
        print(f"\n[inpaint_furniture] {idx:02d} {obj_type} — '{phrase}'"
              + (" [SAM-failed, bbox-rect fallback]" if has_bbox_fallback
                 else (" [SAM-failed, white canvas]" if sam_failed else "")))

        # ── Pre-filter: check if the detected object is actually inside the room ──
        # Segmentation can pick up trees/sky through windows as "plant", or
        # reflections/outdoor objects.  Use VLM on the crop to verify.
        crop_file = seg.get("crop_file")
        # SCENEWEAVE_NO_INDOOR_CHECK=1 disables the outdoor/reflection pre-filter.
        # The check is meant to drop trees-through-windows / reflections, but it
        # also over-rejects built-in-but-wanted furniture (base cabinets, counters,
        # vanities, kitchen islands) as "room structure", which then meshes from the
        # raw canvas into a thin panel and mis-places. Bypass when a scene's
        # cabinetry keeps getting dropped.
        if crop_file and (seg_dir / crop_file).exists() and not seg.get("indoor_verified") \
                and obj_type != "fireplace" \
                and os.environ.get("SCENEWEAVE_NO_INDOOR_CHECK") != "1":
            if not _vlm_is_indoor_object(seg_dir / crop_file, obj_type):
                print(f"  [indoor-check] not an indoor object — skipping")
                seg["skipped_reason"] = "not_indoor"
                continue

        out_name = f"inpaint_{idx:02d}_{obj_type}.png"
        out_path = inpaint_dir / out_name

        if out_path.exists():
            print(f"  reusing existing {out_name}")
            seg["inpaint_file"] = out_name
            continue

        if sam_failed:
            canvas = None
            crop_img = None
            masked_canvas = None
        else:
            # Load crop for VLM context (shows object in room with surroundings)
            crop_file = seg.get("crop_file")
            crop_img = None
            if crop_file and (seg_dir / crop_file).exists():
                crop_img = Image.open(seg_dir / crop_file).convert("RGB")

            # Load masked canvas (object isolated on grey)
            masked_canvas = None
            if canvas_file and (seg_dir / canvas_file).exists():
                masked_canvas = Image.open(seg_dir / canvas_file).convert("RGB")

            # Build a hybrid reference: crop with non-target areas greyed out.
            # This preserves the object's full shape from the crop (including
            # parts the SAM mask missed) while removing other furniture.
            # The mask is dilated so items ON the object (books, etc.) are
            # included in the silhouette — the prompt handles their removal.
            canvas = _build_hybrid_ref(seg, seg_dir, crop_img, masked_canvas)
            if canvas is None:
                print("  no reference found — skipping.")
                continue
            if masked_canvas is None:
                masked_canvas = canvas

        prompt_override = seg.get("inpaint_prompt_override")
        if prompt_override:
            print(f"  using inpaint_prompt_override: {prompt_override[:100]}…")

        result = _call_edit(canvas, obj_type, phrase,
                            crop=crop_img, masked_canvas=masked_canvas,
                            prompt_override=prompt_override,
                            reconstruct_mode=(has_bbox_fallback
                                              and obj_type in _RECONSTRUCT_TYPES))

        if result is None:
            print("  edit failed — skipping.")
            continue

        # Flat objects (carpet / rug): enforce a clear grey border so the
        # downstream SAM re-segmentation pass can tell where the object ends.
        # The edit model routinely ignores "leave margin" instructions.
        if obj_type in {"carpet", "rug"}:
            result = _enforce_grey_border(result, margin_frac=0.10)
            print(f"  enforced ≥10 % grey border (for segmentation)")

        result.save(out_path)
        seg["inpaint_file"] = out_name
        print(f"  saved → {out_path}")

    # ── Heavily-occluded side table / nightstand → borrow its bed-flanking pair ──
    # A nightstand mostly hidden behind the bed inpaints as a degenerate top-slab
    # sliver (tiny flat mask) → a useless 3D-gen reference.  When a matching full
    # side table exists (the symmetric bed pair), reuse its inpaint, mirrored, for
    # the occluded one.  See memory feedback_occluded_sidetable_borrow.
    _ST_TYPES = {"side_table", "nightstand", "end_table", "bedside_table"}
    _sts = [s for s in segments if s.get("type") in _ST_TYPES
            and s.get("box_px") and s.get("inpaint_file")]
    if len(_sts) >= 2:
        _ba = lambda s: max(1, (s["box_px"][2] - s["box_px"][0]) *
                               (s["box_px"][3] - s["box_px"][1]))
        full = max(_sts, key=_ba)
        for s in _sts:
            if s is not full and _ba(s) < 0.4 * _ba(full):
                try:
                    fp = inpaint_dir / full["inpaint_file"]
                    sp = inpaint_dir / s["inpaint_file"]
                    if fp.exists():
                        Image.open(fp).convert("RGB").transpose(
                            Image.FLIP_LEFT_RIGHT).save(sp)
                        print(f"  [borrow] heavily-occluded {s['inpaint_file']} "
                              f"← {full['inpaint_file']} (mirrored bed pair)")
                except Exception as _e:
                    print(f"  [borrow] side-table borrow failed: {_e}")

    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)

    done = sum(1 for s in segments if "inpaint_file" in s)
    print(f"\n[inpaint_furniture] Done. {done}/{len(segments)} objects inpainted → {inpaint_dir}/")
    return inpaint_dir


def main():
    ap = argparse.ArgumentParser(
        description="Inpaint segmented furniture with Qwen image-edit."
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Pipeline output dir containing furniture/segment_results.json",
    )
    ap.add_argument(
        "--indices", type=int, nargs="+", default=None,
        help="Only re-inpaint these segment indices (e.g. --indices 0 5). "
             "Other segments are left untouched.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Delete existing inpaint output for targeted segments before "
             "regenerating (otherwise existing files are reused).",
    )
    args = ap.parse_args()
    run(output_dir=args.output_dir, indices=args.indices, force=args.force)


if __name__ == "__main__":
    main()
