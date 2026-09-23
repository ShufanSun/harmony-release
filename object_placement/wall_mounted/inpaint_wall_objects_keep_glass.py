"""
inpaint_wall_objects_keep_glass.py — same as inpaint_wall_objects.py but windows
                                     and doors are NOT sent to the AI at all.

All non-window/door objects are inpainted identically to the original script.
Windows / doors:
  - The original segmented canvas is copied directly to the inpainted folder.
  - No AI edit call, no deglaze pass — interior pixels are completely unmodified.

Usage:
    python -m object_placement.wall_mounted.inpaint_wall_objects_keep_glass \\
        --output-dir outputs/living_room9

Outputs written to <output_dir>/wall_mounted/inpainted/
Segment JSON is updated in-place with inpaint_file fields (same as original script).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import requests
from PIL import Image

from Qwen.image_edit_adapter import edit_image as _edit_image

EDIT_API_URL = "http://localhost:8000/generate"   # legacy; routed via adapter
OBJ_IMG_SIZE = 512
EDIT_STEPS   = 20

# ── Per-type descriptors ──────────────────────────────────────────────────────
# Window descriptor: generate a real glass window, NOT frame-only.
# All other descriptors are identical to inpaint_wall_objects.py.

_TYPE_DESCRIPTOR: dict[str, str] = {
    "window": (
        "a complete photorealistic window showing all its glass panes and the same outdoor "
        "view as the reference, on a plain grey background; sharp detail"
    ),
    "door": (
        "a single photorealistic architectural interior door set in its own door frame / jamb, "
        "viewed straight-on and filling the image, with a visible handle and (if glazed) its "
        "glass panels — it is a real DOOR, not a picture and not a painting; "
        "perfectly straight vertical and horizontal edges, full height visible, "
        "no picture-frame border, no white mat, no artwork, no surrounding room or furniture, "
        "plain grey background outside the door frame, ultra-sharp, 4K detail"
    ),
    "curtain": (
        "the same curtain panel as in the reference — identical fabric colour, shade, "
        "texture, and drape pattern; "
        "complete the shape into a full rectangle: straight left edge, straight right edge, "
        "continuous fabric from top header to full hem at the bottom; "
        "fill any missing or occluded regions with the same fabric seamlessly; "
        "no holes, no cutouts, no irregular edges, no wall, no furniture behind it, "
        "ultra-sharp fabric detail, 4K"
    ),
    "shelf": (
        "a photorealistic wall shelf unit, clean horizontal boards, "
        "sharp bracket edges, empty shelves, no wall behind, "
        "ultra-sharp woodwork detail, 4K"
    ),
    "tv": (
        "a photorealistic flat-screen television, perfectly straight bezel on all four sides, "
        "blank dark screen, no wall behind, ultra-sharp, 4K"
    ),
    "art": (
        "EXACTLY ONE single framed painting, isolated on a plain grey background; "
        "only this one painting is visible — nothing else in the entire image; "
        "same frame style, artwork subject, and colour palette as the reference, "
        "but rendered sharply at high resolution; "
        "frame is a complete rectangle with perfectly straight edges on all four sides; "
        "artwork surface rendered with visible detail, rich colours, fine brushwork or print quality; "
        "no wall, no other paintings, no other objects, no furniture, museum-quality, 4K"
    ),
    "mirror": (
        "a photorealistic decorative wall mirror, "
        "complete frame with perfectly straight or curved edges, "
        "smooth reflective surface, no wall behind, ultra-sharp, 4K"
    ),
    "clock": (
        "a photorealistic wall clock, complete circular or rectangular housing, "
        "clearly visible clock face with sharp numerals and hands, "
        "no wall behind, ultra-sharp, 4K"
    ),
    "light": (
        "the light fixture EXACTLY as shown in the reference — PRESERVE its actual "
        "form and orientation: if the reference shows a HANGING PENDANT (a glass or "
        "metal shade suspended VERTICALLY from a cord, chain or rod descending from "
        "above), render that vertical hanging pendant with its cord/rod at the very "
        "top going up; if it shows a WALL SCONCE (a bracket fixed to the wall with a "
        "short horizontal or angled arm out to a bulb/shade), render that sconce. "
        "Do NOT convert a hanging pendant into a wall sconce or a sconce into a "
        "pendant. Match the reference's shade shape, proportions, materials and "
        "colour; smooth clean surfaces, slim elegant profile; the fixture does NOT "
        "connect to the floor or any furniture; full fixture visible (cord/bracket "
        "to bulb), no wall behind, ultra-sharp, 4K"
    ),
    "ceiling_light": (
        "a photorealistic ceiling-mounted light fixture viewed from below, "
        "such as a recessed downlight, pendant lamp, or flush-mount chandelier, "
        "complete housing, shade, or diffuser fully visible, "
        "no ceiling surface, no wall, no room context, "
        "ultra-sharp, 4K detail"
    ),
    "radiator": (
        "a photorealistic radiator, "
        "clean straight panel edges top and bottom, detailed fin structure, "
        "no wall behind, ultra-sharp, 4K"
    ),
    "other": (
        "a photorealistic isolated object, clean sharp edges, "
        "no wall, no background texture, ultra-sharp, 4K"
    ),
}

_GLOBAL_NEG = (
    "blurry, out of focus, low resolution, low quality, pixelated, jpeg artefacts, "
    "grainy, noisy, soft focus, hazy, "
    "wall, brick wall, plaster, concrete, stone wall, painted wall, drywall, "
    "wood paneling, wallpaper, tile, background surface, wall behind object, "
    "room interior, floor, ceiling, furniture, other objects, "
    "multiple objects, two objects, several items, group of objects, "
    "additional object, second object, extra object, side object, neighbouring object, "
    "foreground object, overlapping furniture, "
    "outdoor view, sky, clouds, buildings, trees, landscape, "
    "people, border, watermark, "
    "cropped edges, cut-off, incomplete shape, missing parts, "
    "distorted, warped, deformed"
)

_TYPE_NEG: dict[str, str] = {
    "shelf": (
        # NOTE: deliberately omits the "other objects / additional object" clauses so
        # the items sitting ON the shelf survive; adds "empty shelf" as a negative.
        "blurry, low resolution, low quality, soft focus, "
        "wall, plaster, painted wall, background surface, wall behind, "
        "room interior, floor, ceiling, "
        "empty shelf, empty shelves, bare shelf, cleared shelf, removed items, "
        "missing objects, stripped shelf, "
        "people, border, watermark, cropped, cut-off, distorted"
    ),
    "door": (
        "blurry, low resolution, low quality, soft focus, "
        "picture frame, framed picture, painting, poster, artwork, canvas, wall art, "
        "white mat, matted border, gallery frame, framed scene, photo frame, "
        "wall, plaster, painted wall, background surface, wall behind, "
        "room interior, bedroom, bed, lamp, nightstand, sofa, furniture, floor, ceiling, "
        "multiple doors, second door, "
        "people, border, watermark, cropped, cut-off, distorted"
    ),
    "window": (
        "blurry, low resolution, low quality, "
        "furniture, armchair, table, plant, people, wall, room interior, "
        "added thick white frame, simplified window, fewer panes than reference, "
        "watermark, distorted"
    ),
    "art": (
        "blurry, low resolution, low quality, soft focus, grainy, "
        "wall, plaster, concrete, painted wall, background surface, wall behind, "
        "room interior, floor, ceiling, furniture, "
        "second painting, additional painting, multiple paintings, two paintings, "
        "gallery wall, picture collage, art collection, row of frames, side-by-side artworks, "
        "lamp, wall lamp, wall sconce, sconce, brass lamp, metal lamp, light fixture, "
        "lamp shade, lamp head, lamp arm, bulb, light, "
        "curtain, drape, fabric panel, sheer, blinds, "
        "shelf, mirror, clock, plant, vase, "
        "any object in front of the painting, foreground occluder, "
        "any object other than the single framed painting, "
        "bent frame, warped edges, uneven frame, crooked frame, "
        "people, border, watermark, "
        "cropped, cut-off, distorted"
    ),
    "curtain": (
        "blurry, low resolution, low quality, soft focus, "
        "wall, plaster, concrete, painted wall, background surface, wall behind, "
        "window frame, window glass, window blind, venetian blind, roller blind, shutter, "
        "outdoor view, sky, trees, plant, leaves, foliage, potted plant, "
        "room interior, floor, ceiling, furniture, sofa, table, "
        "lamp, sconce, mirror, painting, picture frame, "
        "any object other than the single curtain panel, "
        "multiple curtain panels, row of curtains, repeated panels, "
        "irregular shape, jagged edges, stepped edges, ragged silhouette, torn edges, "
        "holes, cutouts, missing fabric, partial curtain, incomplete shape, asymmetric cutout, "
        "people, border, watermark, "
        "cropped, cut-off, distorted"
    ),
}

_TYPE_CFG: dict[str, float] = {
    "art":           7.0,   # high: reference often has lamp/curtain occluding painting
    "curtain":       2.5,
    "mirror":        4.0,
    "window":        6.5,   # simple "complete the window, no thick frame" prompt
    "ceiling_light": 5.0,
}
_DEFAULT_CFG = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_prompt(obj_type: str, phrase: str) -> str:
    import os as _os
    if _os.environ.get("SCENEWEAVE_INPAINT_SHORT"):
        t = obj_type.replace("_", " ")
        return (f"Extract the {t} shown and amodally complete it into a single, whole "
                f"{t} on a plain grey background. Keep its exact shape, aspect ratio, "
                f"colour, material and proportions — do not redesign it. Remove "
                f"everything else (room, other objects, people). Fill any occluded or "
                f"cut-off parts so nothing is cropped or floating.")
    descriptor = _TYPE_DESCRIPTOR.get(obj_type, _TYPE_DESCRIPTOR["other"])

    colour_m = re.match(
        r"^((?:light |dark |bright |deep )?\w+(?:\s+\w+)?)\s+" + re.escape(obj_type),
        phrase, re.IGNORECASE,
    )
    style_prefix = f"{colour_m.group(1).strip()}, " if colour_m else ""

    if obj_type == "curtain":
        reference_instruction = (
            "Preserve the reference exactly — same aspect ratio, same drape pattern, "
            "same fabric colour and folds. "
            "The only change needed: the reference has concave gaps or missing edge regions "
            "caused by foreground objects (plants, furniture) that were in front of the curtain. "
            "Fill those gaps seamlessly with matching fabric so the curtain silhouette becomes "
            "a clean complete panel. Everything else stays identical to the reference."
        )
    elif obj_type == "art":
        reference_instruction = (
            "The reference is a partial cut-out of a single framed painting on a plain "
            "grey background — only the visible (un-occluded) portion of the painting and "
            "its frame is shown; the rest of the canvas is solid grey because those parts "
            "were hidden by foreground objects in the original photo. "
            "Your task: EXTEND the visible painting into a COMPLETE rectangular framed "
            "painting. Fill every grey region with artwork or frame content that is "
            "consistent with the visible portion — same artwork style, brushwork, colours, "
            "and palette; same frame profile, colour, and thickness. "
            "Snap the outer frame edges to perfectly straight horizontal and vertical "
            "lines so the final shape is a clean rectangle on all four sides. "
            "Do NOT introduce any new object — no lamp, no sconce, no curtain, no second "
            "painting, no furniture. Background outside the rectangular frame stays the "
            "same plain grey. The result is one isolated complete framed painting."
        )
    elif obj_type == "window":
        reference_instruction = (
            "Complete this partial, occluded window into one whole, complete window: remove "
            "any furniture or plant in front of it and fill those areas (and any grey gaps) "
            "with the same glass panes, mullions and outdoor view. Do not add a thick white "
            "outer frame; a later rectification step refines the geometry."
        )
    elif obj_type == "shelf":
        # A shelf is defined by what sits on it — keep the objects, don't empty it.
        # Return early so the generic "EXACTLY ONE object / no other items" tail (which
        # would strip the shelf contents) is NOT appended.
        return (
            f"{style_prefix}a photorealistic wall-mounted shelf unit, isolated on a solid "
            "plain grey background, WITH every object that rests on its shelves kept exactly "
            "as in the reference — books, vases, frames, boxes, plants and decor stay in their "
            "original positions, colours, sizes and proportions. Do NOT empty the shelves and "
            "do NOT remove or relocate anything sitting on them. Complete only the shelf "
            "structure itself (boards, side panels, brackets) where it is occluded or cut off, "
            "and fill any grey gaps to match. No wall behind, no floor, no surrounding room; "
            "the shelf and the items on it form ONE unit. Maximum sharpness and detail."
        )
    else:
        reference_instruction = (
            "The reference image may be partial, occluded, or low-resolution — "
            "use it only for color, material, and style. "
            "Generate a fully complete, structurally whole object with no missing regions, "
            "no holes, no ragged edges, no irregular cutouts."
        )

    return (
        f"{style_prefix}{descriptor}. "
        "EXACTLY ONE object in the entire image, fully isolated on a solid plain grey background. "
        "No second object of any kind, no other items, no neighbouring objects, no wall, no room. "
        f"{reference_instruction} "
        "Render at maximum sharpness and detail."
    )


def _aspect_size(img: Image.Image, base: int = OBJ_IMG_SIZE) -> tuple[int, int]:
    """Return (width, height) rounded to nearest 8 that preserves aspect ratio."""
    w, h = img.size
    if w >= h:
        out_w = base
        out_h = max(8, round(base * h / w / 8) * 8)
    else:
        out_h = base
        out_w = max(8, round(base * w / h / 8) * 8)
    return out_w, out_h


def _raw_edit(canvas: Image.Image, prompt: str, neg: str, cfg: float) -> Image.Image | None:
    out_w, out_h = _aspect_size(canvas)
    payload = {
        "prompt":              prompt,
        "negative_prompt":     neg,
        "reference_image":     _encode_pil(canvas),
        "width":               out_w,
        "height":              out_h,
        "num_inference_steps": EDIT_STEPS,
        "true_cfg_scale":      cfg,
        "num_images":          1,
    }
    try:
        data = _edit_image(payload, timeout=300)
        img_bytes = base64.b64decode(data["images"][0])
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"  edit call failed: {e}")
        return None


def _canvas_completeness(canvas: Image.Image) -> float:
    """Return fraction of the 4 edges that have foreground pixels near them.
    1.0 = all four edges have window content touching them (complete).
    <1.0 = some edges are all-grey (frame missing on that side).
    """
    import numpy as np
    arr  = np.array(canvas, dtype=np.float32)
    grey = np.array([185., 185., 185.])
    tol  = 25.0
    is_fg = (np.abs(arr - grey) > tol).any(axis=2)
    H, W  = is_fg.shape
    rim   = max(4, H // 20)
    edges = [
        is_fg[:rim,  :].any(),   # top
        is_fg[-rim:, :].any(),   # bottom
        is_fg[:, :rim].any(),    # left
        is_fg[:, -rim:].any(),   # right
    ]
    return sum(edges) / 4.0


def _inpaint_window(canvas: Image.Image, phrase: str) -> Image.Image | None:
    """Complete a window canvas.
    - If all 4 edges already have foreground content (complete canvas), use as-is.
    - Otherwise run AI edit to complete ONLY the missing region (fill-only): the
      real frame/glass pixels are composited back, so no new frame is invented.
    """
    completeness = _canvas_completeness(canvas)
    print(f"  [window] canvas completeness: {completeness:.2f} (1.0=all edges present)")

    if completeness >= 1.0:
        print("  [window] canvas already complete — using directly")
        return canvas

    prompt = (
        "Complete this PARTIAL window into ONE whole rectangular window. The "
        "reference shows only the visible part of a single window — some panes are "
        "cut off at the edges or hidden behind foreground objects. Reconstruct the "
        "FULL window: keep the SAME frame colour and material, the SAME "
        "mullion/divider GRID at the same spacing and the same number of panes per "
        "row, and CONTINUE the exact same scene seen through the glass into the "
        "missing/occluded area — as if the foreground objects were removed and the "
        "window simply carried on behind them. Output ONE clean complete window on "
        "a plain grey background: no foreground furniture or plants, no added "
        "picture-frame border around the image, no second window."
    )
    neg = (
        "small single-pane window, tiny window, generic blank window, "
        "foreground furniture, chair, sofa, plant, curtain, person, occluder, "
        "added picture-frame border, blank frame, grey gap, missing panes, "
        "second window, additional window, multiple separate windows, row of windows, "
        "restyled frame, different mullion pattern, different view, "
        "wall, room, distorted, watermark"
    )
    print("  [window] canvas incomplete — completing partial window (cfg=3.5) …")
    # Medium CFG: reconstruct the FULL window faithfully from the visible portion
    # (grid + view) rather than fill-only (which leaves occluded holes) or high-CFG
    # regenerate (which collapses a multi-pane window into a generic single pane).
    return _raw_edit(canvas, prompt, neg, cfg=3.5)


def _call_edit(canvas: Image.Image, obj_type: str, phrase: str) -> Image.Image | None:
    prompt = _build_prompt(obj_type, phrase)
    neg    = _TYPE_NEG.get(obj_type, _GLOBAL_NEG)
    cfg    = _TYPE_CFG.get(obj_type, _DEFAULT_CFG)

    print(f"  prompt (cfg={cfg}): {prompt[:120]}")

    out_w, out_h = _aspect_size(canvas)
    payload = {
        "prompt":              prompt,
        "negative_prompt":     neg,
        "reference_image":     _encode_pil(canvas),
        "width":               out_w,
        "height":              out_h,
        "num_inference_steps": EDIT_STEPS,
        "true_cfg_scale":      cfg,
        "num_images":          1,
    }
    try:
        data = _edit_image(payload, timeout=300)
        img_bytes = base64.b64decode(data["images"][0])
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"  image-edit failed: {e}")
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(output_dir: str | Path,
        types: list[str] | None = None,
        indices: list[int] | None = None) -> Path:
    out_dir     = Path(output_dir) / "wall_mounted"
    seg_dir     = out_dir / "segmented"
    inpaint_dir = out_dir / "inpainted"
    results_p   = out_dir / "segment_results.json"

    if not results_p.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run segment_wall_objects first.\n"
            f"Expected: {results_p}"
        )

    inpaint_dir.mkdir(parents=True, exist_ok=True)

    with open(results_p) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[inpaint] No segments in segment_results.json — nothing to do.")
        return inpaint_dir

    for seg in segments:
        idx         = seg["index"]
        obj_type    = seg.get("type", "other")
        phrase      = seg.get("phrase", obj_type)
        canvas_file = seg.get("canvas_file")

        if types is not None and obj_type not in types:
            continue
        if indices is not None and idx not in indices:
            continue

        print(f"\n[inpaint] {idx:02d} {obj_type} — '{phrase}'")

        if not canvas_file:
            print("  no canvas_file — skipping.")
            continue

        canvas_path = seg_dir / canvas_file
        if not canvas_path.exists():
            print(f"  canvas not found: {canvas_path} — skipping.")
            continue

        out_name = f"inpaint_{idx:02d}_{obj_type}.png"
        out_path = inpaint_dir / out_name
        if out_path.exists():
            print(f"  reusing existing {out_name}")
            seg["inpaint_file"] = out_name
            continue

        canvas = Image.open(canvas_path).convert("RGB")

        if obj_type in ("window", "door"):
            # Pad generously on all sides so the model has room to add frame edges.
            pad = max(80, max(canvas.width, canvas.height) // 5)
            padded = Image.new("RGB",
                               (canvas.width + pad*2, canvas.height + pad*2),
                               (185, 185, 185))
            padded.paste(canvas, (pad, pad))
            canvas = padded

        if obj_type in ("window", "door"):
            result = _inpaint_window(canvas, phrase)
        else:
            result = _call_edit(canvas, obj_type, phrase)
        if result is None:
            print("  edit failed — skipping.")
            continue

        result.save(out_path)
        seg["inpaint_file"] = out_name
        print(f"  saved → {out_path}")

    with open(results_p, "w") as f:
        json.dump(data, f, indent=2)

    done = sum(1 for s in segments if "inpaint_file" in s)
    print(f"\n[inpaint] Done. {done}/{len(segments)} objects inpainted → {inpaint_dir}/")
    return inpaint_dir


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Inpaint wall-mounted objects — windows kept as photorealistic glass windows "
            "(no frame-only conversion). All other objects inpainted identically to "
            "inpaint_wall_objects.py."
        )
    )
    ap.add_argument(
        "--output-dir", required=True,
        help="Pipeline output dir containing wall_mounted/segment_results.json",
    )
    ap.add_argument(
        "--types", default=None,
        help="Comma-separated object types to inpaint (e.g. window,curtain). Default: all.",
    )
    ap.add_argument(
        "--indices", default=None,
        help="Comma-separated segment indices to (re)inpaint, e.g. '8' or '0,3,8'. "
             "Combined with --types as an AND filter when both are passed.",
    )
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",")] if args.types else None
    indices = ([int(i.strip()) for i in args.indices.split(",")]
               if args.indices else None)
    run(output_dir=args.output_dir, types=types, indices=indices)


if __name__ == "__main__":
    main()
