"""
inpaint_decorations.py — complete partial decoration objects using Qwen image-edit.

Pipeline:
  1. Load segment_results.json from <output_dir>/decorations/
  2. For each segmented decoration:
     a. If text_only=True (SAM failed or not a single object):
        generate from text description alone using the crop as loose reference.
     b. If text_only=False (valid segmented single object):
        use the canvas (partial object on grey) + VLM description to inpaint
        missing regions.
  3. Save inpainted output as inpaint_<idx>_<name>.png
  4. Update segment_results.json with inpaint_file fields.

Usage:
    python -m object_placement.decorations.inpaint_decorations \
        --output-dir outputs/living_room9
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

# Preserve COMPLETE decorations exactly (composite the real segmented pixels on
# grey) instead of regenerating them — the generative editor drifts both colour
# and shape (a rust angled pillow becomes a brown square).  Cut-off / occluded
# objects still go through generative completion.  Set =0 to always generate.
_DECOR_PRESERVE = os.environ.get("SCENEWEAVE_DECOR_PRESERVE", "1").lower() \
    not in ("0", "false", "no", "")

from Qwen.image_edit_adapter import edit_image as _edit_image

EDIT_API_URL = "http://localhost:8000/generate"   # legacy; routed via adapter
VLM_API_URL  = "http://localhost:8080/v1/chat/completions"
from object_placement.vlm_backend import vlm_post as _vlm_post
OBJ_IMG_SIZE = 768
EDIT_STEPS   = 30
MAX_RETRIES  = 1

_BG_GREY = (185, 185, 185)

# ── Helpers ──────────────────────────────────────────────────────────────────

# Simple colour vocabulary for crop-based colour extraction
_COLOUR_NAMES: list[tuple[tuple[int,int,int], str]] = [
    ((220, 220, 220), "white"),
    ((160, 160, 160), "light grey"),
    ((100, 100, 100), "grey"),
    ((50,  50,  50),  "dark grey"),
    ((20,  20,  20),  "black"),
    ((200, 100, 100), "pink"),
    ((200, 50,  50),  "red"),
    ((200, 80,  30),  "orange"),
    ((200, 180, 50),  "yellow"),
    ((80,  160, 80),  "green"),
    ((50,  80,  160), "blue"),
    ((100, 50,  150), "purple"),
    ((160, 110, 60),  "brown"),
    ((210, 170, 120), "beige"),
    ((200, 150, 90),  "tan"),
]

_COLOUR_KEYWORDS = {
    "white", "light grey", "grey", "gray", "dark grey", "black",
    "pink", "red", "orange", "yellow", "green", "blue", "purple",
    "brown", "beige", "tan", "cream", "ivory", "navy", "teal",
    "olive", "rust", "mustard", "charcoal", "off-white",
}


def _nearest_colour_name(rgb: tuple[int, int, int]) -> str:
    best_name, best_dist = "grey", float("inf")
    for (cr, cg, cb), name in _COLOUR_NAMES:
        d = (rgb[0]-cr)**2 + (rgb[1]-cg)**2 + (rgb[2]-cb)**2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def _extract_crop_colour(crop: Image.Image) -> str | None:
    """Extract the dominant non-background colour from a crop image.

    Excludes pixels close to the canvas background grey and returns a
    human-readable colour name, or None if the crop is too small/ambiguous.
    """
    arr = np.array(crop.convert("RGB"), dtype=np.float32)
    bg  = np.array(_BG_GREY, dtype=np.float32)

    # Mask out background-grey pixels (within 35 units in each channel)
    diff = np.abs(arr - bg).max(axis=2)
    fg_mask = diff > 35
    if fg_mask.sum() < 20:
        return None

    fg_pixels = arr[fg_mask].astype(np.int32)

    # Compute mean colour of foreground pixels
    mean_rgb = tuple(int(v) for v in fg_pixels.mean(axis=0))
    name = _nearest_colour_name(mean_rgb)  # type: ignore[arg-type]
    return name


def _strip_colour_words(text: str) -> str:
    """Remove colour adjectives from a category hint.

    The VLM scene-inventory/analysis often labels a decoration with the WRONG
    colour (e.g. a rose pillow described as "orange", a beige one as "gray").
    When a reference image is available the model should read the true colour
    from the pixels, so we drop the colour word from the text hint to stop it
    overriding the reference. Multi-word colours ("light grey") are removed
    before single words. Falls back to the original text if stripping empties it.
    """
    out = text
    for c in ("light grey", "light gray", "dark grey", "dark gray", "off-white"):
        out = re.sub(rf"\b{re.escape(c)}\b", "", out, flags=re.IGNORECASE)
    for c in _COLOUR_KEYWORDS:
        out = re.sub(rf"\b{re.escape(c)}\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.")
    return out or text


def _inject_colour_into_description(description: str, colour: str) -> str:
    """Prepend colour to description if no colour word is already present."""
    desc_lower = description.lower()
    if any(kw in desc_lower for kw in _COLOUR_KEYWORDS):
        return description  # already has a colour word
    # Insert colour after any leading quantity word
    words = description.split()
    if words and words[0].lower() in {"a", "an", "the", "one", "two", "three",
                                       "four", "five", "six", "2", "3", "4", "5", "6"}:
        return f"{words[0]} {colour} {' '.join(words[1:])}"
    return f"{colour} {description}"


def _encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _canvas_fg_fraction(canvas: Image.Image) -> float:
    """Fraction of pixels that are not background grey."""
    arr = np.array(canvas.convert("RGB"), dtype=np.float32)
    bg = np.array(_BG_GREY, dtype=np.float32)
    diff = np.abs(arr - bg).max(axis=2)
    return float((diff > 25).mean())


# ── Description lookup from decoration_analysis.json ─────────────────────────

def _load_decoration_descriptions(decor_dir: Path) -> dict[tuple[int, str], str]:
    """Load original decoration descriptions from decoration_analysis.json.

    Returns a dict mapping (furniture_index, decoration_name) → description.
    These descriptions come from the full-res reference image comparison and
    are much more reliable than re-describing blurry segmented crops.
    """
    analysis_path = decor_dir / "decoration_analysis.json"
    if not analysis_path.exists():
        return {}

    with open(analysis_path) as f:
        analysis = json.load(f)

    lookup: dict[tuple[int, str], str] = {}
    for furn in analysis.get("furniture_decorations", []):
        furn_idx = furn["furniture_index"]
        for deco in furn.get("missing_decorations", []):
            if isinstance(deco, dict):
                lookup[(furn_idx, deco["name"])] = deco.get("description", "")
    return lookup


# ── Inpainting ───────────────────────────────────────────────────────────────

_GLOBAL_NEG = (
    "blurry, low resolution, low quality, pixelated, jpeg artefacts, "
    "room interior, wall, floor, ceiling, "
    "background objects, second object, multiple objects, "
    "people, person, hands, border, watermark, "
    "cropped edges, cut-off, distorted, warped, deformed"
)

_TEXT_ONLY_NEG = (
    _GLOBAL_NEG + ", "
    "furniture, sofa, couch, chair, table, bed, desk"
)


_QUANTITY_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
}


def _strip_quantity(desc: str) -> tuple[str, int]:
    """Strip leading quantity words from a description.

    Returns (singularised description, count).
    E.g. "two grey throw pillows" → ("grey throw pillow", 2)
    """
    words = desc.split()
    count = 1
    if words and words[0].lower() in _QUANTITY_WORDS:
        count = _QUANTITY_WORDS[words[0].lower()]
        words = words[1:]

    # Singularise trailing 's' if count was > 1
    if count > 1 and words and words[-1].endswith("s") and len(words[-1]) > 2:
        words[-1] = words[-1][:-1]

    return " ".join(words), count


def _build_inpaint_prompt(
    phrase: str,
    description: str | None,
    text_only: bool,
    use_crop: bool = False,
    colour: str | None = None,
) -> str:
    """Build the generation prompt for a decoration object.

    Always generates exactly ONE object. Quantity is handled separately
    via the `quantity` field in segment metadata.
    """
    base_phrase = phrase.split()[-1] if " " in phrase else phrase
    desc = description or base_phrase

    # SHORT-PROMPT mode: when a reference image is available (not text-only),
    # use a short image-led prompt instead of the long descriptive one.
    if os.environ.get("SCENEWEAVE_INPAINT_SHORT") and not text_only:
        d, _ = _strip_quantity(desc)
        d = _strip_colour_words(d)
        return (f"Extract the {d} shown and amodally complete it into a single, whole "
                f"{d} on a plain grey background. Keep its exact shape, colour, material "
                f"and proportions — do not redesign it. Remove everything else; nothing "
                f"cropped or floating.")

    # Strip quantity — we always generate a single object
    desc, _ = _strip_quantity(desc)

    # When a reference image is available (canvas / crop), drop the colour word
    # from the text hint: the VLM colour label is frequently wrong and biases the
    # generation off the true tone (e.g. a rose pillow labelled "orange"). The
    # image carries the real colour. For text-only there is no image, so keep it.
    if not text_only:
        desc = _strip_colour_words(desc)

    # Colour anchor from the actual reference pixels (not the VLM label). Gemini
    # tends to warm-/grey-shift upholstery, so we both name the true tone and
    # explicitly forbid drifting toward the common wrong tones.
    colour_clause = ""
    if colour and not text_only:
        colour_clause = (
            f" The {desc} is {colour} in colour — reproduce this EXACT {colour} tone; "
            f"do NOT shift it toward orange, rust, grey, or any other colour."
        )

    if text_only:
        return (
            f"A product photo of a single {desc}. "
            "The object is isolated on a plain solid grey background. "
            "No furniture, no sofa, no table, no room. "
            "High-resolution, photorealistic, sharp detail, studio lighting. "
            "Only ONE object, nothing else."
        )

    if use_crop:
        # Crop reference includes room background — explicitly tell the model to
        # ignore everything except the target object and reproduce only that.
        return (
            f"A product photo of a single complete {desc}. "
            f"The reference image is a room photo crop — focus only on the {desc} "
            "visible in it; ignore the furniture, background, walls, and floor. "
            f"Reproduce the {desc}'s exact colour, shape, and material.{colour_clause} "
            "Place it alone on a plain solid grey background. "
            "Photorealistic, sharp, studio lighting. "
            "Exactly ONE object — no furniture, no room, nothing else."
        )

    # Canvas-based: the reference image IS the source of truth for what the
    # object looks like.  Treat the noun (`desc`) only as a hint of category;
    # the visible colour, shape, material, and pattern in the reference must
    # win when there is any conflict with the text.  No explicit colour-keyword
    # injection — the model can read the colour from the canvas pixels, and
    # forcing a text colour overrides the reference (e.g. a beige pillow gets
    # painted "white" because the description happened to say "white-ish").
    return (
        f"A product photo of the exact object shown in the reference image. "
        f"The reference image is the source of truth: reproduce its specific "
        f"colour, pattern, texture, and shape exactly as visible.{colour_clause}  "
        f"Treat '{desc}' only as a category hint — when the text and the "
        f"reference disagree, follow the reference.  "
        "Complete any cut-off or partially visible parts so the object looks "
        "fully intact, but do not invent details that aren't suggested by the "
        "reference.  Plain solid grey background. Photorealistic, sharp, "
        "studio lighting.  Exactly ONE object, nothing else — no furniture, "
        "no room, no extra objects."
    )


def _call_edit(
    canvas: Image.Image | None,
    phrase: str,
    description: str | None,
    text_only: bool,
    cfg_scale: float = 5.0,
    use_crop: bool = False,
) -> Image.Image | None:
    """Generate or complete a decoration image."""
    # True colour from the reference pixels (not the VLM colour label) anchors the
    # prompt so Gemini doesn't drift the tone (e.g. rose → orange).
    colour = _extract_crop_colour(canvas) if (canvas is not None and not text_only) else None
    prompt = _build_inpaint_prompt(phrase, description, text_only,
                                   use_crop=use_crop, colour=colour)
    print(f"    colour(ref pixels)={colour}  prompt: {prompt[:110]}…")

    neg = _TEXT_ONLY_NEG if text_only else _GLOBAL_NEG
    payload: dict = {
        "prompt": prompt,
        "negative_prompt": neg,
        "width": OBJ_IMG_SIZE,
        "height": OBJ_IMG_SIZE,
        "num_inference_steps": EDIT_STEPS,
        "true_cfg_scale": cfg_scale,
        "num_images": 1,
    }

    # Use canvas as reference if available (not for pure text-only)
    if canvas is not None:
        payload["reference_image"] = _encode_pil(canvas)

    try:
        data = _edit_image(payload, timeout=300)
        img_bytes = base64.b64decode(data["images"][0])
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        print(f"    generation failed: {e}")
        return None


# ── VLM quality check ───────────────────────────────────────────────────────

_VLM_QUALITY_PROMPT = """\
The image shows an AI-generated object. It should be: "{description}"

Rate it:
1. Does it clearly match the description "{description}"? (e.g. if it should be \
"two stacked books", does it show books? If it should be a "desk lamp", is there \
a lamp?)
2. Is the object complete and not cut off?
3. Is it on a plain grey background with no room, no furniture, no other objects?

Return JSON: {{"ok": true/false, "score": 1-5, "reason": "short explanation"}}
Only JSON, no markdown.
"""


def _vlm_quality_check(result: Image.Image, phrase: str,
                        description: str | None = None) -> dict:
    """Quick quality check on generated decoration."""
    try:
        b64 = _encode_pil(result)
        desc_str = description or phrase
        payload = {
            "model": "qwen3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": _VLM_QUALITY_PROMPT.format(description=desc_str)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "max_tokens": 100,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"    [quality] Failed: {e}")
    return {"ok": True}


# ── Crop clarity check ──────────────────────────────────────────────────────

_VLM_CROP_CLARITY_PROMPT = """\
This image is a crop from a room photo. The target object should be: "{description}"

Is the target object clearly visible and identifiable in this crop?
- YES: the object is clearly visible, recognisable, and occupies a significant portion of the image
- NO: the crop is too blurry, too small, too ambiguous, cluttered, or the object is barely visible

Return JSON: {{"clear": true/false, "reason": "short explanation"}}
Only JSON, no markdown.
"""


def _vlm_crop_is_clear(crop: Image.Image, phrase: str,
                        description: str | None = None) -> bool:
    """Returns True if the crop clearly shows the target object."""
    try:
        b64 = _encode_pil(crop)
        desc_str = description or phrase
        payload = {
            "model": "qwen3",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": _VLM_CROP_CLARITY_PROMPT.format(description=desc_str)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "max_tokens": 80,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group())
            clear = result.get("clear", True)
            print(f"    [crop clarity] clear={clear} — {result.get('reason', '')}")
            return bool(clear)
    except Exception as e:
        print(f"    [crop clarity] Failed: {e}")
    return True  # default: trust the crop


# ── Completeness check + exact-pixel preservation ────────────────────────────

_VLM_COMPLETE_PROMPT = """\
This image shows a single segmented object ("{description}") isolated on a plain \
grey background.

Is the WHOLE object fully visible, or is part of it cut off / clipped at an edge \
/ hidden behind something (so a piece is missing)?
- COMPLETE: the entire object is shown, no part is cut off or occluded.
- PARTIAL: part of the object is missing — clipped by an edge, or hidden/cropped.

Return JSON: {{"complete": true/false, "reason": "short explanation"}}
Only JSON, no markdown.
"""


def _vlm_is_complete(canvas: Image.Image, phrase: str,
                     description: str | None = None) -> bool:
    """True if the segmented object is whole (nothing cut off / occluded).

    Complete objects are preserved exactly; partial ones need generative
    completion.  Defaults to False (generate) on any failure — safer to
    complete than to keep a clipped object."""
    try:
        b64 = _encode_pil(canvas)
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _VLM_COMPLETE_PROMPT.format(
                    description=description or phrase)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "max_tokens": 100,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _vlm_post(payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            r = json.loads(m.group())
            comp = bool(r.get("complete", False))
            print(f"    [complete?] {comp} — {r.get('reason', '')}")
            return comp
    except Exception as e:
        print(f"    [complete?] failed: {e}")
    return False


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(output_dir: str | Path, no_vlm: bool = False,
        indices: "list[int] | None" = None) -> Path:
    out_dir = Path(output_dir)
    decor_dir = out_dir / "decorations"
    seg_dir = decor_dir / "segmented"
    inpaint_dir = decor_dir / "inpainted"
    results_path = decor_dir / "segment_results.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run segment_decorations first.\n"
            f"Expected: {results_path}"
        )

    inpaint_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("[inpaint_decorations] No segments — nothing to do.")
        return inpaint_dir

    # Load original descriptions from decoration_analysis.json (full-res ref)
    # These are much better than re-describing blurry segmented crops.
    deco_descs = _load_decoration_descriptions(decor_dir)

    for seg in segments:
        seg_idx = seg["seg_index"]
        if indices is not None and seg_idx not in indices:
            continue
        phrase = seg.get("phrase", "decoration")
        ftype = seg.get("furniture_type", "furniture")
        text_only = seg.get("text_only", False)
        furn_idx = seg.get("furniture_index", -1)

        # Tiny / low-resolution source: a decoration that occupies only a few
        # pixels in the photo upscales to a blurry, unrecognisable blob, and the
        # preserve-canvas path would just keep that blur. Force text-reconstruct
        # instead (the true pixel colour is still extracted below and anchored
        # into the description), so the model regenerates a CLEAN object of the
        # right colour/style rather than preserving the blur.
        if float(seg.get("coverage", 1.0) or 1.0) < 0.002:
            if not text_only:
                print(f"    tiny source (coverage={seg.get('coverage')}) — "
                      f"text-reconstruct (colour-anchored) instead of blurry preserve")
            text_only = True

        # Slugify: keep only filename-safe chars. Raw VLM phrases can contain
        # "/" (e.g. "soundbar/media device") or "#" which break the path.
        safe_phrase = "".join(c if c.isalnum() else "_" for c in phrase).strip("_") or "decoration"
        out_name = f"inpaint_{seg_idx:02d}_{safe_phrase}.png"
        out_path = inpaint_dir / out_name

        print(f"\n[inpaint_decorations] {seg_idx:02d} '{phrase}' on {ftype}"
              + (" [text-only]" if text_only else ""))

        if out_path.exists():
            print(f"    reusing existing {out_name}")
            seg["inpaint_file"] = out_name
            continue

        # Load the original photo crop for clarity checking only.
        # The canvas (SAM-segmented object on grey) is used for generation —
        # the raw crop includes surrounding room context and causes the model
        # to reproduce the full scene patch instead of a single isolated object.
        crop_ref = None
        canvas = None
        if not text_only:
            crop_file = seg.get("crop_file")
            if crop_file and (seg_dir / crop_file).exists():
                crop_ref = Image.open(seg_dir / crop_file).convert("RGB")
            canvas_file = seg.get("canvas_file")
            if canvas_file and (seg_dir / canvas_file).exists():
                canvas = Image.open(seg_dir / canvas_file).convert("RGB")

            # Skip segments whose source files are missing.  The JSON entry
            # may list crop_file/canvas_file paths from a previous run, but
            # if the actual PNGs aren't on disk the inpainter would silently
            # fall through to text-only generation and produce a generic
            # invented object — usually that's worse than dropping the segment.
            if crop_ref is None and canvas is None:
                print(f"    skipped — no crop/canvas/mask files on disk "
                      f"(crop_file={seg.get('crop_file')}, "
                      f"canvas_file={seg.get('canvas_file')})")
                seg["skipped_reason"] = "source_files_missing"
                continue

        # Always prefer decoration_analysis.json descriptions — they come from
        # careful per-furniture VLM comparison of the reference image and are
        # the most accurate account of what the object actually is.
        # Fall back to object_description from the segmentation check only if
        # no analysis entry exists.
        obj_desc = seg.get("object_description")
        description = deco_descs.get((furn_idx, phrase))
        if not description:
            for (fi, name), desc in deco_descs.items():
                if fi == furn_idx and (phrase in name or name in phrase):
                    description = desc
                    break
        if not description:
            description = obj_desc

        # Strip all furniture/spatial context from description so it doesn't
        # bleed into the generation prompt (e.g. "on the sofa", "on the backrest",
        # "arranged on the left side of the table", ": two grey and two orange...").
        if description:
            # 1. Remove everything after a colon (often "two X and two Y, arranged...")
            description = re.sub(r'\s*:.*$', '', description, flags=re.IGNORECASE).strip()
            # 2. Strip "on/in/at/near/placed on/arranged on the <furniture word>"
            _FURN_WORDS = (
                r'sofa|couch|chair|armchair|table|desk|shelf|shelves|cabinet|bookcase|'
                r'backrest|back rest|armrest|arm rest|seat|cushion|bed|dresser|'
                r'furniture|surface|floor|wall|corner|room'
            )
            description = re.sub(
                r'\s*,?\s*(arranged\s+)?(placed\s+)?(on|in|at|near|along|against|'
                r'leaning\s+against)\s+(the\s+)?(' + _FURN_WORDS + r')[\w\s-]*',
                '', description, flags=re.IGNORECASE
            ).strip().rstrip(',. ')
            # 3. Strip positional phrases ("on the left/right/center side of the...")
            description = re.sub(
                r'\s*(on|at|near|placed on|placed at|located on|in)\s+the\s+'
                r'(left|right|center|centre|far|near|front|back|middle)'
                r'[\w\s-]*(of|side|edge|corner)?\s*(of\s+)?the\s+\w+',
                '', description, flags=re.IGNORECASE
            ).strip().rstrip(',. ')

        # Extract and store quantity — we always generate a single object
        if description:
            _, quantity = _strip_quantity(description)
            if quantity > 1:
                seg["quantity"] = quantity
                print(f"    quantity: {quantity} (will generate 1, annotated as {quantity})")

        # Colour: the VLM colour label is frequently WRONG (a rose pillow called
        # "orange", a beige one "gray") and drags the generation off-tone. Read
        # the TRUE colour from the segment's own pixels and make it authoritative —
        # strip any existing colour word from the description, then inject the
        # pixel-derived one. Load the canvas/crop from disk even for text-only
        # segments (no generation reference, but we can still read the colour).
        _colour_ref = canvas if canvas is not None else crop_ref
        if _colour_ref is None:
            _cf = seg.get("canvas_file") or seg.get("crop_file")
            if _cf and (seg_dir / _cf).exists():
                try:
                    _colour_ref = Image.open(seg_dir / _cf).convert("RGB")
                except Exception:
                    _colour_ref = None
        if _colour_ref is not None:
            extracted_colour = _extract_crop_colour(_colour_ref)
            if extracted_colour:
                orig_desc = description or phrase
                description = _inject_colour_into_description(
                    _strip_colour_words(orig_desc), extracted_colour)
                if description != orig_desc:
                    print(f"    [colour] true tone '{extracted_colour}' "
                          f"(was '{orig_desc}') → '{description}'")
                else:
                    print(f"    [colour] already has colour word, extracted '{extracted_colour}'")

        print(f"    description: {description}")

        # ── Exact-pixel preservation (BEFORE the crop-clarity/text-only gate) ──
        # A faithfully-segmented, COMPLETE object should be kept verbatim — the
        # generative editor drifts colour + shape (a rust angled pillow becomes a
        # brown square), and the crop-clarity gate can even send a clear pillow to
        # generic text-only generation (e.g. when the quantity in the description
        # doesn't match the one visible instance).  So if we have a well-covered
        # canvas of a complete single object, composite its real pixels on grey
        # and skip generation entirely.  Completeness is judged on the SINGULAR
        # `phrase` (we always emit one instance; quantity is annotated separately)
        # so a 'two pillows' description doesn't mark a whole single pillow partial.
        if _DECOR_PRESERVE and not text_only and canvas is not None:
            # The CANVAS is already the SAM-segmented object on grey, correctly
            # aligned and sized — use it verbatim (the crop is a tiny zoomed box
            # and the mask is full-image, so they don't co-register).
            cov = _canvas_fg_fraction(canvas)
            if cov >= 0.20:
                is_complete = True if no_vlm else _vlm_is_complete(canvas, phrase)
                if is_complete:
                    try:
                        canvas.save(out_path)
                        seg["inpaint_file"] = out_name
                        seg["preserved"] = True
                        print(f"    preserved canvas — exact segmented pixels "
                              f"(complete, coverage={cov:.2f}) → {out_path}")
                        continue
                    except Exception as e:
                        print(f"    preserve failed ({e}) — falling back to generation")
                else:
                    print(f"    object cut-off/occluded — generative completion")

        # Determine reference image and CFG scale.
        # The crop is used only for clarity gating (is the object visible?).
        # Generation uses the canvas (SAM-segmented object on grey) — it isolates
        # just the target object without surrounding room context.
        # If the crop is unclear, skip straight to text-only regardless of canvas.
        fg = 0.0
        use_crop = False
        if text_only:
            cfg = 6.0
            ref_canvas = None
        else:
            # Gate on crop clarity first: if the crop can't confirm the object
            # is visible, don't trust the canvas either — go text-only.
            crop_clear = True
            if crop_ref is not None and not no_vlm:
                crop_clear = _vlm_crop_is_clear(crop_ref, phrase, description)

            if not crop_clear:
                print(f"    crop unclear — using text-only generation")
                ref_canvas = None
                text_only = True
                cfg = 6.0
            else:
                fg = _canvas_fg_fraction(canvas) if canvas else 0.0
                if fg < 0.15:
                    # Canvas too sparse — fall back to crop if available and clear.
                    # Mark use_crop=True so the prompt tells the model to ignore
                    # the room background in the crop and extract only the object.
                    if crop_ref is not None:
                        print(f"    fg_fraction={fg:.3f} too low — using photo crop as fallback reference")
                        ref_canvas = crop_ref
                        use_crop = True
                        fg = 1.0
                        # Crop-based: still primarily reference-driven (model
                        # has to extract the object from a busy room photo),
                        # so keep cfg moderate to follow the "ignore background"
                        # instruction without overriding the visible object.
                        cfg = 3.5
                    else:
                        print(f"    fg_fraction={fg:.3f} too low — using text only")
                        ref_canvas = None
                        text_only = True
                        # Text-only: no reference at all — must follow the
                        # description, so cfg stays high.
                        cfg = 6.0
                else:
                    # Canvas-based with good foreground coverage: the reference
                    # IS the object. Use a low cfg so the model preserves the
                    # canvas's visible identity instead of biasing toward the
                    # generic noun in the prompt.
                    cfg = 3.0
                    ref_canvas = canvas
                    print(f"    using canvas as reference")

        print(f"    fg_fraction: {fg:.3f}  cfg: {cfg:.1f}")

        # Generate with retry.  Strategy:
        #   - First attempt: trust the reference at low cfg.
        #   - On weak result (score >= 3 but not ok): nudge cfg DOWN (0.8×)
        #     so the model leans even harder on the reference.
        #   - On poor result (score < 3): the reference itself is too
        #     unrecognizable — escalate to the next fallback tier (crop, then
        #     text-only) rather than cranking cfg up, which only forces the
        #     generic-noun bias to dominate identity.
        best_result = None
        best_score = 0

        for attempt in range(MAX_RETRIES + 1):
            cur_cfg = cfg * (0.8 if attempt > 0 and best_score >= 3 else 1.0)
            if attempt > 0 and best_score < 3:
                # Reference unrecognizable — promote to the next fallback tier
                if not use_crop and not text_only and crop_ref is not None:
                    print(f"    score<3 — escalating to photo-crop fallback")
                    ref_canvas = crop_ref
                    use_crop = True
                    cur_cfg = 3.5
                elif not text_only:
                    print(f"    score<3 — escalating to text-only fallback")
                    ref_canvas = None
                    text_only = True
                    use_crop = False
                    cur_cfg = 6.0
                else:
                    # Already text-only; small bump but stay below text-prompt
                    # saturation to avoid producing cartoon-like outputs.
                    cur_cfg = min(cfg * 1.2, 7.5)

            result = _call_edit(ref_canvas, phrase, description, text_only,
                                cfg_scale=cur_cfg, use_crop=use_crop)
            if result is None:
                continue

            if no_vlm:
                score, ok = 5, True
                print(f"    quality: skipped (--no-vlm)")
            else:
                qc = _vlm_quality_check(result, phrase, description=description)
                score = qc.get("score", 3)
                ok = qc.get("ok", True)
                print(f"    quality: ok={ok} score={score} — {qc.get('reason', '')}")

            if score > best_score:
                best_score = score
                best_result = result

            if ok:
                break

        if best_result is None:
            print(f"    all attempts failed — skipping")
            continue

        best_result.save(out_path)
        seg["inpaint_file"] = out_name
        print(f"    saved → {out_path}")

    # Update results
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    done = sum(1 for s in segments if "inpaint_file" in s)
    print(f"\n[inpaint_decorations] Done. {done}/{len(segments)} decorations "
          f"inpainted → {inpaint_dir}/")
    return inpaint_dir


def main():
    ap = argparse.ArgumentParser(
        description="Inpaint partial decoration objects with Qwen image-edit."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output dir containing decorations/segment_results.json")
    ap.add_argument("--no-vlm", action="store_true",
                    help="Skip VLM crop-clarity and quality-check calls (saves GPU memory)")
    args = ap.parse_args()
    run(output_dir=args.output_dir, no_vlm=args.no_vlm)


if __name__ == "__main__":
    main()
