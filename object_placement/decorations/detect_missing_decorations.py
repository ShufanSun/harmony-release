"""
detect_missing_decorations.py — identify missing small decorations on each furniture piece.

Pipeline:
  1. Load placement_analysis.json + segment_results.json to get furniture bboxes
     and crop files.
  2. For each furniture piece, send the VLM its reference crop (from the original
     photo, with decorations) and its inpainted image (clean furniture, no
     decorations). The VLM sees exactly what's missing per piece.
  3. Save decoration_analysis.json.

Usage:
    python -m object_placement.decorations.detect_missing_decorations \
        --output-dir outputs/living_room9 \
        --image data/indoor_images/living_room9.jpg
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import requests
from PIL import Image

VLM_API_URL = "http://localhost:8080/v1/chat/completions"
from object_placement.vlm_backend import vlm_post as _vlm_post


def _encode_image(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1)


def _bbox_center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _bbox_area(box: list[int]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def _dedup_cross_furniture(results: list[dict],
                           proximity_px: int = 150) -> list[dict]:
    """Remove duplicate decoration names across nearby furniture of the same type.

    When two nearby furniture pieces (same type, bboxes within proximity_px) both
    list the same decoration name (e.g. "books"), keep it only on the piece whose
    bbox is larger — that piece more likely actually has the object on it, while
    the smaller/overlapping piece saw the same object leak into its context crop.
    """
    from collections import defaultdict
    import math

    # Group results by furniture type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["furniture_type"]].append(r)

    # For each type group, find duplicate decoration names on nearby pieces
    to_remove: dict[int, set[str]] = defaultdict(set)  # furn_idx → deco names to drop

    for ftype, group in by_type.items():
        if len(group) < 2:
            continue

        for i, a in enumerate(group):
            for b in group[i + 1:]:
                # Check proximity
                ca = _bbox_center(a["box_px"])
                cb = _bbox_center(b["box_px"])
                dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                if dist > proximity_px:
                    continue

                # Find shared decoration names
                a_names = {d["name"] for d in a["missing_decorations"]
                           if isinstance(d, dict)}
                b_names = {d["name"] for d in b["missing_decorations"]
                           if isinstance(d, dict)}
                shared = a_names & b_names
                if not shared:
                    continue

                # Keep on the larger bbox piece, remove from smaller
                area_a = _bbox_area(a["box_px"])
                area_b = _bbox_area(b["box_px"])
                loser = b if area_a >= area_b else a
                print(f"[detect_decorations] Cross-dedup: removing {shared} "
                      f"from {loser['label']} (smaller bbox, duplicate of "
                      f"{'a' if loser is b else 'b'})")
                to_remove[loser["furniture_index"]].update(shared)

    # Apply removals
    for r in results:
        drop_names = to_remove.get(r["furniture_index"])
        if drop_names:
            r["missing_decorations"] = [
                d for d in r["missing_decorations"]
                if not (isinstance(d, dict) and d["name"] in drop_names)
            ]

    # Drop decoration types we never place: throw blankets drape over furniture
    # and don't reconstruct/place well as a rigid GLB (they end up as a flat slab
    # over a backrest), so exclude them entirely.
    _SKIP_DECOR = ("throw blanket", "blanket", "throw")
    for r in results:
        before = len(r["missing_decorations"])
        r["missing_decorations"] = [
            d for d in r["missing_decorations"]
            if not (isinstance(d, dict)
                    and any(s in (d.get("name", "").lower()) for s in _SKIP_DECOR)
                    and "pillow" not in d.get("name", "").lower())
        ]
        if len(r["missing_decorations"]) != before:
            print(f"[detect_decorations] dropped blanket-type item(s) on "
                  f"{r.get('label')} (not placed)")

    return results


def _dedup_furniture(items: list[dict], iou_thresh: float = 0.40) -> list[dict]:
    """Remove furniture items whose bboxes heavily overlap with an already-kept
    item. Keeps the item with the larger bbox (more likely to be the primary
    detection)."""
    # Sort by bbox area descending — larger items kept first
    sorted_items = sorted(items,
                          key=lambda it: (it["box_px"][2] - it["box_px"][0]) *
                                         (it["box_px"][3] - it["box_px"][1]),
                          reverse=True)
    kept: list[dict] = []
    for item in sorted_items:
        box = item["box_px"]
        if any(_iou(box, k["box_px"]) >= iou_thresh for k in kept):
            print(f"[detect_decorations] Dedup: dropping {item['index']:02d} "
                  f"{item['type']} (overlaps with kept item)")
            continue
        kept.append(item)
    return kept


def _annotated_context_crop(image_np: np.ndarray, box_px: list[int],
                            other_boxes: list[list[int]] | None = None,
                            context_mult: float = 3.0,
                            canvas_size: int = 768) -> Image.Image:
    """Crop a region around the furniture bbox with enough context, then draw:
      - RED rectangle on the TARGET furniture
      - BLUE rectangles on OTHER nearby furniture visible in the crop

    This lets the VLM see which objects belong to other furniture (blue)
    and should be ignored.
    """
    from PIL import ImageDraw

    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = box_px
    bw, bh = x2 - x1, y2 - y1

    expand_x = int(bw * context_mult) // 2
    expand_y = int(bh * context_mult) // 2
    min_half = 80
    expand_x = max(expand_x, min_half)
    expand_y = max(expand_y, min_half)

    cx1 = max(0, x1 - expand_x)
    cy1 = max(0, y1 - expand_y)
    cx2 = min(W, x2 + expand_x)
    cy2 = min(H, y2 + expand_y)

    crop = Image.fromarray(image_np[cy1:cy2, cx1:cx2])
    draw = ImageDraw.Draw(crop)

    # Draw OTHER furniture in blue first (so red target is on top)
    for ob in (other_boxes or []):
        ox1, oy1 = ob[0] - cx1, ob[1] - cy1
        ox2, oy2 = ob[2] - cx1, ob[3] - cy1
        # Only draw if at least partially visible in the crop
        cw_px, ch_px = cx2 - cx1, cy2 - cy1
        if ox2 > 0 and oy2 > 0 and ox1 < cw_px and oy1 < ch_px:
            for offset in range(2):
                draw.rectangle(
                    [ox1 - offset, oy1 - offset, ox2 + offset, oy2 + offset],
                    outline=(0, 100, 255),
                )

    # Draw TARGET in red
    rx1, ry1 = x1 - cx1, y1 - cy1
    rx2, ry2 = x2 - cx1, y2 - cy1
    for offset in range(3):
        draw.rectangle(
            [rx1 - offset, ry1 - offset, rx2 + offset, ry2 + offset],
            outline=(255, 0, 0),
        )

    # Resize to square canvas
    cw, ch = crop.size
    side = max(cw, ch, 1)
    canvas = Image.new("RGB", (side, side), (185, 185, 185))
    canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return canvas.resize((canvas_size, canvas_size), Image.LANCZOS)


# ── VLM prompt ──────────────────────────────────────────────────────────────

_DETECT_PER_FURNITURE_PROMPT = """\
You are comparing two images to identify every object that is resting ON TOP of \
one specific furniture piece in the reference photo but is absent from the clean \
3D-reconstructed version.

Room type: {room_type}
This furniture is: {label} ({furniture_type})
Location in room: {notes}

IMAGE 1 (REFERENCE CROP): a zoomed-in crop of the room around this furniture. \
A RED RECTANGLE marks the exact furniture piece to analyze. \
BLUE RECTANGLES mark OTHER nearby furniture — objects on blue-boxed furniture \
are NOT on the target piece. Ignore them completely.

IMAGE 2 (CLEAN PIECE): the 3D-reconstructed version of THIS SPECIFIC furniture \
piece, isolated on a grey background — shown for size and shape reference only.

TASK:
Look at the furniture inside the RED RECTANGLE in IMAGE 1. List every object that \
is physically resting ON its surface in IMAGE 1 that does NOT appear in IMAGE 2. \
Do not try to re-identify the furniture by shape — the red rectangle already marks \
the exact piece to focus on.

RULES:
- ONLY list objects resting on the surface of the MATCHED furniture piece.
- The TOP SURFACE of the furniture is at (or just ABOVE) the TOP EDGE of the RED \
rectangle. Objects sitting on it therefore appear mostly ABOVE the red box — \
INCLUDE them. This matters for small side tables / nightstands / consoles whose \
body is boxed low while a potted plant, succulent, candle, lamp, vase, or lantern \
sits on top, extending above the rectangle. Look carefully just above the red box.
- The reference crop may show OTHER furniture nearby — ignore objects on those.
- Include ANY object on the surface, not just decorative ones. This includes: \
throw pillows, blankets, books, bowls, vases, plants, potted plants, desk lamps, \
table lamps, candles, trays, sculptures, figurines, photo frames, coasters, \
laptops, monitors, keyboards, mice, phones, tablets, remote controls, speakers, \
clocks, alarm clocks, water bottles, mugs, cups, and any other visible item.
- Do NOT list cushions — they are considered part of the sofa/chair itself.
- Do NOT list items that are part of the furniture structure (legs, armrests, etc.).
- Be specific about quantity, colour, material, and placement:
  e.g. "two orange throw pillows on the left side of the backrest"
  e.g. "an open silver laptop on the right side of the desk surface"
  e.g. "a white desk lamp on the left corner of the table"
- If you see NO objects on this piece in the reference, return an empty list.
- Trust what you can SEE in the reference image over the furniture type label — \
the label may be approximate (e.g. something called "coffee table" may be a desk). \
If you can clearly see a monitor, lamp, or keyboard on the surface, list it.
- When unsure of an object's exact identity, use the room type to pick the most \
likely interpretation: e.g. in a home_office, a small rectangular object is more \
likely a computer mouse than a remote control.
- Do NOT hallucinate or infer objects that are not visible in the reference.

OUTPUT — a JSON object:
{{
  "label": "{label}",
  "missing_decorations": [
    {{"name": "throw pillows", "description": "two orange velvet throw pillows placed on the left side of the sofa backrest"}},
    {{"name": "laptop", "description": "an open silver laptop on the right half of the desk surface"}},
    {{"name": "books", "description": "a stack of three books on the left corner of the coffee table"}},
    ...
  ]
}}

No markdown, no explanation — ONLY the JSON.
"""


def _detect_per_furniture(
    annotated_ref: Image.Image,
    clean_img: Image.Image,
    label: str,
    furniture_type: str,
    notes: str,
    room_type: str = "room",
) -> dict | None:
    """VLM call for one furniture piece: compare annotated reference vs clean inpaint."""
    ref_b64, ref_mime = _encode_image(annotated_ref)
    clean_b64, clean_mime = _encode_image(clean_img)

    prompt = _DETECT_PER_FURNITURE_PROMPT.format(
        label=label,
        furniture_type=furniture_type.replace("_", " "),
        notes=notes or "no additional context",
        room_type=room_type.replace("_", " "),
    )

    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:{clean_mime};base64,{clean_b64}"}},
            ],
        }],
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    try:
        resp = _vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])

        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            print(f"    [VLM] Unexpected format: {raw[:200]}")
            return None

        return json.loads(m.group())

    except Exception as e:
        print(f"    [VLM] Failed: {e}")
        return None


# Reference-presence verification — drop hallucinated detections.  Disable with
# SCENEWEAVE_NO_PRESENCE_CHECK=1.
import os as _os
VERIFY_PRESENCE = _os.environ.get("SCENEWEAVE_NO_PRESENCE_CHECK") != "1"

_PRESENCE_PROMPT = """\
This is a crop of a REAL interior photograph.  A thick RED rectangle marks one
piece of furniture: a {label}.

Decide whether this specific object is CLEARLY and UNAMBIGUOUSLY visible, resting
ON or against the marked {label}, in THIS photo:

  object: {name} — {description}

Answer NO if ANY of these hold:
- you cannot clearly see the object,
- it is actually part of the wall, floor, ceiling, a shadow, reflection, seam,
  trim, or a thin line / pole that is not really this object,
- it is a different kind of object than described,
- it belongs to a different piece of furniture, not the marked one,
- or you are at all unsure it is genuinely present.

Answer YES only if a real, distinct {name} is plainly visible on the marked {label}.
Respond with a single word: YES or NO.
"""


def _verify_present_in_ref(ref_crop: Image.Image, decoration: dict, label: str,
                           timeout: int = 60) -> bool:
    """Return True if `decoration` is genuinely visible in the REFERENCE crop.

    Skeptical by design (the prompt rejects on any doubt) so hallucinations are
    dropped, but lenient on transport errors (keeps) so a VLM hiccup never
    silently deletes a real decoration."""
    name = decoration.get("name", "") if isinstance(decoration, dict) else str(decoration)
    desc = decoration.get("description", name) if isinstance(decoration, dict) else name
    try:
        b64, mime = _encode_image(ref_crop)
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _PRESENCE_PROMPT.format(
                    label=label, name=name, description=desc)},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        resp = _vlm_post(payload, timeout=timeout)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"]).upper()
        toks = re.findall(r"\b(YES|NO)\b", raw)
        verdict = toks[-1] if toks else "YES"     # keep if unparseable
        return verdict != "NO"
    except Exception as e:
        print(f"      [presence] verify error ({e}) — keeping (lenient)")
        return True


_UNPLACED_PROMPT = """\
This image is a render of the CURRENT 3D-reconstructed scene — the furniture and
fixtures that have ALREADY been placed.  A RED rectangle marks one piece: a {label}.

Small decorations may or may not have been placed yet.  Decide whether THIS specific
decoration is ALREADY present and clearly visible on the marked {label} in THIS render:

  decoration: {name} — {description}

Answer YES if it is ALREADY there (it has been placed — do NOT add it again).
Answer NO  if it is ABSENT from the render (it still needs to be placed).

Judge only the marked piece.  Respond with a single word: YES or NO.
"""


def _verify_unplaced(scene_crop: Image.Image, decoration: dict, label: str,
                     timeout: int = 60) -> bool:
    """Return True if `decoration` is NOT yet present in the current scene crop
    (i.e. it is UNPLACED and should be added).

    Lenient: on any error / unparseable answer it returns True (keep) so a VLM
    hiccup never silently drops a real decoration."""
    name = decoration.get("name", "") if isinstance(decoration, dict) else str(decoration)
    desc = decoration.get("description", name) if isinstance(decoration, dict) else name
    try:
        b64, mime = _encode_image(scene_crop)
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _UNPLACED_PROMPT.format(
                    label=label, name=name, description=desc)},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        resp = _vlm_post(payload, timeout=timeout)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"]).upper()
        toks = re.findall(r"\b(YES|NO)\b", raw)
        verdict = toks[-1] if toks else "NO"      # default keep (treat as unplaced)
        return verdict != "YES"                    # YES = already placed → drop
    except Exception as e:
        print(f"      [unplaced] verify error ({e}) — keeping (lenient)")
        return True


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(output_dir: str | Path, image_path: str) -> Path:
    out_dir = Path(output_dir)
    furniture_dir = out_dir / "furniture"
    seg_dir = furniture_dir / "segmented"
    inpaint_dir = furniture_dir / "inpainted"
    decor_dir = out_dir / "decorations"
    decor_dir.mkdir(parents=True, exist_ok=True)

    placement_path = furniture_dir / "placement_analysis.json"
    if not placement_path.exists():
        raise FileNotFoundError(
            f"placement_analysis.json not found — run furniture placement first.\n"
            f"Expected: {placement_path}"
        )

    seg_results_path = furniture_dir / "segment_results.json"
    if not seg_results_path.exists():
        raise FileNotFoundError(
            f"segment_results.json not found — run segment_furniture first.\n"
            f"Expected: {seg_results_path}"
        )

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(placement_path) as f:
        placement = json.load(f)
    with open(seg_results_path) as f:
        seg_data = json.load(f)

    # Load room type for context-aware plausibility checks
    floorplan_path = out_dir / "floorplan_analysis.json"
    _room_type = "room"
    if floorplan_path.exists():
        try:
            _fp = json.load(open(floorplan_path))
            _room_type = _fp.get("room", {}).get("room_type", "room")
        except Exception:
            pass

    # Build index → segment data lookup
    seg_by_index: dict[int, dict] = {}
    for seg in seg_data.get("segments", []):
        seg_by_index[seg["index"]] = seg

    image_np = np.array(Image.open(image_path).convert("RGB"))

    furniture_items = placement.get("placement_order", [])
    if not furniture_items:
        print("[detect_decorations] No furniture in placement_analysis.json.")
        return decor_dir

    # Types that typically have decorations on them
    _SURFACE_TYPES = {
        "sofa", "chair", "armchair", "coffee_table", "dining_table",
        "desk", "side_table", "bookcase", "cabinet", "bed",
        # Other table/surface types that commonly hold decorations (statues,
        # vases, candles) — without these they were skipped and their decor
        # never detected (005212: pedestal_table statues missed).
        "pedestal_table", "end_table", "console_table", "console",
        "nesting_table", "nesting_side_table", "nightstand", "round_table",
        "tv_stand", "sideboard", "dresser", "credenza", "shelf", "shelving",
        "buffet", "chest_of_drawers", "drawer", "stand",
        # Ottomans/poufs are seating but VERY commonly used as coffee-table
        # surfaces (a tray, plant or books on top); without these their decor
        # is never detected (pexels_2343465: the plant on the round ottoman was
        # missed). Safe to add — the VLM only reports what is actually on top.
        "ottoman", "pouf", "footstool",
        # Fireplace mantels very commonly hold decorations (clock, candlesticks,
        # vases, framed photos). Without this the mantel-top decor is skipped
        # (elegant: objects on the fireplace mantel were missed).
        "fireplace", "mantel", "mantelpiece",
    }

    # Filter to surface types and build unique labels
    type_counters: dict[str, int] = {}
    surface_items = []
    for item in furniture_items:
        ftype = item["type"]
        # Substring match so compound/variant labels are still recognised as
        # surfaces (004992: "nesting_coffee_table_set" contains "coffee_table" but
        # failed the exact-set test, so its vases/books were never detected).
        if not any(k in ftype for k in _SURFACE_TYPES):
            print(f"[detect_decorations] Skipping {item['index']:02d} {ftype} "
                  f"— not a surface type")
            continue
        surface_items.append(item)

    # Deduplicate near-overlapping bboxes (keeps larger bbox)
    surface_items = _dedup_furniture(surface_items)

    # Build unique labels after dedup
    for item in surface_items:
        ftype = item["type"]
        type_counters[ftype] = type_counters.get(ftype, 0) + 1
        item["_label"] = f"{ftype.replace('_', ' ')} {type_counters[ftype]}"

    if not surface_items:
        print("[detect_decorations] No surface furniture found.")
        return decor_dir

    # ── Per-furniture VLM calls ──
    results = []
    for item in surface_items:
        idx = item["index"]
        ftype = item["type"]
        label = item["_label"]
        box_px = item["box_px"]
        notes = item.get("notes", "")

        print(f"\n[detect_decorations] {idx:02d} {label}")

        # Zoomed crop with red rectangle on target, blue on other furniture
        seg = seg_by_index.get(idx)
        other_boxes = [it["box_px"] for it in surface_items
                       if it["index"] != idx]
        annotated_ref = _annotated_context_crop(
            image_np, box_px, other_boxes=other_boxes)

        # Clean reconstruction: the inpainted furniture image (no decorations)
        inpaint_file = seg.get("inpaint_file") if seg else None
        if inpaint_file and (inpaint_dir / inpaint_file).exists():
            clean_img = Image.open(inpaint_dir / inpaint_file).convert("RGB")
        elif inpaint_file and (seg_dir / inpaint_file).exists():
            clean_img = Image.open(seg_dir / inpaint_file).convert("RGB")
        else:
            print(f"  No inpainted image found — skipping")
            continue

        # Save comparison pair for debugging
        annotated_ref.save(decor_dir / f"compare_{idx:02d}_{ftype}_ref.png")
        clean_img.save(decor_dir / f"compare_{idx:02d}_{ftype}_clean.png")

        vlm_result = _detect_per_furniture(
            annotated_ref, clean_img, label, ftype, notes, room_type=_room_type)

        if vlm_result is None:
            print(f"  VLM failed — skipping")
            continue

        missing = vlm_result.get("missing_decorations", [])
        # Cushions are part of the sofa/chair — skip them
        _SKIP_NAMES = {"cushion", "cushions", "seat cushion", "seat cushions",
                       "back cushion", "back cushions",
                       "folded towel", "folded towels", "towel", "towels"}

        def _skip_decor(nm: str) -> bool:
            nm = nm.lower().strip()
            if nm in _SKIP_NAMES:
                return True
            # Floor / standing / tripod lamps are FLOOR FURNITURE, not tabletop
            # decorations — they are handled by the furniture stage and must not
            # be placed on a table here.  (Small table/desk lamps that genuinely
            # sit on a surface are kept.)
            if "floor lamp" in nm or "standing lamp" in nm or "tripod lamp" in nm:
                return True
            return False

        missing = [
            m for m in missing
            if isinstance(m, dict) and not _skip_decor(m.get("name", ""))
        ]

        # ── Reference-presence gate ──────────────────────────────────────────
        # The detector occasionally hallucinates an object that isn't actually
        # on the piece (e.g. reading a 'desk lamp' off a thin wall line).  Show
        # the VLM the REFERENCE crop and ask, skeptically, whether each detected
        # decoration is genuinely visible on the marked furniture; drop on a
        # clear NO so only items truly present in the photo get reconstructed.
        if VERIFY_PRESENCE and missing:
            _kept = []
            for _m in missing:
                if _verify_present_in_ref(annotated_ref, _m, label):
                    _kept.append(_m)
                else:
                    _nm = _m.get("name", _m) if isinstance(_m, dict) else _m
                    print(f"    [presence] drop '{_nm}' — not clearly visible in reference")
            missing = _kept

        entry = {
            "furniture_index": idx,
            "furniture_type": ftype,
            "label": label,
            "box_px": box_px,
            "notes": notes,
            "missing_decorations": missing,
        }
        results.append(entry)

        print(f"  Missing: "
              f"{[m['name'] if isinstance(m, dict) else m for m in missing]}")

    # ── Keep only UNPLACED decorations ───────────────────────────────────────
    # The per-furniture detector lists every decoration visible in the PHOTO.
    # Some may already exist in the reconstructed scene (baked into a furniture
    # GLB, or placed by an earlier decoration run).  Per requirement, only place
    # decorations that are NOT already in the scene: check each against the
    # current furnished+ceiling render and drop any the VLM confirms are present.
    _scene_render_np = None
    for _cand in (decor_dir.parent / "ceiling" / "render_ceiling_placed.png",
                  furniture_dir / "render_furniture_placed.png"):
        if _cand.exists():
            try:
                _si = Image.open(_cand).convert("RGB").resize(
                    (image_np.shape[1], image_np.shape[0]), Image.LANCZOS)
                _scene_render_np = np.array(_si)
                print(f"[detect_decorations] unplaced-check against {_cand.name}")
                break
            except Exception as _e:
                print(f"[detect_decorations] could not load {_cand.name}: {_e}")
    import os
    # The unplaced-check is a VLM judgment that can false-positive (e.g. a table
    # lamp on a thin desk wrongly judged "already present" against a render that
    # has no lamp), silently dropping a real decoration. SCENEWEAVE_DECOR_NO_UNPLACED_CHECK=1
    # skips it so detected decorations are kept.
    if _scene_render_np is not None and not os.environ.get("SCENEWEAVE_DECOR_NO_UNPLACED_CHECK"):
        for entry in results:
            _missing = entry.get("missing_decorations", [])
            if not _missing:
                continue
            _other = [it["box_px"] for it in surface_items
                      if it["index"] != entry["furniture_index"]]
            _scene_crop = _annotated_context_crop(
                _scene_render_np, entry["box_px"], other_boxes=_other)
            _kept = []
            for _deco in _missing:
                if _verify_unplaced(_scene_crop, _deco, entry.get("label", "piece")):
                    _kept.append(_deco)
                else:
                    _nm = _deco.get("name", _deco) if isinstance(_deco, dict) else _deco
                    print(f"  [unplaced] idx={entry['furniture_index']} drop "
                          f"'{_nm}' — already present in scene")
            entry["missing_decorations"] = _kept

    # Cross-furniture dedup: remove duplicate decoration names on nearby
    # same-type furniture (e.g. two coffee tables both listing "books")
    results = _dedup_cross_furniture(results)

    # Save decoration analysis
    analysis_path = decor_dir / "decoration_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "scene_summary": placement.get("scene_summary", ""),
            "furniture_decorations": results,
        }, f, indent=2)

    print(f"\n[detect_decorations] {len(results)} furniture piece(s) analyzed")
    print(f"[detect_decorations] Results → {analysis_path}")
    return decor_dir


def main():
    ap = argparse.ArgumentParser(
        description="Detect missing decorations on furniture using VLM."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output dir (decorations/ is created inside)")
    ap.add_argument("--image", required=True,
                    help="Reference room photograph")
    args = ap.parse_args()
    run(output_dir=args.output_dir, image_path=args.image)


if __name__ == "__main__":
    main()
