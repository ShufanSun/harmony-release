"""
segment_wall_objects.py — segment wall-mounted objects using Grounded-SAM.

Pipeline:
  1. Call VLM to identify which TYPES of wall-mounted objects are present
     (window, curtain, art, shelf, sconce, …) — no per-instance details needed.
  2. Build a single GroundingDINO text prompt from all unique types.
  3. Run GroundingDINO on the full image → bounding boxes for all instances.
  4. Run SAM (box-prompted) on each box → pixel-precise segmentation mask.
  5. Save per-instance outputs to <output_dir>/wall_mounted/segmented/:
       segment_00_window_mask.png   — RGBA, object opaque / bg transparent
       segment_00_window_canvas.png — object on grey square canvas
  6. Write segment_results.json with all detections + file paths.

Usage:
    python -m object_placement.wall_mounted.segment_wall_objects \\
        --output-dir outputs/20260331_031530 \\
        --image data/indoor_images/living_room9.jpg
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

# ── Grounded-SAM paths ────────────────────────────────────────────────────────

_REPO_ROOT  = Path(__file__).resolve().parents[2]
_GSA_ROOT   = _REPO_ROOT / "Grounded-Segment-Anything"
_GDINO_CFG  = _GSA_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_GDINO_CKPT = _GSA_ROOT / "weights/groundingdino_swint_ogc.pth"
_SAM_CKPT   = _GSA_ROOT / "weights/sam_vit_h_4b8939.pth"

sys.path.insert(0, str(_GSA_ROOT / "GroundingDINO"))
sys.path.insert(0, str(_GSA_ROOT / "segment_anything"))

# ── Config ────────────────────────────────────────────────────────────────────

VLM_API_URL    = "http://localhost:8080/v1/chat/completions"

# Route VLM chat through the shared backend router (Qwen local / gpt-5.5 NVIDIA).
from object_placement.vlm_backend import vlm_post as _vlm_post
BOX_THRESHOLD  = 0.22   # was 0.28 — lowered so unusual window shapes
                         # (floor-to-ceiling glass walls, narrow transom
                         # strips) survive GroundingDINO's box filter.
TEXT_THRESHOLD = 0.18   # was 0.22 — paired drop so multi-phrase prompts
                         # (e.g. "floor-to-ceiling window") still match.
SAM_MODEL_TYPE = "vit_h"
OBJ_IMG_SIZE   = 768      # canvas output size
_BG_GREY       = (185, 185, 185)
_CANVAS_PAD        = 0.08   # fraction of object size added as padding on canvas (default)
_CANVAS_PAD_FRAME  = 0.15   # larger pad for windows/doors — frame reaches bbox edge
_MAX_BOX_AREA  = 0.65     # ignore boxes covering >65% of image area (whole-scene matches)
# After GroundingDINO (deliberately low thresholds let unusual windows through,
# which also lets false positives through — a thin frame-edge strip scored 0.24
# as a "horizontal window strip"), ask the VLM to confirm each box in BROADER
# context (full photo with the box drawn). Drops only on a clear NO; lenient on
# errors so a VLM hiccup never removes a real object. Disable with
# SCENEWEAVE_NO_SEGMENT_VERIFY=1.
VERIFY_SEGMENTS = os.environ.get("SCENEWEAVE_NO_SEGMENT_VERIFY") != "1"

# ── Type → GroundingDINO noun phrase ─────────────────────────────────────────
# Kept short — GroundingDINO works best with simple noun phrases.
_TYPE_GDINO: dict[str, str] = {
    # Multi-phrase window prompt — bare "window" misses (a) horizontal
    # transom / clerestory strip windows tucked under the ceiling and
    # (b) full-height glass-wall installations (floor-to-ceiling glass
    # patios / sliding-door walls).  Both routinely get mis-classified
    # as "painting" or skipped entirely.  Listing several specific noun
    # phrases broadens GDINO's recall without sacrificing precision —
    # the per-detection IoU dedup downstream collapses any duplicates.
    "window":        ("window . large window . picture window . "
                       "glass window . floor-to-ceiling window . "
                       "glass wall . sliding glass door . "
                       "transom window . clerestory window . "
                       "horizontal window strip"),
    "door":          "door",
    "curtain":       "curtain",
    "shelf":         "shelf",
    "tv":            "television",
    "art":           "painting . framed picture . poster",
    "mirror":        "mirror . bathroom mirror . large wall mirror . framed mirror . vanity mirror",
    "cabinet":       "wall cabinet . wall-mounted cabinet . bathroom wall cabinet . medicine cabinet . wall cupboard . hanging storage cabinet",
    "decorative_panel": ("decorative wall panel . lattice screen . fretwork screen . "
                          "carved wood screen . room divider panel . perforated panel . "
                          "decorative wall screen . ornamental panel . wood partition screen"),
    "panel":         ("decorative wall panel . lattice screen . fretwork screen . "
                      "carved wood screen . room divider panel . perforated panel . "
                      "decorative wall screen . ornamental panel . wood partition screen"),
    "screen":        ("decorative screen . lattice screen . fretwork screen . room divider . "
                      "carved wood panel . perforated wall panel"),
    "clock":         "clock",
    "light":         "lamp",
    # "ceiling_light" excluded — too easily confused with wall lamps
    "radiator":      "radiator",
    "air_conditioner": ("air conditioner . split air conditioner . wall mounted ac unit . "
                        "mini split indoor unit . hvac wall unit"),
    "speaker":       "wall speaker . mounted speaker",
    "other":         "object",
}

# Types we always feed to GroundingDINO regardless of what the VLM returned.
# The VLM type-detector is non-deterministic and routinely misses obvious
# items (windows in well-lit living rooms, curtains, art) that GDINO would
# otherwise find easily. Letting GDINO + its score thresholds gate detection
# is far more reliable than letting one Qwen call decide what's in the room.
# Keep this list to types that (a) are common in residential interiors and
# (b) GDINO recognises well from a short noun phrase.
_ALWAYS_INCLUDE_TYPES: list[str] = ["window", "curtain", "art", "light", "air_conditioner"]

# Scene-specific GroundingDINO phrases the VLM writes per object (type -> phrase),
# populated by detect_object_types(). build_gdino_prompt() prefers these precise,
# per-scene phrases over the generic _TYPE_GDINO dictionary — this is what closes
# the detection-precision gap vs a dense all-at-once segmenter (the phrasing, not
# the masks, is what differs). Reset per detection call.
_VLM_PHRASES: dict[str, str] = {}


# ── VLM: ask which types are present ─────────────────────────────────────────

_TYPE_DETECTION_PROMPT = """\
Look at this interior room photograph.

List EVERY wall-mounted or wall-attached object that is clearly visible, as a
short lowercase noun (open vocabulary — name what you actually see). Common ones:
  window, door, curtain, shelf, tv, painting (art), mirror, clock, wall light
  (sconce), radiator, air conditioner, wall speaker, vent, thermostat, intercom,
  fuse box, wall fan
Include anything else mounted on the wall/ceiling even if not listed above.
Be thorough — do NOT miss units high on the wall (e.g. a split air-conditioner
above a window or door). Objects that REST ON or hang above a piece of furniture
(a TV on a media console, items on a shelf) are handled by the decoration stage,
not here — list only things FIXED to the bare wall.

Rules:
- "light"         = wall sconce whose bracket is physically bolted to the wall surface;
                    the arm or rod must originate FROM the wall, not from the floor or furniture.
                    EXCLUDE: floor lamps (base on the floor), torchieres, desk lamps (base on table),
                    any lamp whose rod/stick connects downward to furniture or the ground,
                    and any lamp that spatially overlaps with or appears to rest on top of
                    furniture (sofa, table, shelf, cabinet) — if it overlaps furniture it is
                    almost certainly a desk or floor lamp, not a wall sconce.
- "curtain"       = fabric panel hanging from a wall-mounted rod
- Include a category only if at least one instance is clearly visible.
- Do NOT include furniture, floor lamps, desk lamps, table lamps, or anything not attached to a wall/ceiling.

For EACH object, also write a PRECISE, SHORT grounding phrase describing what you
actually see — this drives an open-vocabulary object detector (GroundingDINO),
which works best on concrete 1-3 word noun phrases. Make the phrase specific to the
ACTUAL object (shape / material / kind), not the generic category, and add 1-2
alternative phrasings separated by " . ". Examples:
  round mirror  →  "round mirror . circular wall mirror"
  a carved wood screen  →  "lattice screen . carved wood screen . fretwork panel"
  an arched window  →  "arched window . round-top window"
  a recessed downlight  →  "recessed light . downlight . ceiling spotlight"
Prefer the distinctive visual phrase a detector could localize; avoid vague words
like "object" or "decor".

OUTPUT — a JSON array of objects, each {"type": "<snake_case category>",
"gdino": "<short specific noun phrase . alt phrasing>"}. Example:
  [{"type":"mirror","gdino":"round mirror . circular wall mirror"},
   {"type":"window","gdino":"arched window . round-top window"},
   {"type":"decorative_panel","gdino":"lattice screen . fretwork panel"}]
No markdown, no explanation.
"""


def _encode_image(path: str) -> tuple[str, str]:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def detect_object_types(image_path: str) -> list[str]:
    """Ask the VLM which wall-mounted object types are present. Returns a list of type strings."""
    b64, mime = _encode_image(image_path)
    payload = {
        "model": "qwen3",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",      "text": _TYPE_DETECTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    print("[segment] Calling VLM to identify object types …")
    global _VLM_PHRASES
    _VLM_PHRASES = {}
    resp = _vlm_post(payload, timeout=120)
    resp.raise_for_status()
    raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])

    # Parse JSON array (greedy — the array holds nested {…} objects)
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        print(f"[segment] VLM returned unexpected format: {raw[:200]}")
        return []
    try:
        items = json.loads(m.group())
        if isinstance(items, list):
            # OPEN vocabulary: keep whatever wall-mounted objects the VLM names.
            # New format = [{"type","gdino"}]; old format = ["type", …] (still
            # accepted). The per-object "gdino" phrase is stashed in _VLM_PHRASES
            # so build_gdino_prompt() grounds a precise scene-specific query;
            # novel/unknown types fall back to _TYPE_GDINO or the noun itself and
            # are confirmed by the per-box VLM _verify_segment pass.
            norm: list[str] = []
            for it in items:
                if isinstance(it, dict):
                    t = str(it.get("type", "")).strip().lower()
                    phrase = str(it.get("gdino", "")).strip()
                elif isinstance(it, str):
                    t, phrase = it.strip().lower(), ""
                else:
                    continue
                t = re.sub(r"\s+", "_", t)
                if not t:
                    continue
                if phrase:
                    _VLM_PHRASES[t] = phrase
                if t not in norm:
                    norm.append(t)
            print(f"[segment] VLM identified wall-mounted objects (open list): {norm}")
            if _VLM_PHRASES:
                print(f"[segment] VLM scene-specific gdino phrases: {_VLM_PHRASES}")
            return norm
    except json.JSONDecodeError:
        pass
    print(f"[segment] Could not parse VLM response: {raw[:200]}")
    return []


_VERIFY_PROMPT = """\
This interior room photograph has ONE region outlined by a thick red rectangle.

GroundingDINO detected the outlined region as a "{phrase}" (category: {obj_type}).
Using the BROADER CONTEXT of the whole room, decide whether the outlined region is
GENUINELY a real, distinct {obj_type} that is fixed to a WALL (a vertical wall
surface). This stage handles WALL-MOUNTED objects ONLY.

A mirror or picture hung on the wall above a dresser/console is still wall-mounted
(accept it). But a TV or object that rests on / hangs above a media console is
handled by the decoration stage — reject it here.

Answer NO (reject) if the outlined region is any of:
- a CEILING fixture — a chandelier, pendant light, hanging lamp, flush-mount
  ceiling light, track light, or ceiling fan. Anything attached to or hanging from
  the CEILING is NOT wall-mounted; it is handled by a different (furniture/lighting)
  stage, so reject it here even if it is a real light.
- a floor lamp, torchiere, desk/table lamp, or any lamp resting on the floor or on
  furniture,
- the wall–ceiling junction, crown molding, a cove/strip light, or a ceiling line,
- a shadow, glare, or seam, OR a reflection cast on a glossy surface (floor,
  countertop, glass) — BUT a framed WALL MIRROR is a real wall-mounted object:
  accept it, do NOT reject it merely because it shows a reflection of the room,
- a free-standing or floor-resting cabinet, vanity, dresser, sink unit, or base
  cabinetry that sits on the floor (this is FURNITURE, handled elsewhere) — accept
  a "cabinet" ONLY if it is a distinct unit mounted high on the wall, clear of the
  floor, with empty wall visible beneath it,
- the top or edge of a piece of furniture, a plant, or another object,
- a toilet, WC, bidet, urinal, cistern, sink, or washbasin — sanitaryware is a
  fixture handled by the furniture/fixture stage, NOT a wall-mounted decor object,
  even when it is wall-hung; reject it here,
- a TV / screen / object resting on or hanging above a media console (decoration stage),
- plain empty wall, floor, or curtain fabric mistaken for glass,
- only a thin sliver running along the very edge of the photo,
- or anything that is NOT a clearly visible, real {obj_type}.

For a "light", answer YES ONLY for a WALL SCONCE whose bracket/arm is bolted to and
originates from the vertical wall. Reject EVERY ceiling-hung or ceiling-mounted light.

Answer YES only if it is unmistakably a real, distinct {obj_type} fixed to a wall.
Respond with a single word: YES or NO.
"""


def _verify_segment(image_np: np.ndarray, box_px: list[int], obj_type: str,
                    phrase: str, timeout: int = 60) -> bool:
    """Ask the VLM to confirm a detected box is really a wall-mounted `obj_type`,
    shown IN CONTEXT (the full photo with the box drawn). Returns True to keep.

    Lenient by design: keeps the segment on any error / unparseable answer, and
    drops ONLY on an explicit NO — so a VLM outage can never silently delete real
    objects, while obvious false positives (a 0.24-score frame-edge 'window
    strip') get filtered. Gated by VERIFY_SEGMENTS."""
    try:
        from PIL import ImageDraw
        im = Image.fromarray(image_np).convert("RGB")
        draw = ImageDraw.Draw(im)
        x1, y1, x2, y2 = [int(v) for v in box_px]
        w = max(4, int(0.006 * max(im.size)))
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=w)
        buf = io.BytesIO(); im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _VERIFY_PROMPT.format(obj_type=obj_type, phrase=phrase)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        resp = _vlm_post(payload, timeout=timeout)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"]).upper()
        toks = re.findall(r"\b(YES|NO)\b", raw)
        verdict = toks[-1] if toks else "YES"     # default keep if unparseable
        print(f"          VLM verify '{obj_type}' ({phrase}): {verdict} "
              f"→ {'keep' if verdict != 'NO' else 'DROP'}")
        return verdict != "NO"
    except Exception as e:
        print(f"          VLM verify error ({e}) — keeping (lenient)")
        return True


def build_gdino_prompt(types: list[str]) -> str:
    """Convert types to a GroundingDINO ' . '-separated prompt.

    Prefers the VLM's PRECISE, scene-specific phrase (``_VLM_PHRASES``, e.g.
    "round mirror . circular wall mirror") for localization precision, and UNIONS
    it with the generic ``_TYPE_GDINO`` phrase (if the type is known) for recall.
    Falls back to the bare noun for unknown types with no VLM phrase.
    """
    seen: set[str] = set()
    phrases: list[str] = []

    def _add(chunk: str) -> None:
        for part in str(chunk).split(" . "):
            part = part.strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                phrases.append(part)

    for t in types:
        vlm   = _VLM_PHRASES.get(t)
        fixed = _TYPE_GDINO.get(t)
        if vlm:
            _add(vlm)
        if fixed:
            _add(fixed)
        if not vlm and not fixed:
            _add(t.replace("_", " "))
    return " . ".join(phrases)


# ── Grounded-SAM inference ────────────────────────────────────────────────────

def _load_models(device: str):
    from groundingdino.util.inference import load_model
    from segment_anything import SamPredictor, sam_model_registry

    print(f"[segment] Loading GroundingDINO …")
    gdino = load_model(str(_GDINO_CFG), str(_GDINO_CKPT), device=device)
    print(f"[segment] Loading SAM …")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(_SAM_CKPT))
    sam.to(device=device)
    return gdino, SamPredictor(sam)


def _gdino_predict(gdino_model, image_path: str, prompt: str, device: str):
    from groundingdino.util.inference import load_image, predict
    _, image_tensor = load_image(image_path)
    boxes, logits, phrases = predict(
        model=gdino_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device,
    )
    return boxes.numpy(), logits.numpy(), phrases


def _boxes_to_px(boxes_norm: np.ndarray, img_w: int, img_h: int) -> list[list[int]]:
    result = []
    for b in boxes_norm:
        x1 = max(0, int((b[0] - b[2] / 2) * img_w))
        y1 = max(0, int((b[1] - b[3] / 2) * img_h))
        x2 = min(img_w, int((b[0] + b[2] / 2) * img_w))
        y2 = min(img_h, int((b[1] + b[3] / 2) * img_h))
        result.append([x1, y1, x2, y2])
    return result


def _filter_large_boxes(
    boxes_norm: np.ndarray, logits: np.ndarray, phrases: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    areas = boxes_norm[:, 2] * boxes_norm[:, 3]
    keep  = areas <= _MAX_BOX_AREA
    return boxes_norm[keep], logits[keep], [p for p, k in zip(phrases, keep) if k]


def _filter_contained_boxes(
    boxes_norm: np.ndarray, logits: np.ndarray, phrases: list[str],
    containment_thresh: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop boxes that are mostly contained inside a higher-scoring box.

    A box B is dropped if its intersection with any other box A covers
    ≥ containment_thresh of B's own area (i.e. B is a sub-detection of A).
    """
    n = len(boxes_norm)
    if n == 0:
        return boxes_norm, logits, phrases

    # Convert cx,cy,w,h → x1,y1,x2,y2
    x1 = boxes_norm[:, 0] - boxes_norm[:, 2] / 2
    y1 = boxes_norm[:, 1] - boxes_norm[:, 3] / 2
    x2 = boxes_norm[:, 0] + boxes_norm[:, 2] / 2
    y2 = boxes_norm[:, 1] + boxes_norm[:, 3] / 2
    areas = boxes_norm[:, 2] * boxes_norm[:, 3]

    # Group phrases that represent collections (e.g. "wall art") rather than
    # individual items — if a group box contains individual detections of the
    # same type, drop the group box and keep the individuals.
    _GROUP_PHRASES = {"wall art", "gallery wall", "artwork", "artworks",
                       "gallery", "art collection"}

    # First pass: count how many individual boxes each group box contains
    contained_count: dict[int, int] = {}   # group_box_j → count of contained
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ix1_ = max(x1[i], x1[j])
            iy1_ = max(y1[i], y1[j])
            ix2_ = min(x2[i], x2[j])
            iy2_ = min(y2[i], y2[j])
            if ix2_ <= ix1_ or iy2_ <= iy1_:
                continue
            inter_ = (ix2_ - ix1_) * (iy2_ - iy1_)
            if (inter_ / (areas[i] + 1e-6) >= containment_thresh
                    and phrases[j].lower() in _GROUP_PHRASES):
                contained_count[j] = contained_count.get(j, 0) + 1

    keep = [True] * n
    # Drop group boxes that contain multiple individual detections
    for j, cnt in contained_count.items():
        if cnt >= 2:
            print(f"  [filter] dropping group box {j} '{phrases[j]}' "
                  f"(contains {cnt} individual detections)")
            keep[j] = False

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # Intersection of box i with box j
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            # If box i is mostly inside box j, drop i (keep higher-score j)
            if inter / (areas[i] + 1e-6) >= containment_thresh and logits[i] <= logits[j]:
                print(f"  [filter] dropping contained box {i} '{phrases[i]}' "
                      f"(inside box {j} '{phrases[j]}')")
                keep[i] = False
                break

    keep_arr = np.array(keep)
    return boxes_norm[keep_arr], logits[keep_arr], [p for p, k in zip(phrases, keep) if k]


def _sam_segment_box(predictor, image_np: np.ndarray, box_px: list[int],
                     obj_type: str = "other") -> tuple[np.ndarray, float]:
    """Run SAM and return a mask tightly bounded to the GDINO box.

    For wall-mounted detections we constrain the resulting mask to lie inside
    the GroundingDINO bounding box (with a small dilation so that frames /
    edges that extend slightly past the box are still kept). Without this
    constraint SAM occasionally returns a "scene" mask that bleeds into nearby
    curtains, walls, or furniture — producing canvases that look like a
    rectangular crop of the source photo rather than just the target object.
    Among the three multimask candidates we pick the one whose own bbox
    overlaps the GDINO box best (IoU), preferring tighter object masks over
    the broad scene mask.
    """
    predictor.set_image(image_np)
    box_arr = np.array(box_px)
    masks, scores, _ = predictor.predict(box=box_arr[None], multimask_output=True)

    H, W = image_np.shape[:2]
    bx1, by1, bx2, by2 = box_px
    bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
    # Allow a small frame-overhang outside the GDINO box.
    overhang = 0.10 if obj_type in ("window", "door") else 0.05
    dx1 = max(0, int(bx1 - bw * overhang))
    dy1 = max(0, int(by1 - bh * overhang))
    dx2 = min(W, int(bx2 + bw * overhang))
    dy2 = min(H, int(by2 + bh * overhang))

    box_area = bw * bh

    def _score_candidate(m: np.ndarray, s: float) -> float:
        if not m.any():
            return -1.0
        ys, xs = np.where(m)
        mx1, my1 = int(xs.min()), int(ys.min())
        mx2, my2 = int(xs.max()), int(ys.max())
        ix1 = max(mx1, bx1); iy1 = max(my1, by1)
        ix2 = min(mx2, bx2); iy2 = min(my2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return -1.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        m_area = (mx2 - mx1) * (my2 - my1)
        union = m_area + box_area - inter
        iou = inter / max(union, 1)
        # Penalize masks whose bbox is much larger than the GDINO box —
        # those are the "scene" candidates that include neighbours.
        oversize = max(1.0, m_area / max(box_area, 1))
        return float(s) * iou / (oversize ** 0.5)

    ranked = [(_score_candidate(m, s), i) for i, (m, s) in enumerate(zip(masks, scores))]
    ranked.sort(reverse=True)
    best = int(ranked[0][1])

    mask = masks[best].astype(bool)

    # Hard-clip to the dilated GDINO box: removes any sprawl into far-away
    # curtains / walls / furniture while leaving the frame edges intact.
    clip = np.zeros_like(mask)
    clip[dy1:dy2, dx1:dx2] = True
    mask = mask & clip

    # Keep only the largest connected component so disconnected blobs
    # (e.g. a chair beneath the window that SAM grouped in) are dropped.
    if mask.any():
        try:
            from scipy.ndimage import label
            labeled, n = label(mask)
            if n > 1:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                keep = int(sizes.argmax())
                mask = labeled == keep
        except ImportError:
            pass

    return mask, float(scores[best])


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_mask_rgba(image_np: np.ndarray, mask: np.ndarray, path: Path):
    """RGBA PNG: object pixels opaque, background transparent."""
    h, w   = mask.shape
    rgba   = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, :3] = image_np[mask]
    rgba[mask,  3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def _save_canvas(image_np: np.ndarray, mask: np.ndarray, path: Path,
                 box_px: list[int] | None = None, obj_type: str = "other"):
    """Object cropped (with padding), background grey, resized to OBJ_IMG_SIZE square.

    For non-window types: only the SAM-masked pixels are kept — the rest of
    the crop is grey, so the AI inpainter (or downstream texture builder)
    sees a clean object isolated on neutral background.

    For windows / doors: the SAM mask captures only the dark wooden frame —
    the BRIGHT GLASS INTERIOR (sky / blinds / view) is outside the mask and
    would get blanked to grey by the masking step.  That stripped the visual
    content downstream needed to reconstruct a complete window.  We special-
    case windows + doors to keep the FULL bounding-box crop intact (frame +
    glass + a small surround), so the inpainted texture mirrors the photo.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        Image.new("RGB", (OBJ_IMG_SIZE, OBJ_IMG_SIZE), _BG_GREY).save(path)
        return

    H, W = image_np.shape[:2]
    is_window_or_door = obj_type in ("window", "door")

    # Window/door: take the OUTER bounding box of (SAM mask ∪ detection bbox).
    # The mask and bbox may not perfectly agree — combining both ensures the
    # entire frame + glass region is captured.
    # Other types: use the SAM mask bounds only (current behaviour).
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    if is_window_or_door and box_px is not None:
        bx1, by1, bx2, by2 = [int(v) for v in box_px]
        r0 = min(r0, by1); r1 = max(r1, by2 - 1)
        c0 = min(c0, bx1); c1 = max(c1, bx2 - 1)

    pad_frac = _CANVAS_PAD_FRAME if is_window_or_door else _CANVAS_PAD
    pad_r  = int((r1 - r0) * pad_frac)
    pad_c  = int((c1 - c0) * pad_frac)
    r0     = max(0, r0 - pad_r)
    r1     = min(H - 1, r1 + pad_r)
    c0     = max(0, c0 - pad_c)
    c1     = min(W - 1, c1 + pad_c)

    crop_rgb  = image_np[r0:r1+1, c0:c1+1].copy()
    crop_mask = mask[r0:r1+1, c0:c1+1]
    if is_window_or_door:
        # Keep the FULL crop — frame + glass + small wall surround.  The
        # downstream window-plane mesh wants the actual photo content as
        # its texture, not a grey-ground frame outline.
        bg = crop_rgb
    else:
        bg = np.full_like(crop_rgb, _BG_GREY)
        bg[crop_mask] = crop_rgb[crop_mask]

    obj   = Image.fromarray(bg)
    cw, ch = obj.size
    side  = max(cw, ch, 1)
    canvas = Image.new("RGB", (side, side), _BG_GREY)
    canvas.paste(obj, ((side - cw) // 2, (side - ch) // 2))
    canvas.resize((OBJ_IMG_SIZE, OBJ_IMG_SIZE), Image.LANCZOS).save(path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(output_dir: str | Path, image_path: str,
        types: list[str] | None = None) -> Path:
    out_dir = Path(output_dir) / "wall_mounted"
    seg_dir = out_dir / "segmented"
    seg_dir.mkdir(parents=True, exist_ok=True)

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Partial re-run: keep existing entries for types we are NOT
    # re-segmenting, drop & re-do entries for the requested --types.
    results_path = out_dir / "segment_results.json"
    kept_existing: list[dict] = []
    prior_payload: dict = {}
    if types is not None and results_path.exists():
        try:
            prior_payload = json.loads(results_path.read_text())
            existing = prior_payload.get("segments", [])
            for seg in existing:
                if seg.get("type") in types:
                    # Drop the old files so they don't linger.
                    for k in ("mask_file", "canvas_file"):
                        f = seg.get(k)
                        if f:
                            try:
                                (seg_dir / f).unlink()
                            except FileNotFoundError:
                                pass
                else:
                    kept_existing.append(seg)
            print(f"[segment] partial-rerun: keeping {len(kept_existing)} prior "
                  f"segment(s); replacing types {types}")
        except Exception as e:
            print(f"[segment] could not load prior segment_results.json: {e}")

    # ── Step 1: VLM → object types ───────────────────────────────────────
    # Use the FULL VLM-detected list to build the GDINO prompt so detection
    # quality stays the same as a non-filtered run; we filter by phrase
    # after detection when --types is set.
    detected = detect_object_types(image_path)
    # Force-include common residential types regardless of VLM output —
    # see _ALWAYS_INCLUDE_TYPES note for rationale.
    for t in _ALWAYS_INCLUDE_TYPES:
        if t not in detected:
            detected.append(t)
    if detected:
        print(f"[segment] effective types (VLM ∪ always-include): {detected}")
    if types is not None:
        # If VLM missed the requested type entirely, still inject it into the
        # prompt so we have a chance of finding it.
        for t in types:
            if t not in detected:
                detected.append(t)
    if not detected:
        print("[segment] No wall-mounted object types detected — done.")
        if types is not None and kept_existing:
            payload = {
                "detected_types": prior_payload.get("detected_types", []),
                "gdino_prompt":   prior_payload.get("gdino_prompt", ""),
                "segments":       kept_existing,
            }
            with open(results_path, "w") as f:
                json.dump(payload, f, indent=2)
        return seg_dir
    types_to_segment = detected

    prompt = build_gdino_prompt(types_to_segment)
    print(f"[segment] GroundingDINO prompt: '{prompt}'")

    # ── Step 2: GroundingDINO → boxes ────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gdino, sam_predictor = _load_models(device)

    image_np          = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w      = image_np.shape[:2]

    boxes_norm, logits, phrases = _gdino_predict(gdino, image_path, prompt, device)
    print(f"[segment] GroundingDINO: {len(boxes_norm)} raw box(es) — {phrases}")

    boxes_norm, logits, phrases = _filter_large_boxes(boxes_norm, logits, phrases)
    print(f"[segment] After size filter: {len(boxes_norm)} box(es)")

    boxes_norm, logits, phrases = _filter_contained_boxes(boxes_norm, logits, phrases)
    print(f"[segment] After containment filter: {len(boxes_norm)} box(es)")

    # Phrase-level filter for partial reruns — keep only boxes whose phrase
    # maps to one of the requested types.
    if types is not None:
        keep_idx = []
        for i, ph in enumerate(phrases):
            phl = ph.lower()
            for t in types:
                gd = _TYPE_GDINO.get(t, t).lower()
                if phl in gd or gd in phl:
                    keep_idx.append(i)
                    break
        boxes_norm = boxes_norm[keep_idx]
        logits = logits[keep_idx]
        phrases = [phrases[i] for i in keep_idx]
        print(f"[segment] After --types filter ({types}): {len(boxes_norm)} box(es)")

    if len(boxes_norm) == 0:
        print("[segment] No boxes remain after filtering — done.")
        # In partial-rerun mode the prior entries for these types were already
        # deleted; rewrite the JSON so it reflects what's actually on disk.
        if types is not None and kept_existing:
            payload = {
                "detected_types": prior_payload.get("detected_types", []),
                "gdino_prompt":   prior_payload.get("gdino_prompt", ""),
                "segments":       kept_existing,
            }
            with open(results_path, "w") as f:
                json.dump(payload, f, indent=2)
        return seg_dir

    boxes_px = _boxes_to_px(boxes_norm, img_w, img_h)

    # ── Step 3: SAM → masks, save outputs ────────────────────────────────
    # Offset new indices so they don't collide with kept_existing file names.
    idx_offset = (max((s["index"] for s in kept_existing), default=-1) + 1
                  if kept_existing else 0)

    results = []
    for new_i, (box_px, phrase, logit) in enumerate(zip(boxes_px, phrases, logits)):
        idx = new_i + idx_offset
        print(f"[segment] {idx:02d} '{phrase}' (score {logit:.3f})  box {box_px}")

        # Map detected phrase back to canonical type key (needed for SAM
        # constraints — windows get a slightly larger frame-overhang).
        # Multi-phrase prompts (e.g. window: "window . large window . glass
        # window . ...") need per-sub-phrase matching, not whole-string
        # substring match — and GDINO sometimes returns a concatenation
        # of multiple matched sub-phrases ("large window floor-to-ceiling
        # window transom window"), which never appears verbatim in the
        # prompt.  Strategy:
        #   1. Try each sub-phrase ("window", "glass window", ...) as a
        #      substring of the matched phrase, longest first.
        #   2. Fallback: any-keyword match (any sub-phrase shares ≥1 word
        #      with the matched phrase).
        #   3. Last resort: keep the raw phrase as type (current behaviour).
        # GDINO frequently MERGES several prompt categories into one box with a
        # garbled phrase (e.g. a painting that sits near a window comes back as
        # "painting picture poster window picture window clerestory window").
        # Committing to a single type and verifying only that type used to drop
        # the box wholesale when the longest matching sub-phrase happened to be
        # the wrong category (here "clerestory window" beat "painting", so the
        # art was typed 'window', verified NO, and discarded with the art lost).
        # Instead, build a RANKED list of candidate types and verify each in
        # turn — keep on the first YES.
        phrase_l = phrase.lower()
        # Pass 1: per-sub-phrase substring matches → (type, sub) pairs.
        candidates: list[tuple[str, str]] = []
        for t, gdino_phrase in _TYPE_GDINO.items():
            for sub in gdino_phrase.split("."):
                sub = sub.strip().lower()
                if not sub:
                    continue
                if sub in phrase_l or phrase_l in sub:
                    candidates.append((t, sub))
        # Rank distinct candidate TYPES by their longest matching sub-phrase
        # (longest = most specific first), preserving first-seen order on ties.
        cand_types: list[str] = []
        if candidates:
            best_sub_len: dict[str, int] = {}
            for t, sub in candidates:
                best_sub_len[t] = max(best_sub_len.get(t, 0), len(sub))
            cand_types = sorted(best_sub_len, key=lambda t: best_sub_len[t],
                                reverse=True)
            # Door priority: the "window" prompt deliberately includes "sliding
            # glass door" (for glass-wall patios), so a real door matches that
            # phrase and would otherwise be tried as a window first.  When the
            # phrase literally says "door" and a door was requested, try the
            # door reading first (a glass/sliding door verifies YES as a door).
            if ("door" in phrase_l
                    and (types is None or "door" in types)
                    and "door" in best_sub_len):
                cand_types = ["door"] + [t for t in cand_types if t != "door"]
        # Pass 2: keyword overlap (any word in phrase matches any sub-phrase
        # word) — catches GDINO's concatenated multi-match outputs.
        if not cand_types:
            phrase_words = set(re.findall(r"[a-z]+", phrase_l))
            for t, gdino_phrase in _TYPE_GDINO.items():
                gdino_words = set(re.findall(r"[a-z]+", gdino_phrase.lower()))
                if phrase_words & gdino_words:
                    cand_types.append(t)
        # Pass 3: last resort — the raw phrase as its own type.
        if not cand_types:
            cand_types = [phrase.replace(" ", "_").lower()]

        # VLM context check — reject GroundingDINO false positives (e.g. a thin
        # frame-edge strip mis-detected as a 'horizontal window strip') by showing
        # the VLM the box IN the full room photo.  Try each candidate type and
        # keep on the first YES; only drop when EVERY candidate is rejected, so a
        # merged art+window box still survives as 'art'.  Lenient: _verify_segment
        # returns True on error/unparseable, so a flaky call won't drop a real
        # object.
        obj_type = cand_types[0]
        if VERIFY_SEGMENTS:
            verified = next(
                (t for t in cand_types
                 if _verify_segment(image_np, box_px, t, phrase)), None)
            if verified is None:
                print(f"          REJECTED by VLM context check "
                      f"(tried {cand_types}) — skip '{phrase}'")
                continue
            obj_type = verified

        mask, sam_score = _sam_segment_box(sam_predictor, image_np, box_px, obj_type)
        coverage = float(mask.mean())
        print(f"          SAM score {sam_score:.3f}  coverage {coverage:.2%}")

        stem        = f"segment_{idx:02d}_{obj_type}"
        mask_file   = f"{stem}_mask.png"
        canvas_file = f"{stem}_canvas.png"

        _save_mask_rgba(image_np, mask, seg_dir / mask_file)
        _save_canvas(image_np, mask, seg_dir / canvas_file,
                     box_px=box_px, obj_type=obj_type)
        print(f"          saved: {mask_file}  {canvas_file}")

        results.append({
            "index":       idx,
            "type":        obj_type,
            "phrase":      phrase,
            "gdino_score": round(float(logit), 4),
            "sam_score":   round(sam_score, 4),
            "coverage":    round(coverage, 4),
            "box_px":      box_px,
            "mask_file":   mask_file,
            "canvas_file": canvas_file,
        })

    # ── Same-type bbox dedup ──────────────────────────────────────────────
    # Two boxes of the same type with high bbox overlap are almost always the
    # same physical object detected twice (common for curtains and stacked
    # artwork).  Drop the lower-scoring one.  We also drop a box that is
    # ≥90% contained inside a same-type box (e.g. a window pane inside its
    # own frame).  Applied after SAM so we compare actual pixel boxes.
    _IOU_DEDUP = 0.50    # same-type IoU threshold → drop lower-scoring
    def _iou(b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0, 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = max((b1[2]-b1[0]) * (b1[3]-b1[1]), 1)
        a2 = max((b2[2]-b2[0]) * (b2[3]-b2[1]), 1)
        union = a1 + a2 - inter
        return inter / max(union, 1), inter / min(a1, a2)

    drop: set[int] = set()
    for i in range(len(results)):
        if i in drop:
            continue
        for j in range(len(results)):
            if j == i or j in drop:
                continue
            if results[i]["type"] != results[j]["type"]:
                continue
            iou, containment = _iou(results[i]["box_px"], results[j]["box_px"])
            if iou < _IOU_DEDUP and containment < 0.90:
                continue
            # Keep the higher-scoring detection; break ties by larger box.
            si, sj = results[i]["gdino_score"], results[j]["gdino_score"]
            ai = (results[i]["box_px"][2] - results[i]["box_px"][0]) * \
                 (results[i]["box_px"][3] - results[i]["box_px"][1])
            aj = (results[j]["box_px"][2] - results[j]["box_px"][0]) * \
                 (results[j]["box_px"][3] - results[j]["box_px"][1])
            loser = i if (si, ai) < (sj, aj) else j
            winner = j if loser == i else i
            print(f"  [dedup] dropping {results[loser]['type']} {loser} "
                  f"(iou={iou:.2f} cont={containment:.2f} vs {winner})")
            drop.add(loser)
            if loser == i:
                break
    if drop:
        # Drop dedup losers and clean up any files we already wrote for them
        # so they don't linger on disk.
        for k in drop:
            for fk in ("mask_file", "canvas_file"):
                f = results[k].get(fk)
                if f:
                    try:
                        (seg_dir / f).unlink()
                    except FileNotFoundError:
                        pass
        results = [r for k, r in enumerate(results) if k not in drop]
        # Re-index so downstream code sees contiguous indices, starting after
        # any kept_existing entries when partial-rerunning a subset of types.
        for new_local, r in enumerate(results):
            r["index"] = idx_offset + new_local

    # ── Save results JSON ─────────────────────────────────────────────────
    # Merge with kept_existing in partial-rerun mode so non-targeted entries
    # (e.g. art / curtain when --types window) survive.
    merged_segments = kept_existing + results
    if kept_existing:
        merged_types = sorted(set(prior_payload.get("detected_types", []))
                              | set(types_to_segment))
    else:
        merged_types = types_to_segment

    results_path = out_dir / "segment_results.json"
    payload = {
        "detected_types":  merged_types,
        "gdino_prompt":    prompt,
        "segments":        merged_segments,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[segment] {len(results)} new + {len(kept_existing)} kept → "
          f"{seg_dir}/  (total {len(merged_segments)})")
    print(f"[segment] Results → {results_path}")
    return seg_dir


def regen_canvas(output_dir: str | Path, image_path: str | Path,
                 types: list[str] | None = None) -> None:
    """Regenerate canvas PNGs from existing masks without re-running SAM/DINO.

    Useful when canvas padding/crop logic changes and you want to re-crop
    without paying the cost of re-running detection + segmentation.
    """
    out_dir   = Path(output_dir) / "wall_mounted"
    seg_dir   = out_dir / "segmented"
    results_p = out_dir / "segment_results.json"

    image_np = np.array(Image.open(image_path).convert("RGB"))
    data     = json.loads(results_p.read_text())
    segments = data.get("segments", [])

    for seg in segments:
        obj_type = seg.get("type", "other")
        if types is not None and obj_type not in types:
            continue
        idx        = seg["index"]
        mask_file  = seg_dir / f"segment_{idx:02d}_{obj_type}_mask.png"
        canvas_out = seg_dir / f"segment_{idx:02d}_{obj_type}_canvas.png"
        box_px     = seg.get("box_px")
        if not mask_file.exists():
            print(f"[regen_canvas] mask not found: {mask_file} — skipping")
            continue
        mask_rgba = np.array(Image.open(mask_file).convert("RGBA"))
        mask      = mask_rgba[:, :, 3] > 0
        _save_canvas(image_np, mask, canvas_out, box_px=box_px, obj_type=obj_type)
        print(f"[regen_canvas] {idx:02d} {obj_type} → {canvas_out.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Segment wall-mounted objects with Grounded-SAM."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output dir (segmented/ is created inside wall_mounted/)")
    ap.add_argument("--image", required=True,
                    help="Reference room photograph")
    ap.add_argument("--regen-canvas", action="store_true",
                    help="Regenerate canvas PNGs from existing masks (no SAM/DINO re-run)")
    ap.add_argument("--types", default=None,
                    help="Comma-separated types to (re)segment, e.g. 'window'. "
                         "When passed in normal mode, only those types are "
                         "re-detected/segmented and existing entries for other "
                         "types in segment_results.json are preserved. "
                         "When passed with --regen-canvas, restricts which "
                         "canvases are regenerated.")
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",")] if args.types else None
    if args.regen_canvas:
        regen_canvas(output_dir=args.output_dir, image_path=args.image, types=types)
    else:
        run(output_dir=args.output_dir, image_path=args.image, types=types)


if __name__ == "__main__":
    main()
