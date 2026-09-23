"""
place_missing_items.py — VLM-driven detection and placement of items missing
from the rendered scene after furniture + decoration placement.

Run as a standalone post-decoration stage (does NOT modify decorations/):

    python -m object_placement.missing_items.place_missing_items \
        --output-dir outputs/Demo/living_room9

Pipeline per missing-item type (e.g. "lamp"):

    1. _vlm_detect_missing(ref_photo, render) → list of types
       Excludes types already present in decorations/objects/ (those would
       be duplicates of items the decoration stage handled).
    2. _segment_one(image, prompt) — GroundingDINO + SAM, NMS-deduped
    3. _inpaint_optional(crop, mask) — clean source for Hunyuan (best-effort)
    4. _generate_glb(crop|inpaint, dst) — Hunyuan3D server
    5. _vlm_pick_furniture(crop, scene_render, type, furn_list) → furn idx
    6. _place_item(...) — anchor on furniture surface, silhouette-scale,
       rotate via VLM feedback hill-climb

Per-type outputs land at:

    outputs/<scene>/<type>/
        segmented/<type>_<i>_{mask,crop,canvas}.png
        inpainted/<type>_<i>_inpaint.png      (when inpainter available)
        objects/<type>_<i>.glb
        placements/<type>_<i>_render.png      (per-item render)
        placements.json                        (final placements)

The scene composite (furniture + decorations + missing items) is rendered to
    outputs/<scene>/render_with_missing_items.png
and a combined GLB is written to
    outputs/<scene>/scene_with_missing_items.glb
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from PIL import Image, ImageDraw

# Reuse helpers from existing modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Config ────────────────────────────────────────────────────────────────────

VLM_API_URL = os.environ.get(
    "VLM_API_URL",
    "http://localhost:8080/v1/chat/completions",
)

# What this stage may recover, grouped by the domain that owns placement.
#
# This used to be decoration-scale ONLY, on the reasoning that "furniture comes
# from upstream stages" and "the wall_mounted stage handles paintings".  That
# reasoning assumes the upstream stage SUCCEEDED.  When it doesn't, nothing
# else ever looks: elegant lost the foreground table that carried the room's
# main lamp (the lamp was then hosted onto the fireplace mantel), and
# pexels_12881062 lost a wall-sized painting AND its windows — which left the
# lighting stage with no sources at all and shipped a black render.
#
# A missing-object detector that cannot report the objects that actually go
# missing is not a safety net.  So: any domain, and the domain decides where
# the recovered object is routed.
_DECOR_TYPES = {
    "lamp", "desk lamp", "table lamp", "floor lamp",
    "sculpture", "figurine", "trophy",
    "candle", "candle holder",
    "vase", "decorative vase",
    "clock", "small clock",
    "framed photo", "picture frame",
    "books", "bowl", "tray", "bottle", "cushion", "throw pillow",
    "potted plant", "plant",
}

_LARGE_FURNITURE_TYPES = {
    "sofa", "couch", "loveseat", "armchair", "chair", "stool", "ottoman",
    "coffee_table", "side_table", "end_table", "dining_table",
    "desk", "console_table", "tv_stand", "sideboard", "dresser",
    "bed", "nightstand",
    "shelf", "bookshelf", "bookcase", "cabinet", "wardrobe",
    "fireplace", "carpet", "rug",
}

_WALL_TYPES = {
    "painting", "framed art", "artwork", "poster", "canvas", "wall art",
    "mirror", "window", "door", "curtain", "blinds",
    "wall lamp", "sconce", "tv", "wall clock", "shelf unit",
    "radiator", "air conditioner", "tapestry", "wall hanging",
}

# type → owning domain, for routing a recovered object to the right placer.
_TYPE_DOMAIN = {}
for _t in _DECOR_TYPES:
    _TYPE_DOMAIN[_t] = "decoration"
for _t in _LARGE_FURNITURE_TYPES:
    _TYPE_DOMAIN.setdefault(_t, "furniture")
for _t in _WALL_TYPES:
    _TYPE_DOMAIN.setdefault(_t, "wall")

_ELIGIBLE_TYPES = set(_TYPE_DOMAIN)


# ── VLM utilities ────────────────────────────────────────────────────────────

def _encode_pil(img: "Image.Image") -> str:
    import base64
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _vlm_call_json(content: list[dict],
                   max_tokens: int = 400) -> "dict | None":
    """POST a chat-completion to the local VLM server, parse JSON from the
    response.  Strips <think>…</think> and ```fences``` defensively."""
    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = requests.post(VLM_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            print(f"[vlm] no JSON in response: {raw[:200]}")
            return None
        return json.loads(m.group())
    except Exception as e:
        print(f"[vlm] call failed: {e}")
        return None


# ── Step 1: detect missing items ──────────────────────────────────────────────

_DETECT_MISSING_PROMPT = """\
You are auditing a 3D scene reconstruction.

IMAGE 1: the original reference photo of the room.
IMAGE 2: the current rendered reconstruction.

Compare IMAGE 1 to IMAGE 2 and list ANY object that is clearly visible in
IMAGE 1 but is NOT present in IMAGE 2.

Report objects of ANY size and ANY kind, including:
  - furniture: sofa, chair, stool, table, coffee table, side table, desk,
    cabinet, shelf, bed, fireplace, rug
  - wall-mounted: painting, framed art, poster, mirror, window, door,
    curtain, wall lamp, sconce, tv, air conditioner, radiator
  - decoration: lamp, vase, books, candle, plant, bowl, cushion, figurine

An earlier stage was SUPPOSED to place these, but stages fail silently, so
do not assume something was handled elsewhere — if it is in the photo and
not in the render, report it.

EXCLUDE only these, which are already placed:
  {already_placed}

The photograph is {w} x {h} pixels. For each missing object return:
  - "type": the most specific name from the lists above.
  - "box_px": a tight bounding box [x0, y0, x1, y1] in PIXEL coordinates
    measured from the TOP-LEFT corner, with x0<x1 and y0<y1.  This is used
    to segment the object directly, so it must actually enclose it.
  - "description": one short sentence — colour, shape, where it sits.
  - "confidence": 0.0-1.0, how sure you are it is missing from IMAGE 2.

Report an object ONLY if you can see it in IMAGE 1 and can point to where
it is.  Do not list objects that merely look different between the two
images, and do not invent things a room like this usually contains.

Return ONLY valid JSON, no markdown:
{{"missing": [
   {{"type": "...", "box_px": [x0, y0, x1, y1],
     "description": "...", "confidence": 0.0}}
]}}
If nothing is missing, return {{"missing": []}}.
"""


def _vlm_detect_missing(
    ref_photo: "Image.Image",
    current_render: "Image.Image",
    already_placed: "set[str]",
) -> "list[dict]":
    excl = ", ".join(sorted(already_placed)) if already_placed else "(none)"
    _w, _h = ref_photo.size
    content = [
        {"type": "text",
         "text": _DETECT_MISSING_PROMPT.format(already_placed=excl,
                                               w=_w, h=_h)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_photo)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(current_render)}"}},
    ]
    result = _vlm_call_json(content, max_tokens=1200)
    if not result:
        return []
    out = []
    for item in result.get("missing", []) or []:
        if not isinstance(item, dict):
            continue
        t = str(item.get("type", "")).strip().lower()
        if not t:
            continue
        if t not in _ELIGIBLE_TYPES:
            print(f"  [detect] dropping ineligible type '{t}'")
            continue
        if t in already_placed:
            print(f"  [detect] dropping already-placed type '{t}'")
            continue
        # The box is what lets us skip the box detector that already failed:
        # SAM is promptable by rectangle, so a usable box means we never have
        # to ask GroundingDINO to find this object again.  A missing or
        # nonsensical box is not fatal — it falls back to the GDINO path.
        box = _sanitise_box(item.get("box_px"), *ref_photo.size)
        out.append({
            "type": t,
            "domain": _TYPE_DOMAIN.get(t, "decoration"),
            "box_px": box,
            "confidence": float(item.get("confidence", 0.0) or 0.0),
            "description": str(item.get("description", "")).strip(),
        })
        print(f"  [detect] missing: {t} ({out[-1]['domain']}) "
              f"box={box or 'NONE → will fall back to GDINO'} "
              f"conf={out[-1]['confidence']:.2f}")
    return out


def _sanitise_box(b, w: int, h: int) -> "list[int] | None":
    """Clamp a VLM-supplied box to the image; reject degenerate/absurd ones.

    A model asked for pixel coordinates sometimes answers in normalised or
    0-1000 units, or transposes a pair.  SAM will segment whatever rectangle
    it is handed, so a bad box silently produces a bad object.
    """
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
    except Exception:
        return None
    if max(x0, y0, x1, y1) <= 1.0:                       # normalised 0-1
        x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
    elif max(x0, y0, x1, y1) <= 1000 and max(w, h) > 1000:   # 0-1000 grid
        x0, x1 = x0 / 1000.0 * w, x1 / 1000.0 * w
        y0, y1 = y0 / 1000.0 * h, y1 / 1000.0 * h
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    if ((x1 - x0) * (y1 - y0)) / float(w * h) > 0.9:     # ~the whole image
        return None
    return [x0, y0, x1, y1]


def _placed_categories(out_dir: Path) -> "set[str]":
    """Categories already represented in decorations/objects.  Anything
    here is considered 'covered' so the missing-item detector won't add
    duplicates."""
    cats: set[str] = set()
    objs_dir = out_dir / "decorations" / "objects"
    if objs_dir.is_dir():
        for f in objs_dir.glob("decor_*.glb"):
            # Filename pattern: decor_<idx>_<phrase>.glb
            stem = f.stem
            parts = stem.split("_", 2)
            if len(parts) >= 3:
                phrase = parts[2].replace("_", " ").lower().strip()
                cats.add(phrase)
                # Also add the LAST word as a coarse type ("orange pillow" → "pillow")
                cats.add(phrase.split()[-1])
    return cats


# ── Step 2: segment ───────────────────────────────────────────────────────────

def _load_gdino_sam(device: str = "cuda:0"):
    """Lazy import + load GroundingDINO + SAM-ViT-H.  Returns (gdino, sam_predictor)."""
    repo_root = Path(__file__).resolve().parents[2]
    gsa_root = repo_root / "Grounded-Segment-Anything"
    sys.path.insert(0, str(gsa_root / "GroundingDINO"))
    sys.path.insert(0, str(gsa_root / "segment_anything"))
    from groundingdino.util.inference import load_model as _load_gdino
    from segment_anything import SamPredictor, sam_model_registry

    gdino_cfg  = gsa_root / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_ckpt = gsa_root / "weights/groundingdino_swint_ogc.pth"
    sam_ckpt   = gsa_root / "weights/sam_vit_h_4b8939.pth"

    print(f"[seg] loading GroundingDINO + SAM-ViT-H on {device} …")
    gdino = _load_gdino(str(gdino_cfg), str(gdino_ckpt), device=device)
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt)).to(device)
    return gdino, SamPredictor(sam)


def _gdino_detect(gdino, image_path: Path, prompt: str, device: str,
                  box_thresh: float = 0.20,
                  text_thresh: float = 0.18) -> "list[tuple[list[int], float]]":
    from groundingdino.util.inference import load_image, predict
    pil = Image.open(image_path).convert("RGB")
    W, H = pil.size
    _, image = load_image(str(image_path))
    boxes, logits, _ = predict(
        model=gdino, image=image, caption=prompt,
        box_threshold=box_thresh, text_threshold=text_thresh, device=device,
    )
    if boxes is None or len(boxes) == 0:
        return []
    bb = boxes.cpu().numpy()
    sc = logits.cpu().numpy()
    out = []
    for (cx, cy, w, h), score in zip(bb, sc):
        x1 = max(0, int((cx - w / 2) * W));  y1 = max(0, int((cy - h / 2) * H))
        x2 = min(W, int((cx + w / 2) * W));  y2 = min(H, int((cy + h / 2) * H))
        out.append(([x1, y1, x2, y2], float(score)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _nms_dedup(detections: "list[tuple[list[int], float]]",
               iou_thresh: float = 0.40) -> "list[tuple[list[int], float]]":
    if not detections:
        return []
    kept: list = []
    for box, score in detections:
        x1, y1, x2, y2 = box
        bx_area = max((x2 - x1) * (y2 - y1), 1)
        is_dup = False
        for kbox, _ in kept:
            kx1, ky1, kx2, ky2 = kbox
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            ka = max((kx2 - kx1) * (ky2 - ky1), 1)
            iou = inter / (bx_area + ka - inter)
            if iou >= iou_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append((box, score))
    return kept


def _sam_mask(sam_predictor, image_np: np.ndarray, bbox_xyxy: list[int]):
    sam_predictor.set_image(image_np)
    masks, _, _ = sam_predictor.predict(
        box=np.asarray(bbox_xyxy, dtype=np.float32),
        multimask_output=False,
    )
    return masks[0].astype(bool)


def _save_seg_outputs(out_dir: Path, type_safe: str, idx: int,
                      image_pil: "Image.Image", mask: np.ndarray,
                      bbox_xyxy: list[int],
                      crop_pad: int = 12) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image_pil)
    # mask
    mask_file = f"{type_safe}_{idx:02d}_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out_dir / mask_file)
    # crop on neutral grey canvas
    grey = np.full_like(arr, 185)
    composite = np.where(mask[..., None], arr, grey)
    x1, y1, x2, y2 = bbox_xyxy
    cx1 = max(0, x1 - crop_pad); cy1 = max(0, y1 - crop_pad)
    cx2 = min(arr.shape[1], x2 + crop_pad); cy2 = min(arr.shape[0], y2 + crop_pad)
    crop_file = f"{type_safe}_{idx:02d}_crop.png"
    Image.fromarray(composite[cy1:cy2, cx1:cx2]).save(out_dir / crop_file)
    # canvas (full-image with mask, useful for VLM context)
    canvas_file = f"{type_safe}_{idx:02d}_canvas.png"
    Image.fromarray(composite).save(out_dir / canvas_file)
    return {
        "mask_file": mask_file,
        "crop_file": crop_file,
        "canvas_file": canvas_file,
        "bbox_px": list(bbox_xyxy),
    }


# ── Step 3: optional inpaint (best-effort; skips silently if unavailable) ────

def _inpaint_optional(crop_path: Path, mask_path: Path,
                      dst: Path) -> "Path | None":
    """Try to call the existing decoration inpainter if available.  This
    is a best-effort hook; many setups don't have it wired and that's OK
    — Hunyuan can consume the raw crop directly."""
    try:
        from object_placement.decorations.inpaint_decorations import inpaint_one
    except Exception:
        return None
    try:
        inpaint_one(crop_path, mask_path, dst)
        return dst if dst.exists() else None
    except Exception as e:
        print(f"  [inpaint] best-effort inpaint failed ({e}) — using raw crop")
        return None


# ── Step 4: Hunyuan3D GLB generation ──────────────────────────────────────────

def _generate_glb(image_path: Path, glb_path: Path,
                  texture: bool = True) -> bool:
    from object_placement.furniture.object_generation import (
        _generate_hunyuan, _hunyuan_available,
    )
    if not _hunyuan_available():
        print("  [hunyuan] server not reachable — skipping generation")
        return False
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    return _generate_hunyuan(image_path, glb_path, texture=texture)


# ── Step 5: pick which furniture this missing item belongs on ────────────────

_PICK_FURNITURE_PROMPT = """\
You are placing a missing decoration in a 3D scene.

IMAGE 1: a crop showing the missing item — a "{type_}".
IMAGE 2: the current scene render with furniture LABELED by index.

The missing item must be placed on (or next to) ONE piece of furniture
visible in IMAGE 2.  Look at the photo of the item and the scene, decide
which furniture index is the most plausible support.

Available furniture (index → type):
{furn_list}

If the item should sit on the FLOOR (e.g. floor lamp), return -1.
Otherwise return the index of the furniture it sits on.

Return ONLY valid JSON, no markdown:
{{"furniture_index": <int>, "placement_kind": "on_surface"|"on_floor",
  "reasoning": "..."}}
"""


_LAMP_LIKE_TYPES = {"lamp", "desk lamp", "table lamp"}
_FLOOR_LAMP_TYPES = {"floor lamp"}
_SEATING_TYPES = {"sofa", "couch", "loveseat", "armchair", "chair"}
_NON_SURFACE_TYPES = {"plant", "tree", "sculpture", "statue",
                      "lamp", "floor lamp", "floor_lamp"}


def _geo_pick_furniture(item_bbox_px: "list[int]",
                        item_type: str,
                        furn_list: "list[dict]",
                        out_dir: Path,
                        camera: dict,
                        img_w: int, img_h: int) -> "list[dict]":
    """Project each furniture's TOP CENTRE into image space and rank by
    pixel distance to the item's mask centre.  Mirrors the `_proj_dist`
    + type-filter pattern from place_decorations._resolve_furniture so
    desk lamps end up on tables (even *inner* tables that don't X-overlap
    the lamp's bbox) instead of nearby sofas / chairs.

    Returns up to top 5 candidates as
        [{index, type, d2, kind_hint}, ...]
    sorted by d² ascending.
    """
    if not furn_list:
        return []

    # Project a 3D world point → image pixel using the same convention as
    # the rest of the pipeline (right = fwd × up, up_c = right × fwd).
    cam_pos = np.array(camera["position_m"], dtype=np.float64)
    look_at = np.array(camera["look_at_m"],  dtype=np.float64)
    up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    fwd = look_at - cam_pos
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    right = np.cross(fwd, up_world)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up_c = np.cross(right, fwd)
    fx = img_w / (2.0 * np.tan(np.radians(float(camera["hfov_deg"]) / 2.0)))
    cx_p, cy_p = img_w / 2.0, img_h / 2.0

    def _project_top(f: dict) -> "tuple[float, float, float] | None":
        try:
            p = np.array(f["position_m"], dtype=np.float64)
            top_y = float(p[1]) + float(f.get("size_m", {}).get("height_m", 0.5))
            wp = np.array([p[0], top_y, p[2]], dtype=np.float64) - cam_pos
            zc = float(np.dot(wp, fwd))
            if zc <= 0.01:
                return None
            xc = float(np.dot(wp, right)) / zc
            yc = float(np.dot(wp, up_c)) / zc
            return cx_p + xc * fx, cy_p - yc * fx, zc
        except Exception:
            return None

    lx1, ly1, lx2, ly2 = item_bbox_px
    mx = (lx1 + lx2) / 2.0
    my = (ly1 + ly2) / 2.0

    # Type filter: a "lamp"-class item should land on a TABLE/DESK surface,
    # not on seating, not on a plant, not on the floor.  A "floor lamp"
    # gets the inverse — it stands on the floor and the nearest furniture
    # is a spatial anchor only.
    item_lower = item_type.lower()
    is_table_lamp = any(kw in item_lower for kw in _LAMP_LIKE_TYPES) and not any(
        kw in item_lower for kw in _FLOOR_LAMP_TYPES)
    is_floor_lamp = any(kw in item_lower for kw in _FLOOR_LAMP_TYPES)

    def _is_valid(f: dict) -> bool:
        idx = int(f.get("index", -1))
        if idx < 0:
            return False
        ftype = f.get("type", "").lower()
        if ftype in _NON_SURFACE_TYPES:
            return False
        if is_table_lamp and ftype in _SEATING_TYPES:
            return False           # desk lamp must not go on sofa/chair
        if "position_m" not in f:
            return False
        return True

    cands: list[dict] = []
    for f in furn_list:
        if not _is_valid(f):
            continue
        proj = _project_top(f)
        if proj is None:
            continue
        px, py, _zc = proj
        d2 = (px - mx) ** 2 + (py - my) ** 2
        # `kind_hint` defaults to on_surface for desk lamps; floor lamps
        # always land on_floor regardless of which table they're nearest to.
        kind = "on_floor" if is_floor_lamp else "on_surface"
        cands.append({
            "index": int(f.get("index", -1)),
            "type":  f.get("type", ""),
            "d2":    float(d2),
            "proj_top_px": (float(px), float(py)),
            "kind_hint": kind,
        })
    cands.sort(key=lambda c: c["d2"])
    return cands[:5]


def _vlm_pick_furniture(item_crop: "Image.Image",
                        scene_render: "Image.Image",
                        type_: str,
                        furn_list: "list[dict]") -> "dict":
    listing = "\n".join(f"  [{f['index']}] {f.get('type','furniture')}"
                        for f in furn_list)
    content = [
        {"type": "text",
         "text": _PICK_FURNITURE_PROMPT.format(type_=type_, furn_list=listing)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(item_crop)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(scene_render)}"}},
    ]
    res = _vlm_call_json(content, max_tokens=200)
    if not res:
        return {"furniture_index": -1, "placement_kind": "on_floor",
                "reasoning": "VLM call failed — defaulting to floor"}
    return res


def _annotate_scene_with_indices(scene_render: "Image.Image",
                                 furn_list: "list[dict]",
                                 camera: dict) -> "Image.Image":
    """Draw an index label near each furniture piece's projected centroid
    so the picker VLM sees what each index refers to."""
    base = scene_render.convert("RGB").copy()
    draw = ImageDraw.Draw(base)
    cam_pos = np.array(camera["position_m"], dtype=np.float64)
    look_at = np.array(camera["look_at_m"], dtype=np.float64)
    up = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    fwd = (look_at - cam_pos)
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    right = np.cross(fwd, up)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up_c = np.cross(right, fwd)
    W, H = base.size
    fx = W / (2.0 * np.tan(np.radians(float(camera["hfov_deg"]) / 2.0)))
    cx_p, cy_p = W / 2.0, H / 2.0
    for f in furn_list:
        try:
            p = np.array(f["position_m"], dtype=np.float64)
            sm = f.get("size_m", {})
            top_y = float(p[1]) + float(sm.get("height_m", 0.5))
            target = np.array([p[0], top_y, p[2]], dtype=np.float64) - cam_pos
            zc = float(np.dot(target, fwd))
            if zc <= 0.01:
                continue
            xc = float(np.dot(target, right)) / zc
            yc = float(np.dot(target, up_c)) / zc
            px = cx_p + xc * fx
            py = cy_p - yc * fx
            draw.rectangle([px - 14, py - 10, px + 28, py + 12],
                           fill=(255, 80, 0))
            draw.text((px - 10, py - 8), str(f.get("index", "?")),
                      fill=(255, 255, 255))
        except Exception:
            pass
    return base


# ── Step 6: place item ──────────────────────────────────────────────────────

def _glb_native_extents(glb_path: Path) -> "tuple[float, float, float]":
    import trimesh
    m = trimesh.load(str(glb_path), force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump())
    b = m.bounds
    return (float(b[1, 0] - b[0, 0]),
            float(b[1, 1] - b[0, 1]),
            float(b[1, 2] - b[0, 2]))


def _force_upright_rotation(glb_path: Path) -> "list[list[float]]":
    """Pick a 90°-step rotation that puts the GLB's LONGEST axis along world Y.

    Hunyuan3D often emits tall objects (lamps, vases, candles) lying on their
    side — the mesh's natural Y-extent is the SHORTEST dimension instead of the
    tallest.  `_dec_level_base` only handles small (≤45°) tilts, so we need an
    explicit 90° flip first.  Returns a 3×3 rotation matrix (as nested lists)
    that, when right-multiplied onto the centred mesh, makes the longest axis
    point along +Y.
    """
    nw, nh, nd = _glb_native_extents(glb_path)
    extents = (nw, nh, nd)
    longest = int(np.argmax(extents))
    if longest == 1:
        return np.eye(3).tolist()                  # already vertical
    if longest == 0:                               # X is longest → rotate so X→Y
        # Rotate around Z by +90°: (x,y,z) → (-y, x, z); but we want X to become Y.
        # Use rotation that maps +X to +Y: rotation about Z by -90° gives (y,-x,z).
        # We want (y, x, z) → matrix that sends e_x → e_y, e_y → -e_x, e_z → e_z
        # which is rotation about Z by +90°.
        return [[0.0, -1.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0]]
    # longest == 2: Z is longest → rotate so Z→Y (rotate about X by -90°)
    # sends e_z → e_y, e_y → -e_z, e_x → e_x
    return [[1.0, 0.0,  0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0,  0.0]]


def _robust_base_offset(glb_path: Path,
                        rotation_3x3: "list[list[float]]",
                        scale: float,
                        percentile: float = 1.0) -> float:
    """How far ABOVE the absolute mesh-min Y the *true base* of the mesh is,
    once the upright rotation + uniform scale have been applied.

    Returns base_y_robust − base_y_absolute (always ≥ 0).  The renderer uses
    `verts[:, 1].min()` to anchor pos[1], which gets pulled down by stray
    Hunyuan spike verts → the real base ends up FLOATING above pos[1] by
    this offset.  Adding this offset to pos[1] compensates: the spike still
    pokes through the table by an imperceptible amount but the visible base
    sits flush.
    """
    try:
        import trimesh
        m = trimesh.load(str(glb_path), force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(m.dump())
        # Centre + scale + rotate, mirroring the renderer.
        bnd = m.bounds
        centre = (bnd[0] + bnd[1]) / 2.0
        verts = (m.vertices.astype(np.float64) - centre) * float(scale)
        R = np.asarray(rotation_3x3, dtype=np.float64)
        verts = verts @ R.T
        ys = verts[:, 1]
        if ys.size < 4:
            return 0.0
        y_min  = float(ys.min())
        y_robust = float(np.percentile(ys, percentile))
        return max(0.0, y_robust - y_min)
    except Exception:
        return 0.0


def _floor_anchor_near_furniture(furn: dict,
                                 camera: dict) -> "tuple[float, float]":
    """Pick a floor (X, Z) hint NEAR a furniture piece — used for on_floor
    items like floor lamps that the VLM said belong next to a chair/sofa.
    Offsets the spot by half the furniture's depth + 25 cm in the direction
    AWAY from the camera (so the lamp tucks behind the seating, matching
    typical "lamp behind chair" arrangements)."""
    fp = list(furn.get("position_m", [0.0, 0.0, 0.0]))
    sm = furn.get("size_m", {})
    half_d = float(sm.get("depth_m", 0.4)) / 2.0
    cam_pos = np.array(camera["position_m"], dtype=np.float64)
    look_at = np.array(camera["look_at_m"], dtype=np.float64)
    fwd = look_at - cam_pos
    fwd[1] = 0.0
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        return float(fp[0]), float(fp[2])
    fwd /= n     # unit vector pointing away from camera in XZ
    offset = half_d + 0.25
    return float(fp[0] + fwd[0] * offset), float(fp[2] + fwd[2] * offset)


def _silhouette_anchor(bbox_px: list[int],
                       camera: dict,
                       img_w: int, img_h: int) -> "tuple[float, float] | None":
    """Project the bbox bottom-centre to the floor plane Y=0.  Used as a
    fallback world-space hint when the VLM's furniture pick fails."""
    try:
        cam_pos = np.array(camera["position_m"], dtype=np.float64)
        look_at = np.array(camera["look_at_m"], dtype=np.float64)
        up = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
        fwd = look_at - cam_pos
        fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
        right = np.cross(fwd, up); right /= max(float(np.linalg.norm(right)), 1e-9)
        up_c = np.cross(right, fwd)
        fx = img_w / (2.0 * np.tan(np.radians(float(camera["hfov_deg"]) / 2.0)))
        px = (bbox_px[0] + bbox_px[2]) / 2.0
        py = float(bbox_px[3])
        x_ndc = (px - img_w / 2.0) / fx
        y_ndc = -(py - img_h / 2.0) / fx
        ray = fwd + x_ndc * right + y_ndc * up_c
        ray /= max(float(np.linalg.norm(ray)), 1e-9)
        if ray[1] >= -1e-6:
            return None
        t = -cam_pos[1] / ray[1]
        wp = cam_pos + t * ray
        return float(wp[0]), float(wp[2])
    except Exception:
        return None


def _silhouette_target_height(bbox_px: list[int],
                              furn: "dict | None",
                              camera: dict,
                              img_h: int) -> float:
    """Estimate real-world height of the missing item from its bbox height
    in the photo + the supporting furniture's known top-Y for distance.

    Returns a height in metres.  Coarse but better than a blind 0.3 m.
    """
    if furn is None or "position_m" not in furn:
        return 0.5
    try:
        sm = furn.get("size_m", {})
        top_y = float(furn["position_m"][1]) + float(sm.get("height_m", 0.5))
        cam_pos = np.array(camera["position_m"], dtype=np.float64)
        # Distance from camera to the support's centroid in ground plane.
        dx = float(furn["position_m"][0]) - float(cam_pos[0])
        dz = float(furn["position_m"][2]) - float(cam_pos[2])
        ground_d = float(np.sqrt(dx * dx + dz * dz))
        # Slant distance to the support top
        slant = float(np.sqrt(ground_d * ground_d + (top_y - float(cam_pos[1])) ** 2))
        bbox_h_px = float(bbox_px[3] - bbox_px[1])
        fy = img_h / (2.0 * np.tan(np.radians(float(camera["vfov_deg"]) / 2.0)))
        # h_world ≈ h_px * d_slant / fy
        h = bbox_h_px * slant / max(fy, 1.0)
        return float(np.clip(h, 0.05, 2.5))
    except Exception:
        return 0.5


def _place_one(item_entry: dict,
               glb_path: Path,
               furn: "dict | None",
               bbox_px: list[int],
               camera: dict,
               img_w: int, img_h: int,
               kind: str = "on_surface",
               room_w: float = 5.0,
               room_d: float = 5.0) -> dict:
    """Compute world position + scale + upright rotation for one item.

    Furniture-anchored, NOT silhouette-anchored: the bbox bottom for an
    occluded floor lamp back-projects nearly horizontally and lands far
    outside the room.  We let the VLM's furniture pick decide the (X,Z),
    then snap to the supporting surface (or to floor + behind for on_floor).
    Everything is clamped to room bounds as a last-line safety.
    """
    nat_w, nat_h, nat_d = _glb_native_extents(glb_path)
    # Use the LONGEST native extent as the height proxy — Hunyuan often emits
    # tall objects with their long axis on X or Z, so blindly scaling to the
    # mesh's native height (a tiny dimension) made the lamp hilariously short.
    longest_native = max(nat_w, nat_h, nat_d)
    target_h = _silhouette_target_height(bbox_px, furn, camera, img_h)
    scale = target_h / max(longest_native, 1e-6)
    # After applying _force_upright_rotation, longest axis IS Y, so size_m
    # height = longest_native * scale by construction.
    sm = {
        "width_m":  float(min(nat_w, nat_d) * scale),
        "height_m": float(longest_native * scale),
        "depth_m":  float(min(nat_w, nat_d) * scale),
    }

    upright_R = _force_upright_rotation(glb_path)

    if furn is not None and "position_m" in furn:
        p = list(furn["position_m"])
        if kind == "on_floor":
            # Floor lamp NEAR the picked furniture — tucked behind it
            # (further from camera) by half-furniture-depth + 25 cm.
            ax, az = _floor_anchor_near_furniture(furn, camera)
            pos = [ax, 0.0, az]
        else:
            # On the furniture's top surface, centred over its footprint.
            f_top = float(p[1]) + float(furn.get("size_m", {}).get("height_m", 0.5))
            pos = [float(p[0]), f_top, float(p[2])]
    else:
        # No furniture pick — fall back to silhouette ground anchor, but
        # clamped to room bounds so we never produce negative coords.
        anchor_xz = _silhouette_anchor(bbox_px, camera, img_w, img_h)
        if anchor_xz is None:
            anchor_xz = (camera["position_m"][0], camera["position_m"][2] - 0.6)
        pos = [float(anchor_xz[0]), 0.0, float(anchor_xz[1])]

    # Sink Y by the spike-vs-true-base offset so the visible mesh base
    # touches the table top.  Without this the renderer's
    # `base_y = verts[:, 1].min()` picks an outlier spike vertex from
    # Hunyuan and the lamp ends up floating above the surface.
    base_lift = _robust_base_offset(glb_path, upright_R, scale)
    if base_lift > 1e-4:
        pos[1] = max(0.0, pos[1] - base_lift)

    # Hard clamp to room — keeps the item inside the floor plan even when
    # all anchor heuristics fail (occlusion, missing furniture, etc.).
    margin = 0.10
    pos[0] = float(np.clip(pos[0], margin, max(margin, room_w - margin)))
    pos[2] = float(np.clip(pos[2], margin, max(margin, room_d - margin)))

    item_entry.update({
        "glb_path":        str(glb_path),
        "position_m":      pos,
        "rotation_3x3":    upright_R,
        "scale":           [scale, scale, scale],
        "size_m":          sm,
        "furniture_index": furn.get("index", -1) if furn else -1,
        "furniture_type":  furn.get("type", "floor") if furn else "floor",
        "placement_type":  "on_surface",   # ← triggers _dec_level_base
        "bbox_px":         list(bbox_px),
        "kind":            kind,
    })
    return item_entry


# ── Step 7: rotation feedback (hill-climb) ───────────────────────────────────
#
# Reuses the same idea as the furniture stage: render at current yaw,
# render at +Δ and -Δ, ask the VLM which is closest to the photo, advance
# in that direction, stop when both neighbours look worse.

_ROT_COMPARE_PROMPT = """\
You are choosing the best 3D pose for a "{type_}" placed in a scene.

IMAGE 1: the reference photo of the room.
IMAGE 2 (LEFT) and IMAGE 3 (RIGHT): two candidate renders of the same
scene with the "{type_}" rotated to slightly different yaw angles.

Which candidate's "{type_}" matches the orientation of the one in the
reference photo more closely?

Return ONLY valid JSON: {{"better": "left" | "right" | "tie",
                          "reasoning": "..."}}
"""


def _vlm_pick_rotation(ref_photo: "Image.Image",
                       render_left: "Image.Image",
                       render_right: "Image.Image",
                       type_: str) -> str:
    content = [
        {"type": "text",
         "text": _ROT_COMPARE_PROMPT.format(type_=type_)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_photo)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_left)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_right)}"}},
    ]
    res = _vlm_call_json(content, max_tokens=200)
    if not res:
        return "tie"
    val = str(res.get("better", "tie")).strip().lower()
    return val if val in {"left", "right", "tie"} else "tie"


def _yaw_rotation(deg: float) -> "list[list[float]]":
    r = np.radians(deg)
    c, s = float(np.cos(r)), float(np.sin(r))
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


# ── Rendering helpers ────────────────────────────────────────────────────────

def _render_scene(out_dir: Path, placements: "list[dict]",
                  dst: Path) -> "Image.Image | None":
    """Render furniture + decorations + missing items composited onto the
    base furniture render.  Reuses the decoration renderer."""
    try:
        from object_placement.decorations.place_decorations import _do_render
        _do_render(out_dir, placements, dst)
        if dst.exists():
            return Image.open(dst).convert("RGB")
    except Exception as e:
        print(f"  [render] failed: {e}")
    return None


def _load_camera(out_dir: Path) -> "dict | None":
    for cand in [out_dir / "camera_vggt.json", out_dir / "camera.json"]:
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def _load_scene_placements(out_dir: Path) -> "list[dict]":
    """Read DECORATION placements only — furniture is already rendered
    into furniture/render_furniture_placed.png which `_do_render` uses
    as the base image.  Including furniture entries here would make the
    composite renderer re-draw them as decorations and stomp on the
    base pixels (carpet looks scrubbed, floor plant projects above the
    chair, etc.).  We append missing-item entries to this list, so the
    composite render = (furniture base PNG) + (decorations) + (missing).
    """
    placements: list[dict] = []
    dp = out_dir / "decorations" / "placements" / "decoration_placements.json"
    if dp.exists():
        placements.extend(json.loads(dp.read_text()))
    return placements


def _load_furniture_only(out_dir: Path) -> "list[dict]":
    fp = out_dir / "furniture" / "furniture_placements.json"
    return json.loads(fp.read_text()) if fp.exists() else []


# ── Main orchestration ──────────────────────────────────────────────────────

def run(output_dir: "str | Path",
        missing_types: "Iterable[str] | None" = None,
        device: str = "cuda:0",
        no_texture: bool = False,
        crop_pad: int = 24,
        rot_steps: int = 4,
        rot_step_deg: float = 30.0,
        skip_detect: bool = False,
        animate: bool = False,
        animate_frame_ms: int = 700,
        animate_gif_width: int = 1500) -> Path:
    """Detect → segment → generate → place missing decoration items.

    missing_types : optional explicit list of types to bypass VLM detection
                    (useful for testing one type at a time, or when the
                    detector already ran and you want to re-place).
    """
    out_dir = Path(output_dir).resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(out_dir)

    # 0. Locate the reference photo + load camera.
    cam = _load_camera(out_dir)
    if cam is None:
        raise FileNotFoundError(f"camera.json/camera_vggt.json missing in {out_dir}")
    ref_photo_path = _resolve_ref_photo(out_dir, cam)
    if ref_photo_path is None:
        raise FileNotFoundError(
            f"Could not locate reference photo for {out_dir}")
    ref_photo = Image.open(ref_photo_path).convert("RGB")
    img_w, img_h = ref_photo.size
    print(f"[missing] reference photo: {ref_photo_path} ({img_w}×{img_h})")

    # 1. Render the current scene (furniture + decorations) → "before" image.
    placements = _load_scene_placements(out_dir)
    before_render_path = out_dir / "render_with_decorations_for_missing.png"
    before_img = _render_scene(out_dir, placements, before_render_path)
    if before_img is None:
        # Fall back to whatever final render exists.
        for cand in [out_dir / "decorations" / "placements" / "render_decorations_placed.png",
                     out_dir / "furniture" / "render_furniture_placed.png"]:
            if cand.exists():
                before_img = Image.open(cand).convert("RGB")
                break
    if before_img is None:
        raise FileNotFoundError("No rendered scene found to compare against")

    # 2. Detect missing items.
    placed_cats = _placed_categories(out_dir)
    print(f"[missing] already-placed categories: {sorted(placed_cats) or '(none)'}")
    if missing_types is not None:
        missing = [{"type": str(t).lower().strip(), "description": ""}
                   for t in missing_types]
        print(f"[missing] explicit types: {[m['type'] for m in missing]}")
    elif skip_detect:
        missing = []
        print("[missing] --skip-detect set — no detection run")
    else:
        print("[missing] running VLM detect …")
        missing = _vlm_detect_missing(ref_photo, before_img, placed_cats)
        print(f"[missing] detected {len(missing)} type(s): "
              f"{[m['type'] for m in missing]}")

    if not missing:
        print("[missing] nothing to place. Done.")
        return out_dir

    # 3. Lazy-load segmentation models only when we have work.
    gdino, sam_pred = _load_gdino_sam(device=device)
    image_np = np.asarray(ref_photo)
    furn_list = _load_furniture_only(out_dir)
    scene_annot = _annotate_scene_with_indices(before_img, furn_list, cam)

    all_results: list[dict] = []
    # Per-item ordered list of (label, png_path) for GIF assembly.
    anim_frames_by_type: "dict[str, list[tuple[str, Path]]]" = {}
    for spec in missing:
        type_ = spec["type"]
        type_safe = re.sub(r"\W+", "_", type_).strip("_")
        print(f"\n[missing] === processing '{type_}' ===")
        type_dir = out_dir / type_safe
        seg_dir = type_dir / "segmented"
        ipt_dir = type_dir / "inpainted"
        obj_dir = type_dir / "objects"
        place_dir = type_dir / "placements"
        for d in (seg_dir, ipt_dir, obj_dir, place_dir):
            d.mkdir(parents=True, exist_ok=True)
        anim_frames_by_type[type_safe] = []
        # Frame 0 = baseline scene (no missing item yet).
        if animate:
            _baseline = place_dir / f"{type_safe}_anim00_baseline.png"
            try:
                from shutil import copyfile as _cf
                _cf(before_render_path, _baseline)
                anim_frames_by_type[type_safe].append(("baseline", _baseline))
            except Exception:
                pass

        # --- 3a. Segment ---
        # PREFER THE VLM'S OWN BOX.  The whole reason an object reaches this
        # stage is that the box detector failed to find it; re-running that
        # same detector with a reworded prompt mostly fails the same way.
        # SAM is promptable by rectangle, so the VLM's box goes straight in
        # and the detector is skipped entirely.
        bbox = spec.get("box_px")
        if bbox:
            print(f"  [seg] using VLM box {bbox} for '{type_}' "
                  f"(conf={spec.get('confidence', 0.0):.2f}) — skipping GDINO")
        else:
            # No usable box (model gave none, or it failed sanitisation) —
            # fall back to the detector rather than giving up on the object.
            prompt = " . ".join([type_,
                                 f"{type_} on table",
                                 f"small {type_}"])
            if type_ == "lamp":
                prompt = "lamp . desk lamp . table lamp . floor lamp"
            dets = _gdino_detect(gdino, ref_photo_path, prompt, device)
            dets = _nms_dedup(dets, iou_thresh=0.40)
            if not dets:
                print(f"  [seg] no VLM box and no GDINO detection for "
                      f"'{type_}' — skipping")
                continue
            bbox, score = dets[0]
            print(f"  [seg] GDINO fallback bbox={bbox} score={score:.3f}")
        mask = _sam_mask(sam_pred, image_np, bbox)
        seg_meta = _save_seg_outputs(seg_dir, type_safe, 0,
                                     ref_photo, mask, bbox,
                                     crop_pad=crop_pad)
        crop_path = seg_dir / seg_meta["crop_file"]
        mask_path = seg_dir / seg_meta["mask_file"]

        # --- 3b. Inpaint (best-effort) ---
        ipt_path = _inpaint_optional(crop_path, mask_path,
                                     ipt_dir / f"{type_safe}_00_inpaint.png")
        glb_src = ipt_path if ipt_path is not None else crop_path

        # --- 3c. Generate GLB ---
        glb_path = obj_dir / f"{type_safe}_00.glb"
        if not glb_path.exists():
            ok = _generate_glb(glb_src, glb_path, texture=not no_texture)
            if not ok:
                print(f"  [hunyuan] generation failed for '{type_}' — skipping")
                continue
        else:
            print(f"  [hunyuan] reusing existing {glb_path.name}")

        # --- 3d. Pick furniture ---
        # Geometric pre-pass: project each furniture's TOP-CENTRE into the
        # photo and rank by pixel distance to the lamp's mask centre.
        # This handles inner / occluded tables that don't share an image-
        # bbox X-range with the lamp (which the previous bbox-overlap
        # version missed).  Type-filter excludes seating / plants for
        # lamp-class items so they don't land on a sofa.
        geo_cands = _geo_pick_furniture(bbox, type_, furn_list, out_dir,
                                        cam, img_w, img_h)
        if geo_cands:
            print("  [pick] geometric candidates (top 3, by projected-top d²):")
            for c in geo_cands[:3]:
                px, py = c["proj_top_px"]
                print(f"    [{c['index']}] {c['type']}  d²={c['d2']:.0f}  "
                      f"proj_top=({px:.0f},{py:.0f})  hint={c['kind_hint']}")
        crop_img = Image.open(crop_path).convert("RGB")
        pick = _vlm_pick_furniture(crop_img, scene_annot, type_, furn_list)
        vlm_fi = int(pick.get("furniture_index", -1))
        vlm_kind = str(pick.get("placement_kind", "on_surface")).lower().strip()
        if geo_cands:
            top = geo_cands[0]
            # Trust the geometric pick by default — when the item is a
            # table-lamp class object (which the type filter has already
            # restricted to surface furniture), the projected-distance
            # geometry is more reliable than the VLM's text-only call.
            # We still let the VLM override IF (a) it agrees on type
            # (e.g. table → table) AND (b) its choice is also in the geo
            # top-2 (i.e. close in projected distance, just a tiebreak).
            top2_idx = {c["index"] for c in geo_cands[:2]}
            if vlm_fi in top2_idx and vlm_fi != top["index"]:
                fi, kind = vlm_fi, top["kind_hint"]
                print(f"  [pick] VLM tiebreak within top-2: choosing [{fi}] over geo [{top['index']}]")
            else:
                fi, kind = top["index"], top["kind_hint"]
                if vlm_fi != fi or vlm_kind != kind:
                    print(f"  [pick] geometric override: VLM said "
                          f"[{vlm_fi}] {vlm_kind}, geo says [{fi}] {kind}")
        else:
            fi, kind = vlm_fi, vlm_kind
        # Keep `furn` even for on_floor — `_place_one` uses it as a *spatial
        # anchor* (where to put the lamp on the floor) rather than a
        # support surface.
        furn = next((f for f in furn_list if int(f.get("index", -2)) == fi), None)
        print(f"  [pick] using furniture_index={fi} kind={kind}  "
              f"vlm_reasoning: {str(pick.get('reasoning',''))[:140]}")

        # --- 3e. Place ---
        entry = {
            "type":        type_,
            "phrase":      type_,
            "description": spec.get("description", ""),
            "is_missing_item": True,
        }
        # Pull room dims from the same source the furniture stage uses so the
        # hard clamp respects floorplan_analysis.json (which we already fix
        # to match the actual mesh in the wall_alignment stage).
        try:
            from object_placement.furniture.place_furniture_vggt import _get_room_dims
            _ref_glb = (out_dir / "furniture" / "scene_with_furniture.glb")
            _rw, _rd = _get_room_dims(_ref_glb, cam)
        except Exception:
            _rw, _rd = 5.0, 5.0
        entry = _place_one(entry, glb_path, furn, bbox, cam, img_w, img_h,
                           kind=kind, room_w=_rw, room_d=_rd)

        # --- 3f. Rotation feedback (hill-climb pairwise) ---
        # Best-effort; if any render fails, keep yaw=0.
        try:
            current_yaw = 0.0
            placements_with_item = list(placements) + [entry]
            base_path = place_dir / f"{type_safe}_00_base.png"
            base_render = _render_scene(out_dir, placements_with_item, base_path)
            if animate and base_path.exists():
                anim_frames_by_type[type_safe].append(
                    ("placed_initial", base_path))
            if base_render is not None and rot_steps > 0:
                direction = +1
                last_better = None
                for step in range(rot_steps):
                    cand_yaw = current_yaw + direction * rot_step_deg
                    # Render left = current_yaw, right = cand_yaw
                    entry["rotation_3x3"] = _yaw_rotation(current_yaw)
                    left_path = place_dir / f"{type_safe}_00_yaw{int(round(current_yaw)):+d}.png"
                    left_render = _render_scene(out_dir, placements_with_item, left_path)
                    entry["rotation_3x3"] = _yaw_rotation(cand_yaw)
                    right_path = place_dir / f"{type_safe}_00_yaw{int(round(cand_yaw)):+d}.png"
                    right_render = _render_scene(out_dir, placements_with_item, right_path)
                    if animate:
                        if left_path.exists():
                            anim_frames_by_type[type_safe].append(
                                (f"rot_step{step}_cur{int(current_yaw):+d}", left_path))
                        if right_path.exists():
                            anim_frames_by_type[type_safe].append(
                                (f"rot_step{step}_cand{int(cand_yaw):+d}", right_path))
                    if left_render is None or right_render is None:
                        break
                    pick_rot = _vlm_pick_rotation(ref_photo, left_render,
                                                  right_render, type_)
                    print(f"    [rot] step {step}: cur={current_yaw:+.0f}° "
                          f"vs cand={cand_yaw:+.0f}° → {pick_rot}")
                    if pick_rot == "right":
                        current_yaw = cand_yaw
                        last_better = "right"
                    elif pick_rot == "left" and last_better is None:
                        # First step worse in +; flip direction once.
                        direction = -1
                        last_better = "flip"
                    else:
                        # Worse → revert (current_yaw stays) and stop.
                        break
            entry["rotation_3x3"] = _yaw_rotation(current_yaw)
            entry["yaw_deg"] = float(current_yaw)
            # Final per-item snapshot at the chosen yaw.
            final_path = place_dir / f"{type_safe}_00_final.png"
            _render_scene(out_dir, placements_with_item, final_path)
            if animate and final_path.exists():
                anim_frames_by_type[type_safe].append(("final", final_path))
            print(f"  [rot] final yaw={current_yaw:+.1f}°")
        except Exception as e:
            print(f"  [rot] hill-climb failed: {e} — keeping yaw=0")

        # --- Commit + persist ---
        placements.append(entry)
        all_results.append(entry)
        with open(place_dir / "placement.json", "w") as f:
            json.dump(entry, f, indent=2)

        # Per-item placement GIF — assembled even if rotation skipped, as
        # long as we captured at least baseline + initial.
        if animate and len(anim_frames_by_type.get(type_safe, [])) >= 2:
            try:
                from PIL import Image as _PIL
                gif_path = place_dir / f"{type_safe}_placement.gif"
                frames_pil: list = []
                for _label, _fp in anim_frames_by_type[type_safe]:
                    try:
                        _img = _PIL.open(_fp).convert("RGB")
                        if _img.width != animate_gif_width:
                            _ratio = animate_gif_width / float(_img.width)
                            _img = _img.resize(
                                (animate_gif_width,
                                 int(_img.height * _ratio)),
                                _PIL.LANCZOS)
                        frames_pil.append(_img)
                    except Exception:
                        pass
                if len(frames_pil) >= 2:
                    frames_pil[0].save(
                        str(gif_path), save_all=True,
                        append_images=frames_pil[1:],
                        duration=animate_frame_ms, loop=0, optimize=False)
                    print(f"  [animate] {len(frames_pil)} frames → {gif_path.name}")
            except Exception as _ge:
                print(f"  [animate] GIF assembly failed: {_ge}")

    # 4. Final composite render + save.
    if all_results:
        scene_dst = out_dir / "render_with_missing_items.png"
        _render_scene(out_dir, placements, scene_dst)
        print(f"[missing] final render → {scene_dst}")
        with open(out_dir / "missing_items_placements.json", "w") as f:
            json.dump(all_results, f, indent=2)
    print(f"\n[missing] done. {len(all_results)} item(s) placed.")
    return out_dir


def _resolve_ref_photo(out_dir: Path, cam: dict) -> "Path | None":
    """Find the reference photo path.  Prefer vggt/camera.json image field
    (absolute path to original), then fall back to common locations."""
    # vggt camera.json is a list of frames with "image" path
    vc = out_dir / "vggt" / "camera.json"
    if vc.exists():
        try:
            data = json.loads(vc.read_text())
            if isinstance(data, list) and data and "image" in data[0]:
                p = Path(data[0]["image"])
                if not p.is_absolute():
                    p = Path(__file__).resolve().parents[2] / p
                if p.exists():
                    return p
        except Exception:
            pass
    # Fallbacks
    for cand in [out_dir / "decorations" / "annotated_reference.png",
                 out_dir / "render_vggt.png"]:
        if cand.exists():
            return cand
    # Last resort: any .jpg in scene root
    for f in out_dir.glob("*.jpg"):
        return f
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect and place items missing from the reconstructed scene.",
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output directory (contains furniture/, "
                         "decorations/, camera.json).")
    ap.add_argument("--types", default=None,
                    help="Comma-separated types to bypass VLM detection "
                         "(e.g. 'lamp' or 'lamp,vase'). When omitted, the "
                         "VLM detector decides what's missing.")
    ap.add_argument("--device", default="cuda:0",
                    help="GroundingDINO + SAM device (default cuda:0).")
    ap.add_argument("--no-texture", action="store_true",
                    help="Skip Hunyuan3D texture generation (geometry only).")
    ap.add_argument("--crop-pad", type=int, default=24,
                    help="Padding (px) around segmentation bbox for the "
                         "cropped Hunyuan input. Default 24.")
    ap.add_argument("--rot-steps", type=int, default=4,
                    help="Max hill-climb rotation steps. 0 = skip rotation "
                         "feedback. Default 4.")
    ap.add_argument("--rot-step-deg", type=float, default=30.0,
                    help="Per-step yaw delta for rotation hill-climb (degrees).")
    ap.add_argument("--skip-detect", action="store_true",
                    help="Skip VLM detection entirely (no-op unless --types is set).")
    ap.add_argument("--animate", action="store_true",
                    help="Save a per-item placement GIF (baseline → initial → "
                         "rotation steps → final) at "
                         "<scene>/<type>/placements/<type>_placement.gif.")
    ap.add_argument("--animate-duration", type=int, default=700,
                    help="Per-frame duration in ms for the placement GIF.")
    ap.add_argument("--animate-width", type=int, default=1500,
                    help="GIF width in pixels (frames LANCZOS-resized).")
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",")] if args.types else None
    run(output_dir=args.output_dir,
        missing_types=types,
        device=args.device,
        no_texture=args.no_texture,
        crop_pad=args.crop_pad,
        rot_steps=args.rot_steps,
        rot_step_deg=args.rot_step_deg,
        skip_detect=args.skip_detect,
        animate=args.animate,
        animate_frame_ms=args.animate_duration,
        animate_gif_width=args.animate_width)


if __name__ == "__main__":
    main()
