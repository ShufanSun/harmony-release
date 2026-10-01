#!/usr/bin/env python
"""
la_pipeline_segment.py — LocateAnything+SAM segmenter that emits SceneWeave's
EXACT segment_results.json contract, so the real inpaint / object-generation /
placement stages run downstream unchanged.

Replaces GroundingDINO in segment_wall_objects.py / segment_furniture.py with:
    LocateAnything (boxes + per-type None)  →  VLM verify  →  SAM mask
and writes, into {output_dir}/{wall_mounted|furniture}/:
    segment_results.json
    segmented/segment_NN_<type>_mask.png     (RGBA)
    segmented/segment_NN_<type>_canvas.png   (object on grey 185 canvas)
    segmented/segment_NN_<type>_crop.png      (furniture only — tight bbox crop)

Run in the `eagle-embodied` conda env on a GPU node. Uses the SceneWeave VLM
backend (gpt-5.5 via SCENEWEAVE_VLM_BACKEND=gpt55) for the verify step.

Usage:
    python la_pipeline_segment.py --domain wall      --image REF.jpg --output-dir SCENE
    python la_pipeline_segment.py --domain furniture --image REF.jpg --output-dir SCENE
"""
import argparse
import base64
import io
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# harmony-release root (this file lives in object_placement/la_sam2_workers/);
# needed so _verify / _discover_types can import object_placement.vlm_backend.
_SCENEWEAVE = os.path.dirname(os.path.dirname(_HERE))
_GSA = os.path.join(_SCENEWEAVE, "Grounded-Segment-Anything")
_SAM_CKPT_DEFAULT = os.path.join(_GSA, "weights/sam_vit_h_4b8939.pth")
sys.path.insert(0, os.path.join(_GSA, "segment_anything"))
sys.path.insert(0, _SCENEWEAVE)
# locateanything_worker.py is NVIDIA's, unmodified, from the Embodied/ folder of
# https://github.com/NVlabs/Eagle — point SCENEWEAVE_EAGLE_EMBODIED_DIR at it.
_EAGLE_EMBODIED = os.environ.get("SCENEWEAVE_EAGLE_EMBODIED_DIR")
if _EAGLE_EMBODIED:
    sys.path.insert(0, _EAGLE_EMBODIED)

from locateanything_worker import LocateAnythingWorker  # noqa: E402

SAM_MODEL_TYPE = "vit_h"
OBJ_IMG_SIZE = 768
_BG_GREY = (185, 185, 185)
_CANVAS_PAD = 0.08
_CANVAS_PAD_FRAME = 0.15
_WINDOW_LIKE = ("window", "door")
# Large soft furniture: SAM's tight mask grabs only the rigid frame and drops
# the mattress/cushions — prefer the whole-object (largest in-box) mask instead.
_PREFER_FULL_MASK = {"bed", "sofa", "couch", "sectional", "daybed",
                     "mattress", "armchair", "loveseat"}

# ---- domain configs --------------------------------------------------------
_WALL_TYPES = ["window", "door", "curtain", "painting", "mirror", "clock",
               "wall lamp", "shelf", "tv", "radiator", "air conditioner"]
_FURN_TYPES = ["bed", "sofa", "chair", "armchair", "stool", "coffee table",
               "side table", "end table",
               "dining table", "desk", "bookcase", "cabinet", "nightstand",
               "fireplace", "floor lamp", "plant", "carpet", "rug"]

# normalize free-form LA labels → canonical pipeline type tokens
_ALIASES = {
    "painting": "art", "framed picture": "art", "picture": "art",
    "wall lamp": "light", "sconce": "light", "wall sconce": "light",
    "television": "tv", "air conditioner": "air_conditioner",
    "nightstand": "cabinet", "bedside table": "cabinet",
    "side table": "side_table", "end table": "side_table",
}


def _canon_type(label: str) -> str:
    low = label.lower().strip()
    if low in _ALIASES:
        return _ALIASES[low]
    return re.sub(r"[^a-z0-9]+", "_", low).strip("_") or "object"


# ============================ LocateAnything ===============================
def _parse_labeled_boxes(answer: str, w: int, h: int, min_px: int = 6):
    boxes, absent, current = [], [], ""
    for m in re.finditer(r"<ref>(?P<ref>.*?)</ref>|<box>(?P<box>.*?)</box>", answer):
        if m.group("ref") is not None:
            current = m.group("ref").strip()
            continue
        coords = re.findall(r"<(\d+)>", m.group("box"))
        if len(coords) == 4:
            x1, y1, x2, y2 = (int(c) for c in coords)
            # normalize 0-1000 → px, order corners, clamp to image
            px1, px2 = sorted((x1 / 1000 * w, x2 / 1000 * w))
            py1, py2 = sorted((y1 / 1000 * h, y2 / 1000 * h))
            px1, px2 = max(0, int(px1)), min(w, int(round(px2)))
            py1, py2 = max(0, int(py1)), min(h, int(round(py2)))
            if px2 - px1 < min_px or py2 - py1 < min_px:
                continue  # drop degenerate / zero-area detections
            boxes.append({"label": current, "box_px": [px1, py1, px2, py2]})
        elif current:
            absent.append(current)
    return boxes, absent


# ============================ VLM verify ===================================
_VERIFY_WALL = """\
This interior photograph has ONE region outlined by a thick red rectangle, which
a detector labelled "{label}". Using the whole-room context, decide whether the
outlined region is GENUINELY a real, distinct {label} fixed to a vertical WALL.
Answer NO if it is a CEILING fixture (chandelier/pendant/flush light/fan), a
floor/table lamp, crown molding or a ceiling line, a shadow/reflection/seam, the
edge of furniture or a plant, plain wall/curtain fabric, a thin frame-edge sliver,
or anything that is not clearly a real wall-mounted {label}. For a light, YES only
for a wall sconce bolted to the wall. Respond with one word: YES or NO."""

_VERIFY_FURN = """\
This interior photograph has ONE region outlined by a thick red rectangle, which
a detector labelled "{label}". Using the whole-room context, decide whether the
outlined region is GENUINELY a real, distinct, free-standing {label} — a piece of
furniture / large decor resting on the FLOOR (or, for a carpet/rug, lying on the
floor). Answer NO if it is wall-mounted or ceiling-hung, only a shadow/reflection/
seam, a tiny sliver at the photo edge, part of a different object, plain floor or
wall, or anything that is not clearly a real {label}. Respond one word: YES or NO."""


def _strip_thinking(t):
    return re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.IGNORECASE).strip()


def _verify(image_np, box_px, label, domain, timeout=60):
    try:
        from object_placement.vlm_backend import vlm_post
    except Exception as e:
        return True, f"SKIP(no-backend:{type(e).__name__})"
    try:
        im = Image.fromarray(image_np).convert("RGB")
        d = ImageDraw.Draw(im)
        x1, y1, x2, y2 = (int(v) for v in box_px)
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0),
                    width=max(4, int(0.006 * max(im.size))))
        buf = io.BytesIO(); im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        tpl = _VERIFY_WALL if domain == "wall" else _VERIFY_FURN
        payload = {"model": "qwen3", "max_tokens": 220,
                   "chat_template_kwargs": {"enable_thinking": True},
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": tpl.format(label=label)},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
        resp = vlm_post(payload, timeout=timeout)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"]).upper()
        toks = re.findall(r"\b(YES|NO)\b", raw)
        verdict = toks[-1] if toks else "YES"
        return verdict != "NO", verdict
    except Exception as e:
        return True, f"SKIP(error:{type(e).__name__})"


_DISCOVER_FURN = (
    "Look at this indoor room photo. List EVERY distinct piece of FREE-STANDING "
    "FURNITURE that sits on the floor — seating, tables, storage, floor lamps, "
    "floor plants, rugs. Use short generic singular nouns (e.g. \"sofa\", "
    "\"armchair\", \"side table\", \"ottoman\", \"console table\", \"bench\", "
    "\"floor lamp\", \"plant\", \"rug\"). Do NOT list wall-mounted or ceiling "
    "items, small decor sitting on top of furniture, windows, or architecture. "
    "Respond with ONLY a JSON array of strings, nothing else."
)
_DISCOVER_WALL = (
    "Look at this indoor room photo. List EVERY distinct type of WALL-MOUNTED "
    "object visible — windows, curtains, paintings/art, mirrors, wall lamps/"
    "sconces, shelves, clocks, TVs, radiators, AC units. Use short generic "
    "singular nouns. Respond with ONLY a JSON array of strings, nothing else."
)


def _discover_types(image_np, domain, timeout=60):
    """Open-vocabulary type discovery: ask the VLM what object types are actually
    visible in the photo, so new/unlisted pieces (side table, ottoman, console,
    bench, …) get auto-included alongside the base vocabulary instead of being
    missed because they weren't pre-listed. Returns a list of lowercase type
    strings (empty on any failure — detection then just uses the base list)."""
    import json
    try:
        from object_placement.vlm_backend import vlm_post
    except Exception:
        return []
    try:
        im = Image.fromarray(image_np).convert("RGB")
        buf = io.BytesIO(); im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        tpl = _DISCOVER_WALL if domain == "wall" else _DISCOVER_FURN
        payload = {"model": "qwen3", "max_tokens": 300,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": tpl},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
        resp = vlm_post(payload, timeout=timeout)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return []
        out = []
        for it in json.loads(m.group()):
            s = str(it).lower().strip()
            if s and len(s) <= 30 and s not in out:
                out.append(s)
        return out
    except Exception:
        return []


# ============================ SAM ==========================================
def _sam_segment_box(predictor, image_np, box_px, ttype="other", exclude_boxes=None):
    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(box=np.array(box_px)[None], multimask_output=True)
    H, W = image_np.shape[:2]
    bx1, by1, bx2, by2 = (int(v) for v in box_px)
    bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
    over = 0.10 if ttype in _WINDOW_LIKE else 0.05
    dx1, dy1 = max(0, int(bx1 - bw * over)), max(0, int(by1 - bh * over))
    dx2, dy2 = min(W, int(bx2 + bw * over)), min(H, int(by2 + bh * over))
    box_area = bw * bh

    def _sc(m, s):
        if not m.any():
            return -1.0
        ys, xs = np.where(m)
        mx1, my1, mx2, my2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        ix1, iy1, ix2, iy2 = max(mx1, bx1), max(my1, by1), min(mx2, bx2), min(my2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return -1.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        m_area = (mx2 - mx1) * (my2 - my1)
        iou = inter / max(m_area + box_area - inter, 1)
        return float(s) * iou / (max(1.0, m_area / max(box_area, 1)) ** 0.5)

    clip = np.zeros((H, W), bool); clip[dy1:dy2, dx1:dx2] = True
    if ttype in _PREFER_FULL_MASK:
        # Beds / sofas: SAM's tight candidate grabs only the wooden frame and
        # drops the mattress/bedding (soft, different texture). Pick the
        # candidate with the LARGEST coverage inside the box (the whole-object
        # mask), so the mattress + bedding are included — not just the frame.
        best = int(max(range(len(masks)),
                       key=lambda i: int((masks[i].astype(bool) & clip).sum())))
    else:
        best = int(sorted(((_sc(m, s), i) for i, (m, s) in enumerate(zip(masks, scores))), reverse=True)[0][1])
    mask = masks[best].astype(bool)
    mask &= clip
    # Drop pixels that belong to OTHER detected objects (e.g. a nightstand
    # sitting inside the bed's bounding box) so the whole-object mask doesn't
    # swallow neighbours.
    for eb in (exclude_boxes or []):
        ex1, ey1, ex2, ey2 = (int(v) for v in eb)
        mask[max(0, ey1):min(H, ey2), max(0, ex1):min(W, ex2)] = False
    if mask.any():
        try:
            from scipy.ndimage import label as cc
            lab, n = cc(mask)
            if n > 1:
                sz = np.bincount(lab.ravel()); sz[0] = 0
                mask = lab == int(sz.argmax())
        except ImportError:
            pass
    return mask, float(scores[best])


# ============================ save (contract) ==============================
def _save_mask_rgba(image_np, mask, path):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[mask, :3] = image_np[mask]; rgba[mask, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def _save_canvas(image_np, mask, path, box_px=None, ttype="other"):
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        Image.new("RGB", (OBJ_IMG_SIZE, OBJ_IMG_SIZE), _BG_GREY).save(path); return
    H, W = image_np.shape[:2]
    win = ttype in _WINDOW_LIKE
    r0 = int(np.argmax(rows)); r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols)); c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    if win and box_px is not None:
        bx1, by1, bx2, by2 = (int(v) for v in box_px)
        r0, r1, c0, c1 = min(r0, by1), max(r1, by2 - 1), min(c0, bx1), max(c1, bx2 - 1)
    pad = _CANVAS_PAD_FRAME if win else _CANVAS_PAD
    pr, pc = int((r1 - r0) * pad), int((c1 - c0) * pad)
    r0, r1 = max(0, r0 - pr), min(H - 1, r1 + pr)
    c0, c1 = max(0, c0 - pc), min(W - 1, c1 + pc)
    crop = image_np[r0:r1 + 1, c0:c1 + 1].copy()
    cmask = mask[r0:r1 + 1, c0:c1 + 1]
    bg = crop if win else np.where(cmask[..., None], crop, np.array(_BG_GREY, np.uint8))
    obj = Image.fromarray(bg.astype(np.uint8))
    cw, ch = obj.size; side = max(cw, ch, 1)
    canvas = Image.new("RGB", (side, side), _BG_GREY)
    canvas.paste(obj, ((side - cw) // 2, (side - ch) // 2))
    canvas.resize((OBJ_IMG_SIZE, OBJ_IMG_SIZE), Image.LANCZOS).save(path)


def _save_crop(image_np, box_px, path):
    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box_px)
    x2 = min(W, max(x1 + 1, x2)); y2 = min(H, max(y1 + 1, y2))
    crop = image_np[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_np[max(0, y1 - 1):y1 + 1, max(0, x1 - 1):x1 + 1]
    Image.fromarray(crop).save(path)


# ============================ main =========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["wall", "furniture"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--output-dir", required=True, help="scene working dir")
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sam-ckpt", default=_SAM_CKPT_DEFAULT)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    is_wall = args.domain == "wall"
    types = _WALL_TYPES if is_wall else _FURN_TYPES
    dom_dir = os.path.join(args.output_dir, "wall_mounted" if is_wall else "furniture")
    seg_dir = os.path.join(dom_dir, "segmented")
    os.makedirs(seg_dir, exist_ok=True)
    for sub in ("inpainted", "objects"):
        os.makedirs(os.path.join(dom_dir, sub), exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    image_np = np.array(img); w, h = img.size
    print(f"[la-seg:{args.domain}] image {w}x{h} → {dom_dir}")

    worker = LocateAnythingWorker(args.model, device=args.device)
    # SCENEWEAVE_LA_OPEN_DETECT=1 — open detection: instead of grounding a fixed
    # VLM type list (which silently drops anything LA can't localize by name,
    # e.g. a heavily-occluded bench), ask LA to locate ANY furniture/object and
    # return whatever it finds; the VLM-verify filter below decides furniture.
    if os.environ.get("SCENEWEAVE_LA_OPEN_DETECT"):
        _open = ("any piece of furniture or freestanding object in the room, "
                 "including any partially hidden, clipped or occluded item")
        print(f"[la-seg] (1) OPEN-DETECT mode: {_open!r}")
        answer = worker.detect(img, [_open], generation_mode="hybrid",
                               temperature=args.temperature, verbose=False)["answer"]
    else:
        print(f"[la-seg] (1) LocateAnything types: {types}")
        answer = worker.detect(img, types, generation_mode="hybrid",
                               temperature=args.temperature, verbose=False)["answer"]
    boxes, absent = _parse_labeled_boxes(answer, w, h)
    print(f"[la-seg]     located {len(boxes)}; absent {absent or '—'}")

    kept = []
    for b in boxes:
        if args.no_verify:
            b["verdict"] = "SKIP(--no-verify)"; kept.append(b); continue
        keep, verdict = _verify(image_np, b["box_px"], b["label"], args.domain)
        b["verdict"] = verdict
        print(f"          {b['label']:22s} {verdict:16s} → {'keep' if keep else 'DROP'}")
        if keep:
            kept.append(b)
    print(f"[la-seg]     {len(kept)}/{len(boxes)} survived verify")

    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=args.sam_ckpt).to(args.device)
    predictor = SamPredictor(sam)

    segments, detected = [], []
    for i, b in enumerate(kept):
        ttype = _canon_type(b["label"])
        box_px = [int(v) for v in b["box_px"]]
        # For whole-object (bed/sofa) masks, exclude OTHER detections' boxes so
        # the large mask doesn't swallow a neighbour (e.g. a nightstand inside
        # the bed box).
        exclude = ([o["box_px"] for j, o in enumerate(kept) if j != i]
                   if ttype in _PREFER_FULL_MASK else None)
        mask, score = _sam_segment_box(predictor, image_np, box_px, ttype, exclude_boxes=exclude)
        stem = f"segment_{i:02d}_{ttype}"
        mask_file = f"{stem}_mask.png"
        canvas_file = f"{stem}_canvas.png"
        _save_mask_rgba(image_np, mask, os.path.join(seg_dir, mask_file))
        _save_canvas(image_np, mask, os.path.join(seg_dir, canvas_file), box_px, ttype)
        seg = {"index": i, "type": ttype, "phrase": b["label"],
               "gdino_score": 1.0, "sam_score": round(score, 4),
               "coverage": round(float(mask.mean()), 4), "box_px": box_px,
               "mask_file": mask_file, "canvas_file": canvas_file,
               "verify_verdict": b["verdict"], "inpaint_file": None, "glb_file": None}
        if not is_wall:
            crop_file = f"{stem}_crop.png"
            _save_crop(image_np, box_px, os.path.join(seg_dir, crop_file))
            seg["crop_file"] = crop_file
        segments.append(seg); detected.append(ttype)
        print(f"          [{i}] {ttype:20s} sam={score:.2f} → {stem}")

    results = {"detected_types": detected,
               "gdino_prompt": " . ".join(types),
               "source": "locateanything+sam",
               "segments": segments}
    with open(os.path.join(dom_dir, "segment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[la-seg:{args.domain}] wrote {len(segments)} segments → "
          f"{os.path.join(dom_dir, 'segment_results.json')}")


if __name__ == "__main__":
    main()
