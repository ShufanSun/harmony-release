#!/usr/bin/env python
"""
sam2_from_boxes.py — Stage 2 of LocateAnything→SAM2.

Reads a boxes.json from la_emit_boxes.py and runs SAM2 (box-prompted) to produce
high-quality masks — combining LocateAnything's grounding/recall with SAM2's
whole-object masks. Run in the `lang-sam` env.

    python sam2_from_boxes.py --boxes BOXES.json --out OUTDIR

Detection is done entirely by LocateAnything (la_emit_boxes.py). SAM2 only
turns each LocateAnything box into a pixel mask, because LocateAnything outputs
boxes/points, not masks, and the inpaint / object-generation stages need masks.

Model: Meta SAM 2.1 (https://github.com/facebookresearch/sam2), default
sam2.1_hiera_small (--sam-type / SCENEWEAVE_SAM2_TYPE: sam2.1_hiera_tiny|small|
base_plus|large). Loaded through lang_sam
(https://github.com/luca-medeiros/lang-segment-anything; `pip install -e .`
in the lang-sam env), which downloads the checkpoint from
https://dl.fbaipublicfiles.com/segment_anything_2/092824/ into the torch hub
cache on first run. On an offline node, fetch that .pt file ahead of time into
$TORCH_HOME/hub/checkpoints/.

Swapping the segmenter: any box-prompted segmenter works. Replace
`SAM().build_model` / `sam.predict` so it returns one HxW bool mask (plus a
score) per box in boxes.json; everything after that is model-agnostic.
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lang_sam.models.sam import SAM

_BG_GREY = (185, 185, 185)
OBJ_IMG_SIZE = 768
_WINDOW_LIKE = ("window", "door")
_COLORS = np.array([[255, 64, 64], [64, 160, 255], [64, 220, 96], [255, 176, 32],
                    [200, 96, 255], [0, 200, 200], [255, 96, 180], [160, 160, 64],
                    [255, 128, 0], [120, 220, 255]], dtype=np.uint8)


def _font():
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        return ImageFont.load_default()


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "object"


def _save_mask_rgba(image_np, mask, path):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[mask, :3] = image_np[mask]; rgba[mask, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def _save_canvas(image_np, mask, box, path, win=False):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        Image.new("RGB", (OBJ_IMG_SIZE, OBJ_IMG_SIZE), _BG_GREY).save(path); return
    H, W = image_np.shape[:2]
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if win:
        bx1, by1, bx2, by2 = (int(v) for v in box)
        r0, r1, c0, c1 = min(r0, by1), max(r1, by2), min(c0, bx1), max(c1, bx2)
    pad = int((r1 - r0) * (0.15 if win else 0.08)); padc = int((c1 - c0) * (0.15 if win else 0.08))
    r0, r1 = max(0, r0 - pad), min(H - 1, r1 + pad); c0, c1 = max(0, c0 - padc), min(W - 1, c1 + padc)
    crop = image_np[r0:r1 + 1, c0:c1 + 1].copy(); cmask = mask[r0:r1 + 1, c0:c1 + 1]
    if win:
        # Mask-isolate windows too.  Previously windows used the raw crop (no
        # mask), which let room context (sofa/wall inside the padded crop) bleed
        # into the canvas — the inpaint then regenerated the whole scene instead
        # of just the window.  Dilate the mask slightly so the frame around the
        # glass is preserved, then grey out everything outside it.
        from PIL import ImageFilter
        _md = max(3, int(0.03 * max(crop.shape[:2])) | 1)   # odd kernel ≥3
        cm = np.asarray(Image.fromarray((cmask.astype(np.uint8) * 255)).filter(
            ImageFilter.MaxFilter(_md))) > 127
        bg = np.where(cm[..., None], crop, np.array(_BG_GREY, np.uint8))
    else:
        bg = np.where(cmask[..., None], crop, np.array(_BG_GREY, np.uint8))
    obj = Image.fromarray(bg.astype(np.uint8)); cw, ch = obj.size; side = max(cw, ch, 1)
    canvas = Image.new("RGB", (side, side), _BG_GREY); canvas.paste(obj, ((side - cw) // 2, (side - ch) // 2))
    canvas.resize((OBJ_IMG_SIZE, OBJ_IMG_SIZE), Image.LANCZOS).save(path)


def _overlay(img, dets):
    base = np.asarray(img.convert("RGB")).astype(np.float32)
    for i, dd in enumerate(dets):
        col = _COLORS[i % len(_COLORS)].astype(np.float32)
        base[dd["mask"]] = 0.45 * col + 0.55 * base[dd["mask"]]
    vis = Image.fromarray(base.astype(np.uint8)); d = ImageDraw.Draw(vis); font = _font()
    for i, dd in enumerate(dets):
        col = tuple(int(c) for c in _COLORS[i % len(_COLORS)]); x1, y1, x2, y2 = dd["box"]
        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        lbl = f"{i}:{dd['type']}"
        tb = d.textbbox((x1, y1), lbl, font=font); d.rectangle([tb[0], tb[1] - 2, tb[2] + 2, tb[3]], fill=col)
        d.text((x1 + 1, y1), lbl, fill=(0, 0, 0), font=font)
    return vis


def _save_crop(image_np, box, path):
    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x2 = min(W, max(x1 + 1, x2)); y2 = min(H, max(y1 + 1, y2))
    Image.fromarray(image_np[y1:y2, x1:x2]).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True, help="boxes.json from la_emit_boxes.py")
    ap.add_argument("--out", default=None, help="standalone output dir (simple mode)")
    ap.add_argument("--sam-type", default="sam2.1_hiera_small")
    # Contract mode: write the SceneWeave segment_results.json + segment_NN_<type>
    # files into <output-dir>/<wall_mounted|furniture>/ so the pipeline consumes it.
    ap.add_argument("--contract", action="store_true",
                    help="emit the SceneWeave segment_results.json contract")
    ap.add_argument("--output-dir", default=None, help="scene dir (contract mode)")
    ap.add_argument("--domain", default=None, choices=["wall", "furniture", "decoration"])
    args = ap.parse_args()

    data = json.load(open(args.boxes))
    img = Image.open(data["image"]).convert("RGB")
    image_np = np.array(img); w, h = img.size
    segs = data["boxes"]
    domain = args.domain or data.get("domain", "wall")

    if args.contract:
        if not args.output_dir:
            ap.error("--contract requires --output-dir")
        _sub = {"wall": "wall_mounted", "furniture": "furniture", "decoration": "decorations"}[domain]
        dom_dir = os.path.join(args.output_dir, _sub)
        for sub in ("segmented", "inpainted", "objects"):
            os.makedirs(os.path.join(dom_dir, sub), exist_ok=True)
        seg_dir = os.path.join(dom_dir, "segmented")
        out_base = dom_dir
    else:
        if not args.out:
            ap.error("provide --out (simple) or --contract --output-dir")
        seg_dir = os.path.join(args.out, "segmented"); os.makedirs(seg_dir, exist_ok=True)
        out_base = args.out

    print(f"[sam2] {domain}: {len(segs)} boxes from LocateAnything → SAM2 masks "
          f"({'contract' if args.contract else 'simple'} mode)")
    is_wall = domain == "wall"
    is_decor = domain == "decoration"

    sam = SAM(); sam.build_model(args.sam_type)
    masks = np.zeros((0, h, w), bool); scores = np.zeros(0)
    if segs:
        xyxy = np.array([s["box_px"] for s in segs], dtype=float)
        masks, scores, _ = sam.predict(image_np, xyxy)
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None]
        scores = np.asarray(scores).reshape(-1)

    dets, segments, detected = [], [], []
    for i, s in enumerate(segs):
        ttype = s["type"]; box = s["box_px"]; win = ttype in _WINDOW_LIKE
        phrase = s.get("phrase", s.get("label", ttype))
        mask = masks[i].astype(bool)
        score = float(scores[i]) if i < len(scores) else 0.0
        if is_decor:
            stem = f"decor_{i:02d}_{_slug(phrase)}"
        elif args.contract:
            stem = f"segment_{i:02d}_{_slug(ttype)}"
        else:
            stem = f"seg_{i:02d}_{_slug(ttype)}"
        mask_file, canvas_file = f"{stem}_mask.png", f"{stem}_canvas.png"
        _save_mask_rgba(image_np, mask, os.path.join(seg_dir, mask_file))
        _save_canvas(image_np, mask, box, os.path.join(seg_dir, canvas_file), win)
        dets.append({"type": ttype, "box": box, "mask": mask})
        if is_decor:
            # decoration schema: tied to a parent furniture piece, named decor_NN
            crop_file = f"{stem}_crop.png"
            _save_crop(image_np, box, os.path.join(seg_dir, crop_file))
            seg = {"seg_index": i, "furniture_index": s.get("furniture_index"),
                   "furniture_type": s.get("furniture_type"), "phrase": phrase,
                   "gdino_phrase": phrase, "gdino_score": 1.0, "sam_score": round(score, 4),
                   "box_px": box, "coverage": round(float(mask.mean()), 4),
                   "mask_file": mask_file, "canvas_file": canvas_file, "crop_file": crop_file,
                   "object_description": phrase, "text_only": False}
        else:
            # NOTE: do NOT pre-set inpaint_file/glb_file — downstream stages add them.
            # The analysis stage does seg.get("glb_file", "<default path>"), which
            # returns None (not the default) if the key exists as null → placement
            # crashes on Path(None). Omit the keys entirely, matching segment_furniture.
            seg = {"index": i, "type": ttype, "phrase": phrase,
                   "label": s.get("label"), "gdino_score": 1.0, "sam_score": round(score, 4),
                   "sam2_score": round(score, 4), "coverage": round(float(mask.mean()), 4),
                   "box_px": box, "verify_verdict": s.get("verdict"),
                   "mask_file": mask_file, "canvas_file": canvas_file}
            if args.contract and not is_wall:
                crop_file = f"{stem}_crop.png"
                _save_crop(image_np, box, os.path.join(seg_dir, crop_file))
                seg["crop_file"] = crop_file
        segments.append(seg); detected.append(ttype)
        print(f"   [{i}] {(phrase if is_decor else ttype):24s} sam2={score:.2f} cov={mask.mean():.3f} → {stem}")

    _overlay(img, dets).save(os.path.join(out_base, "overlay.png"))
    if args.contract:
        record = {"detected_types": detected, "gdino_prompt": " . ".join(detected),
                  "source": "locateanything+sam2", "absent": data.get("absent", []),
                  "segments": segments}
        json.dump(record, open(os.path.join(out_base, "segment_results.json"), "w"), indent=2)
        print(f"[sam2:{domain}] wrote contract → {out_base}/segment_results.json ({len(segments)} segments)")
    else:
        record = {"image": data["image"], "image_size": [w, h], "domain": domain,
                  "absent": data.get("absent", []), "source": "locateanything+sam2",
                  "segments": segments}
        json.dump(record, open(os.path.join(out_base, "results.json"), "w"), indent=2)
        print(f"[sam2] wrote {len(dets)} → {out_base}/  (overlay.png, results.json, segmented/)")


if __name__ == "__main__":
    main()
