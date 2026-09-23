"""
segment_ceiling_objects.py — segment CEILING-mounted fixtures with Grounded-SAM.

Counterpart to ``wall_mounted/segment_wall_objects.py`` for objects that hang
from or are mounted on the CEILING — pendant lights, chandeliers, hanging lamps,
flush-mount ceiling lights, track lights and ceiling fans.  The wall-mounted
stage deliberately REJECTS every ceiling fixture (it defers them to "a different
stage" — this one), and the furniture / lighting stages only ever treat ceiling
lights as illumination sources, never as reconstructed-and-placed geometry, so
without this stage pendant lights simply never appear in the scene.

Pipeline (mirrors the wall-mounted segmenter, reusing its Grounded-SAM helpers):
  1. VLM → which ceiling-fixture TYPES are present.
  2. Build a GroundingDINO prompt from those types.
  3. GroundingDINO → boxes; size + containment filters.
  4. VLM context-verify each box: keep ONLY genuine ceiling-hung/ceiling-mounted
     fixtures; reject wall sconces, floor/table lamps, the ceiling line, glare …
  5. SAM (box-prompted) → mask; save RGBA mask + grey-canvas crop.
  6. Write <output_dir>/ceiling/segment_results.json.

The output layout (segment_results.json + segmented/ + inpainted/ + objects/) is
intentionally identical to the wall-mounted folder so the existing
``object_generation.run(wall_mounted_dir=<ceiling_dir>)`` reconstructs the
GLBs unchanged.

Usage:
    python -m object_placement.ceiling.segment_ceiling_objects \\
        --output-dir outputs/front3d/rgb_003200 \\
        --image outputs/front3d/rgb_003200/ref.png
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
import torch
from PIL import Image

# ── Reuse the heavy Grounded-SAM machinery from the wall-mounted segmenter ─────
# Detection / segmentation / IO are identical; only the type vocabulary, the VLM
# prompts and the output directory differ.
from object_placement.wall_mounted.segment_wall_objects import (
    _load_models,
    _gdino_predict,
    _boxes_to_px,
    _filter_large_boxes,
    _filter_contained_boxes,
    _sam_segment_box,
    _save_mask_rgba,
    _save_canvas,
    _encode_image,
    _strip_thinking,
)
from object_placement.vlm_backend import vlm_post as _vlm_post

# ── Type → GroundingDINO noun phrase ─────────────────────────────────────────
_TYPE_GDINO: dict[str, str] = {
    "pendant_light": ("pendant light . pendant lamp . hanging lamp . "
                      "hanging light . hanging pendant"),
    "chandelier":    "chandelier . crystal chandelier . candelabra light",
    "ceiling_lamp":  ("ceiling light . ceiling lamp . flush mount light . "
                      "ceiling fixture . dome light"),
    "ceiling_fan":   "ceiling fan",
    "track_light":   "track light . spotlight rail . recessed spotlight",
}

# Types fed to GroundingDINO regardless of what the VLM returned.  The VLM type
# detector routinely misses a pendant/ceiling lamp in a well-lit room; letting
# GDINO + its score thresholds gate detection is far more reliable.  Restricted
# to the two most common residential ceiling fixtures.
_ALWAYS_INCLUDE_TYPES: list[str] = ["pendant_light", "ceiling_lamp"]

# Verify each box in broader context (full photo, box drawn).  Drops only on a
# clear NO.  Disable with SCENEWEAVE_NO_SEGMENT_VERIFY=1.
VERIFY_SEGMENTS = os.environ.get("SCENEWEAVE_NO_SEGMENT_VERIFY") != "1"

OBJ_IMG_SIZE = 768

# ── VLM: which ceiling-fixture types are present ──────────────────────────────

_TYPE_DETECTION_PROMPT = """\
Look at this interior room photograph.

List ONLY the categories of CEILING-mounted light fixtures / objects that are
clearly visible.  Choose from this fixed list:
  pendant_light, chandelier, ceiling_lamp, ceiling_fan, track_light

Definitions:
- "pendant_light" = a lamp hanging from the ceiling on a cord, rod, or chain
                    (one or several drop pendants).
- "chandelier"    = a branched / multi-arm decorative hanging light fixture.
- "ceiling_lamp"  = a flush- or semi-flush-mount fixture fixed directly to the
                    ceiling surface (dome light, ceiling plate light).
- "ceiling_fan"   = a fan mounted to the ceiling (with or without a light kit).
- "track_light"   = a rail/track of spotlights, or recessed ceiling spotlights.

Rules:
- Include a category ONLY if at least one instance hangs from or is mounted on
  the CEILING.
- EXCLUDE wall sconces, floor lamps, table/desk lamps, and any lamp whose base
  rests on the floor or on furniture — those are NOT ceiling fixtures.
- Do NOT include furniture, windows, or anything attached to a vertical wall.

For EACH fixture, also write a PRECISE, SHORT grounding phrase describing what you
actually see — it drives an open-vocabulary detector (GroundingDINO), best on
concrete 1-3 word noun phrases. Be specific to the ACTUAL fixture (shape / kind),
not the generic category, with 1-2 alternatives separated by " . ". Examples:
  a crystal chandelier   →  "crystal chandelier . hanging chandelier"
  a recessed downlight   →  "recessed light . downlight . ceiling spotlight"
  a flush ceiling fan    →  "ceiling fan . flush mount fan"

OUTPUT — a JSON array of objects, each {"type": "<snake_case category>",
"gdino": "<short specific noun phrase . alt phrasing>"}. Example:
  [{"type":"pendant_light","gdino":"pendant light . hanging pendant lamp"}]
No markdown, no explanation.
"""


# Scene-specific GroundingDINO phrases the VLM writes per fixture (type->phrase),
# populated by detect_object_types(); build_gdino_prompt() prefers these. Reset per call.
_VLM_PHRASES: dict[str, str] = {}


def detect_object_types(image_path: str) -> list[str]:
    """Ask the VLM which ceiling-fixture types are present."""
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
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    print("[ceil_seg] Calling VLM to identify ceiling-fixture types …")
    try:
        resp = _vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = _strip_thinking(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[ceil_seg] VLM type-detect failed ({e}) — relying on always-include")
        return []

    global _VLM_PHRASES
    _VLM_PHRASES = {}
    m = re.search(r"\[[\s\S]*\]", raw)   # greedy — array holds nested {…} objects
    if not m:
        print(f"[ceil_seg] VLM returned unexpected format: {raw[:200]}")
        return []
    try:
        items = json.loads(m.group())
        if isinstance(items, list):
            # New format [{"type","gdino"}] (old ["type",…] still accepted); the
            # per-object phrase is stashed for build_gdino_prompt. Keep known types
            # plus any VLM-named type carrying a phrase (open-vocab).
            valid: list[str] = []
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
                if (t in _TYPE_GDINO or phrase) and t not in valid:
                    valid.append(t)
            print(f"[ceil_seg] VLM identified types: {valid}")
            if _VLM_PHRASES:
                print(f"[ceil_seg] VLM scene-specific gdino phrases: {_VLM_PHRASES}")
            return valid
    except json.JSONDecodeError:
        pass
    print(f"[ceil_seg] Could not parse VLM response: {raw[:200]}")
    return []


_VERIFY_PROMPT = """\
This interior room photograph has ONE region outlined by a thick red rectangle.

GroundingDINO detected the outlined region as a "{phrase}" (category: {obj_type}).
Using the BROADER CONTEXT of the whole room, decide whether the outlined region is
GENUINELY a real, distinct light fixture / object that HANGS FROM or is MOUNTED ON
the CEILING.  This stage handles CEILING fixtures ONLY.

Answer YES only if the outlined region is unmistakably a ceiling-hung or
ceiling-mounted fixture: a pendant light, chandelier, hanging lamp, flush-mount
ceiling light, track light, or ceiling fan whose attachment point is the CEILING
(top of the fixture meets the ceiling; cord / chain / rod descends FROM the ceiling).

Answer NO (reject) if the outlined region is any of:
- a WALL sconce or any lamp whose bracket originates from a vertical wall,
- a floor lamp, torchiere, desk/table/bedside lamp, or any lamp resting on the
  floor or on furniture,
- a window, skylight glare, the wall-ceiling junction, crown molding, a ceiling
  beam, or a plain ceiling line,
- a shadow, reflection, glare, or seam,
- the top edge of furniture, a plant, or another object,
- only a thin sliver along the very edge of the photo,
- or anything that is NOT a clearly visible, real ceiling fixture.

Respond with a single word: YES or NO.
"""


def _verify_segment(image_np: np.ndarray, box_px: list[int], obj_type: str,
                    phrase: str, timeout: int = 60) -> bool:
    """Confirm a detected box is really a ceiling fixture, shown IN CONTEXT.

    Lenient: keeps on any error / unparseable answer, drops only on explicit NO."""
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
    """Prefer the VLM's precise scene-specific phrase (_VLM_PHRASES), unioned with
    the generic _TYPE_GDINO phrase; fall back to the bare noun."""
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


def _match_type(phrase: str) -> str:
    """Map a GroundingDINO phrase back to a canonical ceiling type key."""
    phrase_l = phrase.lower()
    candidates: list[tuple[str, str]] = []
    for t, gdino_phrase in _TYPE_GDINO.items():
        for sub in gdino_phrase.split("."):
            sub = sub.strip().lower()
            if sub and (sub in phrase_l or phrase_l in sub):
                candidates.append((t, sub))
    if candidates:
        candidates.sort(key=lambda c: len(c[1]), reverse=True)
        return candidates[0][0]
    phrase_words = set(re.findall(r"[a-z]+", phrase_l))
    for t, gdino_phrase in _TYPE_GDINO.items():
        if phrase_words & set(re.findall(r"[a-z]+", gdino_phrase.lower())):
            return t
    return phrase.replace(" ", "_").lower()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(output_dir: str | Path, image_path: str,
        types: list[str] | None = None) -> Path:
    out_dir = Path(output_dir) / "ceiling"
    seg_dir = out_dir / "segmented"
    seg_dir.mkdir(parents=True, exist_ok=True)

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results_path = out_dir / "segment_results.json"

    # ── Step 1: VLM → types (∪ always-include ∪ requested) ───────────────────
    detected = detect_object_types(image_path)
    for t in _ALWAYS_INCLUDE_TYPES:
        if t not in detected:
            detected.append(t)
    if types is not None:
        for t in types:
            if t not in detected:
                detected.append(t)
    if not detected:
        print("[ceil_seg] No ceiling-fixture types — done.")
        return seg_dir
    print(f"[ceil_seg] effective types (VLM ∪ always-include): {detected}")

    prompt = build_gdino_prompt(detected)
    print(f"[ceil_seg] GroundingDINO prompt: '{prompt}'")

    # ── Step 2: GroundingDINO → boxes ────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gdino, sam_predictor = _load_models(device)

    image_np      = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w  = image_np.shape[:2]

    boxes_norm, logits, phrases = _gdino_predict(gdino, image_path, prompt, device)
    print(f"[ceil_seg] GroundingDINO: {len(boxes_norm)} raw box(es) — {phrases}")

    boxes_norm, logits, phrases = _filter_large_boxes(boxes_norm, logits, phrases)
    boxes_norm, logits, phrases = _filter_contained_boxes(boxes_norm, logits, phrases)
    print(f"[ceil_seg] After size/containment filters: {len(boxes_norm)} box(es)")

    if len(boxes_norm) == 0:
        print("[ceil_seg] No boxes remain — done.")
        with open(results_path, "w") as f:
            json.dump({"detected_types": detected, "gdino_prompt": prompt,
                       "segments": []}, f, indent=2)
        return seg_dir

    boxes_px = _boxes_to_px(boxes_norm, img_w, img_h)

    # ── Step 3: SAM → masks, save outputs ────────────────────────────────────
    results = []
    for new_i, (box_px, phrase, logit) in enumerate(zip(boxes_px, phrases, logits)):
        idx = new_i
        obj_type = _match_type(phrase)
        print(f"[ceil_seg] {idx:02d} '{phrase}' (score {logit:.3f})  box {box_px}  → {obj_type}")

        # Reject GDINO false positives (wall sconces, the ceiling line, glare)
        # by showing the VLM the box IN the full room photo.
        if VERIFY_SEGMENTS and not _verify_segment(image_np, box_px, obj_type, phrase):
            print(f"          REJECTED by VLM context check — skip '{phrase}'")
            continue

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

    # ── Same-type bbox dedup (drop the lower-scoring near-duplicate) ──────────
    _IOU_DEDUP = 0.50

    def _iou(b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0, 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = max((b1[2]-b1[0]) * (b1[3]-b1[1]), 1)
        a2 = max((b2[2]-b2[0]) * (b2[3]-b2[1]), 1)
        return inter / max(a1 + a2 - inter, 1), inter / min(a1, a2)

    drop: set[int] = set()
    for i in range(len(results)):
        if i in drop:
            continue
        for j in range(len(results)):
            if j == i or j in drop or results[i]["type"] != results[j]["type"]:
                continue
            iou, cont = _iou(results[i]["box_px"], results[j]["box_px"])
            if iou < _IOU_DEDUP and cont < 0.90:
                continue
            si, sj = results[i]["gdino_score"], results[j]["gdino_score"]
            loser = i if si < sj else j
            winner = j if loser == i else i
            print(f"  [dedup] dropping {results[loser]['type']} {loser} "
                  f"(iou={iou:.2f} cont={cont:.2f} vs {winner})")
            drop.add(loser)
            if loser == i:
                break
    if drop:
        for k in drop:
            for fk in ("mask_file", "canvas_file"):
                f = results[k].get(fk)
                if f:
                    try:
                        (seg_dir / f).unlink()
                    except FileNotFoundError:
                        pass
        results = [r for k, r in enumerate(results) if k not in drop]
        for new_local, r in enumerate(results):
            r["index"] = new_local

    # ── Save results JSON ─────────────────────────────────────────────────────
    payload = {
        "detected_types": detected,
        "gdino_prompt":   prompt,
        "segments":       results,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[ceil_seg] {len(results)} fixture(s) → {seg_dir}/")
    print(f"[ceil_seg] Results → {results_path}")
    return seg_dir


def main():
    ap = argparse.ArgumentParser(
        description="Segment ceiling-mounted fixtures with Grounded-SAM."
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--types", nargs="+", default=None,
                    help=f"Restrict to these types: {list(_TYPE_GDINO)}")
    args = ap.parse_args()
    run(output_dir=args.output_dir, image_path=args.image, types=args.types)


if __name__ == "__main__":
    main()
