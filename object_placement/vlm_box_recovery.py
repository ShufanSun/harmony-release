"""vlm_box_recovery.py — recover objects the box detector missed, by asking the
VLM to LOCALISE them and feeding its boxes to the existing SAM2 stage.

WHY THIS EXISTS.  LocateAnything grounds a fixed list of type names; a type it
cannot localise verbatim yields ``<box>None</box>`` and lands in ``absent`` with
no trace.  Two scenes in the July set lost everything that way — living_room8
and pexels_12881062 both came back ``located 0; absent [window, door, curtain,
painting, ...]`` for rooms containing an obvious window and a wall-sized
painting.  Downstream, no window meant no light source, which meant Blender
rendered with ``lights: []`` and shipped a black image.

The VLM can see those objects perfectly well.  Asked to list the furniture in
elegant's reference photo it volunteered "Round side table in foreground with
lamp — not against a wall", an object the detector never found.  So the
localisation signal already exists upstream of the model that is missing it.

WHY BOXES, NOT TYPES.  ``missing_items.place_missing_items`` already asks the
VLM what is missing — but it returns only {type, description} and then re-runs
GroundingDINO on the full image, i.e. it asks the detector that just failed to
try again.  SAM2 is *promptable by box*: given a rectangle it segments whatever
is inside without detecting anything.  Passing the VLM's box therefore skips the
failing step entirely.  (That stage is also decoration-only by prompt — it
refuses to report furniture or wall art — so it could not have recovered either
of the objects above.)

Emitting LA's exact ``_la_boxes.json`` schema means the whole downstream chain
(``sam2_from_boxes.py --contract`` → masks → crops → inpaint → Hunyuan) runs
unchanged; this module is simply an alternative box source.

Usage:
    from object_placement import vlm_box_recovery as vbr
    vbr.recover(out_dir, image_path, domain="wall")     # writes _la_boxes.json
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw

from object_placement.vlm_backend import vlm_post

# What each domain is allowed to recover.  Deliberately broad — the point is to
# catch what the grounding detector dropped, and a wrong guess is filtered by
# the verify pass below rather than by a narrow allow-list.
_DOMAIN_HINT = {
    "wall": ("objects mounted on or set into a wall: paintings, framed art, "
             "posters, mirrors, windows, doors, curtains, wall lamps, "
             "sconces, shelves, televisions, clocks, radiators"),
    "furniture": ("free-standing furniture resting on the floor: sofas, "
                  "chairs, armchairs, stools, tables, coffee tables, side "
                  "tables, desks, cabinets, shelves, beds, fireplaces, "
                  "large floor plants"),
    "decoration": ("small objects resting on furniture: lamps, vases, books, "
                   "bowls, candles, plants, figurines, trays, cushions"),
}

_FIND_PROMPT = """\
IMAGE 1 is a photograph of a room.

An automatic object detector was run on it and reported these items as \
ABSENT — it could not locate them:
  {absent}

It DID find these (do not report them again):
  {found}

Your job: look at IMAGE 1 and find any {hint} that is CLEARLY VISIBLE in the \
photograph but is not in the found list. The detector is often wrong about \
"absent" — an object it could not name may still be plainly there.

The image is {w} x {h} pixels. For each object you can see, give a tight \
bounding box in PIXEL coordinates [x0, y0, x1, y1] with x0<x1, y0<y1, \
measured from the TOP-LEFT corner.

Report an object ONLY if you can actually see it and can point to where it \
is. Do not invent objects that would typically be in such a room. If you are \
not confident about an object, leave it out.

Return ONLY valid JSON, no markdown:
{{"found": [
   {{"label": "<short noun, 1-2 words>",
     "box_px": [x0, y0, x1, y1],
     "confidence": <0.0-1.0>,
     "why": "<what you see, one short clause>"}}
]}}
If there is genuinely nothing missing, return {{"found": []}}.
"""

_VERIFY_PROMPT = """\
IMAGE 1 is a room photo with {n} numbered red box(es) drawn on it.

For EACH numbered box, decide whether it tightly contains the object it is \
labelled with. Be strict: a box that is mostly wall, floor or a different \
object is WRONG.

Return ONLY valid JSON, no markdown:
{{"boxes": [
   {{"n": <number>, "ok": true|false,
     "box_px": [x0, y0, x1, y1],
     "note": "<short>"}}
]}}
Set "ok": false for a box that does not contain its object. When a box is \
merely loose or offset but the object IS visible near it, set "ok": true and \
give a corrected tighter "box_px". Omit "box_px" if the original is fine.
"""


def _enc(img: Image.Image, max_px: int = 1280) -> str:
    im = img.copy()
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _call_json(content: list, max_tokens: int = 1500) -> dict | None:
    try:
        r = vlm_post({"messages": [{"role": "user", "content": content}],
                      "max_tokens": max_tokens}, timeout=120)
        if r.status_code != 200:
            print(f"  [vbr] VLM HTTP {r.status_code}: {r.text[:140]}")
            return None
        raw = r.json()["choices"][0]["message"].get("content") or ""
    except Exception as e:
        print(f"  [vbr] VLM call failed: {type(e).__name__}: {e}")
        return None
    if not raw.strip():
        print("  [vbr] VLM returned empty content")
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        print(f"  [vbr] no JSON in reply: {raw[:140]!r}")
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print(f"  [vbr] bad JSON ({e}): {raw[:140]!r}")
        return None


def _sane_box(b, w: int, h: int, *, min_frac: float = 0.0006,
              max_frac: float = 0.85) -> list[int] | None:
    """Clamp to the image and reject degenerate / absurd boxes.

    A VLM asked for pixel coordinates will occasionally return normalised ones,
    a transposed pair, or the whole image.  None of those should reach SAM.
    """
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
    except Exception:
        return None
    # normalised (0-1) or 0-1000 grid → scale back to pixels
    if max(x0, y0, x1, y1) <= 1.0:
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h
    elif max(x0, y0, x1, y1) <= 1000 and max(w, h) > 1000:
        x0, x1 = x0 / 1000.0 * w, x1 / 1000.0 * w
        y0, y1 = y0 / 1000.0 * h, y1 / 1000.0 * h
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(int(round(x0)), w - 1))
    y0 = max(0, min(int(round(y0)), h - 1))
    x1 = max(0, min(int(round(x1)), w))
    y1 = max(0, min(int(round(y1)), h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    frac = ((x1 - x0) * (y1 - y0)) / float(w * h)
    if frac < min_frac or frac > max_frac:
        return None
    return [x0, y0, x1, y1]


def find(image_path: str | Path, *, domain: str = "wall",
         absent: list[str] | None = None, found: list[str] | None = None,
         min_conf: float = 0.5, verify: bool = True,
         debug_dir: str | Path | None = None) -> list[dict]:
    """Ask the VLM to localise objects the detector missed. Returns box dicts."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    hint = _DOMAIN_HINT.get(domain, _DOMAIN_HINT["furniture"])
    res = _call_json([
        {"type": "text", "text": _FIND_PROMPT.format(
            absent=", ".join(absent or []) or "(none listed)",
            found=", ".join(found or []) or "(nothing)",
            hint=hint, w=w, h=h)},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_enc(img)}"}},
    ])
    if not res:
        return []

    cands = []
    for it in res.get("found", []) or []:
        if not isinstance(it, dict):
            continue
        lab = str(it.get("label", "")).strip().lower()
        box = _sane_box(it.get("box_px"), w, h)
        conf = float(it.get("confidence", 0.0) or 0.0)
        if not lab or box is None:
            print(f"  [vbr] dropping '{lab or '?'}' — unusable box {it.get('box_px')}")
            continue
        if conf < min_conf:
            print(f"  [vbr] dropping '{lab}' — confidence {conf:.2f} < {min_conf}")
            continue
        cands.append({"label": lab, "box_px": box, "confidence": conf,
                      "why": str(it.get("why", ""))[:160]})
    if not cands:
        print("  [vbr] VLM localised nothing")
        return []
    print(f"  [vbr] VLM localised {len(cands)}: "
          + ", ".join(f"{c['label']}{c['box_px']}" for c in cands))
    if not verify:
        return cands

    # ── verify pass: draw the boxes and make the model look at its own work ──
    # A model asked for pixel coordinates from a description will place a box
    # roughly right and occasionally badly wrong.  Rendering them back is the
    # cheapest way to catch the bad ones before they reach SAM, which will
    # happily segment whatever rectangle it is handed.
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    lw = max(2, int(min(w, h) * 0.004))
    for i, c in enumerate(cands, 1):
        d.rectangle(c["box_px"], outline=(255, 0, 0), width=lw)
        d.text((c["box_px"][0] + 4, max(0, c["box_px"][1] - 18)),
               f"{i}:{c['label']}", fill=(255, 0, 0))
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        vis.save(Path(debug_dir) / f"_vbr_{domain}_boxes.png")
    ver = _call_json([
        {"type": "text", "text": _VERIFY_PROMPT.format(n=len(cands))},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_enc(vis)}"}},
    ])
    if not ver:
        print("  [vbr] verify unavailable — keeping unverified boxes")
        for c in cands:
            c["verdict"] = "UNVERIFIED"
        return cands

    by_n = {int(b["n"]): b for b in (ver.get("boxes") or [])
            if isinstance(b, dict) and str(b.get("n", "")).isdigit()}
    kept = []
    for i, c in enumerate(cands, 1):
        v = by_n.get(i)
        if v is None:
            c["verdict"] = "UNVERIFIED"
            kept.append(c)
            continue
        if not v.get("ok"):
            print(f"  [vbr] verify REJECTED {i}:{c['label']} — {v.get('note','')}")
            continue
        fixed = _sane_box(v.get("box_px"), w, h) if v.get("box_px") else None
        if fixed and fixed != c["box_px"]:
            print(f"  [vbr] verify tightened {i}:{c['label']} "
                  f"{c['box_px']} → {fixed}")
            c["box_px"] = fixed
        c["verdict"] = "YES"
        kept.append(c)
    print(f"  [vbr] {len(kept)}/{len(cands)} box(es) survived verification")
    return kept


def recover(out_dir: str | Path, image_path: str | Path, *, domain: str = "wall",
            merge: bool = True, min_conf: float = 0.5) -> Path | None:
    """Localise missed objects and MERGE them into the domain's _la_boxes.json.

    Writes LA's schema so `sam2_from_boxes.py --contract` consumes the result
    with no changes.  Returns the path written, or None when nothing was added.
    """
    out_dir = Path(out_dir)
    sub = {"wall": "wall_mounted", "furniture": "furniture",
           "decoration": "decorations"}[domain]
    dom_dir = out_dir / sub
    boxes_p = dom_dir / "_la_boxes.json"

    prior, absent, existing = {}, [], []
    if boxes_p.exists():
        try:
            prior = json.loads(boxes_p.read_text())
            absent = list(prior.get("absent") or [])
            existing = [b for b in (prior.get("boxes") or [])]
        except Exception as e:
            print(f"  [vbr] could not read {boxes_p}: {e}")
    found_labels = [str(b.get("label", "")) for b in existing]

    new = find(image_path, domain=domain, absent=absent, found=found_labels,
               min_conf=min_conf, debug_dir=dom_dir)
    if not new:
        return None

    # Drop anything overlapping a box the detector already has — the recovery
    # pass is for what is MISSING, not for re-proposing what was found.
    def _iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        bb = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(aa + bb - inter)

    merged = list(existing) if merge else []
    added = 0
    for c in new:
        if any(_iou(c["box_px"], b.get("box_px", [0, 0, 0, 0])) > 0.45
               for b in merged):
            print(f"  [vbr] skipping '{c['label']}' — overlaps an existing box")
            continue
        merged.append({
            "index": len(merged),
            "label": c["label"],
            "type": c["label"].replace(" ", "_"),
            "box_px": c["box_px"],
            "verdict": c.get("verdict", "YES"),
            "source": "vlm_box_recovery",
            "confidence": c.get("confidence"),
            "why": c.get("why"),
        })
        added += 1
    if not added:
        return None
    for i, b in enumerate(merged):
        b["index"] = i

    img = Image.open(image_path)
    payload = {
        "image": str(Path(image_path).resolve()),
        "image_size": list(img.size),
        "domain": domain,
        # Anything recovered is no longer absent.
        "absent": [a for a in absent
                   if a not in {b.get("label") for b in merged}],
        "boxes": merged,
    }
    dom_dir.mkdir(parents=True, exist_ok=True)
    if boxes_p.exists():
        boxes_p.replace(boxes_p.with_suffix(".json.pre_vbr"))
    boxes_p.write_text(json.dumps(payload, indent=2))
    print(f"[vbr] {domain}: recovered {added} object(s) → {boxes_p} "
          f"({len(merged)} total)")
    return boxes_p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--domain", default="wall",
                    choices=["wall", "furniture", "decoration"])
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true",
                    help="localise and print, write nothing")
    a = ap.parse_args()
    if a.dry_run:
        for c in find(a.image, domain=a.domain, debug_dir=a.output_dir,
                      min_conf=a.min_conf):
            print("  ", json.dumps(c))
    else:
        recover(a.output_dir, a.image, domain=a.domain, min_conf=a.min_conf)
