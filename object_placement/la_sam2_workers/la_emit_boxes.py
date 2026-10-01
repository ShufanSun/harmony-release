#!/usr/bin/env python
"""
la_emit_boxes.py — Stage 1 of LocateAnything→SAM2.

Runs LocateAnything detection (+ optional gpt-5.5 wall/furniture verify) and
writes the kept boxes to a JSON for SAM2 to mask in the `lang-sam` env.
Run in `eagle-embodied`.

    python la_emit_boxes.py --domain wall|furniture --image IMG --out BOXES.json

Model: NVIDIA LocateAnything-3B (https://huggingface.co/nvidia/LocateAnything-3B,
NVIDIA model license). It downloads into the HF cache on first run. To fetch it
ahead of time, e.g. on a compute node with no internet access:
    hf download nvidia/LocateAnything-3B --local-dir LocateAnything-3B
    export SCENEWEAVE_LA_MODEL=/path/to/LocateAnything-3B   # → --model
The inference wrapper (locateanything_worker.py) comes from the Embodied/ folder
of https://github.com/NVlabs/Eagle. Set SCENEWEAVE_EAGLE_EMBODIED_DIR to that
folder, and build the eagle-embodied env from its README (`pip install -e .`,
which pins transformers==4.57.1).

Swapping the detector: any model works if this script still writes the same
boxes.json, i.e. {"image", "image_size", "domain", "absent", "boxes": [{"index",
"label", "type", "box_px": [x1, y1, x2, y2], "verdict"}]} in full-resolution
pixel coordinates. Replace the `L.LocateAnythingWorker(...)` / `worker.detect`
calls; the stages after it only read boxes.json.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la_pipeline_segment as L  # reuse LA detect / parse / verify / type map


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1)


def _dedup_boxes(boxes, iou_thr=0.6):
    """Collapse near-duplicate detections (e.g. the same pillow matched by both
    'throw pillow' and 'throw pillows') — keep the first of each overlap group."""
    kept = []
    for b in boxes:
        if any(_iou(b["box_px"], k["box_px"]) > iou_thr for k in kept):
            continue
        kept.append(b)
    return kept


def _base_overlap(inner, outer):
    """Fraction of the inner box's BOTTOM strip area that lies inside outer —
    decorations rest on a furniture surface, so the base is the key contact."""
    ix1, iy1, ix2, iy2 = inner
    h = max(1, iy2 - iy1)
    by1 = iy2 - int(0.35 * h)           # bottom 35% strip
    bx1, bz1, bx2, bz2 = ix1, by1, ix2, iy2
    ox1, oy1, ox2, oy2 = outer
    ax1, ay1 = max(bx1, ox1), max(bz1, oy1)
    ax2, ay2 = min(bx2, ox2), min(bz2, oy2)
    if ax2 <= ax1 or ay2 <= ay1:
        return 0.0
    inter = (ax2 - ax1) * (ay2 - ay1)
    area = (bx2 - bx1) * (bz2 - bz1)
    return inter / max(area, 1)


def _load_decor_types(analysis_path):
    """From decoration_analysis.json → (sorted decor names, furniture list)."""
    data = json.load(open(analysis_path))
    names, furn = set(), []
    for fe in data.get("furniture_decorations", []):
        furn.append({"furniture_index": fe.get("furniture_index"),
                     "furniture_type": fe.get("furniture_type"),
                     "box_px": fe.get("box_px")})
        for md in fe.get("missing_decorations", []):
            n = (md.get("name") or "").lower().strip()
            if n:
                names.add(n)
    return sorted(names), furn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["wall", "furniture", "decoration"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output boxes.json path")
    ap.add_argument("--analysis", default=None,
                    help="decoration_analysis.json (required for --domain decoration)")
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-side", type=int, default=1536,
                    help="downscale longest image side for LA detection to cap "
                         "ViT attention memory (boxes are resolution-independent; "
                         "0 disables). 1536 keeps small-object recall while staying "
                         "well under the full-res OOM. Verify + SAM2 use full-res coords.")
    args = ap.parse_args()

    is_decor = args.domain == "decoration"
    furn = []
    if is_decor:
        if not args.analysis:
            ap.error("--domain decoration requires --analysis")
        types, furn = _load_decor_types(args.analysis)
        if not types:
            print("[la-boxes:decoration] no missing decorations in analysis — nothing to do")
            json.dump({"image": os.path.abspath(args.image), "domain": "decoration",
                       "boxes": []}, open(args.out, "w"), indent=2)
            return
    else:
        types = list(L._WALL_TYPES if args.domain == "wall" else L._FURN_TYPES)

    img = Image.open(args.image).convert("RGB")
    image_np = np.array(img); w, h = img.size

    # Open-vocabulary discovery: ask the VLM what object types are actually in the
    # photo and auto-include any not in the base list, so new furniture (side
    # table, ottoman, console, bench, …) isn't missed just for not being listed.
    if not is_decor:
        extra = [t for t in L._discover_types(image_np, args.domain) if t not in types]
        if extra:
            types += extra
            print(f"[la-boxes:{args.domain}] VLM auto-included types: {extra}")

    # ── drop structural surfaces from the query list ─────────────────────────
    # A multi-word type can come back split into its separate words, each
    # grounded on its own (the decoration path documents this: "desk lamp" →
    # "desk" + "lamp", and the bare "desk" then grounded onto the actual desk).
    # In the wall domain the same split turned "wall lamp" into a bare "wall"
    # query, LocateAnything grounded it on the literal wall, and verification
    # passed it: h300_medium_office_01 shipped a wall-mounted GLB of a wall,
    # box [0,0,1878,872] of an 1880x1057 image, while every real wall object
    # was reported absent.  A wall/ceiling/floor can never be a wall-MOUNTED
    # object, so never ask for one.
    _STRUCTURAL = {"wall", "ceiling", "floor", "walls", "room"}
    _dropped = [t for t in types if t.lower().strip() in _STRUCTURAL]
    if _dropped:
        types = [t for t in types if t.lower().strip() not in _STRUCTURAL]
        print(f"[la-boxes:{args.domain}] dropped structural query term(s) "
              f"{_dropped} — a surface is not an object on that surface")

    # Run LocateAnything on a downscaled copy so the ViT's O(N^2) full-attention
    # over image patches doesn't OOM on large photos. Output coords are
    # normalized 0-1000, so they map straight back to the full-res (w, h).
    det_img = img
    if args.max_side and max(w, h) > args.max_side:
        s = args.max_side / max(w, h)
        det_img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    print(f"[la-boxes:{args.domain}] image {w}x{h} (detect@{det_img.size}); types={types}")

    worker = L.LocateAnythingWorker(args.model, device=args.device)
    # SCENEWEAVE_LA_OPEN_DETECT=1 — open detection instead of grounding the fixed
    # type list. Grounding a named type that LA can't localize just yields
    # <box>None</box> and the object is silently dropped (e.g. a heavily-occluded
    # bench). Asking LA to locate ANY furniture/object lets it surface whatever it
    # sees; the downstream VLM-verify filter then decides furniture-or-not.
    if os.environ.get("SCENEWEAVE_LA_OPEN_DETECT"):
        _desc = ("any piece of furniture or freestanding object in the room, "
                 "including any partially hidden, clipped or occluded item")
        print(f"[la-boxes:{args.domain}] OPEN-DETECT: {_desc!r}")
        answer = worker.detect(det_img, [_desc], generation_mode="hybrid",
                               temperature=args.temperature, verbose=False)["answer"]
    elif is_decor and os.environ.get("SCENEWEAVE_DECO_OPEN", "1") == "1":
        # Grounding ONLY the analysis's exact phrases is brittle: a name the
        # model cannot localize verbatim yields <box>None</box> and the object
        # vanishes without a trace, while a multi-word name can be split and
        # each word grounded separately.  office8 asked for "computer monitor"
        # and "desk lamp" and got back 'desk' + 'lamp' — the monitor silently
        # lost, and the stray "desk" grounded onto the actual desk.
        #
        # Widen the vocabulary, but keep LA's EXPECTED INPUT SHAPE: a list of
        # short noun phrases.  A long descriptive sentence does not work —
        # LA returned `located 0` and echoed the whole sentence back as one
        # absent label.  So: the requested names PLUS a generic decoration
        # vocabulary, each a couple of words.  _sem_match() below maps whatever
        # comes back onto the requested concepts, and unmatched-but-real items
        # are kept as discovered decorations.
        _GENERIC = ["monitor", "screen", "laptop", "lamp", "books", "book",
                    "pillow", "cushion", "vase", "plant", "tray", "bowl",
                    "cup", "mug", "candle", "clock", "picture frame",
                    "basket", "blanket", "ornament", "bottle", "box"]
        _seen, types_open = set(), []
        for _t in list(types) + _GENERIC:
            _tl = _t.lower().strip()
            if _tl and _tl not in _seen:
                _seen.add(_tl); types_open.append(_tl)
        print(f"[la-boxes:decoration] OPEN-VOCAB ({len(types_open)} terms): {types_open}")
        answer = worker.detect(det_img, types_open, generation_mode="hybrid",
                               temperature=args.temperature, verbose=False)["answer"]
    else:
        answer = worker.detect(det_img, types, generation_mode="hybrid",
                               temperature=args.temperature, verbose=False)["answer"]
    boxes, absent = L._parse_labeled_boxes(answer, w, h)
    print(f"[la-boxes] located {len(boxes)}; absent {absent or '—'}")

    if is_decor:
        # ── Reconcile labels against what was actually REQUESTED ──────────
        # A multi-word decoration name can come back split into its separate
        # words, each grounded on its own.  office8 asked for
        #   ['books', 'computer monitor', 'desk lamp', 'throw pillow']
        # and got back ['books', 'desk', 'lamp', 'throw_pillow']: "desk lamp"
        # was split, and the bare word "desk" was then grounded onto the
        # actual DESK — producing a "decoration" whose box equalled the desk's
        # own bbox, which downstream inpainted and meshed a second desk and
        # placed it on top of the first.  Meanwhile the real monitor was never
        # found and, because it emitted no <ref>, never reached `absent`
        # either, so nothing reported it missing.
        _req = {t.lower().strip() for t in types}

        _STOP = {"a", "an", "the", "of", "with", "small", "large", "some"}
        # Light synonym groups so a detector's word choice ("screen", "rug")
        # still lands on the requested concept ("computer monitor", "carpet").
        # NOT an exhaustive vocabulary — unmatched labels are KEPT, not dropped.
        _SYN = [
            {"monitor", "screen", "display", "computer", "imac", "tv"},
            {"lamp", "light", "lighting"},
            {"book", "books", "magazine", "magazines"},
            {"pillow", "cushion", "throw"},
            {"plant", "pot", "planter", "vase"},
            {"rug", "carpet", "mat"},
        ]

        def _toks(s):
            return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w and w not in _STOP}

        def _sem_match(lab):
            """Best requested name for `lab`, or None. Token overlap + synonyms."""
            lt = _toks(lab)
            if not lt:
                return None
            best, best_s = None, 0.0
            for t in _req:
                tt = _toks(t)
                if not tt:
                    continue
                inter = len(lt & tt)
                # synonym credit: any group containing a word from each side
                syn = sum(1 for g in _SYN if (lt & g) and (tt & g))
                score = (inter + syn) / max(len(tt), 1)
                if score > best_s:
                    best, best_s = t, score
            return best if best_s >= 0.5 else None

        _kept = []
        for _b in boxes:
            _lab = (_b.get("label") or "").lower().strip()
            # Reject malformed labels before anything else.  LA's <ref> parsing
            # occasionally emits token garbage ('<523>ush<587>' came back on
            # living_room004); previously such labels were dropped simply for
            # not being requested, but now that unmatched labels are KEPT as
            # discovered decorations, junk would reach meshing.
            if (not _lab or "<" in _lab or ">" in _lab
                    or not re.search(r"[a-z]{3}", _lab)):
                print(f"   [label_map] {_lab!r} malformed → DROP")
                continue
            if _lab in _req:
                _kept.append(_b); continue
            _m = _sem_match(_lab)
            if _m:
                print(f"   [label_map] {_lab!r} → {_m!r} (semantic match)")
                _b["label"] = _m
            else:
                # Open vocabulary: an object the analysis never thought to ask
                # for is still a real object.  Keep it under the detector's own
                # label — the furniture-overlap, self-IoU and downstream VLM
                # verify decide whether it survives.
                print(f"   [label_map] {_lab!r} kept as discovered decoration")
            _kept.append(_b)
        boxes = _kept
        # anything requested that produced no box at all is genuinely missing
        _found = {(b.get("label") or "").lower().strip() for b in boxes}
        for _t in sorted(_req - _found):
            if _t not in absent:
                absent.append(_t)
                print(f"   [label_fix] {_t!r} never localized → recorded ABSENT")
        # Collapse near-duplicate boxes first — the analysis often lists the same
        # item under singular+plural names (throw pillow / throw pillows), so LA
        # detects each physical instance multiple times with identical boxes.
        n_before = len(boxes)
        boxes = _dedup_boxes(boxes)
        print(f"[la-boxes:decoration] deduped {n_before} → {len(boxes)} boxes")
        # Assign each decoration box to the furniture piece whose box its base
        # sits inside (max base-overlap). Drop boxes that don't rest on any.
        segs = []
        for b in boxes:
            best_f, best_ov = None, 0.0
            for fe in furn:
                if not fe.get("box_px"):
                    continue
                ov = _base_overlap(b["box_px"], fe["box_px"])
                if ov > best_ov:
                    best_ov, best_f = ov, fe
            _ov_min = float(__import__("os").environ.get("SCENEWEAVE_DECO_OV_MIN", "0.15"))
            if best_f is not None and best_ov < _ov_min:
                # Back-of-surface rescue.  _base_overlap asks how much of the
                # decoration's bottom strip falls INSIDE the furniture bbox.
                # An object standing at the BACK of a surface fails that under
                # perspective: its base projects at or just above the surface's
                # top edge.  office8's monitor sits against the wall at the back
                # of the desk — desk bbox top y=1248, monitor bottom y≈1260 —
                # scoring ov=0.11 and being dropped, even though it is plainly
                # on the desk.  Accept when the decoration is horizontally well
                # inside the furniture and its base sits near that top edge.
                _bx, _fx = b["box_px"], best_f["box_px"]
                _fw = max(1, _fx[2] - _fx[0])
                _hx1, _hx2 = max(_bx[0], _fx[0]), min(_bx[2], _fx[2])
                _hcov = max(0, _hx2 - _hx1) / max(1, _bx[2] - _bx[0])
                _fh = max(1, _fx[3] - _fx[1])
                _tol = float(__import__("os").environ.get("SCENEWEAVE_DECO_BACKLINE", "0.25")) * _fh
                _base_y = _bx[3]
                if _hcov >= 0.6 and (_fx[1] - _tol) <= _base_y <= _fx[3]:
                    print(f"   {b['label']:22s} → {best_f['furniture_type']}"
                          f"#{best_f['furniture_index']} (back-of-surface rescue: "
                          f"ov={best_ov:.2f} but {_hcov:.0%} horizontally inside and "
                          f"base y={_base_y} at its top edge {_fx[1]})")
                    best_ov = _ov_min      # accept
            if best_f is None or best_ov < _ov_min:
                print(f"   {b['label']:22s} no-furniture (ov={best_ov:.2f}) → DROP")
                continue
            # A decoration sits ON a piece of furniture — it can never BE that
            # piece.  When the detector grounds a stray word onto the support
            # itself the two boxes come back near-identical (office8: the
            # spurious "desk" box was [1470,1246,2520,1899] against the desk's
            # own [1470,1248,2520,1899]).  Left in, it gets inpainted and
            # meshed as a duplicate of the furniture and then placed on top of
            # it.  Reject on IoU with the assigned support.
            # Compare against EVERY furniture box, not just the assigned support.
            # Two distinct failures need this:
            #   * the detector grounds a stray word onto the support itself
            #     (office8: a "desk" box identical to desk#3), and
            #   * open-vocabulary detection re-finds a piece that is ALREADY
            #     furniture and offers it as a decoration (office8: the floor
            #     "plant", which is furniture#4, proposed as a decoration to sit
            #     on the office chair).  That one is invisible to a
            #     support-only check, because the duplicate's support is a
            #     different object than the piece it duplicates.
            _IOU_MAX = float(__import__("os").environ.get("SCENEWEAVE_DECO_SELF_IOU", "0.85"))
            _bx = b["box_px"]
            _dup = None
            for _fe in furn:
                _fx = _fe.get("box_px")
                if not _fx:
                    continue
                _ix = max(0, min(_bx[2], _fx[2]) - max(_bx[0], _fx[0]))
                _iy = max(0, min(_bx[3], _fx[3]) - max(_bx[1], _fx[1]))
                _inter = _ix * _iy
                _union = ((_bx[2]-_bx[0])*(_bx[3]-_bx[1])
                          + (_fx[2]-_fx[0])*(_fx[3]-_fx[1]) - _inter)
                _iou = _inter / _union if _union > 0 else 0.0
                if _iou > _IOU_MAX:
                    _dup = (_fe, _iou)
                    break
            if _dup is not None:
                _fe, _iou = _dup
                _same = _fe.get("furniture_index") == best_f.get("furniture_index")
                print(f"   {b['label']:22s} duplicates "
                      f"{'its support ' if _same else 'existing furniture '}"
                      f"{_fe['furniture_type']}#{_fe['furniture_index']} "
                      f"(IoU={_iou:.2f} > {_IOU_MAX}) → DROP")
                continue
            print(f"   {b['label']:22s} → {best_f['furniture_type']}#{best_f['furniture_index']} (ov={best_ov:.2f})")
            segs.append({"index": len(segs), "label": b["label"], "phrase": b["label"],
                         "type": L._canon_type(b["label"]), "box_px": b["box_px"],
                         "furniture_index": best_f["furniture_index"],
                         "furniture_type": best_f["furniture_type"], "verdict": "OVERLAP"})
        # ── Budget ────────────────────────────────────────────────────────
        # Widening the vocabulary makes the detector generous: 'elegant' asked
        # for 9 decorations and kept 24, the extras being generic terms
        # (bowl/cup/mug/candle/basket/bottle/box) all landing on the fireplace
        # mantel.  Some may be real — a styled mantel does hold a candle and a
        # bowl — but every extra costs a Hunyuan mesh and adds clutter, and
        # unlike the wall/furniture domains there is no VLM verify filter here
        # to reject them.  So bound it rather than guess: keep the names the
        # analysis actually ASKED for first, then fill the remaining budget
        # with discovered items (largest first, as a crude salience proxy).
        _MAX = int(__import__("os").environ.get("SCENEWEAVE_DECO_MAX", "12"))
        if len(segs) > _MAX:
            _reqset = {t.lower().strip() for t in types}
            def _area(s):
                b = s["box_px"]; return (b[2]-b[0]) * (b[3]-b[1])
            # Round-robin by NAME, not a flat sort: a flat sort spent the whole
            # budget on five copies of "vase with flowers" and dropped distinct
            # requested items (throw pillow, candlestick).  Give every name its
            # first instance before any name gets a second, requested names
            # ahead of discovered ones.
            _by_name = {}
            for _s in sorted(segs, key=_area, reverse=True):
                _by_name.setdefault((_s["label"] or "").lower().strip(), []).append(_s)
            _names = ([n for n in _by_name if n in _reqset]
                      + [n for n in _by_name if n not in _reqset])
            _keep, _round = [], 0
            while len(_keep) < _MAX and any(len(v) > _round for v in _by_name.values()):
                for _n in _names:
                    if len(_keep) >= _MAX:
                        break
                    if len(_by_name[_n]) > _round:
                        _keep.append(_by_name[_n][_round])
                _round += 1
            _cut = [s["label"] for s in segs if s not in _keep]
            _n_req = sum(1 for s in _keep
                         if (s["label"] or "").lower().strip() in _reqset)
            print(f"[la-boxes:decoration] budget {len(segs)} → {len(_keep)} "
                  f"(max {_MAX}; {_n_req} requested, {len(_keep) - _n_req} "
                  f"discovered); dropped: {_cut}")
            segs = _keep
            for _i, _s in enumerate(segs):
                _s["index"] = _i
        out = {"image": os.path.abspath(args.image), "image_size": [w, h],
               "domain": "decoration", "absent": absent, "boxes": segs}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[la-boxes:decoration] wrote {len(segs)} boxes → {args.out}")
        return

    # Collapse pixel-aligned duplicates: LocateAnything detects the SAME physical
    # object under multiple synonymous types (chair/armchair, carpet/rug,
    # side table/cabinet), yielding near-identical boxes. A high IoU between two
    # boxes ⇒ the same thing ⇒ keep one (the first / most-canonical label) so we
    # don't place duplicate objects. Distinct furniture rarely exceeds this IoU.
    n_before = len(boxes)
    boxes = _dedup_boxes(boxes, iou_thr=0.7)
    if len(boxes) != n_before:
        print(f"[la-boxes:{args.domain}] deduped {n_before} → {len(boxes)} "
              f"boxes (same-object synonyms)")

    # Belt-and-braces for the same failure: even with the query list cleaned,
    # a label can still come back naming a surface, or a box can swallow the
    # frame.  Nothing mounted on a wall covers most of the photo, so a box that
    # does is the wall (or the room), not an object on it.  This runs BEFORE
    # the VLM verify because verify passed the wall box in office_01.
    _AREA_MAX = float(os.environ.get("SCENEWEAVE_WALL_BOX_AREA_MAX", "0.40"))
    if args.domain == "wall":
        _kept = []
        for b in boxes:
            x1, y1, x2, y2 = b["box_px"]
            frac = abs((x2 - x1) * (y2 - y1)) / float(max(w * h, 1))
            lab = str(b.get("label", "")).lower().strip()
            if lab in _STRUCTURAL:
                print(f"   {lab:22s} {'STRUCTURAL':14s} DROP  (a surface, not an object)")
                continue
            if frac > _AREA_MAX:
                print(f"   {lab:22s} {'OVERSIZE':14s} DROP  "
                      f"({frac:.0%} of frame > {_AREA_MAX:.0%})")
                continue
            _kept.append(b)
        if len(_kept) != len(boxes):
            print(f"[la-boxes:wall] sanity gate: {len(boxes)} → {len(_kept)} box(es)")
        boxes = _kept

    segs = []
    for b in boxes:
        if args.no_verify:
            verdict, keep = "SKIP", True
        else:
            keep, verdict = L._verify(image_np, b["box_px"], b["label"], args.domain)
        print(f"   {b['label']:22s} {verdict:14s} {'keep' if keep else 'DROP'}")
        if keep:
            segs.append({"index": len(segs), "label": b["label"],
                         "type": L._canon_type(b["label"]),
                         "box_px": [int(v) for v in b["box_px"]], "verdict": verdict})

    out = {"image": os.path.abspath(args.image), "image_size": [w, h],
           "domain": args.domain, "absent": absent, "boxes": segs}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[la-boxes:{args.domain}] wrote {len(segs)} boxes → {args.out}")


if __name__ == "__main__":
    main()
