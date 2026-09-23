"""geometry_audit.py — quantitative scene audit + corrective resolution, and the
top-down orthographic plan render.

Two jobs, kept together because the plan render is how a human (or a VLM) reads
the audit's findings:

  audit(placements, room_w, room_d)   → findings + in-place corrections
  render_plan(...)                    → top-down PNG with oriented footprints

WHY ORIENTED FOOTPRINTS.  seating_group's collision pass works on the rotated
AABB, which over-states a 45°-yawed object's footprint by up to 41% and cannot
tell "flush against the wall" from "corner poking through it".  A desk whose
long axis runs along a wall and one rotated 90° have identical AABBs only when
square, so AABB-only reasoning silently accepts the wrong one.  Everything here
uses the true rectangle.

WHY CORRECTIVE, NOT ADVISORY.  The existing `⚠ SIZE-SUSPECT` print reports a
67 cm overlap and then does nothing with it.  A finding that nobody acts on is
a finding that ships.  Every finding here carries the metres it is off by and,
where a safe move exists, is resolved before the placements are saved.

Convention (shared with seating_group._aabb): with
``ang = arctan2(fr[0], fr[2])`` for ``fr = R @ _local_front``, the object's
local +Z (depth/front axis) maps to world ``(sin ang, cos ang)`` in XZ and its
local +X (width axis) to ``(cos ang, -sin ang)``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Who yields to whom when two footprints overlap: the LOWER priority moves.
# Mirrors seating_group._PRIORITY so the two passes never fight each other.
_PRIORITY = {
    "sofa": 5, "sectional": 5, "bed": 5,
    "bookcase": 4, "cabinet": 4, "shelf": 4, "wardrobe": 4, "dresser": 4,
    "fireplace": 4,
    "desk": 3, "dining_table": 3,
    "coffee_table": 2, "side_table": 2, "end_table": 2, "nightstand": 2,
    "chair": 1, "armchair": 1, "stool": 1, "ottoman": 1, "plant": 1,
}

# An overlap below this is contact, not interpenetration (a chair tucked under a
# desk legitimately shares floor area).
_OVERLAP_TOL_M2 = 0.02
# Wall penetration below this is within the thickness of a baseboard.
_WALL_TOL_M = 0.02
# Leave this much air after separating two pieces.
_MARGIN_M = 0.04


def _prio(o) -> int:
    return _PRIORITY.get(str(o.get("type", "")).lower(), 2)


def _yaw(o) -> float:
    """World yaw of the object's front axis, matching seating_group._aabb."""
    R = np.asarray(o.get("rotation_3x3", np.eye(3).tolist()), float)
    fr = R @ np.asarray(o.get("_local_front", [0, 0, 1]), float)
    return float(np.arctan2(fr[0], fr[2]))


def footprint(o) -> np.ndarray | None:
    """The object's true floor rectangle as 4 world-XZ corners, or None."""
    if "position_m" not in o:
        return None
    s = o.get("size_m") or {}
    w = float(s.get("width_m", 0.0))
    d = float(s.get("depth_m", 0.0))
    if w <= 0 or d <= 0:
        return None
    ang = _yaw(o)
    ex = np.array([math.cos(ang), -math.sin(ang)])   # local +X (width)
    ez = np.array([math.sin(ang), math.cos(ang)])    # local +Z (depth/front)
    c = np.array([float(o["position_m"][0]), float(o["position_m"][2])])
    return np.array([c - ex * w / 2 - ez * d / 2,
                     c + ex * w / 2 - ez * d / 2,
                     c + ex * w / 2 + ez * d / 2,
                     c - ex * w / 2 + ez * d / 2])


def facing(o) -> np.ndarray:
    """Unit world-XZ vector the object's front faces."""
    ang = _yaw(o)
    return np.array([math.sin(ang), math.cos(ang)])


# ── separating-axis test for two convex rectangles ──────────────────────────

def _mtv(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Minimum translation vector separating rect A from rect B.

    Returns (unit axis pointing away from B, depth in metres), or None when the
    two do not overlap.  Only the 4 edge normals need testing for rectangles.
    """
    best_axis, best_depth = None, float("inf")
    for rect in (A, B):
        for i in range(4):
            edge = rect[(i + 1) % 4] - rect[i]
            n = np.array([-edge[1], edge[0]])
            ln = np.linalg.norm(n)
            if ln < 1e-9:
                continue
            n = n / ln
            pa, pb = A @ n, B @ n
            overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if overlap <= 0:
                return None                      # separating axis found
            if overlap < best_depth:
                best_depth = overlap
                # orient the axis so it pushes A away from B
                best_axis = n if A.mean(0) @ n >= B.mean(0) @ n else -n
    if best_axis is None:
        return None
    return best_axis, float(best_depth)


def _overlap_area(A: np.ndarray, B: np.ndarray) -> float:
    """Intersection area of two convex rects (Sutherland–Hodgman clip)."""
    poly = [p for p in A]
    for i in range(4):
        a, b = B[i], B[(i + 1) % 4]
        edge = b - a
        nrm = np.array([-edge[1], edge[0]])
        # B's corners are ordered consistently, so "inside" is a fixed sign
        inside = lambda p: (p - a) @ nrm >= -1e-12
        clipped, n = [], len(poly)
        for j in range(n):
            cur, prv = poly[j], poly[(j - 1) % n]
            if inside(cur):
                if not inside(prv):
                    clipped.append(_isect(prv, cur, a, b))
                clipped.append(cur)
            elif inside(prv):
                clipped.append(_isect(prv, cur, a, b))
        poly = clipped
        if not poly:
            return 0.0
    area = 0.0
    for i in range(len(poly)):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % len(poly)]
        area += x0 * z1 - x1 * z0
    return abs(area) / 2.0


def _isect(p0, p1, a, b):
    d1, d2 = p1 - p0, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return p1
    t = ((a[0] - p0[0]) * d2[1] - (a[1] - p0[1]) * d2[0]) / den
    return p0 + t * d1


# ── the audit ───────────────────────────────────────────────────────────────

def _wall_axis(o) -> int | None:
    """Axis index (0=X, 1=Z) the object is pinned along by its wall, or None.

    A back-wall object may slide in X but must keep its Z; a side-wall object
    the reverse.  Respecting this is what keeps a corrective push from sliding
    a sofa off the wall it is supposed to back onto.
    """
    wa = str(o.get("wall_affinity") or "centre").lower()
    if wa in ("back", "front"):
        return 1
    if wa in ("left", "right"):
        return 0
    return None


def audit(placements: list[dict], room_w: float, room_d: float,
          *, correct: bool = True, out_path: str | Path | None = None) -> dict:
    """Audit floor geometry; optionally fix what is safely fixable, in place.

    Findings, each with the metres it is off by:
      wall_penetration — footprint crosses a room boundary
      overlap          — two footprints interpenetrate
      off_floor        — a floor object floating above / sunk below y=0

    Returns the audit dict; also writes it to `out_path` when given.
    """
    objs = [o for o in placements
            if not o.get("is_carpet") and footprint(o) is not None]
    findings: list[dict] = []
    corrections: list[dict] = []

    # Anything with no usable floor footprint is REPORTED, never silently
    # dropped: that is how living_room004's glass side table disappeared — the
    # pipeline reclassified it wall_mounted with depth 1.7 cm and no
    # position_m, so every floor-space pass skipped it and nothing said so.
    for o in placements:
        if o.get("is_carpet") or footprint(o) is not None:
            continue
        sz = o.get("size_m") or {}
        findings.append({
            "kind": "no_floor_footprint", "index": o.get("index"),
            "type": o.get("type"),
            "reason": ("no position_m" if "position_m" not in o
                       else "zero/absent size_m"),
            "wall_mounted": bool(o.get("wall_mounted")),
            "size_m": sz,
        })
    # A footprint this thin is a collapsed object, not a thin one.
    for o in objs:
        sz = o.get("size_m") or {}
        thin = min(float(sz.get("width_m", 1)), float(sz.get("depth_m", 1)))
        if thin < 0.05:
            findings.append({"kind": "degenerate_footprint",
                             "index": o.get("index"), "type": o.get("type"),
                             "min_extent_m": round(thin, 4)})

    # 1. Room containment ---------------------------------------------------
    for o in objs:
        fp = footprint(o)
        lo, hi = fp.min(axis=0), fp.max(axis=0)
        pen = [max(0.0, -lo[0]), max(0.0, hi[0] - room_w),
               max(0.0, -lo[1]), max(0.0, hi[1] - room_d)]
        worst = max(pen)
        if worst <= _WALL_TOL_M:
            continue
        wall = ("left", "right", "back", "front")[int(np.argmax(pen))]
        f = {"kind": "wall_penetration", "index": o.get("index"),
             "type": o.get("type"), "wall": wall,
             "penetration_m": round(float(worst), 4)}
        findings.append(f)
        if not correct:
            continue
        # Push straight in along the violated axis. Never widen the object —
        # a piece genuinely larger than the room is a sizing bug, not a
        # placement bug, and silently shrinking it would hide that.
        span_x, span_z = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        p = o["position_m"]
        before = [round(float(p[0]), 4), round(float(p[2]), 4)]
        # float() throughout: a numpy scalar written into position_m propagates
        # into furniture_placements.json, which json.dump cannot serialise.
        if span_x <= room_w:
            p[0] = float(p[0]) + float(max(0.0, -lo[0]) - max(0.0, hi[0] - room_w))
        if span_z <= room_d:
            p[2] = float(p[2]) + float(max(0.0, -lo[1]) - max(0.0, hi[1] - room_d))
        after = [round(float(p[0]), 4), round(float(p[2]), 4)]
        if before != after:
            corrections.append({"index": o.get("index"), "type": o.get("type"),
                                "reason": f"{wall}-wall penetration "
                                          f"{float(worst) * 100:.1f} cm",
                                "from_xz": before, "to_xz": after})
            f["corrected"] = True

    # 2. Pairwise interpenetration -----------------------------------------
    for _pass in range(3):                     # a push can create a new overlap
        moved = False
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                fa, fb = footprint(a), footprint(b)
                area = _overlap_area(fa, fb)
                if area <= _OVERLAP_TOL_M2:
                    continue
                res = _mtv(fa, fb)
                if res is None:
                    continue
                axis, depth = res
                # The lower-priority piece yields; ties go to the smaller one.
                if _prio(a) == _prio(b):
                    sa = float((fa.max(0) - fa.min(0)).prod())
                    sb = float((fb.max(0) - fb.min(0)).prod())
                    mover, other = (a, b) if sa <= sb else (b, a)
                elif _prio(a) < _prio(b):
                    mover, other = a, b
                else:
                    mover, other = b, a
                sign = 1.0 if mover is a else -1.0
                push = axis * sign * (depth + _MARGIN_M)
                if _pass == 0:
                    findings.append({
                        "kind": "overlap", "index_a": a.get("index"),
                        "index_b": b.get("index"),
                        "type_a": a.get("type"), "type_b": b.get("type"),
                        "overlap_area_m2": round(float(area), 4),
                        "penetration_m": round(float(depth), 4),
                    })
                if not correct:
                    continue
                # Keep a wall-flush piece on its wall: zero the push component
                # along the wall normal so it can only slide sideways.
                pinned = _wall_axis(mover)
                if pinned is not None:
                    push[pinned] = 0.0
                    if abs(push[1 - pinned]) < 1e-4:
                        continue               # nowhere legal to go — leave it
                p = mover["position_m"]
                before = [round(float(p[0]), 4), round(float(p[2]), 4)]
                p[0] = min(max(float(p[0]) + float(push[0]), 0.0), room_w)
                p[2] = min(max(float(p[2]) + float(push[1]), 0.0), room_d)
                after = [round(float(p[0]), 4), round(float(p[2]), 4)]
                if before != after:
                    moved = True
                    corrections.append({
                        "index": mover.get("index"), "type": mover.get("type"),
                        "reason": f"{depth * 100:.1f} cm into "
                                  f"idx={other.get('index')} "
                                  f"({other.get('type')})",
                        "from_xz": before, "to_xz": after})
        if not moved:
            break

    # 3. Floor contact (report only — a Y fix belongs to the height pass) ----
    for o in objs:
        if o.get("on_top_of") is not None or o.get("flat_on_floor"):
            continue
        y = float(o["position_m"][1])
        if abs(y) > 0.05:
            findings.append({"kind": "off_floor", "index": o.get("index"),
                             "type": o.get("type"), "offset_m": round(y, 4)})

    report = {
        "room": {"width_m": round(float(room_w), 4),
                 "depth_m": round(float(room_d), 4)},
        "objects": [
            {"index": o.get("index"), "type": o.get("type"),
             "wall_affinity": o.get("wall_affinity"),
             "yaw_deg": round(math.degrees(_yaw(o)), 1),
             "footprint_m": [round(v, 4) for v in
                             (footprint(o).min(0).tolist()
                              + footprint(o).max(0).tolist())],
             "size_m": o.get("size_m")}
            for o in objs
        ],
        "findings": findings,
        "corrections": corrections,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))

    # Summarise EVERY finding kind. A summary that counts only the kinds it
    # knows about prints "0, 0, 0" over a scene that has findings.
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    detail = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "clean"
    print(f"[geom_audit] {len(objs)} object(s) with a floor footprint "
          f"(of {len(placements)} placed): {detail} — "
          f"{len(corrections)} correction(s) applied")
    for f in findings:
        if f["kind"] in ("no_floor_footprint", "degenerate_footprint"):
            print(f"  [geom_audit] idx={f['index']} {f['type']}: {f['kind']}"
                  f" — {f.get('reason') or f.get('min_extent_m')}")
    for c in corrections:
        print(f"  [geom_audit] idx={c['index']} {c['type']}: {c['reason']} → "
              f"({c['from_xz'][0]:.2f},{c['from_xz'][1]:.2f}) → "
              f"({c['to_xz'][0]:.2f},{c['to_xz'][1]:.2f})")
    return report


# ── top-down orthographic plan ──────────────────────────────────────────────

def render_plan(placements: list[dict], room_w: float, room_d: float,
                out_path: str | Path, *, decorations: list[dict] | None = None,
                findings: list[dict] | None = None, title: str = "",
                cam: dict | None = None) -> Path:
    """Orthographic top-down plan: oriented footprints + facing arrows.

    Image +x is world +X, image +y is world +Z, so the BACK wall (z=0) is at the
    top.  Wall adjacency and depth ordering are unambiguous here; in the
    reference-camera view they are not, which is why this exists.

    PASS `cam` WHEN THE PLAN IS SHOWN TO A VLM.  World-X order is NOT screen-X
    order: the reference camera is usually oblique, so sorting office8's objects
    by world X gives pixel columns 672, 1427, 882, 1710, 2059, 1036 — not
    monotonic.  A model told only "image x = world X" will therefore read
    left/right off this plan in a way that contradicts the photograph.  With
    `cam` the camera marker and its view wedge are drawn, so "left of the
    camera" is readable from the plan itself instead of assumed from the page.
    """
    from PIL import Image, ImageDraw

    S = 120                                  # px per metre
    pad = 56
    W = int(room_w * S) + 2 * pad
    H = int(room_d * S) + 2 * pad
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img, "RGBA")

    def px(x, z):
        return pad + x * S, pad + z * S

    bad = set()
    for f in (findings or []):
        for k in ("index", "index_a", "index_b"):
            if f.get(k) is not None:
                bad.add(f[k])

    d.rectangle([pad, pad, pad + room_w * S, pad + room_d * S],
                fill=(255, 255, 255), outline=(40, 40, 40), width=4)
    d.text((pad + 6, pad - 22), "BACK wall  (z=0)", fill=(40, 40, 40))
    d.text((pad + 6, pad + room_d * S + 8), "FRONT (camera side)", fill=(40, 40, 40))
    d.text((6, pad + 6), "LEFT", fill=(40, 40, 40))
    d.text((pad + room_w * S + 8, pad + 6), "RIGHT", fill=(40, 40, 40))
    if title:
        d.text((6, 6), title, fill=(20, 20, 20))
    d.text((6, H - 18),
           f"room {room_w:.2f} × {room_d:.2f} m   •   {S} px/m", fill=(90, 90, 90))

    # carpets first, underneath everything
    for o in placements:
        if not o.get("is_carpet"):
            continue
        wc = o.get("world_corners")
        if not wc:
            continue
        pts = [px(p[0], p[2]) for p in wc]
        d.polygon(pts, fill=(226, 222, 214, 180), outline=(190, 186, 178))

    for o in placements:
        if o.get("is_carpet"):
            continue
        fp = footprint(o)
        if fp is None:
            continue
        idx = o.get("index")
        flagged = idx in bad
        col = (208, 52, 44) if flagged else (54, 96, 168)
        fill = (208, 52, 44, 46) if flagged else (54, 96, 168, 46)
        d.polygon([px(p[0], p[1]) for p in fp], fill=fill, outline=col)
        # re-stroke: PIL's polygon outline is 1 px
        pts = [px(p[0], p[1]) for p in fp]
        for i in range(4):
            d.line([pts[i], pts[(i + 1) % 4]], fill=col, width=3)
        # facing arrow from centre
        c = np.array([float(o["position_m"][0]), float(o["position_m"][2])])
        f = facing(o)
        depth = float((o.get("size_m") or {}).get("depth_m", 0.4))
        tip = c + f * max(0.22, depth * 0.55)
        d.line([px(*c), px(*tip)], fill=col, width=3)
        head = np.array([-f[1], f[0]]) * 0.07
        d.polygon([px(*tip), px(*(tip - f * 0.11 + head)),
                   px(*(tip - f * 0.11 - head))], fill=col)
        wa = str(o.get("wall_affinity") or "centre")[:1].upper()
        d.text((px(*c)[0] - 18, px(*c)[1] - 6),
               f"{idx}:{str(o.get('type', '?'))[:10]}[{wa}]", fill=(20, 20, 20))

    for dec in (decorations or []):
        p = dec.get("position_m")
        if not p:
            continue
        x, y = px(float(p[0]), float(p[2]))
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(230, 145, 30))

    # Camera marker + view wedge — the only thing that makes "left"/"right" on
    # this plan mean the same as "left"/"right" in the reference photo.
    if cam and cam.get("position_m") and cam.get("look_at_m"):
        cp = np.array(cam["position_m"], float)
        la = np.array(cam["look_at_m"], float)
        fw = np.array([la[0] - cp[0], la[2] - cp[2]])
        n = np.linalg.norm(fw)
        if n > 1e-6:
            fw = fw / n
            half = math.radians(float(cam.get("hfov_deg", 60.0)) / 2.0)
            c0 = np.array([cp[0], cp[2]])
            reach = max(room_w, room_d) * 1.4
            for sgn, lab in ((+1, None), (-1, None)):
                ca, sa = math.cos(sgn * half), math.sin(sgn * half)
                ray = np.array([fw[0] * ca - fw[1] * sa, fw[0] * sa + fw[1] * ca])
                d.line([px(*c0), px(*(c0 + ray * reach))],
                       fill=(120, 120, 120), width=2)
            d.line([px(*c0), px(*(c0 + fw * reach))], fill=(160, 160, 160), width=1)
            cx, cy = px(*c0)
            d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7],
                      fill=(30, 30, 30), outline=(255, 255, 255))
            # Which way is screen-left for THIS camera (y-up, right-handed):
            # screen-right in XZ is (fw x up) = (fw_z, -fw_x).
            sr = np.array([fw[1], -fw[0]])
            tip = c0 + fw * 0.55 + sr * 0.55
            d.line([px(*(c0 + fw * 0.55)), px(*tip)], fill=(30, 30, 30), width=2)
            d.text((px(*tip)[0] + 3, px(*tip)[1] - 6), "screen-RIGHT",
                   fill=(30, 30, 30))
            d.text((cx + 10, cy - 4), "CAMERA", fill=(30, 30, 30))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    print(f"[topdown] plan → {out_path}")
    return Path(out_path)
