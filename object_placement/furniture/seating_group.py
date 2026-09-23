"""
seating_group.py — geometry-driven correction of a living-room seating group,
applied automatically at the end of place_furniture_vggt.run() (and runnable
standalone via scripts/anchor_seating_group.py / resolve_furniture_collisions.py).

Per-object VGGT depth is unreliable for the CENTRED pieces of a conversation
group: a coffee table whose image bbox sits dead-centre in front of the sofa can
still back-project into a corner, accent chairs can be left facing the wrong way,
and neighbours can interpenetrate (a side table swallowed by a bookcase). These
rules re-derive the group from the SOFA's geometry (which the engine places
reliably, flush to a wall) and then separate overlaps:

  anchor_group():
    • Central coffee table = largest "coffee_table" whose image bbox overlaps the
      sofa horizontally → re-centred in front of the sofa, LONG side parallel
      (long edge facing the sofa).
    • Free-standing accent chairs (wall_affinity "centre") → rotated to face the
      central table, and pulled to a minimum camera clearance so they aren't
      clipped by the near plane.

  resolve_collisions():
    • Damped relaxation that slides each lower-priority object out of every
      higher-priority / wall-anchored neighbour until footprints clear.
    • Caps obviously over-tall pieces (occluded base inflates the height estimate).

Everything is computed from the scene's own sofa/table/camera geometry — no
per-scene constants beyond the type tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_TABLE_TYPES = {"coffee_table", "coffee table", "cocktail_table"}
_CHAIR_TYPES = {"chair", "armchair", "accent_chair", "lounge_chair"}
_SOFA_TYPES  = {"sofa", "couch", "loveseat", "sectional", "sectional_sofa",
                "corner_sofa", "l_sofa", "l_shaped_sofa"}

GAP_TABLE_M   = 0.30   # clearance between sofa front and table near edge
MIN_CAM_CLEAR = 1.8    # chairs kept at least this far from the camera
CHAIR_FWD     = 0.65   # chairs seated this far IN FRONT OF the table (toward cam)
MARGIN        = 0.08   # clearance left between pieces after separation

_PRIORITY = {
    "sofa": 5, "sectional": 5, "bed": 5,
    "bookcase": 4, "cabinet": 4, "shelf": 4, "wardrobe": 4, "dresser": 4,
    "desk": 3, "dining_table": 3,
    "coffee_table": 2, "side_table": 2, "end_table": 2, "nightstand": 2,
    "chair": 1, "armchair": 1, "stool": 1, "ottoman": 1, "plant": 1,
}
_MAX_H = {
    # Tall shelving is legitimate (full-height bookcases ~2 m) — only cap clearly
    # implausible values. Low pieces stay tightly capped.
    "bookcase": 2.2, "cabinet": 2.1, "shelf": 2.2, "dresser": 1.3, "wardrobe": 2.3,
    "side_table": 0.7, "coffee_table": 0.6, "nightstand": 0.8,
    "chair": 1.2, "armchair": 1.1, "sofa": 1.1, "plant": 2.0,
}


def _rot_to_dir(local_front, dx, dz):
    lf = np.asarray(local_front, float)
    phi = np.arctan2(lf[0], lf[2])
    psi = np.arctan2(dx, dz)
    th = psi - phi
    c, s = np.cos(th), np.sin(th)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _bbox_overlaps_x(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0])


def _footprint(o) -> float:
    s = o.get("size_m", {})
    return float(s.get("width_m", 0)) * float(s.get("depth_m", 0))


def _prio(o) -> int:
    p = _PRIORITY.get(o.get("type"), 2)
    if o.get("_anchored_to_sofa"):
        p += 1
    return p


def _aabb(o):
    p = o["position_m"]
    s = o.get("size_m", {})
    R = np.asarray(o.get("rotation_3x3", np.eye(3).tolist()), float)
    fr = R @ np.asarray(o.get("_local_front", [0, 0, 1]), float)
    ang = np.arctan2(fr[0], fr[2])
    w, d = s.get("width_m", 0.3), s.get("depth_m", 0.3)
    hx = abs(np.cos(ang)) * w / 2 + abs(np.sin(ang)) * d / 2
    hz = abs(np.sin(ang)) * w / 2 + abs(np.cos(ang)) * d / 2
    return p[0] - hx, p[0] + hx, p[2] - hz, p[2] + hz


def _attach_boxes(placements, scene: Path) -> None:
    """furniture_placements.json drops box_px — recover from the segments."""
    if all(o.get("box_px") for o in placements):
        return
    seg_p = scene / "furniture" / "segment_results.json"
    if not seg_p.exists():
        return
    src = json.load(open(seg_p))
    segs = src.get("segments", src) if isinstance(src, dict) else src
    box_by_idx = {s.get("index"): s.get("box_px") for s in segs}
    for o in placements:
        if o.get("box_px") is None:
            o["box_px"] = box_by_idx.get(o.get("index"))


def anchor_group(placements, cam, scene: Path,
                 gap_table_m: float = GAP_TABLE_M) -> None:
    _attach_boxes(placements, scene)
    cam_pos = np.asarray(cam["position_m"], float)

    sofa = max((o for o in placements if o.get("type") in _SOFA_TYPES),
               key=_footprint, default=None)
    if sofa is None:
        return

    s_pos = np.asarray(sofa["position_m"], float)
    s_R = np.asarray(sofa["rotation_3x3"], float)
    s_front = s_R @ np.asarray(sofa.get("_local_front", [0, 0, 1]), float)
    s_front = s_front / (np.linalg.norm(s_front) + 1e-9)
    s_front_xz = np.array([s_front[0], s_front[2]])
    s_depth = float(sofa.get("size_m", {}).get("depth_m", 0.8))
    s_box = sofa.get("box_px")

    tables = [o for o in placements if o.get("type") in _TABLE_TYPES]
    central = None
    if tables and s_box is not None:
        cands = [o for o in tables
                 if o.get("box_px") and _bbox_overlaps_x(o["box_px"], s_box)]
        if cands:
            central = max(cands, key=_footprint)

    # PHOTO-ALIGNED policy: keep each object's back-projected (photo) position;
    # correct only ORIENTATION. The one exception is a back-projection that
    # clearly FAILED — a central table that drifted laterally off the sofa or
    # landed behind it (the depth-noise "table in the corner" case) — which we
    # re-center so it matches where the photo shows it.
    # Wall-inward normal = "in front of the sofa".  Reliable for L-sectionals
    # whose mesh front-axis (s_front_xz) can point sideways — used for BOTH the
    # central-vs-side demotion test and the in-front anchor below, so a table's
    # deep +Z back-projection isn't miscounted as a huge LATERAL offset (which
    # was wrongly demoting the central table to a "side table" in deep rooms).
    _wall_front = {
        "right": np.array([-1.0, 0.0]),   # right wall (x=room_w) → into room is -X
        "left":  np.array([1.0, 0.0]),
        "back":  np.array([0.0, 1.0]),     # back wall (z=0) → into room is +Z
    }
    # For a wall-flush sofa the wall-inward normal is the reliable "in front of
    # the sofa" direction.  For a CENTRE/free-standing sofa there is no wall, so
    # default-to-+Z is wrong when the sofa faces ±X (the table then anchors at the
    # sofa's far END instead of centred in front) — use the sofa's own facing.
    _wa = sofa.get("wall_affinity")
    if _wa in _wall_front:
        front_dir = _wall_front[_wa]
    else:
        _fn = float(np.linalg.norm(s_front_xz))
        front_dir = (s_front_xz / _fn) if _fn > 1e-6 else np.array([0.0, 1.0])
    if central is not None:
        cur = np.array([central["position_m"][0], central["position_m"][2]])
        rel = cur - np.array([s_pos[0], s_pos[2]])
        along = float(rel @ front_dir)                        # forward dist from sofa
        lateral = float(np.linalg.norm(rel - along * front_dir))
        # A table is the central coffee table only if it sits roughly in FRONT of
        # the sofa — i.e. its lateral offset is within the sofa's own half-length
        # (a coffee table before a long sofa can be ~1 m off-centre and still be
        # central).  A table laterally BEYOND the sofa's end is a SIDE table
        # (beside an arm) — leave it at its photo position.  Threshold is relative
        # to the sofa length, not a fixed 0.80 m, so the central table isn't
        # mis-demoted on a long sofa.
        # Lateral half-length = the sofa's world extent PERPENDICULAR to front_dir.
        # Don't assume width_m is the seat length: a sofa facing ±X has its seat
        # length along Z stored as depth_m (004242: width_m=0.87, depth_m=2.19), so
        # width_m/2 grossly understates the half-length and a centred table near the
        # sofa's end gets mis-demoted to a side table.
        _sw = float(sofa.get("size_m", {}).get("width_m", 1.8))
        _sd = float(sofa.get("size_m", {}).get("depth_m", 0.9))
        _shx = (abs(s_R[0, 0]) * _sw + abs(s_R[0, 2]) * _sd) / 2.0
        _shz = (abs(s_R[2, 0]) * _sw + abs(s_R[2, 2]) * _sd) / 2.0
        _lat_dir = np.array([-front_dir[1], front_dir[0]])
        s_half_len = abs(_shx * _lat_dir[0]) + abs(_shz * _lat_dir[1])
        # An L-shaped sectional seats in two directions, so the central table can
        # legitimately sit further off the width-centre (e.g. in front of the
        # chaise return) — widen the lateral tolerance so it isn't mis-demoted.
        _is_sectional = sofa.get("type", "") in {
            "sectional", "sectional_sofa", "corner_sofa", "l_sofa", "l_shaped_sofa"}
        lateral_cap = max(0.80, 0.9 * s_half_len) + (0.7 if _is_sectional else 0.0)
        # A table whose footprint OVERLAPS the sofa is the central coffee table
        # mis-placed (you never park a side table inside the sofa) — re-anchor it in
        # front rather than demoting it and leaving the collision (004242).
        _tw = float(central.get("size_m", {}).get("width_m", 0.5))
        _td = float(central.get("size_m", {}).get("depth_m", 0.5))
        _tR = np.asarray(central.get("rotation_3x3", np.eye(3).tolist()), float)
        _thx = (abs(_tR[0, 0]) * _tw + abs(_tR[0, 2]) * _td) / 2.0
        _thz = (abs(_tR[2, 0]) * _tw + abs(_tR[2, 2]) * _td) / 2.0
        _ovx = min(cur[0] + _thx, s_pos[0] + _shx) - max(cur[0] - _thx, s_pos[0] - _shx)
        _ovz = min(cur[1] + _thz, s_pos[2] + _shz) - max(cur[1] - _thz, s_pos[2] - _shz)
        _overlaps_sofa = (_ovx > 0.02 and _ovz > 0.02)
        if lateral > lateral_cap and not _overlaps_sofa:
            print(f"[seating] table idx{central.get('index')} is beside the sofa "
                  f"(lateral={lateral:.2f}m > {lateral_cap:.2f}) → side table, "
                  f"keeping photo position")
            central = None
        elif lateral > lateral_cap:
            print(f"[seating] table idx{central.get('index')} laterally off "
                  f"(lat={lateral:.2f}>{lateral_cap:.2f}) but OVERLAPS sofa → central, "
                  f"re-anchoring in front")
    if central is not None:
        # Anchor the central table just in front of the sofa (front_dir computed
        # above — the wall-inward normal), preserving its photo lateral offset.
        along = float(rel @ front_dir)
        lateral_vec = rel - along * front_dir             # component beside the sofa axis
        # Pull the table toward the sofa's lateral CENTRE: a coffee table sits in
        # front of the SEATING, not way off to one side.  In a deep room the photo
        # back-projection can place it far off-centre (e.g. ~1.9 m left of an
        # L-sectional), so cap the lateral offset so it stays roughly in front of
        # the sofa (still allowing a modest off-centre nudge from the photo).
        # Keep the table near its SILHOUETTE lateral position — only clear the
        # collision, don't yank it to the sofa's dead centre (004242: pulling it
        # fully central put it far from where the photo shows it). Allow up to the
        # sofa's lateral half-length (so it can sit toward a long sofa's end) but
        # never wildly off to one side.
        _LAT_CAP = float(np.clip(s_half_len, 0.6, 1.1))
        _lat_mag = float(np.linalg.norm(lateral_vec))
        if _lat_mag > _LAT_CAP:
            lateral_vec = lateral_vec * (_LAT_CAP / _lat_mag)
        t_depth = float(central.get("size_m", {}).get("depth_m", 0.5))
        # Cap the sofa's effective seat-front distance from its bbox centre: a
        # sectional's depth_m includes the chaise return, so s_depth/2 grossly
        # over-states how far ahead the seat face actually is, shoving the table
        # metres into the room.  A real sofa seat-front is ≲0.55 m from centre.
        # Forward extent = the sofa's half-extent ALONG front_dir — NOT depth_m/2,
        # which for a ±X-facing sofa is the seat WIDTH (004242: depth_m=2.19), wildly
        # over-stating the seat-front and shoving the table far past its silhouette.
        _fwd_half = abs(_shx * front_dir[0]) + abs(_shz * front_dir[1])
        _seat_front = min(_fwd_half, 0.55)
        # Cap the table's own depth contribution: a mis-scaled coffee table can
        # read ~1.2 m deep, which over-inflates the clearance and shoves the table
        # forward (rendering BELOW its silhouette bbox).  A real coffee table is
        # ≲0.7 m front-to-back, so cap it — pulls the table back toward its bbox.
        min_off = _seat_front + gap_table_m + min(t_depth, 0.7) / 2.0
        # IN-FRONT-OF-SOFA ANCHOR (3D-plausible): a central coffee table belongs
        # just in front of the sofa, inside the L.  In an over-deep room the
        # silhouette/VGGT back-projection both place the table too far forward
        # (the foreground floor maps to large z), stranding it mid-room.  The
        # forward distance is the single-view-ambiguous dimension, so CLAMP it to
        # a tight band just in front of the sofa — keeping the lateral (photo)
        # offset so it still sits where the photo shows it left-right.
        # "In front of the sofa" is DIRECTIONAL, not a strict distance: the table
        # is already mask-aligned, so its current forward distance (`along`) is
        # the photo silhouette's depth, and the silhouette is authoritative.  So
        # only push the table FORWARD when it is too close (inside the sofa's
        # seat-front clearance) — never pull it back toward the sofa.  The old
        # tight band (+0.15) yanked silhouette-correct tables back, rendering them
        # SMALLER (further from camera) and off their silhouette bbox
        # (living_room9: z 2.30→1.81).  Room-containment is the backstop for any
        # egregiously over-deep back-projection.
        new_along = float(max(along, min_off))
        s_pos_xz = np.array([s_pos[0], s_pos[2]])
        new_xz = s_pos_xz + front_dir * new_along + lateral_vec
        central["position_m"] = [float(new_xz[0]),
                                 float(central["position_m"][1]), float(new_xz[1])]
        central["_anchored_to_sofa"] = True
        if abs(new_along - along) > 1e-3:
            print(f"[seating] central table idx{central.get('index')} anchored in front of "
                  f"sofa (along {along:.2f}→{new_along:.2f}m, lat={lateral:.2f}) "
                  f"→ ({new_xz[0]:.2f},{new_xz[1]:.2f})")
        else:
            print(f"[seating] central table idx{central.get('index')} already in front "
                  f"(along={along:.2f} lat={lateral:.2f})")
        # Orientation: long side parallel to the WALL the sofa backs onto, table
        # facing the sofa.  Use the wall-inward normal (front_dir), NOT the sofa's
        # mesh front-axis — an L-sectional's mesh front is ambiguous and can leave
        # the table edge-on (perpendicular) to the wall instead of parallel.
        central["rotation_3x3"] = _rot_to_dir(
            central.get("_local_front", [0, 0, 1]), -front_dir[0], -front_dir[1])
        central["_anchored_to_sofa"] = True
        t_pos_xz = np.array([central["position_m"][0], central["position_m"][2]])
    else:
        t_pos_xz = np.array([s_pos[0], s_pos[2]]) + s_front_xz * (s_depth / 2 + 0.6)

    tx, tz = float(t_pos_xz[0]), float(t_pos_xz[1])
    for o in placements:
        if o.get("type") not in _CHAIR_TYPES:
            continue
        if o.get("wall_affinity") not in (None, "centre"):
            continue
        # Keep the chair's photo-aligned position; only rotate it to face the
        # central table. With the per-mesh front now VLM-detected, this aims the
        # SEAT at the table without pulling the chair off its reference footprint.
        cx, cz = float(o["position_m"][0]), float(o["position_m"][2])
        # Use the backrest-detected seat-front (same authority as the dining
        # chair-group) rather than the raw _local_front, which is frequently the
        # BACK of the mesh and would aim the chair 180° away from the table.
        _sf = _chair_seat_front_local(o.get("glb_path")) or o.get("_local_front", [0, 0, 1])
        o["rotation_3x3"] = _rot_to_dir(_sf, tx - cx, tz - cz)
        o["_faces_table"] = True


def _cap_heights(placements) -> None:
    for o in placements:
        cap = _MAX_H.get(o.get("type"))
        s = o.get("size_m", {})
        h = s.get("height_m")
        if cap and h and h > cap + 1e-3:
            f = cap / h
            s["height_m"] = cap
            if o.get("eff_h"):
                o["eff_h"] = float(o["eff_h"]) * f
            sc = o.get("scale")
            if isinstance(sc, list) and len(sc) == 3:
                o["scale"] = [sc[0], sc[1] * f, sc[2]]
            print(f"[seating] idx{o.get('index')} {o.get('type')} height "
                  f"{h:.2f}→{cap:.2f} m")


def resolve_collisions(placements, room_w: float, room_d: float) -> None:
    _cap_heights(placements)
    DAMP = 0.5
    for _ in range(40):
        max_step = 0.0
        for b in placements:
            push = np.zeros(2)
            for a in placements:
                if a is b or _prio(a) <= _prio(b):
                    continue
                ax0, ax1, az0, az1 = _aabb(a)
                bx0, bx1, bz0, bz1 = _aabb(b)
                ox = min(ax1, bx1) - max(ax0, bx0)
                oz = min(az1, bz1) - max(az0, bz0)
                if ox <= 0.0 or oz <= 0.0:
                    continue
                pb, pa = b["position_m"], a["position_m"]
                if ox <= oz:
                    push[0] += (1.0 if pb[0] >= pa[0] else -1.0) * (ox + MARGIN)
                else:
                    push[1] += (1.0 if pb[2] >= pa[2] else -1.0) * (oz + MARGIN)
            if push[0] or push[1]:
                pb = b["position_m"]
                pb[0] = min(max(pb[0] + DAMP * push[0], 0.2), room_w - 0.2)
                pb[2] = min(max(pb[2] + DAMP * push[1], 0.2), room_d - 0.2)
                max_step = max(max_step, abs(DAMP * push[0]), abs(DAMP * push[1]))
        if max_step < 0.01:
            break


def _remaining_overlaps(placements, room_w, room_d):
    """Return list of (idx_a, idx_b, overlap_area) for footprints that still overlap."""
    objs = [o for o in placements if not o.get("is_carpet")]
    out = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            ax0, ax1, az0, az1 = _aabb(a)
            bx0, bx1, bz0, bz1 = _aabb(b)
            ox = min(ax1, bx1) - max(ax0, bx0)
            oz = min(az1, bz1) - max(az0, bz0)
            if ox > 0.03 and oz > 0.03:
                out.append((a.get("index"), b.get("index"), float(ox * oz)))
    return out


def _render_layout_schematic(placements, room_w, room_d, path, overlaps):
    """Top-down schematic: room rectangle + labelled footprint box per object,
    overlapping ones outlined red.  Image x = world X, image y = world Z (back
    wall at top), so the VLM's left/right/front/back matches world axes."""
    from PIL import Image, ImageDraw
    S = 90  # px per metre
    pad = 40
    W = int(room_w * S) + 2 * pad
    H = int(room_d * S) + 2 * pad
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([pad, pad, pad + int(room_w * S), pad + int(room_d * S)],
                outline=(0, 0, 0), width=3)
    d.text((pad + 4, 4), "BACK wall (z=0)  •  image x=world X, image y=world Z", fill=(0, 0, 0))
    in_overlap = set()
    for ia, ib, _ in overlaps:
        in_overlap.add(ia); in_overlap.add(ib)
    for o in placements:
        if o.get("is_carpet"):
            continue
        x0, x1, z0, z1 = _aabb(o)
        px0 = pad + int(x0 * S); px1 = pad + int(x1 * S)
        py0 = pad + int(z0 * S); py1 = pad + int(z1 * S)
        col = (220, 40, 40) if o.get("index") in in_overlap else (70, 110, 200)
        d.rectangle([px0, py0, px1, py1], outline=col, width=3)
        d.text((px0 + 2, py0 + 2), f"{o.get('index')}:{o.get('type','?')[:8]}", fill=col)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return path


def _vlm_layout_round(placements, scene: Path, room_w, room_d, overlaps) -> int:
    """One VLM relocation round.  Renders the labelled top-down, asks the VLM for
    moves that clear the listed overlaps WITHOUT creating new ones, applies them.
    Returns the number of objects actually moved."""
    import base64, json as _json, re
    from object_placement.vlm_backend import vlm_post

    sch = Path(scene) / "furniture" / "_layout_schematic.png"
    _render_layout_schematic(placements, room_w, room_d, sch, overlaps)
    b64 = base64.b64encode(open(sch, "rb").read()).decode()

    by_idx = {o.get("index"): o for o in placements if not o.get("is_carpet")}
    lines = []
    for o in by_idx.values():
        s = o.get("size_m", {})
        wa = o.get("wall_affinity")
        if wa in ("left", "right"):
            mob = "wall-flush on a SIDE wall (slides along Z, stays flush in X)"
        elif wa == "back":
            mob = "wall-flush on the BACK wall (slides along X, stays flush in Z)"
        else:
            mob = "free-standing (moves anywhere)"
        lines.append(f"  idx={o.get('index')} type={o.get('type')} "
                     f"pos=(x={o['position_m'][0]:.2f},z={o['position_m'][2]:.2f}) "
                     f"size=({s.get('width_m',0):.2f}x{s.get('depth_m',0):.2f}) — {mob}")
    ov_txt = "; ".join(f"idx{a}↔idx{b}" for a, b, _ in overlaps)
    prompt = (
        "Top-down floor plan of a room. Image X = world X (right), image Y = world Z "
        f"(down = toward camera; back wall at top, z=0). Room is {room_w:.2f}m (X) "
        f"by {room_d:.2f}m (Z). Red boxes overlap and must be separated.\n\n"
        "Objects:\n" + "\n".join(lines) + "\n\n"
        f"Overlapping pairs: {ov_txt}.\n\n"
        "Reposition objects so NO footprints overlap. EVERY object is movable — choose "
        "the move that resolves the conflict with the LEAST disruption AND the most "
        "open space:\n"
        "- Look at the schematic for the LARGEST empty floor areas. Move pieces INTO "
        "that open space rather than squeezing them into a tight gap between other "
        "objects.\n"
        "- If a piece has no room near its spot, prefer SLIDING the wall-flush sofa/"
        "cabinet along its wall into open space (the sofa often has the most room to "
        "slide) instead of cramming a small table somewhere it will hit a chair.\n"
        "- CRITICAL: do NOT create any NEW overlap — your new positions must clear of "
        "EVERY other object (check all of them, not just the listed pair).\n"
        "- A wall-flush piece stays flush against its wall (slide only); free pieces "
        "move anywhere; keep each object's role (side table beside sofa, coffee table "
        "in front of sofa); keep everything inside the room.\n\n"
        "Return ONLY JSON: {\"moves\":[{\"index\":N,\"x\":<m>,\"z\":<m>}]} for every "
        "object you move (include the sofa if sliding it helps)."
    )
    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "max_tokens": 1024,
    }
    try:
        resp = vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.I)
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            print("[vlm_layout] no JSON in VLM response")
            return 0
        moves = _json.loads(m.group()).get("moves", [])
    except Exception as e:
        print(f"[vlm_layout] VLM call failed ({e})")
        return 0

    n_moved = 0
    for mv in moves:
        o = by_idx.get(mv.get("index"))
        if o is None:
            continue
        # Clamp the CENTRE so the object's FOOTPRINT stays in the room — using
        # rotation-aware half-extents (size_m is local; a yaw=±90 sofa's width
        # runs along Z).  Clamping only the centre to a fixed margin let large /
        # rotated pieces (e.g. a deep sectional) clip metres through a wall.
        _sz = o.get("size_m", {}) or {}
        _w = float(_sz.get("width_m", 0.4)); _d = float(_sz.get("depth_m", 0.4))
        _R = o.get("rotation_3x3")
        if _R is not None:
            _c = abs(float(_R[0][0])); _s = abs(float(_R[0][2]))
            _hw = (_c * _w + _s * _d) / 2.0; _hd = (_s * _w + _c * _d) / 2.0
        else:
            _hw, _hd = _w / 2.0, _d / 2.0
        _lox, _hix = _hw + 0.02, room_w - _hw - 0.02
        _loz, _hiz = _hd + 0.02, room_d - _hd - 0.02
        nx = (room_w / 2.0 if _lox > _hix
              else float(np.clip(mv.get("x", o["position_m"][0]), _lox, _hix)))
        nz = (room_d / 2.0 if _loz > _hiz
              else float(np.clip(mv.get("z", o["position_m"][2]), _loz, _hiz)))
        # No piece is immovable, but a wall-flush piece only SLIDES ALONG its wall.
        wa = o.get("wall_affinity")
        if wa in ("left", "right"):
            nx = o["position_m"][0]          # flush in X → slide along Z only
        elif wa == "back":
            nz = o["position_m"][2]          # flush in Z → slide along X only
        # Silhouette is the authority: a free-standing piece (chair, ottoman)
        # must not be teleported far from where its segmentation mask placed it.
        # Without this the VLM "resolves" a collision by jumping a lounge chair to
        # the opposite side of the room (004160: left chair → far right). Bound the
        # move to a radius around _mask_origin_xz; a minor residual overlap on the
        # correct side beats a chair on the wrong side.
        _mo = o.get("_mask_origin_xz")
        if _mo and wa not in ("left", "right", "back"):
            _MAXDEV = 1.0
            nx = float(np.clip(nx, _mo[0] - _MAXDEV, _mo[0] + _MAXDEV))
            nz = float(np.clip(nz, _mo[1] - _MAXDEV, _mo[1] + _MAXDEV))
        if abs(nx - o["position_m"][0]) < 1e-3 and abs(nz - o["position_m"][2]) < 1e-3:
            continue
        print(f"[vlm_layout] idx={o.get('index')} {o.get('type')}: "
              f"({o['position_m'][0]:.2f},{o['position_m'][2]:.2f}) → ({nx:.2f},{nz:.2f})")
        o["position_m"][0] = nx
        o["position_m"][2] = nz
        n_moved += 1
    return n_moved


def _vlm_resolve_collisions(placements, cam, scene: Path, room_w, room_d,
                            max_rounds: int = 3) -> None:
    """Iteratively resolve footprint overlaps the local push can't escape, using a
    VLM that reasons over the whole top-down layout — moving whichever piece (any
    piece, including the sofa) into open space gives the cleanest result.  Iterates
    so a move that introduces a NEW overlap (e.g. a table shoved into a chair) gets
    caught and re-resolved instead of left."""
    for rnd in range(max_rounds):
        overlaps = _remaining_overlaps(placements, room_w, room_d)
        if not overlaps:
            if rnd:
                print("[vlm_layout] all overlaps resolved")
            return
        print(f"[vlm_layout] round {rnd + 1}: {len(overlaps)} overlap(s) → asking VLM")
        n = _vlm_layout_round(placements, scene, room_w, room_d, overlaps)
        resolve_collisions(placements, room_w, room_d)   # local cleanup of small residue
        if n == 0:
            break   # VLM made no usable move — stop
    rem = _remaining_overlaps(placements, room_w, room_d)
    print(f"[vlm_layout] overlaps after VLM relocation: {len(rem)}")


# ── dining / meeting chair-group arrangement ─────────────────────────────────
_DINING_TABLE_TYPES = {"dining_table", "dining table", "meeting_table",
                       "meeting table", "round_table", "round table", "table"}
_DINING_CHAIR_TYPES = {"chair", "dining_chair", "dining chair", "armchair",
                       "side_chair", "office_chair"}
CHAIR_TABLE_GAP = -0.08  # chair-to-table offset. NEGATIVE (tuck under): chairs &
                         # tables are placed at their PRE-clamp (oversized) sizes
                         # but render shrunk by the per-type scale-clamp, leaving a
                         # ~0.15 m phantom gap. A small tuck cancels it so chairs sit
                         # AT the table (collision-exempt for dining sets, so safe).


def _chair_seat_front_local(glb_path):
    """Detect a chair's SEAT-FRONT in local coords from its backrest: the tall
    upper portion is concentrated on the BACK side, so the seat faces the
    opposite way.  Robust to a wrong/defaulted `_local_front` (which on these
    GLBs points at the BACKREST, making chairs face away from the table)."""
    import trimesh
    try:
        m = trimesh.load(str(glb_path), force="scene")
        gl = (max(m.geometry.values(), key=lambda x: len(x.faces))
              if hasattr(m, "geometry") else m)
        v = gl.vertices
        ymin, ymax = float(v[:, 1].min()), float(v[:, 1].max())
        H = ymax - ymin
        if H < 1e-3:
            return None
        up = v[v[:, 1] > ymin + 0.55 * H]          # upper portion = backrest band
        if len(up) < 10:
            return None
        d = up[:, [0, 2]].mean(0) - v[:, [0, 2]].mean(0)   # → toward backrest
        n = float(np.linalg.norm(d))
        if n < 0.02:                                # symmetric (e.g. stool) — unknown
            return None
        b = d / n
        return [-float(b[0]), 0.0, -float(b[1])]    # seat-front = opposite backrest
    except Exception:
        return None


def _chair_mean_color(glb_path):
    """Mean RGB of a chair mesh (vertex colours / texture), or None.  Used to pick
    a COLOUR-TYPICAL representative so a 3D-gen colour artifact (e.g. a purple
    seat) on one mesh isn't adopted for the whole matching set."""
    import trimesh
    try:
        m = trimesh.load(str(glb_path), force="scene")
        gl = (max(m.geometry.values(), key=lambda x: len(x.faces))
              if hasattr(m, "geometry") else m)
        vis = gl.visual
        if hasattr(vis, "to_color"):
            vc = np.asarray(vis.to_color().vertex_colors)
            if vc.ndim == 2 and len(vc):
                return [float(x) for x in vc[:, :3].mean(0)]
    except Exception:
        pass
    return None


def _place_chair_facing(c, px, pz, tc, face_dir=None):
    """Drop a chair at (px,pz) on the floor, rotated so its SEAT-FRONT points in
    `face_dir` (dx,dz); when face_dir is None it faces the table centre tc.
    Rectangular tables pass an axis-aligned face_dir so chairs sit SQUARE
    (perpendicular) to the edge instead of angling toward the centre point."""
    if face_dir is not None:
        dx, dz = float(face_dir[0]), float(face_dir[1])
    else:
        dx, dz = float(tc[0] - px), float(tc[1] - pz)
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        dz = 1.0
    sf = _chair_seat_front_local(c.get("glb_path")) or c.get("_local_front", [0, 0, 1])
    c["_local_front"] = list(sf)                    # keep downstream consistent
    c["position_m"] = [round(float(px), 3), 0.0, round(float(pz), 3)]
    c["rotation_3x3"] = _rot_to_dir(sf, dx, dz)
    # Record the chair-group's intended spot so a later collision/VLM round can't
    # nudge a chair out of its aligned pair — apply() restores this at the end.
    c["_dining_target_xz"] = [round(float(px), 3), round(float(pz), 3)]


def _default_sides(n, long_is_z):
    """Fallback distribution: fill the two LONG sides evenly, add one chair per
    short end only when crowded (n>=6).  Sides are world directions relative to
    the table: 'front'=+Z, 'back'=-Z, 'left'=-X, 'right'=+X."""
    longs = ("front", "back") if long_is_z else ("left", "right")
    shorts = ("left", "right") if long_is_z else ("front", "back")
    counts = {"front": 0, "back": 0, "left": 0, "right": 0}
    n_ends = 2 if n >= 6 else 0
    n_long = n - n_ends
    counts[longs[0]] = (n_long + 1) // 2
    counts[longs[1]] = n_long // 2
    if n_ends:
        counts[shorts[0]] = 1
        counts[shorts[1]] = 1
    return counts


def _vlm_chair_arrangement(table, chairs, scene, room_w, room_d, is_round):
    """Ask the VLM, over the top-down schematic, how to distribute the chairs.
    Returns {"shape": "round"|"rectangular", "around": N} or
    {"shape":"rectangular","sides":{front,back,left,right}}.  None on failure."""
    import base64, json as _json, re
    from object_placement.vlm_backend import vlm_post
    sch = Path(scene) / "furniture" / "_chairgroup_schematic.png"
    try:
        _render_layout_schematic([table] + chairs, room_w, room_d, sch, [])
        b64 = base64.b64encode(open(sch, "rb").read()).decode()
    except Exception as e:
        print(f"[chair_group] schematic render failed ({e})")
        return None
    ts = table.get("size_m", {})
    prompt = (
        "Top-down floor plan. Image X = world X (right), image Y = world Z "
        "(down; back wall at top). A dining/meeting TABLE (blue box, "
        f"{ts.get('width_m',0):.2f}m X by {ts.get('depth_m',0):.2f}m Z) is "
        f"surrounded by {len(chairs)} chairs. Decide how to arrange the chairs "
        "around the table, each facing it.\n"
        "Report how many chairs sit on EACH side (front=+Z toward camera, "
        "back=-Z, left=-X, right=+X). A side that is against a WALL has 0.\n"
        "- A round/square table in a CORNER has chairs only on its 1-2 OPEN sides "
        "(e.g. two chairs side-by-side on each of two open sides); a free-standing "
        "round table spreads them evenly across all open sides.\n"
        "- A rectangular table puts most chairs on the two LONG sides, splitting "
        "evenly, and at most one at each short end.\n"
        'Return ONLY JSON: {"shape":"round"|"rectangular",'
        '"sides":{"front":n,"back":n,"left":n,"right":n}} with the counts '
        f"summing to {len(chairs)} (0 for any wall-blocked side)."
    )
    payload = {"model": "qwen3", "max_tokens": 400, "messages": [{"role": "user",
               "content": [{"type": "text", "text": prompt},
                           {"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
    try:
        resp = vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.I)
        m = re.search(r"\{[\s\S]*\}", raw)
        arr = _json.loads(m.group()) if m else None
        print(f"[chair_group] VLM arrangement: {arr}")
        return arr
    except Exception as e:
        print(f"[chair_group] VLM call failed ({e})")
        return None


def _vlm_count_chairs(scene):
    """Ask the VLM to COUNT the dining chairs in the REFERENCE photo. This is the
    AUTHORITATIVE count: GDINO gives noisy/duplicate/occluded/blob chair boxes, so
    len(detected_chairs) is unstable across runs. Returns an int (1..12) or None."""
    import base64, re
    from object_placement.vlm_backend import vlm_post
    cand = (list(Path(scene).glob("rgb_*.jpeg")) + list(Path(scene).glob("*.jpeg"))
            + list(Path(scene).glob("manhattan_reference.png")))
    ref = next((r for r in cand if r.exists()), None)
    if ref is None:
        return None
    try:
        b64 = base64.b64encode(open(ref, "rb").read()).decode()
        ext = "png" if ref.suffix.lower() == ".png" else "jpeg"
        prompt = ("Count the CHAIRS arranged around the dining/meeting table in this "
                  "room photo. Count only real chairs a person sits on — NOT stools, "
                  "ottomans, benches, sofas, or the table itself. Include chairs that "
                  "are partly hidden behind the table or behind other chairs. "
                  "Return ONLY the integer count, nothing else.")
        payload = {"model": "qwen3", "max_tokens": 200, "messages": [{"role": "user",
                   "content": [{"type": "text", "text": prompt},
                               {"type": "image_url",
                                "image_url": {"url": f"data:image/{ext};base64,{b64}"}}]}]}
        resp = vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.I)
        m = re.search(r"\d+", raw)
        val = int(m.group()) if m else None
        if val is not None:
            print(f"[chair_group] VLM ref-photo chair count = {val}")
        return val if (val and 1 <= val <= 12) else None
    except Exception as e:
        print(f"[chair_group] VLM chair count failed ({e})")
        return None


def arrange_chair_group(placements, cam, scene: Path, room_w: float,
                        room_d: float, use_vlm: bool = True) -> None:
    """Table-FIRST dining/meeting layout: keep the table where it was placed,
    then arrange its chairs evenly around it (round → around the rim; rectangular
    → split on the long sides), every chair facing the table.  A chair whose mesh
    is missing borrows a sibling chair's GLB from the same group."""
    def _is_dining_table(o):
        t = str(o.get("type", "")).lower().replace("_", " ")
        if "table" not in t:
            return False
        if any(x in t for x in ("coffee", "side", "end", "console",
                                 "nightstand", "cocktail", "tv", "bedside")):
            return False
        return ("dining" in t or "meeting" in t or "round" in t or t.strip() == "table")

    def _is_dining_chair(o):
        t = str(o.get("type", "")).lower()
        if "chair" not in t:
            return False
        # Exclude lounge/wicker/recliner seating: these belong to a separate
        # (e.g. "foreground lounge") group and must not be absorbed into the
        # dining-table arrangement, displacing real dining chairs.
        if any(x in t for x in ("lounge", "wicker", "recliner", "rocking", "papasan")):
            return False
        return True   # dining/side/office/armchair around a dining table

    def _is_real_chair(o):
        # Drop a FALSE 'chair' detection — a flat/round blob (no backrest and not
        # taller than wide) that 3D-gen produced from a mis-segment; counting it
        # adds a phantom extra chair to the set.
        gp = o.get("glb_path")
        if not (gp and Path(gp).exists()):
            return True
        try:
            import trimesh
            m = trimesh.load(gp, force="scene")
            gl = max(m.geometry.values(), key=lambda x: len(x.faces))
            ext = np.abs(gl.bounds[1] - gl.bounds[0])
            height_dom = float(ext[1]) >= 0.8 * max(float(ext[0]), float(ext[2]))
        except Exception:
            return True
        return height_dom or (_chair_seat_front_local(gp) is not None)

    # ── Keep accent/lounge seating OUT of the dining-chair set ───────────────
    # placement_analysis assigns each piece a free-text `group` (e.g. "transparent
    # dining chair set" vs "red armchair seating group").  An accent armchair placed
    # geometrically near the table would otherwise be absorbed here, mixing types and
    # disabling same-style unification.  Read the analysis groups and drop chairs the
    # VLM put in a SEPARATE seating arrangement (so the real dining set unifies).
    _grp = {}
    try:
        import json as _json
        _an = _json.load(open(Path(scene) / "furniture" / "placement_analysis.json"))
        for _e in _an.get("placement_order", []):
            if _e.get("index") is not None and _e.get("group"):
                _grp[_e["index"]] = str(_e["group"]).lower()
    except Exception:
        _grp = {}

    def _is_separate_seating(o):
        g = _grp.get(o.get("index"))
        if not g or "dining" in g or "meeting" in g:
            return False
        return any(k in g for k in ("armchair", "accent", "lounge", "sofa",
                                    "seating group", "reading", "foreground"))

    tables = [o for o in placements if _is_dining_table(o)]
    chairs = [o for o in placements if _is_dining_chair(o)]
    _off = [c for c in chairs if _is_separate_seating(c)]
    for c in _off:
        print(f"[chair_group] idx{c.get('index')} excluded from dining set "
              f"(analysis group: '{_grp.get(c.get('index'))}')")
    chairs = [c for c in chairs if c not in _off]
    if tables and chairs:
        _dropped_chairs = []   # blob/dedup casualties — may be recovered onto empty short ends
        real = [c for c in chairs if _is_real_chair(c)]
        for d in [c for c in chairs if c not in real]:
            print(f"[chair_group] dropped non-chair blob idx{d.get('index')} "
                  f"(flat/symmetric — false detection)")
            if d in placements:
                placements.remove(d)
            _dropped_chairs.append(d)
        chairs = real
        # Drop DUPLICATE chair detections: GDINO sometimes finds the same chair
        # twice with overlapping (non-contained) boxes that the segment-stage
        # dedup misses; they back-project to two spots and inflate the chair count
        # (005440: a 3rd "chair" that was a duplicate of the back chair). Collapse
        # same-chair boxes by IoU, keeping the larger box.
        def _box_iou(a, b):
            if not a or not b:
                return 0.0
            ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
            ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
            iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
            return inter / float(aa + bb - inter) if (aa + bb - inter) > 0 else 0.0
        def _barea(o):
            b = o.get("box_px") or [0, 0, 0, 0]
            return (b[2] - b[0]) * (b[3] - b[1])
        _kept = []
        for c in sorted(chairs, key=lambda o: -_barea(o)):
            if any(_box_iou(c.get("box_px"), k.get("box_px")) > 0.40 for k in _kept):
                print(f"[chair_group] dropped duplicate chair idx{c.get('index')} "
                      f"(box IoU>0.40 with a kept chair — same chair detected twice)")
                if c in placements:
                    placements.remove(c)
                _dropped_chairs.append(c)
            else:
                _kept.append(c)
        chairs = _kept
    if not tables or not chairs:
        return

    table = max(tables, key=_footprint)
    tc = np.array([float(table["position_m"][0]), float(table["position_m"][2])])
    # Pin the table to this anchor: the chairs are arranged around it, so a later
    # collision/VLM round must not drift the table away from its chairs (the VLM
    # layout round is non-deterministic and was relocating the table). apply()
    # restores _dining_target_xz at the end, keeping the whole set locked together.
    table["_dining_target_xz"] = [round(float(tc[0]), 3), round(float(tc[1]), 3)]
    # 1 chair at a table: don't redistribute — just FACE it toward the table
    # (fixes a reversed single chair). 2+ chairs get arranged below (2 → opposite
    # sides, 4 → 2+2, etc.).
    if len(chairs) < 2:
        for c in chairs:
            _place_chair_facing(c, c["position_m"][0], c["position_m"][2], tc)
        print(f"[chair_group] {len(chairs)} chair(s) at table idx{table.get('index')} "
              f"→ faced toward the table")
        return

    # walls_metadata depth is frequently 0.0 (VLM omits floor_depth_m); fall back
    # to the actual walls.obj box so the in-room clamp / side spans aren't broken.
    if room_w <= 0.5 or room_d <= 0.5:
        try:
            wobj = Path(scene) / "walls.obj"
            vs = [list(map(float, l.split()[1:4]))
                  for l in open(wobj) if l.startswith("v ")]
            vs = np.asarray(vs, float)
            ow = float(vs[:, 0].max() - vs[:, 0].min())
            od = float(vs[:, 2].max() - vs[:, 2].min())
            if room_w <= 0.5:
                room_w = ow
            if room_d <= 0.5:
                room_d = od
            print(f"[chair_group] room dims from walls.obj → {room_w:.2f}×{room_d:.2f}")
        except Exception as e:
            print(f"[chair_group] could not read walls.obj dims ({e})")
    ts = table.get("size_m", {})
    tw = float(ts.get("width_m", 1.0)); td = float(ts.get("depth_m", 1.0))
    aspect = max(tw, td) / max(min(tw, td), 1e-3)
    is_round = ("round" in str(table.get("type", "")).lower()) or aspect < 1.25
    R = np.asarray(table.get("rotation_3x3", np.eye(3).tolist()), float)
    if not is_round:
        # Un-tilt the desk/table: snap its yaw to the nearest 90° so its edges are
        # axis-aligned.  A tilted rectangular dining table is a placement artifact;
        # the chairs must sit SQUARE (perpendicular) to its edges, not to a tilt.
        lf = np.asarray(table.get("_local_front", [0, 0, 1]), float)
        fr = R @ lf
        yaw = float(np.arctan2(fr[0], fr[2]))
        snapped = round(yaw / (np.pi / 2.0)) * (np.pi / 2.0)
        if abs(((yaw - snapped + np.pi) % (2 * np.pi)) - np.pi) > np.radians(3):
            table["rotation_3x3"] = _rot_to_dir(lf.tolist(),
                                                float(np.sin(snapped)), float(np.cos(snapped)))
            R = np.asarray(table["rotation_3x3"], float)
            print(f"[chair_group] un-tilted table idx{table.get('index')}: "
                  f"yaw {np.degrees(yaw):.0f}° → {np.degrees(snapped):.0f}°")
    hx = (abs(R[0, 0]) * tw + abs(R[0, 2]) * td) / 2.0   # world half-extent X
    hz = (abs(R[2, 0]) * tw + abs(R[2, 2]) * td) / 2.0   # world half-extent Z

    # ── recover dropped short-end chairs ─────────────────────────────────────
    # A rectangular dining table expects chairs on its SHORT ends too. The
    # blob-drop / IoU-dedup above can over-cull a real short-end chair when it is
    # seen edge-on (looks "flat") or sits behind a long-side chair (boxes overlap
    # → looks like a duplicate). If a short end is otherwise EMPTY and a dropped
    # candidate back-projects near it, restore it (006375: idx9 right end seen
    # edge-on, idx12 left end behind the front chair). 005440-style true
    # duplicates sit on an already-occupied LONG side → never restored here.
    if "_dropped_chairs" in dir() and _dropped_chairs and not is_round:
        ends = ([np.array([tc[0] - hx, tc[1]]), np.array([tc[0] + hx, tc[1]])]
                if hx >= hz else
                [np.array([tc[0], tc[1] - hz]), np.array([tc[0], tc[1] + hz])])
        def _cxz(o):
            return np.array([float(o["position_m"][0]), float(o["position_m"][2])])
        def _end_empty(e):
            return not any(np.linalg.norm(_cxz(k) - e) < 0.55 for k in chairs)
        def _recover(best, e, tag):
            _dropped_chairs.remove(best)
            chairs.append(best)
            if best not in placements:
                placements.append(best)
            print(f"[chair_group] recovered short-end chair idx{best.get('index')} "
                  f"near table end ({e[0]:.2f},{e[1]:.2f}) — {tag}")
        # Pass 1: a dropped chair that back-projects near an EMPTY short end.
        for e in ends:
            if not _end_empty(e):
                continue
            cand = [d for d in _dropped_chairs
                    if d.get("position_m") and np.linalg.norm(_cxz(d) - e) < 0.85]
            if cand:
                _recover(min(cand, key=lambda d: np.linalg.norm(_cxz(d) - e)),
                         e, "dining set expects it")
        # Pass 2 (symmetric completion): exactly ONE short end ended up filled, the
        # OPPOSITE is still empty, the set is already substantial (>=4 chairs), and a
        # dropped candidate remains → that leftover is the symmetric partner whose
        # base was mis-back-projected toward the filled end (006375: idx9 detected
        # edge-on). Place it at the empty end so the set completes to 2+2+1+1.
        # Guarded so a true-duplicate drop on a 2-chair desk (005440) never fires:
        # there NO short end is filled, so _filled==0.
        _empty = [e for e in ends if _end_empty(e)]
        _filled = len(ends) - len(_empty)
        # Only resurrect a REAL chair for symmetric completion — never a blob-dropped
        # FALSE detection. A 2+2-on-opposite-sides set (short ends empty) must stay 4
        # chairs; resurrecting a flat/symmetric blob turned 003847 into a phantom 5th
        # chair (2-1-1-1). Dedup-dropped chairs are real (detected twice) and pass
        # _is_real_chair; blob-dropped ones fail it and are excluded here.
        _real_dropped = [d for d in _dropped_chairs if _is_real_chair(d)]
        if _empty and _filled == 1 and len(chairs) >= 4 and _real_dropped:
            e = _empty[0]
            best = min(_real_dropped,
                       key=lambda d: (np.linalg.norm(_cxz(d) - e)
                                      if d.get("position_m") else 1e9))
            if best.get("position_m"):
                best["position_m"][0] = float(e[0]); best["position_m"][2] = float(e[1])
                _recover(best, e, "symmetric dining completion")

    # ── mesh unify / borrow ──────────────────────────────────────────────────
    # A matching dining/meeting set is one chair design repeated.  The per-chair
    # 3D-gen produces a slightly different mesh from each crop, so the set renders
    # as N mismatched chairs.  When every chair shares the same TYPE (same style),
    # use ONE representative mesh — the clearest/most-complete one (largest
    # segmentation bbox) — for all of them.  Mixed styles: only fill in a chair
    # whose own mesh is missing (borrow a sibling).
    _attach_boxes(placements, Path(scene))      # recover box_px from segments
    _W = int(cam.get("width_px", 1296)); _H = int(cam.get("height_px", 968))

    def _bbox_area(o):
        b = o.get("box_px")
        return ((b[2] - b[0]) * (b[3] - b[1])) if (b and len(b) == 4) else _footprint(o) * 1e5

    def _clipped(o):
        b = o.get("box_px")
        if not (b and len(b) == 4):
            return False
        return (b[0] <= 0.01 * _W or b[1] <= 0.01 * _H
                or b[2] >= 0.99 * _W or b[3] >= 0.99 * _H)

    valid = [c for c in chairs if c.get("glb_path") and Path(c["glb_path"]).exists()]
    # A footstool / round stool / pouf / ottoman is NOT a chair: its mesh has no
    # backrest, so it must never become the shared chair representative (008868: the
    # whole set rendered as tiny round stools because inpaint_05_footstool won the
    # vote). Exclude stool-type meshes from the representative pool (keep them only
    # if there is genuinely nothing else).
    def _is_stool_mesh(c):
        g = str(c.get("glb_path", "")).lower()
        t = str(c.get("type", "")).lower()
        return any(k in g or k in t for k in ("stool", "footstool", "pouf", "ottoman"))
    valid = [c for c in valid if not _is_stool_mesh(c)] or valid
    if valid:
        # Representative = a REAL chair: it must have a detectable BACKREST
        # (symmetric blobs — a mis-segmented round stool/cushion that 3D-gen'd as
        # a disc — return no seat-front and must never be the shared mesh) AND be
        # a COMPLETE, non-edge-clipped crop (clipped crops 3D-gen as broken
        # meshes).  Among those, the largest/clearest silhouette wins.
        # Real chairs = those with a detectable backrest.  Do NOT hard-exclude an
        # edge-clipped crop here — a clipped crop often still 3D-gens a clean mesh
        # (the best brown is frequently a clipped one); non-clipped is only a soft
        # tie-break below.
        real = [c for c in valid
                if _chair_seat_front_local(c.get("glb_path")) is not None]
        cands = real or valid
        # Hunyuan/3D-gen often tints a chair purple/blue (cool: B>=G).  Drop those
        # colour-artifacted meshes and adopt the WARMEST clean brown (max R-B) — a
        # real wooden/upholstered chair is warm — preferring a non-clipped, clearer
        # silhouette only to break ties.
        colmap = {id(c): _chair_mean_color(c.get("glb_path")) for c in cands}
        warm = [c for c in cands
                if colmap[id(c)] and colmap[id(c)][2] < colmap[id(c)][1]]
        pool = warm or cands

        def _warmth(c):
            col = colmap[id(c)]
            return (col[0] - col[2]) if col else 0.0
        # Representative by SHAPE first, color only as a tie-break (user: prefer the
        # most-similar SHAPE over the most-similar color). Same-style chairs share a
        # design, so pick the mesh whose bbox PROPORTIONS are closest to the group
        # MEDIAN (the typical shape) — a colour-right but oddly-shaped 3D-gen result
        # must not become the shared mesh.
        def _shape(c):
            try:
                import trimesh
                _g = trimesh.load(c.get("glb_path"), force="scene")
                _gl = max(_g.geometry.values(), key=lambda x: len(x.faces))
                e = np.abs(_gl.bounds[1] - _gl.bounds[0])
                h = max(float(e[1]), 1e-3)
                return np.array([float(e[0]) / h, float(e[2]) / h])   # (w/h, d/h)
            except Exception:
                return None
        _shp = {id(c): _shape(c) for c in pool}
        _sv = [s for s in _shp.values() if s is not None]
        if _sv:
            _med = np.median(np.stack(_sv), axis=0)
            rep = min(pool, key=lambda c: (
                round(float(np.linalg.norm(_shp[id(c)] - _med)), 2)
                if _shp[id(c)] is not None else 1e9,
                0 if not _clipped(c) else 1,        # non-clipped next
                -round(_warmth(c) / 15.0),          # warmer color only as a tie-break
                -_bbox_area(c)))
        else:
            rep = max(pool, key=lambda c: (round(_warmth(c) / 15.0),
                                           0 if _clipped(c) else 1, _bbox_area(c)))
        types = {str(c.get("type", "")).lower().replace("-", "_") for c in chairs}
        unify = len(types) == 1
        # Unified SIZE = MEDIAN of the chairs' silhouette-fit heights (robust to an
        # over-scaled rep), realised uniformly on the rep's mesh — so the set is
        # consistent AND matches the reference chair size, not one outlier's.
        rep_scale = list(rep.get("scale", [1, 1, 1]))
        rep_size = dict(rep.get("size_m", {}))
        if unify:
            _hs = [float(c.get("size_m", {}).get("height_m", 0))
                   for c in chairs if c.get("size_m", {}).get("height_m")]
            if _hs:
                try:
                    import trimesh
                    _g = trimesh.load(rep["glb_path"], force="scene")
                    _gl = max(_g.geometry.values(), key=lambda x: len(x.faces))
                    _ext = np.abs(_gl.bounds[1] - _gl.bounds[0])
                    _med_h = float(np.median(_hs))
                    _us = _med_h / max(float(_ext[1]), 1e-3)
                    rep_scale = [_us, _us, _us]
                    rep_size = {"width_m": round(float(_ext[0]) * _us, 3),
                                "height_m": round(_med_h, 3),
                                "depth_m": round(float(_ext[2]) * _us, 3)}
                except Exception:
                    pass
        adopted = 0
        for c in chairs:
            missing = not (c.get("glb_path") and Path(c["glb_path"]).exists())
            if unify or (missing and c is not rep):
                c["glb_path"] = rep["glb_path"]
                c["scale"] = list(rep_scale)
                c["size_m"] = dict(rep_size)
                c["_local_front"] = list(rep.get("_local_front", [0, 0, 1]))
                if c is not rep:
                    adopted += 1
        if unify:
            print(f"[chair_group] same-style set '{list(types)[0]}' → all "
                  f"{len(chairs)} chairs use representative mesh idx{rep.get('index')} "
                  f"({adopted} re-meshed)")
        elif adopted:
            print(f"[chair_group] mixed styles → {adopted} missing chair(s) "
                  f"borrowed mesh from idx{rep.get('index')}")

    chairs = sorted(chairs, key=lambda o: o.get("index", 0))
    n = len(chairs)
    # ── Reconcile chair COUNT to the VLM's read of the REF photo ──────────────
    # The detection count (after dedup/blob-drop/recovery) is unstable run-to-run
    # (008868: 4→3→2). Trust the VLM's direct count of chairs in the reference
    # instead: trim extras (keep the clearest by bbox) or CLONE a chair into the
    # missing seats (the clones inherit the already-unified representative mesh, so
    # an occluded/merged 4th chair is filled rather than lost).
    _vn = _vlm_count_chairs(scene) if use_vlm else None
    if _vn and _vn != n and not is_round:
        if _vn < n:
            chairs = sorted(chairs, key=lambda c: -_bbox_area(c))[:_vn]
            print(f"[chair_group] VLM ref-count {_vn} < detected {n} → trimmed to {_vn}")
        elif chairs:
            import copy as _copy
            _src = max(chairs, key=_bbox_area)
            for _i in range(_vn - n):
                _c = _copy.deepcopy(_src)
                _c["index"] = 9000 + _i
                _c["_cloned_for_count"] = True
                chairs.append(_c)
                placements.append(_c)
            print(f"[chair_group] VLM ref-count {_vn} > detected {n} → cloned "
                  f"{_vn - n} chair(s) into the missing seats")
        chairs = sorted(chairs, key=lambda o: o.get("index", 0))
        n = len(chairs)
    # Depth used to offset each chair from the table edge. Cap it to a realistic
    # dining-chair depth: the raw meshes are frequently oversized BEFORE the
    # per-type scale-clamp shrinks them at render time, so using the raw depth
    # parks the chair too far out and leaves a phantom gap once it shrinks
    # (004286 chairs looked stranded). 0.52 m ≈ a real chair's seat depth.
    cd = float(np.clip(np.median([float(c.get("size_m", {}).get("depth_m", 0.55))
                                  for c in chairs]) or 0.55, 0.40, 0.52))
    arr = _vlm_chair_arrangement(table, chairs, scene, room_w, room_d, is_round) \
        if use_vlm else None
    if isinstance(arr, dict) and arr.get("shape"):
        is_round = arr["shape"].startswith("round")

    # Seating rule: FOUR chairs are arranged two-per-side on two opposite sides
    # (square style, each pair facing across the table) even when the table is
    # round; only 5+ chairs ring a round table radially. Pick the side pair
    # (front/back vs left/right) that is OPEN — clear of walls — so a table near a
    # wall (e.g. 004286 against the left wall) seats its chairs on the free axis.
    # 2 chairs → one on each of two OPPOSITE open sides (facing across); 4 chairs
    # → 2 per side on two opposite open sides. (6+ chairs use _default_sides, which
    # gives 2+2 on the long sides + 1 on each short end.)
    # 2 → 1+1 opposite; 4 → 2+2 opposite; 6 → 2+2 on the long sides + 1 on each
    # short end (the user's "1-2-1-2" arrangement, via _default_sides). Use this
    # SIDE-based layout even for square/low-aspect tables that the shape detector
    # called "round" — only seat chairs radially for a genuinely circular table.
    force_pairs = n in (2, 4, 6)
    force_2x2 = force_pairs   # back-compat name used below
    # Use WORLD half-extents (hx, hz — already rotation-aware) to pick the long
    # side, NOT the local size_m (tw/td): a rotated table's local long axis isn't
    # its world long axis, which put chairs on the SHORT world side and shrank
    # them to it (008868: chairs scaled to 0.26 m). Chairs on front/back run along
    # X → that's the long side when hx >= hz.
    _force_long_is_z = (hx >= hz)
    if force_pairs:
        is_round = False
        _gap2 = CHAIR_TABLE_GAP + cd / 2.0
        _x_fits = (tc[0] - hx - _gap2 > 0.05) and (tc[0] + hx + _gap2 < room_w - 0.05)
        _z_fits = (tc[1] - hz - _gap2 > 0.05) and (tc[1] + hz + _gap2 < room_d - 0.05)
        if _z_fits and not _x_fits:
            _force_long_is_z = True
        elif _x_fits and not _z_fits:
            _force_long_is_z = False
        _cnts = _default_sides(n, _force_long_is_z)
        print(f"[chair_group] {n} chairs → sides {_cnts} (long={'Z' if _force_long_is_z else 'X'}), "
              f"facing across")

    if is_round:
        radius = max(hx, hz) + CHAIR_TABLE_GAP + cd / 2.0
        # angle convention: atan2(x, z) → +Z(front)=0, +X(right)=90°, -X(left)=-90°
        _SECTOR = {"front": 0.0, "right": np.pi / 2, "back": np.pi, "left": -np.pi / 2}
        sides = None
        if isinstance(arr, dict) and isinstance(arr.get("sides"), dict):
            sd = {k: int(arr["sides"].get(k, 0) or 0) for k in _SECTOR}
            if 0 < sum(sd.values()) <= n:
                sides = sd
        it = iter(chairs)
        if sides:
            # Cluster each side's chairs side-by-side within a ~55° arc on that
            # side of the rim (a corner table then seats only its open sides).
            placed = 0
            for side, k in sides.items():
                if k <= 0:
                    continue
                c0 = _SECTOR[side]
                for j in range(k):
                    frac = (j - (k - 1) / 2.0) / max(k, 1)
                    ang = c0 + frac * np.radians(55)
                    try:
                        c = next(it)
                    except StopIteration:
                        break
                    _place_chair_facing(c, tc[0] + radius * np.sin(ang),
                                        tc[1] + radius * np.cos(ang), tc)
                    placed += 1
            for c in list(it):                       # leftovers → even angles
                ang = 2.0 * np.pi * placed / max(n, 1)
                _place_chair_facing(c, tc[0] + radius * np.sin(ang),
                                    tc[1] + radius * np.cos(ang), tc)
                placed += 1
            print(f"[chair_group] round table idx{table.get('index')}: "
                  f"chairs by sector {sides}, facing centre")
        else:
            for i, c in enumerate(chairs):
                ang = 2.0 * np.pi * i / n
                _place_chair_facing(c, tc[0] + radius * np.sin(ang),
                                    tc[1] + radius * np.cos(ang), tc)
            print(f"[chair_group] round table idx{table.get('index')}: "
                  f"{n} chairs spaced {360.0/n:.0f}° apart (even), facing centre")
    else:
        long_is_z = hx >= hz   # world extents (rotation-aware), not local tw/td
        if force_pairs:
            # 2 or 4 chairs: force them onto the open opposite sides (1+1 or 2+2);
            # ignore the VLM 'sides' (computed for a round ring) which would scatter.
            long_is_z = _force_long_is_z
            counts = _default_sides(n, long_is_z)
        elif isinstance(arr, dict) and isinstance(arr.get("sides"), dict):
            sd = arr["sides"]
            counts = {k: int(sd.get(k, 0) or 0) for k in ("front", "back", "left", "right")}
            if sum(counts.values()) != n:        # VLM miscount → fall back
                counts = _default_sides(n, long_is_z)
        else:
            counts = _default_sides(n, long_is_z)
        # Scale chairs DOWN to fit the table when they're oversized for it: a
        # small table with 2 chairs per side otherwise looks crowded AND the
        # chairs overlap the rim, which makes the collision pass shove the table
        # off-centre. Keep the set proportional — the chairs on the busiest side
        # should span about the table's side length. Only shrinks (never grows),
        # and locks the scale so the silhouette re-fit won't re-inflate them.
        cw0 = float(np.median([float(c.get("size_m", {}).get("width_m", 0.5))
                               for c in chairs]) or 0.5)
        _kmax = max(counts.values()) if counts else 1
        _busy_fb = max(counts.get("front", 0), counts.get("back", 0))
        _busy_lr = max(counts.get("left", 0), counts.get("right", 0))
        _spread_half = hx if _busy_fb >= _busy_lr else hz
        _table_side = 2.0 * _spread_half
        # Target the k chairs on the busiest side spanning ~0.8× the table side.
        # The 0.8 leaves margin AND compensates for the table shrinking at render
        # (it's oversized here at chair-group time but clamped smaller later), so
        # the chairs end up proportional to the RENDERED table, not this one.
        fit_scale = min(1.0, 0.80 * _table_side / max(_kmax * cw0, 1e-3))
        if fit_scale < 0.95:
            for c in chairs:
                _sc = np.array(c.get("scale", [1.0, 1.0, 1.0]), float)
                if _sc.ndim == 0:
                    _sc = np.array([float(_sc)] * 3)
                c["scale"] = list(_sc * fit_scale)
                _sm = dict(c.get("size_m", {}) or {})
                for _k in ("width_m", "depth_m", "height_m"):
                    if _k in _sm:
                        _sm[_k] = float(_sm[_k]) * fit_scale
                c["size_m"] = _sm
                c["_chair_group_scaled"] = True
            cd = float(min(np.median([float(c.get("size_m", {}).get("depth_m", 0.52))
                                      for c in chairs]) or 0.52, 0.52))
            print(f"[chair_group] chairs scaled ×{fit_scale:.2f} to fit table "
                  f"(side {_table_side:.2f} m, {_kmax}/side, chair w {cw0:.2f} m)")
        # side → (fixed-axis position, spread-axis, spread half-range, inward sign)
        edge = {
            "front": ("z", tc[1] + hz + CHAIR_TABLE_GAP + cd / 2.0, "x", hx),
            "back":  ("z", tc[1] - hz - CHAIR_TABLE_GAP - cd / 2.0, "x", hx),
            "right": ("x", tc[0] + hx + CHAIR_TABLE_GAP + cd / 2.0, "z", hz),
            "left":  ("x", tc[0] - hx - CHAIR_TABLE_GAP - cd / 2.0, "z", hz),
        }
        # inward (toward table), perpendicular to each edge
        _INWARD = {"front": (0.0, -1.0), "back": (0.0, 1.0),
                   "right": (-1.0, 0.0), "left": (1.0, 0.0)}
        cw = float(np.median([float(c.get("size_m", {}).get("width_m", 0.5))
                              for c in chairs]) or 0.5)
        it = iter(chairs)
        for side, k in counts.items():
            if k <= 0:
                continue
            fixed_axis, fixed_val, spread_axis, half = edge[side]
            centre = tc[0] if spread_axis == "x" else tc[1]
            # Tight packing: space chairs by their own width (+small gap), centred
            # on the side, capped so they stay within the table edge.
            pitch = min(cw + 0.06, (2.0 * half) / max(k, 1))
            for j in range(k):
                spread = centre + (j - (k - 1) / 2.0) * pitch
                try:
                    c = next(it)
                except StopIteration:
                    break
                if fixed_axis == "z":
                    px, pz = spread, fixed_val
                else:
                    px, pz = fixed_val, spread
                # face SQUARE to the edge (perpendicular), pointing at the table
                _place_chair_facing(c, px, pz, tc, face_dir=_INWARD[side])
        print(f"[chair_group] rect table idx{table.get('index')}: chairs by side "
              f"{counts} (long={'Z' if long_is_z else 'X'}), facing centre")

    # keep chairs inside the room
    for c in chairs:
        c["position_m"][0] = float(np.clip(c["position_m"][0], 0.2, room_w - 0.2))
        c["position_m"][2] = float(np.clip(c["position_m"][2], 0.2, room_d - 0.2))


def apply(placements, cam, scene: Path, room_w: float, room_d: float,
          gap_table_m: float = GAP_TABLE_M) -> None:
    """Full correction: anchor the seating group, then separate collisions."""
    try:
        anchor_group(placements, cam, Path(scene), gap_table_m=gap_table_m)
        # Dining/meeting chair groups: table-first, then arrange chairs around it.
        arrange_chair_group(placements, cam, Path(scene), room_w, room_d)
        resolve_collisions(placements, room_w, room_d)
        # Whole-layout VLM fallback for overlaps the local push can't escape
        # (e.g. a free object wedged in a corner that must relocate across the room).
        _vlm_resolve_collisions(placements, cam, Path(scene), room_w, room_d)
        # Chair-group is authoritative for dining chairs: restore each to its
        # arranged spot so the collision/VLM rounds above can't stagger a pair
        # (the chairs are intentionally tight around the table — overlap there is
        # expected, not a conflict to resolve).
        _restored = 0
        for c in placements:
            tgt = c.get("_dining_target_xz")
            if tgt is not None:
                if abs(c["position_m"][0] - tgt[0]) > 1e-3 or abs(c["position_m"][2] - tgt[1]) > 1e-3:
                    _restored += 1
                c["position_m"][0] = float(tgt[0])
                c["position_m"][2] = float(tgt[1])
        if _restored:
            print(f"[chair_group] restored {_restored} dining chair(s) to arranged spot "
                  f"(undo collision/VLM nudge)")

        # ── General silhouette clamp (silhouette-authority) ───────────────────
        # No layout/seating/collision pass may teleport a FREE-STANDING object far
        # from where its segmentation mask placed it (the silhouette is authority).
        # anchor_group/collision/VLM rounds were silently relocating pieces — a left
        # lounge chair to the far right (005476), and tall objects (plant, shelf)
        # off their silhouette. Snap any centre/free-standing piece that drifted
        # > _SIL_MAXDEV from its _mask_origin_xz back to the mask spot.
        # Centre pieces: snap BOTH axes back to the mask. Wall-flush pieces
        # (back/left/right): keep the wall-perpendicular snap but clamp the
        # ALONG-wall coordinate to the mask, so a cabinet/shelf can't drift along
        # its wall into a corner (005241). Exempt: dining set (restored above) and
        # TABLES (a coffee/side table is intentionally anchored beside/in-front-of
        # seating by anchor_group).
        _SIL_MAXDEV = 1.0
        _TABLE_T = ("coffee_table", "cocktail_table", "side_table", "end_table",
                    "console_table", "nesting", "dining_table", "round_table",
                    "pedestal_table")
        _snapped = 0
        for o in placements:
            if o.get("is_carpet") or o.get("wall_mounted"):
                continue
            if o.get("_dining_target_xz") is not None or o.get("_anchor_target_xz") is not None:
                continue
            _ot = str(o.get("type", "")).lower().replace("-", "_")
            if any(k in _ot for k in _TABLE_T):   # tables are intentionally anchored
                continue
            mo = o.get("_mask_origin_xz")
            if mo is None:
                continue
            _wa = o.get("wall_affinity")
            if _wa == "back":
                # along-wall = X; clamp X to mask, keep wall-snapped Z
                if abs(float(o["position_m"][0]) - float(mo[0])) > _SIL_MAXDEV:
                    o["position_m"][0] = float(mo[0]); _snapped += 1
                    print(f"[seating] idx{o.get('index')} ({_ot}) slid along back wall to "
                          f"silhouette X={mo[0]:.2f} (was drifting toward a corner)")
            elif _wa in ("left", "right"):
                # along-wall = Z; clamp Z to mask, keep wall-snapped X
                if abs(float(o["position_m"][2]) - float(mo[1])) > _SIL_MAXDEV:
                    o["position_m"][2] = float(mo[1]); _snapped += 1
                    print(f"[seating] idx{o.get('index')} ({_ot}) slid along {_wa} wall to "
                          f"silhouette Z={mo[1]:.2f} (was drifting toward a corner)")
            else:
                dx = float(o["position_m"][0]) - float(mo[0])
                dz = float(o["position_m"][2]) - float(mo[1])
                if (dx * dx + dz * dz) ** 0.5 > _SIL_MAXDEV:
                    o["position_m"][0] = float(mo[0])
                    o["position_m"][2] = float(mo[1])
                    _snapped += 1
                    print(f"[seating] idx{o.get('index')} ({_ot}) snapped back to silhouette "
                          f"({mo[0]:.2f},{mo[1]:.2f}) — a layout pass had teleported it")
        if _snapped:
            print(f"[seating] silhouette clamp: snapped {_snapped} seating piece(s) back to mask")

        # ── Final de-collision: free-standing floor object embedded in a seat ──
        # A tall floor object whose BASE is occluded by a sofa/bed back-projects
        # its silhouette INTO the seat footprint (006476: a floor plant landing in
        # the sofa cushions), and the silhouette clamp above re-pins it there. Last
        # pass: if such an object's CENTER sits INSIDE a seat footprint (genuinely
        # embedded — not a nightstand/ottoman merely touching an edge), slide it to
        # the seat edge its MASK BOX overflows (else nearest in-bounds edge). The
        # exit is kept in-bounds so the downstream bounds_clamp won't pull it back.
        _SEAT_T = ("sofa", "couch", "sectional", "loveseat", "settee", "bed", "daybed")
        def _aabb_xz(o):
            sz = o.get("size_m") or {}
            w = float(sz.get("width_m", 0)); d = float(sz.get("depth_m", 0))
            if w <= 0 or d <= 0:
                return None
            R = o.get("rotation_3x3")
            if R is not None:
                c = abs(float(R[0][0])); s = abs(float(R[0][2]))
                hw = (c * w + s * d) / 2.0; hd = (s * w + c * d) / 2.0
            else:
                hw, hd = w / 2.0, d / 2.0
            x = float(o["position_m"][0]); z = float(o["position_m"][2])
            return [x - hw, z - hd, x + hw, z + hd, hw, hd]
        _seats = [o for o in placements
                  if any(k in str(o.get("type", "")).lower() for k in _SEAT_T)
                  and _aabb_xz(o) is not None]
        for o in placements:
            if o.get("is_carpet") or o.get("wall_mounted"):
                continue
            _ot = str(o.get("type", "")).lower().replace("-", "_")
            if any(k in _ot for k in _SEAT_T) or any(k in _ot for k in _TABLE_T):
                continue
            if "nightstand" in _ot or "bedside" in _ot or "ottoman" in _ot:
                continue   # these legitimately abut a seat
            if o.get("_dining_target_xz") is not None:
                continue
            oa = _aabb_xz(o)
            if oa is None:
                continue
            for s in _seats:
                sa = _aabb_xz(s)
                if sa is None:
                    continue
                cur_x = float(o["position_m"][0]); cur_z = float(o["position_m"][2])
                if not (sa[0] < cur_x < sa[2] and sa[1] < cur_z < sa[3]):
                    continue   # center not embedded in this seat — leave it
                gap = 0.04
                # These free-standing pieces (plant / lamp / stool) are small and get
                # scale-clamped smaller later; at THIS stage they can be transiently
                # oversized, which wrongly rejects the beside-the-seat exit. Use a
                # capped effective half-extent so a narrow object can still tuck
                # against the wall beside the seat (006476 plant → right of armrest).
                ehw = min(oa[4], 0.5); ehd = min(oa[5], 0.5)
                def _exit(side):
                    if side == "right":
                        x = min(sa[2] + ehw + gap, room_w - ehw - 0.02)
                        return (x, cur_z) if (x - ehw) >= (sa[2] - 0.05) else None
                    if side == "left":
                        x = max(sa[0] - ehw - gap, ehw + 0.02)
                        return (x, cur_z) if (x + ehw) <= (sa[0] + 0.05) else None
                    if side == "back":
                        z = min(sa[3] + ehd + gap, room_d - ehd - 0.02)
                        return (cur_x, z) if (z - ehd) >= (sa[3] - 0.05) else None
                    # front
                    z = max(sa[1] - ehd - gap, ehd + 0.02)
                    return (cur_x, z) if (z + ehd) <= (sa[1] + 0.05) else None
                cand = {k: _exit(k) for k in ("right", "left", "back", "front")}
                ok = {k: v for k, v in cand.items() if v is not None}
                if not ok:
                    continue
                pref = None
                ob = o.get("box_px"); sb = s.get("box_px")
                if ob and sb:
                    over_r = ob[2] - sb[2]   # >0 → object extends right of seat in image
                    over_l = sb[0] - ob[0]   # >0 → object extends left of seat in image
                    if over_r > 20 and over_r >= over_l and "right" in ok:
                        pref = "right"
                    elif over_l > 20 and over_l > over_r and "left" in ok:
                        pref = "left"
                if pref is None:
                    # Prefer a LATERAL exit (beside the seat) over front/back — a
                    # plant beside a sofa reads far better than one in front of it.
                    # Break ties by nearest displacement.
                    _lat = [k for k in ok if k in ("left", "right")]
                    _pool = _lat if _lat else list(ok)
                    pref = min(_pool, key=lambda k: (ok[k][0] - cur_x) ** 2 + (ok[k][1] - cur_z) ** 2)
                nx, nz = ok[pref]
                o["position_m"][0] = nx; o["position_m"][2] = nz
                o["_anchor_target_xz"] = [round(nx, 3), round(nz, 3)]
                print(f"[seating] de-collide idx{o.get('index')} ({_ot}) embedded in "
                      f"{s.get('type')} → {pref} edge ({nx:.2f},{nz:.2f})")
                oa = _aabb_xz(o)   # refresh for the next seat
    except Exception as e:               # never let a layout heuristic break placement
        print(f"[seating] skipped ({e})")
