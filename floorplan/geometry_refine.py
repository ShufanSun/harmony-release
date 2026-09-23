"""Code-driven geometry refinement for Front3D scenes — replaces hand-editing
walls.obj / camera_vggt.json / walls_metadata.json.

Given a scene (camera_vggt.json + walls.obj + vggt/depth_0.npy) it:
  1. Rebuilds walls_metadata.json if it is corrupt (a camera-json copy, which
     silently crashes render_room — keys must be ⊆ {back,left,right,front,
     ceiling,floor}).
  2. Estimates the room box extents (x_lo,x_hi,z_lo,z_hi and ceiling height):
       - robust wall-plane fit from the back-projected depth + surface normals
         (histogram-peak per wall — furniture spreads out, the wall is a sharp
         peak; plane *intersections* give corners, so corner rooms are NOT
         flattened) WHEN the depth is reliable;
       - otherwise keep the calibrated extents (don't guess) and only do the
         deterministic fixes below.
  3. Enforces camera-inside-box (if position_m is outside in x or z the floor
     backprojection samples the wall) by extending the violated bound.
  4. Re-origins so the box starts at [0,0,0] (texture pipeline assumes it),
     applying ONE rigid transform consistently to walls.obj verts, the camera
     position_m/look_at_m, and the metadata lengths — no scattered edits.

CLI:  python -m floorplan.geometry_refine <scene_dir> [--dry-run] [--no-depth]
"""
import os
import json
import argparse
import numpy as np

from floorplan.depth_wall_refine import (
    _load_camera, _ray_dirs, _box_bounds, _predict_depth, _align_depth,
    _point_normals,
)

PERWALL_KEYS = {"back", "left", "right", "front", "ceiling", "floor"}
CAM_MARGIN = 0.15          # keep camera this far inside an extended wall
DEPTH_RESID_OK = 0.06      # m — above this the depth-derived extents are distrusted
DEPTH_SCALE_MIN = 1.0      # affine scale below this ⇒ depth fit failed
MIN_WALL_PX = 6000         # a side-wall plane needs this many pixels to be trusted
                           # (a grazing wall fit from a few hundred px is noise)
MAX_EXTENT_DELTA = 1.5     # don't move an extent more than this from the calibration


# ── metadata ────────────────────────────────────────────────────────────────
def _metadata_is_corrupt(scene):
    mp = os.path.join(scene, "walls_metadata.json")
    if not os.path.exists(mp):
        return True
    try:
        m = json.load(open(mp))
    except Exception:
        return True
    return not (isinstance(m, dict) and set(m.keys()) <= PERWALL_KEYS)


def _strip_openings(scene, dry_run=False):
    """Remove window/door features from walls_metadata.json AND camera_vggt.json
    wall_context. The empty-room reference we backproject from has its doors and
    windows removed, so leaving phantom opening features makes the texture
    pipeline tile/fill spurious frames. Done in code so no scene needs a manual
    JSON edit before backprojection."""
    removed = 0
    for fn in ("walls_metadata.json", "camera_vggt.json"):
        fp = os.path.join(scene, fn)
        if not os.path.exists(fp):
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        # per-wall dicts live at the top level (metadata) or under wall_context (camera)
        ctx = d if set(d.keys()) <= PERWALL_KEYS else d.get("wall_context", {})
        if not isinstance(ctx, dict):
            continue
        touched = False
        for w in ("back", "left", "right", "front"):
            wd = ctx.get(w)
            if isinstance(wd, dict) and wd.get("features"):
                keep = [f for f in wd["features"]
                        if not (isinstance(f, dict) and f.get("type") in ("window", "door"))]
                if len(keep) != len(wd["features"]):
                    removed += len(wd["features"]) - len(keep)
                    wd["features"] = keep
                    touched = True
        if touched and not dry_run:
            json.dump(d, open(fp, "w"), indent=1)
    return removed


def _rebuild_metadata(scene, lo, hi, dry_run=False):
    mp = os.path.join(scene, "walls_metadata.json")
    W = round(float(hi[0] - lo[0]), 3)
    D = round(float(hi[2] - lo[2]), 3)
    H = round(float(hi[1] - lo[1]), 3)
    m = {
        "back":  {"length_m": W, "height_m": H, "features": [], "tile_size_m": W},
        "left":  {"length_m": D, "height_m": H, "features": [], "tile_size_m": D},
        "right": {"length_m": D, "height_m": H, "features": [], "tile_size_m": D},
        "front": {"length_m": W, "height_m": H, "features": [], "tile_size_m": W},
        "ceiling": {"tile_size_m": 2.0},
        "floor": {"tile_size_m": W},
    }
    if not dry_run:
        if os.path.exists(mp):
            os.replace(mp, mp + ".corrupt_bak")
        json.dump(m, open(mp, "w"), indent=2)
    return m


# ── robust wall-plane fit from depth ────────────────────────────────────────
def _hist_peak(vals, lo, hi, bins=120, min_n=300):
    """Robust plane coordinate: histogram peak then median within the peak bin.
    A planar wall makes a sharp peak; furniture/occluders spread out and are
    rejected."""
    v = vals[np.isfinite(vals)]
    if v.size < min_n:
        return None, 0
    h, e = np.histogram(v, bins=bins, range=(lo, hi))
    i = int(np.argmax(h))
    c = (e[i] + e[i + 1]) / 2
    w = (e[1] - e[0]) * 1.5
    sel = v[np.abs(v - c) < w]
    return (float(np.median(sel)) if sel.size > 50 else float(c)), int(sel.size)


def _fit_planes(scene):
    """Back-project depth, classify pixels by surface normal axis + position,
    robust-fit each wall plane. Returns (planes, depth_resid, scale)."""
    cam, W, H, pos, fwd, right, up, fx, fy = _load_camera(scene)
    dirs = _ray_dirs(W, H, pos, fwd, right, up, fx, fy)
    lo, hi = _box_bounds(scene)
    Dpred, wid = _predict_depth(dirs, pos, fwd, lo, hi)
    dv = np.load(os.path.join(scene, "vggt", "depth_0.npy")).astype(float)
    valid = np.isfinite(Dpred) & (Dpred > 0.1) & (dv > 0.05)
    Dmeas, a, b, inl = _align_depth(dv, Dpred, valid)
    resid = float(np.std((Dmeas - Dpred)[inl])) if inl.any() else 9.0
    valid &= np.isfinite(Dmeas) & (Dmeas > 0.1) & (Dmeas < 30)
    dirfwd = (dirs * fwd).sum(2)
    P = pos[None, None, :] + (Dmeas / dirfwd)[..., None] * dirs
    n, nok = _point_normals(P, valid)
    ax = np.argmax(np.abs(n), 2)
    mag = np.max(np.abs(n), 2)
    ok = valid & nok & (mag > 0.6)
    midz = (lo[2] + hi[2]) / 2
    midx = (lo[0] + hi[0]) / 2
    backw = ok & (ax == 2) & (P[..., 2] < midz)        # by POSITION not normal sign
    leftw = ok & (ax == 0) & (P[..., 0] < midx)
    rightw = ok & (ax == 0) & (P[..., 0] >= midx)
    back_z, nb = _hist_peak(P[..., 2][backw], lo[2] - 1.0, hi[2], 120)
    left_x, nl = _hist_peak(P[..., 0][leftw], lo[0] - 3.0, hi[0], 160)
    right_x, nr = _hist_peak(P[..., 0][rightw], lo[0], hi[0] + 3.0, 160)
    # ceiling height: y-up surfaces high in the room
    ceilw = ok & (ax == 1) & (P[..., 1] > (lo[1] + hi[1]) * 0.6)
    ceil_y, nc = _hist_peak(P[..., 1][ceilw], (lo[1] + hi[1]) * 0.5, hi[1] + 1.5, 100)
    # EDGE-CALC: where the back-wall plane (z=z_lo) meets the L/R image edges,
    # plus whether the BACK wall (not a side wall) actually reaches that edge —
    # so a frontal back wall running off-frame is extended, but a corner is not.
    # Use the BOX back-wall plane z=lo[2] (the plane render_room actually draws),
    # NOT the depth-measured back_z — when depth scale is off (rgb_008298 back_z
    # 0.5 vs box 0) the edge extent must still match the rendered geometry.
    zb = lo[2]

    def _xedge(u):
        ndcx = (u + 0.5 - W / 2) / fx
        d = ndcx * right + fwd
        t = (zb - pos[2]) / d[2]
        return float(pos[0] + t * d[0]) if t > 0 else None
    ew = max(8, int(0.04 * W))                 # edge strip width
    midrow = slice(int(0.2 * H), int(0.7 * H))
    reachL = backw[midrow, :ew].mean() > 0.25  # back wall dominates the left edge column
    reachR = backw[midrow, W - ew:].mean() > 0.25
    planes = {"back_z": back_z, "left_x": left_x, "right_x": right_x, "ceil_y": ceil_y,
              "n_back": nb, "n_left": nl, "n_right": nr, "n_ceil": nc,
              "xL_edge": _xedge(0), "xR_edge": _xedge(W - 1),
              "reachL": bool(reachL), "reachR": bool(reachR)}
    return planes, resid, float(a)


# ── apply (atomic, re-origined) ─────────────────────────────────────────────
def _apply(scene, new_lo, new_hi, dry_run=False):
    """Re-origin so min=0 and write the transform consistently to walls.obj,
    camera_vggt.json (position/look_at + wall_context lengths) and
    walls_metadata.json (lengths/heights). new_lo/new_hi are the target box
    bounds in the CURRENT world frame (may be negative)."""
    shift = -np.asarray(new_lo, float)            # so min → 0
    new_lo = np.asarray(new_lo, float) + shift
    new_hi = np.asarray(new_hi, float) + shift
    op = os.path.join(scene, "walls.obj")
    old = _box_bounds(scene)                       # (lo,hi) current verts
    olo, ohi = old
    lines = []
    for l in open(op):
        if l.startswith("v "):
            p = [float(t) for t in l.split()[1:4]]
            for k in range(3):
                # remap the box-corner coords to the new bounds, then shift
                if abs(p[k] - olo[k]) < 1e-3:
                    p[k] = float(new_lo[k]) - shift[k]
                elif abs(p[k] - ohi[k]) < 1e-3:
                    p[k] = float(new_hi[k]) - shift[k]
                p[k] += shift[k]
            lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            lines.append(l)
    Wn = float(new_hi[0] - new_lo[0]); Dn = float(new_hi[2] - new_lo[2]); Hn = float(new_hi[1] - new_lo[1])
    cam = json.loads(open(os.path.join(scene, "camera_vggt.json")).read())
    cam["position_m"] = (np.array(cam["position_m"], float) + shift).tolist()
    cam["look_at_m"] = (np.array(cam["look_at_m"], float) + shift).tolist()
    wc = cam.get("wall_context", {})
    if isinstance(wc.get("back"), dict) and wc["back"].get("length_m"):
        wc["back"]["length_m"] = round(Wn, 3)
    mp = os.path.join(scene, "walls_metadata.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else None
    if isinstance(meta, dict) and set(meta.keys()) <= PERWALL_KEYS:
        for w, L in (("back", Wn), ("front", Wn), ("left", Dn), ("right", Dn)):
            if w in meta and isinstance(meta[w], dict):
                if meta[w].get("length_m") is not None:
                    meta[w]["length_m"] = round(L, 3)
                if meta[w].get("tile_size_m") is not None and w in ("back", "front"):
                    meta[w]["tile_size_m"] = round(Wn, 3)
                if meta[w].get("height_m") is not None:
                    meta[w]["height_m"] = round(Hn, 3)
    if not dry_run:
        # one-time backup so the code's transform is always restorable
        for f in (op, os.path.join(scene, "camera_vggt.json"), mp):
            if os.path.exists(f) and not os.path.exists(f + ".prerefine"):
                import shutil
                shutil.copy2(f, f + ".prerefine")
        open(op, "w").writelines(lines)
        json.dump(cam, open(os.path.join(scene, "camera_vggt.json"), "w"), indent=1)
        if meta is not None:
            json.dump(meta, open(mp, "w"), indent=2)
    return new_lo, new_hi, shift


# ── orchestration ───────────────────────────────────────────────────────────
def refine_geometry(scene, dry_run=False, use_depth=True, max_iters=4,
                    strip_openings=True):
    """Iterate the single-pass refine to convergence. The depth fit is only
    trustworthy when the camera is INSIDE the box; on the raw calibration the
    camera is often outside, so pass 1 fixes camera-inside (re-origin) and the
    depth extents are then re-evaluated on the corrected box in pass 2 (e.g.
    rgb_005476: pass1 distrusts depth + re-origins, pass2 fires edge-R). The box
    grows monotonically (extend-only), so this converges; max_iters caps it.

    strip_openings: remove window/door features from the metadata. True for the
        empty-room reference pipeline (its backprojection source has no openings);
        pass False from the full pipeline (main), which detects and places
        wall-mounted windows/doors downstream and must keep those features."""
    reports = []
    for _ in range(max_iters):
        rep = _refine_once(scene, dry_run=dry_run, use_depth=use_depth,
                           strip_openings=strip_openings)
        reports.append(rep)
        if dry_run or rep.get("applied") is None:   # dry_run can't iterate; or converged
            break
    final = reports[-1]
    if len(reports) > 1:
        final["box_old"] = reports[0]["box_old"]    # original calibration box
        final["actions"] = [a for r in reports for a in r["actions"]]
        final["iters"] = len(reports)
    return final


def _refine_once(scene, dry_run=False, use_depth=True, strip_openings=True):
    report = {"scene": os.path.basename(scene.rstrip("/")), "actions": []}
    lo, hi = _box_bounds(scene)
    cam = json.loads(open(os.path.join(scene, "camera_vggt.json")).read())
    pos = np.array(cam["position_m"], float)

    # 1. metadata sanity
    if _metadata_is_corrupt(scene):
        _rebuild_metadata(scene, lo, hi, dry_run)
        report["actions"].append("rebuilt corrupt walls_metadata.json")

    # 1b. strip phantom window/door features (empty-room ref has none) — in code,
    #     so backprojection needs no manual JSON edit.  Skipped for the full
    #     pipeline, which keeps openings for wall-mounted placement.
    nrm = _strip_openings(scene, dry_run) if strip_openings else 0
    if nrm:
        report["actions"].append(f"stripped {nrm} window/door feature(s)")

    new_lo = lo.copy(); new_hi = hi.copy()

    # Visible-corner gate: manhattan_reference.png (the calibration box) already
    # places any IN-FRAME wall corner correctly. If the current left/right back
    # corner projects inside the frame, the calibration captured a real corner
    # there — extending that side outward would shove the corner off-frame and
    # flatten the room (rgb_004411 left, rgb_004587 left+right). Only a side
    # whose corner is off-frame (wall runs to/past the image edge) may be
    # extended. Use the ORIGINAL calibrated box for this test.
    _W = int(cam["width_px"]); _H = int(cam["height_px"])
    _pos = np.array(cam["position_m"], float); _look = np.array(cam["look_at_m"], float)
    _fx = _W / (2 * np.tan(np.radians(cam["hfov_deg"] / 2)))
    _fwd = _look - _pos; _fwd /= np.linalg.norm(_fwd)
    # EXACT render_room mesh projection (_camera_axes): full fwd + stored up, so
    # every projected-corner decision matches what render_room rasterizes. (The
    # tilt model in render_room is only for texture-UV sampling, not geometry.)
    _up = np.array(cam.get("up", [0.0, 1.0, 0.0]), float)
    _right = np.cross(_fwd, _up); _rn = np.linalg.norm(_right)
    _right = _right / _rn if _rn > 1e-9 else np.array([1.0, 0.0, 0.0])
    _up_c = np.cross(_right, _fwd); _up_c /= np.linalg.norm(_up_c)
    _ymid = (lo[1] + hi[1]) / 2

    def _imgx(x, z, y=None):
        d = np.array([x, _ymid if y is None else y, z], float) - _pos
        zc = d @ _fwd
        return (_W / 2 + _fx * (d @ _right) / zc) / _W if zc > 1e-3 else None

    def _imgy(x, y, z):
        d = np.array([x, y, z], float) - _pos
        zc = d @ _fwd
        return (_H / 2 - _fx * (d @ _up_c) / zc) / _H if zc > 1e-3 else None
    _xl = _imgx(lo[0], lo[2]); _xr = _imgx(hi[0], lo[2])
    visL = _xl is not None and 0.03 < _xl < 0.97   # calibration already shows a left corner
    visR = _xr is not None and 0.03 < _xr < 0.97
    report["visible_corner"] = {"left": bool(visL), "right": bool(visR),
                                "img_x_left": None if _xl is None else round(_xl, 3),
                                "img_x_right": None if _xr is None else round(_xr, 3)}

    # 2. depth-based extents (plane-intersection = no flattening)
    if use_depth:
        planes, resid, scale = _fit_planes(scene)
        report["depth"] = {"resid_sd": round(resid, 3), "scale": round(scale, 2), "planes": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in planes.items()}}
        # `reliable` (strict): trust measured side-wall PLANES — needs a low global
        # residual. `back_ok` (loose): the BACK-wall fit alone is trustworthy even
        # when furniture inflates the global residual (many back pixels + sane
        # back_z); enough for the edge-calc, which depends only on back_z + camera.
        back_ok = (planes["back_z"] is not None and abs(planes["back_z"] - lo[2]) < 0.4
                   and (planes["n_back"] or 0) > 50000 and scale >= DEPTH_SCALE_MIN)
        reliable = back_ok and resid <= DEPTH_RESID_OK
        applied = []
        if reliable:
            # EXTEND-ONLY: depth may only grow the box, never shrink it (a
            # furniture-contaminated fit would cut real wall). Plane extends are
            # gated by the visible-corner test — never push an in-frame corner
            # off-frame (rgb_004587 left).
            # A VERY strong side-wall plane (≫ furniture pixel counts) overrides
            # the visible-corner gate: when depth reliably sees a big receding side
            # wall BEYOND the box edge, the calibration's in-frame corner is simply
            # short and the back wall must extend out to that real wall (rgb_007467
            # left_x -1.52, n_left 71k → corner moves 0.382→0.169, matching the
            # photo). The 40k threshold keeps furniture out (rgb_004587 lamp n 14k).
            strongL = (planes["n_left"] or 0) > 40000
            strongR = (planes["n_right"] or 0) > 40000
            if ((not visL or strongL) and planes["left_x"] is not None and planes["n_left"] > MIN_WALL_PX
                    and planes["left_x"] < lo[0] - 0.05
                    and lo[0] - planes["left_x"] < (2.0 if strongL else MAX_EXTENT_DELTA)):
                new_lo[0] = planes["left_x"]; applied.append("left")
            if ((not visR or strongR) and planes["right_x"] is not None and planes["n_right"] > MIN_WALL_PX
                    and planes["right_x"] > hi[0] + 0.05
                    and planes["right_x"] - hi[0] < (2.0 if strongR else MAX_EXTENT_DELTA)):
                new_hi[0] = planes["right_x"]; applied.append("right")
            if (planes["ceil_y"] is not None and planes["n_ceil"] > MIN_WALL_PX
                    and planes["ceil_y"] > hi[1] + 0.1
                    and planes["ceil_y"] - hi[1] < MAX_EXTENT_DELTA):
                new_hi[1] = planes["ceil_y"]; applied.append("ceiling")
        # Edge-calc needs only the box back-wall plane + camera + the normal-based
        # reach signal — NOT a trustworthy back_z. So allow it whenever the back
        # wall is strongly frontal (many back-normal pixels), even if back_ok is
        # False because the depth scale/offset is wrong (rgb_008298).
        if back_ok or (planes["n_back"] or 0) > 150000:
            # EDGE-CALC — a FRONTAL back wall that calibration cut short: depth
            # says the back wall reaches the frame edge (reach*), yet the box puts
            # a corner well inside the frame. That "corner" is spurious unless a
            # REAL side wall sits there. A side-wall plane is real only if it has
            # enough pixels AND lies at/beyond the box edge; if those "side" pixels
            # sit well INSIDE the room they are furniture (a chair/screen), not a
            # wall — n_side alone can't tell them apart, the plane POSITION can:
            #   rgb_004411 left_x 0.04 ≈ edge → real wall → protect (keep corner)
            #   rgb_004587 left_x -1.07 beyond edge → real wall → protect
            #   rgb_005011 left_x 1.56 / rgb_004980 left_x 1.57 → 1.5 m inside →
            #     furniture, not a wall → extend the frontal back wall to the edge.
            # A genuine side-wall corner sits in a BAND around the box edge. A
            # plane well INSIDE the room is furniture (rgb_005011/004980); a plane
            # well BEYOND the box edge means the box is too narrow and the back
            # wall should still extend out to it (rgb_005516 right_x 5.39 vs hi
            # 4.8). Only a plane near the edge is a real corner to protect. (A real
            # corner that genuinely sits beyond — a thin side sliver — has
            # reach*=False, which independently blocks the edge extend.)
            realLeft = ((planes["n_left"] or 0) >= MIN_WALL_PX and planes["left_x"] is not None
                        and abs(planes["left_x"] - lo[0]) <= 0.5)
            realRight = ((planes["n_right"] or 0) >= MIN_WALL_PX and planes["right_x"] is not None
                         and abs(planes["right_x"] - hi[0]) <= 0.5)
            # FRONTAL-DOMINANT (n_back ≫ n_sides AND back wall reaches both edges):
            # the room is a flat frontal wall, not a corner. Any weak side-plane
            # pixels (even sitting near an edge — rgb_007406 left_x 0.02 n 6.6k) are
            # noise/furniture, NOT real corners. WIDEN the back wall to the frame
            # edges (edge-calc) to fill the correct length while KEEPING the
            # calibrated depth/orthography — what the user asked for. (Earlier this
            # rescaled DEPTH instead, which wrongly changed the orthography.)
            # Require BOTH side walls weak — a true frontal room has no strong side
            # wall. (rgb_004411 is a CORNER with a 46k-px real left wall yet a huge
            # back wall, so an n_back/n_sides ratio test wrongly called it frontal
            # and flattened its corner; the max-side-wall test excludes it.)
            _nb = planes["n_back"] or 0
            _maxside = max(planes["n_left"] or 0, planes["n_right"] or 0)
            if planes["reachL"] and planes["reachR"] and _nb > 150000 and _maxside < 20000:
                realLeft = realRight = False
            report["depth"]["real_side"] = {"left": bool(realLeft), "right": bool(realRight)}
            if (planes["reachL"] and planes["xL_edge"] is not None and not realLeft
                    and planes["xL_edge"] < new_lo[0] - 0.05
                    and new_lo[0] - planes["xL_edge"] < 3.0):
                new_lo[0] = planes["xL_edge"]; applied.append("edge-L")
            if (planes["reachR"] and planes["xR_edge"] is not None and not realRight
                    and planes["xR_edge"] > new_hi[0] + 0.05
                    and planes["xR_edge"] - new_hi[0] < 3.0):
                new_hi[0] = planes["xR_edge"]; applied.append("edge-R")
        if reliable or back_ok or applied:
            report["actions"].append(f"depth extend [{','.join(applied) or 'no-extension'}]")
        else:
            report["actions"].append("depth distrusted (kept calibrated extents)")

    # 3. camera-inside enforcement (deterministic, safe for corner rooms)
    if pos[0] < new_lo[0] - 1e-6:
        new_lo[0] = pos[0] - CAM_MARGIN; report["actions"].append(f"extend x_lo to contain camera ({pos[0]:.2f})")
    if pos[0] > new_hi[0] + 1e-6:
        new_hi[0] = pos[0] + CAM_MARGIN; report["actions"].append(f"extend x_hi to contain camera ({pos[0]:.2f})")
    if pos[2] < new_lo[2] - 1e-6:
        new_lo[2] = pos[2] - CAM_MARGIN; report["actions"].append(f"extend z_lo to contain camera ({pos[2]:.2f})")
    if pos[2] > new_hi[2] + 1e-6:
        new_hi[2] = pos[2] + CAM_MARGIN; report["actions"].append(f"extend z_hi to contain camera ({pos[2]:.2f})")

    changed = not (np.allclose(new_lo, lo) and np.allclose(new_hi, hi))
    report["box_old"] = {"lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()}
    report["box_new"] = {"lo": new_lo.round(3).tolist(), "hi": new_hi.round(3).tolist()}
    report["reorigin_needed"] = bool((new_lo < -1e-6).any())
    if changed:
        a_lo, a_hi, shift = _apply(scene, new_lo, new_hi, dry_run)
        report["applied"] = {"shift": np.asarray(shift).round(3).tolist(),
                             "final_box": {"lo": a_lo.round(3).tolist(), "hi": a_hi.round(3).tolist()}}
    else:
        report["applied"] = None
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-depth", action="store_true")
    a = ap.parse_args()
    rep = refine_geometry(a.scene, dry_run=a.dry_run, use_depth=not a.no_depth)
    print(json.dumps(rep, indent=2))
