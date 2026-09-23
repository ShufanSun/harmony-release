"""
post_refine_ceiling.py — make the rendered ceiling pixel match VGGT's
ceiling-anchor target, regardless of what `walls.obj`'s ceiling height
happens to be.

Why this exists:
    Stage 2 (`align_camera`) aligns the FLOOR anchor pixel exactly via the
    iterative orbit + pin loop, but the pin only enforces the floor↔ceiling
    pixel SPAN.  If `walls.obj`'s ceiling height differs from the H_room
    value the pin used (e.g. mesh generated at Y=1.8m while pin used 2.7m),
    the rendered ceiling lands at the wrong pixel even though the floor is
    correct and the span is correct.

What this does:
    1. Reads `camera_vggt.json`'s `_ceil_anchor_px` (target ceiling pixel).
    2. Solves analytically for the `ceiling_h` that makes the back-wall top
       corner (at the same X/Z as the floor anchor) project to that pixel.
    3. Rewrites `walls.obj` (moves every vertex at the current ceiling Y to
       the new height) and `floorplan_analysis.json` (`ceiling_height_m`).
    4. Re-renders `render_vggt.png` + `render_vggt_overlay.png` so the new
       overlay reflects the corrected mesh.

This is a post-processor — `align_to_walls.py` is left untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sync_room_dims_to_mesh(output_dir, verbose: bool = True) -> bool:
    """Refresh floorplan_analysis.json + walls_metadata.json room dimensions from
    the ACTUAL walls.obj geometry (width = X-extent, depth = Z-extent, height =
    Y-extent).

    Stage-2 VGGT alignment can resize the wall mesh — especially the WIDTH — after
    Stage-1 wrote floorplan_analysis.json, and `_sync_floor_depth_metadata` only
    refreshes DEPTH.  That leaves `room.floor_width_m` stale, and furniture
    placement (place_furniture_vggt reads floor_width_m) then scales/grounds the
    furniture against the wrong room size — objects float / mis-scale.  This syncs
    all three dims so the sidecars match the mesh.  Returns True if anything changed."""
    out = Path(output_dir)
    wp = out / "walls.obj"
    if not wp.exists():
        return False
    verts = [[float(p) for p in l.split()[1:4]]
             for l in wp.read_text().splitlines() if l.startswith("v ")]
    if not verts:
        return False
    va = np.array(verts)
    W  = round(float(va[:, 0].max() - va[:, 0].min()), 3)   # x extent
    Dd = round(float(va[:, 2].max() - va[:, 2].min()), 3)   # z extent
    H  = round(float(va[:, 1].max() - va[:, 1].min()), 3)   # y extent
    changed = False
    fap = out / "floorplan_analysis.json"
    if fap.exists():
        fa = json.loads(fap.read_text())
        room = fa.get("room") if isinstance(fa, dict) else None
        if isinstance(room, dict):
            for k, val in (("floor_width_m", W), ("floor_depth_m", Dd),
                           ("ceiling_height_m", H)):
                if abs(float(room.get(k) or 0) - val) > 0.02:
                    if verbose:
                        print(f"[dim_sync] floorplan_analysis {k}: "
                              f"{room.get(k)} → {val}")
                    room[k] = val
                    changed = True
            if changed:
                fap.write_text(json.dumps(fa, indent=2))
    mp = out / "walls_metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
        if isinstance(meta, dict):
            m_changed = False
            for w, length in (("back", W), ("front", W), ("left", Dd), ("right", Dd)):
                e = meta.get(w)
                if isinstance(e, dict):
                    if e.get("length_m") is not None and abs(float(e["length_m"] or 0) - length) > 0.02:
                        e["length_m"] = length; m_changed = True
                    if e.get("height_m") is not None and abs(float(e["height_m"] or 0) - H) > 0.02:
                        e["height_m"] = H; m_changed = True
            if m_changed:
                mp.write_text(json.dumps(meta, indent=2)); changed = True
    if verbose and changed:
        print(f"[dim_sync] room dims synced to walls.obj: W={W} D={Dd} H={H}")
    return changed


def _sync_floor_depth_metadata(out: Path, depth_m: float, verbose: bool = True) -> None:
    """Write the room DEPTH (= walls.obj Z extent) into walls_metadata.json
    (left/right wall length + tile_size) and floorplan_analysis.json
    (room.floor_depth_m).  The VLM frequently leaves floor_depth_m = 0, so these
    must be refreshed from the actual geometry after any depth (re)calibration."""
    if not depth_m or depth_m <= 0.1:
        return
    try:
        mp = out / "walls_metadata.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
            if isinstance(meta, dict):
                for w in ("left", "right"):
                    if isinstance(meta.get(w), dict):
                        if meta[w].get("length_m") is not None:
                            meta[w]["length_m"] = depth_m
                        if meta[w].get("tile_size_m") is not None:
                            meta[w]["tile_size_m"] = depth_m
                mp.write_text(json.dumps(meta, indent=2))
        fap = out / "floorplan_analysis.json"
        if fap.exists():
            fa = json.loads(fap.read_text())
            if isinstance(fa, dict) and isinstance(fa.get("room"), dict):
                fa["room"]["floor_depth_m"] = depth_m
                fap.write_text(json.dumps(fa, indent=2))
        if verbose:
            print(f"[depth_extend] floor depth synced → {depth_m:.2f}m "
                  f"(walls_metadata left/right + floorplan_analysis)")
    except Exception as e:
        if verbose:
            print(f"[depth_extend] floor-depth metadata sync failed: {e}")


def extend_room_depth_to_camera(output_dir: str | Path,
                                 margin_m: float = 0.4,
                                 verbose: bool = True) -> bool:
    """
    Extend walls.obj depth so the front wall is behind the calibrated camera.

    After VGGT floor-first calibration the camera may land past the room's
    front wall (happens when floor_depth_m was estimated too conservatively
    from dist_to_back_m × 1.3).  Any part of the FOV that looks past the
    side-wall edges then shows void.  This function moves every vertex at
    the current Z-max (front wall) to camera_z + margin_m.

    Returns True if the mesh was extended, False if skipped.
    """
    out = Path(output_dir)
    cam_p   = out / "camera_vggt.json"
    walls_p = out / "walls.obj"

    if not (cam_p.exists() and walls_p.exists()):
        if verbose:
            print("[depth_extend] missing camera_vggt.json or walls.obj — skipping")
        return False

    cam     = json.loads(cam_p.read_text())
    cam_z   = float(cam["position_m"][2])

    lines   = walls_p.read_text().splitlines(keepends=True)
    verts   = [[float(p) for p in l.split()[1:4]]
               for l in lines if l.startswith("v ")]
    if not verts:
        return False
    z_max_old = float(max(v[2] for v in verts))
    z_min     = float(min(v[2] for v in verts))

    needed_z = cam_z + margin_m
    if needed_z <= z_max_old:
        if verbose:
            print(f"[depth_extend] room depth ({z_max_old:.2f}m) already covers "
                  f"camera_z ({cam_z:.2f}m) + margin — skipping")
        # Even when no extension is needed, keep the floor/room DEPTH dimension in
        # sync with the actual mesh (the VLM often leaves floor_depth_m = 0).
        _sync_floor_depth_metadata(out, round(z_max_old - z_min, 3), verbose)
        return False

    if verbose:
        print(f"[depth_extend] camera_z={cam_z:.2f}m > room z_max={z_max_old:.2f}m "
              f"→ extending front wall to {needed_z:.2f}m")

    new_lines: list[str] = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            if abs(z - z_max_old) < 0.05:
                z = needed_z
            new_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        else:
            new_lines.append(line)
    walls_p.write_text("".join(new_lines))
    if verbose:
        print(f"[depth_extend] walls.obj front-wall Z: {z_max_old:.3f} → {needed_z:.3f}")

    # The floor/room DEPTH dimension must follow the extended mesh, or downstream
    # (furniture placement, side-wall textures, chair groups) reads a stale depth.
    _sync_floor_depth_metadata(out, round(needed_z - z_min, 3), verbose)

    # Re-render so the overlay reflects the extended mesh
    try:
        from floorplan.wall_line.render_room import render_room
        from floorplan.vggt_estimates.align_to_walls import (
            render_manhattan_overlay, _find_back_anchor,
        )
        from floorplan.vggt_estimates.manhattan import estimate_manhattan

        render_path = out / "render_vggt.png"
        render_room(
            mesh_path=str(walls_p),
            camera_json_path=str(cam_p),
            out_path=str(render_path),
            texture_dir=None,
            bg_color=(40, 40, 40),
        )
        result = estimate_manhattan(str(out / "vggt"), frame=0, subsample=4)
        floor_w, ceil_w = _find_back_anchor(result)
        render_manhattan_overlay(
            str(render_path), result, floor_w, ceil_w,
            str(out / "render_vggt_overlay.png"),
        )
        if verbose:
            print(f"[depth_extend] re-rendered render_vggt.png + overlay")
    except Exception as e:
        if verbose:
            print(f"[depth_extend] re-render failed: {e}")

    return True


def _ceiling_crease_color(image_path, col_x: float, W: int, H: int,
                          wall_band=(0.30, 0.42), win: int = 40,
                          dist_thr: float = 26.0, frac_thr: float = 0.6):
    """Detect the ceiling-wall crease y at image column col_x by COLOUR
    transition (robust where edge-energy fails — cove/lamp/patterned walls).

    Samples the wall colour from an upper-mid band at that column (above most
    furniture, below the ceiling), then scans down from the top to the first row
    that is mostly that wall colour — i.e. where the wall begins below the
    ceiling. Returns y in image pixels, or None."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.resize(img, (W, H))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    x0 = max(0, int(col_x) - win); x1 = min(W, int(col_x) + win)
    if x1 - x0 < 8:
        return None
    wall = np.median(
        lab[int(wall_band[0] * H):int(wall_band[1] * H), x0:x1].reshape(-1, 3), 0)
    for y in range(int(0.02 * H), int(0.5 * H)):
        d = np.sqrt(((lab[y, x0:x1] - wall) ** 2).sum(1))
        if (d < dist_thr).mean() > frac_thr:
            return float(y)
    return None


def refine_ceiling_to_vggt_target(output_dir: str | Path, verbose: bool = True) -> bool:
    """Returns True if the ceiling was adjusted, False if skipped."""
    out = Path(output_dir)
    cam_p   = out / "camera_vggt.json"
    walls_p = out / "walls.obj"
    fp_p    = out / "floorplan_analysis.json"

    if not (cam_p.exists() and walls_p.exists() and fp_p.exists()):
        if verbose:
            print(f"[ceil_refine] missing one of camera_vggt.json/walls.obj/"
                  f"floorplan_analysis.json — skipping")
        return False

    cam = json.loads(cam_p.read_text())
    floor_target = cam.get("_floor_anchor_px")
    ceil_target  = cam.get("_ceil_anchor_px")
    if not floor_target or not ceil_target:
        if verbose:
            print(f"[ceil_refine] camera_vggt.json missing _floor/_ceil_anchor_px — "
                  f"can't refine without targets")
        return False

    # ── Camera basis ─────────────────────────────────────────────────────────
    pos  = np.array(cam["position_m"], dtype=np.float64)
    look = np.array(cam["look_at_m"],  dtype=np.float64)
    hfov = float(cam.get("hfov_deg", 70.0))
    vfov = float(cam.get("vfov_deg", hfov))
    W_r  = int(cam["width_px"]);  H_r = int(cam["height_px"])
    fx = W_r / (2.0 * np.tan(np.radians(hfov / 2.0)))
    fy = H_r / (2.0 * np.tan(np.radians(vfov / 2.0)))
    cx_img = W_r / 2.0
    cy_img = H_r / 2.0

    fwd = look - pos
    fwd_n = float(np.linalg.norm(fwd))
    if fwd_n < 1e-6:
        return False
    fwd /= fwd_n
    up_world = np.array([0., 1., 0.])
    right = np.cross(fwd, up_world)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        return False
    right /= rn
    up_cam = np.cross(right, fwd)

    # ── Find the back-wall floor anchor in 3D from walls.obj ──────────────────
    lines = walls_p.read_text().splitlines(keepends=True)
    verts = [[float(p) for p in l.split()[1:4]]
             for l in lines if l.startswith("v ")]
    if not verts:
        return False
    va = np.array(verts)
    floor_y    = float(va[:, 1].min())
    ceil_y_old = float(va[:, 1].max())
    back_z     = float(va[:, 2].min())   # mesh convention: back wall at min Z
    h_old = ceil_y_old - floor_y

    # back-wall floor candidate corners (mesh: y≈floor_y AND z≈back_z)
    back_floor_pts = [v for v in verts
                      if abs(v[1] - floor_y) < 0.05 and abs(v[2] - back_z) < 0.05]
    if not back_floor_pts:
        if verbose:
            print(f"[ceil_refine] no back-wall floor corners found in mesh "
                  f"(floor_y={floor_y:.3f}, back_z={back_z:.3f}) — skipping")
        return False

    def _project(pt):
        d = np.array(pt) - pos
        z_cam = float(d @ fwd)
        if z_cam <= 1e-6:
            return None
        x_cam = float(d @ right)
        y_cam = float(d @ up_cam)
        return np.array([cx_img + fx * x_cam / z_cam,
                         cy_img - fy * y_cam / z_cam])

    # Pick the back-wall floor corner whose projection is closest to the
    # floor-anchor target — that's the "anchor" the iterative loop already
    # locked to.
    anchor_pt = None
    best_err = float("inf")
    floor_target_arr = np.array(floor_target, dtype=np.float64)
    for pt in back_floor_pts:
        pp = _project(pt)
        if pp is None:
            continue
        err = float(np.linalg.norm(pp - floor_target_arr))
        if err < best_err:
            best_err = err; anchor_pt = pt
    if anchor_pt is None:
        if verbose:
            print(f"[ceil_refine] no back-wall floor corner projects in front "
                  f"of camera — skipping")
        return False
    if verbose:
        print(f"[ceil_refine] floor anchor: world=({anchor_pt[0]:.2f},"
              f"{anchor_pt[1]:.2f},{anchor_pt[2]:.2f})  "
              f"projects to ({_project(anchor_pt)[0]:.0f},"
              f"{_project(anchor_pt)[1]:.0f}) "
              f"vs target ({floor_target[0]:.0f},{floor_target[1]:.0f}) "
              f"→ floor err {best_err:.1f}px")

    anchor_x = float(anchor_pt[0])

    # ── Solve analytically for ceiling_h ──────────────────────────────────────
    # Point Q = (anchor_x, h, back_z).  delta = Q - pos.
    # z_cam(h) = a + b*h    where a = (anchor_x-px)*fwd[0] - py*fwd[1] + (back_z-pz)*fwd[2]
    #                              b = fwd[1]
    # y_cam(h) = e + f*h    where e = (anchor_x-px)*up[0] - py*up[1] + (back_z-pz)*up[2]
    #                              f = up[1]
    # v_proj(h) = cy_img - fy * y_cam / z_cam = v_target  →  solve
    #   fy*(e + f*h) = (cy_img - v_target) * (a + b*h)
    #   h * (fy*f - (cy_img - v_t)*b) = (cy_img - v_t)*a - fy*e
    px, py, pz = pos
    a_c = (anchor_x - px) * fwd[0] - py * fwd[1] + (back_z - pz) * fwd[2]
    b_c = fwd[1]
    e_c = (anchor_x - px) * up_cam[0] - py * up_cam[1] + (back_z - pz) * up_cam[2]
    f_c = up_cam[1]

    # Robustness for non-converged calibrations: the VGGT ceiling anchor can be
    # far off (edge-energy detection fooled by cove/lamp/patterned walls). For
    # those scenes ONLY, re-detect the crease by colour transition and prefer it
    # on clear disagreement. Converged scenes are trusted as-is (no regression).
    if not cam.get("_floor_first_converged", True):
        _ref = next((p for p in (out / f"{out.name}.jpeg", out / f"{out.name}.jpg")
                     if p.exists()), None)
        if _ref is None:
            _cands = sorted(out.glob("*.jpeg")) + sorted(out.glob("*.jpg"))
            _ref = _cands[0] if _cands else None
        if _ref is not None:
            _ycc = _ceiling_crease_color(_ref, ceil_target[0], W_r, H_r)
            # Conservative: only fix a too-SHORT room — crease ABOVE the anchor
            # (smaller y) by >50px — AND only when the crease sits in a plausible
            # ceiling-junction band (top 4–25%). This corrects the known failure
            # (anchor sunk into the wall) while skipping the cases where the
            # colour detector is unreliable (two-tone walls → spuriously low
            # crease, or it latches onto the very top), which would regress.
            _plaus = (0.04 * H_r) <= (_ycc or 1e9) <= (0.25 * H_r)
            if _ycc is not None and _ycc < ceil_target[1] - 50 and _plaus:
                if verbose:
                    print(f"[ceil_refine] non-converged scene: colour-crease "
                          f"y={_ycc:.0f} above VGGT anchor y={ceil_target[1]:.0f} "
                          f"(room too short) — using colour-crease")
                ceil_target = [ceil_target[0], float(_ycc)]
            elif verbose and _ycc is not None:
                print(f"[ceil_refine] colour-crease y={_ycc:.0f} vs anchor "
                      f"y={ceil_target[1]:.0f} — keeping VGGT anchor (guarded)")

    v_t = float(ceil_target[1])
    denom = fy * f_c - (cy_img - v_t) * b_c
    if abs(denom) < 1e-9:
        if verbose:
            print(f"[ceil_refine] degenerate denominator in ceiling solve — skipping")
        return False
    h_new = ((cy_img - v_t) * a_c - fy * e_c) / denom

    # Sanity: ceiling_h must be in a plausible range
    if not (1.5 <= h_new <= 4.5):
        if verbose:
            print(f"[ceil_refine] computed ceiling_h={h_new:.3f}m out of range "
                  f"[1.5, 4.5] — skipping (probably bad anchor)")
        return False
    if abs(h_new - h_old) < 0.01:
        if verbose:
            print(f"[ceil_refine] mesh ceiling already correct "
                  f"(h_old={h_old:.3f}m, h_new={h_new:.3f}m, Δ<1cm) — skipping")
        return False

    if verbose:
        print(f"[ceil_refine] mesh ceiling height: {h_old:.3f}m → "
              f"{h_new:.3f}m  (target ceiling pixel y={v_t:.0f})")

    # ── Rewrite walls.obj ─────────────────────────────────────────────────────
    new_ceil_y = floor_y + h_new
    new_lines: list[str] = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            if abs(y - ceil_y_old) < 0.05:
                y = new_ceil_y
            new_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        else:
            new_lines.append(line)
    walls_p.write_text("".join(new_lines))
    if verbose:
        print(f"[ceil_refine] walls.obj ceiling vertices Y: "
              f"{ceil_y_old:.3f} → {new_ceil_y:.3f}")

    # ── DO NOT update floorplan_analysis.json ────────────────────────────────
    # The previous behaviour wrote h_new back to the analysis, which made the
    # adjusted ceiling persist across runs.  When cam_y collapses (pin's
    # Y/Z coupling) the refine reduces ceiling height; writing that back means
    # the next run starts with the reduced ceiling, ceiling shrinks again,
    # compounding to absurd values (saw 2.4m → 1.59m → eventually below 1m).
    # Update walls.obj for THIS run's render so the corner edge endpoints
    # match cyan in the overlay, but don't pollute the source-of-truth analysis.
    if verbose:
        print(f"[ceil_refine] walls.obj updated to ceiling={h_new:.3f}m for this "
              f"render only; analysis ceiling_height_m kept unchanged "
              f"(prevents compound shrinkage across reruns)")

    # ── Update camera wall_context too (used by furniture placement) ─────────
    try:
        wctx = cam.setdefault("wall_context", {})
        for side in ("back", "left", "right"):
            if side in wctx and isinstance(wctx[side], dict):
                wctx[side]["height_m"] = round(float(h_new), 3)
        cam_p.write_text(json.dumps(cam, indent=2))
    except Exception:
        pass

    # ── Re-render render_vggt.png + render_vggt_overlay.png ──────────────────
    try:
        from floorplan.wall_line.render_room import render_room
        from floorplan.vggt_estimates.align_to_walls import (
            render_manhattan_overlay, _find_back_anchor,
        )
        from floorplan.vggt_estimates.manhattan import estimate_manhattan

        render_path = out / "render_vggt.png"
        render_room(
            mesh_path=str(walls_p),
            camera_json_path=str(cam_p),
            out_path=str(render_path),
            texture_dir=None,
            bg_color=(40, 40, 40),
        )
        # Re-fit Manhattan to draw the overlay (cheap with cached VGGT outputs)
        result = estimate_manhattan(str(out / "vggt"), frame=0, subsample=4)
        floor_w, ceil_w = _find_back_anchor(result)
        render_manhattan_overlay(
            str(render_path), result, floor_w, ceil_w,
            str(out / "render_vggt_overlay.png"),
        )
        if verbose:
            print(f"[ceil_refine] re-rendered render_vggt.png + "
                  f"render_vggt_overlay.png with corrected ceiling")
    except Exception as e:
        if verbose:
            print(f"[ceil_refine] re-render failed: {e}")

    return True
