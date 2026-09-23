"""
place_furniture.py — place furniture GLBs on the floor and render the scene.

For each entry in furniture/placement_analysis.json (in placement order):
  1. Back-project bbox centre pixel onto the floor plane (Y=0) → 3D (X, Z).
  2. Set Y = object_height / 2 so the bottom sits flush on the floor.
  3. Scale GLB to canonical furniture dimensions for the type.
  4. Rotate to face the camera (wall-affinity objects face their wall inward normal).
  5. Composite all objects using the same rasteriser as wall_mounted_object_placement.
  6. Save render to furniture/render_furniture_placed.png.

Usage:
    python -m object_placement.furniture.place_furniture \\
        --output-dir outputs/20260331_031530 \\
        [--indices 0]       # only place the first N objects (0 = all)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Reuse rendering utilities from wall_mounted ────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from object_placement.wall_mounted.wall_mounted_object_placement import (
    _camera_axes,
    _backproject_pixel,
    _rasterize_tri,
    _project_vertex,
    _mesh_face_color,
    NEAR_CLIP,
)
from object_placement.wall_mounted.placements.fill_openings import (
    _get_vertex_colors,
    _rasterize_vc_tri,
)
from object_placement.furniture.vggt_refine import (
    compute_vggt_calibration as _vggt_compute_calibration,
    compute_pseudo_to_metric_scale as _vggt_compute_pseudo_to_metric_scale,
    compute_pseudo_to_metric_polyfit as _vggt_compute_pseudo_to_metric_polyfit,
    refine_placement as _vggt_refine_placement,
    render_diff_refine_placement as _vggt_render_diff_refine,
    save_summary_plot as _vggt_save_summary,
)
from object_placement.wall_mounted.placements.place_objects import (
    _transform_glb_to_wall,
)
from object_placement.furniture.object_placement import cap_camera_resolution

# ── Default physical sizes per furniture type (width_x, height_y, depth_z) m ──
_DEFAULT_SIZES: dict[str, tuple[float, float, float]] = {
    "sofa":         (1.90, 0.85, 0.90),
    "chair":        (0.70, 0.90, 0.70),
    "armchair":     (0.85, 0.90, 0.85),
    "coffee_table": (0.90, 0.45, 0.60),
    "dining_table": (1.60, 0.75, 0.90),
    "desk":         (1.40, 0.75, 0.65),
    "bookcase":     (0.90, 1.80, 0.35),
    "cabinet":      (1.00, 0.90, 0.40),
    "floor_lamp":   (0.40, 1.55, 0.40),
    "bed":          (1.80, 0.55, 2.10),
    "plant":        (0.55, 1.60, 0.55),
}
_DEFAULT_SIZE_FALLBACK = (0.80, 0.80, 0.80)

_FLAT_TYPES = {"carpet", "rug"}

# Gap (metres) left between object back-face and wall to avoid z-fighting
_WALL_GAP = 0.05

# Bbox area (px²) below which perspective back-projection gives unreliable
# size estimates (object is partially occluded / only a sliver is visible).
# For such objects we skip size estimation and reuse a reference size from
# an already-placed object of the same type, or fall back to the default.
_MIN_RELIABLE_BBOX_AREA_PX = 10_000  # 100×100 px

# ── Room dimension helpers ────────────────────────────────────────────────────

def _get_room_dims(glb_path: Path, cam: dict) -> tuple[float, float]:
    """Return (room_w, room_d) from the best available source.

    Priority:
      0. walls.obj bounds — the ACTUAL rendered room geometry.  Placement,
         flush-snap, collision resolution and the final render must all agree on
         the room extent; walls.obj is what gets drawn (and what seating_group
         already uses), so deriving the room box from anything else (floorplan
         JSON width, camera wall_context, camera-Z depth) leaves objects snapped
         to the wrong wall position when those disagree (e.g. floorplan=4.2,
         wall_context=3.8, walls.obj=4.67 → a "right-wall" sofa lands ~0.9m short
         of the rendered wall).
      1. floorplan_analysis.json
      2. wall_context in camera JSON (may be zero in camera_vggt.json)
      3. Hardcoded fallback (5.0 m)

    When room depth is zero or missing (side walls not detected), falls back
    to the camera Z position as a conservative depth estimate.
    """
    best_w: float = 0.0

    # Navigate: glb_path → furniture/objects/... → output_dir
    output_dir = Path(str(glb_path))
    for _ in range(6):  # walk up at most 6 levels
        output_dir = output_dir.parent
        # Priority 0: the rendered wall mesh — authoritative, keeps placement,
        # collision and render on the SAME room box.
        _wobj = output_dir / "walls.obj"
        if _wobj.exists():
            try:
                _v = np.array([[float(x) for x in _l.split()[1:4]]
                               for _l in open(_wobj) if _l.startswith("v ")],
                              dtype=np.float64)
                if len(_v):
                    _w = float(_v[:, 0].max() - _v[:, 0].min())
                    _d = float(_v[:, 2].max() - _v[:, 2].min())
                    if _w > 0.1 and _d > 0.1:
                        return _w, _d
            except Exception:
                pass
        fp_path = output_dir / "floorplan_analysis.json"
        if fp_path.exists():
            try:
                with open(fp_path) as f:
                    fp = json.load(f)
                room = fp.get("room", {})
                w = float(room.get("floor_width_m", 0))
                d = float(room.get("floor_depth_m", 0))
                if w > 0.1:
                    best_w = w
                if w > 0.1 and d > 0.1:
                    return w, d
            except Exception:
                pass
            break

    # Fall back to wall_context
    wall_ctx = cam.get("wall_context", {})
    wc_w = float(wall_ctx.get("back", {}).get("length_m", 0))
    d = float(wall_ctx.get("right", {}).get("length_m", 0))
    # Also try left wall for depth (some cameras report left but not right)
    if d < 0.1:
        d = float(wall_ctx.get("left", {}).get("length_m", 0))
    if wc_w > 0.1:
        best_w = wc_w       # wall_context back-wall length is authoritative for width
    if wc_w > 0.1 and d > 0.1:
        return wc_w, d

    # If width is known but depth is zero (side walls undetected), estimate
    # depth from camera Z position — the camera is roughly at the front of the room.
    if best_w > 0.1:
        cam_z = float(cam.get("position_m", [0, 0, 5.0])[2])
        d_est = cam_z if cam_z > 0.5 else 5.0
        return best_w, d_est

    return 5.0, 5.0


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _ray_floor_intersect(
    cam_pos: np.ndarray, ray_dir: np.ndarray
) -> np.ndarray | None:
    """Intersect ray with the floor plane Y=0.  Returns world point or None."""
    denom = ray_dir[1]
    if abs(denom) < 1e-9 or denom >= 0.0:   # parallel or pointing up
        return None
    t = -cam_pos[1] / denom
    if t < 0.0:
        return None
    return cam_pos + t * ray_dir


def _rotation_y(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix around world Y axis."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], dtype=np.float64)


def _is_bed(t: str) -> bool:
    """True for any bed variant (bed, bunk_bed, loft_bed, daybed, king_bed, …).

    The VLM types beds with many labels; treating only the literal 'bed' as a bed
    made bunk/loft beds bypass the flush-to-wall scaling, the bed yaw silhouette
    pick, and the headboard guard — so they placed small and mid-room instead of
    filling their (often bottom-clipped) foreground silhouette."""
    t = (t or "").lower().replace("-", "_").replace(" ", "_")
    return t == "bed" or t.endswith("_bed") or t in ("bunk_bed", "loft_bed", "daybed")


def _enforce_seating_front_into_room(placements: list[dict]) -> None:
    """Deterministic orientation guard for wall-adjacent seating.

    A sofa/chair against a wall must have its BACK on the wall and its FRONT
    (the seat) facing into the room.  The front-detection + OBB-detilt pipeline
    occasionally bakes a 180° flip, leaving the seat facing the wall.  Here we
    check each seating object's final facing against its wall and flip 180° if
    the front points into the wall — independent of the upstream detection.

    Hunyuan seating meshes have their front at local +Z (the convention the
    no-VLM path assumes), so front_world = R @ [0,0,1].
    """
    _SEAT = {"sofa", "couch", "loveseat", "sectional",
             "chair", "armchair", "office_chair", "stool"}
    into_room = {
        "back":  np.array([0.0, 0.0, 1.0]),   # away from z=0
        "left":  np.array([1.0, 0.0, 0.0]),   # away from x=0
        "right": np.array([-1.0, 0.0, 0.0]),  # away from x=room_w
    }
    for p in placements:
        wa = p.get("wall_affinity")
        if wa not in into_room:
            continue
        if p.get("type", "").lower().replace("-", "_") not in _SEAT:
            continue
        if p.get("_facing_applied"):
            continue   # deliberately angled toward a desk/conversation focal — keep it
        R = np.array(p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
        tgt = into_room[wa]                       # room-ward normal of the wall
        tgt_h = np.array([tgt[0], tgt[2]])         # horizontal (x,z)

        # STEP 1 — GEOMETRY: the sofa's LONGER horizontal footprint axis (seating
        # length) must run PARALLEL to the wall; the SHORT (front-to-back) axis
        # perpendicular, into the room. The front detection frequently assigns the
        # front to the LONG axis on L-shaped/sectional sofas, so trusting it rotates
        # the long side perpendicular ("side against the wall"). Fix by geometry:
        # if the long axis is perpendicular to the wall, turn 90°.
        sm = p.get("size_m", {}) or {}
        wm = float(sm.get("width_m", 0.0)); dm = float(sm.get("depth_m", 0.0))
        if wm > 0 and dm > 0:
            long_local = np.array([1.0, 0.0, 0.0]) if wm >= dm else np.array([0.0, 0.0, 1.0])
            long_w = R @ long_local
            long_h = np.array([long_w[0], long_w[2]])
            n = np.linalg.norm(long_h)
            if n > 1e-6:
                long_h /= n
                # parallel to wall ⇒ long_h ⟂ tgt_h (|dot|≈0). If |dot|>0.5 the long
                # axis leans toward the wall normal (perpendicular to wall) → rotate 90°.
                if abs(float(long_h @ tgt_h)) > 0.5:
                    R = _rotation_y(np.pi / 2.0) @ R
                    print(f"  [seat_guard] idx={p.get('index')} {p.get('type')}: long axis was "
                          f"PERPENDICULAR to {wa} wall → turned 90° (seating length now along wall)")

        # STEP 2 — FRONT: with the long axis parallel, ensure the seat FRONT faces
        # into the room (flip 180° if it faces the wall). Uses the detected front's
        # component along the room-ward (short) axis.
        lf = np.array(p.get("_local_front", [0.0, 0.0, 1.0]), dtype=np.float64)
        front = R @ lf
        if float(np.array([front[0], front[2]]) @ tgt_h) < -0.05:
            R = _rotation_y(np.pi) @ R
            print(f"  [seat_guard] idx={p.get('index')} {p.get('type')}: front faced the "
                  f"{wa} wall → flipped 180° (seat now faces into the room)")
        p["rotation_3x3"] = R.tolist()

    # ── Bed: headboard against its wall ──────────────────────────────────────
    # A bed's FOOT (geometric front = away from the tall headboard end, stored in
    # _local_front) must point INTO the room, so the headboard sits flush against
    # the wall.  Unlike the seat guard above this may need a 90° turn — when the
    # bed's long head↔foot axis ended up PARALLEL to the wall (headboard on a side
    # instead of the wall the bed is flush to), which a 180°-only flip can't fix.
    for p in placements:
        if not _is_bed(p.get("type", "")):
            continue
        wa = p.get("wall_affinity")
        if wa not in into_room:
            continue
        R = np.array(p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
        lf = np.array(p.get("_local_front", [0.0, 0.0, 1.0]), dtype=np.float64)
        foot = R @ lf
        yaw_cur = float(np.arctan2(foot[0], foot[2]))
        tgt = into_room[wa]
        yaw_tgt = float(np.arctan2(tgt[0], tgt[2]))
        d = yaw_tgt - yaw_cur
        if abs(((d + np.pi) % (2 * np.pi)) - np.pi) > np.radians(5):
            p["rotation_3x3"] = (_rotation_y(d) @ R).tolist()
            print(f"  [bed_guard] idx={p.get('index')} bed: foot→into-room ({wa}); "
                  f"headboard now flush to the wall (turned {np.degrees(d):+.0f}°)")


def _pick_bed_yaw_by_silhouette(placements, cam, out_root) -> None:
    """Choose each bed's cardinal yaw by matching its rendered silhouette to the
    segmentation mask.  A bed's orientation is otherwise under-constrained — front
    detect/OBB defaults to a fixed yaw (headboard toward -X), which is right for a
    side-wall bed but wrong for a back-wall bed.  Neither a fixed default nor a
    nearest-wall rule works (the headboard wall only follows from the silhouette),
    so project the bed mesh at 0/90/180/270 and keep the yaw with the best mask IoU
    (the tall headboard makes the silhouette yaw-dependent)."""
    try:
        import trimesh
        from PIL import Image as _PILImage
        from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes
    except Exception as _e:
        print(f"  [bed_yaw] unavailable: {_e}")
        return
    beds = [p for p in placements
            if _is_bed(p.get("type", "")) and p.get("glb_path")]
    if not beds:
        return
    # The placement dict often lacks mask_file — resolve it from segment_results.json by index.
    seg_map = {}
    try:
        _sr = json.loads((Path(out_root) / "furniture" / "segment_results.json").read_text())
        for _s in _sr.get("segments", []):
            if _s.get("mask_file") is not None:
                seg_map[_s.get("index")] = _s["mask_file"]
    except Exception:
        pass
    W = int(cam.get("width_px", 1296)); H = int(cam.get("height_px", 968))
    cam_pos = np.array(cam["position_m"], float)
    look_at = np.array(cam["look_at_m"], float)
    up_world = np.array(cam.get("up", [0.0, 1.0, 0.0]), float)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    fx = W / (2.0 * np.tan(np.radians(float(cam["hfov_deg"]) / 2.0)))
    cx, cy = W / 2.0, H / 2.0
    GS = 192
    for p in beds:
        # VLM-authority for ORIENTATION: when the VLM has already determined which
        # wall the headboard is against (wall_affinity), the bed flush guard
        # (_enforce_seating_front_into_room, run just before this) has turned the
        # headboard to that wall using the geometric head↔foot axis. A ragged or
        # partial bed mask must NOT override that decision — the silhouette is only
        # trustworthy for SCALING to size, not for deciding which side is the back.
        # So skip the silhouette yaw override whenever there is a clear VLM wall,
        # and fall back to it only when orientation is genuinely under-constrained
        # (wall_affinity 'centre'/None).
        if p.get("wall_affinity") in ("back", "front", "left", "right"):
            print(f"  [bed_yaw] idx={p.get('index')} bed: VLM wall="
                  f"'{p.get('wall_affinity')}' is authoritative — keeping VLM/flush "
                  f"orientation (silhouette used for scaling only, not orientation)")
            continue
        try:
            mfile = p.get("mask_file") or seg_map.get(p.get("index"))
            if not mfile:
                continue
            mpath = Path(out_root) / "furniture" / "segmented" / mfile
            if not mpath.exists():
                continue
            m = np.array(_PILImage.open(mpath))
            fg = (m.sum(2) > 0) if m.ndim == 3 else (m > 0)
            ys, xs = np.where(fg)
            if len(xs) < 50:
                continue
            # The mask is at NATIVE resolution; the mesh grid below is binned with
            # (W,H) at the capped render resolution. Rescale the mask pixels into
            # (W,H) so both grids share one pixel space (correct IoU).
            _mh, _mw = fg.shape[:2]
            if _mw > 0 and _mh > 0 and (_mw, _mh) != (W, H):
                xs = (xs * (W / float(_mw))).astype(np.int64)
                ys = (ys * (H / float(_mh))).astype(np.int64)
            tgt = np.zeros((GS, GS), bool)
            tgt[np.clip(ys * GS // H, 0, GS - 1), np.clip(xs * GS // W, 0, GS - 1)] = True
            mesh = trimesh.load(p["glb_path"], force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            verts = np.asarray(mesh.vertices, float)
            if len(verts) > 4000:
                verts = verts[np.random.default_rng(0).choice(len(verts), 4000, replace=False)]
            vs = verts * np.array(p["scale"], float)
            R0 = np.array(p["rotation_3x3"], float)
            pos = np.array(p["position_m"], float)
            best_yaw, best_iou = 0.0, -1.0
            for yaw in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
                R = _rotation_y(yaw) @ R0
                vw = vs @ R.T + pos
                grid = np.zeros((GS, GS), bool)
                for v in vw:
                    px, py, zc = _project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
                    if zc <= 0.05:
                        continue
                    gx, gy = int(px * GS // W), int(py * GS // H)
                    if 0 <= gx < GS and 0 <= gy < GS:
                        grid[max(0, gy - 1):gy + 2, max(0, gx - 1):gx + 2] = True  # splat 3x3
                inter = int((grid & tgt).sum()); uni = int((grid | tgt).sum())
                iou = inter / uni if uni else 0.0
                if iou > best_iou:
                    best_iou, best_yaw = iou, yaw
            # Only override the geometry default when the silhouette match is
            # actually reliable. A bottom-clipped or rough bunk-bed mask gives a
            # low IoU, and a 90°/270° pick there turns a wall-flush bed to jut
            # INTO the room (the length should stay ALONG the wall). Keep R0 when
            # the best IoU is weak, and never accept a perpendicular turn for a
            # wall-flush bed unless it clearly wins.
            _wall = p.get("wall_affinity")
            _perp = abs(best_yaw - np.pi / 2) < 1e-3 or abs(best_yaw - 3 * np.pi / 2) < 1e-3
            _MIN_YAW_IOU = 0.45
            if abs(best_yaw) > 1e-3 and best_iou >= _MIN_YAW_IOU and not (
                    _perp and _wall in ("left", "right") and best_iou < 0.6):
                p["rotation_3x3"] = (_rotation_y(best_yaw) @ R0).tolist()
                print(f"  [bed_yaw] idx={p.get('index')} bed: silhouette yaw "
                      f"{np.degrees(best_yaw):+.0f}° (IoU={best_iou:.2f})")
            else:
                print(f"  [bed_yaw] idx={p.get('index')} bed: kept default orientation "
                      f"(best yaw {np.degrees(best_yaw):+.0f}° IoU={best_iou:.2f} too weak)")
        except Exception as _be:
            print(f"  [bed_yaw] idx={p.get('index')} failed: {_be}")


def _fine_yaw_refine(p, cam, out_root) -> None:
    """CHANGE 2: fine slight-yaw silhouette refine for ALL placed furniture.

    Generalisation of the BED-only cardinal yaw pick (_pick_bed_yaw_by_silhouette):
    reuses the SAME mask-loading + mesh→grid projection + IoU machinery, but sweeps
    a FINER set of yaw candidates (±20° around the current yaw in 5° steps) and picks
    the yaw that maximises silhouette IoU.  Runs in place on a single placement dict,
    AFTER position (mask_align) AND AFTER the VGGT depth-approximate fix, gated by the
    env flag SCENEWEAVE_FINE_YAW (default off → behaviour unchanged).

    Same safety gate the bed code uses: only override when best IoU ≥ _MIN_YAW_IOU,
    and never accept a near-perpendicular turn for a wall-flush object unless it
    clearly wins.
    """
    if not os.environ.get("SCENEWEAVE_FINE_YAW"):
        return
    if not p.get("glb_path"):
        return
    if p.get("wall_mounted") or p.get("is_carpet"):
        return
    try:
        import trimesh
        from PIL import Image as _PILImage
        from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes
    except Exception as _e:
        print(f"  [fine_yaw] unavailable: {_e}")
        return
    try:
        # Resolve the segment mask the same way the bed yaw code does.
        mfile = p.get("mask_file")
        if not mfile:
            try:
                _sr = json.loads((Path(out_root) / "furniture" / "segment_results.json").read_text())
                for _s in _sr.get("segments", []):
                    if _s.get("index") == p.get("index") and _s.get("mask_file"):
                        mfile = _s["mask_file"]
                        break
            except Exception:
                pass
        if not mfile:
            return
        mpath = Path(out_root) / "furniture" / "segmented" / mfile
        if not mpath.exists():
            return
        W = int(cam.get("width_px", 1296)); H = int(cam.get("height_px", 968))
        cam_pos = np.array(cam["position_m"], float)
        look_at = np.array(cam["look_at_m"], float)
        up_world = np.array(cam.get("up", [0.0, 1.0, 0.0]), float)
        right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
        fx = W / (2.0 * np.tan(np.radians(float(cam["hfov_deg"]) / 2.0)))
        cx, cy = W / 2.0, H / 2.0
        GS = 192
        m = np.array(_PILImage.open(mpath))
        fg = (m.sum(2) > 0) if m.ndim == 3 else (m > 0)
        ys, xs = np.where(fg)
        if len(xs) < 50:
            return
        # Rescale the NATIVE-resolution mask into the capped (W,H) pixel space so
        # the mask grid and the mesh-projection grid (both binned with W,H) align.
        _mh, _mw = fg.shape[:2]
        if _mw > 0 and _mh > 0 and (_mw, _mh) != (W, H):
            xs = (xs * (W / float(_mw))).astype(np.int64)
            ys = (ys * (H / float(_mh))).astype(np.int64)
        tgt = np.zeros((GS, GS), bool)
        tgt[np.clip(ys * GS // H, 0, GS - 1), np.clip(xs * GS // W, 0, GS - 1)] = True
        mesh = trimesh.load(p["glb_path"], force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        verts = np.asarray(mesh.vertices, float)
        if len(verts) > 4000:
            verts = verts[np.random.default_rng(0).choice(len(verts), 4000, replace=False)]
        vs = verts * np.array(p["scale"], float)
        R0 = np.array(p["rotation_3x3"], float)
        pos = np.array(p["position_m"], float)

        def _iou_at(yaw: float) -> float:
            R = _rotation_y(yaw) @ R0
            vw = vs @ R.T + pos
            grid = np.zeros((GS, GS), bool)
            for v in vw:
                px, py, zc = _project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
                if zc <= 0.05:
                    continue
                gx, gy = int(px * GS // W), int(py * GS // H)
                if 0 <= gx < GS and 0 <= gy < GS:
                    grid[max(0, gy - 1):gy + 2, max(0, gx - 1):gx + 2] = True  # splat 3x3
            inter = int((grid & tgt).sum()); uni = int((grid | tgt).sum())
            return inter / uni if uni else 0.0

        # DEFECT FIX: the old sweep was ±20° ONLY — it could never correct a 90°
        # wrong-cardinal facing (lamp/desk/sofa). Do a COARSE 4-cardinal IoU pick
        # first (catches the 90°/180°/270° error), THEN a FINE ±20° refine around
        # the winning cardinal.
        cur_iou = _iou_at(0.0)
        best_yaw, best_iou = 0.0, cur_iou
        for c_deg in (90.0, 180.0, 270.0):          # coarse cardinals
            iou = _iou_at(np.radians(c_deg))
            if iou > best_iou:
                best_iou, best_yaw = iou, np.radians(c_deg)
        base = best_yaw
        for deg in range(-20, 21, 5):               # fine refine around winner
            if deg == 0:
                continue
            yaw = base + np.radians(deg)
            iou = _iou_at(yaw)
            if iou > best_iou:
                best_iou, best_yaw = iou, yaw

        # Override only on a real silhouette win. A near-cardinal (≈90/270) turn
        # for a wall-flush object must clear a higher bar so we don't spin a
        # flush sofa/bed perpendicular on a noisy mask.
        _wall = p.get("wall_affinity")
        _deg = abs(np.degrees(best_yaw)) % 180.0
        _cardinal_turn = 70.0 <= _deg <= 110.0
        _MIN_YAW_IOU = 0.40
        if (abs(best_yaw) > 1e-3 and best_iou >= _MIN_YAW_IOU
                and best_iou > cur_iou + 0.02
                and not (_cardinal_turn and _wall in ("left", "right", "back", "front")
                         and best_iou < 0.60)):
            p["rotation_3x3"] = (_rotation_y(best_yaw) @ R0).tolist()
            print(f"  [fine_yaw] idx={p.get('index')} yaw {np.degrees(best_yaw):+.0f}° "
                  f"(IoU={best_iou:.2f}, was {cur_iou:.2f})")
        else:
            print(f"  [fine_yaw] idx={p.get('index')} kept (best IoU={best_iou:.2f} "
                  f"cur={cur_iou:.2f})")
    except Exception as _fe:
        print(f"  [fine_yaw] idx={p.get('index')} failed: {_fe}")


def _silhouette_lateral_pin(p, cam, out_root) -> None:
    """Pin an object's LATERAL (left/right) position to its silhouette.

    The silhouette gives a reliable image-x even when VGGT depth is unreliable
    (occluded/dark objects), so we trust left/right from the mask and only let
    depth stay where it is.  Shifts the object along the GROUND-projected
    camera-right axis until its rendered centre-x matches the mask centre-x.
    Gated by SCENEWEAVE_SILH_LATERAL (default off → behaviour unchanged).
    """
    if not os.environ.get("SCENEWEAVE_SILH_LATERAL"):
        return
    if not p.get("glb_path") or p.get("wall_mounted") or p.get("is_carpet"):
        return
    try:
        import trimesh
        from PIL import Image as _PILImage
        from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes
    except Exception:
        return
    try:
        mfile = p.get("mask_file")
        if not mfile:
            try:
                _sr = json.loads((Path(out_root) / "furniture" / "segment_results.json").read_text())
                for _s in _sr.get("segments", []):
                    if _s.get("index") == p.get("index") and _s.get("mask_file"):
                        mfile = _s["mask_file"]; break
            except Exception:
                pass
        if not mfile:
            return
        mpath = Path(out_root) / "furniture" / "segmented" / mfile
        if not mpath.exists():
            return
        W = int(cam.get("width_px", 1296)); H = int(cam.get("height_px", 968))
        cam_pos = np.array(cam["position_m"], float)
        look_at = np.array(cam["look_at_m"], float)
        up_world = np.array(cam.get("up", [0.0, 1.0, 0.0]), float)
        right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
        fx = W / (2.0 * np.tan(np.radians(float(cam["hfov_deg"]) / 2.0)))
        cx, cy = W / 2.0, H / 2.0
        m = np.array(_PILImage.open(mpath))
        fg = (m.sum(2) > 0) if m.ndim == 3 else (m > 0)
        ys, xs = np.where(fg)
        if len(xs) < 50:
            return
        # The mask is at the NATIVE photo resolution; (W,H)/fx/cx are at the
        # capped render resolution — rescale the mask pixels into (W,H) so the
        # silhouette centre matches the capped-res projection below.
        _mh, _mw = fg.shape[:2]
        if _mw > 0 and _mh > 0 and (_mw, _mh) != (W, H):
            xs = xs * (W / float(_mw))
            ys = ys * (H / float(_mh))
        silh_cx = float(np.median(xs))          # robust mask centre-x (px)
        mesh = trimesh.load(p["glb_path"], force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        verts = np.asarray(mesh.vertices, float)
        if len(verts) > 3000:
            verts = verts[np.random.default_rng(0).choice(len(verts), 3000, replace=False)]
        R = np.array(p["rotation_3x3"], float)
        vs = (verts * np.array(p["scale"], float)) @ R.T   # rot/scale, no translate
        pos = np.array(p["position_m"], float)
        gr = np.array([right_v[0], 0.0, right_v[2]], float)  # ground camera-right
        if np.linalg.norm(gr) < 1e-6:
            return
        gr /= np.linalg.norm(gr)

        def _proj_cx(position):
            d = (vs + position) - cam_pos
            zc = d @ fwd_v
            ok = zc > 0.05
            if not ok.any():
                return None
            px = cx + fx * ((d @ right_v)[ok] / zc[ok])
            return float(np.mean(px))

        moved = 0.0
        for _ in range(10):
            cur_cx = _proj_cx(pos)
            if cur_cx is None:
                break
            err = silh_cx - cur_cx
            if abs(err) < 4.0:
                break
            zc0 = max(float((pos - cam_pos) @ fwd_v), 0.2)
            s = float(np.clip(err * zc0 / (fx * max(1e-3, float(gr @ right_v))), -1.5, 1.5))
            pos = pos + s * gr
            moved += s
        if abs(moved) > 0.03:
            p["position_m"] = [float(pos[0]), float(p["position_m"][1]), float(pos[2])]
            print(f"  [silh_lateral] idx={p.get('index')} shifted {moved:+.2f}m "
                  f"→ x,z=({pos[0]:.2f},{pos[2]:.2f})")
        else:
            print(f"  [silh_lateral] idx={p.get('index')} already aligned")
    except Exception as _le:
        print(f"  [silh_lateral] idx={p.get('index')} failed: {_le}")


# Realistic maximum HORIZONTAL footprint (longest floor dimension, metres) per
# furniture type.  The bbox-fit scaler can inflate an object well past life-size
# when it foreshortens in a diagonal corner view; this caps it back to plausible.
_MAX_FOOTPRINT_M = {
    "sofa": 2.7, "couch": 2.7, "sectional": 3.3, "loveseat": 1.9,
    "chair": 0.95, "armchair": 1.05, "office_chair": 0.8, "stool": 0.6,
    "bench": 1.8,
    "coffee_table": 1.5, "side_table": 0.8, "round_table": 1.1, "end_table": 0.7,
    "dining_table": 2.4, "desk": 1.9, "console_table": 1.6,
    "bed": 2.2,
    "bookcase": 2.0, "cabinet": 2.2, "wardrobe": 2.6, "dresser": 1.8, "shelf": 2.0,
    "tv_stand": 2.0, "nightstand": 0.7, "floor_lamp": 0.6,
    "plant": 1.2, "tall_plant": 1.2, "large_plant": 1.5, "potted_plant": 0.9,
    "leafy_plant": 1.2, "tree": 1.8,
}


def _clamp_furniture_scale(placements: list[dict]) -> None:
    """Cap each placed object's scale so its real-world footprint stays plausible.

    Fires only to SHRINK an object whose longest horizontal dimension exceeds the
    per-type maximum (never enlarges).  Reduces the scale uniformly so proportions
    are preserved; _apply_floor_transform re-grounds and re-wall-snaps afterwards.
    """
    import trimesh
    for p in placements:
        if p.get("is_carpet") or "glb_path" not in p:
            continue
        t = p.get("type", "").lower().replace("-", "_")
        max_h = _MAX_FOOTPRINT_M.get(t)
        if max_h is None:
            continue
        # Silhouette-authoritative: mask_align fits each object's scale to its
        # rendered silhouette (the reference-match goal).  The per-type footprint
        # cap is now only a LOOSE backstop against absurd sizes (e.g. a runaway
        # bbox-fit) — it fires at 1.5× the nominal cap so a genuinely large piece
        # that legitimately matches its silhouette is NOT shrunk back down.
        # EXCEPTION: small tables foreshorten heavily when they sit close to the
        # camera in a foreground/corner view, so the bbox-fit inflates them well
        # past life-size; give them a TIGHT cap so they snap back to their true
        # silhouette size instead of dominating the room.
        _TIGHT_FOOTPRINT = {"coffee_table", "side_table", "end_table",
                            "round_table", "console_table", "nightstand"}
        max_h = max_h * (0.85 if t in _TIGHT_FOOTPRINT else 1.5)
        gp = Path(p["glb_path"])
        if not gp.exists():
            continue
        try:
            m = trimesh.load(str(gp), force="scene")
            geoms = list(m.geometry.values()) if isinstance(m, trimesh.Scene) else [m]
            g = max(geoms, key=lambda x: len(x.faces))
            ext = np.abs(g.bounds[1] - g.bounds[0])
        except Exception:
            continue
        sc = np.array(p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if sc.ndim == 0:
            sc = np.array([float(sc)] * 3)
        scaled = ext * sc
        cur_max = float(max(scaled[0], scaled[2]))   # longest horizontal extent
        if cur_max > max_h + 1e-3:
            f = max_h / cur_max
            p["scale"] = (sc * f).tolist()
            # Keep size_m in sync — downstream passes (collision / seating_group)
            # re-derive scale from size_m, so leaving it unclamped silently
            # re-inflates the object back to its pre-clamp size.
            if isinstance(p.get("size_m"), dict):
                for _k in ("width_m", "height_m", "depth_m"):
                    if _k in p["size_m"]:
                        p["size_m"][_k] = float(p["size_m"][_k]) * f
            print(f"  [scale_clamp] idx={p.get('index')} {t}: "
                  f"{cur_max:.2f}m > {max_h:.2f}m max → ×{f:.2f} → {max_h:.2f}m")


# Minimum plausible HEIGHT (m) for surface furniture.  A distant or partly
# occluded small table can be crushed by the silhouette-fit (its visible mask is
# tiny), leaving it far too short next to a full-height sofa.  These floors scale
# such pieces UP — in Y only, so the footprint (and the footprint clamp) are
# untouched — so a side table reaches roughly armrest height, a desk reaches
# work-surface height, etc.  Only ever raises height, never lowers it.
_MIN_SURFACE_HEIGHT_M: dict[str, float] = {
    "coffee_table": 0.36, "side_table": 0.48, "end_table": 0.48,
    "nightstand": 0.48, "console_table": 0.70, "desk": 0.68,
    "dining_table": 0.70, "tv_stand": 0.40,
}


# Low tables whose floor-contact is ~at the (bottom-clipped) mask edge, so
# mask_align may move them in depth to their silhouette box (see is_low_surface).
_LOW_SURFACE_TYPES = {"coffee_table", "side_table", "end_table", "round_table",
                      "console_table", "nightstand", "cocktail_table"}


def _enforce_min_surface_height(placements: list[dict]) -> None:
    """Scale under-tall surface furniture UP in Y so its height reaches a
    plausible per-type floor (keeps footprint; stretch capped at ×1.8)."""
    import trimesh
    for p in placements:
        t = p.get("type", "").lower().replace("-", "_")
        floor = _MIN_SURFACE_HEIGHT_M.get(t)
        if floor is None or "glb_path" not in p or p.get("is_carpet"):
            continue
        gp = Path(p["glb_path"])
        if not gp.exists():
            continue
        try:
            g = trimesh.load(str(gp), force="scene")
            geoms = list(g.geometry.values()) if isinstance(g, trimesh.Scene) else [g]
            m = max(geoms, key=lambda x: len(x.faces))
        except Exception:
            continue
        sc = np.array(p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if sc.ndim == 0:
            sc = np.array([float(sc)] * 3)
        cur_h = float(m.extents[1]) * float(sc[1])
        if cur_h < 1e-3 or cur_h >= floor:
            continue
        k = min(floor / cur_h, 1.8)
        sc[1] *= k
        p["scale"] = sc.tolist()
        print(f"  [min_height] idx={p.get('index')} {t}: H {cur_h:.2f}→{cur_h*k:.2f}m "
              f"(×{k:.2f}, Y-only — was short vs larger furniture)")


def _fit_scale_to_silhouette(placements: list[dict], camera: dict) -> None:
    """Re-scale each placed object so its PROJECTED silhouette matches its
    segmentation-mask bbox (box_px).

    The placement's mask-fit runs before the wall reconciliation / orientation
    guard; once those change how the object projects, its rendered size drifts
    from the mask (e.g. a sofa re-oriented onto the right wall renders 24% wider
    and 51% taller than its mask).  This re-fits the uniform scale to the mask so
    the silhouette matches.  Uses the WIDTH ratio for wall-flush pieces (their
    bbox width = the along-wall length, the most reliable cue) and the smaller of
    width/height for free pieces; the factor is clamped to avoid wild jumps.
    """
    import trimesh
    from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes
    cam_pos = np.array(camera["position_m"], dtype=np.float64)
    look_at = np.array(camera["look_at_m"],  dtype=np.float64)
    up_w    = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_w)
    W = int(camera["width_px"]); H = int(camera["height_px"])
    hfov = float(camera["hfov_deg"])
    fx = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0
    _FLUSH = {"sofa", "couch", "loveseat", "sectional", "bookcase", "cabinet",
              "desk", "bed", "dresser", "tv_stand", "console_table", "bench"}

    for p in placements:
        box = p.get("box_px")
        if not box or "glb_path" not in p or p.get("is_carpet"):
            continue
        # The chair-group may deliberately shrink dining chairs to fit a small
        # table; don't let the silhouette re-fit re-inflate them back to the
        # crowded ref size.
        if p.get("_chair_group_scaled"):
            continue
        # A CONVERGED VLM size beats a TRUNCATED silhouette.
        #
        # This pass runs last, so whatever it decides is what gets saved — and
        # it fits to box_px unconditionally.  That silently discarded the
        # reflection loop's answer: on elegant the size critique converged on a
        # 50% depth correction (0.80 → 0.40 m over two iterations) and the
        # placement that shipped differed from the no-VLM run by 0.01 cm.  The
        # loop reasoned, converged, and changed nothing.
        #
        # Only override it when the mask is actually authoritative.  A box
        # touching a frame border is a truncated silhouette — the object runs
        # off-screen, so its projected extent is not its real extent.  Every
        # object we have seen mis-sized this way was edge-clipped: office8's
        # desk, elegant's fireplace (x0=0), pexels_2343465's sofa (x0=0).
        _m = 5.0
        if p.get("_vlm_size_converged") and (
                box[0] < _m or box[1] < _m or box[2] > W - _m or box[3] > H - _m):
            print(f"  [silhouette_fit] idx={p.get('index')} {p.get('type')}: "
                  f"keeping converged VLM size "
                  f"{p.get('_vlm_size_m')} — bbox {list(map(int, box))} is "
                  f"clipped by the frame, so its silhouette is truncated")
            continue
        gp = Path(p["glb_path"])
        if not gp.exists():
            continue
        try:
            m = trimesh.load(str(gp), force="scene")
            geoms = list(m.geometry.values()) if isinstance(m, trimesh.Scene) else [m]
            g = max(geoms, key=lambda x: len(x.faces))
            if not p.get("skip_ground_removal"):
                res = _remove_ground_faces(g)
                g = res[0] if isinstance(res, tuple) else res
        except Exception:
            continue
        sc = np.array(p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if sc.ndim == 0:
            sc = np.array([float(sc)] * 3)
        R  = np.array(p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
        P  = np.array(p["position_m"], dtype=np.float64)
        wa = p.get("wall_affinity", "centre")
        rw_cam, _ = _get_room_dims(gp, camera)
        try:
            verts = _apply_floor_transform(
                g.vertices, g.bounds, sc, R, P, wall_affinity=wa, room_w=rw_cam,
                back_flush=(wa in ("back", "left", "right")
                            and (p.get("type", "") in _FLUSH or _is_bed(p.get("type", "")))))
        except Exception:
            continue
        xs, ys = [], []
        for v in verts[::25]:
            d = v - cam_pos
            zc = float(np.dot(d, fwd_v))
            if zc <= 0.05:
                continue
            xs.append(cx + fx * float(np.dot(d, right_v)) / zc)
            ys.append(cy - fx * float(np.dot(d, up_c_v)) / zc)
        if len(xs) < 8:
            continue
        rend_w = max(xs) - min(xs)
        rend_h = max(ys) - min(ys)
        if rend_w < 1 or rend_h < 1:
            continue
        tgt_w = float(box[2] - box[0])
        tgt_h = float(box[3] - box[1])
        is_flush = (p.get("type", "").lower().replace("-", "_") in _FLUSH
                    or _is_bed(p.get("type", "")))
        if is_flush:
            f = tgt_w / rend_w            # match along-wall length
        else:
            f = min(tgt_w / rend_w, tgt_h / rend_h)   # fit within the mask
        # Beds are commonly bottom-clipped (silhouette runs off the photo), so the
        # along-wall width is the reliable cue but may need a bigger upscale than
        # other flush pieces — allow more headroom for beds.
        _hi = 2.2 if _is_bed(p.get("type", "")) else 1.6
        f = float(np.clip(f, 0.5, _hi))
        if abs(f - 1.0) > 0.04:
            p["scale"] = (sc * f).tolist()
            print(f"  [silhouette_fit] idx={p.get('index')} {p.get('type')}: "
                  f"rendered {rend_w:.0f}×{rend_h:.0f}px vs mask {tgt_w:.0f}×{tgt_h:.0f}px "
                  f"→ scale ×{f:.2f}")


def _clamp_positions_to_room(placements: list[dict], camera: dict) -> None:
    """Keep free-standing (centre) objects fully inside the room footprint.

    A centre object's position comes from back-projecting its bbox to the floor;
    in diagonal corner views the far-image edge back-projects PAST a wall, leaving
    e.g. a coffee table floating outside the room.  Clamp the (x, z) so the
    object's whole footprint sits within [0, room_w] × [0, room_d].

    Only centre/None-affinity objects are clamped — wall-adjacent pieces are
    positioned by the wall snap and must not be pulled inward by their half-length.
    """
    import trimesh
    for p in placements:
        if p.get("is_carpet") or "glb_path" not in p or "position_m" not in p:
            continue   # skip carpets and any object that never got positioned
        if p.get("wall_affinity") in ("back", "left", "right"):
            continue
        gp = Path(p["glb_path"])
        if not gp.exists():
            continue
        room_w, room_d = _get_room_dims(gp, camera)
        if not room_w or not room_d:
            continue
        try:
            m = trimesh.load(str(gp), force="scene")
            geoms = list(m.geometry.values()) if isinstance(m, trimesh.Scene) else [m]
            g = max(geoms, key=lambda x: len(x.faces))
            ext = np.abs(g.bounds[1] - g.bounds[0])
        except Exception:
            continue
        sc = np.array(p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if sc.ndim == 0:
            sc = np.array([float(sc)] * 3)
        hh = 0.5 * float(max(ext[0] * sc[0], ext[2] * sc[2]))  # horizontal half-extent
        x, y, z = (float(v) for v in p["position_m"])
        lo_x, hi_x = hh, room_w - hh
        lo_z, hi_z = hh, room_d - hh
        nx = (room_w / 2.0) if lo_x > hi_x else min(max(x, lo_x), hi_x)
        nz = (room_d / 2.0) if lo_z > hi_z else min(max(z, lo_z), hi_z)
        if abs(nx - x) > 0.01 or abs(nz - z) > 0.01:
            p["position_m"] = [nx, y, nz]
            print(f"  [pos_clamp] idx={p.get('index')} {p.get('type')}: "
                  f"({x:.2f},{z:.2f}) outside room → ({nx:.2f},{nz:.2f})")


def _room_box_from_walls(any_glb_path) -> "tuple[float,float,float,float,float,float] | None":
    """Full room AABB (xmin,xmax,zmin,zmax,ymin,ymax) from walls.obj — the same
    authoritative box used for placement/render."""
    output_dir = Path(str(any_glb_path))
    for _ in range(6):
        output_dir = output_dir.parent
        wobj = output_dir / "walls.obj"
        if wobj.exists():
            try:
                v = np.array([[float(x) for x in l.split()[1:4]]
                              for l in open(wobj) if l.startswith("v ")], dtype=np.float64)
                if len(v):
                    return (float(v[:, 0].min()), float(v[:, 0].max()),
                            float(v[:, 2].min()), float(v[:, 2].max()),
                            float(v[:, 1].min()), float(v[:, 1].max()))
            except Exception:
                pass
    return None


def _enforce_room_containment(placements: list[dict], camera: dict, tol: float = 0.04) -> None:
    """FINAL safety net: keep every placed object fully inside the room box.

    The earlier clamps (_clamp_positions_to_room / inline bounds_clamp) only
    SHIFT position and skip wall-adjacent pieces, so an object that is simply too
    big to fit (e.g. a bed scaled up to fill a bottom-clipped silhouette) still
    pokes through the walls. This pass, run last before the render, also SCALES
    the object down (about its base-centre, so it stays on the floor) when its
    rotated world AABB exceeds the room on any axis, then shifts it in on the
    axes where it still overhangs. Applies to wall-flush pieces too (containment
    wins over silhouette fill)."""
    import trimesh
    box = None
    _cache: dict[str, "np.ndarray | None"] = {}
    for p in placements:
        if p.get("is_carpet") or "glb_path" not in p or "position_m" not in p:
            continue
        gp = Path(p["glb_path"])
        if not gp.exists():
            continue
        if box is None:
            box = _room_box_from_walls(gp)
            if box is None:
                return
        xmin, xmax, zmin, zmax, ymin, ymax = box
        key = str(gp)
        if key not in _cache:
            try:
                _cache[key] = np.asarray(trimesh.load(str(gp), force="mesh").vertices, dtype=np.float64)
            except Exception:
                _cache[key] = None
        verts0 = _cache[key]
        if verts0 is None or not len(verts0):
            continue
        sc = np.array(p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if sc.ndim == 0:
            sc = np.array([float(sc)] * 3)
        R = np.array(p.get("rotation_3x3", np.eye(3)), dtype=np.float64)
        pos = np.array(p["position_m"], dtype=np.float64)

        def _world(scale, position):
            return (verts0 * scale) @ R.T + position

        w = _world(sc, pos); b0, b1 = w.min(0), w.max(0)
        room_ext = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        obj_ext = b1 - b0
        # 1) scale down if too big on x, z (footprint) or y (height)
        fac = 1.0
        for ax, slack in ((0, 2 * tol), (2, 2 * tol), (1, tol)):
            if obj_ext[ax] > room_ext[ax] - slack and obj_ext[ax] > 1e-6:
                fac = min(fac, (room_ext[ax] - slack) / obj_ext[ax])
        if fac < 0.999:
            anchor = np.array([(b0[0] + b1[0]) / 2.0, b0[1], (b0[2] + b1[2]) / 2.0])
            pos = anchor + (pos - anchor) * fac
            sc = sc * fac
            p["scale"] = sc.tolist(); p["position_m"] = pos.tolist()
            if isinstance(p.get("size_m"), dict):
                for k in p["size_m"]:
                    p["size_m"][k] = float(p["size_m"][k]) * fac
            w = _world(sc, pos); b0, b1 = w.min(0), w.max(0)
            print(f"  [contain] idx={p.get('index')} {p.get('type')}: too big for room "
                  f"→ scaled ×{fac:.2f}")
        # 2) shift in on x/z where still overhanging.
        # NOTE: only X/Z (footprint).  The Y axis is deliberately NOT touched:
        # b0/b1 come from the RAW (un-grounded) mesh verts, whose local origin
        # sits mid-body, so b0[1] = pos[1] - half_height ≈ -0.35 m looks "below
        # the floor" even for a perfectly grounded piece.  Lifting pos[1] to
        # bring that raw bottom up to ymin sets pos[1] ≈ half_height; geo_save /
        # render then ground the BASE (via _find_base_y) to that lifted pos[1],
        # so the object hovers half its height.  Grounding already guarantees
        # base-on-floor, so no Y containment is needed here.
        shift = np.zeros(3)
        for ax, lo, hi in ((0, xmin, xmax), (2, zmin, zmax)):
            if b0[ax] < lo - tol:
                shift[ax] = lo - b0[ax]
            elif b1[ax] > hi + tol:
                shift[ax] = hi - b1[ax]
        if np.abs(shift).max() > 0.01:
            p["position_m"] = (pos + shift).tolist()
            print(f"  [contain] idx={p.get('index')} {p.get('type')}: shifted "
                  f"(dx={shift[0]:.2f}, dz={shift[2]:.2f}, dy={shift[1]:.2f}) inside room box")


def _rotation_for_affinity(
    wall_affinity: str,
    obj_pos: np.ndarray,
    cam_pos: np.ndarray,
    local_front: np.ndarray | None = None,
) -> np.ndarray:
    """Return a Y-axis rotation so the object's front faces the correct world direction.

    local_front:  unit vector in mesh-local space indicating where the front is
                  (from VLM detection).  Defaults to [0, 0, -1] (the 3D generator convention).

    Room coordinate system:
      X: 0=left wall → room_w=right wall
      Z: 0=back wall → room_d=front/camera
      Camera is at large Z, looking toward Z=0.

    Desired facing direction (world space):
      back   → face +Z (toward camera, away from back wall)
      left   → face +X (into room from left wall)
      right  → face -X (into room from right wall)
      centre → face toward camera on XZ plane

    Rotation R is applied as verts @ R.T.  We compute R so that R maps
    local_front to the desired world facing direction.
    """
    if local_front is None:
        local_front = np.array([0, 0, -1], dtype=np.float64)

    # Desired world facing direction
    if wall_affinity == "back":
        target = np.array([0, 0, 1], dtype=np.float64)
    elif wall_affinity == "left":
        target = np.array([1, 0, 0], dtype=np.float64)
    elif wall_affinity == "right":
        target = np.array([-1, 0, 0], dtype=np.float64)
    else:
        # centre: face toward camera
        dx = cam_pos[0] - obj_pos[0]
        dz = cam_pos[2] - obj_pos[2]
        norm = np.sqrt(dx*dx + dz*dz)
        if norm < 1e-6:
            target = np.array([0, 0, 1], dtype=np.float64)
        else:
            target = np.array([dx/norm, 0, dz/norm], dtype=np.float64)

    # Compute Y-rotation angle from local_front to target.
    # Both are XZ-plane vectors.  Use atan2 for each, take difference.
    angle_from = np.arctan2(float(local_front[0]), float(local_front[2]))
    angle_to   = np.arctan2(float(target[0]),      float(target[2]))
    angle = angle_to - angle_from

    return _rotation_y(angle)


# ── VLM front-face detection ──────────────────────────────────────────────────
# Cache: glb_path → local front direction vector (so we only query VLM once per GLB)
_front_face_cache: dict[str, np.ndarray] = {}

def _load_front_cache(furniture_dir: Path) -> None:
    """Load persisted front directions from front_cache.json."""
    cache_path = furniture_dir / "front_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            for key, vec in data.items():
                _front_face_cache[key] = np.array(vec, dtype=np.float64)
            print(f"[front_cache] loaded {len(data)} cached front direction(s)")
        except Exception as e:
            print(f"[front_cache] load failed: {e}")

def _save_front_cache(furniture_dir: Path) -> None:
    """Persist front directions to front_cache.json."""
    cache_path = furniture_dir / "front_cache.json"
    data = {k: (v.tolist() if hasattr(v, 'tolist') else list(v)) for k, v in _front_face_cache.items()}
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[front_cache] saved {len(data)} front direction(s) → {cache_path}")


def _render_cardinal_views(mesh_verts: np.ndarray, mesh_vc: np.ndarray | None,
                           size: int = 512,
                           top_view: bool = False) -> "Image.Image":
    """Render a GLB from 4 cardinal directions and stitch into a 2×2 grid.

    Layout:  top-left = View A (from +Z)   top-right = View B (from -Z)
             bot-left = View C (from -X)   bot-right = View D (from +X)

    If top_view=True, a 5th panel (View E, straight down from +Y) is appended
    as a full-width strip below the 2×2 grid.

    Uses face-based rendering via trimesh if available, otherwise falls back
    to high-quality vertex-cloud rendering with large splat dots.
    """
    from PIL import Image as PILImage, ImageDraw

    verts = mesh_verts.astype(np.float64)
    if mesh_vc is None or len(mesh_vc) != len(verts):
        vc = np.full((len(verts), 3), 160, dtype=np.uint8)
    else:
        vc = mesh_vc[:, :3].astype(np.uint8) if mesh_vc.shape[1] >= 3 else mesh_vc

    # 4 cardinal views + optional top-down
    views = [
        ("View A",   15,   0),   # looking from +Z toward -Z
        ("View B",   15, 180),   # looking from -Z toward +Z
        ("View C",   15,  90),   # looking from -X toward +X
        ("View D",   15, 270),   # looking from +X toward -X
    ]
    if top_view:
        views.append(("View E (top-down)", 90, 0))  # straight down from +Y

    panels = []
    centre = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts_c = verts - centre

    # Splat radius scaled to the projected vertex spacing so the dots OVERLAP into
    # a solid-looking surface instead of a sparse point cloud — the old fixed r=2
    # left gaps the VLM couldn't read for front/back (005241 cabinet reversed).
    # ~half the verts face the camera; spacing ≈ span_px / sqrt(n_visible).
    n_verts = len(verts)
    dot_r = int(np.clip(round((size - 20) / max(1.0, (n_verts / 2.0) ** 0.5)), 2, 7))

    for label, el_deg, az_deg in views:
        el, az = np.radians(el_deg), np.radians(az_deg)
        ca, sa = np.cos(az), np.sin(az)
        ce, se = np.cos(el), np.sin(el)
        Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
        R = Rx @ Ry
        vr = verts_c @ R.T

        span = max(np.ptp(vr[:, 0]), np.ptp(vr[:, 1])) * 1.15
        if span < 1e-6:
            panels.append(np.full((size, size, 3), 240, dtype=np.uint8))
            continue
        sc = (size - 20) / span
        mx = (vr[:, 0].min() + vr[:, 0].max()) / 2.0
        my = (vr[:, 1].min() + vr[:, 1].max()) / 2.0
        px = ((vr[:, 0] - mx) * sc + size / 2.0).astype(np.int32)
        py = (-(vr[:, 1] - my) * sc + size / 2.0).astype(np.int32)
        order = np.argsort(vr[:, 2])
        buf = np.full((size, size, 3), 240, dtype=np.uint8)
        r = dot_r
        valid = (px >= r) & (px < size - r) & (py >= r) & (py < size - r)
        for i in order:
            if not valid[i]:
                continue
            x, y = int(px[i]), int(py[i])
            buf[y-r:y+r, x-r:x+r] = vc[i]

        # Draw label
        panel_img = PILImage.fromarray(buf)
        draw = ImageDraw.Draw(panel_img)
        draw.text((8, 8), label, fill=(0, 0, 0))
        panels.append(np.array(panel_img))

    # 2×2 grid (+ optional full-width top-down strip)
    top = np.concatenate([panels[0], panels[1]], axis=1)
    bot = np.concatenate([panels[2], panels[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    if top_view and len(panels) == 5:
        # Stretch the top-down panel to full grid width and append below
        top_panel = PILImage.fromarray(panels[4]).resize((size * 2, size), PILImage.LANCZOS)
        grid = np.concatenate([grid, np.array(top_panel)], axis=0)
    return PILImage.fromarray(grid)


def _vlm_detect_front_direction(glb_path: str | Path, mesh_verts: np.ndarray,
                                mesh_vc: np.ndarray | None,
                                obj_type: str,
                                cache_suffix: str = "",
                                bbox_crop_path: Path | None = None,
                                wall_affinity: str = "centre") -> np.ndarray:
    """Ask VLM which cardinal direction is the front of this mesh.

    Uses TWO signals:
      1. A bbox crop from the reference image (how the object looks in the scene)
      2. 4 cardinal views of the 3D mesh

    The VLM matches the bbox crop to one of the 4 views to determine which
    mesh direction faces the camera.  Combined with wall_affinity, this tells
    us the mesh front direction.

    Returns a unit vector in LOCAL mesh space pointing in the front direction
    (e.g. [0, 0, 1] if front faces +Z).  Cached per GLB path + cache_suffix.
    """
    import base64, re, requests
    from io import BytesIO

    key = str(glb_path) + cache_suffix

    # ── Bed: GEOMETRIC front (overrides the VLM AND any stale cache — the VLM
    # often mistakes a long mattress side for the front).  A bed's headboard is
    # the tall end of its LONG axis; the FOOT (opposite the headboard) is the
    # front.  Detect the headboard as the end with the greater vertical reach,
    # then point the front away from it — so wall-snap puts the headboard (back)
    # against the wall.  Runs BEFORE the cache check so it can't return a stale
    # VLM result for a bed.
    if obj_type == "bed":
        v = np.asarray(mesh_verts, dtype=np.float64)
        # The headboard is the TALL structure at one end; the front is the FOOT
        # (away from it). Detect the headboard DIRECTLY from the tallest vertices
        # rather than "longest footprint axis + which end is taller" — that axis
        # choice is wrong on BROAD/LOW beds (near-square footprint), which made the
        # flush guard turn a SIDE of the bed to the wall instead of the headboard.
        cen_xz = v[:, [0, 2]].mean(0)
        tall = v[v[:, 1] > np.percentile(v[:, 1], 85)]        # tallest 15% = headboard
        hb = (tall[:, [0, 2]].mean(0) - cen_xz) if len(tall) else np.array([0.0, -1.0])
        if abs(hb[0]) >= abs(hb[1]):                          # snap to nearest cardinal axis
            hb = np.array([np.sign(hb[0]) or 1.0, 0.0]); axis = 0
        else:
            hb = np.array([0.0, np.sign(hb[1]) or 1.0]); axis = 2
        front = np.array([-hb[0], 0.0, -hb[1]], dtype=np.float64)   # foot = away from headboard
        _front_face_cache[key] = front
        print(f"  [front_detect] bed GEOMETRIC (tall-vertex): headboard at "
              f"{'+' if (hb[0]+hb[1])>0 else '-'}{'XYZ'[axis]} end → front={front.tolist()}")
        return front

    if key in _front_face_cache:
        return _front_face_cache[key]

    # SCENEWEAVE_ANIMATE_NO_TOPDOWN suppresses the top-down (View E) panel.
    _no_topdown = bool(os.environ.get("SCENEWEAVE_ANIMATE_NO_TOPDOWN"))
    grid_img = _render_cardinal_views(mesh_verts, mesh_vc,
                                      top_view=(obj_type == "bed" and not _no_topdown))

    stem = Path(str(glb_path)).stem
    debug_path = Path(str(glb_path)).parent / f"{stem}{cache_suffix}.front_detect.png"
    grid_img.save(str(debug_path))

    buf = BytesIO()
    grid_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Type-specific front description
    front_desc = {
        "sofa": "The FRONT of a sofa is the side with the seat cushions visible — where you would sit down. The BACK is the tall flat panel behind the cushions.",
        "chair": "The FRONT of a chair is the side with the seat visible — where you sit. You can see the seat surface and maybe armrests from the front. The BACK shows the rear of the backrest.",
        "armchair": "The FRONT is where you would sit — seat cushion and armrests visible. The BACK shows the rear panel.",
        "desk": "The FRONT of a desk is where you sit — the opening for your legs. The BACK is the panel behind the desk surface.",
        "coffee_table": "Coffee tables often have no strong front. Pick the view that shows the most usable/open side.",
        "plant": "Plants have no strong front. Pick any view — prefer the view showing the most foliage.",
        "bookcase": "The FRONT of a bookcase shows the open shelves. The BACK is the flat panel.",
        "cabinet": "The FRONT shows the doors/drawers. The BACK is flat.",
        "bed": (
            "The FRONT of a bed is the FOOT side — the open END of the mattress where you climb in, "
            "OPPOSITE the pillows and headboard. The BACK is the headboard side (tall vertical panel "
            "with pillows in front of it). Do NOT pick the headboard side as the front, even if its "
            "design is more prominent — beds are oriented headboard-against-the-wall, foot-into-the-room. "
            "Use View E (top-down) to clearly identify which SHORT end is the foot (open, no headboard) "
            "vs which SHORT end is the headboard (taller structure). The foot end is the FRONT."
        ),
    }
    desc = front_desc.get(obj_type, "The front is the main designed face of the object.")

    # Wall affinity context for the VLM
    affinity_hint = ""
    if wall_affinity in ("back", "left", "right"):
        affinity_hint = (
            f"\nCONTEXT: This {obj_type} is placed against the {wall_affinity} wall. "
            f"The camera sees its FRONT face (the side facing into the room). "
            f"The reference crop shows the camera-visible side."
        )
    else:
        affinity_hint = (
            f"\nCONTEXT: This {obj_type} is in the centre of the room. "
            f"The reference crop shows the camera-visible side."
        )

    # Build prompt — if we have a bbox crop, use it as the primary signal
    if bbox_crop_path and bbox_crop_path.exists():
        prompt = f"""You have TWO images to compare.

**Image 1** (reference photo): A crop of a **{obj_type.replace('_', ' ')}** from the actual room photo.
**Image 2** (3D mesh views): orthogonal views of the same object's 3D mesh:
  - View A (top-left) and View B (top-right) are OPPOSITE sides (front vs back)
  - View C (bottom-left) and View D (bottom-right) are the LEFT and RIGHT sides
  - View E (bottom strip, if present): straight top-down view — use this to identify the foot vs headboard end of a bed

{desc}
{affinity_hint}

**STEP 1 — Identify the FRONT of the object from the 4 mesh views:**
{desc}
Pick which view (A, B, C, or D) shows the FRONT of the object based on its functional design.
For a chair/sofa: the front shows the SEAT SURFACE and open leg area where you SIT DOWN.
For a desk: the front shows the LEGROOM OPENING where you put your legs.
For a bookcase: the front shows OPEN SHELVES.

**STEP 2 — Match the reference photo:**
Which mesh view most closely matches the ANGLE and SIDE visible in the reference photo (Image 1)?
Compare the overall SILHOUETTE SHAPE — proportions, outline, visible features.
Note: the reference may be a 3/4 angle, not exactly one of the 4 views. Pick the CLOSEST match.

Respond with ONLY a JSON object:
{{"front_view": "A" or "B" or "C" or "D", "camera_view": "A" or "B" or "C" or "D", "reason": "brief explanation"}}

front_view: which mesh view (A/B/C/D) shows the object's FUNCTIONAL FRONT (View E is context only, not a valid answer)
camera_view: which mesh view best matches the reference photo angle"""

        crop_b64 = _encode_image(bbox_crop_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]
    else:
        prompt = f"""You see 4 views of a 3D {obj_type.replace('_', ' ')} mesh from 4 different angles arranged in a 2×2 grid:
- View A (top-left)
- View B (top-right) — opposite side from A (180°)
- View C (bottom-left) — 90° from A
- View D (bottom-right) — 90° from A, opposite from C

{desc}

Look carefully at all 4 views. Which one shows the FRONT?

Respond with ONLY a JSON object:
{{"front_view": "A" or "B" or "C" or "D", "reason": "brief explanation"}}"""

        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]

    # View label → local-space front direction vector.
    # Camera positions:
    #   A: camera at +Z → we see the side facing +Z → front direction = [0, 0, +1]
    #   B: camera at -Z → we see the side facing -Z → front direction = [0, 0, -1]
    #   C: camera at -X → we see the side facing -X → front direction = [-1, 0, 0]
    #   D: camera at +X → we see the side facing +X → front direction = [+1, 0, 0]
    view_to_front = {
        "A": np.array([0, 0,  1], dtype=np.float64),
        "B": np.array([0, 0, -1], dtype=np.float64),
        "C": np.array([-1, 0, 0], dtype=np.float64),
        "D": np.array([ 1, 0, 0], dtype=np.float64),
    }

    default_front = np.array([0, 0, -1], dtype=np.float64)  # the 3D generator convention

    # Check if any variant of this path exists in cache (absolute vs relative).
    # GLB paths may be resolved to absolute during placement but stored as
    # relative in the cache file.
    # IMPORTANT: raw meshes (objects/) and canonical meshes (objects_detilted/)
    # have different coordinate frames.  Only match within the same category.
    glb_name = Path(str(glb_path)).name
    glb_str = str(glb_path)
    _is_detilted = "objects_detilted" in glb_str
    for cached_key, cached_val in _front_face_cache.items():
        if cached_key.endswith(glb_name + cache_suffix):
            _cached_is_detilted = "objects_detilted" in cached_key
            if _cached_is_detilted != _is_detilted:
                continue  # skip cross-category matches
            print(f"  [front_detect] fuzzy cache hit: {cached_key} → {cached_val.tolist()}")
            _front_face_cache[key] = cached_val  # also store under canonical key
            return cached_val

    try:
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group())
            # Always use front_view — the functional front of the object.
            # This is the most reliable signal: the VLM identifies the SEAT
            # side of a chair, the OPEN SHELVES of a bookcase, etc.
            # camera_view (which view matches the reference photo) is logged
            # for debugging but NOT used for orientation.
            if bbox_crop_path and result.get("front_view"):
                cam_view = str(result.get("camera_view", "")).upper().strip()
                front_view = str(result.get("front_view", "A")).upper().strip()
                reason = result.get("reason", "")
                front_dir = view_to_front.get(front_view, default_front)
                print(f"  [front_detect] bbox+VLM: camera=View {cam_view}, "
                      f"front=View {front_view} → dir={front_dir.tolist()}  "
                      f"(wall_affinity={wall_affinity})  ({reason})")
            else:
                view = str(result.get("front_view", "A")).upper().strip()
                reason = result.get("reason", "")
                front_dir = view_to_front.get(view, default_front)
                print(f"  [front_detect] front=View {view} → local_dir={front_dir.tolist()}  ({reason})")
            _front_face_cache[key] = front_dir
            return front_dir
    except Exception as e:
        print(f"  [front_detect] VLM failed: {e} — assuming -Z (not cached)")

    # Do NOT cache the default — preserves disk-loaded values for future runs.
    return default_front


def _compute_scale(
    glb_bounds: np.ndarray,  # (2, 3)
    obj_type: str,
) -> np.ndarray:
    """Return per-axis scale (3,) so GLB fits default furniture dimensions."""
    tw, th, td = _DEFAULT_SIZES.get(obj_type, _DEFAULT_SIZE_FALLBACK)
    bmin, bmax = glb_bounds
    gw = float(bmax[0] - bmin[0]) or 1.0
    gh = float(bmax[1] - bmin[1]) or 1.0
    gd = float(bmax[2] - bmin[2]) or 1.0
    return np.array([tw / gw, th / gh, td / gd], dtype=np.float64)


# ── Wall-mounted fallback placement ──────────────────────────────────────────

def _determine_wall_from_bbox(
    bx1: float, by1: float, bx2: float, by2: float,
    cam: dict,
    room_w: float,
    room_d: float,
) -> tuple[str, np.ndarray]:
    """Determine which wall an object is on and its world-space centre.

    Primary: bbox horizontal edge position (right edge > 75 % → right wall;
    left edge < 25 % → left wall).  Falls back to back-projecting the bbox
    centre ray to each wall plane and picking the closest hit.

    Returns (wall_name, world_pt).
    """
    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    W   = int(cam["width_px"])
    H   = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])
    fx  = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    # Horizontal edge heuristic — reliable for side-wall objects.
    # Require BOTH edges on the same side of center to avoid classifying large
    # central objects (e.g. a wide sofa spanning bx1=35%–bx2=90%) as side-wall.
    bx2_frac = float(bx2) / W
    bx1_frac = float(bx1) / W
    _z_mid = room_d / 2.0 if room_d > 0 else 1.5
    if bx2_frac > 0.75 and bx1_frac > 0.50:
        return "right", np.array([room_w, 1.5, _z_mid])
    if bx1_frac < 0.25 and bx2_frac < 0.50:
        return "left", np.array([0.0, 1.5, _z_mid])

    px_c = (bx1 + bx2) / 2.0
    py_c = (by1 + by2) / 2.0
    ray = _backproject_pixel(px_c, py_c, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)

    # Try intersecting with each wall plane
    walls: list[tuple[str, float, np.ndarray]] = []  # (name, t, point)
    # Back wall: Z = 0
    if abs(ray[2]) > 1e-9:
        t = (0.0 - cam_pos[2]) / ray[2]
        if t > 0:
            pt = cam_pos + t * ray
            if 0 <= pt[0] <= room_w and 0 <= pt[1] <= 3.5:
                walls.append(("back", t, pt))
    # Left wall: X = 0
    if abs(ray[0]) > 1e-9:
        t = (0.0 - cam_pos[0]) / ray[0]
        if t > 0:
            pt = cam_pos + t * ray
            if 0 <= pt[2] <= room_d and 0 <= pt[1] <= 3.5:
                walls.append(("left", t, pt))
    # Right wall: X = room_w
    if abs(ray[0]) > 1e-9:
        t = (room_w - cam_pos[0]) / ray[0]
        if t > 0:
            pt = cam_pos + t * ray
            if 0 <= pt[2] <= room_d and 0 <= pt[1] <= 3.5:
                walls.append(("right", t, pt))

    if not walls:
        # Fallback: back wall at image centre
        return "back", np.array([room_w / 2, 1.5, 0.0])

    # Pick closest wall (smallest t)
    walls.sort(key=lambda w: w[1])
    return walls[0][0], walls[0][2]


def _estimate_wall_object_size(
    bx1: float, by1: float, bx2: float, by2: float,
    world_pt: np.ndarray,
    cam: dict,
) -> dict:
    """Estimate width_m and height_m for a wall-mounted object from its bbox."""
    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    W   = int(cam["width_px"])
    H   = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])
    fx  = W / (2.0 * np.tan(np.radians(hfov / 2.0)))

    # Camera-space depth
    z_c = float(np.dot(world_pt - cam_pos, fwd_v))
    if z_c < 0.1:
        z_c = 2.0

    pixel_w = float(bx2 - bx1)
    pixel_h = float(by2 - by1)
    physical_w = pixel_w / fx * z_c
    physical_h = pixel_h / fx * z_c

    # Clamp to reasonable wall-mounted object sizes
    physical_w = float(np.clip(physical_w, 0.1, 2.5))
    physical_h = float(np.clip(physical_h, 0.1, 2.5))

    return {
        "width_m":  round(physical_w, 3),
        "height_m": round(physical_h, 3),
        "depth_m":  0.05,  # wall-mounted objects are thin
    }


def _backproject_bbox_to_wall(
    bx1: float, by1: float, bx2: float, by2: float,
    wall: str,
    cam: dict,
    room_w: float,
    room_d: float,
) -> np.ndarray | None:
    """Back-project bbox centre onto a specific wall plane.

    Returns world-space point on the wall surface, or None if the ray
    doesn't hit the wall.
    """
    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    W   = int(cam["width_px"])
    H   = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])
    fx  = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    px_c = (bx1 + bx2) / 2.0
    py_c = (by1 + by2) / 2.0
    ray = _backproject_pixel(px_c, py_c, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)

    # Intersect with the specified wall plane
    if wall == "back":
        if abs(ray[2]) < 1e-9:
            return None
        t = (0.0 - cam_pos[2]) / ray[2]
    elif wall == "left":
        if abs(ray[0]) < 1e-9:
            return None
        t = (0.0 - cam_pos[0]) / ray[0]
    elif wall == "right":
        if abs(ray[0]) < 1e-9:
            return None
        t = (room_w - cam_pos[0]) / ray[0]
    else:
        return None

    if t <= 0:
        return None
    return cam_pos + t * ray


def compute_wall_mounted_placement(
    entry: dict,
    cam: dict,
    output_dir: Path,
    detilt_meta: dict | None = None,
    use_vlm: bool = True,
) -> dict | None:
    """Place a wall-mounted object detected in the furniture phase.

    Uses _transform_glb_to_wall from the wall-mounted pipeline to handle
    GLB scaling, orientation, and wall snapping.

    Returns a placement dict or None on failure.
    """
    glb_path = Path(entry.get("glb_file", ""))
    if not glb_path.exists():
        print(f"  [wm_fallback] GLB not found: {glb_path}")
        return None

    bx1, by1, bx2, by2 = [float(v) for v in entry["box_px"]]
    room_w, room_d = _get_room_dims(glb_path, cam)

    # Use VLM-specified wall if available; fall back to wall_affinity; then bbox ray-casting
    wall = entry.get("wall")
    if not wall or wall not in ("back", "left", "right"):
        wa = entry.get("wall_affinity", "")
        if wa in ("back", "left", "right"):
            wall = wa
    if wall and wall in ("back", "left", "right"):
        world_pt = _backproject_bbox_to_wall(bx1, by1, bx2, by2, wall, cam, room_w, room_d)
        if world_pt is None:
            print(f"  [wm_fallback] back-projection to {wall} wall failed, trying ray-cast")
            wall, world_pt = _determine_wall_from_bbox(bx1, by1, bx2, by2, cam, room_w, room_d)
    else:
        wall, world_pt = _determine_wall_from_bbox(bx1, by1, bx2, by2, cam, room_w, room_d)

    # Estimate size from bbox perspective projection
    size = _estimate_wall_object_size(bx1, by1, bx2, by2, world_pt, cam)

    # Load room info for _transform_glb_to_wall
    room = {"floor_width_m": room_w, "floor_depth_m": room_d, "ceiling_height_m": 2.7}
    fp_path = output_dir / "floorplan_analysis.json"
    if fp_path.exists():
        try:
            with open(fp_path) as f:
                fp = json.load(f)
            room.update(fp.get("room", {}))
        except Exception:
            pass

    cam_pos = np.array(cam["position_m"], dtype=np.float64)
    obj_type = entry.get("type", "wall_art")

    print(f"  [wm_fallback] wall={wall}  world_pt={np.round(world_pt, 2).tolist()}  "
          f"size={size['width_m']:.2f}×{size['height_m']:.2f}m")

    # ── Front-face check for wall-mounted objects ──────────────────────────
    # Determine which side of the mesh is the functional front (open shelves,
    # painted canvas, etc.) so we can ensure it faces outward from the wall.
    # Determine flip_z: whether to negate the wall-normal axis after OBB.
    #
    # After OBB, depth is canonicalized to +Z → maps to +wall_normal (room side).
    # Hunyuan3D meshes have front at +Z (camera at +Z looking at -Z), so after
    # OBB the front already faces the room.
    #   → Front at +Z: no flip needed (flip_z=False)
    #   → Front at -Z: front faces wall → flip needed (flip_z=True)
    #
    # Priority: detilt_results → VLM live → default (no flip, assume +Z front).
    import trimesh as _wm_tm
    _flip_z = False
    _front_resolved = False

    _is_canon = bool(entry.get("is_canonical_glb"))
    if _is_canon:
        # Canonical GLB: front is at -Z (baked by detilt_and_orient.py).
        # After OBB, -Z face ends up at min wall-normal → facing wall.
        # flip_z=True puts it facing the room.
        _flip_z = True
        _front_resolved = True
        print(f"  [wm_front] canonical GLB — front at -Z, flip_z=True (face outward)")
    elif detilt_meta:
        _fla = detilt_meta.get("front_local_axis", "")
        if _fla == "+Z":
            _flip_z = False   # front already faces room after OBB
            _front_resolved = True
            print(f"  [wm_front] detilt says front=+Z → no flip needed")
        elif _fla == "-Z":
            _flip_z = True    # front faces wall → flip to face room
            _front_resolved = True
            print(f"  [wm_front] detilt says front=-Z → flip_z=True")
        elif _fla:
            _front_resolved = True
            print(f"  [wm_front] detilt says front={_fla} — using default orientation")

    # 2) Fallback: VLM live front detection
    if not _front_resolved and use_vlm:
        try:
            _wm_mesh = _wm_tm.load(str(glb_path), force="mesh")
            if isinstance(_wm_mesh, _wm_tm.Scene):
                _wm_mesh = _wm_tm.util.concatenate(_wm_mesh.dump())
            _wm_vc = None
            try:
                if hasattr(_wm_mesh.visual, "to_color"):
                    _wm_vc = _wm_mesh.visual.to_color().vertex_colors
            except Exception:
                pass
            crop_file = entry.get("crop_file", "")
            _wm_crop = None
            if crop_file:
                furn_dir = glb_path.parent.parent
                _wm_crop = furn_dir / "segmented" / crop_file
                if not _wm_crop.exists():
                    _wm_crop = None
            local_front = _vlm_detect_front_direction(
                glb_path, _wm_mesh.vertices, _wm_vc, obj_type,
                cache_suffix=f"_wm_{wall}",
                bbox_crop_path=_wm_crop,
                wall_affinity=wall,
            )
            if local_front is not None:
                if local_front[2] > 0.5:
                    _flip_z = False  # front at +Z → already faces room
                    _front_resolved = True
                    print(f"  [wm_front] VLM says front is +Z → no flip needed")
                elif abs(local_front[0]) > 0.5:
                    _front_resolved = True
                    print(f"  [wm_front] VLM says front on X-axis ({local_front.tolist()}) — using default")
                else:
                    _flip_z = True   # front at -Z → faces wall → flip
                    _front_resolved = True
                    print(f"  [wm_front] VLM says front is -Z → flip_z=True")
        except Exception as e:
            print(f"  [wm_front] VLM front detection failed: {e}")

    if not _front_resolved:
        print(f"  [wm_front] no front info — assuming Hunyuan +Z front (no flip)")

    # Hunyuan3D meshes are Y-up — no Y-flip needed.
    # (The furniture placement path works correctly without Y-flip, confirming
    # Y-up convention for these meshes.)
    _flip_y = False

    result = _transform_glb_to_wall(
        glb_path, wall, world_pt, size, room,
        cam_pos=cam_pos,
        flip_y=_flip_y,
        flip_z=_flip_z,
        detilt=False,       # wall-mounted objects are flat
        force_upright=True,
    )
    if result is None:
        print(f"  [wm_fallback] _transform_glb_to_wall returned None")
        return None

    verts, faces, vc = result
    return {
        "index":         entry["index"],
        "type":          obj_type,
        "glb_path":      str(glb_path),
        "wall":          wall,
        "world_pt":      world_pt.tolist(),
        "size_m":        size,
        "wall_mounted":  True,
        "_verts":        verts,
        "_faces":        faces,
        "_vc":           vc,
    }


# ── Convex-hull XZ footprint helpers ──────────────────────────────────────────

def _compute_xz_hull(verts: np.ndarray) -> np.ndarray | None:
    """Return (K,2) convex hull of mesh vertices projected onto XZ plane.
    Returned points are in local mesh space (before scale/rotation/translation).
    Returns None if the hull cannot be computed (degenerate mesh)."""
    try:
        from scipy.spatial import ConvexHull
        pts = verts[:, [0, 2]].astype(np.float64)
        # Deduplicate (many meshes have repeated vertices)
        pts = np.unique(pts, axis=0)
        if len(pts) < 3:
            return None
        ch = ConvexHull(pts)
        return pts[ch.vertices]      # (K,2) hull vertices
    except Exception:
        return None


def _col_xz_half_extents(p: dict) -> tuple[float, float]:
    """World-space XZ half-extents for a placement dict.

    Module-level so non-`run` callers (e.g. _vlm_scene_review at L~3812) can
    use the same formula as the placement loop's local helper at L~6678 — the
    two definitions must match. If you change one, change both.
    """
    sm = p.get("size_m", {})
    # width_m/depth_m are semantic (along-front vs perpendicular); undo the
    # [axis_swap] relabel to get raw mesh-X/mesh-Z halves before combining
    # with R, which rotates the RAW local axes.
    _pf = p.get("_local_front", [0, 0, -1])
    if abs(float(_pf[0])) > abs(float(_pf[2])):
        hx_local = sm.get("depth_m", 0.5) / 2.0
        hz_local = sm.get("width_m", 0.5) / 2.0
    else:
        hx_local = sm.get("width_m", 0.5) / 2.0
        hz_local = sm.get("depth_m", 0.5) / 2.0
    R = np.array(p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
    hx = abs(float(R[0][0])) * hx_local + abs(float(R[0][2])) * hz_local
    hz = abs(float(R[2][0])) * hx_local + abs(float(R[2][2])) * hz_local
    return hx, hz


def _poly_collides_sat(
    hull_a: np.ndarray, cx_a: float, cz_a: float, R_a: np.ndarray,
    sc_a: np.ndarray,
    hull_b: np.ndarray, cx_b: float, cz_b: float, R_b: np.ndarray,
    sc_b: np.ndarray,
    margin: float = 0.0,
) -> bool:
    """Separating Axis Theorem (SAT) test for two convex polygons in XZ.

    hull_X : (K,2) XZ vertices in local mesh space
    cx/cz  : world-space centre
    R_X    : 3×3 rotation matrix
    sc_X   : scale vector [sx, sy, sz]
    margin : extra separation distance treated as collision

    Returns True if the polygons overlap (i.e. there is a collision).
    """
    def _transform(hull, cx, cz, R, sc):
        # Scale along the mesh-local X and Z axes then apply the XZ 2-D
        # rotation block from the full 3×3 matrix, then translate.
        h = hull * np.array([float(sc[0]), float(sc[2])])  # local scale
        rot2 = np.array([[float(R[0, 0]), float(R[0, 2])],
                         [float(R[2, 0]), float(R[2, 2])]])
        return h @ rot2.T + np.array([cx, cz])

    R_a = np.asarray(R_a, dtype=np.float64)
    R_b = np.asarray(R_b, dtype=np.float64)
    poly_a = _transform(hull_a, cx_a, cz_a, R_a, sc_a)
    poly_b = _transform(hull_b, cx_b, cz_b, R_b, sc_b)

    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = float(np.linalg.norm(axis))
            if norm < 1e-9:
                continue
            axis /= norm
            proj_a = poly_a @ axis
            proj_b = poly_b @ axis
            # Gap > margin means this axis separates the polygons
            gap = max(proj_a.min(), proj_b.min()) - min(proj_a.max(), proj_b.max())
            if gap > margin:
                return False      # separating axis found — no collision
    return True                   # no separating axis found — collision


# ── Floor-placement computation ────────────────────────────────────────────────

def compute_floor_placement(
    entry: dict,
    cam: dict,
    support_y: float = 0.0,
    support_xz: tuple[float, float] | None = None,
    allow_rotation_types: frozenset = frozenset(),
    use_vlm: bool = True,
) -> dict | None:
    """Compute position_m, rotation_3x3, scale for one furniture object.

    support_y:  Y position of the surface this object rests on (0 = floor).
                Used for objects placed on top of other furniture.
    support_xz: If provided, override the back-projected XZ position with the
                (X, Z) centre of the supporting object (centres the item on it).

    Returns a placement dict compatible with wall_mounted renderer, or None
    if floor intersection fails.
    """
    import trimesh

    glb_rel = entry.get("glb_file", "")
    glb_path = Path(glb_rel)
    if not glb_path.exists():
        print(f"  [place] GLB not found: {glb_path}")
        return None

    # Camera parameters
    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)

    W   = int(cam["width_px"])
    H   = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])
    fx  = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    # Back-project the bottom-centre of the bbox to the floor.
    # Using the bottom edge (by2) is more accurate: it's where the object
    # contacts the floor rather than the visual centre of the object.
    # For on-top-of objects the floor_pt is only needed for size estimation;
    # the final XZ position is overridden by support_xz.
    bx1, by1, bx2, by2 = entry["box_px"]
    px_img = (bx1 + bx2) / 2.0
    py_img = float(by2)

    # Edge-aware adjustment for partial visibility.  When the bbox touches
    # a vertical image edge the visible portion isn't the full object — the
    # cut-off side extends beyond the frame.  Using the visible-bbox centre
    # as the back-projection anchor biases the resulting world position
    # toward the visible side; for a chair half cut off at the left edge,
    # for example, the chair lands ~half a chair-width too far right.
    # Shift px_img toward the cut-off side by a fraction of the visible
    # bbox width so the back-projected position moves toward where the
    # full object would be.  The shift is a heuristic — true cut-off
    # fraction is unknown — but ~30 % of visible width gives a workable
    # approximation when the cut-off is roughly half.
    img_w_full = float(cam.get("width_px", 0)) or 1.0
    _edge_thr = 5  # px tolerance for "touching the edge"
    _cut_off_frac = 0.30
    _bbox_w = float(bx2 - bx1)
    if bx1 < _edge_thr:
        px_img -= _bbox_w * _cut_off_frac
        print(f"  [backproj] bbox touches LEFT edge — shifting anchor "
              f"left by {_bbox_w * _cut_off_frac:.0f}px "
              f"({(bx1+bx2)/2.0:.0f} → {px_img:.0f})")
    elif bx2 > img_w_full - _edge_thr:
        px_img += _bbox_w * _cut_off_frac
        print(f"  [backproj] bbox touches RIGHT edge — shifting anchor "
              f"right by {_bbox_w * _cut_off_frac:.0f}px "
              f"({(bx1+bx2)/2.0:.0f} → {px_img:.0f})")

    ray = _backproject_pixel(px_img, py_img, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
    floor_pt = _ray_floor_intersect(cam_pos, ray)
    if floor_pt is not None:
        print(f"  [backproj] bbox_bot=({px_img:.0f},{py_img:.0f}) → "
              f"floor=({floor_pt[0]:.3f}, {floor_pt[2]:.3f})")
    if floor_pt is None:
        if support_xz is not None:
            # On-top-of object whose bbox bottom doesn't hit the floor — use a
            # dummy floor_pt at the same XZ as the supporting object.
            floor_pt = np.array([support_xz[0], 0.0, support_xz[1]], dtype=np.float64)
            print(f"  [place] floor intersection failed; using support XZ {support_xz}")
        else:
            print(f"  [place] floor intersection failed for idx={entry['index']} — skipping")
            return None

    # Load GLB.  For canonical (pre-de-tilted) GLBs the the 3D generator ground slab has
    # already been removed by detilt_and_orient.py — skip the slab pass to
    # avoid double-processing.
    is_canonical = bool(entry.get("is_canonical_glb"))
    try:
        mesh = trimesh.load(str(glb_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        if not is_canonical:
            result = _remove_ground_faces(mesh)
            mesh = result[0] if isinstance(result, tuple) else result
        else:
            print(f"  [place] using canonical GLB (ground + detilt already baked)")
        bounds = mesh.bounds  # (2, 3)
    except Exception as e:
        print(f"  [place] trimesh load failed: {e}")
        return None

    obj_type = entry.get("type", "other")
    wall_affinity = entry.get("wall_affinity", "centre")
    # back_flush: object's back face touches the wall (sofa, desk, bookcase…).
    # Fall back to a type-based heuristic if not provided by the VLM.
    _FLUSH_TYPES = {"sofa", "armchair", "bookcase", "cabinet", "desk", "bed"}
    back_flush = entry.get(
        "back_flush",
        wall_affinity != "centre" and obj_type in _FLUSH_TYPES,
    )

    # Room dimensions: prefer floorplan_analysis.json (always correct),
    # fall back to wall_context (can be zero in camera_vggt.json).
    room_w, room_d = _get_room_dims(glb_path, cam)
    room_h_ceil = 2.7
    # For flush-against-wall objects the snap in _apply_floor_transform handles
    # the gap; keep a token offset here just to avoid z-fighting.
    WALL_GAP  = 0.01 if back_flush else _WALL_GAP

    # ── Wall-affinity sanity check via bbox-ray ───────────────────────────────
    # The placement_analysis VLM picks wall_affinity by looking at the IMAGE,
    # which can confuse "right of the bed in image space" with "world right
    # wall" when the camera is rotated.  Only override when the analysis's
    # wall is BEHIND or BESIDE the camera (i.e. invisible) — for walls that
    # ARE in the camera's forward half-space, trust the analysis even if the
    # bbox-centre ray hits a different wall first (large objects like beds
    # extend across the room and their bbox-centre ray can hit a side wall).
    if back_flush and wall_affinity in ("back", "left", "right"):
        try:
            _cam_pos_xz = np.array(
                [cam["position_m"][0], cam["position_m"][2]], dtype=np.float64
            )
            _look_at_xz = np.array(
                [cam["look_at_m"][0], cam["look_at_m"][2]], dtype=np.float64
            )
            _fwd = _look_at_xz - _cam_pos_xz
            _fnorm = float(np.linalg.norm(_fwd))
            if _fnorm > 1e-6:
                _fwd = _fwd / _fnorm
                # A wall is "visible" if the camera is on the inside face of
                # it (inside the room).  We intentionally do NOT require the
                # camera forward to point toward the wall — the camera looks
                # toward the back wall but can still see left/right walls in
                # its peripheral view.
                _wall_visible = {
                    "back":  _cam_pos_xz[1] > 0,
                    "left":  _cam_pos_xz[0] > 0,
                    "right": _cam_pos_xz[0] < room_w,
                }
                if not _wall_visible.get(wall_affinity, True):
                    _bx1, _by1, _bx2, _by2 = entry["box_px"]
                    _ray_wall, _ray_pt = _determine_wall_from_bbox(
                        float(_bx1), float(_by1), float(_bx2), float(_by2),
                        cam, room_w, room_d,
                    )
                    if _ray_wall in ("back", "left", "right") and _ray_wall != wall_affinity:
                        print(f"  [wall_affinity] '{wall_affinity}' wall is "
                              f"behind/beside camera (cam_xz={_cam_pos_xz.tolist()}, "
                              f"fwd_xz={_fwd.round(2).tolist()}) — overriding to "
                              f"bbox-ray-hit '{_ray_wall}' at "
                              f"({_ray_pt[0]:.2f},{_ray_pt[2]:.2f})")
                        wall_affinity = _ray_wall
                        entry["wall_affinity"] = _ray_wall
        except Exception as _wa_e:
            print(f"  [wall_affinity] bbox-ray check skipped: {_wa_e}")

    # ── Perspective-based size estimation ────────────────────────────────────
    bx1, by1, bx2, by2 = entry["box_px"]
    def_w, def_h, def_d = _DEFAULT_SIZES.get(obj_type, _DEFAULT_SIZE_FALLBACK)

    _bbox_area_px = (float(bx2) - float(bx1)) * (float(by2) - float(by1))
    _ref_size_m   = entry.get("_ref_size_m")   # already-placed sibling's size
    _ref_box_px   = entry.get("_ref_box_px")   # pre-scanned same-type reliable bbox
    _bbox_too_small = _bbox_area_px < _MIN_RELIABLE_BBOX_AREA_PX

    if _bbox_too_small:
        # Priority: (1) borrow perspective sizing from a reliable sibling's
        # bbox, (2) reuse an already-placed sibling's size, (3) defaults.
        _ref_used = False
        if _ref_box_px is not None and len(_ref_box_px) == 4:
            rbx1, rby1, rbx2, rby2 = [float(v) for v in _ref_box_px]
            _ref_ray = _backproject_pixel(
                (rbx1 + rbx2) / 2.0, float(rby2),
                cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy,
            )
            _ref_floor = _ray_floor_intersect(cam_pos, _ref_ray)
            if _ref_floor is not None:
                raw_w, raw_h, raw_d = _estimate_size_from_bbox(
                    rbx1, rby1, rbx2, rby2,
                    _ref_floor, cam_pos, fwd_v, right_v, up_c_v, fx, cx, cy, obj_type,
                )
                est_w = float(np.clip(raw_w, def_w * 0.5, def_w * 2.5))
                est_h = float(np.clip(raw_h, def_h * 0.5, def_h * 2.5))
                est_d = raw_d
                _ref_used = True
                print(f"  [small_bbox] area={_bbox_area_px:.0f} px² < {_MIN_RELIABLE_BBOX_AREA_PX} "
                      f"— borrowing size from reliable sibling bbox: "
                      f"W={est_w:.2f} H={est_h:.2f} D={est_d:.2f}")
        if not _ref_used and _ref_size_m:
            est_w = float(_ref_size_m.get("width_m",  def_w))
            est_h = float(_ref_size_m.get("height_m", def_h))
            est_d = float(_ref_size_m.get("depth_m",  def_d))
            _ref_used = True
            print(f"  [small_bbox] area={_bbox_area_px:.0f} px² — using already-placed "
                  f"sibling size W={est_w:.2f} H={est_h:.2f} D={est_d:.2f}")
        if not _ref_used:
            est_w, est_h, est_d = def_w, def_h, def_d
            print(f"  [small_bbox] area={_bbox_area_px:.0f} px² — no reliable sibling, "
                  f"using default W={est_w:.2f} H={est_h:.2f} D={est_d:.2f}")
    else:
        raw_w, raw_h, raw_d = _estimate_size_from_bbox(
            float(bx1), float(by1), float(bx2), float(by2),
            floor_pt, cam_pos, fwd_v, right_v, up_c_v, fx, cx, cy, obj_type,
        )
        est_w = float(np.clip(raw_w, def_w * 0.5, def_w * 2.5))
        est_h = float(np.clip(raw_h, def_h * 0.5, def_h * 2.5))
        est_d = raw_d
        print(f"  [size] perspective W={raw_w:.2f} H={raw_h:.2f}  "
              f"→ clamped W={est_w:.2f} H={est_h:.2f} D={est_d:.2f} m")

    # ── VLM numerical pre-estimation (position + size) ────────────────────────
    furn_dir  = glb_path.parent.parent
    crop_file = entry.get("crop_file", "")
    crop_path = furn_dir / "segmented" / crop_file if crop_file else None
    output_dir = furn_dir.parent
    scene_img_candidates = [
        output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
        output_dir / "render_final.png",
        output_dir / "render.png",
    ]
    scene_img = next((p for p in scene_img_candidates if p.exists()), None)

    vlm_num: dict = {}
    if use_vlm and crop_path and crop_path.exists() and not _bbox_too_small:
        vlm_num = _vlm_estimate_placement(
            crop_path, scene_img, obj_type, wall_affinity, room_w, room_d, cam,
        )
        if vlm_num:
            conf = float(vlm_num.get("confidence", 0))
            print(f"  [vlm_num] offset={vlm_num.get('wall_offset_m')}m  "
                  f"W={vlm_num.get('width_m')} H={vlm_num.get('height_m')} D={vlm_num.get('depth_m')}  "
                  f"conf={conf:.2f}  {vlm_num.get('notes','')}")
            # VLM sets initial reasonable scale; silhouette mask_align refines.
            if conf >= 0.6:
                vw = float(vlm_num.get("width_m") or 0)
                vh = float(vlm_num.get("height_m") or 0)
                vd = float(vlm_num.get("depth_m") or 0)
                if 0.1 < vw < 6.0: est_w = float(np.clip(vw, def_w*0.4, def_w*2.5))
                if 0.1 < vh < 4.0: est_h = float(np.clip(vh, def_h*0.4, def_h*2.5))
                if 0.1 < vd < 4.0: est_d = float(np.clip(vd, def_d*0.4, def_d*2.5))
                print(f"  [vlm_num] overriding size → W={est_w:.2f} H={est_h:.2f} D={est_d:.2f}")

                # ── Per-type depth sanity cap ─────────────────────────────────
                # The VLM occasionally hallucinates a depth larger than is
                # physically plausible for the type (e.g. a 1.2 m chair). When
                # combined with the room-bounds clamp later in the pipeline,
                # an oversized depth pulls the object away from its silhouette
                # to make the (now-too-large) footprint fit inside the room.
                # Cap depth based on type so the silhouette wins.
                _t = obj_type.lower().replace("-", "_")
                _depth_max: float | None = None
                if _t in {"chair", "armchair", "office_chair", "stool", "ottoman"}:
                    _depth_max = max(est_w * 0.85, 0.65)  # chairs: depth ≤ 85% of width or 0.65 m
                elif _t in {"sofa", "loveseat", "couch"}:
                    _depth_max = 1.05                    # sofas: depth ≤ 1.05 m
                elif _t in {"desk", "dining_table"}:
                    _depth_max = max(est_w * 0.9, 0.90)  # desks: typically deeper than wide-ish
                elif _t in {"bench"}:
                    _depth_max = 0.55
                # coffee_table / side_table / end_table / nightstand / shelf / etc.
                # are intentionally left uncapped — those *can* legitimately be
                # deeper (long sectional coffee tables, wide nightstands, etc.).
                if _depth_max is not None and est_d > _depth_max:
                    print(f"  [size_sanity] {obj_type} depth {est_d:.2f} > cap {_depth_max:.2f} "
                          f"→ clamping (prevents room-clamp drift)")
                    est_d = float(_depth_max)

    # Track VLM-estimated width for wall_plane capping
    _vlm_est_w = est_w if (vlm_num and float(vlm_num.get("confidence", 0)) >= 0.6) else None

    # ── Wall-plane back-projection for flush wall-affinity objects ───────────
    # Back-project the bbox edges to a plane parallel to the affinity wall,
    # offset by est_d/2 (the object's centreline depth).  This gives:
    #   - silhouette width directly in world coordinates
    #   - along-wall position from the bbox centre
    wall_plane_pt:  np.ndarray | None = None
    wall_plane_w:   float | None = None
    _silhouette_w_m: float | None = None
    if back_flush:
        if wall_affinity == "back":
            plane_axis, plane_val = 2, est_d / 2.0 + 0.01
        elif wall_affinity == "left":
            plane_axis, plane_val = 0, est_d / 2.0 + 0.01
        else:   # right
            plane_axis, plane_val = 0, room_w - est_d / 2.0 - 0.01

        def _bp_plane(px, py):
            ray = _backproject_pixel(
                px, py, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy
            )
            denom = ray[plane_axis]
            if abs(denom) < 1e-9:
                return None
            t = (plane_val - cam_pos[plane_axis]) / denom
            if t <= 0:
                return None
            return cam_pos + t * ray

        px_c = (float(bx1) + float(bx2)) / 2.0
        py_c = (float(by1) + float(by2)) / 2.0
        c_pt = _bp_plane(px_c, py_c)
        l_pt = _bp_plane(float(bx1), py_c)
        r_pt = _bp_plane(float(bx2), py_c)
        if c_pt is not None:
            wall_plane_pt = c_pt
        if l_pt is not None and r_pt is not None:
            wp_axis = 0 if wall_affinity == "back" else 2
            wall_plane_w = abs(float(r_pt[wp_axis] - l_pt[wp_axis]))
            _wp_clipped = float(np.clip(wall_plane_w, def_w * 0.3, def_w * 2.5))
            if _vlm_est_w is not None and _wp_clipped > _vlm_est_w * 1.3:
                _wp_clipped = _vlm_est_w * 1.3
                print(f"  [wall_plane] silhouette width @plane={plane_val:.2f}: "
                      f"raw={wall_plane_w:.2f}  capped to {_wp_clipped:.2f} "
                      f"(VLM est_w={_vlm_est_w:.2f} × 1.3)")
            else:
                print(f"  [wall_plane] silhouette width @plane={plane_val:.2f}: "
                      f"raw={wall_plane_w:.2f}  → est_w={_wp_clipped:.2f}")
            est_w = _wp_clipped
            _silhouette_w_m = _wp_clipped
        if wall_plane_pt is not None:
            print(f"  [wall_plane] centre @plane: "
                  f"{np.round(wall_plane_pt, 2).tolist()}")

    # ── Front direction resolution ───────────────────────────────────────
    # Priority: detilt_and_orient.py front_local_axis (VLM grid comparison
    # on OBB-aligned mesh, pre-computed) → live VLM detection → default [0,0,1].
    #
    # For canonical GLBs the front_local_axis is used directly (the OBB frame
    # IS the mesh frame).  For raw meshes we transform the front direction
    # from the OBB frame back to the raw mesh frame using detilt_R.
    _detilt_axis_map = {
        "-Z": np.array([0, 0, -1], dtype=np.float64),
        "+Z": np.array([0, 0,  1], dtype=np.float64),
        "+X": np.array([1, 0,  0], dtype=np.float64),
        "-X": np.array([-1, 0, 0], dtype=np.float64),
    }
    _dfa = entry.get("_detilt_front_axis", "")
    # Hunyuan3D convention: camera at +Z looking at -Z, so the object's
    # front faces +Z.  Detilt VLM consistently picks panel A (-Z) for seating
    # because the concave side (seat) looks like the "front" from -Z.
    # Override: for seating, flip -Z to +Z to match camera convention.
    _SEATING_DETILT_OVERRIDE = {"sofa", "couch", "loveseat", "chair",
                                "armchair", "stool", "office_chair"}
    if (_dfa == "-Z"
            and obj_type.lower().replace("-", "_") in _SEATING_DETILT_OVERRIDE):
        print(f"  [front] Hunyuan override: detilt said -Z, flipping to +Z for {obj_type}")
        _dfa = "+Z"
    # Per-segment yaw override: a `yaw_offset_deg` field in segment_results
    # for the matching segment is applied later as an extra rotation.  See
    # the apply block downstream.
    _use_detilt_front = use_vlm and _dfa and _dfa in _detilt_axis_map
    if _use_detilt_front:
        _front_obb = _detilt_axis_map[_dfa]
        if entry.get("is_canonical_glb"):
            # Canonical GLB: front is always at -Z after yaw baking
            local_front = _front_obb
            print(f"  [front] using detilt front_axis={_dfa} → {local_front.tolist()}")
        else:
            # Raw mesh: transform front from OBB frame to raw mesh frame.
            # Detilt applies: v_obb = (v_raw - centroid) @ R.T  (R.T maps raw→OBB)
            # Inverse for directions: d_raw = R.T @ d_obb  (column convention)
            _detilt_R = entry.get("_detilt_R")
            if _detilt_R is not None:
                _R_det = np.array(_detilt_R, dtype=np.float64)
                _front_raw = _R_det.T @ _front_obb
                # Project to XZ plane (Y-rotation only) and normalise
                _front_raw[1] = 0.0
                _fn = np.linalg.norm(_front_raw)
                if _fn > 1e-6:
                    local_front = _front_raw / _fn
                else:
                    local_front = _front_obb  # degenerate — fall back
                print(f"  [front] detilt front_axis={_dfa} → raw frame "
                      f"{np.round(local_front, 3).tolist()}")
            else:
                local_front = _front_obb
                print(f"  [front] using detilt front_axis={_dfa} → {local_front.tolist()}")
    else:
        # Without detilt results, the VLM vertex-cloud front detection is
        # unreliable — it consistently confuses front/back for seating
        # furniture because the vertex cloud looks similar from both sides.
        # Hunyuan3D reconstructs meshes with the camera along +Z,
        # so the object's front faces +Z.
        # Hunyuan3D seating: sofas have front at +Z (override flips detilt -Z → +Z).
        # Chairs without detilt: use -Z since the mesh visual front faces -Z
        # and vlm_orient will correct if wrong.
        _SEATING_FRONT_TYPES = {"sofa", "couch", "loveseat", "chair",
                                "armchair", "stool", "office_chair"}
        if obj_type.lower().replace("-", "_") in _SEATING_FRONT_TYPES and use_vlm:
            # Seating has NO consistent Hunyuan front axis — the old hardcoded
            # [0,0,+1] is right for some sofas but wrong for many chair meshes,
            # leaving the seat facing away from the room. Detect the front from
            # rendered cardinal views matched to the reference crop (reliable;
            # same path used for cabinets/tables) instead of guessing.
            mesh_vc = None
            try:
                if hasattr(mesh.visual, "to_color"):
                    mesh_vc = mesh.visual.to_color().vertex_colors
            except Exception:
                pass
            _front_crop = crop_path if (crop_path and crop_path.exists()) else None
            local_front = _vlm_detect_front_direction(
                glb_path, mesh.vertices, mesh_vc, obj_type,
                cache_suffix=f"_raw_{wall_affinity}",
                bbox_crop_path=_front_crop,
                wall_affinity=wall_affinity,
            )
            print(f"  [front] seating VLM-detected front = "
                  f"{np.round(local_front, 3).tolist()} for {obj_type}")
        elif obj_type.lower().replace("-", "_") in _SEATING_FRONT_TYPES:
            local_front = np.array([0, 0, 1], dtype=np.float64)
            print(f"  [front] no detilt / VLM disabled — Hunyuan default "
                  f"[0,0,+1] for {obj_type}")
        elif use_vlm:
            mesh_vc = None
            try:
                if hasattr(mesh.visual, "to_color"):
                    mesh_vc = mesh.visual.to_color().vertex_colors
            except Exception:
                pass
            _front_crop = crop_path if (crop_path and crop_path.exists()) else None
            local_front = _vlm_detect_front_direction(
                glb_path, mesh.vertices, mesh_vc, obj_type,
                cache_suffix=f"_raw_{wall_affinity}",
                bbox_crop_path=_front_crop,
                wall_affinity=wall_affinity,
            )
        else:
            local_front = np.array([0, 0, 1], dtype=np.float64)
            print(f"  [front] no detilt / VLM disabled — using Hunyuan default [0,0,+1] for {obj_type}")

    # wo_feedback_loop ablation: place the raw (un-canonicalized) mesh in its
    # NATIVE orientation — override any VLM/detilt front-detection with the
    # Hunyuan default (+Z). The object then only gets the geometric wall-affinity
    # rotation, so a mis-oriented raw mesh stays mis-oriented (the ablation's point).
    if __import__("os").environ.get("SCENEWEAVE_ABLATE_NO_FEEDBACK") == "1":
        local_front = np.array([0, 0, 1], dtype=np.float64)
        print(f"  [front] ABLATION wo_feedback_loop → raw mesh orientation (+Z) for {obj_type}")

    # ── Front-aware size measurement ─────────────────────────────────────
    # Once we know the front direction, we can measure width (cross-axis),
    # depth (front-back axis), and height from the raw mesh.
    # local_front is one of [0,0,1], [0,0,-1], [1,0,0], [-1,0,0].
    # Depth axis = local_front direction, Width axis = perpendicular on XZ.
    _cv = mesh.vertices.astype(np.float64) - mesh.vertices.mean(axis=0)
    _abs_front = np.abs(local_front)
    if _abs_front[2] > _abs_front[0]:
        # Front is along Z → depth=Z extent, width=X extent
        gw = float(np.ptp(_cv[:, 0])) or 1.0
        gd = float(np.ptp(_cv[:, 2])) or 1.0
    else:
        # Front is along X → depth=X extent, width=Z extent
        gw = float(np.ptp(_cv[:, 2])) or 1.0
        gd = float(np.ptp(_cv[:, 0])) or 1.0
    gh = float(np.ptp(_cv[:, 1])) or 1.0
    print(f"  [front_size] local_front={local_front.tolist()}  "
          f"raw extents w={gw:.3f} h={gh:.3f} d={gd:.3f}")

    # ── Swapped-axis detection ───────────────────────────────────────────
    # The mesh's front axis can be mislabelled: a wide desk generated from a
    # sparse mask often comes back narrow-and-deep, i.e. its WIDTH sits on the
    # axis we are calling depth.  Uniform scaling can never repair that — the
    # aspect ratio is frozen at generation time — so the only fix is to
    # relabel which mesh axis is the front.  wall_plane has already measured
    # the true along-wall width from the image silhouette (est_w), so compare
    # the mesh's footprint aspect against the target aspect and swap when the
    # mesh is clearly holding its long side on the depth axis.
    #
    # Relabelling local_front (rather than post-multiplying R by 90°) is what
    # keeps the front pointing INTO the room: _rotation_for_affinity maps
    # local_front onto the into-room normal whichever axis it names.  A blind
    # 90° on R would swing the front along the wall instead — that is why the
    # old wall_align correction was neutered rather than fixed.
    if (os.environ.get("SCENEWEAVE_AXIS_SWAP", "1") == "1"
            and _silhouette_w_m is not None
            and wall_affinity in ("back", "left", "right")
            and est_d > 1e-6 and gw > 1e-6 and gd > 1e-6):
        _mesh_ar   = gw / gd
        _target_ar = est_w / est_d
        _as_is     = abs(np.log(_mesh_ar)       - np.log(_target_ar))
        _swapped   = abs(np.log(1.0 / _mesh_ar) - np.log(_target_ar))
        _AR_MIN    = float(os.environ.get("SCENEWEAVE_AXIS_SWAP_AR",     "1.25"))
        _AR_MARGIN = float(os.environ.get("SCENEWEAVE_AXIS_SWAP_MARGIN", "0.35"))
        if (max(_mesh_ar, 1.0 / _mesh_ar) >= _AR_MIN
                and _swapped < _as_is - _AR_MARGIN):
            local_front = np.array(
                [float(local_front[2]), 0.0, -float(local_front[0])],
                dtype=np.float64)
            gw, gd = gd, gw
            print(f"  [axis_swap] mesh long axis was on the DEPTH axis "
                  f"(mesh w:d={_mesh_ar:.2f}, image silhouette wants "
                  f"{_target_ar:.2f}) → front relabelled to "
                  f"{local_front.tolist()}, w={gw:.3f} d={gd:.3f}")
        else:
            print(f"  [axis_swap] keeping axes (mesh w:d={_mesh_ar:.2f}, "
                  f"target {_target_ar:.2f}, as_is={_as_is:.2f} "
                  f"swapped={_swapped:.2f})")

    # ── Uniform scale ────────────────────────────────────────────────────
    uniform_scale = est_w / gw
    _h_cap = def_h * (1.8 if wall_affinity != "centre" else 1.5)
    _d_cap = def_d * (1.8 if wall_affinity != "centre" else 1.5)
    uniform_scale = min(uniform_scale, _h_cap / gh, _d_cap / gd)
    scale = np.array([uniform_scale, uniform_scale, uniform_scale], dtype=np.float64)
    eff_w = gw * uniform_scale
    eff_h = gh * uniform_scale
    eff_d = gd * uniform_scale
    print(f"  [scale] uniform={uniform_scale:.3f}  "
          f"effective W={eff_w:.2f} H={eff_h:.2f} D={eff_d:.2f} m")

    # ── Opening-based along-wall position ─────────────────────────────────────
    # Load floorplan_analysis.json to find windows/openings on the affinity wall
    # and snap the object to be centred under the nearest window to the
    # back-projected floor position.
    fp_path = furn_dir.parent / "floorplan_analysis.json"
    # ── Silhouette-aligned position ───────────────────────────────────────────
    # Use the back-projected floor contact point directly as the XZ position.
    # For wall-affinity objects, only snap the PERPENDICULAR axis (distance
    # from the wall) — the ALONG-WALL axis comes straight from back-projection
    # so the rendered object aligns with its silhouette in the source image.
    # Opening snap and VLM position overrides are intentionally omitted: they
    # shift objects away from where they appear in the reference image.
    # For flush wall-affinity objects that successfully back-projected to the
    # centreline plane, use that point's wall-parallel coordinate directly.
    # Otherwise fall back to floor_pt (bbox bottom-centre back-projection).
    src_pt = wall_plane_pt if wall_plane_pt is not None else floor_pt
    if wall_affinity == "back":
        pos_x = float(np.clip(src_pt[0], eff_w/2 + WALL_GAP, room_w - eff_w/2 - WALL_GAP))
        pos_z = eff_d / 2.0 + WALL_GAP
    elif wall_affinity == "left":
        pos_z = float(np.clip(src_pt[2], eff_w/2 + WALL_GAP, room_d - eff_w/2 - WALL_GAP))
        pos_x = eff_d / 2.0 + WALL_GAP
    elif wall_affinity == "right":
        pos_z = float(np.clip(src_pt[2], eff_w/2 + WALL_GAP, room_d - eff_w/2 - WALL_GAP))
        pos_x = room_w - eff_d / 2.0 - WALL_GAP
    else:
        pos_x = float(np.clip(floor_pt[0], eff_w/2 + WALL_GAP, room_w - eff_w/2 - WALL_GAP))
        pos_z = float(np.clip(floor_pt[2], eff_d/2 + WALL_GAP, room_d - eff_d/2 - WALL_GAP))

    # On-top-of: override XZ with the supporting object's centre so the item
    # sits centred on it rather than at the back-projected floor position.
    if support_xz is not None:
        pos_x, pos_z = float(support_xz[0]), float(support_xz[1])
        print(f"  [on_top] XZ overridden by support → ({pos_x:.3f}, {pos_z:.3f})")

    # ── Manual position override from placement_analysis.json ────────────────
    # Set "position_override_m": [x, z] on an entry to pin XZ regardless of
    # wall_affinity / bbox back-projection.  wall_affinity still controls
    # rotation and wall_snap flushing.
    _pos_ov = entry.get("position_override_m")
    _has_pos_override = False
    if _pos_ov is not None:
        try:
            _ov_x, _ov_z = float(_pos_ov[0]), float(_pos_ov[1])
            pos_x, pos_z = _ov_x, _ov_z
            _has_pos_override = True
            print(f"  [pos_override] XZ pinned to ({pos_x:.3f}, {pos_z:.3f})")
        except Exception as _e:
            print(f"  [pos_override] invalid position_override_m ({_pos_ov}): {_e}")

    # ── Safety clamp: keep *entire object* inside the scene box ──────────────
    # Account for object half-extents so the mesh doesn't poke outside the room.
    _hw = eff_w / 2.0   # half-width  (lateral extent)
    _hd = eff_d / 2.0   # half-depth  (front-back extent)
    pos_x = float(np.clip(pos_x, _hw, max(room_w - _hw, _hw)))
    pos_z = float(np.clip(pos_z, _hd, max(room_d - _hd, _hd)))

    print(f"  [pos] affinity={wall_affinity}  → ({pos_x:.3f}, {pos_z:.3f})  Y={support_y:.3f}")
    position_m = np.array([pos_x, support_y, pos_z])

    # ── Rotation from front detection ────────────────────────────────────
    # local_front was resolved above (detilt front_axis → VLM → default).
    # _rotation_for_affinity computes a Y-rotation so the object's front
    # faces into the room (away from its wall) or toward the camera.
    _rot_affinity = entry.get("_orient_affinity", wall_affinity)
    if not use_vlm:
        # No VLM: use raw Hunyuan orientation (identity — no wall-affinity rotation).
        R = np.eye(3, dtype=np.float64)
        print(f"  [rotation] VLM disabled — using identity rotation (raw Hunyuan frame)")
    elif _rot_affinity == "_face_toward" and "_orient_face_pos" in entry:
        # Face toward a specific object position
        _tgt = np.array(entry["_orient_face_pos"], dtype=np.float64)
        dx = _tgt[0] - position_m[0]
        dz = _tgt[2] - position_m[2]
        norm = np.sqrt(dx*dx + dz*dz)
        if norm > 1e-6:
            _face_target = np.array([dx/norm, 0.0, dz/norm], dtype=np.float64)
        else:
            _face_target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle_from = np.arctan2(float(local_front[0]), float(local_front[2]))
        angle_to   = np.arctan2(float(_face_target[0]), float(_face_target[2]))
        R = _rotation_y(angle_to - angle_from)
        print(f"  [rotation] facing toward object at {np.round(_tgt, 2).tolist()} "
              f"(target dir={np.round(_face_target, 2).tolist()})")
    elif _rot_affinity not in (wall_affinity, "_face_toward"):
        print(f"  [rotation] using orient_affinity={_rot_affinity} (from analysis) "
              f"instead of wall_affinity={wall_affinity}")
        R = _rotation_for_affinity(_rot_affinity, position_m, cam_pos, local_front=local_front)
    else:
        R = _rotation_for_affinity(wall_affinity, position_m, cam_pos, local_front=local_front)

    # ── Wall-alignment check ──────────────────────────────────────────────
    # For wall-affinity objects, verify the object's WIDTH axis (perpendicular
    # to its front direction) is parallel to the wall after rotation.
    # Using the width axis instead of the longest raw mesh axis avoids
    # breaking the front direction when the mesh's depth exceeds its width.
    if wall_affinity in ("back", "left", "right"):
        # Honest footprint test: measure the object's ACTUAL world-space XZ
        # extents after R and compare them against the wall direction.
        #
        # The previous implementation derived a "width axis" from
        # cross(up, local_front).  Because R is built by
        # _rotation_for_affinity precisely to map local_front onto the
        # into-room normal, that axis is parallel to the wall BY
        # CONSTRUCTION: the test printed par=1.00 perp=0.00 for every wall
        # object at every yaw it could actually be handed, so it could never
        # fire.  Measure the mesh instead.
        _wv     = _cv @ R.T
        _ext_x  = float(np.ptp(_wv[:, 0]))
        _ext_z  = float(np.ptp(_wv[:, 2]))
        if wall_affinity == "back":
            _par_ext, _perp_ext = _ext_x, _ext_z
        else:   # left / right walls run along Z
            _par_ext, _perp_ext = _ext_z, _ext_x
        _fp_ratio = _perp_ext / max(_par_ext, 1e-6)
        if _fp_ratio > 1.15:
            # Deliberately NOT rotating R here.  Post-multiplying by 90° would
            # swing the object's FRONT along the wall as well; the correct
            # repair is the axis relabel in [axis_swap] above, which preserves
            # the front direction.  Report loudly instead.
            print(f"  [wall_align] \u26a0 footprint still perpendicular to the "
                  f"{wall_affinity} wall: along-wall={_par_ext:.2f} "
                  f"off-wall={_perp_ext:.2f} (ratio {_fp_ratio:.2f}) — "
                  f"axis_swap did not fire, or the mesh is genuinely deep")
        else:
            print(f"  [wall_align] footprint parallel to {wall_affinity} wall "
                  f"\u2713 (along-wall={_par_ext:.2f} off-wall={_perp_ext:.2f})")

    # ── VLM rotation / scale check ────────────────────────────────────────────
    # crop_path and scene_img already resolved in the numerical estimation block above
    if use_vlm and crop_path and crop_path.exists():
        # Iterate the size critique to a fixed point instead of taking one shot.
        # One-shot left office8's desk at 0.90×1.46 m (a desk rotated 90° in
        # proportion) even though the VLM's own first answer was 1.93×0.65 —
        # the estimate was never fed back and re-checked.  Feed each answer back
        # in and stop when the proposal stops moving (<5% on every axis).
        _SIZE_TOL   = float(os.environ.get("SCENEWEAVE_SIZE_TOL", "0.05"))
        _SIZE_ITERS = int(os.environ.get("SCENEWEAVE_SIZE_ITERS", "4"))
        cur_w, cur_h, cur_d = est_w, est_h, est_d
        vlm = None
        for _sz_it in range(_SIZE_ITERS):
            vlm = _vlm_refine_placement(
                crop_path, obj_type, wall_affinity, cur_w, cur_h, cur_d,
                position_m=position_m, cam=cam, scene_image_path=scene_img,
            )
            nw = float(vlm.get("width_m",  cur_w))
            nh = float(vlm.get("height_m", cur_h))
            nd = float(vlm.get("depth_m",  cur_d))
            if "UNVERIFIED" in str(vlm.get("notes", "")):
                print(f"  [vlm_size] iter {_sz_it + 1}: unverified — stopping, "
                      f"keeping {cur_w:.2f}×{cur_h:.2f}×{cur_d:.2f}m")
                break
            rel = max(abs(nw - cur_w) / max(cur_w, 1e-6),
                      abs(nh - cur_h) / max(cur_h, 1e-6),
                      abs(nd - cur_d) / max(cur_d, 1e-6))
            print(f"  [vlm_size] iter {_sz_it + 1}: "
                  f"{cur_w:.2f}×{cur_h:.2f}×{cur_d:.2f} → {nw:.2f}×{nh:.2f}×{nd:.2f}m "
                  f"(max Δ {rel:.1%})")
            cur_w, cur_h, cur_d = nw, nh, nd
            if rel < _SIZE_TOL:
                print(f"  [vlm_size] converged after {_sz_it + 1} iteration(s)")
                entry["_vlm_size_converged"] = True
                entry["_vlm_size_m"] = [round(cur_w, 4), round(cur_h, 4), round(cur_d, 4)]
                break
        else:
            print(f"  [vlm_size] did NOT converge in {_SIZE_ITERS} iterations — "
                  f"using last proposal {cur_w:.2f}×{cur_h:.2f}×{cur_d:.2f}m")
        if vlm is None:
            vlm = {"rotation_deg": 0, "width_m": est_w, "height_m": est_h,
                   "depth_m": est_d, "notes": "no vlm"}
        vlm_w, vlm_h, vlm_d = cur_w, cur_h, cur_d
        print(f"  [vlm_refine] rot={vlm['rotation_deg']}°  "
              f"target={vlm_w:.2f}×{vlm_h:.2f}×{vlm_d:.2f}m  {vlm['notes']}")

        # Phase-1 rotation: VLM's suggested Y rotation only applies to
        # centre-affinity objects.  Wall-affinity items keep the
        # deterministic "back against wall" rotation regardless of what
        # the VLM thinks — text-only reasoning has consistently picked the
        # wrong cardinal flip for Hunyuan-reconstructed seating, and
        # ±90° would rotate the object off the wall entirely.
        #
        # All small-angle perspective tweaks (±25–45°) come from phase 2
        # (`_vlm_fine_rotation_correction`, the `[fine_rot]` log lines)
        # plus the post-placement pass (`[post_fine_rot]`), which sees
        # the full composited scene.  Both phases compare rendered
        # candidates against the reference image, so they're far more
        # reliable than vlm_refine's text-only cardinal pick for the
        # wall-affinity case.
        extra_deg = float(vlm.get("rotation_deg", 0))
        _xz_aspect = max(gw, gd) / max(min(gw, gd), 1e-6)
        # `_orient_affinity` (from photo analysis "back against the X wall")
        # also drives a deterministic rotation above (line ~1310-1315), so
        # the same VLM-rotation drop must apply when it's set.  Without
        # this, a centre-affinity chair with orient_affinity="left" gets:
        #   1. Rotation applied so chair faces +X (away from "virtual" left wall)
        #   2. VLM compares to GLB native +Z front, says "rotate 90° CW"
        #   3. The +90° CW gets composed on top of step 1 → chair over-rotated
        # Treat orient_affinity != "centre" the same as wall_affinity != "centre"
        # for the rotation-drop logic.
        _orient_aff = entry.get("_orient_affinity", "centre")
        _is_wall_aff = (wall_affinity != "centre") or (_orient_aff != "centre"
                                                       and _orient_aff
                                                       != "_face_toward")

        # Wall-affine items: trust the VLM's cardinal rotation pick.  The
        # deterministic pose assumes `local_front` (set by `front_detect`) is
        # correct, but for symmetric Hunyuan meshes (e.g. a square bed where
        # the actual foot is on a different axis from what View A shows)
        # `front_detect` can pick the wrong axis.  In that case `vlm_refine`
        # — which sees the rendered scene — is the authority that catches the
        # error.  We snap to the nearest cardinal so a noisy 87°/91° doesn't
        # rotate the object off the wall by a sub-cardinal amount, and
        # downstream `wall_snap` re-flushes the rotated bounds against the
        # nearest wall regardless.
        if _is_wall_aff and abs(extra_deg) > 1:
            _drop_reason = (f"wall_aff={wall_affinity}"
                            if wall_affinity != "centre"
                            else f"orient_aff={_orient_aff}")
            _snapped = round(extra_deg / 90.0) * 90.0
            # Types with a clean front/back — VLM's cardinal pick is
            # unreliable for these (e.g. sofa back-against-wall is
            # correctly set by R_for_affinity; VLM's 90° suggestion puts
            # the piece sideways).  Trust the deterministic rotation.
            _DETERMINISTIC_R_TYPES = {
                # Seating
                "sofa", "loveseat", "chair", "armchair", "bench", "ottoman",
                # Box furniture (front/back well defined)
                "cabinet", "dresser", "bookcase", "sideboard", "console",
                "wardrobe", "tv_stand", "shelf", "nightstand", "chest",
                "buffet", "credenza",
                # Table-like with clear wall orientation
                "desk", "dining_table",
                # Beds: headboard-against-wall is set deterministically;
                # vlm_refine's cardinal snap was rotating beds 90° sideways.
                "bed", "mattress", "bunk_bed",
            }
            # Wall affinity no longer vetoes the VLM. The deterministic pose
            # derives its "long axis" from the generated mesh's bounding box, so
            # a mis-proportioned mesh yields a confidently wrong pose that the
            # reflection loop was then forbidden to fix — office8's desk aligned
            # "parallel to the back wall ✓" while sitting 0.90 wide × 1.46 deep.
            # The VLM sees the rendered scene, so it is the better authority; we
            # still snap to the nearest cardinal and wall_snap re-flushes bounds.
            # Set SCENEWEAVE_WALL_DETERMINISTIC_ROT=1 to restore the old veto
            # (it was added because sofas/beds were being rotated sideways).
            _wall_det = os.environ.get("SCENEWEAVE_WALL_DETERMINISTIC_ROT", "0") == "1"
            if _wall_det and obj_type in (_DETERMINISTIC_R_TYPES - allow_rotation_types):
                print(f"  [vlm_rot] dropping {_snapped:+.0f}° for {obj_type} — "
                      f"deterministic wall-affinity rotation is authoritative "
                      f"({_drop_reason}) [SCENEWEAVE_WALL_DETERMINISTIC_ROT=1]")
                extra_deg = 0.0
            elif abs(_snapped) < 1:
                print(f"  [vlm_rot] dropped {extra_deg:+.0f}° — {_drop_reason}, "
                      f"snapped to 0° (small-angle tweak deferred to fine_rot)")
                extra_deg = 0.0
            else:
                _det_note = (" [was previously vetoed for this type]"
                             if obj_type in (_DETERMINISTIC_R_TYPES - allow_rotation_types)
                             else "")
                print(f"  [vlm_rot] keeping {_snapped:+.0f}° (VLM said "
                      f"{extra_deg:+.0f}°) — {_drop_reason}, snapped to "
                      f"nearest cardinal; corrects front-axis ambiguity in "
                      f"symmetric mesh (wall_snap will re-flush bounds)"
                      f"{_det_note}")
                extra_deg = _snapped

        if abs(extra_deg) > 1:
            R_extra = _rotation_y(np.radians(extra_deg))
            R = R_extra @ R
            print(f"  [vlm_rot] applied {extra_deg:+.0f}° "
                  f"(wall_aff={wall_affinity}, aspect={_xz_aspect:.2f})")

        # ── Uniform scale from VLM target dimensions ────────────────────────
        # VLM sets a reasonable initial scale; silhouette mask_align refines
        # it afterward.
        vlm_w = float(np.clip(vlm_w, def_w * 0.3, def_w * 2.5))
        vlm_h = float(np.clip(vlm_h, def_h * 0.3, def_h * 2.5))
        vlm_d = float(np.clip(vlm_d, def_d * 0.3, def_d * 2.5))

        cur_w = gw * float(scale[0]) or 1.0
        cur_h = gh * float(scale[1]) or 1.0
        cur_d = gd * float(scale[2]) or 1.0

        rw = vlm_w / cur_w
        rh = vlm_h / cur_h
        rd = vlm_d / cur_d
        uniform_factor = float(np.clip(np.cbrt(rw * rh * rd), 0.5, 2.0))
        scale *= uniform_factor
        print(f"  [vlm_scale] uniform factor={uniform_factor:.3f}  "
              f"scale → [{scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}]")
    else:
        print(f"  [vlm_refine] no crop found — skipping VLM check")

    # ── Wall-affinity cardinal snap ─────────────────────────────────────────
    # Guarantee that wall-affinity objects get a clean axis-aligned rotation.
    # Extract the effective yaw from R and snap to the nearest 90° increment.
    # This prevents floating-point drift or OBB imprecision from producing
    # tilted desks, sofas, etc. against walls.
    # Centre low tables (coffee/side/round/end) are conventionally parallel to
    # the room, so snap them to cardinal too — otherwise the front-detect can
    # leave them at an odd tilted angle (e.g. 154°) in a diagonal-view scene.
    _CARDINAL_CENTRE_TYPES = {"coffee_table", "side_table", "round_table",
                              "end_table", "console_table", "cocktail_table"}
    if (wall_affinity in ("back", "left", "right")
            or obj_type in _CARDINAL_CENTRE_TYPES):
        yaw = np.arctan2(float(R[0][2]), float(R[0][0]))
        snapped = round(yaw / (np.pi / 2)) * (np.pi / 2)
        if abs(yaw - snapped) > 1e-4:
            R = _rotation_y(snapped)
            print(f"  [cardinal_snap] yaw {np.degrees(yaw):.1f}° → {np.degrees(snapped):.1f}° "
                  f"(wall_affinity={wall_affinity}, type={obj_type})")

    # ── Centre low-table long-axis default ──────────────────────────────────
    # A coffee/cocktail/centre table almost always sits with its LONG edge
    # parallel to the sofa it fronts — i.e. parallel to the back wall (world X).
    # front-detect / the VLM size estimate frequently leave the long axis
    # pointing front-to-back (toward the camera). This is more reliable than
    # the VLM's text reasoning (which judges its own size estimate, not the
    # actual mesh OBB), so deterministically rotate the long axis to world X.
    if obj_type in _CARDINAL_CENTRE_TYPES and wall_affinity == "centre":
        _lx = float(bounds[1][0] - bounds[0][0])   # local X extent
        _lz = float(bounds[1][2] - bounds[0][2])   # local Z extent
        _R2 = np.array([[R[0][0], R[0][2]], [R[2][0], R[2][2]]])
        _wX = _R2 @ np.array([_lx, 0.0])           # world span of local X axis
        _wZ = _R2 @ np.array([0.0, _lz])           # world span of local Z axis
        _world_x_span = abs(_wX[0]) + abs(_wZ[0])
        _world_z_span = abs(_wX[1]) + abs(_wZ[1])
        if _world_z_span > _world_x_span + 0.10:
            R = _rotation_y(np.pi / 2) @ R
            print(f"  [table_long_axis] {obj_type} long axis was front-back "
                  f"(worldX={_world_x_span:.2f} worldZ={_world_z_span:.2f}) → "
                  f"rotated 90° to lie parallel to the back wall")

    # ── Manual per-segment yaw override (POST-snap) ──────────────────────────
    # Applied AFTER cardinal_snap so non-cardinal offsets (e.g. -45°) survive.
    # `yaw_offset_deg` is added to the segment_results entry; rotation_y(+)
    # is clockwise from above (+Y), so positive = CW, negative = CCW.
    _manual_yaw = entry.get("yaw_offset_deg")
    if _manual_yaw is not None:
        try:
            _manual_yaw = float(_manual_yaw)
        except (TypeError, ValueError):
            _manual_yaw = None
    if _manual_yaw is not None and abs(_manual_yaw) > 0.1:
        R_manual = _rotation_y(np.radians(_manual_yaw))
        R = R_manual @ R
        print(f"  [manual_yaw] applied {_manual_yaw:+.1f}° from segment_results "
              f"(post cardinal_snap)")

    # ── Manual per-segment ROLL override (around wall-normal axis) ──────────
    # When the generator's OBB swaps height↔width on a tall piece, the mesh ends up
    # wider than tall.  A `roll_offset_deg` field rotates the mesh around the
    # wall-normal axis so the long axis becomes vertical.  For back/front
    # walls the normal is world Z; for left/right walls it's world X.
    _manual_roll = entry.get("roll_offset_deg")
    if _manual_roll is not None:
        try:
            _manual_roll = float(_manual_roll)
        except (TypeError, ValueError):
            _manual_roll = None
    if _manual_roll is not None and abs(_manual_roll) > 0.1:
        _ar = float(np.radians(_manual_roll))
        _cr, _sr = float(np.cos(_ar)), float(np.sin(_ar))
        if wall_affinity in ("back", "front"):
            R_roll = np.array([[ _cr, -_sr, 0.0],
                               [ _sr,  _cr, 0.0],
                               [ 0.0,  0.0, 1.0]], dtype=np.float64)
            _axis_label = "Z (back/front normal)"
        else:
            R_roll = np.array([[1.0, 0.0, 0.0],
                               [0.0, _cr, -_sr],
                               [0.0, _sr,  _cr]], dtype=np.float64)
            _axis_label = "X (left/right normal)"
        R = R_roll @ R
        print(f"  [manual_roll] applied {_manual_roll:+.1f}° around "
              f"wall-normal axis {_axis_label}")

    # Compute preliminary effective dims from the rotated bounding box.
    # After rotation R, the along-wall and into-room axes depend on which
    # cardinal rotation was applied.  Compute the rotated corner extents
    # directly for accuracy.
    _s0 = float(scale[0])
    _corners_raw = np.array([
        [bounds[0][0], bounds[0][2]],  # min X, min Z
        [bounds[0][0], bounds[1][2]],  # min X, max Z
        [bounds[1][0], bounds[0][2]],  # max X, min Z
        [bounds[1][0], bounds[1][2]],  # max X, max Z
    ])
    _R2d = np.array([[R[0][0], R[0][2]], [R[2][0], R[2][2]]])  # 2D XZ rotation
    _corners_rot = _corners_raw @ _R2d.T
    _world_x_ext = float(np.ptp(_corners_rot[:, 0]))
    _world_z_ext = float(np.ptp(_corners_rot[:, 1]))
    if wall_affinity == "back":
        # back wall runs along X → width=X extent, depth=Z extent
        eff_w = _world_x_ext * _s0
        eff_d = _world_z_ext * _s0
    else:
        # left/right wall runs along Z → width=Z extent, depth=X extent
        eff_w = _world_z_ext * _s0
        eff_d = _world_x_ext * _s0
    eff_h = float(bounds[1][1] - bounds[0][1]) * _s0

    # ── Manual width override (`target_width_m`) ────────────────────────────
    # Hard-pins the world-space width when VLM/silhouette estimation
    # produces a piece that's visibly too wide (or narrow).  Applies a
    # uniform factor on top of all earlier scaling so eff_w matches the
    # target; height and depth scale proportionally.
    _target_w = entry.get("target_width_m")
    if _target_w is not None:
        try:
            _target_w = float(_target_w)
        except (TypeError, ValueError):
            _target_w = None
    if _target_w is not None and eff_w > 1e-6 and abs(eff_w - _target_w) > 0.02:
        _w_factor = float(_target_w) / eff_w
        scale *= _w_factor
        eff_w *= _w_factor
        eff_h *= _w_factor
        eff_d *= _w_factor
        print(f"  [target_width] {obj_type} width override: "
              f"eff_w → {eff_w:.2f} m (×{_w_factor:.2f}, "
              f"H→{eff_h:.2f} D→{eff_d:.2f})")

    # ── Min-depth floor for box-shaped furniture ────────────────────────────
    # the 3D generator often collapses depth on box-shaped meshes (cabinets, dressers,
    # bookcases, sideboards): the resulting mesh has paper-thin Z extent
    # even though the real piece is 30–50 cm deep.  When the post-rotation
    # effective world depth is below a per-type floor, stretch the
    # mesh-local depth axis up to that floor.
    _MIN_DEPTH_BY_TYPE = {
        "cabinet": 0.40, "dresser": 0.40, "bookcase": 0.30,
        "tv_stand": 0.40, "sideboard": 0.40, "console": 0.30,
        "wardrobe": 0.55, "shelf": 0.30, "nightstand": 0.35,
    }
    _depth_floor = _MIN_DEPTH_BY_TYPE.get(obj_type)
    if _depth_floor is not None and eff_d < _depth_floor and wall_affinity in ("back", "front", "left", "right"):
        if wall_affinity in ("back", "front"):
            _depth_local = 2 if abs(R[2][2]) > abs(R[2][0]) else 0   # world Z
        else:
            _depth_local = 0 if abs(R[0][0]) > abs(R[0][2]) else 2   # world X
        _depth_factor = _depth_floor / eff_d
        scale[_depth_local] *= _depth_factor
        print(f"  [depth_floor] {obj_type}: eff_d {eff_d:.3f} → "
              f"{_depth_floor:.3f} m (×{_depth_factor:.2f}, "
              f"local axis {'XYZ'[_depth_local]})")
        eff_d = _depth_floor

    # ── CHANGE 1: Orthographic initial facing (env flag SCENEWEAVE_ORTHO_INIT) ──
    # Snap the per-object initial world yaw to the nearest cardinal (0/90/180/270°)
    # so the object starts axis-aligned to the world, BEFORE silhouette mask_align
    # and BEFORE the VGGT depth refine.  Default behaviour (flag unset) is unchanged.
    if os.environ.get("SCENEWEAVE_ORTHO_INIT"):
        _front_w = R @ np.array([0.0, 0.0, 1.0])
        _yaw_cur = float(np.arctan2(_front_w[0], _front_w[2]))
        _yaw_snap = float(np.round(_yaw_cur / (np.pi / 2.0)) * (np.pi / 2.0))
        _yaw_delta = _yaw_snap - _yaw_cur
        if abs(((_yaw_delta + np.pi) % (2 * np.pi)) - np.pi) > 1e-4:
            R = _rotation_y(_yaw_delta) @ R
        _snap_deg = int(round(np.degrees(_yaw_snap))) % 360
        print(f"  [ortho_init] idx={entry.get('index')} yaw snapped to {_snap_deg}°")

    result = {
        "index":        entry["index"],
        "type":         obj_type,
        "glb_path":     str(glb_path),
        "position_m":   position_m.tolist(),
        "rotation_3x3": R.tolist(),
        "scale":        scale.tolist(),
        "size_m":       {"width_m": eff_w, "height_m": eff_h, "depth_m": eff_d},
        "eff_h":        eff_h,
        "_silhouette_w_m": _silhouette_w_m,
        "wall_affinity": wall_affinity,
        "_pos_override": _has_pos_override,
        "on_top_of":     entry.get("on_top_of"),
        "_local_front":  local_front.tolist(),
        "skip_ground_removal": is_canonical or (wall_affinity != "centre"),
        "flat_on_floor": bool(entry.get("flat_on_floor", False)),
        # 2-D XZ convex hull of mesh vertices in local space — used for
        # accurate footprint collision detection instead of AABB.
        "_xz_hull_pts":  _compute_xz_hull(mesh.vertices),
        # Deferred mask_fit data — used by main loop after position is final,
        # then stripped before saving to JSON.
        "_mask_fit_data": {
            "mesh_verts": mesh.vertices,
            "mesh_bounds": bounds,
            "bbox": (bx1, by1, bx2, by2),
            "cam_pos": cam_pos,
            "right_v": right_v,
            "up_c_v": up_c_v,
            "fwd_v": fwd_v,
            "fx": fx, "cx": cx, "cy": cy,
            "back_flush": back_flush,
            "def_w": def_w, "def_h": def_h, "def_d": def_d,
            "room_h_ceil": room_h_ceil, "room_w": room_w, "room_d": room_d,
        },
    }
    return result


# ── Carpet (flat textured quad) placement ────────────────────────────────────

def _vlm_estimate_carpet_placement(
    scene_image_path: Path,
    crop_path: Path | None,
    room_w: float,
    room_d: float,
    ceiling_h: float,
) -> dict:
    """Ask VLM to estimate carpet position, rotation, and size from the scene.

    The room coordinate system:
      X: 0 = left wall, {room_w} = right wall
      Z: 0 = back wall, {room_d} = front (camera side)
      The camera looks from front towards the back wall.

    Returns dict with keys:
      centre_x_m, centre_z_m  — carpet centre position on the floor
      width_m, depth_m        — carpet dimensions (width along its long edge)
      rotation_deg            — rotation in degrees, 0 = aligned with walls,
                                positive = counter-clockwise from above
      confidence              — 0–1
      notes                   — brief explanation
    Falls back to empty dict on failure.
    """
    import re, requests

    prompt = f"""You are estimating the **real-world placement** of a carpet/rug on the floor in a room photo.

Room coordinate system (bird's-eye view):
  X axis: 0 = left wall, {room_w:.1f} = right wall  (room is {room_w:.1f} m wide)
  Z axis: 0 = back wall, {room_d:.1f} = front/camera side  (room is {room_d:.1f} m deep)
  Ceiling height = {ceiling_h:.1f} m.
  The camera is near the front wall looking towards the back wall.

**Your task — look at the room photo and estimate:**

1. **Position**: Where is the centre of the carpet on the floor?
   - centre_x_m: distance from the left wall (0 to {room_w:.1f})
   - centre_z_m: distance from the back wall (0 to {room_d:.1f})
   Think about which corner or area of the room the carpet is in.

2. **Size**: How big is the carpet?
   - width_m: the carpet's shorter dimension (typically 1.5–3.0 m)
   - depth_m: the carpet's longer dimension (typically 2.0–4.0 m)
   Use the wall height ({ceiling_h:.1f} m) and room width ({room_w:.1f} m) as scale references.

3. **Rotation**: Is the carpet axis-aligned with the walls, or rotated?
   - rotation_deg: 0 = edges parallel to walls, positive = counter-clockwise
   - Most carpets are axis-aligned (0°), but some are placed at an angle (e.g. 45°).
   - If the carpet's long edge runs along the room's depth (Z axis), that is 90°.

Image 1: the full room scene — study the carpet's position, size, and angle.
{('Image 2: a cropped view of just the carpet.' if crop_path else '')}

Respond with ONLY a JSON object, no markdown:
{{"centre_x_m": 0.0, "centre_z_m": 0.0, "width_m": 0.0, "depth_m": 0.0, "rotation_deg": 0.0, "confidence": 0.0, "notes": "brief explanation"}}"""

    content: list[dict] = [{"type": "text", "text": prompt}]
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(scene_image_path)}"}})
    if crop_path and crop_path.exists():
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_encode_image(crop_path)}"}})

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    try:
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group())
            print(f"  [vlm_carpet] pos=({result.get('centre_x_m')}, {result.get('centre_z_m')})  "
                  f"size={result.get('width_m')}×{result.get('depth_m')}m  "
                  f"rot={result.get('rotation_deg')}°  "
                  f"conf={result.get('confidence')}  "
                  f"notes={result.get('notes','')}")
            return result
    except Exception as e:
        print(f"  [vlm_carpet] VLM call failed: {e}")
    return {}


def compute_carpet_placement(
    entry: dict,
    cam: dict,
    carpet_img_path: Path,
    output_dir: Path | None = None,
) -> dict | None:
    """Place carpet using VLM for position/rotation/size, refined by mask.

    1. VLM reasons about where the carpet is, how big it is, and whether
       it's rotated relative to the walls.
    2. Mask back-projection provides a lower-bound size refinement.
    3. Build a uniform rectangle at the decided position, size, and rotation.

    Returns a placement dict with 'is_carpet': True and 'world_corners'.
    """
    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)

    W    = int(cam["width_px"])
    H    = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    bx1, by1, bx2, by2 = [float(v) for v in entry["box_px"]]

    # ── Mask-projected position & size ───────────────────────────────────────
    # Use the actual segmentation mask pixel extents (tighter than bbox) when
    # available.  This gives a much better silhouette match.
    mx1, my1, mx2, my2 = bx1, by1, bx2, by2   # default to bbox
    mask_fill: float | None = None             # mask area / bbox area
    if output_dir is not None:
        from PIL import Image as _PILImage
        mask_file = entry.get("mask_file", "")
        if mask_file:
            mask_path = output_dir / "furniture" / "segmented" / mask_file
            if mask_path.exists():
                try:
                    _marr = np.array(_PILImage.open(mask_path))
                    if _marr.ndim == 3:
                        _fg = _marr.sum(axis=2) > 0
                    else:
                        _fg = _marr > 0
                    _ys, _xs = np.where(_fg)
                    if len(_xs) > 0:
                        mx1, my1 = float(_xs.min()), float(_ys.min())
                        mx2, my2 = float(_xs.max()), float(_ys.max())
                        # mask_fill is a resolution-invariant ratio — compute it
                        # from the NATIVE mask before rescaling the extents below.
                        _bbox_area = max(1.0, (mx2 - mx1 + 1) * (my2 - my1 + 1))
                        mask_fill = float(_fg.sum()) / _bbox_area
                        # The mask is stored at the NATIVE photo resolution; the
                        # camera intrinsics (fx/cx/cy) are at the capped render
                        # resolution (W,H). Rescale the mask pixel extents into
                        # (W,H) so the back-projection below uses one pixel space.
                        _mh, _mw = _fg.shape[:2]
                        if _mw > 0 and _mh > 0 and (_mw, _mh) != (W, H):
                            _sx, _sy = W / float(_mw), H / float(_mh)
                            mx1, mx2 = mx1 * _sx, mx2 * _sx
                            my1, my2 = my1 * _sy, my2 * _sy
                        print(f"  [carpet] mask extent: [{mx1:.0f},{my1:.0f},{mx2:.0f},{my2:.0f}]  "
                              f"(bbox was [{bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f}])  "
                              f"fill={mask_fill:.2f} "
                              f"({'round-ish' if mask_fill < 0.85 else 'rect-ish'})")
                except Exception:
                    pass

    ray_c = _backproject_pixel(
        (mx1+mx2)/2, (my1+my2)/2, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy
    )
    mask_centre = _ray_floor_intersect(cam_pos, ray_c)
    if mask_centre is None:
        print(f"  [carpet] floor intersection failed — skipping")
        return None

    # Back-project the 4 mask corners to the floor to get the world-space quad
    # directly from the silhouette.  This handles perspective foreshortening
    # correctly (near edge is wider than far edge on the floor).
    corner_pixels = [
        (mx1, my1), (mx2, my1),  # top-left, top-right
        (mx2, my2), (mx1, my2),  # bottom-right, bottom-left
    ]
    floor_corners = []
    for cpx, cpy in corner_pixels:
        ray = _backproject_pixel(cpx, cpy, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
        fp = _ray_floor_intersect(cam_pos, ray)
        if fp is not None:
            floor_corners.append(fp)

    mask_w, mask_d = None, None
    if len(floor_corners) == 4:
        # Width: average of top and bottom edge lengths
        top_w = float(np.linalg.norm(floor_corners[1] - floor_corners[0]))
        bot_w = float(np.linalg.norm(floor_corners[2] - floor_corners[3]))
        mask_w = (top_w + bot_w) / 2.0
        # Depth: average of left and right edge lengths
        left_d = float(np.linalg.norm(floor_corners[3] - floor_corners[0]))
        right_d = float(np.linalg.norm(floor_corners[2] - floor_corners[1]))
        mask_d = (left_d + right_d) / 2.0
        print(f"  [carpet] mask-projected: centre=({mask_centre[0]:.2f}, {mask_centre[2]:.2f})  "
              f"size={mask_w:.2f}×{mask_d:.2f} m  (from 4-corner backprojection)")
    else:
        # Fallback: use mid-row left/right and mid-col top/bottom
        py_mid = (my1 + my2) / 2.0
        ray_ml = _backproject_pixel(mx1, py_mid, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
        ray_mr = _backproject_pixel(mx2, py_mid, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
        fp_ml = _ray_floor_intersect(cam_pos, ray_ml)
        fp_mr = _ray_floor_intersect(cam_pos, ray_mr)
        px_mid = (mx1 + mx2) / 2.0
        ray_mt = _backproject_pixel(px_mid, my1, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
        ray_mb = _backproject_pixel(px_mid, my2, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
        fp_mt = _ray_floor_intersect(cam_pos, ray_mt)
        fp_mb = _ray_floor_intersect(cam_pos, ray_mb)
        if fp_ml is not None and fp_mr is not None:
            mask_w = float(np.linalg.norm(fp_mr - fp_ml))
            if fp_mt is not None and fp_mb is not None:
                mask_d = float(np.linalg.norm(fp_mb - fp_mt))
            else:
                bbox_w_px = mx2 - mx1
                bbox_h_px = my2 - my1
                aspect = bbox_h_px / max(bbox_w_px, 1)
                mask_d = mask_w * aspect
            print(f"  [carpet] mask-projected: centre=({mask_centre[0]:.2f}, {mask_centre[2]:.2f})  "
                  f"size={mask_w:.2f}×{mask_d:.2f} m")

    # ── Room dimensions ──────────────────────────────────────────────────────
    room_w, room_d = _get_room_dims(carpet_img_path, cam)
    ceiling_h = 2.7
    if output_dir is not None:
        fp_path = output_dir / "floorplan_analysis.json"
        if fp_path.exists():
            try:
                with open(fp_path) as _fp:
                    ceiling_h = float(json.load(_fp).get("room", {}).get("ceiling_height_m", 2.7))
            except Exception:
                pass

    # ── VLM: estimate position, rotation, and size ───────────────────────────
    scene_img = None
    crop_path = None
    if output_dir is not None:
        for cand in [output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
                     output_dir / "render_final.png",
                     output_dir / "render.png"]:
            if cand.exists():
                scene_img = cand
                break
        if entry.get("crop_file"):
            cp = output_dir / "furniture" / "segmented" / entry["crop_file"]
            if cp.exists():
                crop_path = cp

    vlm = {}
    if scene_img is not None:
        vlm = _vlm_estimate_carpet_placement(scene_img, crop_path, room_w, room_d, ceiling_h)

    # ── Decide final position ────────────────────────────────────────────────
    # Mask back-projection is more reliable than VLM for position because it
    # derives from actual pixel coordinates.  VLM position estimates are
    # stochastic and can be far off, causing mask_align to diverge.
    cx_w = float(mask_centre[0])
    cz_w = float(mask_centre[2])
    print(f"  [carpet] position from mask: ({cx_w:.2f}, {cz_w:.2f})")
    if vlm.get("centre_x_m") is not None and vlm.get("centre_z_m") is not None:
        print(f"  [carpet] (VLM suggested ({vlm['centre_x_m']:.2f}, {vlm['centre_z_m']:.2f}) — ignored)")

    # ── Decide final rotation ────────────────────────────────────────────────
    rot_deg = float(vlm.get("rotation_deg", 0.0))
    rot_rad = np.radians(rot_deg)
    # Base edge directions: X-axis (width) and Z-axis (depth)
    edge_dir = np.array([np.cos(rot_rad), 0.0, -np.sin(rot_rad)], dtype=np.float64)
    perp_dir = np.array([np.sin(rot_rad), 0.0,  np.cos(rot_rad)], dtype=np.float64)
    if abs(rot_deg) > 0.5:
        print(f"  [carpet] rotation from VLM: {rot_deg:.1f}°")

    # ── Decide final size ────────────────────────────────────────────────────
    # The segmented mask, back-projected to the floor, gives the most accurate
    # carpet shape and size — it already accounts for perspective foreshortening
    # and preserves the correct aspect ratio.  Use mask dimensions as the
    # primary authority; VLM is only a fallback when mask back-projection fails.
    # Mask back-projection gives the most accurate size — it accounts for
    # perspective foreshortening via ray-floor intersection.  Per-axis 2D
    # pixel scaling fails for floor objects because Z-depth compresses to
    # very few pixels.  VLM is only a fallback when back-projection fails.
    vlm_w = float(vlm["width_m"]) if vlm.get("width_m") else None
    vlm_d = float(vlm["depth_m"]) if vlm.get("depth_m") else None
    if mask_w is not None and mask_d is not None:
        final_w, final_d = mask_w, mask_d
        print(f"  [carpet] initial size from mask back-projection: {final_w:.2f} × {final_d:.2f} m")
    elif vlm_w is not None and vlm_d is not None:
        final_w = max(0.5, min(room_w * 0.95, vlm_w))
        final_d = max(0.5, min(room_d * 0.95, vlm_d))
        print(f"  [carpet] initial size from VLM (fallback): {final_w:.2f} × {final_d:.2f} m")
    else:
        print(f"  [carpet] no size estimate — skipping")
        return None

    # ── Extend size when mask is clipped at image border ───────────────────
    # If the mask touches the image border, the carpet extends beyond the
    # visible frame.  Mirror the visible half to estimate the full extent,
    # capped at 90% of the room dimension.
    _border_px = max(40, int(min(W, H) * 0.10))  # ~10% of image short edge

    # ── Round + bottom-clipped: anchor far edge to visible top ─────────────
    # When the carpet's mask is round-ish AND its bottom edge runs off-screen,
    # the visible silhouette is roughly the top half of a circle.  The TOP edge
    # is fully visible so we can back-project it directly; the bottom edge is
    # truncated by the image, so mask_d is meaningless.  Rebuild the world
    # quad as a square sized by the un-clipped horizontal extent (mask_w),
    # with its FAR edge anchored to the back-projection of the visible top
    # silhouette point.  The square then extends toward the camera, off-screen
    # if necessary — that's correct because the bottom of the rug really does
    # extend past the image edge in the source photo.
    bottom_clipped = mask_d is not None and my2 > H - _border_px
    # "Round" means the silhouette fill is close to a circle's π/4 ≈ 0.785 (a
    # round/oval rug).  The old test (fill < 0.85) also caught LOW fills, but a
    # rug that is mostly occluded by the furniture standing on it (sofas, coffee
    # table) reads a low fill (≈0.5-0.6) while actually being RECTANGULAR.
    # Squaring such a rug doubles its depth to cover the whole floor
    # (living_room9: 2.44→4.67 m).  Require a genuine round-ish fill band so
    # occluded rectangular rugs keep their back-projected rectangle.
    is_round = mask_fill is not None and 0.65 <= mask_fill <= 0.88
    if is_round and bottom_clipped and mask_w is not None:
        far_ray = _backproject_pixel(
            (mx1 + mx2) / 2.0, my1,
            cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
        far_pt = _ray_floor_intersect(cam_pos, far_ray)
        if far_pt is not None:
            new_side = float(min(mask_w, room_w * 0.95, room_d * 0.95))
            new_cx = float(far_pt[0])
            new_cz = float(far_pt[2] + new_side / 2.0)
            print(f"  [carpet] round + bottom-clipped → anchor far edge at "
                  f"z={far_pt[2]:.2f}; square side={new_side:.2f}m "
                  f"(was {final_w:.2f}×{final_d:.2f}); centre "
                  f"({cx_w:.2f},{cz_w:.2f}) → ({new_cx:.2f},{new_cz:.2f})")
            final_w = final_d = new_side
            cx_w, cz_w = new_cx, new_cz
    if mask_w is not None:
        mask_cx_px = (mx1 + mx2) / 2.0
        if mx1 < _border_px and mx2 > _border_px:
            # Left edge clipped — extend: mirror the right half
            visible_right_half = mask_w / 2.0   # what we measured
            # The centre may also be shifted; use the visible right extent
            # mirrored to estimate the full width
            estimated_w = visible_right_half * 2.0 * 1.1  # +10% for hidden part
            if estimated_w > final_w:
                print(f"  [carpet] left edge clipped (mx1={mx1:.0f}) — "
                      f"extending width {final_w:.2f} → {min(estimated_w, room_w*0.95):.2f}")
                final_w = min(estimated_w, room_w * 0.95)
        if mx2 > W - _border_px and mx1 < W - _border_px:
            # Right edge clipped
            estimated_w = (mask_w / 2.0) * 2.0 * 1.1
            if estimated_w > final_w:
                print(f"  [carpet] right edge clipped (mx2={mx2:.0f}) — "
                      f"extending width {final_w:.2f} → {min(estimated_w, room_w*0.95):.2f}")
                final_w = min(estimated_w, room_w * 0.95)
    if mask_d is not None:
        if my1 < _border_px and my2 > _border_px:
            estimated_d = (mask_d / 2.0) * 2.0 * 1.1
            if estimated_d > final_d:
                print(f"  [carpet] top edge clipped (my1={my1:.0f}) — "
                      f"extending depth {final_d:.2f} → {min(estimated_d, room_d*0.95):.2f}")
                final_d = min(estimated_d, room_d * 0.95)
        if my2 > H - _border_px and my1 < H - _border_px:
            estimated_d = (mask_d / 2.0) * 2.0 * 1.1
            if estimated_d > final_d:
                print(f"  [carpet] bottom edge clipped (my2={my2:.0f}) — "
                      f"extending depth {final_d:.2f} → {min(estimated_d, room_d*0.95):.2f}")
                final_d = min(estimated_d, room_d * 0.95)

    # ── Build world-space quad ───────────────────────────────────────────────
    # When all four mask-corner back-projections succeeded AND the mask is
    # not clipped at the image border, use the back-projected trapezoid
    # directly — its screen projection matches the photo silhouette exactly.
    # Falls back to the axis-aligned rectangle when mask is clipped or
    # back-projection failed.
    _mask_clipped = (
        mx1 < _border_px or mx2 > W - _border_px
        or my1 < _border_px or my2 > H - _border_px
    )
    if len(floor_corners) == 4 and not _mask_clipped:
        # floor_corners pixel order: [TL_px, TR_px, BR_px, BL_px]
        # carpet world_corners order: [BL, BR, TR, TL] (Y=0 floor plane).
        world_corners = [
            floor_corners[3].tolist(),  # BL ← back-projected BL pixel
            floor_corners[2].tolist(),  # BR ← back-projected BR pixel
            floor_corners[1].tolist(),  # TR ← back-projected TR pixel
            floor_corners[0].tolist(),  # TL ← back-projected TL pixel
        ]
        _ctr = np.mean(np.array(world_corners), axis=0)
        cx_w, cz_w = float(_ctr[0]), float(_ctr[2])
        print(f"  [carpet] silhouette-matched quad from mask back-projection  "
              f"centre=({cx_w:.2f}, {cz_w:.2f})")
    else:
        # mask_align will refine position + scale iteratively afterward.
        half_w = final_w / 2.0
        half_d = final_d / 2.0
        c3 = np.array([cx_w, 0.0, cz_w])
        world_corners = [
            (c3 - half_w * edge_dir - half_d * perp_dir).tolist(),  # BL
            (c3 + half_w * edge_dir - half_d * perp_dir).tolist(),  # BR
            (c3 + half_w * edge_dir + half_d * perp_dir).tolist(),  # TR
            (c3 - half_w * edge_dir + half_d * perp_dir).tolist(),  # TL
        ]
        print(f"  [carpet] axis-aligned quad (mask clipped or back-projection "
              f"failed)  centre=({cx_w:.2f}, {cz_w:.2f})  "
              f"size={final_w:.2f}×{final_d:.2f} m")
    return {
        "index":         entry["index"],
        "type":          "carpet",
        "is_carpet":     True,
        "world_corners": world_corners,   # [BL, BR, TR, TL] on Y=0
        "carpet_img":    str(carpet_img_path),
        "position_m":    [cx_w, 0.0, cz_w],
        "_mask_bbox":    (mx1, my1, mx2, my2),  # for mask_align refinement
    }


def _carpet_mask_align(
    cp: dict,
    cam: dict,
    passes: int = 4,
) -> dict:
    """Iteratively adjust the carpet quad position so its 2D projection centre
    matches the mask bbox centre.

    Only adjusts POSITION — carpet size comes from mask back-projection which
    correctly handles perspective foreshortening via ray-floor intersection.
    Per-axis 2D pixel scaling is wrong for floor objects (Z-depth compresses
    to very few pixels in perspective, crushing the carpet).

    Returns the updated carpet placement dict.
    """
    mask_bbox = cp.get("_mask_bbox")
    if mask_bbox is None:
        return cp

    mx1, my1, mx2, my2 = mask_bbox
    target_cx = (mx1 + mx2) / 2.0
    target_cy = (my1 + my2) / 2.0
    target_w = float(mx2 - mx1)
    target_h = float(my2 - my1)
    if target_w < 2 or target_h < 2:
        return cp

    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)

    W    = int(cam["width_px"])
    H    = int(cam["height_px"])
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx_px, cy_px = W / 2.0, H / 2.0

    room_w, room_d = _get_room_dims(Path(cp.get("carpet_img", ".")), cam)

    for _pass in range(passes):
        corners = [np.array(c, dtype=np.float64) for c in cp["world_corners"]]

        # Project corners to 2D
        pxs, pys = [], []
        for c in corners:
            px, py, zc = _project_vertex(c, cam_pos, right_v, up_c_v, fwd_v, fx, cx_px, cy_px)
            if zc > NEAR_CLIP:
                pxs.append(px)
                pys.append(py)

        if len(pxs) < 3:
            break

        proj_cx = (min(pxs) + max(pxs)) / 2.0
        proj_cy = (min(pys) + max(pys)) / 2.0
        proj_w  = max(pxs) - min(pxs)
        proj_h  = max(pys) - min(pys)

        # ── Position shift (ray-floor back-projection) ─────────────────
        # Back-project projected and target pixel centres to the floor plane
        # to get accurate XZ correction without perspective coupling.
        ray_proj = _backproject_pixel(
            proj_cx, proj_cy, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx_px, cy_px)
        ray_tgt = _backproject_pixel(
            target_cx, target_cy, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx_px, cy_px)
        floor_proj = _ray_floor_intersect(cam_pos, ray_proj)
        floor_tgt = _ray_floor_intersect(cam_pos, ray_tgt)

        if floor_proj is not None and floor_tgt is not None:
            shift_x = float(floor_tgt[0] - floor_proj[0])
            shift_z = float(floor_tgt[2] - floor_proj[2])
        else:
            # Fallback: camera-axis shift
            dpx = target_cx - proj_cx
            dpy = target_cy - proj_cy
            centre = np.mean(corners, axis=0)
            depth = float(np.dot(centre - cam_pos, fwd_v))
            if depth < 0.1:
                depth = 1.0
            dx_world = (dpx / fx) * depth
            dz_world = -(dpy / fx) * depth
            shift_3d = dx_world * right_v - dz_world * up_c_v
            shift_x = float(shift_3d[0])
            shift_z = float(shift_3d[2])

        # Cap per-pass shift to 40cm — carpet alignment is important but
        # shouldn't overshoot in a single pass.
        _MAX_CARPET_SHIFT = 0.40
        shift_x = float(np.clip(shift_x, -_MAX_CARPET_SHIFT, _MAX_CARPET_SHIFT))
        shift_z = float(np.clip(shift_z, -_MAX_CARPET_SHIFT, _MAX_CARPET_SHIFT))

        if abs(shift_x) > 0.005 or abs(shift_z) > 0.005:
            # Cap the shift uniformly so the worst-positioned corner stays
            # inside the room. Clipping each corner independently (the
            # previous behaviour) distorts the carpet into a trapezoid when
            # any single corner would land outside the bounds.
            _xs = [float(corners[i][0]) for i in range(4)]
            _zs = [float(corners[i][2]) for i in range(4)]
            _max_pos_x = max(0.0, room_w - max(_xs))    # how far we can shift +X
            _max_neg_x = max(0.0, min(_xs))             # how far we can shift -X
            _max_pos_z = max(0.0, room_d - max(_zs))    # how far we can shift +Z
            _max_neg_z = max(0.0, min(_zs))             # how far we can shift -Z
            _capped_shift_x = float(np.clip(shift_x, -_max_neg_x, _max_pos_x))
            _capped_shift_z = float(np.clip(shift_z, -_max_neg_z, _max_pos_z))
            if (abs(_capped_shift_x - shift_x) > 0.001
                    or abs(_capped_shift_z - shift_z) > 0.001):
                print(f"  [carpet_mask_align] pass {_pass}: shift ({shift_x:.3f}, "
                      f"{shift_z:.3f}) capped to ({_capped_shift_x:.3f}, "
                      f"{_capped_shift_z:.3f}) to keep carpet rectangular within room")
            else:
                print(f"  [carpet_mask_align] pass {_pass}: shift "
                      f"({_capped_shift_x:.3f}, {_capped_shift_z:.3f})")
            for i in range(4):
                cp["world_corners"][i] = [
                    float(corners[i][0] + _capped_shift_x),
                    0.0,
                    float(corners[i][2] + _capped_shift_z),
                ]

        # No per-axis scale — size from mask back-projection is authoritative.

    # Update position_m from new centroid
    corners = [np.array(c, dtype=np.float64) for c in cp["world_corners"]]
    centroid = np.mean(corners, axis=0)
    cp["position_m"] = [float(centroid[0]), 0.0, float(centroid[2])]

    return cp


def _load_base_capped(base_path, cam: dict):
    """Load the base render image and downscale it to the camera's (possibly
    capped) width_px×height_px.

    The base renders (render_final.png etc.) are written at the NATIVE photo
    resolution. After cap_camera_resolution() shrinks cam["width_px"/"height_px"],
    every renderer here sizes its framebuffer + intrinsics off the loaded base
    image, so the base must be downscaled to the capped resolution too — that is
    what actually caps the offscreen framebuffer (preventing the 21 MP OOM).
    Returns a PIL RGB image at (cam W, cam H). 3-D math is untouched: fx/cx/cy
    are still derived from the image W,H, which now equals the capped cam W,H.
    """
    from PIL import Image as _PImg
    img = _PImg.open(base_path).convert("RGB")
    tgt_w = int(cam.get("width_px", img.width))
    tgt_h = int(cam.get("height_px", img.height))
    if tgt_w > 0 and tgt_h > 0 and (img.width, img.height) != (tgt_w, tgt_h):
        img = img.resize((tgt_w, tgt_h), _PImg.LANCZOS)
    return img


def _render_carpet_preview(
    cp: dict,
    cam: dict,
    output_dir: Path,
) -> Path | None:
    """Render carpet onto the scene base image and return the preview path."""
    from PIL import Image

    candidates = [
        output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
        output_dir / "render_final.png",
        output_dir / "render.png",
    ]
    base_path = next((p for p in candidates if p.exists()), None)
    if base_path is None:
        return None

    buf = np.array(_load_base_capped(base_path, cam), dtype=np.uint8)
    H, W = buf.shape[:2]

    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    ok = _render_flat_carpet_quad(buf, cp, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
    if not ok:
        return None

    preview_path = output_dir / "furniture" / "_carpet_preview.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(buf).save(str(preview_path))
    return preview_path


# ── Iterative VLM placement refinement for furniture objects ─────────────────

def _resolve_glb(raw: "str | Path", output_dir: "str | Path") -> Path:
    """Resolve a placement's glb_path, tolerant of stale absolute paths after a
    scene is moved/renamed (e.g. into outputs/reallife/_good/<scene>/). If the
    stored path no longer exists, fall back to the scene's own object store:
    <output_dir>/furniture/objects/<basename>. Returns the original Path if no
    candidate exists (callers still log "GLB missing")."""
    gp = Path(raw)
    if gp.exists():
        return gp
    alt = Path(output_dir) / "furniture" / "objects" / gp.name
    return alt if alt.exists() else gp


def _render_object_preview(
    p: dict,
    cam: dict,
    output_dir: Path,
    existing_placements: list[dict] | None = None,
) -> Path | None:
    """Render a single placed object onto the base scene and return preview path.

    If *existing_placements* is provided, render those first (so context objects
    are visible), then render the target object *p* on top.
    """
    import trimesh
    from PIL import Image

    candidates = [
        output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
        output_dir / "render_final.png",
        output_dir / "render.png",
    ]
    base_path = next((pp for pp in candidates if pp.exists()), None)
    if base_path is None:
        return None

    buf = np.array(_load_base_capped(base_path, cam), dtype=np.uint8)
    H, W = buf.shape[:2]

    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    # Render all objects (existing + target)
    all_to_render = list(existing_placements or []) + [p]
    for pl in all_to_render:
        if pl.get("is_carpet") and "world_corners" in pl:
            _render_flat_carpet_quad(buf, pl, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
            continue
        if pl.get("wall_mounted") and "_verts" in pl:
            verts = pl["_verts"]
            faces = pl["_faces"]
            vert_colors = pl["_vc"]
            proj = [_project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy) for v in verts]
            for fi in range(len(faces)):
                i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
                px0, py0, zc0 = proj[i0]; px1, py1, zc1 = proj[i1]; px2, py2, zc2 = proj[i2]
                if zc0 <= NEAR_CLIP or zc1 <= NEAR_CLIP or zc2 <= NEAR_CLIP:
                    continue
                pts = np.array([[px0, py0, zc0], [px1, py1, zc1], [px2, py2, zc2]], dtype=np.float32)
                col = np.array([vert_colors[i0], vert_colors[i1], vert_colors[i2]], dtype=np.uint8)
                _rasterize_vc_tri(buf, zbuf, pts, col)
            continue
        if "glb_path" not in pl:
            continue
        glb_path = _resolve_glb(pl["glb_path"], output_dir)
        if not glb_path.exists():
            continue
        try:
            scene_or_mesh = trimesh.load(str(glb_path), force="scene")
            if isinstance(scene_or_mesh, trimesh.Scene):
                sub_meshes = list(scene_or_mesh.geometry.values())
                mesh = trimesh.util.concatenate(sub_meshes) if len(sub_meshes) > 1 else sub_meshes[0]
            else:
                mesh = scene_or_mesh
            if not pl.get("skip_ground_removal"):
                result = _remove_ground_faces(mesh)
                mesh = result[0] if isinstance(result, tuple) else result
        except Exception:
            continue

        bounds   = mesh.bounds
        scale    = np.array(pl["scale"],        dtype=np.float64)
        R        = np.array(pl["rotation_3x3"], dtype=np.float64)
        pos      = np.array(pl["position_m"],   dtype=np.float64)
        wall_aff = pl.get("wall_affinity", "centre")
        _FLUSH_TYPES_R = {"sofa", "armchair", "bookcase", "cabinet", "desk", "bed"}
        p_back_flush = pl.get("back_flush", wall_aff != "centre" and pl.get("type", "") in _FLUSH_TYPES_R)
        room_w_cam, _ = _get_room_dims(glb_path, cam)

        if pl.get("skip_floor_transform"):
            verts = mesh.vertices.astype(np.float64).copy()
            verts[:, 1] = 0.0
        else:
            p_obb_R = np.array(pl["obb_R"], dtype=np.float64) if "obb_R" in pl else None
            verts = _apply_floor_transform(
                mesh.vertices, bounds, scale, R, pos,
                wall_affinity=wall_aff, room_w=room_w_cam, back_flush=p_back_flush,
                obb_R=p_obb_R, quiet=True,
                skip_level_base=bool(pl.get("flat_on_floor")),
                skip_wall_snap=bool(pl.get("_pos_override")),
            )

        faces = mesh.faces
        vert_colors = _get_vertex_colors(mesh)
        proj = [_project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy) for v in verts]
        for fi in range(len(faces)):
            i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
            px0, py0, zc0 = proj[i0]; px1, py1, zc1 = proj[i1]; px2, py2, zc2 = proj[i2]
            if zc0 <= NEAR_CLIP or zc1 <= NEAR_CLIP or zc2 <= NEAR_CLIP:
                continue
            pts = np.array([[px0, py0, zc0], [px1, py1, zc1], [px2, py2, zc2]], dtype=np.float32)
            col = np.array([vert_colors[i0], vert_colors[i1], vert_colors[i2]], dtype=np.uint8)
            _rasterize_vc_tri(buf, zbuf, pts, col)

    preview_path = output_dir / "furniture" / "_object_preview.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(buf).save(str(preview_path))
    return preview_path


def _vlm_fine_rotation_correction(
    preview_path: Path,
    reference_path: Path,
    obj_type: str,
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    max_deg: int = 25,
    wall_affinity: str = "centre",
    obj_width_m: float | None = None,
    obj_depth_m: float | None = None,
    obj_world_pos: tuple[float, float] | None = None,
    room_w: float | None = None,
    room_d: float | None = None,
) -> dict:
    """Ask VLM for a *fine* yaw correction in [-max_deg, +max_deg] degrees so
    the object's orientation in the render matches the reference photo.

    Returns dict with:
      delta_deg: float in [-max_deg, +max_deg], 0 if no correction
      notes:     short reasoning
    """
    import re, requests
    bx1, by1, bx2, by2 = bbox
    nx1, ny1 = bx1 / img_w, by1 / img_h
    nx2, ny2 = bx2 / img_w, by2 / img_h

    # Wall-affinity + size context: tell the VLM about the layout role and
    # the object's footprint. Size matters for the prior: a 2.5 m sectional
    # almost always anchors a wall (flush, ≤5° tilt), while a 1.0 m loveseat
    # or 0.6 m accent chair has much more layout flexibility (often 15-30°).
    if obj_width_m is not None and obj_depth_m is not None:
        long_axis = max(obj_width_m, obj_depth_m)
        if long_axis >= 1.8:
            size_prior = (
                f"This is a LARGE piece (long axis {long_axis:.2f} m). Large sofas/"
                f"sectionals/dining tables almost always anchor a wall flush — expect "
                f"0-5° tilt; >10° would be unusual."
            )
        elif long_axis >= 1.0:
            size_prior = (
                f"This is a MEDIUM piece (long axis {long_axis:.2f} m, e.g. loveseat / "
                f"single-seat sofa). Often angled into the room — common range 0-20°."
            )
        else:
            size_prior = (
                f"This is a SMALL piece (long axis {long_axis:.2f} m, e.g. accent chair / "
                f"stool). High layout flexibility — can be at any angle, common 0-30°."
            )
        size_ctx = f" {size_prior}\n"
    else:
        size_ctx = ""

    wall_ctx = (
        f"\nThe {obj_type}'s wall_affinity is '{wall_affinity}' — meaning the layout plan "
        f"placed it near the {wall_affinity if wall_affinity in ('back','left','right') else 'centre of the room'}. "
        f"This is just context; what matters is the actual visible angle in the photo.{size_ctx}"
    )

    if wall_affinity in ("back", "left", "right"):
        # Wall-affinity baseline is 0° (long axis parallel to wall, back
        # against wall).  We don't need the VLM to compare two images —
        # just ask it to read the angle of the long axis off the
        # reference photo.  That number IS the delta to apply.
        # Tell VLM exactly which side of the wall the object sits on so
        # it can disambiguate "tilts toward room centre" by handedness:
        # an object on the LEFT half of the back wall should tilt CCW
        # (its right side forward) to face room centre, while one on the
        # RIGHT half should tilt CW (left side forward).  Without this
        # cue the VLM applies the same sign to both halves.
        # Decide left-half vs right-half from WORLD position when available
        # (image-space bbox can place both sofas on the same image side
        # even though they sit on opposite halves of the back wall).
        _wall_side = None
        _to_centre = "(insufficient context — pick by visible tilt)"
        if obj_world_pos is not None:
            wx, wz = float(obj_world_pos[0]), float(obj_world_pos[1])
            if wall_affinity == "back" and room_w:
                _wall_side = "left half" if wx < room_w / 2.0 else "right half"
                _to_centre = ("CCW (its right side rotates forward toward camera, integer +X°)"
                              if wx < room_w / 2.0
                              else "CW (its left side rotates forward toward camera, integer −X°)")
            elif wall_affinity == "front" and room_w:
                _wall_side = "left half" if wx < room_w / 2.0 else "right half"
                _to_centre = ("CW toward room centre (−X°)"
                              if wx < room_w / 2.0
                              else "CCW toward room centre (+X°)")
            elif wall_affinity == "left" and room_d:
                _wall_side = "near half" if wz > room_d / 2.0 else "far half"
                _to_centre = ("CW toward room centre (−X°)"
                              if wz > room_d / 2.0
                              else "CCW toward room centre (+X°)")
            elif wall_affinity == "right" and room_d:
                _wall_side = "near half" if wz > room_d / 2.0 else "far half"
                _to_centre = ("CCW toward room centre (+X°)"
                              if wz > room_d / 2.0
                              else "CW toward room centre (−X°)")
        if _wall_side is not None:
            position_ctx = (
                f"\nPositional context: this {obj_type} sits on the {_wall_side} of "
                f"the {wall_affinity} wall.  For a typical conversational "
                f"arrangement where it angles toward the room centre rather than "
                f"flush against the wall, the expected tilt direction is {_to_centre}.\n"
            )
        else:
            position_ctx = ""
        prompt = f"""Estimate the **{obj_type}**'s yaw angle from its nearest wall in the reference photo.

Image 1: REFERENCE photo (ground truth pose).
Image 2: current 3D render — IGNORE this; the rendered baseline is 0° (long axis parallel to the {wall_affinity} wall, back flush against it).  We only need the reference angle.

The {obj_type}'s region in the reference is at normalised bbox [{nx1:.2f}, {ny1:.2f}, {nx2:.2f}, {ny2:.2f}].
{wall_ctx}{position_ctx}
Method:
  1. In the REFERENCE photo, trace the {obj_type}'s long axis (the line of the
     sofa's back / seat-front, the chair's seat-front, etc.).
  2. Compare that axis to the {wall_affinity} wall the object sits against.
  3. Measure the deviation in degrees, signed by which side rotates forward:
       • 0°  = long axis perfectly parallel to that wall (back fully flush).
       • +X° = counter-clockwise viewed from above (the side at LOW X / LOW Z
               of the wall, i.e. the LEFT side from camera, rotates forward).
       • −X° = clockwise viewed from above (the RIGHT side rotates forward).
  4. Return that angle as delta_deg (integer, in [-{max_deg}, +{max_deg}]).

Pay attention to the positional context above when picking the sign — the
direction is determined by which side of the {obj_type} comes forward in
the photo.  If the sofa to the LEFT of the room and the sofa to the RIGHT
both tilt toward each other, they will have OPPOSITE signs.

Don't return 0 unless the {obj_type} is genuinely flush (within ~3°) in
the photo.

Respond with ONLY a JSON object, no markdown:
{{"delta_deg": <int>, "notes": "<1 short sentence: estimated reference angle>"}}"""
    else:
        # Centre-affinity (or no wall): no fixed baseline, compare both images
        prompt = f"""Compare the **{obj_type}** yaw rotation between the reference photo and the 3D render.

Image 1: REFERENCE photo (ground truth pose).
Image 2: current 3D RENDER after cardinal-orient pass.

The {obj_type}'s region is at normalised bbox [{nx1:.2f}, {ny1:.2f}, {nx2:.2f}, {ny2:.2f}].
{wall_ctx}
Step-by-step:
  1. In the REFERENCE: trace the {obj_type}'s long axis (line of the back/seat-front of a sofa,
     the front edge of a chair). Which compass direction does it point? Is it parallel to a
     wall, or visibly angled? Estimate the angle in degrees from the nearest wall.
  2. In the RENDER: trace the same axis. What angle from the same wall?
  3. Δyaw = (reference_angle) − (render_angle). Positive = counter-clockwise viewed from above.
  4. If both look essentially the same orientation (within ~3°), return delta_deg=0.
  5. Otherwise return the integer delta in [-{max_deg}, +{max_deg}]. Common cases:
       • Chair facing room centre instead of straight ahead → small ±5 to ±15.
       • Both look identical → 0.

Be honest about what you see. Don't default to 0 just because items "should" be flush —
many real living rooms have angled wall-seating arrangements.

Respond with ONLY a JSON object, no markdown:
{{"delta_deg": <int>, "notes": "<1 short sentence: ref angle vs render angle>"}}"""

    content: list[dict] = [{"type": "text", "text": prompt}]
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(reference_path)}"}})
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(preview_path)}"}})

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    # Same silent-degradation trap as _vlm_refine_placement: no JSON meant
    # delta_deg=0 with nothing logged. Retry once, and report when unverified.
    for _try in range(2):
        try:
            resp = _vlm_post(payload, timeout=90)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"] or ""
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
            m = re.search(r"\{[\s\S]*?\}", raw)
            if m:
                result = json.loads(m.group())
                d = int(round(float(result.get("delta_deg", 0))))
                d = max(-max_deg, min(max_deg, d))
                return {"delta_deg": d, "notes": str(result.get("notes", ""))[:200]}
            print(f"  [vlm_fine_rot] no JSON in reply (try {_try + 1}/2), "
                  f"raw={raw[:160]!r}")
        except Exception as e:
            print(f"  [vlm_fine_rot] call failed (try {_try + 1}/2): {e}")
    print("  [vlm_fine_rot] UNVERIFIED — no rotation correction applied")
    return {"delta_deg": 0, "notes": "vlm unavailable (UNVERIFIED)"}


def _apply_y_rotation_deg(p: dict, deg: float) -> None:
    """Apply an extra Y-axis rotation (degrees) to p['rotation_3x3'] in place.
    The SAT collision code reads rotation_3x3 each pass so the OBB hull
    automatically rotates with it (no need to update _xz_hull_pts)."""
    if abs(deg) < 0.1:
        return
    rad = np.radians(float(deg))
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R_corr = np.array([[ cos_a, 0.0, sin_a],
                       [   0.0, 1.0,   0.0],
                       [-sin_a, 0.0, cos_a]], dtype=np.float64)
    R_base = np.array(p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
    p["rotation_3x3"] = (R_corr @ R_base).tolist()


def vlm_iterative_placement_refine(
    p: dict,
    entry: dict,
    cam: dict,
    output_dir: Path,
    existing_placements: list[dict],
    max_iters: int = 2,
) -> dict:
    """Refine orientation by rendering multiple rotation candidates and asking
    the VLM to pick the one that best matches the reference image.

    Instead of asking "estimate a rotation correction" (unreliable), we render
    the object at 4 candidate Y-rotations (0°, 90°, 180°, -90°) relative to
    the current pose, show all candidates alongside the reference crop, and ask
    the VLM to choose the best match.  This is far more robust because the VLM
    only has to *compare* images, not estimate geometric transforms.

    For wall-affinity objects (back/left/right), only 0° and 180° are offered
    (90°/-90° would rotate the object off the wall).

    Position and scale are NOT modified — those are handled by mask-projection.
    """
    import re, requests, copy
    from PIL import Image as _PILImage

    obj_type = entry.get("type", "object")

    # Run orientation check for objects with a clear front/back distinction.
    # Hunyuan meshes have no consistent front axis, so VLM comparison against
    # the reference image is the most reliable way to get orientation right.
    # Skip only for truly symmetric objects (plants, cylindrical tables).
    _SKIP_ORIENT_TYPES = {"plant", "lamp", "vase"}
    if obj_type.lower() in _SKIP_ORIENT_TYPES:
        print(f"  [vlm_orient] skipping for {obj_type} (symmetric/no clear front)")
        return p

    # If facing_toward rotation was applied, don't override it — the object
    # is deliberately oriented toward its target (e.g. chair facing desk)
    if p.get("_facing_applied"):
        print(f"  [vlm_orient] skipping — facing_toward rotation already applied")
        return p

    # Wall-affinity objects (back/left/right) — including seating —
    # use the deterministic back-against-wall rotation only.  vlm_orient
    # was tried for seating but consistently flipped seats to face the
    # wall instead of the room (VLM mis-reads the candidate-vs-reference
    # comparison for cluttered seating poses).  Any small-angle tweak
    # comes from fine_rot; for genuine front/back errors use the manual
    # `yaw_offset_deg: 180` field in segment_results.
    wall_aff = p.get("wall_affinity", "centre")
    _orient_aff = entry.get("_orient_affinity")
    if wall_aff in ("back", "left", "right") or _orient_aff in ("back", "left", "right"):
        _skip_reason = f"wall_affinity={wall_aff}" if wall_aff != "centre" else f"orient_affinity={_orient_aff}"
        print(f"  [vlm_orient] skipping — {_skip_reason}, "
              f"base rotation is deterministic")
        p["_vlm_orient_done"] = True
        return p

    # Find reference image
    ref_candidates = [
        output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
        output_dir / "render_final.png",
        output_dir / "render.png",
    ]
    ref_path = next((rp for rp in ref_candidates if rp.exists()), None)
    if ref_path is None:
        print("  [vlm_orient] no reference image — skipping")
        return p

    bbox = entry.get("box_px")
    if bbox is None or len(bbox) != 4:
        print("  [vlm_orient] no bbox — skipping")
        return p

    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    obj_type = entry.get("type", "object")

    # Centre-affinity: test all 4 rotation candidates.
    # Wall-affinity seating: restrict to {0°, 180°} so the long axis stays
    # parallel to the wall — only the front/back facing is allowed to flip.
    # (±90° would rotate the object off the wall, which is never what we
    # want for a wall-affinity placement.)
    R_base = np.array(p["rotation_3x3"], dtype=np.float64)
    if wall_aff in ("back", "left", "right") or _orient_aff in ("back", "left", "right"):
        candidate_degs = [0, 180]
    else:
        candidate_degs = [0, 90, 180, -90]
    candidate_previews: list[tuple[int, Path]] = []  # (deg, path)

    for deg in candidate_degs:
        p_cand = copy.deepcopy(p)
        if deg != 0:
            rad = np.radians(float(deg))
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            R_corr = np.array([
                [ cos_a, 0.0, sin_a],
                [   0.0, 1.0,   0.0],
                [-sin_a, 0.0, cos_a],
            ], dtype=np.float64)
            p_cand["rotation_3x3"] = (R_corr @ R_base).tolist()
        preview = _render_object_preview(p_cand, cam, output_dir, existing_placements)
        if preview is None or not Path(preview).exists():
            print(f"  [vlm_orient] render failed for {deg}° — skipping candidate")
            continue
        # Save with unique name (copy, not rename — _render_object_preview reuses a
        # single _object_preview.png path, so a rename can race a later candidate;
        # copy + verify keeps each candidate file independent).
        cand_path = output_dir / "furniture" / f"_orient_cand_{deg}.png"
        try:
            import shutil as _sh
            _sh.copyfile(str(preview), str(cand_path))
        except Exception as _ce:
            print(f"  [vlm_orient] could not save candidate {deg}° ({_ce}) — skipping")
            continue
        if not cand_path.exists():
            continue
        candidate_previews.append((deg, cand_path))

    if not candidate_previews:
        print("  [vlm_orient] no candidates rendered — skipping")
        return p

    # If only one candidate rendered, nothing to compare
    if len(candidate_previews) == 1:
        for _, cp in candidate_previews:
            cp.unlink(missing_ok=True)
        return p

    # Draw a highlight rectangle around the target object in each image so
    # the VLM knows exactly which object to compare.
    from PIL import ImageDraw as _PILDraw

    def _highlight_bbox(img_path: Path, out_path: Path) -> "Path | None":
        if not Path(img_path).exists():
            return None
        img = _PILImage.open(img_path).convert("RGB")
        draw = _PILDraw.Draw(img)
        # Draw a bright rectangle around the object bbox
        for offset in range(3):
            draw.rectangle(
                [bx1 - offset, by1 - offset, bx2 + offset, by2 + offset],
                outline=(255, 0, 0),
            )
        img.save(str(out_path))
        img.close()
        return out_path

    img_w, img_h = _PILImage.open(ref_path).size

    ref_highlighted = output_dir / "furniture" / "_orient_ref.png"
    _highlight_bbox(ref_path, ref_highlighted)

    cand_highlighted: list[tuple[int, Path]] = []
    for deg, cp in candidate_previews:
        hl_path = output_dir / "furniture" / f"_orient_cand_hl_{deg}.png"
        if _highlight_bbox(cp, hl_path) is None:
            print(f"  [vlm_orient] candidate {deg}° image missing — skipping")
            continue
        cand_highlighted.append((deg, hl_path))
    # Need ≥2 valid candidates to compare; otherwise keep the object as-is.
    if len(cand_highlighted) < 2:
        print("  [vlm_orient] <2 valid candidates — keeping current orientation")
        for _, cp in candidate_previews:
            Path(cp).unlink(missing_ok=True)
        return p

    # Build VLM prompt with all candidates
    label_map = {}
    for i, (deg, _) in enumerate(cand_highlighted):
        label_map[deg] = chr(ord("A") + i)

    cand_desc = ", ".join(f"{label_map[deg]}={deg}°" for deg, _ in cand_highlighted)

    # Build nearby-object context
    nearby_lines = []
    if "position_m" in p:
        obj_xz = np.array([p["position_m"][0], p["position_m"][2]])
        for ep in existing_placements:
            if ep.get("is_carpet") or ep.get("wall_mounted"):
                continue
            if "position_m" not in ep:
                continue
            ep_xz = np.array([ep["position_m"][0], ep["position_m"][2]])
            dist = float(np.linalg.norm(obj_xz - ep_xz))
            if dist < 3.0:
                ep_type = ep.get("type", "object")
                dx = ep["position_m"][0] - p["position_m"][0]
                dz = ep["position_m"][2] - p["position_m"][2]
                dirs = []
                if abs(dx) > 0.1:
                    dirs.append("to the right" if dx > 0 else "to the left")
                if abs(dz) > 0.1:
                    dirs.append("behind (toward camera)" if dz > 0 else "in front (toward back wall)")
                dir_str = " and ".join(dirs) if dirs else "very close"
                nearby_lines.append(f"  - {ep_type} at {dist:.1f}m, {dir_str}")

    context_block = ""
    if nearby_lines:
        context_block = f"""
Nearby furniture:
{chr(10).join(nearby_lines)}
Consider which object the {obj_type} should face toward (e.g., a chair faces toward a table or the room centre).
"""

    # Resolve the detilt grid image — 4 canonical views of the mesh
    _idx = entry.get("index", 0)
    _grid_path = (output_dir / "furniture" / "detilt_grids"
                  / f"idx{_idx:02d}_{obj_type}_grid.png")
    _has_grid = _grid_path.exists()

    # Build prompt — include the grid so the VLM can learn what front/back
    # look like on this specific mesh before comparing candidates.
    _grid_desc = ""
    if _has_grid:
        _grid_desc = f"""
Image 2: MESH VIEWS — a 2×2 grid showing 4 orthogonal views of this {obj_type}'s 3D mesh:
  A (top-left): 0° view
  B (top-right): 90° view (right side)
  C (bottom-left): 180° view (opposite of A)
  D (bottom-right): 270° view (left side)
Use these to learn what the FRONT and BACK of this specific {obj_type} look like.
The FRONT of a chair/sofa shows the open SEAT where you SIT DOWN. The BACK shows the rear of the backrest (flat/solid panel).

"""
        _cand_img_label = "Images 3+"
    else:
        _cand_img_label = "Images 2+"

    prompt = f"""Which rotation of the **{obj_type}** best matches the reference?

Image 1: REFERENCE — a crop of the {obj_type} from the original room photo showing how it should look and which direction it faces.
{_grid_desc}{_cand_img_label}: CANDIDATES — crops of the {obj_type} rendered at different Y-axis rotations from the same camera viewpoint.
Left in the image is the same direction across all candidate images.

Candidates (rotation labels): {cand_desc}
{context_block}
Steps:
1. {"From the MESH VIEWS grid, identify which view (A/B/C/D) shows the FRONT of the object (the seat/open side for chairs/sofas). This is your ground truth for what front vs back looks like." if _has_grid else "Identify what the FRONT of this object looks like (seat/open side for chairs/sofas)."}
2. In the REFERENCE image, determine which direction the FRONT faces: left, right, toward viewer, or away.
3. For each CANDIDATE, determine which direction the FRONT faces — use the mesh views to distinguish front from back.
4. Pick the candidate whose FRONT faces the SAME direction as in the reference.

IMPORTANT: Front and back can look similar from a distance. Use the mesh views to tell them apart.
For chairs/sofas: the FRONT shows the open seat (concave, low). The BACK shows the backrest rear (tall, flat/solid).

Respond with ONLY a JSON object, no markdown:
{{"ref_facing": "front faces toward [direction in image]", "best": "A", "notes": "brief explanation"}}"""

    content: list[dict] = [{"type": "text", "text": prompt}]

    # Image 1: segmented canvas crop from the original photo
    _seg_canvas = entry.get("canvas_file", "")
    _seg_canvas_path = (output_dir / "furniture" / "segmented" / _seg_canvas
                        if _seg_canvas else None)
    if _seg_canvas_path and _seg_canvas_path.exists():
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_encode_image(_seg_canvas_path)}"}})
    else:
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_encode_image(ref_highlighted)}"}})

    # Image 2 (optional): detilt grid — 4 canonical views of the mesh
    if _has_grid:
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_encode_image(_grid_path)}"}})

    # Candidate images: crop the bbox region from each rendered scene
    _pad = 20  # pixels padding around bbox
    for deg, cp in cand_highlighted:
        try:
            _cand_img = _PILImage.open(cp).convert("RGB")
            _cx1 = max(0, bx1 - _pad)
            _cy1 = max(0, by1 - _pad)
            _cx2 = min(_cand_img.width, bx2 + _pad)
            _cy2 = min(_cand_img.height, by2 + _pad)
            _crop = _cand_img.crop((_cx1, _cy1, _cx2, _cy2))
            _crop_path = cp.parent / f"_orient_crop_{deg}.png"
            _crop.save(str(_crop_path))
            _cand_img.close()
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{_encode_image(_crop_path)}"}})
            _crop_path.unlink(missing_ok=True)
        except Exception:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{_encode_image(cp)}"}})

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    best_deg = 0
    try:
        resp = _vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        print(f"  [vlm_orient] raw: {raw[:300]}")
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group())
            best_label = result.get("best", "A").strip().upper()
            notes = result.get("notes", "")
            # Map label back to degree
            for deg, _ in cand_highlighted:
                if label_map[deg] == best_label:
                    best_deg = deg
                    break
            # Notes-based override: VLM sometimes contradicts itself — the
            # reasoning in "notes" is often more reliable than the "best" field.
            # If notes say "only candidate X ... matches/correct", use X.
            _notes_match = re.search(
                r'\bonly\s+candidate\s+([A-D])\b[^.]*?'
                r'(?:match(?:es|ing)|correct|right)',
                notes, re.IGNORECASE)
            if _notes_match:
                _override = _notes_match.group(1).upper()
                if _override != best_label:
                    _override_deg = best_deg
                    for deg, _ in cand_highlighted:
                        if label_map[deg] == _override:
                            _override_deg = deg
                            break
                    print(f"  [vlm_orient] notes-override: best={best_label} "
                          f"contradicted by notes → using {_override} ({_override_deg}°)")
                    best_label = _override
                    best_deg = _override_deg
            print(f"  [vlm_orient] best={best_label} → {best_deg}°  notes={notes}")
    except Exception as e:
        print(f"  [vlm_orient] VLM call failed: {e}")

    # Apply the chosen rotation
    if best_deg != 0 and "rotation_3x3" in p:
        rad = np.radians(float(best_deg))
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        R_corr = np.array([
            [ cos_a, 0.0, sin_a],
            [   0.0, 1.0,   0.0],
            [-sin_a, 0.0, cos_a],
        ], dtype=np.float64)
        p["rotation_3x3"] = (R_corr @ R_base).tolist()
        print(f"  [vlm_orient] applied {best_deg}° rotation")
    else:
        print(f"  [vlm_orient] current orientation correct (0°)")

    # Mark as checked so post-verify doesn't override this decision.
    p["_vlm_orient_done"] = True

    # Clean up temp files
    for _, cp in candidate_previews:
        cp.unlink(missing_ok=True)
    for _, cp in cand_highlighted:
        cp.unlink(missing_ok=True)
    ref_highlighted.unlink(missing_ok=True)

    return p


# ── Post-placement orientation verification ──────────────────────────────────

def _vlm_post_placement_verify(
    placements: list[dict],
    seg_by_idx: dict,
    cam: dict,
    output_dir: Path,
) -> list[dict]:
    """After all objects are placed, render the full scene and verify each
    seating object's orientation against the reference photo.

    For each chair/sofa, crop the rendered scene around the object, show the
    VLM the crop + reference crop + detilt grid, and ask if orientation is
    correct.  If not, apply the suggested rotation correction.

    This catches orientation errors that the pre-placement VLM orient missed,
    because the full scene context (other furniture, walls) is now visible.
    """
    import re, requests
    from PIL import Image as _PILImage

    _SEATING_TYPES = {"chair", "armchair", "stool", "office_chair",
                      "sofa", "couch", "loveseat"}

    # Find seating placements that need verification.
    # Skip objects that already went through VLM orient — it tested all 4
    # candidates against the reference and made a reliable decision.
    # Post-verify only checks objects that skipped VLM orient (e.g.
    # _facing_applied chairs whose orientation was set geometrically).
    seating = []
    for p in placements:
        if p.get("is_carpet") or p.get("wall_mounted"):
            continue
        obj_type = p.get("type", "object").lower().replace("-", "_")
        if obj_type not in _SEATING_TYPES:
            continue
        if p.get("_vlm_orient_done"):
            print(f"  [post_verify] idx={p.get('index')} {obj_type}: "
                  f"skipping — VLM orient already checked")
            continue
        if p.get("_facing_applied"):
            print(f"  [post_verify] idx={p.get('index')} {obj_type}: "
                  f"skipping — facing_toward rotation applied")
            continue
        seating.append(p)

    if not seating:
        print("[post_verify] no seating furniture to verify")
        return placements

    # Render the full scene as-is
    preview_path = output_dir / "furniture" / "_post_verify_scene.png"
    _render_full_scene_preview(placements, cam, output_dir, preview_path)
    if not preview_path.exists():
        print("[post_verify] failed to render scene preview")
        return placements

    # Find reference image
    ref_candidates = [
        output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
        output_dir / "render_final.png",
        output_dir / "render.png",
    ]
    ref_path = next((rp for rp in ref_candidates if rp.exists()), None)
    if ref_path is None:
        print("[post_verify] no reference image — skipping")
        preview_path.unlink(missing_ok=True)
        return placements

    any_corrected = False

    for p in seating:
        idx = p.get("index", 0)
        obj_type = p.get("type", "object")
        entry = seg_by_idx.get(idx, {})
        bbox = entry.get("box_px")
        if bbox is None or len(bbox) != 4:
            print(f"  [post_verify] idx={idx} {obj_type}: no bbox — skipping")
            continue

        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        _pad = 30  # generous padding for context

        # Crop rendered scene around this object
        try:
            scene_img = _PILImage.open(preview_path).convert("RGB")
            cx1 = max(0, bx1 - _pad)
            cy1 = max(0, by1 - _pad)
            cx2 = min(scene_img.width, bx2 + _pad)
            cy2 = min(scene_img.height, by2 + _pad)
            render_crop = scene_img.crop((cx1, cy1, cx2, cy2))
            render_crop_path = output_dir / "furniture" / f"_post_verify_crop_{idx}.png"
            render_crop.save(str(render_crop_path))
            scene_img.close()
        except Exception as e:
            print(f"  [post_verify] idx={idx} crop failed: {e}")
            continue

        # Reference crop (segmented canvas from original photo)
        _seg_canvas = entry.get("canvas_file", "")
        _seg_canvas_path = (output_dir / "furniture" / "segmented" / _seg_canvas
                            if _seg_canvas else None)
        if not (_seg_canvas_path and _seg_canvas_path.exists()):
            # Fall back to cropping the reference image
            try:
                ref_img = _PILImage.open(ref_path).convert("RGB")
                ref_crop = ref_img.crop((cx1, cy1, cx2, cy2))
                ref_crop_path = output_dir / "furniture" / f"_post_verify_ref_{idx}.png"
                ref_crop.save(str(ref_crop_path))
                ref_img.close()
                _seg_canvas_path = ref_crop_path
            except Exception:
                print(f"  [post_verify] idx={idx} ref crop failed — skipping")
                render_crop_path.unlink(missing_ok=True)
                continue

        # Detilt grid image
        _grid_path = (output_dir / "furniture" / "detilt_grids"
                      / f"idx{idx:02d}_{obj_type}_grid.png")
        _has_grid = _grid_path.exists()

        # Build VLM prompt
        _grid_section = ""
        if _has_grid:
            _grid_section = f"""
Image 3: MESH VIEWS — a 2×2 grid showing 4 orthogonal views of this {obj_type}'s 3D mesh:
  A (top-left): 0° view
  B (top-right): 90° view
  C (bottom-left): 180° view
  D (bottom-right): 270° view
The FRONT of a chair/sofa shows the open SEAT where you SIT DOWN (concave, lower).
The BACK shows the rear of the backrest (flat/solid panel, taller).
Use these views to learn what the front and back of this specific {obj_type} look like.
"""

        # Build context about nearby furniture for position check
        _nearby_ctx = ""
        if p.get("_facing_applied"):
            # Find the target furniture this chair faces
            for other_p in placements:
                if other_p is p or other_p.get("is_carpet") or other_p.get("wall_mounted"):
                    continue
                other_pos = other_p.get("position_m")
                if other_pos is None:
                    continue
                dx = other_pos[0] - p["position_m"][0]
                dz = other_pos[2] - p["position_m"][2]
                dist = np.sqrt(dx**2 + dz**2)
                if dist < 2.0:
                    _nearby_ctx += f"\nNearby: {other_p.get('type', 'object')} at {dist:.2f}m distance."
            if _nearby_ctx:
                _nearby_ctx = f"""
Also check POSITION: Is this {obj_type} at a natural distance from the nearby furniture?
A chair should be close enough to a desk/table to use it comfortably (within ~0.05-0.15m gap between surfaces).
{_nearby_ctx}
If the {obj_type} is too far from its desk/table, set "move_closer": true.
"""

        prompt = f"""Check if this **{obj_type}** is oriented and positioned correctly in the rendered scene.

Image 1: REFERENCE — how the {obj_type} should look in the original room photo. Pay attention to which direction the FRONT of the {obj_type} faces.
Image 2: RENDERED — the {obj_type} as currently placed in the 3D scene. Check if its front faces the same direction as in the reference.
{_grid_section}
Steps:
1. {"From the MESH VIEWS (Image 3), identify what the FRONT looks like (the seat/open side) vs the BACK (solid backrest rear)." if _has_grid else "Identify what the FRONT of this object looks like (seat/open side for chairs/sofas)."}
2. In the REFERENCE (Image 1), determine which direction the FRONT faces: left, right, toward viewer, or away from viewer.
3. In the RENDERED (Image 2), determine which direction the FRONT currently faces.
4. If they match → orientation is correct. If not → determine the rotation needed.
{_nearby_ctx}
IMPORTANT: For chairs/sofas, the FRONT is the side where you sit (open, concave, lower). The BACK is the rear of the backrest (solid, flat, taller). They can look similar from a distance — use the mesh views to tell them apart.

Respond with ONLY a JSON object, no markdown:
{{"correct": true/false, "ref_front_faces": "direction in reference", "render_front_faces": "direction in render", "correction_deg": 0, "move_closer": false, "notes": "brief explanation"}}

correction_deg should be:
- 0 if correct
- 90 if the object needs to rotate 90° clockwise (when viewed from above)
- -90 if it needs to rotate 90° counter-clockwise
- 180 if it needs to rotate 180°"""

        content: list[dict] = [{"type": "text", "text": prompt}]

        # Image 1: reference
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_encode_image(_seg_canvas_path)}"}})
        # Image 2: rendered crop
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_encode_image(render_crop_path)}"}})
        # Image 3: detilt grid (optional)
        if _has_grid:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{_encode_image(_grid_path)}"}})

        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": True},
        }

        print(f"  [post_verify] idx={idx} {obj_type}: checking orientation...")
        try:
            resp = _vlm_post(payload, timeout=120)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
            print(f"  [post_verify] raw: {raw[:300]}")
            m = re.search(r"\{[\s\S]*?\}", raw)
            if m:
                result = json.loads(m.group())
                is_correct = result.get("correct", True)
                correction = int(result.get("correction_deg", 0))
                notes = result.get("notes", "")
                print(f"  [post_verify] idx={idx} correct={is_correct} "
                      f"correction={correction}° notes={notes}")

                if not is_correct and correction != 0 and "rotation_3x3" in p:
                    rad = np.radians(float(correction))
                    cos_a, sin_a = np.cos(rad), np.sin(rad)
                    R_corr = np.array([
                        [ cos_a, 0.0, sin_a],
                        [   0.0, 1.0,   0.0],
                        [-sin_a, 0.0, cos_a],
                    ], dtype=np.float64)
                    R_cur = np.array(p["rotation_3x3"], dtype=np.float64)
                    p["rotation_3x3"] = (R_corr @ R_cur).tolist()
                    any_corrected = True
                    print(f"  [post_verify] idx={idx} applied {correction}° correction")

                # Position correction: move chair closer to its target
                move_closer = result.get("move_closer", False)
                if move_closer and p.get("_facing_applied"):
                    # Find the closest desk/table and move toward it
                    R_cur = np.array(p["rotation_3x3"], dtype=np.float64)
                    _lf = p.get("_local_front",  [0, 0, -1])
                    front_world = np.array(_lf, dtype=np.float64) @ R_cur.T
                    front_world[1] = 0.0
                    fn = np.linalg.norm(front_world)
                    if fn > 1e-6:
                        front_world /= fn
                    # Nudge 15cm in the front direction (toward desk)
                    nudge = 0.15
                    p["position_m"][0] += float(front_world[0]) * nudge
                    p["position_m"][2] += float(front_world[2]) * nudge
                    any_corrected = True
                    print(f"  [post_verify] idx={idx} nudged {nudge:.2f}m "
                          f"closer (dir={np.round(front_world, 2).tolist()})")
            else:
                print(f"  [post_verify] idx={idx} could not parse VLM response")
        except Exception as e:
            print(f"  [post_verify] idx={idx} VLM call failed: {e}")

        # Clean up temp files
        render_crop_path.unlink(missing_ok=True)
        if f"_post_verify_ref_{idx}" in str(_seg_canvas_path):
            _seg_canvas_path.unlink(missing_ok=True)

    preview_path.unlink(missing_ok=True)

    # ── Re-align corrected objects to their segmentation mask ────────────
    # Post-verify rotations change the object's projected silhouette, so
    # position and scale must be re-fit to the mask bbox.
    if any_corrected:
        print("[post_verify] re-aligning corrected objects to masks...")
        for p in placements:
            if p.get("is_carpet") or p.get("wall_mounted"):
                continue
            if p.get("_facing_applied"):
                continue  # position set by facing_toward, not mask
            mfd = p.get("_mask_fit_data")
            if mfd is None:
                continue
            _bx1, _by1, _bx2, _by2 = mfd["bbox"]
            try:
                _pos_arr, _sc_arr, _ew, _eh, _ed, _ma_err_px = _align_and_scale_to_mask(
                    mfd["mesh_verts"], mfd["mesh_bounds"],
                    np.array(p["scale"], dtype=np.float64),
                    np.array(p["rotation_3x3"], dtype=np.float64),
                    np.array(p["position_m"], dtype=np.float64),
                    _bx1, _by1, _bx2, _by2,
                    mfd["cam_pos"], mfd["right_v"], mfd["up_c_v"], mfd["fwd_v"],
                    mfd["fx"], mfd["cx"], mfd["cy"],
                    p.get("wall_affinity", "centre"), mfd["back_flush"],
                    mfd["room_w"], mfd["room_d"],
                    existing_placements=placements,
                    obj_size_m=p.get("size_m"),
                    vlm_size_locked=bool(p.get("_vlm_size_converged")
                                         or (entry or {}).get("_vlm_size_converged")),
                    mask_fill=_mask_fill_ratio(output_dir, entry or {}),
                    is_low_surface=p.get("type", "").lower().replace("-", "_") in _LOW_SURFACE_TYPES,
                    local_front=np.array(p.get("_local_front", [0, 0, -1]), dtype=np.float64),
                )
                p["position_m"] = _pos_arr.tolist()
                p["scale"] = _sc_arr.tolist()
                p["size_m"] = {"width_m": _ew, "height_m": _eh, "depth_m": _ed}
                p["eff_h"] = _eh
                print(f"  [post_verify] idx={p.get('index')} re-aligned to mask: "
                      f"pos={np.round(_pos_arr, 3).tolist()} "
                      f"scale={np.round(_sc_arr, 3).tolist()}")
            except Exception as e:
                print(f"  [post_verify] idx={p.get('index')} mask re-align failed: {e}")

    return placements


# ── Post-placement scene-level review ─────────────────────────────────────────

def _vlm_scene_review(
    placements: list[dict],
    seg_by_idx: dict,
    cam: dict,
    output_dir: Path,
) -> list[dict]:
    """Render the full scene, compare with the reference image via VLM,
    and apply position corrections for objects that are misaligned.

    Unlike _vlm_post_placement_verify (which only checks seating orientation),
    this checks ALL non-carpet objects for position mismatches: too far left/
    right, too large/small, etc.  The VLM sees the whole scene context.
    """
    import re, requests
    from PIL import Image as _PILImage

    # Render scene preview
    preview_path = output_dir / "furniture" / "_scene_review.png"
    _render_full_scene_preview(placements, cam, output_dir, preview_path)
    if not preview_path.exists():
        print("[scene_review] failed to render scene preview")
        return placements

    # Find reference image
    ref_candidates = [
        output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
        output_dir / "render_final.png",
        output_dir / "render.png",
    ]
    ref_path = next((rp for rp in ref_candidates if rp.exists()), None)
    if ref_path is None:
        print("[scene_review] no reference image — skipping")
        preview_path.unlink(missing_ok=True)
        return placements

    # Build list of placed objects with their mask bboxes
    obj_list = []
    for p in placements:
        if p.get("is_carpet") or p.get("wall_mounted"):
            continue
        idx = p.get("index", -1)
        seg = seg_by_idx.get(idx, {})
        bbox = seg.get("box_px")
        if bbox is None or len(bbox) != 4:
            continue
        obj_list.append({
            "index": idx,
            "type": p.get("type", "object"),
            "bbox": [int(v) for v in bbox],
        })

    if not obj_list:
        print("[scene_review] no objects to review")
        preview_path.unlink(missing_ok=True)
        return placements

    # Format object list for VLM
    obj_desc = "\n".join(
        f"  - idx={o['index']} {o['type']} at bbox [{o['bbox'][0]},{o['bbox'][1]},"
        f"{o['bbox'][2]},{o['bbox'][3]}]"
        for o in obj_list
    )

    prompt = f"""You are reviewing a 3D furniture placement render (Image 2) against the reference photo (Image 1).

Objects placed in the scene:
{obj_desc}

For each object, reason about its POSITION only:
- Where does it appear in Image 1 (reference)? Left/center/right, near/far, what is directly next to it?
- Where does it appear in Image 2 (render)? Same position or different?
- Is it occluding something it shouldn't? Is it too far from where it should be?
- Key spatial relationships: is the chair next to the correct sofa? Is the coffee table in FRONT of the sofa, not BESIDE it? Is the chair facing AWAY from the coffee table (creating space between them), not ON TOP of it?

For example: "In Image 1, the chair is at the far LEFT edge, partially cut off, clearly to the LEFT of the coffee table. In Image 2, the chair overlaps the coffee table. The chair must move LEFT significantly so the coffee table is visible to its right."

For each object with a CLEARLY WRONG position (>40px off), output a correction:
- "reasoning": 1-2 sentences explaining what's wrong and why
- "shift_lr": pixels LEFT (negative) or RIGHT (positive) to reach the reference position
- "shift_ud": pixels UP (negative) or DOWN (positive)

Respond with ONLY valid JSON:
{{"corrections": [
  {{"index": <idx>, "reasoning": "...", "shift_lr": <int>, "shift_ud": <int>}},
  ...
],
"overall_notes": "brief summary"
}}

If positions look correct, return: {{"corrections": [], "overall_notes": "positions match reference"}}
Do NOT suggest scale changes. Only fix objects that are CLEARLY in the wrong position."""

    content: list[dict] = [{"type": "text", "text": prompt}]
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(ref_path)}"}})
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(preview_path)}"}})

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    print("[scene_review] asking VLM to compare rendered scene vs reference...")
    try:
        resp = _vlm_post(payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        print(f"[scene_review] raw: {raw[:500]}")
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            print("[scene_review] could not parse VLM response")
            preview_path.unlink(missing_ok=True)
            return placements

        result = json.loads(m.group())
        corrections = result.get("corrections", [])
        print(f"[scene_review] {len(corrections)} corrections suggested: "
              f"{result.get('overall_notes', '')}")

        if not corrections:
            preview_path.unlink(missing_ok=True)
            return placements

        # Load camera parameters for pixel→world conversion
        cam_pos = np.array(cam["position_m"], dtype=np.float64)
        look_at = np.array(cam["look_at_m"], dtype=np.float64)
        up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
        right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
        ref_img = _PILImage.open(ref_path)
        W, H = ref_img.size
        ref_img.close()
        hfov = float(cam["hfov_deg"])
        _fx = W / (2.0 * np.tan(np.radians(hfov / 2.0)))

        any_changed = False
        for corr in corrections:
            cidx = int(corr.get("index", -1))
            # Find the placement
            p = None
            for _p in placements:
                if _p.get("index") == cidx:
                    p = _p
                    break
            if p is None:
                print(f"  [scene_review] idx={cidx} not found in placements — skip")
                continue

            shift_lr = float(corr.get("shift_lr", 0))
            shift_ud = float(corr.get("shift_ud", 0))
            reasoning = corr.get("reasoning", "")

            # Skip small positional corrections
            if abs(shift_lr) < 40 and abs(shift_ud) < 40:
                print(f"  [scene_review] idx={cidx} {p.get('type')}: "
                      f"shift too small ({shift_lr:.0f}px, {shift_ud:.0f}px) — skip")
                continue

            # Skip facing_applied objects — position anchored to desk/table target.
            if p.get("_facing_applied"):
                print(f"  [scene_review] idx={cidx} {p.get('type')}: "
                      f"skip — facing_toward rotation applied, position anchored to target")
                continue

            # on_top_of objects move with their support — scene_review must not
            # shift them independently or they'll detach from their table/shelf.
            if p.get("on_top_of") is not None:
                print(f"  [scene_review] idx={cidx} {p.get('type')}: "
                      f"skip — on_top_of object, tracks its support")
                continue

            # Note: we no longer skip based on mask_align convergence.  Scene_review
            # sees the full layout (aisle access, furniture relationships) and its
            # corrections are more trustworthy for holistic positioning than
            # mask_align's per-object silhouette alignment.

            # Back-wall objects are already positioned by mask_align (Z locked,
            # X from silhouette).  Scene_review consistently mis-identifies
            # their X in the full-scene render, making things worse.  Skip.
            _sr_wa = p.get("wall_affinity", "centre")
            if _sr_wa in ("back", "left", "right"):
                print(f"  [scene_review] idx={cidx} {p.get('type')}: "
                      f"skip — wall-affinity={_sr_wa}, mask_align already handles position")
                continue

            # For objects near image edges, ask VLM whether the object is
            # actually cut off by the frame or just touching the border.
            # If it's truly truncated, scene_review can't reliably judge
            # position from the visible portion alone.
            _seg_bp = seg_by_idx.get(cidx, {})
            _bp_box = _seg_bp.get("box_px", [])
            if len(_bp_box) == 4:
                _border_m = 5
                _clip_count = sum([
                    float(_bp_box[0]) < _border_m,
                    float(_bp_box[1]) < _border_m,
                    float(_bp_box[2]) > W - _border_m,
                    float(_bp_box[3]) > H - _border_m,
                ])
                if _clip_count >= 2:
                    # Ask VLM: is the object fully visible or cut off?
                    _crop_file = _seg_bp.get("crop_file")
                    _is_truncated = True  # default safe: assume truncated
                    if _crop_file and Path(_crop_file).exists():
                        try:
                            import re as _re_clip, requests as _req_clip
                            _clip_prompt = (
                                f"Look at this image of a {p.get('type', 'object')}. "
                                f"Is the object COMPLETELY VISIBLE in this image, or is "
                                f"part of it cut off by the image border?\n"
                                f"Answer with ONLY: {{\"complete\": true}} or {{\"complete\": false}}"
                            )
                            _clip_payload = {
                                "model": "qwen3",
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": _clip_prompt},
                                    {"type": "image_url", "image_url": {
                                        "url": f"data:image/png;base64,{_encode_image(Path(_crop_file))}"
                                    }},
                                ]}],
                                "max_tokens": 32,
                                "chat_template_kwargs": {"enable_thinking": False},
                            }
                            _clip_resp = _req_clip.post(VLM_API_URL, json=_clip_payload, timeout=30)
                            _clip_resp.raise_for_status()
                            _clip_raw = _clip_resp.json()["choices"][0]["message"]["content"]
                            _clip_m = _re_clip.search(r"\{[^}]*\}", _clip_raw)
                            if _clip_m:
                                _is_truncated = not json.loads(_clip_m.group()).get("complete", False)
                        except Exception:
                            pass  # default to truncated if VLM fails
                    if _is_truncated:
                        print(f"  [scene_review] idx={cidx} {p.get('type')}: "
                              f"skip — object truncated by image border, position not determinable")
                        continue
                    else:
                        print(f"  [scene_review] idx={cidx} {p.get('type')}: "
                              f"VLM says object is complete — applying scene_review correction")

            print(f"  [scene_review] idx={cidx} {p.get('type')}: "
                  f"shift_lr={shift_lr:.0f}px shift_ud={shift_ud:.0f}px")
            if reasoning:
                print(f"    reasoning: {reasoning}")

            # Convert pixel shift → world shift via floor back-projection.
            # Use the mask bbox bottom-centre as the anchor pixel.
            seg = seg_by_idx.get(cidx, {})
            bbox = seg.get("box_px", [0, 0, 0, 0])
            cur_px = (bbox[0] + bbox[2]) / 2.0
            cur_py = float(bbox[3])
            tgt_px = cur_px + shift_lr
            tgt_py = cur_py + shift_ud

            ray_cur = _backproject_pixel(
                cur_px, cur_py, cam_pos, right_v, up_c_v, fwd_v, _fx, _fx,
                W / 2.0, H / 2.0)
            ray_tgt = _backproject_pixel(
                tgt_px, tgt_py, cam_pos, right_v, up_c_v, fwd_v, _fx, _fx,
                W / 2.0, H / 2.0)
            floor_cur = _ray_floor_intersect(cam_pos, ray_cur)
            floor_tgt = _ray_floor_intersect(cam_pos, ray_tgt)

            if floor_cur is not None and floor_tgt is not None:
                dx = float(floor_tgt[0] - floor_cur[0])
                dz = float(floor_tgt[2] - floor_cur[2])
                # Wall-affinity is initial-placement only.  Allow scene_review
                # to move the object FORWARD off its wall (e.g. to clear a
                # collision while staying near where the photo says it should
                # be), but never push it deeper into the wall.
                if _sr_wa == "back" and dz < 0.0:
                    dz = 0.0
                elif _sr_wa == "left" and dx < 0.0:
                    dx = 0.0
                elif _sr_wa == "right" and dx > 0.0:
                    dx = 0.0
                # Cap per-pass shift to 0.40m
                mag = np.sqrt(dx**2 + dz**2)
                if mag < 0.001:
                    continue  # nothing to do after wall constraint
                if mag > 0.40:
                    dx *= 0.40 / mag
                    dz *= 0.40 / mag
                _cand_x = float(p["position_m"][0]) + dx
                _cand_z = float(p["position_m"][2]) + dz
                # Clamp to room bounds only — collisions resolved afterward
                try:
                    _sr_room_w, _sr_room_d = _get_room_dims(
                        Path(p.get("glb_path", ".")), cam)
                    _p_lhx2, _p_lhz2 = _col_xz_half_extents(p)
                    _cand_x = float(np.clip(_cand_x, _p_lhx2, _sr_room_w - _p_lhx2))
                    _cand_z = float(np.clip(_cand_z, _p_lhz2, _sr_room_d - _p_lhz2))
                except Exception:
                    pass
                # Check that shift doesn't create a new collision
                _old_x_sr = p["position_m"][0]
                _old_z_sr = p["position_m"][2]
                p["position_m"][0] = _cand_x
                p["position_m"][2] = _cand_z
                def _sr_collides(pa, pb):
                    # Use BOTH the hull SAT (precise) and the AABB used by
                    # the post-step sr_collision/_final_collides checks. If
                    # either says "collide", treat the shift as colliding —
                    # otherwise scene_review accepts a hull-clean shift that
                    # the looser AABB later sees as overlap and rebounds via
                    # sr_collision, producing the "giant leap" between the
                    # last animation frames.
                    hx_a, hz_a = _col_xz_half_extents(pa)
                    hx_b, hz_b = _col_xz_half_extents(pb)
                    _aabb_hits = ((hx_a + hx_b) > abs(pa["position_m"][0] - pb["position_m"][0])
                                  and (hz_a + hz_b) > abs(pa["position_m"][2] - pb["position_m"][2]))
                    if _aabb_hits:
                        return True
                    hull_a = pa.get("_xz_hull_pts")
                    hull_b = pb.get("_xz_hull_pts")
                    if hull_a is not None and hull_b is not None:
                        R_a  = np.array(pa.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
                        R_b  = np.array(pb.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
                        sc_a = np.array(pa.get("scale", [1, 1, 1]), dtype=np.float64)
                        sc_b = np.array(pb.get("scale", [1, 1, 1]), dtype=np.float64)
                        return _poly_collides_sat(
                            hull_a, pa["position_m"][0], pa["position_m"][2], R_a, sc_a,
                            hull_b, pb["position_m"][0], pb["position_m"][2], R_b, sc_b,
                        )
                    return False
                _sr_shift_collides = any(
                    not _op.get("is_carpet") and not _op.get("wall_mounted")
                    and _op is not p and _sr_collides(p, _op)
                    for _op in placements
                )
                if _sr_shift_collides:
                    p["position_m"][0] = _old_x_sr
                    p["position_m"][2] = _old_z_sr
                    print(f"    → world shift BLOCKED — would cause collision, keeping "
                          f"({_old_x_sr:.3f}, {_old_z_sr:.3f})")
                else:
                    any_changed = True
                    # Mark that scene_review actually moved this object — used
                    # by final_anchor to decide whether the hard-cap force-clamp
                    # (which can re-collide) should fire. Drift caused by
                    # collision resolution alone should NOT trigger force-clamp.
                    p["_scene_review_shifted"] = True
                    print(f"    → world shift dx={dx:.3f} dz={dz:.3f} "
                          f"→ ({p['position_m'][0]:.3f}, {p['position_m'][2]:.3f})")

        if any_changed:
            print("[scene_review] corrections applied")

    except Exception as e:
        print(f"[scene_review] VLM call failed: {e}")

    preview_path.unlink(missing_ok=True)
    return placements


def _render_full_scene_preview(
    placements: list[dict],
    cam: dict,
    output_dir: Path,
    out_path: Path,
) -> None:
    """Render all placed furniture onto the base scene image (1x scale)."""
    import trimesh
    from PIL import Image

    candidates = [
        output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
        output_dir / "render_final.png",
        output_dir / "render.png",
    ]
    base_path = next((pp for pp in candidates if pp.exists()), None)
    if base_path is None:
        return

    buf = np.array(_load_base_capped(base_path, cam), dtype=np.uint8)
    H, W = buf.shape[:2]

    cam_pos  = np.array(cam["position_m"],  dtype=np.float64)
    look_at  = np.array(cam["look_at_m"],   dtype=np.float64)
    up_world = np.array(cam.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    hfov = float(cam["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    for pl in placements:
        if pl.get("is_carpet") and "world_corners" in pl:
            _render_flat_carpet_quad(buf, pl, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
            continue
        if pl.get("wall_mounted") and "_verts" in pl:
            verts = pl["_verts"]
            faces = pl["_faces"]
            vert_colors = pl["_vc"]
            proj = [_project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy) for v in verts]
            for fi in range(len(faces)):
                i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
                px0, py0, zc0 = proj[i0]; px1, py1, zc1 = proj[i1]; px2, py2, zc2 = proj[i2]
                if zc0 <= NEAR_CLIP or zc1 <= NEAR_CLIP or zc2 <= NEAR_CLIP:
                    continue
                pts = np.array([[px0, py0, zc0], [px1, py1, zc1], [px2, py2, zc2]], dtype=np.float32)
                col = np.array([vert_colors[i0], vert_colors[i1], vert_colors[i2]], dtype=np.uint8)
                _rasterize_vc_tri(buf, zbuf, pts, col)
            continue
        if "glb_path" not in pl:
            continue
        glb_path = Path(pl["glb_path"])
        if not glb_path.exists():
            continue
        try:
            scene_or_mesh = trimesh.load(str(glb_path), force="scene")
            if isinstance(scene_or_mesh, trimesh.Scene):
                sub_meshes = list(scene_or_mesh.geometry.values())
                mesh = trimesh.util.concatenate(sub_meshes) if len(sub_meshes) > 1 else sub_meshes[0]
            else:
                mesh = scene_or_mesh
            if not pl.get("skip_ground_removal"):
                result = _remove_ground_faces(mesh)
                mesh = result[0] if isinstance(result, tuple) else result
        except Exception:
            continue

        bounds = mesh.bounds
        scale  = np.array(pl["scale"],        dtype=np.float64)
        R      = np.array(pl["rotation_3x3"], dtype=np.float64)
        pos    = np.array(pl["position_m"],   dtype=np.float64)

        verts_local = mesh.vertices * scale
        verts_world = verts_local @ R.T + pos

        faces_arr = mesh.faces
        vc = _get_vertex_colors(mesh)

        proj = [_project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy) for v in verts_world]
        for fi in range(len(faces_arr)):
            i0, i1, i2 = int(faces_arr[fi, 0]), int(faces_arr[fi, 1]), int(faces_arr[fi, 2])
            px0, py0, zc0 = proj[i0]; px1, py1, zc1 = proj[i1]; px2, py2, zc2 = proj[i2]
            if zc0 <= NEAR_CLIP or zc1 <= NEAR_CLIP or zc2 <= NEAR_CLIP:
                continue
            pts = np.array([[px0, py0, zc0], [px1, py1, zc1], [px2, py2, zc2]], dtype=np.float32)
            col = np.array([vc[i0], vc[i1], vc[i2]], dtype=np.uint8)
            _rasterize_vc_tri(buf, zbuf, pts, col)

    Image.fromarray(buf).save(str(out_path))
    print(f"[post_verify] scene preview saved → {out_path.name}")


# ── Carpet texture orientation check ─────────────────────────────────────────

def _vlm_carpet_orientation_check(
    preview_path: Path,
    reference_path: Path,
) -> tuple[int, int]:
    """Ask the VLM to check the carpet against the reference, on two axes.

    Returns (texture_rotation_deg, footprint_rotation_deg):

      texture_rotation_deg   — pattern/stripe direction (0, 90, -90, 180).
                               Applied to the TEXTURE only when the quad is
                               loaded; rotating the quad instead would swap
                               width/depth on a non-square rug.
      footprint_rotation_deg — whether the rug's RECTANGLE itself runs the
                               wrong way (0 or 90).  The texture check cannot
                               catch this: a rug whose long axis should run
                               left-right but was built front-back has a
                               correctly-oriented pattern on a wrongly-shaped
                               quad.  Answering it needs its own question.
    """
    import re, requests

    prompt = """Compare the carpet/rug in these two images.

Image 1: the reference room photo, showing how the rug SHOULD look.
Image 2: the current 3D render with the rug placed.

Answer TWO independent questions.

(A) TEXTURE direction — look only at the pattern itself (stripes, grain, weave).
Is the pattern in Image 2 turned relative to Image 1?
  0 = correct, 90 = needs 90° clockwise, -90 = needs 90° counter-clockwise,
  180 = needs 180°.

(B) FOOTPRINT direction — ignore the pattern and look at the rug's RECTANGLE
on the floor. In Image 1, does the rug's LONGER side run left-to-right across
the room, or away from the camera (front-to-back)? Compare that with Image 2.
  0  = the rectangle runs the same way in both
  90 = the rectangle is turned a quarter turn (its long and short sides are
       swapped relative to the reference)
If the rug looks square, or you cannot tell which side is longer, answer 0.

Respond with ONLY a JSON object, no markdown:
{"rotation_correction_deg": 0, "footprint_rotation_deg": 0, "reason": "brief explanation"}"""

    content: list[dict] = [{"type": "text", "text": prompt}]
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(reference_path)}"}})
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(preview_path)}"}})

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    try:
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            result = json.loads(m.group())
            rot  = int(result.get("rotation_correction_deg", 0))
            foot = int(result.get("footprint_rotation_deg", 0))
            reason = result.get("reason", "")
            print(f"  [vlm_carpet_orient] texture={rot}°  footprint={foot}°  "
                  f"reason={reason}")
            if rot not in (0, 90, -90, 180):
                rot = 0
            foot = 90 if abs(foot) == 90 else 0
            return rot, foot
    except Exception as e:
        print(f"  [vlm_carpet_orient] VLM call failed: {e}")
    return 0, 0


# ── Shared mesh transform ─────────────────────────────────────────────────────

def _level_base(verts: np.ndarray, max_angle_deg: float = 10.0, quiet: bool = False) -> np.ndarray:
    """Level generated meshes tilt by fitting a plane through base contacts.

    Strategy:
      1. Try 4-quadrant corner contacts (bottom 25%) — best for flat-bottomed objects.
      2. Fallback: fit plane through ALL bottom-fraction vertices, progressively
         widening the search (5%, 10%, 15%, 25%) until the bottom spans enough
         area.  This handles chairs, plants, and other irregular shapes.

    Uses the full Y range (including spike / below-base verts) so that tilted
    bases are detected properly.  Re-anchoring at the end uses _find_base_y
    to skip spikes.
    """
    y_min = float(verts[:, 1].min())
    y_range = float(verts[:, 1].max()) - y_min
    if y_range < 1e-6:
        return verts

    obj_spread = max(np.ptp(verts[:, 0]), np.ptp(verts[:, 2]))

    # Skip the spike below the actual base — picking a quadrant's min Y from
    # spike verts produces a phantom-tilted plane (e.g. low-poly sofas with a
    # vert hanging 17 cm below the seat base → bogus 19° tilt).
    base_y_trim = _find_base_y(verts, quiet=True)

    # Strategy 1: 4-quadrant corner contacts (slice ABOVE the spike-trimmed
    # base, not above the absolute Y_min). Strategy still tolerates real
    # tilt because we use a relatively wide slice (0.20 of y_range above
    # base_y_trim), but ignores hanging spike verts entirely.
    y_thresh = base_y_trim + 0.20 * y_range
    bot = verts[(verts[:, 1] >= base_y_trim - 1e-3) & (verts[:, 1] <= y_thresh)]
    pts = None
    if len(bot) >= 4:
        x_med = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
        z_med = (verts[:, 2].min() + verts[:, 2].max()) / 2.0
        contacts = []
        for x_lo in (True, False):
            for z_lo in (True, False):
                mask = ((bot[:, 0] < x_med) if x_lo else (bot[:, 0] >= x_med)) & \
                       ((bot[:, 2] < z_med) if z_lo else (bot[:, 2] >= z_med))
                if mask.sum() > 0:
                    q = bot[mask]
                    contacts.append(q[q[:, 1].argmin()])
        if len(contacts) >= 3:
            pts = np.array(contacts, dtype=np.float64)

    # Strategy 2: progressively wider bottom slices.
    # Use a tighter angle cap (8°) — large corrections from noisy bottom
    # surfaces (chairs, plants) do more harm than good.
    fallback = False
    if pts is None:
        for frac in (0.05, 0.10, 0.15, 0.25):
            y_lo = y_min + frac * y_range
            cand = verts[verts[:, 1] <= y_lo]
            if len(cand) < 3:
                continue
            xz_spread = max(np.ptp(cand[:, 0]), np.ptp(cand[:, 2]))
            if xz_spread >= 0.15 * obj_spread:
                pts = cand.copy()
                fallback = True
                if not quiet:
                    print(f"  [level_base] using bottom {frac*100:.0f}% "
                          f"({len(pts)} verts, spread={xz_spread:.3f})")
                break
        if pts is None:
            if not quiet:
                print(f"  [level_base] no stable base found — skip")
            return verts

    pts = pts.astype(np.float64)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[1] < 0:
        normal = -normal

    target = np.array([0.0, 1.0, 0.0])
    cross = np.cross(normal, target)
    sin_a = float(np.linalg.norm(cross))
    cos_a = float(np.dot(normal, target))
    angle = float(np.arctan2(sin_a, cos_a))

    # Corner-contact strategy is reliable → allow up to max_angle_deg.
    # Fallback (bottom-slice) is noisier → tighter 8° cap.
    cap = np.radians(8.0) if fallback else np.radians(max_angle_deg)
    if sin_a < 1e-6 or abs(angle) > cap:
        if abs(angle) > cap and not quiet:
            print(f"  [level_base] tilt {np.degrees(angle):.1f}° exceeds "
                  f"{'fallback ' if fallback else ''}cap {np.degrees(cap):.0f}° — skip")
        return verts   # already level, or overcorrection risk

    axis = cross / sin_a
    K = np.array([[     0, -axis[2],  axis[1]],
                  [ axis[2],      0, -axis[0]],
                  [-axis[1],  axis[0],      0]])
    R_corr = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    center = verts.mean(axis=0)
    verts = (verts - center) @ R_corr.T + center
    verts[:, 1] -= _find_base_y(verts, quiet=quiet)   # re-anchor base (not spike) to floor
    if not quiet:
        print(f"  [level_base] corrected {np.degrees(angle):.1f}° tilt")
    return verts


def _rescale_to_mask(
    mesh_verts: np.ndarray,
    mesh_bounds: np.ndarray,
    scale: np.ndarray,
    R: np.ndarray,
    pos: np.ndarray,
    obb_R: np.ndarray | None,
    bx1: float, by1: float, bx2: float, by2: float,
    cam_pos: np.ndarray,
    right_v: np.ndarray,
    up_c_v: np.ndarray,
    fwd_v: np.ndarray,
    fx: float, cx: float, cy: float,
    wall_affinity: str,
    back_flush: bool,
) -> tuple[np.ndarray, float, float, float]:
    """Adjust uniform scale so the projected silhouette matches the mask bbox.

    1. Build the object's 3D AABB from the placement transform.
    2. Project the 8 AABB corners to pixel space.
    3. Compare projected width/height with the target mask bbox.
    4. Compute a correction factor and apply to the uniform scale.

    Returns (adjusted_scale, eff_w, eff_h, eff_d).
    """
    target_w_px = float(bx2 - bx1)
    target_h_px = float(by2 - by1)
    if target_w_px < 2 or target_h_px < 2:
        _na = 2 if wall_affinity == "back" else 0
        _fa = 0 if wall_affinity == "back" else 2
        s = float(scale[0])
        if obb_R is not None:
            c = mesh_verts.mean(axis=0)
            v = (mesh_verts.astype(np.float64) - c) @ obb_R.T
            gw, gh, gd = np.ptp(v[:, _fa]), np.ptp(v[:, 1]), np.ptp(v[:, _na])
        else:
            gw = mesh_bounds[1][0] - mesh_bounds[0][0]
            gh = mesh_bounds[1][1] - mesh_bounds[0][1]
            gd = mesh_bounds[1][2] - mesh_bounds[0][2]
        return scale, gw * s, gh * s, gd * s

    # Transform actual mesh vertices (subsampled) to world space and project.
    # Using real vertices instead of 8 AABB corners gives a tighter silhouette
    # estimate — AABB corners overestimate the projection, causing mask_fit to
    # shrink objects too aggressively.
    if obb_R is not None:
        c = mesh_verts.astype(np.float64).mean(axis=0)
        v = (mesh_verts.astype(np.float64) - c) @ obb_R.T
        lo = v.min(axis=0)
        hi = v.max(axis=0)
    else:
        v = mesh_verts.astype(np.float64)
        lo = mesh_bounds[0].astype(np.float64)
        hi = mesh_bounds[1].astype(np.float64)
        c = np.zeros(3)

    # Centre XZ, anchor Y bottom at 0
    cx_b = (lo[0] + hi[0]) / 2.0
    cz_b = (lo[2] + hi[2]) / 2.0
    offset = np.array([cx_b, lo[1], cz_b])
    lo_c = lo - offset
    hi_c = hi - offset

    # Subsample vertices for projection (max 2000 for speed)
    if obb_R is not None:
        pts = v - offset
    else:
        pts = v - (c + offset)
    _MAX_PTS = 2000
    if len(pts) > _MAX_PTS:
        idx_sub = np.linspace(0, len(pts) - 1, _MAX_PTS, dtype=int)
        pts = pts[idx_sub]

    # Scale, rotate, translate
    pts_s = pts * scale
    pts_w = pts_s @ R.T
    pts_w[:, 1] -= pts_w[:, 1].min()  # ground contact
    pts_w = pts_w + pos[np.newaxis, :]

    # Project to pixels
    pxs, pys = [], []
    for pt in pts_w:
        px, py, zc = _project_vertex(pt, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
        if zc > NEAR_CLIP:
            pxs.append(px)
            pys.append(py)

    if len(pxs) < 2:
        s = float(scale[0])
        _na = 2 if wall_affinity == "back" else 0
        _fa = 0 if wall_affinity == "back" else 2
        gw = hi_c[_fa] - lo_c[_fa]
        return scale, gw * s, (hi_c[1] - lo_c[1]) * s, (hi_c[_na] - lo_c[_na]) * s

    proj_w_px = max(pxs) - min(pxs)
    proj_h_px = max(pys) - min(pys)

    w_ratio = (target_w_px / proj_w_px) if proj_w_px > 2 else None
    h_ratio = (target_h_px / proj_h_px) if proj_h_px > 2 else None

    if w_ratio is None and h_ratio is None:
        s = float(scale[0])
        gw = hi_c[0] - lo_c[0]; gh = hi_c[1] - lo_c[1]; gd = hi_c[2] - lo_c[2]
        return scale, gw * s, gh * s, gd * s

    _na = 2 if wall_affinity == "back" else 0
    _fa = 0 if wall_affinity == "back" else 2

    # ── Step 1: gentle uniform nudge to fit (geometric mean) ──────────────
    # The VLM/default size is the primary authority on physical scale.
    # _rescale_to_mask is only a GENTLE correction — cap at ±40% to avoid
    # over-shrinking objects when position/projection introduces error.
    ratios = []
    if w_ratio is not None:
        ratios.append(float(np.clip(w_ratio, 0.6, 1.6)))
    if h_ratio is not None:
        ratios.append(float(np.clip(h_ratio, 0.6, 1.6)))

    uniform_corr = float(np.sqrt(np.prod(ratios)))
    uniform_corr = float(np.clip(uniform_corr, 0.6, 1.6))

    if abs(uniform_corr - 1.0) > 0.02:
        scale = scale * uniform_corr
        print(f"  [mask_fit] gentle nudge ×{uniform_corr:.3f}  "
              f"(proj {proj_w_px:.0f}×{proj_h_px:.0f} → target {target_w_px:.0f}×{target_h_px:.0f})")

    # Step 2 removed: per-axis stretching distorts mesh proportions.
    # The uniform correction from Step 1 is sufficient.

    if obb_R is not None:
        ew = (hi[_fa] - lo[_fa]) * float(scale[_fa])
        eh = (hi[1] - lo[1]) * float(scale[1])
        ed = (hi[_na] - lo[_na]) * float(scale[_na])
    else:
        ew = (mesh_bounds[1][0] - mesh_bounds[0][0]) * float(scale[0])
        eh = (mesh_bounds[1][1] - mesh_bounds[0][1]) * float(scale[1])
        ed = (mesh_bounds[1][2] - mesh_bounds[0][2]) * float(scale[2])
    return scale, ew, eh, ed


def _mask_fill_ratio(out_root, seg_entry: dict) -> float:
    """Fraction of an object's bbox that its segmentation mask actually covers.

    A heavily-occluded piece comes back as a few slivers: office8's desk mask was
    1,593 px inside a 1050x651 box (0.2%). Silhouette-driven aspect fitting cannot
    mean anything at that density, so callers use this to decide whether the mask
    is allowed to dictate the mesh's proportions. Returns 1.0 when unknown, so an
    unreadable mask never silently disables the silhouette path.
    """
    try:
        import numpy as _np
        from PIL import Image as _Img
        mf = seg_entry.get("mask_file")
        box = seg_entry.get("box_px")
        if not mf or not box:
            return 1.0
        mp = Path(out_root) / "furniture" / "segmented" / mf
        if not mp.exists():
            return 1.0
        arr = _np.array(_Img.open(mp).convert("L")) > 127
        x1, y1, x2, y2 = [int(v) for v in box]
        area = max((x2 - x1) * (y2 - y1), 1)
        inside = int(arr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)].sum())
        return float(inside) / float(area)
    except Exception:
        return 1.0

def _align_and_scale_to_mask(
    mesh_verts: np.ndarray,
    mesh_bounds: np.ndarray,
    scale: np.ndarray,
    R: np.ndarray,
    pos: np.ndarray,
    bx1: float, by1: float, bx2: float, by2: float,
    cam_pos: np.ndarray,
    right_v: np.ndarray,
    up_c_v: np.ndarray,
    fwd_v: np.ndarray,
    fx: float, cx: float, cy: float,
    wall_affinity: str,
    back_flush: bool,
    room_w: float,
    room_d: float,
    n_passes: int = 2,
    existing_placements: list[dict] | None = None,
    obj_size_m: dict | None = None,
    exempt_collision_idxs: set[int] | None = None,
    is_low_surface: bool = False,
    vlm_size_locked: bool = False,
    mask_fill: float = 1.0,
    local_front: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Soft sanity check: nudge world position and scale if they're grossly
    off from the segmentation mask bbox.

    This is a GENTLE refinement — object correspondence constraints (wall
    affinity, facing, relative positions) are the primary authority.  Mask
    alignment just catches large mismatches (e.g. VLM scale way off).

    Returns (adjusted_pos, adjusted_scale, eff_w, eff_h, eff_d).
    """
    # width_m/depth_m are semantic labels (along-front vs perpendicular), not
    # fixed mesh axes.  mesh_bounds[...][0]/[...][2] are raw LOCAL X/Z extents
    # and this function used to always call axis-0 "width" and axis-2 "depth"
    # — true only when local_front points along Z (R never swaps X/Z under a
    # 0°/180° rotation).  A 90°/270° rotation (front along X, e.g. after an
    # [axis_swap] relabel) genuinely swaps which local axis ends up along-wall
    # in world space, so the label must follow local_front the same way
    # [front_size] already does: depth = extent along local_front's axis,
    # width = extent along the other one.
    _front_is_x = (local_front is not None
                   and abs(float(local_front[0])) > abs(float(local_front[2])))
    def _wd_from_local_xz(_lx: float, _lz: float) -> tuple[float, float]:
        """(local-X extent, local-Z extent) → (semantic width, semantic depth)."""
        return (_lz, _lx) if _front_is_x else (_lx, _lz)

    # Ablation wo_placement_refinement: disable silhouette mask-align entirely —
    # keep the VLM/back-projected position + scale as-is (no snapping to the
    # segmentation-mask bbox). Combined with vggt_refine=False (NO_REFINE also sets
    # it), this removes BOTH the silhouette AND the depth/VGGT refinement.
    if __import__("os").environ.get("SCENEWEAVE_ABLATE_NO_MASK_ALIGN") == "1":
        _s = np.asarray(scale, dtype=float).reshape(-1)
        if _s.size == 1:
            _s = np.repeat(_s, 3)
        _ext = np.asarray(mesh_bounds[1]) - np.asarray(mesh_bounds[0])
        _eh = float(_ext[1] * _s[1])
        _ew, _ed = _wd_from_local_xz(float(_ext[0] * _s[0]), float(_ext[2] * _s[2]))
        print("  [mask_align] ABLATION no-mask-align → keeping VLM/back-proj pos+scale (no silhouette)")
        return pos.copy(), (scale.copy() if hasattr(scale, "copy") else scale), _ew, _eh, _ed, []
    target_cx_px = (bx1 + bx2) / 2.0
    target_cy_px = (by1 + by2) / 2.0
    target_w_px  = float(bx2 - bx1)
    target_h_px  = float(by2 - by1)

    # Detect partial visibility: bbox touches image border → the object is
    # clipped and its visible silhouette is smaller than the real object.
    # In this case we still align position (towards the visible centre) but
    # skip scale fitting — the VLM-estimated size is more trustworthy than
    # the clipped mask.
    _img_w = cx * 2.0
    _img_h = cy * 2.0
    _border_margin = 5.0   # pixels
    _clip_left  = bx1 < _border_margin
    _clip_right = bx2 > _img_w - _border_margin
    _clip_top   = by1 < _border_margin
    _clip_bot   = by2 > _img_h - _border_margin
    _partial = _clip_left or _clip_top or _clip_right or _clip_bot
    # An edge-clipped object still has a RELIABLE silhouette in any dimension
    # that doesn't touch a border: a sofa cut off at the left frame edge has a
    # truncated width but its full height is visible.  We use that reliable
    # dimension to scale-fit (silhouette-authoritative) so the rendered object
    # matches the visible mask and its clipped side runs off-frame like the ref.
    _edge_partial    = _partial                 # touched a border (vs small-bbox)
    _width_reliable  = not (_clip_left or _clip_right)
    _height_reliable = not (_clip_top or _clip_bot)
    if _partial:
        print(f"  [mask_align] partial visibility detected "
              f"(bbox [{bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f}] "
              f"in {_img_w:.0f}×{_img_h:.0f}; reliable: "
              f"{'W' if _width_reliable else ''}{'H' if _height_reliable else ''}"
              f"{'none' if not (_width_reliable or _height_reliable) else ''})")

    # Small-bbox guard: when the target mask bbox itself is too small for
    # reliable size estimation (e.g. only the top of a partially-occluded
    # cabinet is visible at 102×33 px), treat it like partial visibility.
    # The size has already been borrowed from a reliable sibling — letting
    # mask_align scale this object down to match a 33-px-tall projection
    # would shrink it to a few centimetres tall.
    _target_area = float(bx2 - bx1) * float(by2 - by1)
    if not _partial and _target_area < _MIN_RELIABLE_BBOX_AREA_PX:
        _partial = True
        print(f"  [mask_align] small target bbox detected "
              f"(area={_target_area:.0f} px² < {_MIN_RELIABLE_BBOX_AREA_PX}) "
              f"— position only, skip scale")

    if target_w_px < 2 or target_h_px < 2:
        s = float(scale[0])
        eh = (mesh_bounds[1][1] - mesh_bounds[0][1]) * s
        ew, ed = _wd_from_local_xz((mesh_bounds[1][0] - mesh_bounds[0][0]) * s,
                                    (mesh_bounds[1][2] - mesh_bounds[0][2]) * s)
        return pos.copy(), scale.copy(), ew, eh, ed

    # ── Build world-space vertices at current pose ────────────────────────
    v = mesh_verts.astype(np.float64)
    lo = mesh_bounds[0].astype(np.float64)
    hi = mesh_bounds[1].astype(np.float64)
    cx_b = (lo[0] + hi[0]) / 2.0
    cz_b = (lo[2] + hi[2]) / 2.0
    offset = np.array([cx_b, lo[1], cz_b])
    pts = v - offset
    _MAX_PTS = 2000
    if len(pts) > _MAX_PTS:
        idx_sub = np.linspace(0, len(pts) - 1, _MAX_PTS, dtype=int)
        pts = pts[idx_sub]

    pos_out = pos.copy()
    scale_out = scale.copy()
    _initial_scale = float(scale[0])  # track for cumulative cap

    # ── Pre-check: which objects are we already colliding with? ─────────────
    # If the object starts colliding with some neighbours, those are "known"
    # collisions.  Mask_align is free to keep (or worsen) overlap with them
    # while moving toward the silhouette — only NEW collisions with OTHER
    # objects are blocked.
    _initial_collision_idxs: set[int] = set()
    # Per-pair AABB overlap area at the initial position.  Tracked so that
    # subsequent silhouette-driven shifts can be accepted only when they do
    # NOT increase the overlap with an "initial collision" pair.  Without
    # this guard, e.g. a chair that starts barely-overlapping a coffee
    # table (~6 cm strip) gets pushed by mask_align to land FULLY inside
    # the table's footprint, because the original code unconditionally
    # ignored all shifts that grew an "initial collision".
    _initial_pair_overlap_area: dict[int, float] = {}
    if existing_placements and obj_size_m:
        _lhx0, _lhz0 = (x / 2.0 for x in _wd_from_local_xz(
            obj_size_m.get("width_m", 0.5), obj_size_m.get("depth_m", 0.5)))
        _hx0 = abs(float(R[0][0])) * _lhx0 + abs(float(R[0][2])) * _lhz0
        _hz0 = abs(float(R[2][0])) * _lhx0 + abs(float(R[2][2])) * _lhz0
        for _ep0 in existing_placements:
            if _ep0.get("is_carpet") or _ep0.get("wall_mounted"):
                continue
            _ep0_sm = _ep0.get("size_m", {})
            _ep0_front = _ep0.get("_local_front", [0, 0, -1])
            _ep0_is_x = abs(float(_ep0_front[0])) > abs(float(_ep0_front[2]))
            _ep0_w2, _ep0_d2 = (_ep0_sm.get("width_m", 0.5) / 2.0,
                                 _ep0_sm.get("depth_m", 0.5) / 2.0)
            _ep0_hx, _ep0_hz = ((_ep0_d2, _ep0_w2) if _ep0_is_x
                                 else (_ep0_w2, _ep0_d2))
            _ep0_R = np.array(_ep0.get("rotation_3x3", np.eye(3).tolist()))
            _ep0_hx_r = abs(float(_ep0_R[0][0])) * _ep0_hx + abs(float(_ep0_R[0][2])) * _ep0_hz
            _ep0_hz_r = abs(float(_ep0_R[2][0])) * _ep0_hx + abs(float(_ep0_R[2][2])) * _ep0_hz
            _ov_x0 = (_hx0 + _ep0_hx_r) - abs(pos[0] - _ep0["position_m"][0])
            _ov_z0 = (_hz0 + _ep0_hz_r) - abs(pos[2] - _ep0["position_m"][2])
            if _ov_x0 > 0.02 and _ov_z0 > 0.02:
                _ep0_idx = _ep0.get("index", -1)
                _initial_collision_idxs.add(_ep0_idx)
                _initial_pair_overlap_area[_ep0_idx] = float(_ov_x0 * _ov_z0)
    # Merge in any colliders from init_slide (objects that were colliding
    # before init_slide pushed this object away)
    if exempt_collision_idxs:
        _initial_collision_idxs |= exempt_collision_idxs
    if _initial_collision_idxs:
        print(f"  [mask_align] initial collisions with idx={_initial_collision_idxs} "
              f"— these will be ignored during collision blocking")

    # Iterate until convergence: keep refining position + scale until the
    # projected bbox matches the target closely, or no progress is made.
    _max_passes = max(n_passes, 8)  # generous upper bound to prevent infinite loop
    for _pass in range(_max_passes):
        pts_s = pts * scale_out
        pts_w = pts_s @ R.T
        pts_w[:, 1] -= pts_w[:, 1].min()
        pts_w = pts_w + pos_out[np.newaxis, :]

        pxs, pys = [], []
        for pt in pts_w:
            px, py, zc = _project_vertex(pt, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
            if zc > NEAR_CLIP:
                pxs.append(px)
                pys.append(py)

        if len(pxs) < 2:
            break

        proj_cx = (min(pxs) + max(pxs)) / 2.0
        proj_cy = (min(pys) + max(pys)) / 2.0
        proj_w  = max(pxs) - min(pxs)
        proj_h  = max(pys) - min(pys)

        # ── Position alignment: ray-floor back-projection ─────────────
        # For fully visible objects: align bottom-centre (floor contact point).
        # For partially clipped objects: align on the NON-CLIPPED horizontal
        # edge.  The clipped side extends beyond the image boundary, so the
        # visible centre is NOT the true object centre.  Using the non-clipped
        # edge gives a more accurate world-space anchor.
        #   - clipped at left (bx1≈0): align right edges
        #   - clipped at right (bx2≈img_w): align left edges
        #   - clipped at both or neither: align centres (default)
        _clip_left  = bx1 < _border_margin
        _clip_right = bx2 > _img_w - _border_margin

        # Horizontal anchor: use the non-clipped horizontal edge.
        # The clipped side extends beyond the image, so visible centre ≠ true centre.
        if _clip_left and not _clip_right:
            proj_bot_cx   = float(max(pxs))   # right edge of projection
            target_bot_cx = float(bx2)         # right edge of mask
        elif _clip_right and not _clip_left:
            proj_bot_cx   = float(min(pxs))   # left edge of projection
            target_bot_cx = float(bx1)         # left edge of mask
        else:
            proj_bot_cx   = proj_cx            # horizontal centre
            target_bot_cx = target_cx_px       # horizontal centre of mask

        # Vertical anchor: always use the BOTTOM edge (floor contact point).
        # Back-projecting the bottom pixel to y=0 gives a reliable world-Z.
        # Top-edge back-projection to the floor plane gives wrong Z because
        # the top of an object is not at floor level.
        proj_bot_cy   = float(max(pys))        # bottom of projected bbox
        target_bot_cy = float(by2)             # bottom of mask bbox
        ray_proj = _backproject_pixel(
            proj_bot_cx, proj_bot_cy, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
        ray_tgt = _backproject_pixel(
            target_bot_cx, target_bot_cy, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
        floor_proj = _ray_floor_intersect(cam_pos, ray_proj)
        floor_tgt = _ray_floor_intersect(cam_pos, ray_tgt)

        if floor_proj is not None and floor_tgt is not None:
            shift_x = float(floor_tgt[0] - floor_proj[0])
            shift_z = float(floor_tgt[2] - floor_proj[2])
        else:
            # Fallback: use depth-based camera-axis conversion (rare edge case)
            dpx = target_bot_cx - proj_bot_cx
            dpy = target_bot_cy - proj_bot_cy
            to_obj = pos_out - cam_pos
            depth = float(np.dot(to_obj, fwd_v))
            if depth < 0.1:
                depth = 1.0
            dx_world = (dpx / fx) * depth
            dz_world = -(dpy / fx) * depth
            shift_3d = dx_world * right_v - dz_world * up_c_v
            shift_x = float(shift_3d[0])
            shift_z = float(shift_3d[2])

        # Wall-affinity is INITIAL placement only — post-placement (here) is
        # free to move the object forward off the wall to match the silhouette,
        # but must not push it deeper into the wall.  This lets a sofa marked
        # "back-wall" come forward when the photo / VGGT depth says it's
        # actually a bit into the room, while still preventing it from being
        # silently shoved through the wall.
        if wall_affinity == "back":
            if shift_z < 0.0:   # negative shift_z = toward back wall (smaller z)
                shift_z = 0.0
        elif wall_affinity == "left":
            if shift_x < 0.0:   # negative shift_x = toward left wall (smaller x)
                shift_x = 0.0
        elif wall_affinity == "right":
            if shift_x > 0.0:   # positive shift_x = toward right wall (larger x)
                shift_x = 0.0

        # Bottom-clipped floor objects: the visible bbox bottom is the image
        # edge, NOT the chair's actual floor contact point (which is below the
        # frame). Back-projecting the clipped bottom-edge to the floor gives a
        # wrong depth, which would pull the object backward to satisfy the
        # bogus anchor. Trust the original back-projection placement instead
        # and cap any mask_align shift to a small magnitude (X for centring,
        # Z near-zero to avoid spurious depth changes).
        # A LOW surface (coffee/side table) whose mask is bottom-clipped has its
        # floor-contact roughly AT the clipped edge, so back-projecting it gives a
        # usable depth — let mask_align move it to its silhouette box.  The cap
        # below is for TALL objects (chairs) whose real base is far below the
        # frame, where the clipped bottom would back-project to a bogus near depth.
        _clip_bottom = by2 > _img_h - _border_margin
        if (_partial and _clip_bottom and not is_low_surface
                and wall_affinity not in ("back", "left", "right")):
            _CAP_X_PARTIAL = 0.15
            _CAP_Z_PARTIAL = 0.05
            _orig_x, _orig_z = shift_x, shift_z
            shift_x = float(np.clip(shift_x, -_CAP_X_PARTIAL, _CAP_X_PARTIAL))
            shift_z = float(np.clip(shift_z, -_CAP_Z_PARTIAL, _CAP_Z_PARTIAL))
            if abs(_orig_x - shift_x) > 0.01 or abs(_orig_z - shift_z) > 0.01:
                print(f"  [mask_align] bottom-clipped: shift "
                      f"({_orig_x:.3f}, {_orig_z:.3f}) → "
                      f"({shift_x:.3f}, {shift_z:.3f}) "
                      f"(trusting back-projection depth)")

        # No artificial shift cap — silhouette position is authoritative.
        # Collision check below is the only constraint on movement.

        # Collision helper: check if placing the object at (cx, cz) with
        # optional scale override would collide with any non-exempt object.
        # Current object's hull (may be None for wall-mounted / carpets)
        _my_hull = None
        if obj_size_m is not None:
            # Reconstruct hull from obj_size_m as a rectangle in local space
            # (used when _xz_hull_pts is not available on the candidate object)
            pass  # hull comes from existing_placements[self] if available

        def _check_collision_at(cx: float, cz: float,
                                s_override: np.ndarray | None = None) -> bool:
            """Return True if (cx, cz) collides with any non-exempt object.
            Uses SAT on hull footprints when available; AABB fallback."""
            if not existing_placements or not obj_size_m:
                return False
            _s_ratio = (float(s_override[0]) / _initial_scale
                        if s_override is not None else
                        float(scale_out[0]) / _initial_scale)
            _sc_candidate = np.array([_s_ratio, _s_ratio, _s_ratio], dtype=np.float64)
            # Footprint of the object being aligned — used for big-anchor priority.
            _my_area = (float(obj_size_m.get("width_m", 0.5))
                        * float(obj_size_m.get("depth_m", 0.5)))

            for _ep in existing_placements:
                if _ep.get("is_carpet") or _ep.get("wall_mounted"):
                    continue
                # Big-anchor priority: a much larger piece (e.g. a sofa reaching
                # its silhouette / wall) must not be held off-target by a small
                # movable item. Let it pass through — the smaller object yields in
                # the later collision-resolution pass, which anchors big wall-flush
                # pieces first. Without this, the sofa stays stuck off its mask
                # ("shift blocked by collision" every pass).
                _ep_area = (float(_ep.get("size_m", {}).get("width_m", 0.5))
                            * float(_ep.get("size_m", {}).get("depth_m", 0.5)))
                if _my_area >= 0.8 and _my_area >= 2.5 * max(_ep_area, 1e-3):
                    continue
                _ep_idx = _ep.get("index", -1)
                if _ep_idx in _initial_collision_idxs:
                    # Tolerated initial collision — but reject shifts that
                    # GROW the overlap area beyond what we started with.
                    # AABB-based area check (cheaper than SAT polygon area).
                    _ep_sm_g = _ep.get("size_m", {})
                    _ep_front_g = _ep.get("_local_front", [0, 0, -1])
                    _ep_isx_g = abs(float(_ep_front_g[0])) > abs(float(_ep_front_g[2]))
                    _ep_w2g, _ep_d2g = (_ep_sm_g.get("width_m", 0.5) / 2.0,
                                         _ep_sm_g.get("depth_m", 0.5) / 2.0)
                    _ep_hx_lg, _ep_hz_lg = ((_ep_d2g, _ep_w2g) if _ep_isx_g
                                             else (_ep_w2g, _ep_d2g))
                    _ep_Rg = np.array(_ep.get("rotation_3x3",
                                              np.eye(3).tolist()))
                    _ep_hx_g = (abs(float(_ep_Rg[0][0])) * _ep_hx_lg
                                + abs(float(_ep_Rg[0][2])) * _ep_hz_lg)
                    _ep_hz_g = (abs(float(_ep_Rg[2][0])) * _ep_hx_lg
                                + abs(float(_ep_Rg[2][2])) * _ep_hz_lg)
                    _my_lhx_g, _my_lhz_g = _wd_from_local_xz(
                        (obj_size_m or {}).get("width_m", 0.5) / 2.0,
                        (obj_size_m or {}).get("depth_m", 0.5) / 2.0)
                    _my_hx_g = (abs(float(R[0][0])) * _my_lhx_g
                                + abs(float(R[0][2])) * _my_lhz_g)
                    _my_hz_g = (abs(float(R[2][0])) * _my_lhx_g
                                + abs(float(R[2][2])) * _my_lhz_g)
                    _ov_x_now = (_my_hx_g + _ep_hx_g) - abs(cx - _ep["position_m"][0])
                    _ov_z_now = (_my_hz_g + _ep_hz_g) - abs(cz - _ep["position_m"][2])
                    _area_now = max(0.0, _ov_x_now) * max(0.0, _ov_z_now)
                    _area_init = _initial_pair_overlap_area.get(_ep_idx, _area_now)
                    # 5 cm² of growth tolerated for FP noise.
                    if _area_now > _area_init + 0.005:
                        return True
                    continue
                _ep_hull = _ep.get("_xz_hull_pts")
                _ep_R    = np.array(_ep.get("rotation_3x3", np.eye(3).tolist()),
                                    dtype=np.float64)
                _ep_sc   = np.array(_ep.get("scale", [1, 1, 1]), dtype=np.float64)
                if _ep_hull is not None:
                    # Build a rectangular hull for the candidate object from obj_size_m
                    # (in local space, centred at origin).  obj_size_m is semantic
                    # (width=perpendicular-to-front, depth=along-front) — undo the
                    # [axis_swap] relabel to get raw local-X/Z half-extents before
                    # building the local-frame rect that R below will rotate.
                    _hw, _hd = (x / 2.0 for x in _wd_from_local_xz(
                        obj_size_m.get("width_m", 0.5), obj_size_m.get("depth_m", 0.5)))
                    _my_rect = np.array([[-_hw, -_hd], [_hw, -_hd],
                                          [_hw,  _hd], [-_hw,  _hd]],
                                         dtype=np.float64)
                    # Scale ratio already baked into rect via obj_size_m,
                    # so candidate scale for rect is identity
                    _collides = _poly_collides_sat(
                        _my_rect, cx, cz, R, np.ones(3),
                        _ep_hull, _ep["position_m"][0], _ep["position_m"][2],
                        _ep_R, _ep_sc,
                    )
                    if _collides:
                        return True
                else:
                    # AABB fallback for objects without hull.
                    # width_m/depth_m are semantic (along-front vs perpendicular)
                    # labels, not raw local-X/local-Z extents — undo the same
                    # swap [axis_swap]/_wd_from_local_xz applies before combining
                    # with R, which operates on raw local axes.
                    _sm = obj_size_m
                    _lhx, _lhz = _wd_from_local_xz(_sm.get("width_m", 0.5) / 2.0,
                                                    _sm.get("depth_m", 0.5) / 2.0)
                    _my_hx = abs(float(R[0][0])) * _lhx + abs(float(R[0][2])) * _lhz
                    _my_hz = abs(float(R[2][0])) * _lhx + abs(float(R[2][2])) * _lhz
                    _ep_sm = _ep.get("size_m", {})
                    _ep_front = _ep.get("_local_front", [0, 0, -1])
                    _ep_front_is_x = abs(float(_ep_front[0])) > abs(float(_ep_front[2]))
                    _ep_w2, _ep_d2 = (_ep_sm.get("width_m", 0.5) / 2.0,
                                       _ep_sm.get("depth_m", 0.5) / 2.0)
                    _ep_hx_l, _ep_hz_l = ((_ep_d2, _ep_w2) if _ep_front_is_x
                                           else (_ep_w2, _ep_d2))
                    _ep_R2 = np.array(_ep.get("rotation_3x3", np.eye(3).tolist()))
                    _ep_hx = (abs(float(_ep_R2[0][0])) * _ep_hx_l
                              + abs(float(_ep_R2[0][2])) * _ep_hz_l)
                    _ep_hz = (abs(float(_ep_R2[2][0])) * _ep_hx_l
                              + abs(float(_ep_R2[2][2])) * _ep_hz_l)
                    _ov_x = (_my_hx + _ep_hx) - abs(cx - _ep["position_m"][0])
                    _ov_z = (_my_hz + _ep_hz) - abs(cz - _ep["position_m"][2])
                    if _ov_x > 0.02 and _ov_z > 0.02:
                        return True
            return False

        if abs(shift_x) > 0.005 or abs(shift_z) > 0.005:
            _prev_x, _prev_z = pos_out[0], pos_out[2]
            _new_x = float(np.clip(pos_out[0] + shift_x, 0.1, room_w - 0.1))
            _new_z = float(np.clip(pos_out[2] + shift_z, 0.1, room_d - 0.1))

            _applied = False
            if not _check_collision_at(_new_x, _new_z):
                pos_out[0] = _new_x
                pos_out[2] = _new_z
                _applied = True
                print(f"  [mask_align] pass {_pass}: shift ({shift_x:.3f}, {shift_z:.3f}) m "
                      f"→ pos=({pos_out[0]:.3f}, {pos_out[2]:.3f})")
            else:
                # Full shift blocked — try X-only, then Z-only
                if abs(shift_x) > 0.005 and not _check_collision_at(_new_x, _prev_z):
                    pos_out[0] = _new_x
                    _applied = True
                    print(f"  [mask_align] pass {_pass}: X-only shift ({shift_x:.3f}, 0) m "
                          f"→ pos=({pos_out[0]:.3f}, {pos_out[2]:.3f})  [Z blocked by collision]")
                if abs(shift_z) > 0.005 and not _check_collision_at(pos_out[0], _new_z):
                    pos_out[2] = _new_z
                    _applied = True
                    print(f"  [mask_align] pass {_pass}: Z-only shift (0, {shift_z:.3f}) m "
                          f"→ pos=({pos_out[0]:.3f}, {pos_out[2]:.3f})  [X blocked by collision]")
                if not _applied:
                    print(f"  [mask_align] pass {_pass}: shift blocked by collision "
                          f"— skipping position, still applying scale")

        # ── Edge-partial scale fit (silhouette-authoritative) ───────────────
        # An object clipped at a frame edge keeps a reliable silhouette in any
        # UNclipped dimension.  Scale UNIFORMLY (proportions preserved) by that
        # dimension's mask/projection ratio so the rendered silhouette matches
        # the visible mask — the clipped side then runs off-frame exactly as in
        # the reference.  Skips small-bbox "partial" (size borrowed from sibling).
        if (proj_h > 2 and proj_w > 2 and _edge_partial
                and (_height_reliable or _width_reliable)):
            _rel_ratio = (target_h_px / proj_h if _height_reliable
                          else target_w_px / proj_w)
            _rel_corr = float(np.clip(_rel_ratio, 0.75, 1.25))
            if abs(_rel_corr - 1.0) > 0.02:
                _cum = (float(scale_out[0]) * _rel_corr) / _initial_scale
                if 0.4 <= _cum <= 1.6:
                    scale_out = scale_out.copy() * _rel_corr
                    print(f"  [mask_align] pass {_pass}: partial uniform scale "
                          f"×{_rel_corr:.3f} (reliable "
                          f"{'H' if _height_reliable else 'W'} ratio "
                          f"{_rel_ratio:.2f}; clipped side off-frame)")
                else:
                    print(f"  [mask_align] pass {_pass}: partial scale ×{_rel_corr:.3f} "
                          f"blocked — cumulative {_cum:.2f}× exceeds ±60% cap")
            continue

        # ── Scale fit: projected height vs mask height ───────────────────
        # For partially clipped objects: skip scale-UP (visible bbox underestimates
        # real size) but allow scale-DOWN (projected bbox clearly too large means
        # the object is genuinely oversized regardless of clipping).
        if proj_h > 2 and proj_w > 2 and not _partial:
            # Silhouette mask is the primary scale authority — use wider
            # per-pass range (±25%) and cumulative cap (±60%).
            h_ratio = float(np.clip(target_h_px / proj_h, 0.75, 1.25))
            w_ratio = float(np.clip(target_w_px / proj_w, 0.75, 1.25))
            # Occlusion detection (GROW branch only): when the mask is
            # taller than the projection but narrower, the height
            # under-reports because the top is visible but a sibling
            # object hides the sides.  Grow Y to match the height
            # without enlarging the FOOTPRINT (uniform growth would
            # push the table past the wall on the corner).
            #
            # IMPORTANT: this branch fires ONLY when h_ratio > 1.05
            # (mask says we need to grow taller).  For shrink cases
            # (h_ratio ≤ 1.0) we want the W/D to shrink too — preserving
            # an oversized footprint just because height happens to be
            # close was the bug behind "cornermost coffee_table bottom
            # too big" (overlapping the corner sofa by 20%).
            _grow_height = h_ratio > 1.05
            # Y-only stretch is only valid when the camera sees the object
            # face-on (back wall).  For side-wall objects (left/right) the
            # camera views them obliquely, so h_ratio >> w_ratio is normal
            # perspective foreshortening — NOT a sign that height under-reports.
            # Applying Y-only stretch there makes beds/cabinets unrealistically tall.
            _width_underreports = (_grow_height
                                   and w_ratio < h_ratio * 0.85
                                   and wall_affinity not in ("left", "right"))
            if _width_underreports:
                _y_corr = float(np.clip(h_ratio, 0.75, 1.25))
                if abs(_y_corr - 1.0) > 0.02:
                    _cum_y = float(scale_out[1]) * _y_corr / _initial_scale
                    if _cum_y < 0.4 or _cum_y > 1.6:
                        print(f"  [mask_align] pass {_pass}: Y scale ×{_y_corr:.3f} "
                              f"blocked — cumulative Y {_cum_y:.2f}× exceeds ±60% cap")
                    else:
                        scale_out = scale_out.copy()
                        scale_out[1] = float(scale_out[1]) * _y_corr
                        print(f"  [mask_align] pass {_pass}: Y-only scale "
                              f"×{_y_corr:.3f} (h_ratio={h_ratio:.2f}, "
                              f"w_ratio={w_ratio:.2f}) — height under-reports, "
                              f"footprint kept fixed")
                # Skip the uniform-scale branch below.
                continue
            # Shrink branch with mismatched ratios: prefer shrinking the
            # FOOTPRINT (X/Z) over the height — matches the "bottom too
            # big" feedback for occluded cornermost items.
            _shrink_footprint = (h_ratio < 1.0
                                 and w_ratio < 1.0
                                 and w_ratio < h_ratio * 0.85)
            if _shrink_footprint:
                _xz_corr = float(np.clip(w_ratio, 0.75, 1.25))
                if abs(_xz_corr - 1.0) > 0.02:
                    _cum_xz = float(scale_out[0]) * _xz_corr / _initial_scale
                    if _cum_xz < 0.4 or _cum_xz > 1.6:
                        print(f"  [mask_align] pass {_pass}: XZ scale ×{_xz_corr:.3f} "
                              f"blocked — cumulative XZ {_cum_xz:.2f}× exceeds ±60% cap")
                    else:
                        # Test that shrinking footprint doesn't introduce a
                        # new collision on the other side (it shouldn't, but
                        # be safe).
                        _candidate_scale = scale_out.copy()
                        _candidate_scale[0] = float(scale_out[0]) * _xz_corr
                        _candidate_scale[2] = float(scale_out[2]) * _xz_corr
                        scale_out = _candidate_scale
                        print(f"  [mask_align] pass {_pass}: XZ-only shrink "
                              f"×{_xz_corr:.3f} (h_ratio={h_ratio:.2f}, "
                              f"w_ratio={w_ratio:.2f}) — width over-reports, "
                              f"height kept fixed")
                continue
            # ── Aspect-mismatch anisotropic fit (silhouette-authoritative) ──
            # When the mask's width/height ratios diverge strongly (one wants
            # WIDER, the other SHORTER) the 3D-gen mesh's intrinsic aspect ratio
            # is wrong for this object — the classic near-CUBE returned for a
            # wide-shallow sideboard/cabinet.  The uniform path below takes
            # sqrt(w·h) and the opposing ratios CANCEL (≈1.0), so the silhouette
            # never matches and the piece reads as an oversized block.  Stretch
            # X and Y toward the mask independently.  Restricted to back-wall
            # (face-on) objects, where projected W/H map cleanly onto the
            # object's X/Y axes — side-wall pieces are viewed obliquely and a
            # large w/h divergence there is just normal perspective foreshortening.
            _aspect_div = max(w_ratio / max(h_ratio, 1e-6),
                              h_ratio / max(w_ratio, 1e-6))
            # When the VLM size critique CONVERGED, the silhouette is no longer
            # the better authority: for a heavily-occluded object the mask is a
            # handful of fragments, and this aspect-fit then stretches the mesh
            # onto that broken outline. office8's desk converged to 1.55×0.65 m
            # and was immediately re-stretched back to 0.90×1.46 m here, which
            # is what produced the 69 cm overlap with the office chair.
            # Skip ONLY the anisotropic stretch — fall through to the uniform
            # scale + position shift below (an early `continue` here would waste
            # the whole pass and leave the object unaligned).
            _MASK_FILL_MIN = float(os.environ.get("SCENEWEAVE_MASK_FILL_MIN", "0.05"))
            _sparse_mask = mask_fill < _MASK_FILL_MIN
            _skip_aspect = ((vlm_size_locked or _sparse_mask)
                            and wall_affinity == "back" and _aspect_div > 1.18)
            if _skip_aspect and _pass == 0:
                _why = (f"mask fills only {mask_fill:.1%} of its bbox "
                        f"(<{_MASK_FILL_MIN:.0%}) — too fragmentary to dictate aspect"
                        if _sparse_mask else
                        "VLM size converged and is authoritative")
                print(f"  [mask_align] aspect-fit disabled — {_why}; silhouette wanted "
                      f"aspect_div={_aspect_div:.2f}; uniform scale + shift still applied")
            if (not _skip_aspect) and wall_affinity == "back" and _aspect_div > 1.18:
                _wc = float(np.clip(w_ratio, 0.7, 1.25))
                _hc = float(np.clip(h_ratio, 0.7, 1.25))
                _cand = scale_out.copy()
                _cum_x = float(scale_out[0]) * _wc / _initial_scale
                _cum_y = float(scale_out[1]) * _hc / _initial_scale
                if 0.45 <= _cum_x <= 2.4 and 0.45 <= _cum_y <= 2.4:
                    _cand[0] = float(scale_out[0]) * _wc
                    _cand[1] = float(scale_out[1]) * _hc
                    # A low, wide cabinet is also shallow — pull depth down with
                    # the (shorter) height so the cube footprint stops poking at
                    # the camera; never grow it.
                    _cand[2] = min(float(scale_out[2]), _cand[1])
                    if not (_wc > 1.0 and _check_collision_at(
                            pos_out[0], pos_out[2], s_override=_cand)):
                        scale_out = _cand
                        print(f"  [mask_align] pass {_pass}: aspect-fit "
                              f"X×{_wc:.3f} Y×{_hc:.3f} Z↓ (mesh aspect ≠ "
                              f"silhouette; proj {proj_w:.0f}×{proj_h:.0f} → "
                              f"target {target_w_px:.0f}×{target_h_px:.0f})")
                        continue
                    print(f"  [mask_align] pass {_pass}: aspect-fit blocked "
                          f"by collision")
                else:
                    print(f"  [mask_align] pass {_pass}: aspect-fit blocked — "
                          f"cumulative X {_cum_x:.2f}× / Y {_cum_y:.2f}× off cap")
            uniform_corr = float(np.clip(np.sqrt(h_ratio * w_ratio), 0.75, 1.25))
            if abs(uniform_corr - 1.0) > 0.02:
                _new_s = float(scale_out[0]) * uniform_corr
                _cum_ratio = _new_s / _initial_scale
                if _cum_ratio < 0.4 or _cum_ratio > 1.6:
                    print(f"  [mask_align] pass {_pass}: scale ×{uniform_corr:.3f} "
                          f"blocked — cumulative {_cum_ratio:.2f}× exceeds ±60% cap")
                else:
                    _candidate_scale = scale_out * uniform_corr
                    if uniform_corr > 1.0 and _check_collision_at(
                            pos_out[0], pos_out[2], s_override=_candidate_scale):
                        print(f"  [mask_align] pass {_pass}: scale ×{uniform_corr:.3f} "
                              f"blocked — would cause collision")
                    else:
                        scale_out = _candidate_scale
                        print(f"  [mask_align] pass {_pass}: scale ×{uniform_corr:.3f}  "
                              f"(proj {proj_w:.0f}×{proj_h:.0f} → target {target_w_px:.0f}×{target_h_px:.0f})")

        # Convergence check: if position and scale barely changed, stop early.
        _dpx = abs(proj_bot_cx - target_bot_cx)
        _dpy = abs(proj_bot_cy - target_bot_cy)
        _dw = abs(proj_w - target_w_px)
        _dh = abs(proj_h - target_h_px)
        if _dpx < 5 and _dpy < 5 and _dw < 10 and _dh < 10:
            break

    # Per-axis stretch is intentionally NOT done here.  Uniform scaling
    # preserves mesh proportions; non-uniform stretch distorts symmetric
    # objects (coffee tables, chairs) and compounds across multiple
    # mask_align calls.  The mask bbox is an approximation — aspect
    # mismatch is expected and acceptable.

    # Final alignment error diagnostic
    pts_s = pts * scale_out
    pts_w = pts_s @ R.T
    pts_w[:, 1] -= pts_w[:, 1].min()
    pts_w = pts_w + pos_out[np.newaxis, :]
    _fpxs, _fpys = [], []
    for pt in pts_w:
        px, py, zc = _project_vertex(pt, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
        if zc > NEAR_CLIP:
            _fpxs.append(px)
            _fpys.append(py)
    if len(_fpxs) >= 2:
        _fcx = (min(_fpxs) + max(_fpxs)) / 2.0
        _fcy = (min(_fpys) + max(_fpys)) / 2.0
        _fw  = max(_fpxs) - min(_fpxs)
        _fh  = max(_fpys) - min(_fpys)
        print(f"  [mask_align] final: proj bbox {_fw:.0f}×{_fh:.0f} @ ({_fcx:.0f},{_fcy:.0f})  "
              f"target {target_w_px:.0f}×{target_h_px:.0f} @ ({target_cx_px:.0f},{target_cy_px:.0f})  "
              f"err: dx={abs(_fcx - target_cx_px):.0f}px dy={abs(_fcy - target_cy_px):.0f}px "
              f"dw={abs(_fw - target_w_px):.0f}px dh={abs(_fh - target_h_px):.0f}px")

    # Compute final effective dims (per-axis).  Keep the raw local-X/local-Z
    # extents (_lx/_lz) for the wall-flush projection below — that math
    # combines them with R's actual matrix entries and must NOT go through
    # the width/depth semantic swap, only the returned ew/ed should.
    eh = (mesh_bounds[1][1] - mesh_bounds[0][1]) * float(scale_out[1])
    _lx = (mesh_bounds[1][0] - mesh_bounds[0][0]) * float(scale_out[0])
    _lz = (mesh_bounds[1][2] - mesh_bounds[0][2]) * float(scale_out[2])
    ew, ed = _wd_from_local_xz(_lx, _lz)

    # ── Re-enforce wall-flush perpendicular position ──────────────────────
    # Scale changes shift the mesh half-extent, so the perpendicular
    # coordinate (distance from wall) must be recalculated.  Use the
    # true mesh bounds (_lx/_lz) rotated into world space — same source of
    # truth as the render/wall_snap step, avoiding drift from vertex subsampling.
    if wall_affinity in ("back", "left", "right"):
        _local_hx = _lx / 2.0
        _local_hz = _lz / 2.0
        _whx = abs(float(R[0][0])) * _local_hx + abs(float(R[0][2])) * _local_hz
        _whz = abs(float(R[2][0])) * _local_hx + abs(float(R[2][2])) * _local_hz
        _gap = 0.01 if back_flush else _WALL_GAP
        if wall_affinity == "back":
            new_z = _whz + _gap
            if abs(pos_out[2] - new_z) > 0.01:
                print(f"  [mask_align] wall re-flush: Z {pos_out[2]:.3f} → {new_z:.3f} (back)")
                pos_out[2] = new_z
        elif wall_affinity == "left":
            new_x = _whx + _gap
            if abs(pos_out[0] - new_x) > 0.01:
                print(f"  [mask_align] wall re-flush: X {pos_out[0]:.3f} → {new_x:.3f} (left)")
                pos_out[0] = new_x
        elif wall_affinity == "right":
            new_x = room_w - _whx - _gap
            if abs(pos_out[0] - new_x) > 0.01:
                print(f"  [mask_align] wall re-flush: X {pos_out[0]:.3f} → {new_x:.3f} (right)")
                pos_out[0] = new_x

    # Return final alignment error so callers can decide whether scene_review
    # should override mask_align's position.
    _final_err_px = (abs(_fcx - target_cx_px), abs(_fcy - target_cy_px)) if len(_fpxs) >= 2 else (999.0, 999.0)
    return pos_out, scale_out, ew, eh, ed, _final_err_px


def _find_base_y(verts: np.ndarray, quiet: bool = False) -> float:
    """Find the Y level where the object's actual base starts.

    generated meshes often have a spike/point at the bottom that doesn't represent
    the real ground contact.  This function scans Y levels from the bottom and
    finds where the XZ cross-section first widens to form a proper base.
    Vertices below that level are spike artifacts.

    Returns the Y value to use as the floor anchor (base level, not Y_min).
    """
    y_min = float(verts[:, 1].min())
    y_range = float(verts[:, 1].max() - y_min)
    if y_range < 1e-6:
        return y_min

    obj_spread = max(np.ptp(verts[:, 0]), np.ptp(verts[:, 2]))
    if obj_spread < 1e-6:
        return y_min

    # Scan from bottom in 2% steps — find where XZ spread first exceeds 15%
    for pct in range(0, 30, 2):
        y_level = y_min + (pct / 100.0) * y_range
        below = verts[verts[:, 1] <= y_level]
        if len(below) < 3:
            continue
        xz_spread = max(np.ptp(below[:, 0]), np.ptp(below[:, 2]))
        if xz_spread >= 0.15 * obj_spread:
            if pct > 0:
                # Base starts above Y_min — there's a spike below
                base_y = y_min + (pct / 100.0) * y_range
                if not quiet:
                    print(f"  [base_y] spike trimmed: base at {pct}% "
                          f"({(pct/100.0)*y_range:.3f} above Y_min)")
                return base_y
            return y_min   # base starts at bottom — no spike

    # No clear base found — just use Y_min
    return y_min


def _apply_floor_transform(
    mesh_verts: np.ndarray,
    mesh_bounds: np.ndarray,
    scale: np.ndarray,
    R: np.ndarray,
    pos: np.ndarray,
    wall_affinity: str = "centre",
    room_w: float | None = None,
    back_flush: bool = False,
    obb_R: np.ndarray | None = None,
    skip_wall_snap: bool = False,
    quiet: bool = False,
    skip_level_base: bool = False,
) -> np.ndarray:
    """Transform mesh vertices so the object sits level on the floor at pos.

    Steps:
      1. [flush] OBB de-tilt → axis-aligned verts.  [other] Centre in XZ.
      2. Scale (non-uniform for flush, uniform otherwise).
      3. Rotate (Y-axis affinity rotation).
      4. [non-flush only] Level base.
      5. Translate to pos.
      6. Snap wall-affinity face to its wall.
    """
    if obb_R is not None:
        # OBB already applied: centre + rotate raw verts
        centroid = mesh_verts.astype(np.float64).mean(axis=0)
        verts = (mesh_verts.astype(np.float64) - centroid) @ obb_R.T
        # Re-anchor: centre XZ at origin, Y at base level
        cx = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
        cz = (verts[:, 2].min() + verts[:, 2].max()) / 2.0
        verts[:, 0] -= cx
        verts[:, 2] -= cz
        # Translate so the detected base sits at Y=0.  _find_base_y finds
        # where the XZ cross-section widens (the real base), ignoring any
        # narrow spike/point cluster below it.  We only translate — never
        # clip or modify individual vertex positions — to preserve the
        # original mesh geometry (legs, feet, etc.) exactly.
        y_base = _find_base_y(verts, quiet=quiet)
        verts[:, 1] -= y_base
    else:
        cx = (mesh_bounds[0][0] + mesh_bounds[1][0]) / 2.0
        cz = (mesh_bounds[0][2] + mesh_bounds[1][2]) / 2.0
        y_bottom = float(mesh_bounds[0][1])
        anchor = np.array([cx, y_bottom, cz], dtype=np.float64)
        verts = mesh_verts.astype(np.float64) - anchor[np.newaxis, :]

    verts = verts * scale[np.newaxis, :]
    verts = verts @ R.T

    # Anchor base to Y=0 using _find_base_y (robust to spike vertices below
    # the real base).  This preserves all vertex positions — only a uniform
    # Y-translation is applied.
    base_y = _find_base_y(verts, quiet=quiet)
    verts[:, 1] -= base_y

    # Level the base: OBB gives approximate alignment but doesn't guarantee a
    # perfectly flat bottom.  _level_base fits a plane through the 4 lowest
    # corner contacts and corrects residual tilt so the object sits stable.
    if not skip_level_base:
        verts = _level_base(verts, max_angle_deg=30.0, quiet=quiet)
    elif not quiet:
        print(f"  [level_base] skipped (flat_on_floor override)")

    # Sink slightly so the lowest visible face aligns with the support surface.
    verts[:, 1] -= 0.025
    # Position in XZ only.  The base is already grounded to Y≈0 above; adding
    # pos[1] here would re-lift the object by any spurious placement-Y (some
    # placement paths set pos_y to ~half the object height to keep the raw
    # mid-body origin from reading as "below floor"), leaving it hovering half
    # its height.  Keep the grounded base so the object always just rests on the
    # floor — lifting/dropping only enough to resolve the ground conflict.
    verts[:, 0] += pos[0]
    verts[:, 2] += pos[2]

    # Snap: re-snap so the back/side face stays flush against the wall.
    # Wall-affinity objects MUST be against their wall — this constraint has
    # higher priority than mask silhouette alignment.  No drift cap: scale
    # changes in mask_align can shift the mesh arbitrarily far from the wall,
    # and the wall constraint must always win.
    snap_gap = 0.0 if back_flush else _WALL_GAP
    if skip_wall_snap:
        pass  # position_override_m was set — don't forcibly flush against wall
    elif wall_affinity == "back":
        drift = float(verts[:, 2].min()) - snap_gap
        if abs(drift) > 1e-3:
            verts[:, 2] -= drift
            if not quiet:
                print(f"  [wall_snap] back-wall Z drift {drift*100:.1f} cm corrected → flush")
    elif wall_affinity == "left":
        drift = float(verts[:, 0].min()) - snap_gap
        if abs(drift) > 1e-3:
            verts[:, 0] -= drift
            if not quiet:
                print(f"  [wall_snap] left-wall X drift {drift*100:.1f} cm corrected → flush")
    elif wall_affinity == "right" and room_w is not None:
        drift = float(verts[:, 0].max()) - (room_w - snap_gap)
        if abs(drift) > 1e-3:
            verts[:, 0] -= drift
            if not quiet:
                print(f"  [wall_snap] right-wall X drift {drift*100:.1f} cm corrected → flush")

    # ── Enforce ground contact ────────────────────────────────────────────────
    # Use _find_base_y again to robustly anchor the base (not the spike tip)
    # to the support surface.  Only a Y-translation — no vertex modification.
    # Ground the base to the FLOOR (Y≈0), never to pos[1]: some placement paths
    # set pos_y to ~half the object height (so the raw mid-body origin doesn't
    # read as "below floor"), and anchoring the base to that made objects hover
    # half their height.  We only lift/drop the object enough to rest its base
    # on the floor, resolving the ground conflict either way.
    target_y = -0.015   # 1.5cm sink to hide floor gap
    base_after = _find_base_y(verts, quiet=quiet)
    y_err = base_after - target_y
    if abs(y_err) > 1e-4:
        verts[:, 1] -= y_err
        if not quiet:
            print(f"  [ground] base Y corrected by {y_err*100:.1f} cm → base at {target_y:.3f}")

    return verts


VLM_API_URL = "http://localhost:8080/v1/chat/completions"

# ── VLM backend router ────────────────────────────────────────────────────────
# All furniture-placement VLM calls go through _vlm_post so the backend can be
# switched between the local Qwen server and NVIDIA-hosted gpt-5.5 with one env
# var.  Both endpoints are OpenAI-compatible chat/completions, so the response
# JSON (choices[0].message.content) is identical and callers are unchanged.
#   SCENEWEAVE_VLM_BACKEND = "qwen" (default) | "gpt55"
#   NVIDIA_API_KEY         = key for the gpt55 backend
_NVIDIA_VLM_URL   = "https://inference-api.nvidia.com/v1/chat/completions"
_GPT55_VLM_MODEL  = "openai/openai/gpt-5.5"


def _vlm_backend() -> str:
    import os
    return os.environ.get("SCENEWEAVE_VLM_BACKEND", "qwen").strip().lower()


def _vlm_post(payload: dict, timeout: int = 90):
    """POST an OpenAI-style chat payload to the active VLM backend.

    Delegates to the SHARED object_placement.vlm_backend.vlm_post, which every
    other placement module already uses.  This module used to carry its own
    stale copy of that function, and the drift silently disabled the entire
    placement reflection loop:

      * it never applied the reasoning-model max_tokens floor.  gpt-5.5 spends
        max_tokens on REASONING before it emits anything, and the refine prompt
        (long prompt + room photo + crop) needs ~1.5k reasoning tokens — but
        the call sends max_tokens=512, so the budget was exhausted mid-reasoning
        and the reply came back with EMPTY content.  That is the `raw=''` seen
        on every object of every office8 run: 12/12 refine calls empty, every
        object falling through to "UNVERIFIED" and leaving the geometric
        passes to do all the work unchecked.  Probed directly, the same prompt
        at the shared backend's floor answers correctly and returns the desk as
        1.9 x 0.76 x 0.7 m — the right wide-shallow shape the pipeline missed.
      * it never stripped temperature/top_p, which gpt-5.5 rejects with 400.
      * it had no retry/backoff for transient 429/502/504.
      * its no-key fallback called ITSELF rather than the Qwen path (infinite
        recursion).
    """
    from object_placement.vlm_backend import vlm_post as _shared_vlm_post
    return _shared_vlm_post(payload, timeout=timeout)


# ── Perspective-based size estimation ─────────────────────────────────────────

def _estimate_size_from_bbox(
    bx1: float, by1: float, bx2: float, by2: float,
    floor_pt: np.ndarray,
    cam_pos: np.ndarray,
    fwd_v: np.ndarray,
    right_v: np.ndarray,
    up_c_v: np.ndarray,
    fx: float, cx: float, cy: float,
    obj_type: str,
) -> tuple[float, float, float]:
    """Estimate (width_m, height_m, depth_m) from the bbox using perspective.

    Width:  back-project left/right bbox edges at the bottom row to the floor,
            measure their world-space distance.
    Height: back-project the top-centre of the bbox; find the world point that
            lies directly above the floor contact (same XZ) and read its Y.
    Depth:  keep the default — depth is not recoverable from a single view.
    """
    # Camera-space depth of the floor contact point (z_c = forward-axis distance)
    z_c = float(np.dot(floor_pt - cam_pos, fwd_v))
    if z_c < 0.1:
        return _DEFAULT_SIZES.get(obj_type, _DEFAULT_SIZE_FALLBACK)

    # ── Width from horizontal bbox span at the floor row ─────────────────────
    ray_l = _backproject_pixel(bx1, by2, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
    ray_r = _backproject_pixel(bx2, by2, cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy)
    fp_l  = _ray_floor_intersect(cam_pos, ray_l)
    fp_r  = _ray_floor_intersect(cam_pos, ray_r)
    if fp_l is not None and fp_r is not None:
        est_w = float(np.linalg.norm(fp_r - fp_l))
    else:
        est_w = float((bx2 - bx1) * z_c / fx)
    est_w = max(0.15, min(5.0, est_w))

    # ── Height from top-centre bbox edge ─────────────────────────────────────
    ray_top = _backproject_pixel(
        (bx1 + bx2) / 2.0, by1,
        cam_pos, right_v, up_c_v, fwd_v, fx, fx, cx, cy
    )
    # Find parameter t such that the XZ of (cam_pos + t*ray_top) matches floor_pt
    ts = []
    if abs(ray_top[0]) > 1e-6:
        ts.append((floor_pt[0] - cam_pos[0]) / ray_top[0])
    if abs(ray_top[2]) > 1e-6:
        ts.append((floor_pt[2] - cam_pos[2]) / ray_top[2])
    if ts:
        t_top  = float(np.mean(ts))
        top_pt = cam_pos + t_top * ray_top
        est_h  = max(0.05, float(top_pt[1]))          # world Y of object top
    else:
        est_h = _DEFAULT_SIZES.get(obj_type, _DEFAULT_SIZE_FALLBACK)[1]
    est_h = max(0.05, min(3.0, est_h))

    # ── Depth: keep default (not recoverable from single view) ───────────────
    est_d = _DEFAULT_SIZES.get(obj_type, _DEFAULT_SIZE_FALLBACK)[2]

    return est_w, est_h, est_d


# ── VLM numerical placement estimation ────────────────────────────────────────

def _vlm_estimate_placement(
    crop_path: Path,
    scene_image_path: Path | None,
    obj_type: str,
    wall_affinity: str,
    room_w: float,
    room_d: float,
    cam: dict,
) -> dict:
    """Ask the VLM to estimate numerical placement values from visual inspection.

    This runs BEFORE the perceptual refinement step and produces:
      wall_offset_m  (float): position along the affinity wall (0 = back/left corner)
      width_m        (float): estimated object width in metres
      height_m       (float): estimated object height in metres
      depth_m        (float): estimated object depth in metres
      confidence     (float): 0–1, how confident the VLM is in these estimates
      notes          (str)

    For wall-affinity objects (back/left/right) the offset is along the wall:
      back  → offset in X (0 = left end of back wall)
      left  → offset in Z (0 = back corner)
      right → offset in Z (0 = back corner)
    For centre objects offset is not used.
    """
    import re, requests

    if not crop_path.exists():
        return {}

    # Build wall feature context
    wall_features: dict[str, list] = {}
    for w in cam.get("wall_context", {}).values():
        pass  # wall_context doesn't carry features; use floorplan_analysis if available
    feat_desc = ""
    # Try to get wall features from floorplan_analysis (same output dir)
    furn_dir = crop_path.parent.parent
    fp_path = furn_dir.parent / "floorplan_analysis.json"
    if fp_path.exists():
        import json as _json
        with open(fp_path) as _f:
            fp = _json.load(_f)
        for w in fp.get("walls", []):
            if w["orientation"] == wall_affinity:
                feats = w.get("features", [])
                if feats:
                    feat_desc = f"  {wall_affinity} wall features: " + ", ".join(
                        f"{f['type']} {f['width_m']}m wide at offset {f['offset_from_left_m']}m"
                        for f in feats
                    )

    affinity_axis = {
        "back":   f"along the back wall (X axis, 0=left end, {room_w:.1f}=right end)",
        "left":   f"along the left wall (Z axis, 0=back corner, {room_d:.1f}=front)",
        "right":  f"along the right wall (Z axis, 0=back corner, {room_d:.1f}=front)",
        "centre": "anywhere in the room (not wall-bound)",
    }.get(wall_affinity, wall_affinity)

    wall_length = room_d if wall_affinity in ("left", "right") else room_w

    # Read ceiling height for the scale-calibration anchor
    ceiling_h = 2.7
    if fp_path.exists():
        try:
            with open(fp_path) as _fp2:
                _fp2_data = json.load(_fp2)
            ceiling_h = float(_fp2_data.get("room", {}).get("ceiling_height_m", 2.7))
        except Exception:
            pass

    prompt = f"""You are estimating the **real-world dimensions and placement** of a {obj_type} in a room.

Room: {room_w:.1f} m wide × {room_d:.1f} m deep, ceiling height = {ceiling_h:.1f} m
Object wall affinity: "{wall_affinity}" — placed {affinity_axis}.
Wall length for this affinity: {wall_length:.1f} m
{feat_desc}

Image 1 (if provided): the full room scene — use it for scale calibration.
Image 2: the {obj_type} crop from the original photo.

**Scale calibration — this is the key step:**
The visible wall in Image 1 is exactly {ceiling_h:.1f} m tall (floor to ceiling).
1. Estimate what fraction of that wall height the {obj_type} occupies.
2. Multiply by {ceiling_h:.1f} m → that is height_m.
3. Estimate width and depth from the object's proportions relative to the wall width
   ({room_w:.1f} m) or window widths {feat_desc.strip() or '(if visible)'}.

**Then provide:**

1. **wall_offset_m**: centre of the {obj_type} along the {wall_affinity} wall
   (0–{wall_length:.1f} m from back/left corner; ignore for "centre" affinity).

2. **width_m, height_m, depth_m**: real-world dimensions derived from the scale
   calibration above.

3. **confidence**: 0.0–1.0 (use ≥0.7 only when wall height is clearly visible).

Respond with ONLY a JSON object, no markdown:
{{"wall_offset_m": 0.0, "width_m": 0.0, "height_m": 0.0, "depth_m": 0.0, "confidence": 0.0, "notes": "fraction of wall height used"}}"""

    content: list[dict] = [{"type": "text", "text": prompt}]
    if scene_image_path and scene_image_path.exists():
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_encode_image(scene_image_path)}"}})
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(crop_path)}"}})

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    try:
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"  [vlm_num] failed: {e}")
    return {}


# ── VLM rotation / scale refinement ───────────────────────────────────────────

def _encode_image(path: Path, max_side: int = 512) -> str:
    """Load an image, downscale if needed, return base64 PNG string."""
    import base64, io
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _vlm_refine_placement(
    crop_path: str | Path,
    obj_type: str,
    wall_affinity: str,
    est_w: float,
    est_h: float,
    est_d: float,
    position_m: np.ndarray | None = None,
    cam: dict | None = None,
    scene_image_path: Path | None = None,
) -> dict:
    """Ask the VLM to reason about orientation and scale in room context.

    Sends the object crop (+ optionally the full scene image) and asks the VLM
    to reason about:
      - Where the object sits relative to walls
      - Furniture arrangement conventions (sofa back against wall, facing centre…)
      - Whether the current wall_affinity rotation is correct
      - Whether the estimated size looks right

    Returns a dict with keys:
      rotation_deg  (int): additional Y rotation in degrees (0, 90, -90, 180)
      scale_factor  (float): uniform scale multiplier (1.0 = no change)
      notes         (str)
    """
    import re, requests

    crop_path = Path(crop_path)
    if not crop_path.exists():
        return {"rotation_deg": 0, "scale_factor": 1.0, "notes": "no crop"}

    # Build room-context description
    room_desc = ""
    nearest_wall = wall_affinity  # fallback
    if position_m is not None and cam is not None:
        room_w, room_d = _get_room_dims(Path(crop_path) if crop_path else Path("."), cam)
        px, pz = float(position_m[0]), float(position_m[2])
        dist_back  = pz                # distance to back wall  (Z=0)
        dist_front = room_d - pz       # distance to front wall (Z=room_d)
        dist_left  = px                # distance to left wall  (X=0)
        dist_right = room_w - px       # distance to right wall (X=room_w)
        dists = {"back wall": dist_back, "front wall": dist_front,
                 "left wall": dist_left, "right wall": dist_right}
        nearest_wall = min(dists, key=dists.get)
        room_desc = (
            f"\nRoom layout (top-down, camera at front):\n"
            f"  Room size: {room_w:.1f} m wide × {room_d:.1f} m deep\n"
            f"  Object position: {px:.2f} m from left wall, {pz:.2f} m from back wall\n"
            f"  Distance to walls — back: {dist_back:.2f} m, front: {dist_front:.2f} m, "
            f"left: {dist_left:.2f} m, right: {dist_right:.2f} m\n"
            f"  → Nearest wall: {nearest_wall}"
        )

    # Facing description from wall_affinity.
    # wall_affinity names the wall the BACK is against; the FRONT faces INTO the room.
    # back-wall object → front (seat/foot) faces toward camera (+Z)
    # left-wall object → front faces right into room (+X)
    # right-wall object → front faces left into room (-X)
    facing_map = {
        "back":   "away from the back wall — toward the camera (foot/seat faces the camera)",
        "left":   "away from the left wall — into the room (toward the right side)",
        "right":  "away from the right wall — into the room (toward the left side)",
        "centre": "toward the camera / room centre",
    }
    facing = facing_map.get(wall_affinity, wall_affinity)

    # Furniture-specific orientation heuristics the VLM should consider
    orient_hints = {
        "sofa":      "Sofas typically have their back against the nearest wall and face into the room.",
        "armchair":  "Armchairs face toward the room centre or a focal point (TV, fireplace, coffee table).",
        "chair":     "Dining/accent chairs face toward a table or room centre.",
        "coffee_table": ("A rectangular coffee table has NO 'front', but its LONG axis is "
                         "NOT free — it must run PARALLEL to the main sofa/seating it sits "
                         "in front of, i.e. parallel to the back wall, exactly as in the "
                         "reference photo. Look at the photo: the table's long, wide edge "
                         "faces the camera and its long axis runs left-to-right across the "
                         "frame. If the CURRENT placement instead has the long axis pointing "
                         "front-to-back (toward the camera) — i.e. its estimated depth is "
                         "larger than its width — it is rotated 90° wrong, so return "
                         "rotation_deg=90 to lay the long edge left-to-right. Match the "
                         "reference orientation; do NOT leave it perpendicular to the sofa."),
        "dining_table": ("A rectangular dining table's LONG axis should match the reference "
                         "photo orientation (usually parallel to the longer wall / the seating "
                         "rows). If the current long axis is perpendicular to how it appears in "
                         "the photo (depth larger than width when the photo shows it wider than "
                         "deep), return rotation_deg=90."),
        "desk":      "Desks face the user — back against a wall.",
        "bookcase":  "Bookcases have their back against a wall, open shelves facing the room.",
        "cabinet":   "Cabinets have their back against the nearest wall.",
        "bed":       "Beds have their headboard against a wall.",
        "plant":     "Plants have no strong orientation.",
    }
    hint = orient_hints.get(obj_type, "Use common interior design conventions.")

    prompt = f"""You are a 3D scene layout assistant refining the placement of a **{obj_type}**.

**Image 1** (if provided): the full room photo for context.
**Image 2**: a cropped view of the {obj_type} as it appears in the original photo.
{room_desc}

Current placement assumption:
  • wall_affinity = "{wall_affinity}" → front faces {facing}.
  • Estimated size: width={est_w:.2f} m, height={est_h:.2f} m, depth={est_d:.2f} m.

Interior design rule for this object type:
  {hint}

**Your tasks:**

1. **Orientation reasoning**: Look at the crop.
   - Which direction is the {obj_type}'s front/face pointing in the photo?
   - Given its position ({nearest_wall} is closest), what orientation makes sense?
   - Does the current wall_affinity rotation ({facing}) match that, or does it need correction?
   - If correction needed, give an additional Y-axis rotation (viewed from above, clockwise positive).
   - Valid values: 0, 90, -90, 180.

2. **Size estimation**: What are the real-world dimensions of this {obj_type}?
   Estimate the target size in metres for each axis:
   - width_m:  horizontal extent (left-right as seen from front)
   - height_m: vertical extent (floor to top)
   - depth_m:  front-to-back extent
   Use your knowledge of typical furniture sizes and the visual context.
   Current estimate: width={est_w:.2f} m, height={est_h:.2f} m, depth={est_d:.2f} m.
   If current looks correct, repeat the same values.

Respond with ONLY a JSON object, no markdown:
{{"rotation_deg": 0, "width_m": {est_w:.2f}, "height_m": {est_h:.2f}, "depth_m": {est_d:.2f}, "notes": "one-sentence reasoning"}}"""

    # Build message content: scene image first (context), then crop
    content: list[dict] = [{"type": "text", "text": prompt}]
    if scene_image_path and scene_image_path.exists():
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_encode_image(scene_image_path)}"}})
    content.append({"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{_encode_image(crop_path)}"}})

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    # A reply that carries no JSON used to fall through to "vlm unavailable"
    # SILENTLY (no exception, no log) — the refinement loop then became a no-op
    # while the pipeline still reported success. Retry once and always say why.
    for _try in range(2):
        try:
            resp = _vlm_post(payload, timeout=90)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"] or ""
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
            m = re.search(r"\{[\s\S]*?\}", raw)
            if m:
                result = json.loads(m.group())
                result.setdefault("rotation_deg",  0)
                result.setdefault("width_m",  est_w)
                result.setdefault("height_m", est_h)
                result.setdefault("depth_m",  est_d)
                result.setdefault("notes", "")
                return result
            print(f"  [vlm_refine] no JSON in reply (try {_try + 1}/2), "
                  f"raw={raw[:160]!r}")
        except Exception as e:
            print(f"  [vlm_refine] call failed (try {_try + 1}/2): {e}")
    print("  [vlm_refine] UNVERIFIED — keeping numerical estimate "
          f"({est_w:.2f}×{est_h:.2f}×{est_d:.2f}m); placement not VLM-checked")
    return {"rotation_deg": 0, "width_m": est_w, "height_m": est_h,
            "depth_m": est_d, "notes": "vlm unavailable (UNVERIFIED)"}


# ── the 3D generator ground-plane removal ────────────────────────────────────────────────

def _remove_ground_faces(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Remove the flat ground slab that the 3D generator bakes into furniture GLBs.

    the 3D generator reconstructs geometry from a single image and adds a horizontal ground
    patch at the base of the object.  We use two complementary detectors:

    1. **Histogram-gap cut** — build a vertical (Y) histogram of face centres;
       if there is a clear valley in the bottom 25 % of the mesh (bin count
       < 5 % of the peak bin), treat everything strictly below the valley as
       the slab.  This handles cases where the slab is a disconnected disc
       below the object (e.g. plants).

    2. **Thin-slab fallback** — drop any face whose full vertical extent lies
       within the bottom 8 % of the mesh height AND whose normal has a
       noticeable upward component (ny > 0.3).  This catches cases where the
       slab is welded into the object (e.g. sofa cushions fused to the floor
       patch), which the histogram-gap detector would miss.

    Returns a new Trimesh (or a (mesh, keep_mask) tuple) so the caller can
    re-index per-face textures correctly.
    """
    import trimesh as _trimesh

    verts   = mesh.vertices
    faces   = mesh.faces
    fn      = mesh.face_normals
    y_min   = float(verts[:, 1].min())
    y_max   = float(verts[:, 1].max())
    h       = y_max - y_min or 1.0

    face_vy     = verts[faces][:, :, 1]
    face_min_y  = face_vy.min(axis=1)
    face_max_y  = face_vy.max(axis=1)
    face_ctr_y  = face_vy.mean(axis=1)

    # ── (1) histogram-gap cut ──────────────────────────────────────────────
    n_bins = 30
    hist, edges = np.histogram(face_ctr_y, bins=n_bins, range=(y_min, y_max))
    peak = int(hist.max()) or 1
    cut_y = None
    # Scan bottom 25% of bins, starting just above the floor, looking for
    # a valley (bin ≤ 5% of peak) with at least one populated bin below it.
    bottom_span = max(2, int(n_bins * 0.25))
    for b in range(1, bottom_span):
        if hist[b] <= max(1, int(peak * 0.05)) and hist[:b].sum() > 0:
            cut_y = float(edges[b])
            break

    is_hist_ground = np.zeros(len(faces), dtype=bool)
    if cut_y is not None:
        # Everything whose centre sits strictly below the valley
        below = face_ctr_y < cut_y
        # Require mostly-horizontal normals (guard against cutting a leg)
        horiz = fn[:, 1] > 0.3
        candidate = below & horiz
        # Only commit to the histogram cut if it actually carves off a slab
        # (>= 0.3 % of faces) — otherwise let the fallback handle it.
        if candidate.sum() >= max(10, int(0.003 * len(faces))):
            is_hist_ground = candidate

    # ── (2) thin-slab fallback ─────────────────────────────────────────────
    slab_thresh = 0.08 * h
    face_height = face_max_y - face_min_y
    is_slab = (
        ((face_max_y - y_min) < slab_thresh)
        & (face_height < slab_thresh)
        & (fn[:, 1] > 0.3)
    )

    is_ground = is_hist_ground | is_slab
    keep = ~is_ground
    if keep.all():
        return mesh          # nothing to remove

    n_removed = int(is_ground.sum())
    new_mesh = _trimesh.Trimesh(
        vertices=verts,
        faces=faces[keep],
        visual=mesh.visual,
        process=False,
    )
    src = []
    if is_hist_ground.any():
        src.append(f"hist@{cut_y:.3f}")
    if is_slab.any():
        src.append("slab")
    print(
        f"  [ground_rm] removed {n_removed}/{len(faces)} ground faces "
        f"({','.join(src) or 'none'}) → {keep.sum()} remain"
    )
    return new_mesh, keep    # also return keep mask for texture re-indexing


# ── Carpet direct per-pixel projection ────────────────────────────────────────

def _render_carpet_direct(
    buf: np.ndarray,
    glb_path: str,
    cam_pos: np.ndarray,
    right_v: np.ndarray,
    up_c_v: np.ndarray,
    fwd_v: np.ndarray,
    fx: float,
    cx: float,
    cy: float,
) -> bool:
    """Render a flat carpet GLB by per-pixel back-projection onto the floor plane.

    For each output pixel whose floor-plane intersection falls inside the carpet
    world-space bounding box, the texture is sampled at the corresponding UV
    coordinates (u = (x-x_min)/(x_max-x_min), v = (z-z_min)/(z_max-z_min)).

    This avoids the per-face colour averaging used by the mesh rasteriser and
    produces correct per-pixel texture sampling with full stripe detail.

    Returns True on success, False if the GLB or its texture could not be loaded.
    """
    import struct, json as _json
    from PIL import Image as _PIL

    H, W = buf.shape[:2]

    # ── Load texture and carpet world bounds from the GLB ─────────────────────
    try:
        with open(glb_path, "rb") as fh:
            data = fh.read()
        magic, _ver, _total = struct.unpack_from("<III", data, 0)
        if magic != 0x46546C67:
            return False
        json_len, _json_type = struct.unpack_from("<II", data, 12)
        gltf = _json.loads(data[20: 20 + json_len])
        bin_start = 20 + json_len
        bin_start = (bin_start + 3) & ~3   # align to 4 bytes

        def _get_bv(bv_idx: int) -> bytes:
            bv = gltf["bufferViews"][bv_idx]
            off = bin_start + 8 + bv.get("byteOffset", 0)
            return data[off: off + bv["byteLength"]]

        def _accessor(acc_idx: int) -> np.ndarray:
            acc = gltf["accessors"][acc_idx]
            _dtype = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                      5123: np.uint16, 5125: np.uint32, 5126: np.float32}[acc["componentType"]]
            _comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
            raw = np.frombuffer(_get_bv(acc["bufferView"]), dtype=_dtype)
            return raw.reshape(acc["count"], _comp) if _comp > 1 else raw

        # Texture image
        tex_bv_idx = gltf["images"][0]["bufferView"]
        tex_bytes  = _get_bv(tex_bv_idx)
        tex_img    = np.array(_PIL.open(__import__("io").BytesIO(tex_bytes)).convert("RGB"),
                              dtype=np.uint8)
        th, tw = tex_img.shape[:2]

        # Vertex positions to determine carpet bounds
        prim  = gltf["meshes"][0]["primitives"][0]
        verts = _accessor(prim["attributes"]["POSITION"]).astype(np.float32)
        uvs   = _accessor(prim["attributes"]["TEXCOORD_0"]).astype(np.float32)
        # GLB UVs: v=0 at top, flip V to match PIL convention for sampling
        uvs[:, 1] = 1.0 - uvs[:, 1]

    except Exception as e:
        print(f"  [carpet_direct] failed to load GLB: {e}")
        return False

    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    if x_max <= x_min or z_max <= z_min:
        return False

    # ── Project carpet corners to screen to get pixel bounding box ────────────
    corners_w = np.array([[x_min, 0.0, z_min], [x_max, 0.0, z_min],
                           [x_max, 0.0, z_max], [x_min, 0.0, z_max]], dtype=np.float64)
    scr_pts = []
    for cw in corners_w:
        d = cw - cam_pos
        fd = float(np.dot(d, fwd_v))
        if fd <= 0:
            continue
        scr_pts.append((cx + fx * np.dot(d, right_v) / fd,
                        cy - fx * np.dot(d, up_c_v)  / fd))
    if not scr_pts:
        return False
    px0 = max(0, int(min(p[0] for p in scr_pts)) - 1)
    px1 = min(W - 1, int(max(p[0] for p in scr_pts)) + 1)
    py0 = max(0, int(min(p[1] for p in scr_pts)) - 1)
    py1 = min(H - 1, int(max(p[1] for p in scr_pts)) + 1)

    # ── Vectorised back-projection over pixel bounding box ────────────────────
    xs = np.arange(px0, px1 + 1, dtype=np.float64)
    ys = np.arange(py0, py1 + 1, dtype=np.float64)
    px_g, py_g = np.meshgrid(xs, ys)   # (h_bb, w_bb)

    # Ray direction for each pixel
    dx = (px_g - cx) / fx
    dy = -(py_g - cy) / fx
    # In camera space: forward=fwd_v, right=right_v, up=up_c_v
    ray_x = dx * right_v[0] + dy * up_c_v[0] + fwd_v[0]
    ray_y = dx * right_v[1] + dy * up_c_v[1] + fwd_v[1]
    ray_z = dx * right_v[2] + dy * up_c_v[2] + fwd_v[2]

    # Intersect with floor plane Y=0: t = -cam_y / ray_y
    denom = ray_y
    valid = (np.abs(denom) > 1e-9) & (denom < 0.0)   # ray pointing down
    t     = np.where(valid, -cam_pos[1] / denom, np.inf)
    valid &= t > 0.0

    wx = cam_pos[0] + t * ray_x
    wz = cam_pos[2] + t * ray_z

    # Check inside carpet bounds
    in_carpet = valid & (wx >= x_min) & (wx <= x_max) & (wz >= z_min) & (wz <= z_max)

    # UV coordinates
    u = np.clip((wx - x_min) / (x_max - x_min), 0.0, 1.0)
    v = np.clip((wz - z_min) / (z_max - z_min), 0.0, 1.0)

    # Texture sample (nearest-neighbour for speed; bilinear optional)
    tx = np.clip((u * (tw - 1)).astype(np.int32), 0, tw - 1)
    ty = np.clip((v * (th - 1)).astype(np.int32), 0, th - 1)

    rows = py_g[in_carpet].astype(np.int32)
    cols = px_g[in_carpet].astype(np.int32)
    buf[rows, cols] = tex_img[ty[in_carpet], tx[in_carpet]]

    print(f"  [carpet_direct] painted {int(in_carpet.sum())} pixels  "
          f"bounds X=[{x_min:.2f},{x_max:.2f}] Z=[{z_min:.2f},{z_max:.2f}]")
    return True


# ── Carpet texture segmentation ──────────────────────────────────────────────

def _segment_carpet_texture(
    tex_path: str,
    existing_mask_path: "str | Path | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load carpet texture and isolate the carpet from background padding.

    Source priority for the mask:
      0. The texture's OWN alpha channel when present.  Inpaint pads are often
         saved as RGBA with alpha=0 outside the carpet — that alpha IS the
         most accurate mask (produced in the texture's own coord frame).
      1. `existing_mask_path` — SAM mask saved during segment_furniture, BUT
         only when its dimensions match the texture (full-image masks for a
         cropped inpaint texture cannot be naively resized — they sit at
         the wrong location after .resize()).
      2. SAM run on the texture with a centre-point prompt (legacy fallback).
      3. Background-threshold of the texture, probing the four corners to
         auto-detect grey vs white padding.

    Returns (tex_rgb, mask_bool) where mask_bool is True for carpet pixels.
    Both are cropped to the tight bounding box of the carpet region.
    """
    from PIL import Image as _PIL

    tex_pil = _PIL.open(tex_path)
    # ── Path 0: texture's own alpha channel ──────────────────────────────────
    tex_alpha_mask: "np.ndarray | None" = None
    if tex_pil.mode in ("RGBA", "LA"):
        _arr = np.array(tex_pil)
        _alpha = _arr[..., -1]
        if int(_alpha.max()) > 0 and int(_alpha.min()) < 250:
            tex_alpha_mask = _alpha > 127
    tex_full = np.array(tex_pil.convert("RGB"), dtype=np.uint8)
    h, w = tex_full.shape[:2]

    mask = None
    if tex_alpha_mask is not None:
        mask = tex_alpha_mask
        print(f"  [carpet_seg] using texture's own alpha channel: "
              f"{int(mask.sum())}/{mask.size} pixels "
              f"({mask.mean()*100:.1f}%) — texture-native silhouette")

    # ── Path 1: pre-existing SAM mask (only if same coord frame) ─────────────
    if mask is None and existing_mask_path is not None and Path(existing_mask_path).exists():
        try:
            m_raw = _PIL.open(str(existing_mask_path))
            mw, mh = m_raw.size
            size_ok = (
                abs(mw - w) <= max(8, int(0.25 * w)) and
                abs(mh - h) <= max(8, int(0.25 * h))
            )
            if not size_ok:
                print(f"  [carpet_seg] skipping pre-existing mask "
                      f"({Path(existing_mask_path).name}): size {mw}×{mh} "
                      f"≠ texture {w}×{h} (different coord frame)")
            else:
                if m_raw.mode in ("RGBA", "LA"):
                    m_arr = np.array(m_raw)
                    alpha = m_arr[..., -1]
                    if int(alpha.max()) > 0:
                        mask_full = alpha > 127
                    else:
                        mask_full = np.array(m_raw.convert("L"), dtype=np.uint8) > 127
                else:
                    mask_full = np.array(m_raw.convert("L"), dtype=np.uint8) > 127
                if mask_full.shape != (h, w):
                    mask_full = np.array(
                        _PIL.fromarray(mask_full.astype(np.uint8) * 255)
                        .resize((w, h), _PIL.NEAREST),
                        dtype=np.uint8) > 127
                mask = mask_full
                print(f"  [carpet_seg] using pre-existing SAM mask: "
                      f"{int(mask.sum())}/{mask.size} pixels "
                      f"({mask.mean()*100:.1f}%)")
        except Exception as _me:
            print(f"  [carpet_seg] pre-existing SAM mask load failed ({_me})")
            mask = None

    # ── Path 2: SAM on the inpaint with centre-point prompt ──────────────────
    if mask is None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
            import torch
            _SAM_TYPE = "vit_h"
            _SAM_CKPTS = [
                Path(__file__).resolve().parents[2] / "checkpoints" / "sam_vit_h_4b8939.pth",
                Path("checkpoints/sam_vit_h_4b8939.pth"),
            ]
            if os.environ.get("SCENEWEAVE_SAM_CKPT"):
                _SAM_CKPTS.insert(0, Path(os.environ["SCENEWEAVE_SAM_CKPT"]))
            ckpt = next((p for p in _SAM_CKPTS if p.exists()), None)
            if ckpt is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam = sam_model_registry[_SAM_TYPE](checkpoint=str(ckpt)).to(device)
                predictor = SamPredictor(sam)
                predictor.set_image(tex_full)
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[w // 2, h // 2]]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                best = int(np.argmax([m.sum() for m in masks]))
                mask = masks[best].astype(bool)
                print(f"  [carpet_seg] SAM mask: {mask.sum()}/{mask.size} pixels "
                      f"({mask.sum()/mask.size*100:.1f}%)")
        except Exception as e:
            print(f"  [carpet_seg] SAM unavailable ({e}), falling back to threshold")

    # ── Path 3: background-threshold (auto-detect bg colour from corners) ────
    if mask is None:
        TOLERANCE = 18
        h_tex, w_tex = tex_full.shape[:2]
        corner_px = np.stack([
            tex_full[0, 0], tex_full[0, w_tex - 1],
            tex_full[h_tex - 1, 0], tex_full[h_tex - 1, w_tex - 1],
        ]).astype(int)
        bg = corner_px.mean(axis=0)
        is_bg = np.all(np.abs(tex_full.astype(int) - bg) < TOLERANCE, axis=2)
        mask = ~is_bg
        try:
            from scipy import ndimage as ndi
            mask = ndi.binary_fill_holes(mask)
            mask = ndi.binary_opening(mask, iterations=2)
            mask = ndi.binary_closing(mask, iterations=2)
        except ImportError:
            pass
        print(f"  [carpet_seg] threshold mask (bg≈{bg.astype(int).tolist()}): "
              f"{mask.sum()}/{mask.size} pixels "
              f"({mask.sum()/mask.size*100:.1f}%)")

    # ── Crop to content bounding box ─────────────────────────────────────────
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return tex_full, np.ones((h, w), dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    tex_crop = tex_full[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1].copy()
    print(f"  [carpet_seg] cropped {w}×{h} → {x1-x0}×{y1-y0}")
    return tex_crop, mask_crop


# ── Flat-quad carpet renderer (no GLB) ───────────────────────────────────────

def _render_flat_carpet_quad(
    buf: np.ndarray,
    p: dict,
    cam_pos: np.ndarray,
    right_v: np.ndarray,
    up_c_v: np.ndarray,
    fwd_v: np.ndarray,
    fx: float,
    cx: float,
    cy: float,
) -> bool:
    """Render a flat carpet quad directly from `world_corners` + `carpet_img`.

    compute_carpet_placement produces a dict with four world-space corners
    [BL, BR, TR, TL] on the Y=0 floor plane and a path to the 2D carpet
    texture.  We project pixels inside the screen-space bounding box of these
    corners back to the floor, check the 2-triangle (BL-BR-TR, BL-TR-TL)
    membership, and sample the texture at perspective-correct barycentric UVs.

    The texture is segmented from its grey padding before use (via SAM or
    grey-threshold fallback), so only actual carpet content is rendered.
    """
    carpet_img_path = p.get("carpet_img")
    world_corners   = p.get("world_corners")
    if not carpet_img_path or not world_corners or len(world_corners) != 4:
        print("  [flat_carpet] missing carpet_img or world_corners")
        return False

    try:
        tex, tex_mask = _segment_carpet_texture(carpet_img_path)
    except Exception as e:
        print(f"  [flat_carpet] texture load failed: {e}")
        return False

    # ── Perspective-rectify the texture to a top-down rectangle ────────────
    # The segmented mask itself is the photo-perspective trapezoid.  Detect
    # the trapezoid's four ACTUAL corners (not its bounding rectangle), then
    # warp those into an axis-aligned rectangle so the rug appears in true
    # top-down view.  Pixels outside the carpet are masked out so the saved
    # texture is just the rug, no background.
    try:
        import cv2 as _cv2
        _tm_u8 = (tex_mask.astype(np.uint8) * 255)
        _contours, _ = _cv2.findContours(
            _tm_u8, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
        )
        if _contours:
            _cnt = max(_contours, key=_cv2.contourArea)
            _hull = _cv2.convexHull(_cnt)
            _perim = float(_cv2.arcLength(_hull, True))
            _approx = None
            for _eps_frac in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18):
                _a = _cv2.approxPolyDP(_hull, _eps_frac * _perim, True)
                if len(_a) == 4:
                    _approx = _a
                    break
                if len(_a) < 4:
                    break
            if _approx is None:
                # Couldn't reduce to 4 — pick the 4 extreme corners by sum/diff
                _all = _hull.reshape(-1, 2).astype(np.float32)
                _sums  = _all.sum(axis=1)
                _diffs = _all[:, 0] - _all[:, 1]
                _approx = np.array([[
                    _all[int(np.argmin(_sums))],
                    _all[int(np.argmax(_diffs))],
                    _all[int(np.argmax(_sums))],
                    _all[int(np.argmin(_diffs))],
                ]], dtype=np.float32).reshape(4, 1, 2)
            _quad = _approx.reshape(4, 2).astype(np.float32)
            # Order corners TL, TR, BR, BL by ANGLE about the centroid.  The old
            # sum/diff sorting collapses two corners to the same point on a
            # ROTATED rug quad (argmax(sum) == argmax(diff)), producing a
            # degenerate warp and a flat grey card.  In image y-down coords,
            # sorting by atan2(y-cy, x-cx) ascending yields TL, TR, BR, BL.
            _cen = _quad.mean(axis=0)
            _ang = np.arctan2(_quad[:, 1] - _cen[1], _quad[:, 0] - _cen[0])
            _src = _quad[np.argsort(_ang)].astype(np.float32)
            # Decide output dims from the average of opposite-edge lengths.
            _w_top = float(np.linalg.norm(_src[1] - _src[0]))
            _w_bot = float(np.linalg.norm(_src[2] - _src[3]))
            _h_lft = float(np.linalg.norm(_src[3] - _src[0]))
            _h_rgt = float(np.linalg.norm(_src[2] - _src[1]))
            _out_w = int(round(max(_w_top, _w_bot)))
            _out_h = int(round(max(_h_lft, _h_rgt)))
            if _out_w >= 16 and _out_h >= 16:
                _dst = np.array([
                    [0,        0],
                    [_out_w-1, 0],
                    [_out_w-1, _out_h-1],
                    [0,        _out_h-1],
                ], dtype=np.float32)
                _M = _cv2.getPerspectiveTransform(_src, _dst)
                tex_warp = _cv2.warpPerspective(
                    tex, _M, (_out_w, _out_h), flags=_cv2.INTER_LINEAR,
                    borderMode=_cv2.BORDER_REPLICATE,
                )
                mask_warp = (
                    _cv2.warpPerspective(
                        _tm_u8, _M, (_out_w, _out_h),
                        flags=_cv2.INTER_NEAREST,
                    ) > 127
                )
                # Zero-out anything outside the warped mask so the saved
                # texture is only the rug (no grey/white background bleed).
                tex_warp[~mask_warp] = 255
                tex      = tex_warp
                tex_mask = mask_warp
                # Save next to source for inspection.
                try:
                    from PIL import Image as _PILDbg
                    _src_p = Path(carpet_img_path)
                    _dbg_path = _src_p.with_name(_src_p.stem + "_rectified.png")
                    _PILDbg.fromarray(tex).save(str(_dbg_path))
                    print(f"  [flat_carpet] rectified via 4-corner trapezoid: "
                          f"src corners (top→bot, l→r): "
                          f"TL={_src[0].tolist()} TR={_src[1].tolist()} "
                          f"BR={_src[2].tolist()} BL={_src[3].tolist()} → "
                          f"dst {_out_w}×{_out_h}  saved → {_dbg_path.name}")
                except Exception:
                    print(f"  [flat_carpet] rectified via 4-corner trapezoid: "
                          f"dst {_out_w}×{_out_h}")
    except Exception as _e:
        print(f"  [flat_carpet] rectification skipped ({_e}) — using raw texture")

    # Apply texture-only rotation if requested — keeps geometry intact
    _tex_rot = p.get("_tex_rotation_deg", 0)
    if _tex_rot != 0:
        _rot_k = {90: 3, -90: 1, 180: 2}.get(_tex_rot, 0)
        if _rot_k:
            tex      = np.rot90(tex,      _rot_k)
            tex_mask = np.rot90(tex_mask, _rot_k)

    th, tw = tex.shape[:2]

    H, W = buf.shape[:2]

    # 2D (x, z) corners — Y is 0 on the floor.
    BL = np.array([world_corners[0][0], world_corners[0][2]], dtype=np.float64)
    BR = np.array([world_corners[1][0], world_corners[1][2]], dtype=np.float64)
    TR = np.array([world_corners[2][0], world_corners[2][2]], dtype=np.float64)
    TL = np.array([world_corners[3][0], world_corners[3][2]], dtype=np.float64)

    # Project 4 corners to screen to get the tight pixel bbox to iterate over.
    def _proj(c_xz: np.ndarray):
        p_world = np.array([c_xz[0], 0.0, c_xz[1]], dtype=np.float64)
        d       = p_world - cam_pos
        fd      = float(np.dot(d, fwd_v))
        if fd <= 0:
            return None
        return (cx + fx * np.dot(d, right_v) / fd,
                cy - fx * np.dot(d, up_c_v)  / fd)

    scr = [_proj(c) for c in (BL, BR, TR, TL)]
    # If some corners are behind the camera, use full screen extent for those
    # dimensions — the per-pixel back-projection handles visibility correctly.
    visible = [s for s in scr if s is not None]
    if not visible:
        print("  [flat_carpet] all corners behind the camera — skipping")
        return False

    if len(visible) < 4:
        # Some corners behind camera — extend bbox to screen edges
        print(f"  [flat_carpet] {4 - len(visible)} corner(s) behind camera — using screen-edge bbox")
        px0 = 0
        px1 = W - 1
        py0 = max(0,     int(min(s[1] for s in visible)) - 1)
        py1 = H - 1
    else:
        px0 = max(0,     int(min(s[0] for s in scr)) - 1)
        px1 = min(W - 1, int(max(s[0] for s in scr)) + 1)
        py0 = max(0,     int(min(s[1] for s in scr)) - 1)
        py1 = min(H - 1, int(max(s[1] for s in scr)) + 1)
    if px1 <= px0 or py1 <= py0:
        return False

    # Per-pixel back-projection to Y=0 for the screen bbox.
    xs = np.arange(px0, px1 + 1, dtype=np.float64)
    ys = np.arange(py0, py1 + 1, dtype=np.float64)
    px_g, py_g = np.meshgrid(xs, ys)   # (h_bb, w_bb)

    dx   = (px_g - cx) / fx
    dy   = -(py_g - cy) / fx
    ray_x = dx * right_v[0] + dy * up_c_v[0] + fwd_v[0]
    ray_y = dx * right_v[1] + dy * up_c_v[1] + fwd_v[1]
    ray_z = dx * right_v[2] + dy * up_c_v[2] + fwd_v[2]

    valid = (np.abs(ray_y) > 1e-9) & (ray_y < 0.0)
    t     = np.where(valid, -cam_pos[1] / ray_y, np.inf)
    valid &= t > 0.0
    wx = cam_pos[0] + t * ray_x
    wz = cam_pos[2] + t * ray_z

    # Two-triangle inside test + barycentric UV: T1 = (BL, BR, TR), T2 = (BL, TR, TL).
    def _bary(Qx, Qz, A, B, C):
        v0x = B[0] - A[0]; v0z = B[1] - A[1]
        v1x = C[0] - A[0]; v1z = C[1] - A[1]
        v2x = Qx  - A[0]; v2z = Qz  - A[1]
        denom = v0x * v1z - v1x * v0z
        safe  = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        beta  = (v2x * v1z - v1x * v2z) / safe
        gamma = (v0x * v2z - v2x * v0z) / safe
        alpha = 1.0 - beta - gamma
        inside = (alpha >= -1e-6) & (beta >= -1e-6) & (gamma >= -1e-6) \
                 & (np.abs(denom) > 1e-12)
        return alpha, beta, gamma, inside

    # UV at each corner — PIL image row 0 is the top, so TL/TR have v=0 and
    # BL/BR have v=1.
    uv_BL = (0.0, 1.0)
    uv_BR = (1.0, 1.0)
    uv_TR = (1.0, 0.0)
    uv_TL = (0.0, 0.0)

    a1, b1, g1, in1 = _bary(wx, wz, BL, BR, TR)
    u1 = a1 * uv_BL[0] + b1 * uv_BR[0] + g1 * uv_TR[0]
    v1 = a1 * uv_BL[1] + b1 * uv_BR[1] + g1 * uv_TR[1]

    a2, b2, g2, in2 = _bary(wx, wz, BL, TR, TL)
    u2 = a2 * uv_BL[0] + b2 * uv_TR[0] + g2 * uv_TL[0]
    v2 = a2 * uv_BL[1] + b2 * uv_TR[1] + g2 * uv_TL[1]

    u = np.where(in1, u1, u2)
    v = np.where(in1, v1, v2)
    inside = valid & (in1 | in2)

    tx = np.clip((u * (tw - 1)).astype(np.int32), 0, tw - 1)
    ty = np.clip((v * (th - 1)).astype(np.int32), 0, th - 1)

    # Only paint pixels where the texture mask says there's actual carpet
    # content (not grey padding).
    inside_content = inside.copy()
    inside_content[inside] &= tex_mask[ty[inside], tx[inside]]

    rows = py_g[inside_content].astype(np.int32)
    cols = px_g[inside_content].astype(np.int32)
    buf[rows, cols] = tex[ty[inside_content], tx[inside_content]]

    print(f"  [flat_carpet] painted {int(inside_content.sum())} pixels "
          f"({int(inside.sum())} in quad, {int(inside.sum() - inside_content.sum())} masked out)  "
          f"BL=({BL[0]:.2f},{BL[1]:.2f}) BR=({BR[0]:.2f},{BR[1]:.2f}) "
          f"TR=({TR[0]:.2f},{TR[1]:.2f}) TL=({TL[0]:.2f},{TL[1]:.2f})")
    return True


# ── Vertex colors with inpainted-texture override ─────────────────────────────

_INPAINT_SUBDIRS_FURN = ["inpainted", "inpainted2", "inpainted5", "inpainted0"]


# ── Renderer (adapted from wall_mounted to use furniture base render) ──────────

_RENDER_SCALE = 3   # 3× native for better small-object visibility


def render_furniture(
    output_dir: Path,
    placements: list[dict],
    camera: dict,
    render_scale: int = _RENDER_SCALE,
    base_image_path: "Path | None" = None,
    out_path: "Path | None" = None,
) -> Path:
    """Composite furniture placements onto the wall_mounted render."""
    import trimesh
    from PIL import Image

    # Base render: use explicit override if provided, else search candidates
    if base_image_path is not None:
        base_path = Path(base_image_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Specified base render not found: {base_path}")
    else:
        candidates = [
            # Wall objects on the reference-textured walls (rendered from the
            # scene's current — promoted ref — wall textures): most complete base.
            output_dir / "wall_mounted" / "placements" / "render_objects_placed_ref_texture.png",
            output_dir / "wall_mounted" / "placements" / "render_objects_placed.png",
            # Reference-baked walls without objects (fallback)
            output_dir / "render_ref_texture.png",
            output_dir / "render_final.png",
            output_dir / "render.png",
        ]
        base_path = next((p for p in candidates if p.exists()), None)
        if base_path is None:
            raise FileNotFoundError(f"No base render found in {output_dir}")
    print(f"[render_furn] Base render: {base_path.name}")

    # Downscale the base to the (possibly capped) camera resolution first — this
    # is what bounds the framebuffer at scale=1 (the per-iteration render path).
    base_img = _load_base_capped(base_path, camera)

    # Upsample base image for higher-res output
    src_w, src_h = base_img.size
    # Cap total render pixels: the _RENDER_SCALE=3 upsample is for SMALL refs; on a
    # large ref (e.g. 6067×3467) 3× = 189 MP, which OOMs / DecompressionBombs and
    # kills the run. Auto-reduce the scale so the output stays within a safe budget.
    _max_render_px = int(os.environ.get("SCENEWEAVE_MAX_RENDER_MP", "60")) * 1_000_000
    while render_scale > 1 and (src_w * render_scale) * (src_h * render_scale) > _max_render_px:
        render_scale -= 1
        print(f"[render_furn] render too large for {src_w}×{src_h} — reducing scale → {render_scale}×")
    tgt_w, tgt_h = src_w * render_scale, src_h * render_scale
    if render_scale != 1:
        base_img = base_img.resize((tgt_w, tgt_h), Image.LANCZOS)
        print(f"[render_furn] Upsampled {src_w}×{src_h} → {tgt_w}×{tgt_h} (scale={render_scale}×)")

    buf = np.array(base_img, dtype=np.uint8).copy()
    H, W = buf.shape[:2]

    cam_pos  = np.array(camera["position_m"],  dtype=np.float64)
    look_at  = np.array(camera["look_at_m"],   dtype=np.float64)
    up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)

    # fx/cx/cy scale with the image — hfov stays fixed, pixel dimensions grow
    hfov = float(camera["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx   = W / 2.0
    cy   = H / 2.0

    # Z-buffer for per-pixel depth testing (same as wall-mounted renderer)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    for p in placements:
        # ── Flat-quad carpet: back-project pixels from texture onto Y=0 ──────
        # compute_carpet_placement returns world_corners + carpet_img but no
        # glb_path.  Render by sampling the carpet image per output pixel.
        if p.get("is_carpet") and "world_corners" in p:
            print(f"[render_furn]  idx={p['index']:>2} {p['type']:<15} (flat quad)")
            _render_flat_carpet_quad(
                buf, p, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy,
            )
            continue

        # ── Wall-mounted fallback: pre-transformed verts from _transform_glb_to_wall
        if p.get("wall_mounted") and "_verts" in p:
            verts = p["_verts"]
            faces = p["_faces"]
            vert_colors = p["_vc"]
            print(f"[render_furn]  idx={p['index']:>2} {p['type']:<15} "
                  f"(wall-mounted) {len(faces)} faces  wall={p.get('wall')}")
            proj = []
            for v in verts:
                px, py, zc = _project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
                proj.append((px, py, zc))
            for fi in range(len(faces)):
                i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
                px0, py0, zc0 = proj[i0]
                px1, py1, zc1 = proj[i1]
                px2, py2, zc2 = proj[i2]
                if zc0 <= NEAR_CLIP or zc1 <= NEAR_CLIP or zc2 <= NEAR_CLIP:
                    continue
                pts = np.array([[px0, py0, zc0], [px1, py1, zc1], [px2, py2, zc2]], dtype=np.float32)
                col = np.array([vert_colors[i0], vert_colors[i1], vert_colors[i2]], dtype=np.uint8)
                _rasterize_vc_tri(buf, zbuf, pts, col)
            continue

        # ── Carpet GLB: per-pixel direct projection ───────────────────────────
        if p.get("skip_floor_transform") and p.get("skip_ground_removal"):
            glb_path = Path(p["glb_path"])
            print(f"[render_furn]  idx={p['index']:>2} {p['type']:<15} (direct projection)")
            if _render_carpet_direct(buf, str(glb_path), cam_pos, right_v, up_c_v,
                                     fwd_v, fx, cx, cy):
                continue
            # fall through to mesh path if direct render failed

        # ── GLB mesh object ───────────────────────────────────────────────────
        if p.get("wall_mounted"):
            print(f"[render_furn] idx={p.get('index')} {p.get('type')}: "
                  f"wall-mounted entry — skipping")
            continue
        if "glb_path" not in p:
            print(f"[render_furn] idx={p.get('index')} {p.get('type')}: "
                  f"no glb_path — skipping")
            continue
        glb_path = _resolve_glb(p["glb_path"], output_dir)
        if not glb_path.exists():
            print(f"[render_furn] GLB missing: {glb_path} — skipping")
            continue
        try:
            # Load as scene to preserve sub-mesh visuals (baseColorTexture accessible)
            scene_or_mesh = trimesh.load(str(glb_path), force="scene")
            if isinstance(scene_or_mesh, trimesh.Scene):
                sub_meshes = list(scene_or_mesh.geometry.values())
                mesh = trimesh.util.concatenate(sub_meshes) if len(sub_meshes) > 1 else sub_meshes[0]
            else:
                mesh = scene_or_mesh
            if not p.get("skip_ground_removal"):
                result = _remove_ground_faces(mesh)
                mesh = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            print(f"[render_furn] Load failed {glb_path.name}: {e}")
            continue
        bounds       = mesh.bounds
        scale        = np.array(p["scale"],        dtype=np.float64)
        R            = np.array(p["rotation_3x3"], dtype=np.float64)
        pos          = np.array(p["position_m"],   dtype=np.float64)
        wall_aff     = p.get("wall_affinity", "centre")
        # Objects positioned by facing_toward (chair at desk) aren't wall-bound
        if p.get("_facing_applied"):
            wall_aff = "centre"
        _FLUSH_TYPES_R = {"sofa", "armchair", "bookcase", "cabinet", "desk", "bed"}
        p_back_flush = p.get(
            "back_flush",
            wall_aff != "centre" and p.get("type", "") in _FLUSH_TYPES_R,
        )
        room_w_cam, _ = _get_room_dims(Path(p["glb_path"]), camera)
        faces = mesh.faces
        vert_colors = _get_vertex_colors(mesh)

        print(f"[render_furn]  idx={p['index']:>2} {p['type']:<15} "
              f"{len(faces)} faces  pos={np.round(pos,2).tolist()}"
              f"  wall_aff={wall_aff}")

        if p.get("skip_floor_transform"):
            verts = mesh.vertices.astype(np.float64).copy()
            verts[:, 1] = 0.0
        else:
            p_obb_R = np.array(p["obb_R"], dtype=np.float64) if "obb_R" in p else None
            verts = _apply_floor_transform(
                mesh.vertices, bounds, scale, R, pos,
                wall_affinity=wall_aff, room_w=room_w_cam, back_flush=p_back_flush,
                obb_R=p_obb_R,
                skip_level_base=bool(p.get("flat_on_floor")),
                skip_wall_snap=bool(p.get("_pos_override")),
            )

        # Project all vertices once
        proj = []
        for v in verts:
            px, py, zc = _project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
            proj.append((px, py, zc))

        for fi in range(len(faces)):
            i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])

            px0, py0, zc0 = proj[i0]
            px1, py1, zc1 = proj[i1]
            px2, py2, zc2 = proj[i2]
            if zc0 <= NEAR_CLIP or zc1 <= NEAR_CLIP or zc2 <= NEAR_CLIP:
                continue

            # No back-face cull: Y-axis rotations (Ry(π) for back-wall objects)
            # flip winding-order normals relative to the camera.  Z-buffer depth
            # testing already ensures only the closest surface is shown.
            pts = np.array([[px0, py0, zc0], [px1, py1, zc1], [px2, py2, zc2]], dtype=np.float32)
            col = np.array([vert_colors[i0], vert_colors[i1], vert_colors[i2]], dtype=np.uint8)
            _rasterize_vc_tri(buf, zbuf, pts, col)

    if out_path is None:
        out_path = output_dir / "furniture" / "render_furniture_placed.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(buf).save(str(out_path))
    print(f"[render_furn] Saved → {out_path}")
    return out_path


# ── Save world-space GLBs ──────────────────────────────────────────────────────


def render_furniture_pyrender(
    output_dir: Path,
    camera: dict,
    placements: "list[dict] | None" = None,
    base_image_path: "Path | None" = None,
    scene_glb: "Path | None" = None,
    out_path: "Path | None" = None,
    render_scale: int = 2,
) -> "Path | None":
    """Composite the placed furniture onto the base render using pyrender for a
    PROPER textured + lit perspective render (per-pixel texture sampling), instead
    of the flat per-vertex rasteriser.

    Renders ``scene_with_furniture.glb`` (furniture already in world coords with
    textures baked) from the VGGT camera, then alpha-composites the furniture
    pixels over the reference-textured background.  The pyrender projection is
    matched exactly to ``_project_vertex`` (fx = W / (2·tan(hfov/2)), right =
    fwd×up) so the furniture lands where the placement put it.

    Returns the output path, or None if pyrender/EGL is unavailable (caller keeps
    the rasteriser output).
    """
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender  # noqa: F401
    except Exception as e:
        print(f"[render_furn_pyr] pyrender unavailable ({e}) — keeping rasteriser output")
        return None
    import trimesh
    from PIL import Image

    furn_dir = Path(output_dir) / "furniture"
    if scene_glb is None:
        scene_glb = furn_dir / "scene_with_furniture.glb"
    scene_glb = Path(scene_glb)
    if not scene_glb.exists():
        print(f"[render_furn_pyr] scene GLB missing: {scene_glb} — skipping")
        return None

    # ── Base background (same priority chain as the rasteriser) ──────────────
    if base_image_path is not None:
        base_path = Path(base_image_path)
    else:
        cands = [
            # Wall objects (window) on the reference-textured walls — the most
            # complete base.  render_objects_placed.png is rendered from the
            # scene's current wall textures, which are the promoted reference
            # textures once texture_from_reference has run.
            Path(output_dir) / "wall_mounted" / "placements" / "render_objects_placed_ref_texture.png",
            Path(output_dir) / "wall_mounted" / "placements" / "render_objects_placed.png",
            Path(output_dir) / "render_ref_texture.png",
            Path(output_dir) / "render_final.png",
            Path(output_dir) / "render.png",
        ]
        base_path = next((p for p in cands if p.exists()), None)
    if base_path is None or not Path(base_path).exists():
        print(f"[render_furn_pyr] no base render found — skipping")
        return None
    print(f"[render_furn_pyr] Base render: {Path(base_path).name}")

    # Downscale the base to the (possibly capped) camera resolution so the
    # pyrender OffscreenRenderer framebuffer (viewport_width×viewport_height = W×H)
    # is bounded — this is the 21 MP OOM site.
    base_img = _load_base_capped(base_path, camera)
    W0, H0 = base_img.size
    W, H = W0 * render_scale, H0 * render_scale
    if render_scale != 1:
        base_img = base_img.resize((W, H), Image.LANCZOS)
    base_arr = np.array(base_img, dtype=np.uint8)

    # ── Camera: match _project_vertex exactly ────────────────────────────────
    cam_pos = np.array(camera["position_m"], dtype=np.float64)
    look_at = np.array(camera["look_at_m"],  dtype=np.float64)
    up_w    = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_w)
    hfov = float(camera["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))     # same fx as rasteriser
    yfov = 2.0 * np.arctan((H / 2.0) / fx)                # → fy == fx (square px)

    # ── Flat carpets first, onto the BASE ────────────────────────────────────
    # pyrender skips carpets (they have no glb_path — just world_corners + a 2D
    # texture), so draw them onto the base image here using the same flat-quad
    # sampler as the rasteriser.  The furniture rendered below composites over
    # this base, so any piece sitting ON the carpet correctly occludes it.
    if placements is not None:
        for _cp in placements:
            if _cp.get("is_carpet") and _cp.get("world_corners"):
                try:
                    _render_flat_carpet_quad(base_arr, _cp, cam_pos,
                                             right_v, up_c_v, fwd_v,
                                             fx, W / 2.0, H / 2.0)
                    print(f"[render_furn_pyr] composited carpet (idx={_cp.get('index')}) onto base")
                except Exception as _ce:
                    print(f"[render_furn_pyr] carpet composite skipped: {_ce}")

    # OpenGL camera-to-world pose: +X=right, +Y=up, +Z=-fwd (pyrender looks -Z)
    pose = np.eye(4)
    pose[:3, 0] = right_v
    pose[:3, 1] = up_c_v
    pose[:3, 2] = -fwd_v
    pose[:3, 3] = cam_pos

    pr_scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                              ambient_light=[1.0, 1.0, 1.0])
    n_added = 0

    if placements is not None:
        # Build from individual placement GLBs so we can (a) apply the SAME
        # _apply_floor_transform as the rasteriser and (b) swap the dark baked
        # GLB texture for the bright inpainted PNG — matching the rasteriser's
        # colours instead of rendering the murky generated albedo.
        from trimesh.visual import TextureVisuals
        from trimesh.visual.material import PBRMaterial
        from PIL import Image as _PIL
        for p in placements:
            if p.get("is_carpet") or "glb_path" not in p:
                continue
            gp = _resolve_glb(p["glb_path"], output_dir)
            if not gp.exists():
                continue
            try:
                _loaded = trimesh.load(str(gp), force="scene")
                subs = list(_loaded.geometry.values()) if isinstance(_loaded, trimesh.Scene) else [_loaded]
                mesh = max(subs, key=lambda g: len(g.faces))
                if not p.get("skip_ground_removal"):
                    res = _remove_ground_faces(mesh)
                    mesh = res[0] if isinstance(res, tuple) else res
            except Exception as e:
                print(f"[render_furn_pyr] load failed {gp.name}: {e}")
                continue

            bounds   = mesh.bounds
            scale    = np.array(p["scale"],        dtype=np.float64)
            R        = np.array(p["rotation_3x3"], dtype=np.float64)
            posv     = np.array(p["position_m"],   dtype=np.float64)
            wall_aff = p.get("wall_affinity", "centre")
            if p.get("_facing_applied"):
                wall_aff = "centre"
            _FLUSH = {"sofa", "armchair", "bookcase", "cabinet", "desk", "bed"}
            back_flush = p.get("back_flush",
                               wall_aff != "centre" and p.get("type", "") in _FLUSH)
            room_w_cam, _ = _get_room_dims(gp, camera)
            if p.get("skip_floor_transform"):
                verts = mesh.vertices.astype(np.float64).copy()
                verts[:, 1] = 0.0
            else:
                obb_R = np.array(p["obb_R"], dtype=np.float64) if "obb_R" in p else None
                verts = _apply_floor_transform(
                    mesh.vertices, bounds, scale, R, posv,
                    wall_affinity=wall_aff, room_w=room_w_cam, back_flush=back_flush,
                    obb_R=obb_R, skip_level_base=bool(p.get("flat_on_floor")),
                    skip_wall_snap=bool(p.get("_pos_override")),
                )

            # Generated meshes bake very dark albedo (mean ≈ 60/255).  Brighten the
            # GLB's OWN texture (keeping its UV atlas mapping intact — swapping in
            # the inpaint PNG would mis-map, since its layout ≠ the GLB atlas) so
            # the rendered fabric reads at a natural tone.
            visual = mesh.visual
            try:
                mat = getattr(mesh.visual, "material", None)
                tex = getattr(mat, "baseColorTexture", None) if mat is not None else None
                if tex is not None and getattr(mesh.visual, "uv", None) is not None:
                    tarr = np.asarray(tex, dtype=np.float32)
                    m = float(tarr[..., :3].mean()) or 1.0
                    # Lift the dark baked albedo to a moderate tone WITHOUT washing
                    # it out — a gentler target + lower cap keeps dark fabrics dark
                    # (vs tan) and preserves fabric↔wood contrast.  pyrender's light
                    # supplies the rest of the brightness.
                    gain = float(np.clip(105.0 / m, 1.0, 2.0))
                    if gain > 1.01:
                        tarr[..., :3] = np.clip(tarr[..., :3] * gain, 0, 255)
                        visual = TextureVisuals(
                            uv=np.array(mesh.visual.uv),
                            material=PBRMaterial(
                                baseColorTexture=_PIL.fromarray(tarr.astype(np.uint8)),
                                metallicFactor=0.0, roughnessFactor=1.0),
                        )
            except Exception:
                visual = mesh.visual

            placed = trimesh.Trimesh(vertices=verts, faces=mesh.faces,
                                     visual=visual, process=False)
            try:
                pr_scene.add(pyrender.Mesh.from_trimesh(placed, smooth=False), pose=np.eye(4))
                n_added += 1
            except Exception as e:
                print(f"[render_furn_pyr] skip {p.get('type')}: {e}")
    else:
        loaded = trimesh.load(str(scene_glb), force="scene")
        geoms = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
        for g in geoms:
            if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
                continue
            try:
                pr_scene.add(pyrender.Mesh.from_trimesh(g, smooth=False), pose=np.eye(4))
                n_added += 1
            except Exception as e:
                print(f"[render_furn_pyr] skip geom: {e}")

    if n_added == 0:
        print("[render_furn_pyr] no renderable furniture geometry — skipping")
        return None

    pr_scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W / H,
                                            znear=0.05, zfar=100.0), pose=pose)
    # Object textures (Gemini inpaint → Hunyuan bake) ALREADY carry realistic
    # baked-in lighting, so ANY added directional light double-lights them and
    # washes the colour. The flat _object_preview (pure ambient, no directional)
    # reads most natural, so match it: full ambient, a token directional only so
    # pyrender always has a light source.
    pr_scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=0.1), pose=pose)

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    try:
        color, depth = r.render(pr_scene)
    finally:
        r.delete()

    # Composite furniture (depth > 0) over the background.
    mask = depth > 0
    out = base_arr.copy()
    out[mask] = color[mask, :3]

    if out_path is None:
        out_path = furn_dir / "render_furniture_placed.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(out_path))
    print(f"[render_furn_pyr] textured composite → {out_path}  "
          f"({n_added} geoms, {W}×{H}, {int(mask.sum())} furniture px)")
    return out_path


def render_placement_buildup(
    output_dir: Path,
    camera: dict,
    placements: list[dict],
    base_image_path: "Path | None" = None,
    frame_ms: int = 700,
    width: int = 960,
) -> "Path | None":
    """Save an animated GIF of the placement built up ONE OBJECT AT A TIME.

    Frame 0 is the empty (reference-textured) room; each subsequent frame adds the
    next placed object via the pyrender textured composite.  Gives a per-step
    "watch the room fill in" animation of the final placement.

    Returns the GIF path, or None if pyrender is unavailable / nothing to render.
    """
    try:
        import pyrender  # noqa: F401
    except Exception as e:
        print(f"[buildup] pyrender unavailable ({e}) — skipping animation")
        return None
    from PIL import Image

    out = Path(output_dir)
    renderable = [p for p in placements
                  if not p.get("is_carpet") and "glb_path" in p
                  and Path(p["glb_path"]).exists()]
    if not renderable:
        print("[buildup] no renderable objects — skipping animation")
        return None

    # Resolve the empty-room base (same chain the composite uses).
    if base_image_path is not None:
        base_path = Path(base_image_path)
    else:
        base_path = next((p for p in [
            out / "wall_mounted" / "placements" / "render_objects_placed_ref_texture.png",
            out / "wall_mounted" / "placements" / "render_objects_placed.png",
            out / "render_ref_texture.png",
            out / "render_final.png",
        ] if p.exists()), None)
    if base_path is None or not Path(base_path).exists():
        print("[buildup] no base render — skipping animation")
        return None

    tmp = out / "furniture" / "_anim"
    tmp.mkdir(parents=True, exist_ok=True)
    # Frame 0 (empty room) must match the capped-resolution frames produced by
    # render_furniture_pyrender below, or the GIF frames would differ in size.
    frames = [_load_base_capped(base_path, camera)]
    for k in range(1, len(renderable) + 1):
        fp = tmp / f"f{k:02d}.png"
        try:
            render_furniture_pyrender(out, camera, placements=renderable[:k],
                                      base_image_path=base_path, out_path=fp,
                                      render_scale=1)
            frames.append(Image.open(fp).convert("RGB"))
        except Exception as e:
            print(f"[buildup] frame {k} failed ({e})")

    if len(frames) < 2:
        return None
    W, H = frames[0].size
    th = int(width * H / W)
    frames = [f.resize((width, th), Image.LANCZOS) for f in frames]
    durations = [frame_ms] * len(frames)
    durations[0] = max(frame_ms, 900)
    durations[-1] = 2000   # hold the finished room
    gif = out / "furniture" / "render_placement_buildup.gif"
    frames[0].save(str(gif), save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"[buildup] animation → {gif}  ({len(frames)} frames)")
    return gif


def save_placed_geometry(output_dir: Path, placements: list[dict]) -> None:
    """Combine the base scene OBJ with all placed furniture and save as a GLB.

    Base scene: wall_mounted/placements/walls_with_objects.obj (walls + wall-mounted objects).
    Output: furniture/scene_with_furniture.glb
    """
    import trimesh

    furn_dir = output_dir / "furniture"
    furn_dir.mkdir(parents=True, exist_ok=True)

    all_meshes: list[trimesh.Trimesh] = []

    # Load base scene (walls + wall-mounted objects)
    base_obj = output_dir / "wall_mounted" / "placements" / "walls_with_objects.obj"
    if base_obj.exists():
        try:
            base = trimesh.load(str(base_obj), force="mesh")
            if isinstance(base, trimesh.Scene):
                base = trimesh.util.concatenate(base.dump())
            all_meshes.append(base)
            print(f"[geo_save] Base scene loaded: {base_obj.name} ({len(base.vertices)} verts)")
        except Exception as e:
            print(f"[geo_save] Warning: could not load base scene: {e}")
    else:
        print(f"[geo_save] Warning: base scene not found at {base_obj}")

    # Apply transforms and append each placed furniture mesh
    for p in placements:
        if p.get("is_carpet"):
            continue  # flat quad — no mesh to save in GLB scene
        # Wall-mounted fallback: pre-transformed verts already available
        if p.get("wall_mounted") and "_verts" in p:
            placed_mesh = trimesh.Trimesh(
                vertices=p["_verts"],
                faces=p["_faces"],
                process=False,
            )
            all_meshes.append(placed_mesh)
            print(f"[geo_save]  idx={p['index']:>2} {p['type']:<15} (wall-mounted) added to scene")
            continue
        glb_path = _resolve_glb(p["glb_path"], output_dir)
        if not glb_path.exists():
            print(f"[geo_save] GLB missing: {glb_path} — skipping")
            continue
        try:
            # Load WITHOUT force="mesh" so per-geometry textures survive.
            # When the GLB is a Scene (multiple textured parts), we keep the
            # first (largest) textured submesh and apply the transform to it.
            # trimesh.util.concatenate would weld them but drops textures.
            _loaded = trimesh.load(str(glb_path))
            if isinstance(_loaded, trimesh.Scene):
                # Pick the largest geometry as the primary textured mesh
                _geoms = list(_loaded.geometry.values())
                if not _geoms:
                    print(f"[geo_save] Empty scene {glb_path.name} — skipping")
                    continue
                mesh = max(_geoms, key=lambda g: len(g.faces))
            else:
                mesh = _loaded
            if not p.get("skip_ground_removal"):
                result = _remove_ground_faces(mesh)
                mesh = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            print(f"[geo_save] Load failed {glb_path.name}: {e}")
            continue

        bounds   = mesh.bounds
        scale    = np.array(p["scale"],        dtype=np.float64)
        R        = np.array(p["rotation_3x3"], dtype=np.float64)
        # Durable orientation lock: when a placement carries `_front_lock_deg`,
        # force its yaw here — this is the FINAL bake step, so the lock wins over
        # whatever VLM front-detection / yaw-refine wrote into rotation_3x3 during
        # furniture_placement, and it persists across full re-runs (the field
        # survives in the placement JSON). Yaw is absolute about +Y, applied to
        # the mesh's upright GLB. Set alongside `_pos_override` to also pin place.
        _flock = p.get("_front_lock_deg")
        if _flock is not None:
            R = _rotation_y(np.radians(float(_flock)))
        pos      = np.array(p["position_m"],   dtype=np.float64)
        wall_aff = p.get("wall_affinity", "centre")
        # Objects positioned by facing_toward (chair at desk) aren't wall-bound
        if p.get("_facing_applied"):
            wall_aff = "centre"
        _FLUSH_TYPES_S = {"sofa", "armchair", "bookcase", "cabinet", "desk", "bed"}
        s_back_flush = p.get(
            "back_flush",
            wall_aff != "centre" and p.get("type", "") in _FLUSH_TYPES_S,
        )
        if p.get("skip_floor_transform"):
            verts = mesh.vertices.astype(np.float64).copy()
            verts[:, 1] = 0.0   # flatten to floor
        else:
            s_obb_R = np.array(p["obb_R"], dtype=np.float64) if "obb_R" in p else None
            verts = _apply_floor_transform(
                mesh.vertices, bounds, scale, R, pos,
                wall_affinity=wall_aff, back_flush=s_back_flush,
                obb_R=s_obb_R,
                skip_level_base=bool(p.get("flat_on_floor")),
                skip_wall_snap=bool(p.get("_pos_override")),
            )

        placed_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=mesh.faces,
            visual=mesh.visual,
            process=False,
        )
        all_meshes.append(placed_mesh)
        print(f"[geo_save]  idx={p['index']:>2} {p['type']:<15} added to scene")

    if not all_meshes:
        print("[geo_save] No meshes to save.")
        return

    # Build a Scene with each mesh as a separate geometry node so textures
    # and per-mesh materials are preserved on export (concatenate() drops them).
    scene = trimesh.Scene()
    for i, m in enumerate(all_meshes):
        scene.add_geometry(m, node_name=f"mesh_{i:03d}")
    out_path = furn_dir / "scene_with_furniture.glb"
    scene.export(str(out_path))
    print(f"[geo_save] Scene saved → {out_path}  ({len(all_meshes)} geometries)")

    # Re-texture the room shell (walls/floor/ceiling) + wall-mounted objects so
    # the saved GLB is FULLY textured.  The furniture is already textured, but
    # geo_save loads the BAKED, flat ``walls_with_objects.obj`` for walls + wall
    # objects.  This is the same retexture the assemble stage applies — running
    # it here means the post-furniture GLB isn't flat-walled.  Best-effort: if
    # the textured wall/floor assets aren't present yet, leave the flat walls.
    try:
        from object_placement.assemble_scene_glb import retexture_scene as _retexture_scene
        _retexture_scene(output_dir, out_path)
        print(f"[geo_save] Re-textured walls/floor/ceiling + wall objects → {out_path}")
    except Exception as _rex:
        print(f"[geo_save] wall/floor retexture skipped (flat walls): {_rex}")


def _unify_group_chair_meshes(furniture_dir: "Path",
                              order: "list[dict]",
                              seg_by_idx: "dict[int, dict]") -> None:
    """Make chairs in the same dining/seating GROUP share ONE representative mesh.

    Reads ``placement_analysis.json`` (in ``furniture_dir``) to map each chair's
    index → its ``group`` string, groups the chairs by group, and for any group
    with >= 3 chairs picks the chair whose GLB exists and has the most faces as
    the representative.  Every other chair in that group has its ``glb_file``
    rewritten (both on the in-memory placement-order entry consumed by the
    placement loop AND on the segment_results entry) to the representative's
    ``glb_file`` so they all render with the same mesh.  Each chair keeps its own
    position/rotation (arrangement around the table is untouched).

    Fully defensive: any missing file / unreadable mesh / absent group is skipped
    silently and never crashes placement.
    """
    _CHAIR_TYPES = {"chair", "armchair", "dining_chair"}

    def _resolve_glb(glb_file: str) -> "Path | None":
        """Resolve a segment glb_file ('objects/foo.glb' or 'foo.glb') under furniture/."""
        if not glb_file:
            return None
        p = Path(glb_file)
        if p.is_absolute():
            return p if p.exists() else None
        cand = furniture_dir / p
        if cand.exists():
            return cand
        # Handle bare filename when stored value lacks the objects/ prefix (or vice-versa)
        cand2 = furniture_dir / "objects" / p.name
        if cand2.exists():
            return cand2
        return cand if cand.exists() else None

    try:
        analysis_p = furniture_dir / "placement_analysis.json"
        if not analysis_p.exists():
            return
        try:
            analysis = json.loads(analysis_p.read_text())
        except Exception:
            return

        # index → group (from placement_analysis), restricted to chair types.
        idx_group: dict[int, str] = {}
        for e in analysis.get("placement_order", []):
            if e.get("type") in _CHAIR_TYPES and e.get("group"):
                try:
                    idx_group[int(e.get("index"))] = e.get("group")
                except Exception:
                    continue

        if not idx_group:
            return

        # Group chairs by their group string.
        groups: dict[str, list[int]] = {}
        for idx, grp in idx_group.items():
            groups.setdefault(grp, []).append(idx)

        # Build quick lookup into the in-memory placement-order entries we'll mutate.
        order_by_idx: dict[int, dict] = {}
        for e in order:
            try:
                order_by_idx.setdefault(int(e.get("index")), e)
            except Exception:
                continue

        def _glb_file_for(idx: int) -> str:
            """Current glb_file for a chair index (prefer placement entry, fall back to segment)."""
            ent = order_by_idx.get(idx)
            if ent and ent.get("glb_file"):
                return ent.get("glb_file")
            seg = seg_by_idx.get(idx)
            if seg and seg.get("glb_file"):
                return seg.get("glb_file")
            return ""

        for grp, idxs in groups.items():
            if len(idxs) < 3:
                continue

            # Pick representative: chair whose GLB exists with the most faces.
            rep_idx = None
            rep_glb_file = None
            rep_faces = -1
            for idx in idxs:
                gf = _glb_file_for(idx)
                abs_p = _resolve_glb(gf)
                if abs_p is None:
                    continue
                try:
                    import trimesh
                    n_faces = len(trimesh.load(str(abs_p), force="mesh").faces)
                except Exception:
                    continue
                if n_faces > rep_faces:
                    rep_faces = n_faces
                    rep_idx = idx
                    rep_glb_file = gf

            if rep_idx is None or not rep_glb_file:
                continue

            # Rewrite every OTHER chair in the group to reuse the representative.
            for idx in idxs:
                if idx == rep_idx:
                    continue
                ent = order_by_idx.get(idx)
                if ent is not None:
                    ent["glb_file"] = rep_glb_file
                seg = seg_by_idx.get(idx)
                if seg is not None:
                    seg["glb_file"] = rep_glb_file

            print(f"[group-mesh] {grp}: {len(idxs)} chairs → reuse {rep_glb_file}")
    except Exception:
        # Never crash placement over a cosmetic mesh-unification step.
        return


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(output_dir: str | Path, n_objects: int = 0, carpet_only: bool = False,
        only_indices: list[int] | None = None,
        vlm_refine_iters: int = 10,
        animate: bool = False,
        animate_gif_width: int = 960,
        animate_frame_ms: int = 1000,
        vggt_refine: bool = True,
        vggt_blend: float = 0.5,
        vggt_max_shift_m: float = 0.6,
        vggt_visible_face_offset: bool = False,
        vggt_lateral_weight: float = 1.0,
        # Depth refinement ON: compare the object's mask depth-centroid in the
        # reconstructed-scene VGGT render vs the reference photo's VGGT cloud and
        # move the object along the camera-forward axis to match (the silhouette
        # mask_align handles the lateral move + scale-to-mask).  Was 0.0 (depth
        # correction computed but discarded); enabled so the per-object depth is
        # pinned by the reference point cloud, not just the 2D silhouette.
        vggt_depth_weight:   float = 0.5,
        vggt_method: str = "render_diff",
        iter_max_dev_m: float = 0.15,
        vlm_fine_rotation: bool = True,
        vlm_fine_rotation_max_deg: int = 25,
        vlm_fine_rotation_iters: int = 3,
        vlm_post_verify: bool = True,
        vlm_scene_review: bool = True,
        allow_rotation_types: frozenset = frozenset(),
        base_image_path: "Path | str | None" = None,
        use_vlm: bool = True) -> Path:
    out_root   = Path(output_dir)
    if not use_vlm:
        vlm_post_verify      = False
        vlm_scene_review     = False
        vlm_fine_rotation    = False

    # ── Ablation toggles (env-controlled; see outputs/_ablations) ─────────────
    import os as _abl_os
    def _abl(name): return _abl_os.environ.get(name) == "1"
    if _abl("SCENEWEAVE_ABLATE_NO_VGGT_REFINE"):   # wo_vggt_refinement: drop depth/vggt, keep silhouette
        vggt_refine = False; vggt_depth_weight = 0.0
    if _abl("SCENEWEAVE_ABLATE_NO_DEPTH"):         # drop only the forward-depth correction
        vggt_depth_weight = 0.0
    if _abl("SCENEWEAVE_ABLATE_NO_SILH"):          # drop silhouette lateral (rely on vggt/back-proj)
        vggt_blend = 1.0
    if _abl("SCENEWEAVE_ABLATE_NO_REFINE"):        # wo_placement_refinement: neither silh nor depth/vggt
        vggt_refine = False; vggt_depth_weight = 0.0; vggt_blend = 1.0
        _abl_os.environ["SCENEWEAVE_ABLATE_NO_MASK_ALIGN"] = "1"
    if _abl("SCENEWEAVE_ABLATE_NO_FEEDBACK"):      # wo_feedback_loop: VLM only for initial placement
        vlm_post_verify = False; vlm_scene_review = False; vlm_fine_rotation = False
    analysis_p = out_root / "furniture" / "placement_analysis.json"

    # Prefer VGGT-calibrated camera; fall back to VLM estimate
    # (SCENEWEAVE_ABLATE_VLM_CAMERA forces the raw VLM camera → wo_camera_calibration)
    camera_p = (out_root / "camera_vggt.json"
                if ((out_root / "camera_vggt.json").exists()
                    and not _abl("SCENEWEAVE_ABLATE_VLM_CAMERA"))
                else out_root / "camera.json")

    for p in (analysis_p, camera_p):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(analysis_p) as f:
        analysis = json.load(f)

    # Load persisted front-face directions (survives across runs when VLM is down)
    _load_front_cache(out_root / "furniture")
    with open(camera_p) as f:
        cam = json.load(f)

    # ── Render-resolution cap ────────────────────────────────────────────────
    # A native 21 MP camera (6067×3467) OOMs the offscreen renderer (it allocates
    # a width_px×height_px framebuffer for every placement iteration). Cap the
    # render resolution to SCENEWEAVE_PLACE_MAX_PX (default 8 MP). All render /
    # placement functions derive intrinsics from width_px/height_px, so this
    # scales the 2-D projection ONLY and leaves the 3-D world placement unchanged.
    # The cap factor _place_res_scale must ALSO be applied to every pixel-space
    # silhouette target (box_px / mask bbox), which lives in the native photo
    # resolution, so the projected render and the snap target stay the same size.
    _cam_W0, _cam_H0 = int(cam["width_px"]), int(cam["height_px"])
    cam = cap_camera_resolution(cam)
    _place_res_scale = float(cam["width_px"]) / float(_cam_W0) if _cam_W0 else 1.0

    def _scale_box_px(_box):
        """Scale a [x1,y1,x2,y2] pixel bbox into the capped render resolution."""
        if not _box or len(_box) != 4:
            return _box
        return [float(_box[0]) * _place_res_scale, float(_box[1]) * _place_res_scale,
                float(_box[2]) * _place_res_scale, float(_box[3]) * _place_res_scale]

    # Enrich placement entries with per-segment data (crop_file, box_px) from
    # segment_results.json which is generated earlier in the pipeline.
    seg_results_p = out_root / "furniture" / "segment_results.json"
    seg_by_idx: dict[int, dict] = {}
    if seg_results_p.exists():
        with open(seg_results_p) as f:
            seg_data = json.load(f)
        for s in seg_data.get("segments", []):
            # Scale the native-resolution silhouette bbox into the capped render
            # resolution so mask_align / silhouette-fit compare like-for-like.
            if _place_res_scale != 1.0 and s.get("box_px"):
                s["box_px"] = _scale_box_px(s["box_px"])
            seg_by_idx[s["index"]] = s

    order: list[dict] = analysis.get("placement_order", [])
    # Placement-analysis entries carry their OWN box_px (set during VLM ordering)
    # at native resolution — scale them too, before they flow into placement.
    if _place_res_scale != 1.0:
        for _e in order:
            if _e.get("box_px"):
                _e["box_px"] = _scale_box_px(_e["box_px"])
    if not order:
        print("[place_furn] No objects in placement_analysis.json — nothing to do.")
        return out_root / "furniture"

    # Chairs in the same dining/seating GROUP (>=3) share ONE representative mesh
    # so the set reads as a matching suite.  Mutates `order` entries (and the
    # segment_results in `seg_by_idx`) in place BEFORE the per-object placement
    # loop loads each entry's glb_file.  Positions/rotations are untouched.
    _unify_group_chair_meshes(out_root / "furniture", order, seg_by_idx)

    to_place = [] if carpet_only else (order if n_objects <= 0 else order[:n_objects])
    # Restrict to a specific set of indices when requested (for targeted re-runs).
    if only_indices is not None:
        _idx_set = set(only_indices)
        to_place = [e for e in to_place if e.get("index") in _idx_set]
        print(f"[place_furn] only_indices={only_indices} → {len(to_place)} entries selected")
    # Filter out excluded items (non-furniture detections like outdoor scenery)
    excluded = [e for e in to_place if e.get("exclude")]
    if excluded:
        for ex in excluded:
            reason = ex.get("exclude_reason", "no reason")
            print(f"[place_furn] EXCLUDED idx={ex.get('index')} {ex.get('type')}: {reason}")
        to_place = [e for e in to_place if not e.get("exclude")]

    # Auto-exclude objects whose GLB or inpainted image doesn't exist.
    # Missing inpainted image means the object was never reconstructed
    # (e.g. outdoor scenery through a window that got segmented but not
    # inpainted).  Missing GLB is similar.
    furn_dir = out_root / "furniture"
    inpaint_dir = furn_dir / "inpainted"
    _valid = []
    for entry in to_place:
        idx_e = entry.get("index", -1)
        obj_type_e = entry.get("type", "?")

        # Check GLB exists
        _has_verified_glb = False
        gf = entry.get("glb_file", "")
        if gf:
            glb_abs = Path(gf) if Path(gf).is_absolute() else (furn_dir / gf)
            if not glb_abs.exists():
                print(f"[place_furn] AUTO-EXCLUDED idx={idx_e} "
                      f"{obj_type_e}: GLB not found ({glb_abs.name})")
                continue
            _has_verified_glb = True

        # Check inpainted image exists (carpets use a different path).
        # Skip this heuristic entirely when a verified GLB is already present —
        # the GLB is the authoritative artefact for placement.  The inpaint
        # check exists to filter "non-furniture" segments that never got a
        # generation pass, so it's redundant once a GLB has been produced.
        # Otherwise, honour an explicit `inpaint_file` on the segment so
        # synthesised entries (mirrored items that re-use another idx's PNG)
        # aren't dropped just because no `inpaint_<idx>_*.png` file exists.
        if not _has_verified_glb and obj_type_e.lower() not in ("carpet", "rug"):
            seg_e = seg_by_idx.get(idx_e, {})
            explicit_inpaint = entry.get("inpaint_file") or seg_e.get("inpaint_file")
            if explicit_inpaint and (inpaint_dir / explicit_inpaint).exists():
                pass  # explicit reused PNG exists — accept
            else:
                inpaint_pattern = f"inpaint_{idx_e:02d}_*"
                inpaint_matches = list(inpaint_dir.glob(inpaint_pattern)) if inpaint_dir.exists() else []
                if not inpaint_matches:
                    print(f"[place_furn] AUTO-EXCLUDED idx={idx_e} "
                          f"{obj_type_e}: no inpainted image found (likely non-furniture)")
                    continue

        _valid.append(entry)
    to_place = _valid
    if carpet_only:
        print("[place_furn] carpet-only mode — skipping all non-carpet objects")
    else:
        print(f"[place_furn] Placing {len(to_place)} object(s) from placement_analysis.json …")

    # Load existing wall-mounted placements to check if wall_mounted objects
    # were already placed — if not, we'll place them using wall-mounted logic.
    wm_placements_p = out_root / "wall_mounted" / "placements" / "object_placements.json"
    existing_wm_types: set[str] = set()
    if wm_placements_p.exists():
        try:
            with open(wm_placements_p) as f:
                for wmp in json.load(f):
                    existing_wm_types.add(wmp.get("type", "").lower())
        except Exception:
            pass

    # Merge segment_results data (crop_file, box_px) into placement entries
    for entry in to_place:
        seg = seg_by_idx.get(entry.get("index", -1), {})
        for key in ("crop_file", "box_px", "mask_file", "canvas_file",
                    "yaw_offset_deg", "roll_offset_deg",
                    "position_offset_xz", "skip_ground_removal",
                    "flat_on_floor", "target_width_m"):
            if key not in entry and key in seg:
                entry[key] = seg[key]

    # Load detilt metadata (if produced by detilt_and_orient.py) so we can
    # swap each source GLB for its canonical (de-tilted, front-aligned) version.
    detilt_p = out_root / "furniture" / "detilt_results.json"
    detilt_by_idx: dict[int, dict] = {}
    if detilt_p.exists():
        try:
            with open(detilt_p) as f:
                _dd = json.load(f)
            for s in _dd.get("segments", []):
                detilt_by_idx[int(s["index"])] = s
            print(f"[place_furn] detilt metadata: {len(detilt_by_idx)} canonical GLB(s) available")
        except Exception as e:
            print(f"[place_furn] detilt_results.json unreadable: {e}")

    # Resolve GLB paths relative to output_dir.
    # Use RAW meshes (objects/) for floor furniture — the OBB rotation baked
    # into canonical GLBs (objects_detilted/) can introduce tilt artifacts that
    # prevent objects from sitting flat on the floor.
    # The detilt_results front_local_axis is still read for front detection.
    for entry in to_place:
        gf = entry.get("glb_file", "")
        if not gf:
            continue
        src_abs = Path(gf) if Path(gf).is_absolute() else (out_root / "furniture" / gf)
        det = detilt_by_idx.get(int(entry.get("index", -1)))
        # Determine if this object is wall-mounted: explicit flag, or thin mesh
        _is_wm = entry.get("wall_mounted", False)
        if not _is_wm and src_abs.exists():
            try:
                import trimesh as _tm_glb
                _m_glb = _tm_glb.load(str(src_abs), force="mesh", process=False)
                _ext_sorted = sorted(_m_glb.bounds[1] - _m_glb.bounds[0])
                if _ext_sorted[2] > 1e-6 and _ext_sorted[0] / _ext_sorted[2] < 0.15:
                    _is_wm = True  # thin mesh → will be wall-mounted
            except Exception:
                pass
        if det:
            fla = det.get("front_local_axis", "")
            if fla:
                entry["_detilt_front_axis"] = fla
            # Store detilt rotation for transforming front axis to raw frame
            if det.get("detilt_R"):
                entry["_detilt_R"] = det["detilt_R"]
            if _is_wm:
                # Wall-mounted objects: keep canonical GLB (OBB alignment helps
                # with wall mounting; front yaw is already baked in)
                canon_rel = det.get("canonical_glb")
                if canon_rel:
                    canon_abs = out_root / canon_rel
                    if canon_abs.exists():
                        entry["glb_file"] = str(canon_abs)
                        entry["is_canonical_glb"] = True
                        print(f"  [glb_resolve] idx={entry.get('index')}: "
                              f"wall-mounted → using canonical GLB")
                        continue
            else:
                # Floor furniture: use raw mesh (OBB baking introduces tilt)
                if fla:
                    print(f"  [glb_resolve] idx={entry.get('index')}: "
                          f"detilt front={fla} (using raw mesh, not canonical)")
        entry["glb_file"] = str(src_abs)

    # Minimum floor distance between two placed objects of the same type (m).
    # Prevents duplicate detections (same sofa/table picked up twice at similar
    # bbox positions) from being placed on top of each other.
    # Minimum XZ distance between two placed objects of the same type.
    # Only dedup truly overlapping objects (same bbox projected to same spot).
    # Keep thresholds low — the VLM layout already handles spatial arrangement.
    _DEDUP_DIST: dict[str, float] = {
        "sofa":         0.30,
        "chair":        0.30,
        "armchair":     0.30,
        "coffee_table": 0.25,
        "dining_table": 0.50,
        "desk":         0.50,
        "bookcase":     0.30,
        "cabinet":      0.30,
        "bed":          0.50,
        "plant":        0.20,
    }
    _DEDUP_FALLBACK = 0.25

    # Pre-filter: remove duplicate detections with heavily overlapping bboxes.
    # If two objects of the same type share > 70% IoU in their detection bbox,
    # keep only the first one (higher priority in placement order).
    _kept_entries: list[dict] = []
    for entry in to_place:
        box = entry.get("box_px")
        if box is None or len(box) != 4:
            _kept_entries.append(entry)
            continue
        bx1, by1, bx2, by2 = box
        dup = False
        for kept in _kept_entries:
            if kept.get("type") != entry.get("type"):
                continue
            kb = kept.get("box_px")
            if kb is None or len(kb) != 4:
                continue
            kx1, ky1, kx2, ky2 = kb
            # Compute IoU
            ix1 = max(bx1, kx1); iy1 = max(by1, ky1)
            ix2 = min(bx2, kx2); iy2 = min(by2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = max(1, (bx2 - bx1) * (by2 - by1))
            area_b = max(1, (kx2 - kx1) * (ky2 - ky1))
            iou = inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0
            if iou > 0.70:
                # Prefer the lower-index duplicate (detected first, usually
                # higher confidence).  If the new entry has a lower index,
                # replace the kept one instead of skipping the new one.
                if entry["index"] < kept["index"]:
                    print(f"  [bbox_dedup] idx={kept['index']} {kept.get('type')} "
                          f"overlaps idx={entry['index']} (IoU={iou:.2f}) — replacing with lower-index")
                    _kept_entries.remove(kept)
                    # don't mark as dup — let it be appended below
                else:
                    print(f"  [bbox_dedup] idx={entry['index']} {entry.get('type')} "
                          f"overlaps idx={kept['index']} (IoU={iou:.2f}) — skipping duplicate")
                    dup = True
                break
        if not dup:
            _kept_entries.append(entry)
    to_place = _kept_entries

    # ── Per-type reference-bbox pre-scan ──────────────────────────────────────
    # For each object type, find the entry with the LARGEST bbox (above the
    # reliability threshold).  Small-bbox entries of the same type then borrow
    # that reference bbox for size estimation, so size doesn't depend on
    # placement order or per-entry back-projection accuracy.
    _type_ref_box: dict[str, list] = {}
    _type_ref_area: dict[str, float] = {}
    for _e in to_place:
        _bp = _e.get("box_px")
        if not _bp or len(_bp) != 4:
            continue
        _area = (float(_bp[2]) - float(_bp[0])) * (float(_bp[3]) - float(_bp[1]))
        if _area < _MIN_RELIABLE_BBOX_AREA_PX:
            continue
        _t = _e.get("type", "")
        if _area > _type_ref_area.get(_t, 0):
            _type_ref_area[_t] = _area
            _type_ref_box[_t]  = list(_bp)
    # Tag every small-bbox entry with the reference bbox (and entry index) for
    # its type, so compute_floor_placement can use it for size estimation.
    for _e in to_place:
        _bp = _e.get("box_px")
        if not _bp or len(_bp) != 4:
            continue
        _area = (float(_bp[2]) - float(_bp[0])) * (float(_bp[3]) - float(_bp[1]))
        if _area >= _MIN_RELIABLE_BBOX_AREA_PX:
            continue
        _t = _e.get("type", "")
        _ref = _type_ref_box.get(_t)
        if _ref is None or _ref == list(_bp):
            continue   # no reliable sibling of same type, or self
        _e["_ref_box_px"] = _ref
        print(f"[ref_bbox] idx={_e.get('index')} {_t}: small bbox area={_area:.0f} px² "
              f"→ borrowing size from reference bbox area="
              f"{_type_ref_area[_t]:.0f} px² ({_ref})")

    # ── Collision helpers (defined early — used by both placement loop and
    # post-placement resolution) ─────────────────────────────────────────────
    def _col_xz_half_extents(p):
        sm = p.get("size_m", {})
        # width_m/depth_m are semantic (along-front vs perpendicular); undo
        # the [axis_swap] relabel to get raw mesh-X/mesh-Z halves before
        # combining with R, which rotates the RAW local axes.
        _pf = p.get("_local_front", [0, 0, -1])
        if abs(float(_pf[0])) > abs(float(_pf[2])):
            hx_local = sm.get("depth_m", 0.5) / 2.0
            hz_local = sm.get("width_m", 0.5) / 2.0
        else:
            hx_local = sm.get("width_m", 0.5) / 2.0
            hz_local = sm.get("depth_m", 0.5) / 2.0
        R = np.array(p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
        hx = abs(float(R[0][0])) * hx_local + abs(float(R[0][2])) * hz_local
        hz = abs(float(R[2][0])) * hx_local + abs(float(R[2][2])) * hz_local
        return hx, hz

    def _poly_collides(pa: dict, pb: dict, margin: float = 0.0) -> bool:
        """Return True if pa and pb collide in the XZ floor plane.
        Uses SAT on convex hull footprints when available; falls back to AABB."""
        hull_a = pa.get("_xz_hull_pts")
        hull_b = pb.get("_xz_hull_pts")
        if hull_a is not None and hull_b is not None:
            R_a  = np.array(pa.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
            R_b  = np.array(pb.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
            sc_a = np.array(pa.get("scale", [1, 1, 1]), dtype=np.float64)
            sc_b = np.array(pb.get("scale", [1, 1, 1]), dtype=np.float64)
            return _poly_collides_sat(
                hull_a, pa["position_m"][0], pa["position_m"][2], R_a, sc_a,
                hull_b, pb["position_m"][0], pb["position_m"][2], R_b, sc_b,
                margin=margin,
            )
        # AABB fallback
        hx_a, hz_a = _col_xz_half_extents(pa)
        hx_b, hz_b = _col_xz_half_extents(pb)
        return ((hx_a + hx_b + margin) > abs(pa["position_m"][0] - pb["position_m"][0])
                and (hz_a + hz_b + margin) > abs(pa["position_m"][2] - pb["position_m"][2]))

    placed_by_idx: dict[int, dict] = {}   # index → placement, for on-top-of lookups
    placements = []

    # ── VGGT ground-truth refinement setup ─────────────────────────────────────
    # Two modes:
    #   render_diff (default): per-iteration, render the current scene + placement,
    #     run VGGT on it, compare masked centroids. Bias-cancelling but slow
    #     (one VGGT inference per furniture).
    #   global_calib:         compute one Sim(3) (photo VGGT pseudo-world →
    #     world meters via walls.obj) and reuse it for all furniture. Fast but
    #     subject to depth bias.
    _vggt_calib = None
    _vggt_ref_depth = None
    _vggt_ref_K = None
    _vggt_ref_E = None
    _vggt_pseudo_to_metric = 1.0
    _vggt_pseudo_to_metric_poly = None
    if vggt_refine:
        if vggt_method == "render_diff":
            try:
                ref_depth_path = out_root / "vggt" / "depth_0.npy"
                ref_cam_path   = out_root / "camera_vggt.json"
                if ref_depth_path.exists() and ref_cam_path.exists():
                    from object_placement.furniture.vggt_refine import (
                        _camera_to_KE as _vggt_camera_to_KE)
                    _vggt_ref_depth = np.load(ref_depth_path)
                    _vggt_ref_K, _vggt_ref_E, _, _ = _vggt_camera_to_KE(
                        json.loads(ref_cam_path.read_text()))
                    # VGGT depths aren't metric.  Try the per-pixel polynomial
                    # calibration first (handles near-field non-uniform
                    # compression), fall back to the global scalar otherwise.
                    try:
                        _vggt_pseudo_to_metric_poly = (
                            _vggt_compute_pseudo_to_metric_polyfit(out_root)
                        )
                    except Exception as _pe:
                        print(f"[vggt-refine] depth polyfit failed ({_pe}); "
                              f"falling back to global scalar")
                        _vggt_pseudo_to_metric_poly = None
                    _s = _vggt_compute_pseudo_to_metric_scale(out_root)
                    _vggt_pseudo_to_metric = float(_s) if _s and _s > 0 else 1.0
                    print(f"[vggt-refine] mode=render_diff (per-iter VGGT-on-render); "
                          f"ref depth {_vggt_ref_depth.shape}; "
                          f"pseudo→metric scalar={_vggt_pseudo_to_metric:.3f}  "
                          f"poly={'enabled' if _vggt_pseudo_to_metric_poly is not None else 'disabled (using scalar)'}")
                else:
                    print(f"[vggt-refine] missing photo VGGT outputs at {ref_depth_path}; "
                          f"disabling VGGT refine")
                    vggt_refine = False
            except Exception as _re:
                print(f"[vggt-refine] render_diff setup failed ({_re}); disabling")
                vggt_refine = False
        else:
            try:
                _vggt_calib = _vggt_compute_calibration(out_root)
            except Exception as _vc_e:
                print(f"[vggt-refine] calibration failed ({_vc_e}); disabling")
                _vggt_calib = None

    # ── Animation setup ────────────────────────────────────────────────────────
    _anim_frame_paths: list[str] = []
    if animate:
        import tempfile as _tmpmod
        _anim_root = out_root / "furniture" / "anim_tmp"
        _anim_root.mkdir(parents=True, exist_ok=True)
        _anim_tmp_dir = _tmpmod.mkdtemp(prefix="furn_anim_", dir=str(_anim_root))

        # Resolve room dimensions + ceiling once for the schematics.
        try:
            _anim_room_w, _anim_room_d = _get_room_dims(
                Path(to_place[0].get("glb_file", ".")) if to_place else Path("."), cam)
        except Exception:
            _anim_room_w, _anim_room_d = 5.0, 5.0
        _anim_ceiling_h = 2.7
        try:
            _fp_path = out_root / "floorplan_analysis.json"
            if _fp_path.exists():
                _anim_ceiling_h = float(json.loads(_fp_path.read_text())
                                        .get("room", {})
                                        .get("ceiling_height_m", 2.7))
        except Exception:
            pass

        def _anim_topdown_schematic(current_placements: list[dict],
                                     highlight_idx: int | None,
                                     img_w: int = 600, img_h: int = 600) -> Path:
            """Bird's-eye floor-plan schematic: room outline + each piece as a
            rotated bbox with idx label and front-direction arrow."""
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            from matplotlib.patches import Rectangle as _Rect, FancyArrow as _Arrow

            _fig, _ax = _plt.subplots(figsize=(img_w / 100, img_h / 100), dpi=100)
            _ax.add_patch(_Rect((0, 0), _anim_room_w, _anim_room_d,
                                fill=False, edgecolor="black", linewidth=2))
            for _pp in current_placements:
                if _pp.get("is_carpet") or _pp.get("wall_mounted"):
                    continue
                _sm = _pp.get("size_m") or {}
                _w = float(_sm.get("width_m", 0.5))
                _d = float(_sm.get("depth_m", 0.5))
                _cx = float(_pp["position_m"][0])
                _cz = float(_pp["position_m"][2])
                _R = np.array(_pp.get("rotation_3x3", np.eye(3).tolist()),
                              dtype=np.float64)
                _yaw_deg = -np.degrees(float(np.arctan2(_R[0, 2], _R[0, 0])))
                _is_hi = (_pp.get("index") == highlight_idx)
                _ax.add_patch(_Rect(
                    (_cx - _w / 2.0, _cz - _d / 2.0), _w, _d,
                    angle=_yaw_deg, rotation_point="center",
                    fill=True, alpha=0.55,
                    facecolor=("gold" if _is_hi else "lightsteelblue"),
                    edgecolor=("crimson" if _is_hi else "navy"),
                    linewidth=(2.5 if _is_hi else 1.0)))
                _ax.text(_cx, _cz,
                         f"{_pp.get('index')}:{(_pp.get('type','?') or '?')[:7]}",
                         ha="center", va="center", fontsize=7,
                         fontweight=("bold" if _is_hi else "normal"))
                _lf = np.asarray(_pp.get("_local_front", [0.0, 0.0, 1.0]),
                                 dtype=np.float64)
                _fw = _R @ _lf
                _ax.add_patch(_Arrow(_cx, _cz, _fw[0] * 0.18, _fw[2] * 0.18,
                                     width=0.06,
                                     color=("red" if _is_hi else "darkgreen"),
                                     alpha=0.85))
            _ax.set_xlim(-0.2, _anim_room_w + 0.2)
            _ax.set_ylim(-0.2, _anim_room_d + 0.2)
            _ax.set_aspect("equal")
            _ax.invert_yaxis()
            _ax.set_xlabel("X (m)")
            _ax.set_ylabel("Z (back→front)")
            _ax.grid(True, alpha=0.25)
            _ax.set_title("top-down")
            _fig.tight_layout()
            _td_path = Path(_anim_tmp_dir) / f"_td_{len(_anim_frame_paths):04d}.png"
            _fig.savefig(_td_path, dpi=100)
            _plt.close(_fig)
            return _td_path

        def _anim_frontview_schematic(current_placements: list[dict],
                                       highlight_idx: int | None,
                                       img_w: int = 600, img_h: int = 600) -> Path:
            """Elevation schematic looking AT the back wall from inside the
            room: X × Y bounding rectangles for each piece, depth-ordered
            (objects with larger z drawn on top so foreground occludes
            background)."""
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            from matplotlib.patches import Rectangle as _Rect

            _fig, _ax = _plt.subplots(figsize=(img_w / 100, img_h / 100), dpi=100)
            # Wall outline: X × ceiling_h at z=0 plane
            _ax.add_patch(_Rect((0, 0), _anim_room_w, _anim_ceiling_h,
                                fill=False, edgecolor="black", linewidth=2))
            # Sort back→front so foreground draws on top
            _sorted = sorted(
                [p for p in current_placements
                 if not p.get("is_carpet") and not p.get("wall_mounted")],
                key=lambda p: float(p["position_m"][2]))
            for _pp in _sorted:
                _sm = _pp.get("size_m") or {}
                _w  = float(_sm.get("width_m", 0.5))
                _h  = float(_sm.get("height_m", _sm.get("eff_h", 0.5)))
                _d  = float(_sm.get("depth_m", 0.5))
                _cx = float(_pp["position_m"][0])
                _cz = float(_pp["position_m"][2])
                _R = np.array(_pp.get("rotation_3x3", np.eye(3).tolist()),
                              dtype=np.float64)
                # Effective X-extent after yaw (axis-aligned bbox of rotated rect)
                _eff_w = abs(float(_R[0, 0])) * _w + abs(float(_R[0, 2])) * _d
                # Y on floor: object base is at y=0, top at h
                _is_hi = (_pp.get("index") == highlight_idx)
                # Depth-based fade for non-highlighted pieces (closer = brighter)
                _depth_norm = 0.0
                if _anim_room_d > 0.01:
                    _depth_norm = float(np.clip(_cz / _anim_room_d, 0.0, 1.0))
                _alpha = 0.45 + 0.35 * _depth_norm   # back=0.45, front=0.80
                _face = ("gold" if _is_hi
                         else (0.62 + 0.25 * _depth_norm,    # R
                               0.69 + 0.20 * _depth_norm,    # G
                               0.84 + 0.10 * _depth_norm))   # B
                _ax.add_patch(_Rect(
                    (_cx - _eff_w / 2.0, 0.0), _eff_w, _h,
                    fill=True, alpha=_alpha,
                    facecolor=_face,
                    edgecolor=("crimson" if _is_hi else "navy"),
                    linewidth=(2.5 if _is_hi else 1.0)))
                _ax.text(_cx, _h * 0.5,
                         f"{_pp.get('index')}:{(_pp.get('type','?') or '?')[:7]}",
                         ha="center", va="center", fontsize=7,
                         fontweight=("bold" if _is_hi else "normal"))
                # Annotate Z position below the rect (top-down depth context)
                _ax.text(_cx, -0.05, f"z={_cz:.2f}", ha="center", va="top",
                         fontsize=6, color="dimgray")
            _ax.set_xlim(-0.2, _anim_room_w + 0.2)
            _ax.set_ylim(-0.25, _anim_ceiling_h + 0.2)
            _ax.set_aspect("equal")
            _ax.set_xlabel("X (m)")
            _ax.set_ylabel("Y (height, m)")
            _ax.grid(True, alpha=0.25)
            _ax.set_title("front view (elevation)")
            _fig.tight_layout()
            _fv_path = Path(_anim_tmp_dir) / f"_fv_{len(_anim_frame_paths):04d}.png"
            _fig.savefig(_fv_path, dpi=100)
            _plt.close(_fig)
            return _fv_path

        def _anim_frame_render(current_placements: list[dict], label: str = "",
                               highlight_idx: int | None = None) -> None:
            """3-panel composite: [camera view | top-down | front view] with
            caption bar at top.  Highlighted object draws bold/yellow in the
            schematics."""
            try:
                from PIL import Image as _PImg, ImageDraw as _PDraw
                _idx = len(_anim_frame_paths)
                _std_path = Path(_anim_tmp_dir) / f"_std_{_idx:04d}.png"
                # Prefer the pyrender textured composite so the step animation
                # shows real materials; fall back to the flat rasteriser if
                # pyrender/EGL is unavailable.
                _rendered = None
                try:
                    _rendered = render_furniture_pyrender(
                        out_root, cam, placements=current_placements,
                        base_image_path=base_image_path, out_path=_std_path,
                        render_scale=1)
                except Exception:
                    _rendered = None
                if _rendered is None:
                    render_furniture(out_root, current_placements, cam,
                                     render_scale=1, out_path=_std_path,
                                     base_image_path=base_image_path)
                _std = _PImg.open(_std_path).convert("RGB")
                _h = _std.height
                _td_path = _anim_topdown_schematic(current_placements, highlight_idx)
                _fv_path = _anim_frontview_schematic(current_placements, highlight_idx)
                _td = _PImg.open(_td_path).convert("RGB")
                _fv = _PImg.open(_fv_path).convert("RGB")
                _td_w = int(_td.width * _h / _td.height)
                _fv_w = int(_fv.width * _h / _fv.height)
                _td_r = _td.resize((_td_w, _h), _PImg.LANCZOS)
                _fv_r = _fv.resize((_fv_w, _h), _PImg.LANCZOS)
                _composed = _PImg.new("RGB",
                                       (_std.width + _td_w + _fv_w, _h),
                                       "white")
                _composed.paste(_std, (0, 0))
                _composed.paste(_td_r, (_std.width, 0))
                _composed.paste(_fv_r, (_std.width + _td_w, 0))
                _bar_h = 36
                _cap = _PImg.new("RGB", (_composed.width, _bar_h), "black")
                _draw = _PDraw.Draw(_cap)
                _txt = f"#{_idx:04d}  {label}"
                if highlight_idx is not None:
                    _txt += f"  (focus idx={highlight_idx})"
                _draw.text((8, 8), _txt, fill="white")
                _final = _PImg.new("RGB",
                                    (_composed.width, _composed.height + _bar_h),
                                    "white")
                _final.paste(_cap, (0, 0))
                _final.paste(_composed, (0, _bar_h))
                _fp = str(Path(_anim_tmp_dir) / f"{_idx:04d}_{label}.png")
                _final.save(_fp)
                _anim_frame_paths.append(_fp)
                # Cleanup intermediates
                for _ip in (_std_path, _td_path, _fv_path):
                    try:
                        _ip.unlink(missing_ok=True)
                    except Exception:
                        pass
                print(f"  [animate] frame {_idx + 1}: {label}"
                      + (f" (focus idx={highlight_idx})" if highlight_idx is not None else ""))
            except Exception as _ae:
                print(f"  [animate] render failed ({label}): {_ae}")
    else:
        _anim_frame_render = None

    for entry in to_place:
        idx      = entry["index"]
        obj_type = entry.get("type", "?")
        on_top   = entry.get("on_top_of")

        # Wall-mounted objects: use wall-mounted placement logic if not already
        # placed during the wall-mounted phase.  Trigger on EITHER:
        #   - wall_affinity == "wall_mounted", OR
        #   - type is inherently wall-mounted (art, tapestry, macramé, etc.)
        #   - mesh is extremely flat (depth < 15% of width) → likely wall art
        #     mislabeled as furniture (e.g. macramé tagged as "bookcase")
        _WALL_MOUNTED_TYPES = {
            "wall_art", "art", "painting", "tapestry", "macrame", "macramé",
            "wall_hanging", "wall_shelf", "wall_clock", "mirror", "sconce",
            "frame", "poster", "wall_decor",
        }
        # Flat-mesh heuristic: load GLB and check if depth is negligible.
        # SKIP for box-shaped types where the 3D generator often collapses depth on
        # legitimately deep furniture — those go through the regular path
        # so the per-type min-depth floor in _size_and_scale_furniture can
        # restore a reasonable thickness.
        _BOX_FURNITURE_TYPES = {
            "cabinet", "dresser", "bookcase", "sideboard", "console",
            "wardrobe", "tv_stand", "shelf", "nightstand", "chest",
            "buffet", "credenza",
        }
        _flat_mesh_wall_mounted = False
        _glb_file = entry.get("glb_file", "")
        if (_glb_file and Path(_glb_file).exists()
                and obj_type.lower() not in ("carpet", "rug")
                and obj_type.lower() not in _BOX_FURNITURE_TYPES):
            try:
                import trimesh as _tm
                _m = _tm.load(str(_glb_file), force="mesh", process=False)
                _b = _m.bounds  # [[min_x,min_y,min_z],[max_x,max_y,max_z]]
                _extents = _b[1] - _b[0]
                _sorted_ext = sorted(_extents)
                # If the thinnest dimension is < 10% of the widest, it's flat
                if _sorted_ext[2] > 1e-6 and _sorted_ext[0] / _sorted_ext[2] < 0.10:
                    _flat_mesh_wall_mounted = True
                    print(f"  [flat_detect] idx={idx} {obj_type}: extents "
                          f"{np.round(_sorted_ext, 3).tolist()} — thin mesh, treating as wall-mounted")
            except Exception:
                pass
        is_wall_mounted = (
            entry.get("wall_affinity") == "wall_mounted"
            or obj_type.lower().replace("-", "_") in _WALL_MOUNTED_TYPES
            or _flat_mesh_wall_mounted
        )
        if is_wall_mounted:
            obj_type_lower = obj_type.lower().replace("_", " ").replace("-", " ")
            # Check if already placed by wall-mounted pipeline (approximate type match)
            already_placed = any(
                t in obj_type_lower or obj_type_lower in t
                for t in existing_wm_types
            )
            if already_placed:
                print(f"\n[place_furn] idx={idx}  {obj_type}  — wall_mounted, "
                      f"already placed in wall-mounted phase, skipping")
                continue
            print(f"\n[place_furn] idx={idx}  {obj_type}  — wall_mounted fallback placement")
            wm_p = compute_wall_mounted_placement(
                entry, cam, out_root,
                detilt_meta=detilt_by_idx.get(idx),
                use_vlm=use_vlm,
            )
            if wm_p is not None:
                placements.append(wm_p)
                print(f"  wall-mounted placement: wall={wm_p['wall']}  "
                      f"world_pt={np.round(wm_p['world_pt'], 2).tolist()}")
            else:
                print(f"  [wm_fallback] placement failed — skipping")
            continue
        # ── Parse opening_relation for orientation hints ───────────────────
        # The analysis VLM often describes "back against the left wall" etc.
        # Use this to override the rotation affinity so the object faces the
        # correct direction deterministically, instead of relying on the
        # render-time VLM to figure it out.
        _opening = (entry.get("opening_relation") or "") + " " + (entry.get("notes") or "")
        _opening_lower = _opening.lower()
        # Parse "back against the <wall>" to set orient_affinity.
        # If the back is against a wall, the front faces the opposite direction.
        # This is deterministic and doesn't depend on VLM front detection.
        import re as _re_orient
        _back_wall_m = _re_orient.search(
            r"back\s+against\s+the\s+(left|right|back|front)\s+wall",
            _opening_lower,
        )
        if _back_wall_m and "_orient_affinity" not in entry:
            _bw = _back_wall_m.group(1)
            # back against left wall → front faces +X (right into room) → affinity "left"
            # back against right wall → front faces -X (left into room) → affinity "right"
            # back against back wall → front faces +Z (toward camera) → affinity "back"
            _orient_map = {"left": "left", "right": "right", "back": "back", "front": "centre"}
            entry["_orient_affinity"] = _orient_map.get(_bw, "centre")
            print(f"  [orient_hint] back against {_bw} wall → orient_affinity={entry['_orient_affinity']}")

        # ── Small-bbox size override ─────────────────────────────────────────
        # If the bbox is too small for reliable perspective size estimation,
        # find an already-placed object of the same type and reuse its size.
        _entry_bp = entry.get("box_px", [0, 0, 0, 0])
        _entry_bbox_area = (float(_entry_bp[2]) - float(_entry_bp[0])) * \
                           (float(_entry_bp[3]) - float(_entry_bp[1]))
        if _entry_bbox_area < _MIN_RELIABLE_BBOX_AREA_PX:
            _ref_placed = next(
                (v for v in placed_by_idx.values()
                 if v.get("type") == obj_type and "size_m" in v),
                None,
            )
            if _ref_placed is not None:
                entry["_ref_size_m"] = _ref_placed["size_m"]
                print(f"  [small_bbox] idx={idx} {obj_type}: "
                      f"bbox area={_entry_bbox_area:.0f} px² — "
                      f"using reference size from idx={_ref_placed.get('index')} "
                      f"({_ref_placed['size_m']['width_m']:.2f}×"
                      f"{_ref_placed['size_m']['depth_m']:.2f}×"
                      f"{_ref_placed['size_m']['height_m']:.2f} m)")
            else:
                entry.pop("_ref_size_m", None)  # fall back to default inside compute_floor_placement
                print(f"  [small_bbox] idx={idx} {obj_type}: "
                      f"bbox area={_entry_bbox_area:.0f} px² — no reference yet, using default size")

        print(f"\n[place_furn] idx={idx}  {obj_type}"
              + (f"  (on top of {on_top})" if on_top is not None else ""))

        # Wrap floor placement in try/except so one failure doesn't skip all
        # remaining objects (e.g. VLM timeout, trimesh load error, etc.)
        try:
            support_y  = 0.0
            support_xz = None
            # on_top_of relationships are currently disabled for furniture
            # placement — all objects are placed on the ground floor.
            if False and on_top is not None:
                sup = placed_by_idx.get(on_top)
                if sup is not None:
                    support_y  = sup["position_m"][1] + sup["eff_h"]
                    support_xz = (sup["position_m"][0], sup["position_m"][2])
                    print(f"  [on_top] support idx={on_top}  surface_y={support_y:.3f} m")
                else:
                    print(f"  [on_top] WARNING: supporting object idx={on_top} not yet placed")

            p = compute_floor_placement(entry, cam, support_y=support_y, support_xz=support_xz,
                                        allow_rotation_types=allow_rotation_types,
                                        use_vlm=use_vlm)
            if p is None:
                continue

            # ── Capture silhouette anchor RIGHT NOW, before any refinement ────
            # `compute_floor_placement` returns the position derived from the
            # segmentation mask. All subsequent steps (vlm_orient, mask_align,
            # fine_rotation, VGGT, post_mask_clearance, sr_collision, ...) can
            # only drift the object away from this anchor by at most the cap
            # (`_ITER_MAX_DEV` / `_FINAL_MAX_DEV` / `0.40 m`). Without this
            # capture, drift accumulates because the fallback at the bottom of
            # the iteration captures a *post-drift* position as the anchor,
            # which makes subsequent caps measure from the wrong reference.
            p["_mask_origin_xz"] = [p["position_m"][0], p["position_m"][2]]

            # Dedup: skip if another object of the same type is already placed nearby
            if on_top is None:
                pos_xz = np.array([p["position_m"][0], p["position_m"][2]])
                min_dist = _DEDUP_DIST.get(obj_type, _DEDUP_FALLBACK)
                too_close = False
                for existing in placements:
                    if existing["type"] != obj_type:
                        continue
                    if "position_m" not in existing:
                        continue
                    ex_xz = np.array([existing["position_m"][0], existing["position_m"][2]])
                    if np.linalg.norm(pos_xz - ex_xz) < min_dist:
                        print(f"  [dedup] skipped — too close to already-placed {obj_type} "
                              f"at {np.round(existing['position_m'],2).tolist()}")
                        too_close = True
                        break
                if too_close:
                    continue

            # ── Initial collision slide ──────────────────────────────────────
            # If the initial position collides with already-placed objects,
            # slide along the wall axis (or camera depth for centre objects)
            # in small steps until clear.  This prevents the later global
            # collision resolution from having to make large pushes.
            # Record which objects we collide with BEFORE sliding, so
            # mask_align knows to ignore them in collision blocking.
            # init_slide fires for floor-standing items.  Previously gated on
            # `on_top is None`, but that misses the case where a tall floor
            # plant is wrongly tagged on_top_of a small coffee table (the
            # plant base is on the floor regardless, and it can collide
            # with neighbouring sofas/chairs).  When `on_top` IS set, we
            # still run init_slide but exclude the support object from
            # collision detection — sitting "on" the support is intentional;
            # colliding with anything ELSE is not.
            if placements:
                _ic_wa = p.get("wall_affinity", "centre")
                _ic_sm = p.get("size_m", {})
                _ic_R = np.array(p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
                _ic_hx_l = _ic_sm.get("width_m", 0.5) / 2.0
                _ic_hz_l = _ic_sm.get("depth_m", 0.5) / 2.0
                _ic_hx = abs(float(_ic_R[0][0])) * _ic_hx_l + abs(float(_ic_R[0][2])) * _ic_hz_l
                _ic_hz = abs(float(_ic_R[2][0])) * _ic_hx_l + abs(float(_ic_R[2][2])) * _ic_hz_l
                # (_ic_hx/_ic_hz still used for room-boundary check below)

                # Record pre-slide collisions so mask_align can ignore them.
                # When on_top is set, exclude that idx — collision with the
                # support is the desired arrangement, not something to slide
                # away from.
                _pre_slide_colliders: set[int] = set()
                for _ep in placements:
                    if _ep.get("is_carpet") or _ep.get("wall_mounted"):
                        continue
                    if on_top is not None and _ep.get("index") == on_top:
                        continue
                    if _poly_collides(p, _ep):
                        _pre_slide_colliders.add(_ep.get("index", -1))
                if _pre_slide_colliders:
                    # Store on placement so mask_align can access it
                    p["_pre_slide_colliders"] = list(_pre_slide_colliders)

                # Determine slide axis based on wall affinity
                if _ic_wa == "back":
                    _slide_axis = 0  # slide along X
                elif _ic_wa in ("left", "right"):
                    _slide_axis = 2  # slide along Z
                else:
                    _slide_axis = 0  # centre: slide along X by default

                _ic_step = 0.05  # 5cm steps
                # Default: 50 cm.  Plants often need a longer slide because
                # they're tagged on_top_of a small coffee table (placing
                # them in a corner) but actually rest on the floor and can
                # collide with the corner sofa for the full sofa depth
                # (~2.5 m).  Allow up to 1.5 m for plants so they can clear.
                _is_plant = p.get("type") == "plant"
                _ic_max_slide = 1.50 if _is_plant else 0.50
                _ic_max_slides = int(_ic_max_slide / _ic_step)
                _ic_room_w, _ic_room_d = _get_room_dims(Path(p.get("glb_path", ".")), cam)
                _ic_room_lim = _ic_room_w if _slide_axis == 0 else _ic_room_d
                _ic_half = _ic_hx if _slide_axis == 0 else _ic_hz
                _ic_origin = p["position_m"][_slide_axis]
                def _ic_any_collide():
                    return any(
                        _poly_collides(p, _ep)
                        for _ep in placements
                        if (not _ep.get("is_carpet")
                            and not _ep.get("wall_mounted")
                            and _ep.get("index") != on_top)
                    )

                for _ic_attempt in range(_ic_max_slides):
                    _ic_collides = _ic_any_collide()
                    if not _ic_collides:
                        break
                    _new_val = p["position_m"][_slide_axis] + _ic_step
                    if _new_val + _ic_half > _ic_room_lim - 0.05:
                        _ic_collides = True  # hit room boundary
                        break
                    p["position_m"][_slide_axis] = _new_val
                if _ic_attempt > 0 and not _ic_collides:
                    print(f"  [init_slide] slid +{_ic_attempt * _ic_step:.2f}m along "
                          f"axis={_slide_axis} to avoid collision")
                    if _anim_frame_render is not None:
                        _anim_frame_render(placements + [p],
                                           label=f"idx{idx}_{obj_type}_post_init_slide",
                                           highlight_idx=idx)
                elif _ic_collides:
                    # Reset and try negative direction
                    p["position_m"][_slide_axis] = _ic_origin
                    for _ic_attempt2 in range(_ic_max_slides):
                        _ic_collides2 = _ic_any_collide()
                        if not _ic_collides2:
                            break
                        _new_val2 = p["position_m"][_slide_axis] - _ic_step
                        if _new_val2 - _ic_half < 0.05:
                            _ic_collides2 = True  # hit room boundary
                            break
                        p["position_m"][_slide_axis] = _new_val2
                    if _ic_attempt2 > 0 and not _ic_collides2:
                        print(f"  [init_slide] slid -{_ic_attempt2 * _ic_step:.2f}m along "
                              f"axis={_slide_axis} to avoid collision")
                        if _anim_frame_render is not None:
                            _anim_frame_render(placements + [p],
                                               label=f"idx{idx}_{obj_type}_post_init_slide",
                                               highlight_idx=idx)
                    elif _ic_collides2:
                        # Wall-axis sliding exhausted in both directions.
                        # Final attempt for plants/decor: slide OFF the wall
                        # (perpendicular axis) — when both ±wall-axis are
                        # blocked by a sofa that runs the full wall length
                        # plus a room boundary, peeling the plant off the
                        # wall can clear the sofa's X/Z while keeping it
                        # near its silhouette anchor.
                        p["position_m"][_slide_axis] = _ic_origin  # reset wall axis
                        _perp_axis = 2 if _slide_axis == 0 else 0
                        _ic_perp_half = _ic_hz if _perp_axis == 2 else _ic_hx
                        _ic_perp_lim = _ic_room_d if _perp_axis == 2 else _ic_room_w
                        _perp_origin = p["position_m"][_perp_axis]
                        _perp_dir = +1.0 if _ic_wa in ("left", "back") else -1.0
                        _perp_cleared = False
                        for _ic_attempt3 in range(_ic_max_slides):
                            if not _ic_any_collide():
                                _perp_cleared = True
                                break
                            _new_perp = p["position_m"][_perp_axis] + _perp_dir * _ic_step
                            # Stay inside room
                            if (_new_perp + _ic_perp_half > _ic_perp_lim - 0.05
                                    or _new_perp - _ic_perp_half < 0.05):
                                break
                            p["position_m"][_perp_axis] = _new_perp
                        if _perp_cleared and _ic_attempt3 > 0:
                            print(f"  [init_slide] wall-axis blocked; slid "
                                  f"{_perp_dir * _ic_attempt3 * _ic_step:+.2f}m "
                                  f"OFF wall (axis={_perp_axis}) to clear collision")
                            if _anim_frame_render is not None:
                                _anim_frame_render(placements + [p],
                                                   label=f"idx{idx}_{obj_type}_post_init_slide",
                                                   highlight_idx=idx)
                        else:
                            p["position_m"][_perp_axis] = _perp_origin  # full reset
                            print(f"  [init_slide] could not clear collision within "
                                  f"{_ic_max_slide:.1f}m on wall axis or perpendicular — "
                                  f"leaving at original position")

            # Snapshot the pre-facing (back-projected) position — mask_align's
            # scale-fit below still needs it: it compares a PROJECTION of the
            # object at a given position against the mask bbox from the source
            # photo, and that bbox was measured at the object's photographed
            # location, not wherever facing_toward later moves it to (e.g. a
            # chair tucked under a desk across the room).  Fitting scale using
            # the post-facing position would match the wrong projection depth.
            _pre_facing_pos = list(p["position_m"])

            # ── facing_toward: rotate to face another object (no reposition) ──
            # "in_front_of" / "facing_toward" describes a spatial relationship,
            # NOT a command to press objects tightly together.  Only rotate the
            # object toward the target — the mask is the position authority.
            # Skip for wall-affinity objects: wall alignment (long side flush)
            # takes priority — rotating to "face" a target would make the
            # object perpendicular to its wall.
            facing_idx = entry.get("facing_toward")
            _wall_aff = p.get("wall_affinity", "centre")
            # facing_toward is applied for chairs/seating facing a desk/table
            # (functional relationship: the chair MUST face the work surface).
            # For chairs facing sofas (conversation layout), skip facing_toward
            # and let the rotation from vlm_refine / wall_align handle it —
            # the chair's orientation should come from the room layout, not
            # pointing directly at the sofa.
            _facing_types = {"chair", "armchair", "stool", "office_chair"}
            # Work-surfaces a chair is TUCKED UNDER (reposition the chair to sit
            # directly in front).  Low occasional tables (coffee/side/round) are
            # NOT tuck-under: a chair beside a coffee table keeps where the photo
            # puts it and merely ROTATES to face — repositioning it "in front of"
            # the coffee table drags it across the room (e.g. a left chair onto
            # the right) and then collides with the table.
            _TUCK_TYPES  = {"desk", "dining_table"}
            _LOW_TABLES  = {"coffee_table", "side_table", "round_table", "table",
                            "end_table", "console_table"}
            _desk_types = _TUCK_TYPES | _LOW_TABLES
            _facing_target = placed_by_idx.get(facing_idx) if facing_idx is not None else None
            _target_type = (_facing_target.get("type", "").lower().replace("-", "_")
                            if _facing_target is not None else "")
            _target_is_desk = _facing_target is not None and _target_type in _desk_types
            _allow_facing = (
                facing_idx is not None
                and (_wall_aff == "centre" or obj_type.lower().replace("-", "_") in _facing_types)
                and _target_is_desk  # only face desks/tables, not sofas
            )
            if _allow_facing:
                target = _facing_target
                p["_facing_applied"] = True
                p["_facing_target_idx"] = facing_idx  # for collision exclusion
                # Chair at desk is free-standing, not wall-bound — set directly
                # so render/geo_save wall_snap never fires regardless of .pyc state
                p["wall_affinity"] = "centre"
                p["skip_ground_removal"] = False

                # Use the local_front already computed during placement
                # (detilt-derived for raw meshes, direct for canonical GLBs)
                _f_lf = p.get("_local_front")
                if _f_lf is not None:
                    _f_local_front = np.array(_f_lf, dtype=np.float64)
                else:
                    _f_local_front = np.array([0, 0, -1], dtype=np.float64)

                # 1) Position chair in front of the target — ONLY for tuck-under
                # work-surfaces (desk / dining table).  Low occasional tables keep
                # the chair's mask/back-projected position untouched (no drag).
                if _target_type in _TUCK_TYPES:
                    tgt_sm = target.get("size_m", {})
                    obj_sm = p.get("size_m", {})
                    tgt_depth = tgt_sm.get("depth_m", 0.5) / 2.0
                    obj_depth = obj_sm.get("depth_m", 0.5) / 2.0
                    offset_dist = tgt_depth + obj_depth + 0.10  # 10cm gap
                    # Use desk's wall_affinity to determine "in front of desk":
                    # The desk's open/user side is opposite its wall.
                    _tgt_wa = target.get("wall_affinity", "centre")
                    if _tgt_wa == "back":
                        offset_dir = np.array([0, 0, 1], dtype=np.float64)
                    elif _tgt_wa == "right":
                        offset_dir = np.array([-1, 0, 0], dtype=np.float64)
                    elif _tgt_wa == "left":
                        offset_dir = np.array([1, 0, 0], dtype=np.float64)
                    else:
                        # Centre desk: fall back to chair-to-desk direction
                        tgt_xz = np.array([target["position_m"][0], target["position_m"][2]])
                        obj_xz = np.array([p["position_m"][0], p["position_m"][2]])
                        dx = tgt_xz[0] - obj_xz[0]
                        dz = tgt_xz[1] - obj_xz[1]
                        dist = np.sqrt(dx*dx + dz*dz)
                        if dist > 0.01:
                            offset_dir = np.array([dx/dist, 0, dz/dist], dtype=np.float64)
                        else:
                            offset_dir = np.array([0, 0, 1], dtype=np.float64)
                    p["position_m"][0] = target["position_m"][0] + float(offset_dir[0]) * offset_dist
                    p["position_m"][2] = target["position_m"][2] + float(offset_dir[2]) * offset_dist
                    print(f"  [facing] tuck-under {_target_type} wall_aff={_tgt_wa} "
                          f"→ offset_dir={offset_dir.tolist()}")
                else:
                    print(f"  [facing] low table ({_target_type}) — keeping mask "
                          f"position, rotate-only")

                # 2) Compute facing rotation from the (possibly unchanged) position
                new_dx = target["position_m"][0] - p["position_m"][0]
                new_dz = target["position_m"][2] - p["position_m"][2]
                new_dist = np.sqrt(new_dx**2 + new_dz**2)
                if new_dist > 0.01:
                    facing_dir = np.array([new_dx/new_dist, 0, new_dz/new_dist], dtype=np.float64)
                else:
                    facing_dir = np.array([0, 0, -1], dtype=np.float64)
                angle_from = np.arctan2(float(_f_local_front[0]), float(_f_local_front[2]))
                angle_to   = np.arctan2(float(facing_dir[0]), float(facing_dir[2]))
                R_facing = _rotation_y(angle_to - angle_from)
                p["rotation_3x3"] = R_facing.tolist()
                print(f"  [facing] idx={facing_idx} ({target.get('type','?')}): "
                      f"{'repositioned + ' if _target_type in _TUCK_TYPES else 'rotate-only, '}"
                      f"faced  local_front={_f_local_front.tolist()}")
            elif facing_idx is not None and not _allow_facing:
                _skip_reason = "wall_affinity" if _wall_aff != "centre" else (
                    f"target is {_facing_target.get('type', '?')}, not a desk/table"
                    if _facing_target else "target not placed")
                print(f"  [facing] skipped rotation — {_skip_reason} "
                      f"(facing_toward={facing_idx})")

            # ── Mask-projection alignment (BEFORE VLM refine) ─────────────────
            # Soft sanity check: nudge position/scale if grossly mismatched
            # with the segmentation mask.  Object correspondence constraints
            # (wall affinity, facing, relative positions) are the PRIMARY
            # authority — mask alignment is secondary and gentle.
            mfd = p.get("_mask_fit_data")
            _exempt = set(p.get("_pre_slide_colliders", []))
            _facing_locked = bool(p.get("_facing_applied"))
            if mfd is not None and not p.get("_pos_override"):
                _bx1, _by1, _bx2, _by2 = mfd["bbox"]
                # _facing_applied objects (e.g. a chair tucked under a desk):
                # the tuck-under offset in the [facing] block above IS the
                # position — mask_align has no way to know the chair is
                # supposed to sit AT the desk rather than at its own
                # back-projected mask position, so it silently dragged the
                # chair back onto the mask and undid the whole offset
                # (office8: 1.34m drag-back → the chair no longer reached
                # the desk at all).  Fit scale using the PRE-facing position
                # (still comparable to the photo's mask bbox — the
                # tucked-under position is at a different depth from camera
                # and would fit the wrong projected size), then keep only
                # the scale/size result and discard the position result.
                _ma_pos_in = (np.array(_pre_facing_pos, dtype=np.float64)
                              if _facing_locked else
                              np.array(p["position_m"], dtype=np.float64))
                _pos_arr, _sc_arr, _ew, _eh, _ed, _ma_err_px = _align_and_scale_to_mask(
                    mfd["mesh_verts"], mfd["mesh_bounds"],
                    np.array(p["scale"], dtype=np.float64),
                    np.array(p["rotation_3x3"], dtype=np.float64),
                    _ma_pos_in,
                    _bx1, _by1, _bx2, _by2,
                    mfd["cam_pos"], mfd["right_v"], mfd["up_c_v"], mfd["fwd_v"],
                    mfd["fx"], mfd["cx"], mfd["cy"],
                    p.get("wall_affinity", "centre"), mfd["back_flush"],
                    mfd["room_w"], mfd["room_d"],
                    n_passes=4,
                    existing_placements=placements,
                    obj_size_m=p.get("size_m"),
                    vlm_size_locked=bool(p.get("_vlm_size_converged")
                                         or (entry or {}).get("_vlm_size_converged")),
                    mask_fill=_mask_fill_ratio(out_root, entry or {}),
                    exempt_collision_idxs=_exempt,
                    is_low_surface=p.get("type", "").lower().replace("-", "_") in _LOW_SURFACE_TYPES,
                    local_front=np.array(p.get("_local_front", [0, 0, -1]), dtype=np.float64),
                )
                if not _facing_locked:
                    p["position_m"] = _pos_arr.tolist()
                p["scale"]  = _sc_arr.tolist()
                p["size_m"] = {"width_m": _ew, "height_m": _eh, "depth_m": _ed}
                p["eff_h"]  = _eh

            # ── Animation frame: object at initial position (pre-VLM refine) ───
            if _anim_frame_render is not None:
                _anim_frame_render(placements + [p],
                                   label=f"idx{idx}_{obj_type}_initial",
                                   highlight_idx=idx)

            # ── Iterative VLM placement refinement (orientation only) ─────────
            # VLM refine handles orientation corrections that mask projection
            # cannot detect.  Scale/position are already anchored by the mask.
            if not p.get("wall_mounted"):
                p = vlm_iterative_placement_refine(
                    p, entry, cam, out_root, placements, max_iters=vlm_refine_iters,
                )
                if _anim_frame_render is not None:
                    _anim_frame_render(placements + [p],
                                       label=f"idx{idx}_{obj_type}_post_vlm_scale_rotation",
                                       highlight_idx=idx)

            # NOTE: wall_face correction removed — it relied on _local_front
            # which can be 180° wrong (detilt VLM confuses front/back).  VLM
            # orient already tests all 4 candidates for seating and visually
            # matches against the reference photo, which is more reliable.

            # ── Final mask-projection recheck ─────────────────────────────────
            # VLM refine may have changed rotation.  One gentle pass to
            # verify scale/position aren't grossly off after orientation change.
            # Same _facing_applied pre-facing-position / position-discard
            # treatment as above — see comment there.
            if mfd is not None and not p.get("_pos_override"):
                _bx1, _by1, _bx2, _by2 = mfd["bbox"]
                _ma_pos_in = (np.array(_pre_facing_pos, dtype=np.float64)
                              if _facing_locked else
                              np.array(p["position_m"], dtype=np.float64))
                _pos_arr, _sc_arr, _ew, _eh, _ed, _ma_err_px = _align_and_scale_to_mask(
                    mfd["mesh_verts"], mfd["mesh_bounds"],
                    np.array(p["scale"], dtype=np.float64),
                    np.array(p["rotation_3x3"], dtype=np.float64),
                    _ma_pos_in,
                    _bx1, _by1, _bx2, _by2,
                    mfd["cam_pos"], mfd["right_v"], mfd["up_c_v"], mfd["fwd_v"],
                    mfd["fx"], mfd["cx"], mfd["cy"],
                    p.get("wall_affinity", "centre"), mfd["back_flush"],
                    mfd["room_w"], mfd["room_d"],
                    n_passes=2,
                    existing_placements=placements,
                    obj_size_m=p.get("size_m"),
                    vlm_size_locked=bool(p.get("_vlm_size_converged")
                                         or (entry or {}).get("_vlm_size_converged")),
                    mask_fill=_mask_fill_ratio(out_root, entry or {}),
                    exempt_collision_idxs=_exempt,
                    is_low_surface=p.get("type", "").lower().replace("-", "_") in _LOW_SURFACE_TYPES,
                    local_front=np.array(p.get("_local_front", [0, 0, -1]), dtype=np.float64),
                )
                if not _facing_locked:
                    p["position_m"] = _pos_arr.tolist()
                p["scale"]  = _sc_arr.tolist()
                p["size_m"] = {"width_m": _ew, "height_m": _eh, "depth_m": _ed}
                p["eff_h"]  = _eh
                # Store final alignment error so scene_review can decide
                # whether to override this position.
                p["_mask_align_final_err_px"] = list(_ma_err_px)

            # ── Capture silhouette anchor BEFORE further refinement ──────────
            # `_mask_origin_xz` is the trust anchor used by the deviation cap
            # in VGGT, post-mask clearance, and collision resolution. Set it
            # NOW (right after mask_align stabilizes) so subsequent steps are
            # measured against the silhouette-matched position, not against
            # whatever they themselves end up producing. The end-of-iteration
            # assignment at the bottom of this block is left in place as a
            # fallback for objects that bypass mask_align (no mfd).
            if mfd is not None:
                p["_mask_origin_xz"] = [p["position_m"][0], p["position_m"][2]]

            # ── Corner-snap heuristic ────────────────────────────────────────
            # Wall-affine objects (back/left/right) only get flushed against
            # ONE wall by initial placement. If the object also sits very close
            # to a perpendicular wall, it almost always belongs IN that corner
            # — silhouette/back-projection X (or Z) tends to lag because the
            # photo bbox of skinny objects (plants, lamps) doesn't pin position
            # precisely. Snap into the corner when the perpendicular-wall gap
            # is under _CORNER_SNAP_DIST, and only if the snap doesn't create
            # a collision with another placed object.
            _CORNER_SNAP_DIST = 0.60   # metres — threshold for "near corner"
            _CORNER_GAP       = 0.05
            _wa_cs = p.get("wall_affinity", "centre")
            if (_wa_cs in ("back", "left", "right")
                and not p.get("wall_mounted")
                and not p.get("is_carpet")):
                _R_cs = np.array(p.get("rotation_3x3", np.eye(3).tolist()),
                                 dtype=np.float64)
                _sm_cs = p.get("size_m", {})
                _lhx_cs = float(_sm_cs.get("width_m", 0.5)) / 2.0
                _lhz_cs = float(_sm_cs.get("depth_m", 0.5)) / 2.0
                _hx_cs = (abs(float(_R_cs[0][0])) * _lhx_cs
                          + abs(float(_R_cs[0][2])) * _lhz_cs)
                _hz_cs = (abs(float(_R_cs[2][0])) * _lhx_cs
                          + abs(float(_R_cs[2][2])) * _lhz_cs)
                try:
                    _rw_cs, _rd_cs = _get_room_dims(
                        Path(p.get("glb_path", ".")), cam)
                except Exception:
                    _rw_cs = mfd["room_w"] if mfd is not None else 5.0
                    _rd_cs = mfd["room_d"] if mfd is not None else 5.0
                _px_cs, _pz_cs = float(p["position_m"][0]), float(p["position_m"][2])
                # Pick the perpendicular axis based on wall_affinity:
                #   "back"  → check x walls (left=0, right=room_w)
                #   "left"  → check z walls (back=0, front=room_d)
                #   "right" → check z walls (back=0, front=room_d)
                _snap = None
                if _wa_cs == "back":
                    _d_l = _px_cs - _hx_cs                     # gap to x=0
                    _d_r = (_rw_cs - _px_cs) - _hx_cs          # gap to x=room_w
                    if _d_l <= _d_r and _d_l < _CORNER_SNAP_DIST and _d_l > -0.05:
                        _snap = (_hx_cs + _CORNER_GAP, _pz_cs, "left")
                    elif _d_r < _CORNER_SNAP_DIST and _d_r > -0.05:
                        _snap = (_rw_cs - _hx_cs - _CORNER_GAP, _pz_cs, "right")
                else:  # left or right
                    _d_b = _pz_cs - _hz_cs                     # gap to z=0 (back)
                    _d_f = (_rd_cs - _pz_cs) - _hz_cs          # gap to z=room_d
                    if _d_b <= _d_f and _d_b < _CORNER_SNAP_DIST and _d_b > -0.05:
                        _snap = (_px_cs, _hz_cs + _CORNER_GAP, "back")
                    elif _d_f < _CORNER_SNAP_DIST and _d_f > -0.05:
                        _snap = (_px_cs, _rd_cs - _hz_cs - _CORNER_GAP, "front")
                if _snap is not None:
                    _new_x, _new_z, _side = _snap
                    # Collision check (AABB; same hull set as mask_align)
                    _old_x, _old_z = _px_cs, _pz_cs
                    p["position_m"][0] = _new_x
                    p["position_m"][2] = _new_z
                    _cs_collides = False
                    for _op_cs in placements:
                        if (_op_cs is p or _op_cs.get("is_carpet")
                                or _op_cs.get("wall_mounted")):
                            continue
                        _opR = np.array(_op_cs.get("rotation_3x3",
                                                    np.eye(3).tolist()))
                        _opsm = _op_cs.get("size_m", {})
                        _ophx_l = float(_opsm.get("width_m", 0.5)) / 2.0
                        _ophz_l = float(_opsm.get("depth_m", 0.5)) / 2.0
                        _ophx = (abs(float(_opR[0][0])) * _ophx_l
                                 + abs(float(_opR[0][2])) * _ophz_l)
                        _ophz = (abs(float(_opR[2][0])) * _ophx_l
                                 + abs(float(_opR[2][2])) * _ophz_l)
                        _ov_x_cs = (_hx_cs + _ophx) - abs(_new_x - _op_cs["position_m"][0])
                        _ov_z_cs = (_hz_cs + _ophz) - abs(_new_z - _op_cs["position_m"][2])
                        if _ov_x_cs > 0.02 and _ov_z_cs > 0.02:
                            _cs_collides = True
                            break
                    if _cs_collides:
                        p["position_m"][0] = _old_x
                        p["position_m"][2] = _old_z
                        print(f"  [corner_snap] {p.get('type','?')} "
                              f"idx={p.get('index')}: would collide → skip "
                              f"({_wa_cs}+{_side})")
                    else:
                        # Refresh silhouette anchor so subsequent VGGT/scene_review
                        # measure deviation from the corner-snapped position, not
                        # from the pre-snap silhouette.
                        p["_mask_origin_xz"] = [_new_x, _new_z]
                        print(f"  [corner_snap] {p.get('type','?')} "
                              f"idx={p.get('index')}: {_wa_cs}+{_side} corner "
                              f"({_old_x:.3f},{_old_z:.3f}) → "
                              f"({_new_x:.3f},{_new_z:.3f})")
                        if _anim_frame_render is not None:
                            _anim_frame_render(placements + [p],
                                               label=f"idx{idx}_{obj_type}_post_corner_snap",
                                               highlight_idx=idx)

            # ── Fine yaw refinement (per-placement, iterative) ───────────────
            # Real photos rarely have seating perfectly flush against a wall;
            # they tilt a few degrees toward the room or each other. The
            # cardinal-orient pass above only checked 0/90/180/270, so this
            # iterates: render → ask VLM for delta → rotate → re-render → ask
            # again. Stops when VLM returns 0 (or |delta|<1°), or after
            # vlm_fine_rotation_iters passes.
            # Skip fine_rot entirely for wall-affinity sofas/loveseats:
            # the deterministic "back-against-wall" rotation set during
            # placement is correct for these, and fine_rot's per-iter +5°
            # nudges only add visible drift (back-wall sofa picked up +15°
            # over 3 iters in the previous run).  Chairs / armchairs /
            # ottomans can still benefit from fine_rot because their facing
            # is room-relative, not wall-relative.
            _is_wall_aff_seating = (p.get("type") in {"sofa", "loveseat"}
                                    and p.get("wall_affinity")
                                    in ("back", "left", "right"))
            if (vlm_fine_rotation
                and not p.get("wall_mounted")
                and not p.get("is_carpet")
                and p.get("_facing_target_idx") is None
                and p.get("type") in {"sofa", "loveseat", "chair", "armchair", "ottoman"}
                and not _is_wall_aff_seating
                and entry.get("box_px")):
                try:
                    # Resolve reference photo path once (used for every iteration).
                    _ref_path = None
                    try:
                        _vc = json.loads((out_root / "vggt" / "camera.json").read_text())
                        _img_field = (_vc[0] if isinstance(_vc, list) else _vc).get("image", "")
                        if _img_field:
                            _candidate = Path(_img_field)
                            if not _candidate.is_absolute():
                                _candidate = Path(__file__).resolve().parents[2] / _candidate
                            if _candidate.exists():
                                _ref_path = _candidate
                    except Exception:
                        _ref_path = None
                    if _ref_path is None:
                        raise FileNotFoundError("reference photo path unresolved")

                    _fr_bbox  = tuple(entry["box_px"])
                    _img_w    = int(cam["width_px"])
                    _img_h    = int(cam["height_px"])
                    _max_deg  = int(vlm_fine_rotation_max_deg)
                    _fr_wa    = p.get("wall_affinity", "centre")
                    # Wall-affinity items: VLM is now told the render baseline
                    # is 0° (back-against-wall) and what the expected sign
                    # for a "tilt toward room centre" is.  In that prompt mode
                    # the VLM often returns a conservative first estimate and
                    # asks to keep rotating in the same direction across
                    # iterations — that's a genuine "still need more"
                    # signal, not hallucination.  Allow up to vlm_fine_rotation_iters
                    # iters, with a slightly higher threshold than centre to
                    # avoid sub-5° noise.
                    if _fr_wa in ("back", "left", "right"):
                        _max_iter      = int(vlm_fine_rotation_iters)
                        _delta_thresh  = 5.0   # ignore <5° (anti-hallucination noise)
                    else:
                        _max_iter      = int(vlm_fine_rotation_iters)
                        _delta_thresh  = 1.0
                    _total_yaw  = 0.0
                    _last_notes = ""
                    for _fr_iter in range(_max_iter):
                        _fr_preview = _render_object_preview(p, cam, out_root, placements)
                        if _fr_preview is None:
                            print(f"  [fine_rot] iter {_fr_iter}: render failed; stopping")
                            break
                        _fr_size = p.get("size_m") or {}
                        _fr_pos_xz = (float(p["position_m"][0]),
                                       float(p["position_m"][2]))
                        try:
                            _fr_room_w, _fr_room_d = _get_room_dims(
                                Path(p.get("glb_path", ".")), cam)
                        except Exception:
                            _fr_room_w, _fr_room_d = None, None
                        _fr_res = _vlm_fine_rotation_correction(
                            preview_path=_fr_preview, reference_path=_ref_path,
                            obj_type=obj_type, bbox=_fr_bbox,
                            img_w=_img_w, img_h=_img_h, max_deg=_max_deg,
                            wall_affinity=_fr_wa,
                            obj_width_m=float(_fr_size.get("width_m", 0)) or None,
                            obj_depth_m=float(_fr_size.get("depth_m", 0)) or None,
                            obj_world_pos=_fr_pos_xz,
                            room_w=_fr_room_w, room_d=_fr_room_d,
                        )
                        _delta = float(_fr_res.get("delta_deg", 0))
                        _last_notes = _fr_res.get("notes", "")
                        if abs(_delta) < _delta_thresh:
                            print(f"  [fine_rot] iter {_fr_iter}: Δyaw={_delta:+.1f}° "
                                  f"< thresh {_delta_thresh:.0f}° "
                                  f"(wa={_fr_wa}, no change)  "
                                  f"notes={_last_notes[:80]}")
                            break
                        # Anti-drift: only apply the same-sign-stop heuristic
                        # for centre-affinity objects (where the VLM compares
                        # two images and may hallucinate a small tilt).  For
                        # wall-aff items the prompt has explicit direction
                        # context and same-sign repeats indicate "VLM wants
                        # to keep rotating in the same direction" — let it
                        # accumulate up to the cap.  Cap the cumulative magnitude
                        # at max_deg to prevent unbounded growth.
                        if (_fr_wa not in ("back", "left", "right")
                                and _fr_iter > 0 and (_delta * _total_yaw) > 0):
                            print(f"  [fine_rot] iter {_fr_iter}: Δyaw={_delta:+.1f}° "
                                  f"same sign as cumulative {_total_yaw:+.1f}° — "
                                  f"VLM likely hallucinating, stopping")
                            break
                        # OPPOSITE sign after a correction: the two estimates
                        # bracket the truth, so applying the reversal in full
                        # lands back where we started.  pexels_2343465's armchair
                        # did exactly that — iter0 -10.0°, iter1 +10.0°,
                        # cumulative +0.0° — spending two VLM calls to change
                        # nothing and leaving the chair at its original angle.
                        # Damp to the midpoint of the bracket and stop there.
                        _stop_after_apply = False
                        if _fr_iter > 0 and (_delta * _total_yaw) < 0:
                            _damped = _delta * 0.5
                            print(f"  [fine_rot] iter {_fr_iter}: Δyaw={_delta:+.1f}° "
                                  f"reverses cumulative {_total_yaw:+.1f}° — estimates "
                                  f"bracket the answer; damping to {_damped:+.1f}° "
                                  f"(settling at {_total_yaw + _damped:+.1f}°) and stopping")
                            _delta = _damped
                            _stop_after_apply = True
                            if abs(_delta) < 0.5:       # already at the midpoint
                                break
                        if abs(_total_yaw + _delta) > _max_deg:
                            _delta = float(np.sign(_delta)) * (_max_deg - abs(_total_yaw))
                            if abs(_delta) < _delta_thresh:
                                print(f"  [fine_rot] iter {_fr_iter}: cumulative "
                                      f"{_total_yaw:+.1f}° already at cap "
                                      f"{_max_deg}°, stopping")
                                break
                            print(f"  [fine_rot] iter {_fr_iter}: clamping Δyaw to "
                                  f"{_delta:+.1f}° to stay within cap {_max_deg}°")
                        _apply_y_rotation_deg(p, _delta)
                        _total_yaw += _delta
                        print(f"  [fine_rot] iter {_fr_iter}: Δyaw={_delta:+.1f}°  "
                              f"(cumulative {_total_yaw:+.1f}°)  notes={_last_notes[:80]}")
                        if _anim_frame_render is not None:
                            _anim_frame_render(placements + [p],
                                               label=f"idx{idx}_{obj_type}_fine_rot_iter{_fr_iter}",
                                               highlight_idx=idx)
                        if _stop_after_apply:
                            break
                    if abs(_total_yaw) >= 0.5:
                        p["_fine_rotation_total_deg"] = _total_yaw
                        p["_fine_rotation_iters"]     = _fr_iter + 1
                        p["_fine_rotation_notes"]     = _last_notes
                except Exception as _fre:
                    print(f"  [fine_rot] failed for idx={p.get('index')}: {_fre}")

            # ── VGGT-based ground-truth refinement ────────────────────────────
            if (vggt_refine
                and not p.get("wall_mounted")
                and not p.get("is_carpet")
                and not p.get("_facing_applied")
                and not p.get("_pos_override")):
                # Animation: snapshot the post-mask_align position BEFORE VGGT
                # so the GIF shows VGGT's effect explicitly.
                if _anim_frame_render is not None:
                    _anim_frame_render(placements + [p],
                                       label=f"idx{idx}_{obj_type}_pre_vggt",
                                       highlight_idx=idx)
                try:
                    if vggt_method == "render_diff" and _vggt_ref_depth is not None:
                        # Render callback: full furniture render at scale=1 (matches camera).
                        def _vggt_render_cb(_pls, _outp):
                            render_furniture(out_root, _pls, cam,
                                             render_scale=1, out_path=Path(_outp),
                                             base_image_path=base_image_path)
                        p = _vggt_render_diff_refine(
                            p, entry, out_root,
                            ref_depth=_vggt_ref_depth, K=_vggt_ref_K, E=_vggt_ref_E,
                            render_callback=_vggt_render_cb,
                            all_placements=placements,
                            blend=vggt_blend,
                            max_shift_m=vggt_max_shift_m,
                            lateral_weight=vggt_lateral_weight,
                            depth_weight=vggt_depth_weight,
                            pseudo_to_metric_scale=_vggt_pseudo_to_metric,
                            pseudo_to_metric_polyfit=_vggt_pseudo_to_metric_poly,
                        )
                    elif _vggt_calib is not None:
                        p = _vggt_refine_placement(
                            p, entry, out_root, _vggt_calib,
                            blend=vggt_blend,
                            max_shift_m=vggt_max_shift_m,
                            respect_y=True,
                            visible_face_offset=vggt_visible_face_offset,
                            lateral_weight=vggt_lateral_weight,
                            depth_weight=vggt_depth_weight,
                            all_placements=placements,
                        )
                    # Animation: snapshot AFTER VGGT (only if it actually applied).
                    if (_anim_frame_render is not None
                        and (p.get("_vggt_refine") or {}).get("status") == "applied"):
                        _anim_frame_render(placements + [p],
                                           label=f"idx{idx}_{obj_type}_post_vggt",
                                           highlight_idx=idx)
                    _vrf = p.get("_vggt_refine") or {}
                    if _vrf.get("status") == "applied":
                        print(f"  [vggt_refine] idx={p.get('index')} method={_vrf.get('method', 'global_calib')} "
                              f"shift_xz={_vrf.get('shift_xz_m', 0.0):.3f}m  "
                              f"clipped={_vrf.get('clipped')}  ")
                except Exception as _vre:
                    print(f"  [vggt_refine] failed for idx={p.get('index')}: {_vre}")

            # (Both the silhouette LATERAL pin AND the fine-yaw refine moved to a
            #  FINAL pass before render — they must run AFTER every wall/corner-snap
            #  AND in order (lateral first), so the yaw IoU is measured at the
            #  CORRECTED lateral position instead of the pre-snap one. See below.)

            # ── Post-mask_align clearance: resolve exempt collisions ──────────
            # Both mask_align calls ignore exempt objects so the object reaches
            # its silhouette target freely.  Now do a minimal slide to clear
            # any remaining overlap with those exempt objects.
            if _exempt and placements:
                _ic_R   = np.array(p.get("rotation_3x3", np.eye(3).tolist()))
                _ic_sm2 = p.get("size_m", {})
                _ic_hx2 = (abs(float(_ic_R[0][0])) * _ic_sm2.get("width_m", 0.5) / 2.0
                          + abs(float(_ic_R[0][2])) * _ic_sm2.get("depth_m", 0.5) / 2.0)
                _ic_hz2 = (abs(float(_ic_R[2][0])) * _ic_sm2.get("width_m", 0.5) / 2.0
                          + abs(float(_ic_R[2][2])) * _ic_sm2.get("depth_m", 0.5) / 2.0)
                for _ep in placements:
                    if _ep is p or _ep.get("is_carpet") or _ep.get("wall_mounted"):
                        continue
                    if _ep.get("index") not in _exempt:
                        continue
                    # Use SAT hull collision for accurate detection
                    if not _poly_collides(p, _ep):
                        continue
                    _ep_sm2 = _ep.get("size_m", {})
                    _ep_R2  = np.array(_ep.get("rotation_3x3", np.eye(3).tolist()))
                    _ep_hx2 = (abs(float(_ep_R2[0][0])) * _ep_sm2.get("width_m", 0.5) / 2.0
                              + abs(float(_ep_R2[0][2])) * _ep_sm2.get("depth_m", 0.5) / 2.0)
                    _ep_hz2 = (abs(float(_ep_R2[2][0])) * _ep_sm2.get("width_m", 0.5) / 2.0
                              + abs(float(_ep_R2[2][2])) * _ep_sm2.get("depth_m", 0.5) / 2.0)
                    _ov_x2 = (_ic_hx2 + _ep_hx2) - abs(p["position_m"][0] - _ep["position_m"][0])
                    _ov_z2 = (_ic_hz2 + _ep_hz2) - abs(p["position_m"][2] - _ep["position_m"][2])
                    # Slide along the axis of MINIMUM overlap (shortest separation)
                    # and always AWAY from the exempt object's centroid.
                    _dx2 = p["position_m"][0] - _ep["position_m"][0]
                    _dz2 = p["position_m"][2] - _ep["position_m"][2]
                    _wa2 = p.get("wall_affinity", "centre")
                    if _wa2 == "back" or (_wa2 != "left" and _wa2 != "right"
                                           and abs(_ov_x2) <= abs(_ov_z2)):
                        _slide2 = (_ov_x2 + 0.05) * (1.0 if _dx2 >= 0 else -1.0)
                        _axis2, _new_coord2 = 0, float(p["position_m"][0] + _slide2)
                    else:
                        _slide2 = (_ov_z2 + 0.05) * (1.0 if _dz2 >= 0 else -1.0)
                        _axis2, _new_coord2 = 2, float(p["position_m"][2] + _slide2)
                        # Mask-deviation cap: don't drag the object more than
                        # _MAX_PMC_DEV m from its silhouette anchor — a slide that
                        # large suggests we should leave the overlap as-is.
                        _MAX_PMC_DEV = 0.15
                        _pmc_origin = p.get("_mask_origin_xz")
                        _check_x = p["position_m"][0]
                        _check_z = _new_coord2
                        if _pmc_origin is not None:
                            _pmc_drift = float(np.sqrt((_check_x - _pmc_origin[0])**2
                                                        + (_check_z - _pmc_origin[1])**2))
                        else:
                            _pmc_drift = 0.0
                        if _pmc_drift > _MAX_PMC_DEV:
                            print(f"  [post_mask_clearance] slide {_slide2:.3f}m abandoned — "
                                  f"would drift {_pmc_drift:.3f}m from silhouette anchor "
                                  f"(cap {_MAX_PMC_DEV:.2f}m)")
                            continue
                        # Don't push past the room walls. Compute the rotated
                        # OBB half-extent on the axis we're sliding along and
                        # ensure the new center keeps the OBB inside the room.
                        try:
                            _pmc_room_w, _pmc_room_d = _get_room_dims(
                                Path(p.get("glb_path", ".")), cam)
                        except Exception:
                            _pmc_room_w, _pmc_room_d = 5.0, 5.0
                        _pmc_R   = np.array(p.get("rotation_3x3", np.eye(3).tolist()),
                                            dtype=np.float64)
                        _pmc_lhz = float(p.get("size_m", {}).get("depth_m", 0.5)) / 2.0
                        _pmc_lhx = float(p.get("size_m", {}).get("width_m", 0.5)) / 2.0
                        _pmc_hz  = (abs(float(_pmc_R[2][0])) * _pmc_lhx
                                  + abs(float(_pmc_R[2][2])) * _pmc_lhz)
                        if _new_coord2 + _pmc_hz > _pmc_room_d - 0.01:
                            _capped_z = _pmc_room_d - _pmc_hz - 0.01
                            print(f"  [post_mask_clearance] slide {_slide2:.3f}m clipped — "
                                  f"would push past back wall (target Z={_new_coord2:.3f} > "
                                  f"limit {_pmc_room_d - _pmc_hz - 0.01:.3f})")
                            _new_coord2 = max(_capped_z, p["position_m"][2])
                            if _new_coord2 <= p["position_m"][2] + 0.01:
                                continue   # nothing useful left
                        if _new_coord2 - _pmc_hz < 0.01:
                            _capped_z = _pmc_hz + 0.01
                            print(f"  [post_mask_clearance] slide {_slide2:.3f}m clipped — "
                                  f"would push past front (target Z={_new_coord2:.3f} < "
                                  f"limit {_pmc_hz + 0.01:.3f})")
                            _new_coord2 = min(_capped_z, p["position_m"][2])
                            if _new_coord2 >= p["position_m"][2] - 0.01:
                                continue
                        # Check that the slide doesn't land in a non-exempt object
                        _old_coord2 = p["position_m"][_axis2]
                        p["position_m"][_axis2] = _new_coord2
                        _slide_collides = any(
                            _poly_collides(p, _op)
                            for _op in placements
                            if _op is not p and not _op.get("is_carpet")
                            and not _op.get("wall_mounted")
                            and _op.get("index") not in _exempt
                        )
                        if _slide_collides:
                            p["position_m"][_axis2] = _old_coord2
                            print(f"  [post_mask_clearance] slide {_slide2:.3f}m blocked — "
                                  f"would collide with non-exempt object")
                        else:
                            print(f"  [post_mask_clearance] cleared overlap with exempt "
                                  f"idx={_ep.get('index')} by sliding {_slide2:.3f}m")

            # ── Per-iteration silhouette anchor enforcement ────────────────────
            # Belt-and-braces: regardless of which step (VGGT, post_mask_clearance,
            # rel_pos, …) ended up moving this object, pull it back to within
            # _ITER_MAX_DEV m of its silhouette anchor before committing the
            # placement. The animation snapshot at the end of this iteration
            # captures the *clamped* pose, not whatever drifted state.
            #
            # The tighter this clamp, the closer the final placement stays to
            # the silhouette/mask_align result; the looser it is, the more
            # authority VGGT has to relocate the object.  Drives all VGGT-
            # related drift in the GIF: bumping this lets `--vggt-blend` /
            # `--vggt-max-shift` actually take effect.
            _ITER_MAX_DEV = float(iter_max_dev_m)
            _iter_origin = p.get("_mask_origin_xz")
            if _iter_origin is not None:
                _io_x, _io_z = float(_iter_origin[0]), float(_iter_origin[1])
                _ic_x = float(p["position_m"][0])
                _ic_z = float(p["position_m"][2])
                _ic_dev = float(np.sqrt((_ic_x - _io_x) ** 2 + (_ic_z - _io_z) ** 2))
                if _ic_dev > _ITER_MAX_DEV:
                    _r = _ITER_MAX_DEV / _ic_dev
                    p["position_m"][0] = _io_x + (_ic_x - _io_x) * _r
                    p["position_m"][2] = _io_z + (_ic_z - _io_z) * _r
                    print(f"  [iter_anchor] idx={p.get('index')} ({obj_type}) "
                          f"drift {_ic_dev:.3f}m → clamped to {_ITER_MAX_DEV:.2f}m: "
                          f"({_ic_x:.3f},{_ic_z:.3f}) → "
                          f"({p['position_m'][0]:.3f},{p['position_m'][2]:.3f})")

            # ── VLM relative position check for wall-affinity objects ────────
            # Check the object's along-wall position relative to nearby
            # wall-mounted objects (art, windows) in the reference image.
            # If visually wrong (e.g. sofa is left of painting but should be
            # centered under it), slide along the wall to correct.
            _waf = p.get("wall_affinity", "centre")
            if False and (_waf in ("back", "left", "right") and not p.get("wall_mounted")
                    and wm_placements_p.exists()):  # disabled: overrides mask_align silhouette
                try:
                    import re as _re_wm, requests as _req_wm
                    _wm_all = json.loads(wm_placements_p.read_text())
                    # Filter wall-mounted objects on the same wall, within 2m laterally
                    _slide_ax = 0 if _waf == "back" else 2
                    _wm_nearby = [
                        _w for _w in _wm_all
                        if _w.get("wall") == _waf
                        and abs((_w.get("world_pt") or [0, 0, 0])[_slide_ax]
                                - p["position_m"][_slide_ax]) < 2.0
                    ]
                    if _wm_nearby:
                        # Render preview of current placement
                        _rp_prev = _render_object_preview(p, cam, out_root, placements)
                        _ref_cands = [
                            out_root / "wall_mounted" / "placements" / "render_objects_placed.png",
                            out_root / "render_final.png",
                            out_root / "render.png",
                        ]
                        _ref_img = next((rp for rp in _ref_cands if rp.exists()), None)
                        if _rp_prev is not None and _ref_img is not None:
                            _wm_desc = "; ".join(
                                f"{_w.get('type','obj')} at world_x={_w['world_pt'][0]:.2f},z={_w['world_pt'][2]:.2f}"
                                for _w in _wm_nearby
                            )
                            _slide_name = "left-right (X axis)" if _slide_ax == 0 else "front-back (Z axis)"
                            _rp_prompt = f"""Compare two images showing the same room:
Image 1: reference photo. Image 2: 3D render.

The {obj_type} is placed against the {_waf} wall.
Nearby wall-mounted objects on the same wall: {_wm_desc}.

Look at the reference (Image 1): where is the {obj_type} relative to the wall-mounted object(s)?
- Is the {obj_type} to the LEFT of the wall-mounted object?
- Is it CENTERED below/near it?
- Is it to the RIGHT?

Now look at the render (Image 2): does the {obj_type}'s position match?

If the relative position is CLEARLY WRONG (not just slightly off), report a correction.
Respond with ONLY valid JSON:
{{"position_correct": true/false, "reference_relation": "left|center|right", "render_relation": "left|center|right"}}
If position looks approximately correct, set position_correct=true."""
                            _content_wm = [
                                {"type": "text", "text": _rp_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(_ref_img)}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(_rp_prev)}"}},
                            ]
                            _payload_wm = {
                                "model": "qwen3",
                                "messages": [{"role": "user", "content": _content_wm}],
                                "max_tokens": 256,
                                "chat_template_kwargs": {"enable_thinking": True},
                            }
                            try:
                                _resp_wm = _req_wm.post(VLM_API_URL, json=_payload_wm, timeout=60)
                                _resp_wm.raise_for_status()
                                _raw_wm = _resp_wm.json()["choices"][0]["message"]["content"]
                                _raw_wm = _re_wm.sub(r"<think>[\s\S]*?</think>", "", _raw_wm, flags=_re_wm.IGNORECASE).strip()
                                _m_wm = _re_wm.search(r"\{[\s\S]*?\}", _raw_wm)
                                if _m_wm:
                                    _res_wm = json.loads(_m_wm.group())
                                    if not _res_wm.get("position_correct", True):
                                        _ref_rel = _res_wm.get("reference_relation", "")
                                        _rnd_rel = _res_wm.get("render_relation", "")
                                        # Compute image-space direction needed
                                        _img_dir = 0
                                        if _ref_rel == "left" and _rnd_rel in ("center", "right"):
                                            _img_dir = -1  # need to go image-left
                                        elif _ref_rel == "right" and _rnd_rel in ("center", "left"):
                                            _img_dir = +1  # need to go image-right
                                        elif _ref_rel == "center":
                                            if _rnd_rel == "left":
                                                _img_dir = +1
                                            elif _rnd_rel == "right":
                                                _img_dir = -1
                                        if _img_dir != 0:
                                            # Only apply if silhouette (mask_align) didn't
                                            # already converge well — skip if mask_origin is
                                            # set (mask_align already handled position).
                                            if p.get("_mask_origin_xz") is not None:
                                                print(f"  [rel_pos] skip — mask_align already "
                                                      f"positioned {obj_type} (ref={_ref_rel} "
                                                      f"render={_rnd_rel})")
                                            else:
                                                if _waf == "back":
                                                    _world_dir = _img_dir
                                                elif _waf == "left":
                                                    _world_dir = -_img_dir
                                                else:
                                                    _world_dir = _img_dir
                                                _slide_wm = _world_dir * 0.30
                                                p["position_m"][_slide_ax] += _slide_wm
                                                print(f"  [rel_pos] {obj_type} slid {_slide_wm:+.2f}m along wall "
                                                      f"(ref={_ref_rel} render={_rnd_rel})")
                            except Exception as _e_wm:
                                print(f"  [rel_pos] VLM call failed: {_e_wm}")
                        if _rp_prev is not None and _rp_prev.exists():
                            _rp_prev.unlink(missing_ok=True)
                except Exception as _e_relpos:
                    pass  # Non-critical — skip silently if anything fails

            # ── Room-bounds position clamp ────────────────────────────────────
            # Skip for _facing_applied objects: their position is determined by
            # their target relationship (e.g. chair at desk).  Clamping can
            # push the chair too close to the table.
            if ("size_m" in p and not p.get("is_carpet")
                    and not p.get("wall_mounted") and not p.get("_facing_applied")):
                _fin_room_w, _fin_room_d = _get_room_dims(Path(p.get("glb_path", ".")), cam)
                # Compute world-space half-extents using rotation matrix so
                # the clamp works correctly regardless of object orientation.
                # width_m/depth_m are semantic (along-front vs perpendicular);
                # undo the [axis_swap] relabel to get raw mesh-X/mesh-Z halves
                # before combining with R, which rotates the RAW local axes.
                _p_front = p.get("_local_front", [0, 0, -1])
                if abs(float(_p_front[0])) > abs(float(_p_front[2])):
                    _local_hx = p["size_m"]["depth_m"] / 2.0   # mesh-X half
                    _local_hz = p["size_m"]["width_m"] / 2.0   # mesh-Z half
                else:
                    _local_hx = p["size_m"]["width_m"] / 2.0   # mesh-X half
                    _local_hz = p["size_m"]["depth_m"] / 2.0   # mesh-Z half
                _R = np.array(p["rotation_3x3"], dtype=np.float64)
                _hx = abs(float(_R[0][0])) * _local_hx + abs(float(_R[0][2])) * _local_hz
                _hz = abs(float(_R[2][0])) * _local_hx + abs(float(_R[2][2])) * _local_hz
                # Step-by-step wall-clearance: instead of snapping the
                # position to the room boundary (which can drag the object
                # into another already-placed item's footprint), nudge it
                # away from the wall in 3 cm increments and stop the moment
                # the move would create a collision. Trades a small amount
                # of wall-clipping for "no new collision".
                _start_x = float(p["position_m"][0])
                _start_z = float(p["position_m"][2])
                _target_x = float(np.clip(_start_x, _hx, max(_fin_room_w - _hx, _hx)))
                _target_z = float(np.clip(_start_z, _hz, max(_fin_room_d - _hz, _hz)))
                _shift_x = _target_x - _start_x
                _shift_z = _target_z - _start_z
                _shift_mag = float(np.sqrt(_shift_x ** 2 + _shift_z ** 2))
                if _shift_mag > 1e-3:
                    _step_size = 0.03  # 3 cm
                    _n_steps = max(1, int(np.ceil(_shift_mag / _step_size)))
                    _last_safe_x, _last_safe_z = _start_x, _start_z
                    _stopped_at = _shift_mag
                    for _i in range(1, _n_steps + 1):
                        _t = min(_i * _step_size, _shift_mag)
                        _try_x = _start_x + (_shift_x / _shift_mag) * _t
                        _try_z = _start_z + (_shift_z / _shift_mag) * _t
                        p["position_m"][0] = _try_x
                        p["position_m"][2] = _try_z
                        if any(_poly_collides(p, _op)
                               for _op in placements
                               if _op is not p and not _op.get("is_carpet")
                               and not _op.get("wall_mounted")):
                            p["position_m"][0] = _last_safe_x
                            p["position_m"][2] = _last_safe_z
                            _stopped_at = _t - _step_size
                            print(f"  [wall_clearance] idx={idx} ({obj_type}) "
                                  f"moved {max(_stopped_at, 0):.3f}m of {_shift_mag:.3f}m "
                                  f"away from wall (stopped: would collide)")
                            break
                        _last_safe_x, _last_safe_z = _try_x, _try_z
                    else:
                        if _shift_mag > 0.05:
                            print(f"  [wall_clearance] idx={idx} ({obj_type}) "
                                  f"moved {_shift_mag:.3f}m away from wall (clean)")

            # Store mask-projected origin for collision resolution —
            # the solver will prefer moving whichever object deviates less.
            # Only set here if the per-iteration mask_align block didn't
            # already capture a silhouette-anchor (i.e. mfd was None).
            if "_mask_origin_xz" not in p:
                p["_mask_origin_xz"] = [p["position_m"][0], p["position_m"][2]]

            placed_by_idx[idx] = p
            placements.append(p)
            print(f"  position_m = {np.round(p['position_m'], 3).tolist()}")
            print(f"  scale      = {np.round(p['scale'], 3).tolist()}")

            # ── Animation frame: object at final position (post-refine) ────────
            if _anim_frame_render is not None:
                _anim_frame_render(placements,
                                   label=f"idx{idx}_{obj_type}_final",
                                   highlight_idx=idx)

        except Exception as _floor_exc:
            import traceback
            print(f"\n  [place_furn] EXCEPTION placing idx={idx} {obj_type}: {_floor_exc}")
            traceback.print_exc()
            continue

    # ── Same-set scale harmonization ─────────────────────────────────────────
    # Objects of the same type and wall_affinity that appear to be a matching
    # set (e.g. two identical sofas placed symmetrically) should have the same
    # scale.  Normalize them to their geometric mean so neither looks oddly
    # larger or smaller than its pair.  Only applies when scales are within
    # 50% of each other (so intentionally different-sized objects are untouched).
    _floor_placed = [p for p in placements
                     if not p.get("is_carpet") and not p.get("wall_mounted")
                     and p.get("on_top_of") is None]
    # Group by (type, wall_affinity)
    from itertools import combinations as _combinations
    _seen_harmonized: set[tuple[int, int]] = set()
    # Only types that *physically* tend to come as matched sets — sectional
    # sofas, dining stools, etc. Chairs/tables vary widely in real spaces
    # (e.g. a desk chair next to a lounge chair) so we don't pair them.
    _HARMONIZE_TYPES = {"sofa", "stool"}
    for _pa, _pb in _combinations(_floor_placed, 2):
        if _pa.get("type") != _pb.get("type"):
            continue
        if _pa.get("type") not in _HARMONIZE_TYPES:
            continue
        # Harmonize if same wall_affinity (clearly a matched pair on one wall)
        # OR if both have back-wall affinity on different walls but same type
        # (e.g. two sofas forming an L-shaped seating arrangement).
        _wa_a = _pa.get("wall_affinity", "centre")
        _wa_b = _pb.get("wall_affinity", "centre")
        if _wa_a != _wa_b and not (_wa_a in ("back", "left", "right") and
                                    _wa_b in ("back", "left", "right")):
            continue
        _sa = float(_pa["scale"][0])
        _sb = float(_pb["scale"][0])
        if _sa <= 0 or _sb <= 0:
            continue
        _ratio = max(_sa, _sb) / min(_sa, _sb)
        if _ratio > 1.5:
            continue  # too different — not the same set
        _pair_key = (min(_pa["index"], _pb["index"]), max(_pa["index"], _pb["index"]))
        if _pair_key in _seen_harmonized:
            continue
        _seen_harmonized.add(_pair_key)
        # Use the smaller scale — only scale DOWN to match, never scale UP.
        # Scaling up causes collisions; the smaller object is already correct.
        _mean_s = min(_sa, _sb)
        for _p_harm in (_pa, _pb):
            _old_s = float(_p_harm["scale"][0])
            if abs(_old_s - _mean_s) < 0.005:
                continue
            _scale_factor = _mean_s / _old_s
            _p_harm["scale"] = [_mean_s, _mean_s, _mean_s]
            sm = _p_harm.get("size_m", {})
            if sm:
                _p_harm["size_m"] = {
                    "width_m":  sm.get("width_m",  1.0) * _scale_factor,
                    "height_m": sm.get("height_m", 1.0) * _scale_factor,
                    "depth_m":  sm.get("depth_m",  1.0) * _scale_factor,
                }
                _p_harm["eff_h"] = _p_harm["size_m"]["height_m"]
            print(f"  [set_scale] idx={_p_harm['index']} ({_p_harm['type']}) "
                  f"harmonized {_old_s:.3f} → {_mean_s:.3f} "
                  f"(pair with idx={_pa['index'] if _p_harm is _pb else _pb['index']})")

    # ── Animation frame: after scale harmonization ───────────────────────────
    if _anim_frame_render is not None and _seen_harmonized:
        _anim_frame_render(placements, label="scale_harmonized")

    # ── Centre-seating conversation orientation ──────────────────────────────
    # A free-standing chair/armchair that was NOT tucked at a desk (no
    # facing_toward-desk rotation applied) should face the room's conversation
    # focal point — the coffee table if one exists, else the sofa, else the room
    # centre.  Without this it keeps whatever the VLM front-view check guessed,
    # which is unreliable for centre pieces (it can end up facing a wall — e.g.
    # an armchair the VLM tagged "facing_toward sofa", which the desk-only facing
    # path skips, falling through to a wrong orientation).  Rotation only — the
    # mask-derived position is preserved.
    _CONV_SEATING = {"chair", "armchair", "stool", "office_chair", "ottoman"}
    _focus = None
    for _ftypes in ({"coffee_table", "dining_table", "table"},
                    {"sofa", "couch", "loveseat", "sectional"}):
        for _q in placements:
            if (_q.get("type", "").lower().replace("-", "_") in _ftypes
                    and _q.get("position_m")):
                _focus = _q
                break
        if _focus is not None:
            break
    if _focus is not None:
        _fx, _fz = float(_focus["position_m"][0]), float(_focus["position_m"][2])
        for _ch in placements:
            if _ch.get("type", "").lower().replace("-", "_") not in _CONV_SEATING:
                continue
            if _ch.get("_facing_applied") or _ch is _focus:
                continue   # already oriented to a desk/table, or it's the focus
            if not _ch.get("position_m"):
                continue   # unplaced detection still in the list — skip, don't crash
            _wa_ch = _ch.get("wall_affinity", "centre")
            _lf = np.array(_ch.get("_local_front", [0, 0, -1]), dtype=np.float64)
            _cx, _cz = float(_ch["position_m"][0]), float(_ch["position_m"][2])
            _ddx, _ddz = _fx - _cx, _fz - _cz
            _dd = float(np.hypot(_ddx, _ddz))
            if _dd < 0.20:
                continue
            _face = np.array([_ddx / _dd, 0.0, _ddz / _dd], dtype=np.float64)
            # Wall-adjacent chairs: also angle them toward the conversation focal
            # (a wall chair facing straight off the wall reads as "facing away" of
            # the group), but ONLY if facing the focal still points its front INTO
            # the room — never rotate it to face into its own wall.
            _into_room_dir = {"back": (0.0, 1.0), "left": (1.0, 0.0),
                              "right": (-1.0, 0.0), "front": (0.0, -1.0)}.get(_wa_ch)
            if _into_room_dir is not None and (
                    _face[0] * _into_room_dir[0] + _face[2] * _into_room_dir[1]) <= 0.1:
                continue   # focal is behind the wall — keep the off-wall facing
            _a_from = np.arctan2(float(_lf[0]), float(_lf[2]))
            _a_to   = np.arctan2(float(_face[0]), float(_face[2]))
            _ch["rotation_3x3"] = _rotation_y(_a_to - _a_from).tolist()
            _ch["_facing_applied"] = True
            _ch["_facing_target_idx"] = _focus.get("index")
            print(f"  [conv_orient] idx={_ch.get('index')} {_ch.get('type')} → face "
                  f"focus idx={_focus.get('index')} ({_focus.get('type')}) "
                  f"dir={np.round(_face, 2).tolist()}")

    # ── Chair/seating height sanity check vs. desk/table ─────────────────────
    # A chair back slightly above the desk surface is normal (e.g. 0.9 m chair
    # vs 0.75 m desk, ratio ≈1.2).  Only intervene when the chair is grossly
    # oversized — more than 50% taller than the desk (ratio > 1.5), which
    # indicates a reconstruction error.  Target 1.3× desk height so the back
    # still sits above the desk surface at the correct proportion.
    _SEATING_H_TYPES = {"chair", "armchair", "stool", "office_chair"}
    # ONLY desks / dining tables — a chair is tucked UNDER these, so it must not
    # tower over them.  A coffee table is NOT a tuck-under surface: an armchair
    # beside a coffee table is a conversation layout and is naturally 2–2.5×
    # taller, so including coffee_table here wrongly shrank wingbacks to ~0.5 m.
    _DESK_H_TYPES    = {"desk", "dining_table"}
    for _ch in placements:
        if _ch.get("type", "").lower().replace("-", "_") not in _SEATING_H_TYPES:
            continue
        _tgt_idx = _ch.get("_facing_target_idx") or _ch.get("facing_toward")
        if _tgt_idx is None:
            continue
        _desk = placed_by_idx.get(_tgt_idx)
        if _desk is None or _desk.get("type", "").lower().replace("-", "_") not in _DESK_H_TYPES:
            continue
        _chair_h = float(_ch.get("eff_h", _ch.get("size_m", {}).get("height_m", 0)))
        _desk_h  = float(_desk.get("eff_h", _desk.get("size_m", {}).get("height_m", 0)))
        if _desk_h < 0.1 or _chair_h < 0.1:
            continue
        _ratio = _chair_h / _desk_h
        if _ratio > 1.5:
            _max_chair_h = _desk_h * 1.3
            _h_scale = _max_chair_h / _chair_h
            _old_s = float(_ch["scale"][0])
            _new_s = _old_s * _h_scale
            _ch["scale"] = [_new_s, _new_s, _new_s]
            for _dim in ("width_m", "height_m", "depth_m"):
                if _dim in _ch.get("size_m", {}):
                    _ch["size_m"][_dim] *= _h_scale
            _ch["eff_h"] = _chair_h * _h_scale
            print(f"  [chair_height] idx={_ch.get('index')} ({_ch.get('type')}) "
                  f"height {_chair_h:.3f}m >> desk {_desk_h:.3f}m (ratio {_ratio:.2f}) — "
                  f"scaled down {_old_s:.3f} → {_new_s:.3f} (new h={_ch['eff_h']:.3f}m)")
        else:
            print(f"  [chair_height] idx={_ch.get('index')} ({_ch.get('type')}) "
                  f"height {_chair_h:.3f}m vs desk {_desk_h:.3f}m "
                  f"(ratio {_ratio:.2f}) — within normal range, no adjustment")

    # ── Collision resolution ────────────────────────────────────────────────
    # Check all pairs of placed furniture for XZ bounding box overlap.
    # For each collision, evaluate BOTH objects as candidate movers and pick
    # the one whose push causes the smallest deviation from its mask-projected
    # silhouette position (_mask_origin_xz).  This keeps objects close to where
    # the segmentation mask says they should be.
    _COLLISION_MARGIN = 0.05  # 5cm gap after resolution

    # Get room dims once for collision resolution
    try:
        _col_room_w, _col_room_d = _get_room_dims(
            Path(placements[0].get("glb_path", ".")) if placements else Path("."), cam)
    except Exception:
        _col_room_w, _col_room_d = 5.0, 5.0

    def _available_space(p):
        """How much room p has to move in each direction before hitting a wall.
        Returns (space_+x, space_-x, space_+z, space_-z)."""
        hx, hz = _col_xz_half_extents(p)
        x, z = p["position_m"][0], p["position_m"][2]
        return (
            _col_room_w - x - hx,   # +X (toward right wall)
            x - hx,                  # -X (toward left wall)
            _col_room_d - z - hz,    # +Z (toward camera)
            z - hz,                  # -Z (toward back wall)
        )

    # Camera depth direction for collision push (projected to XZ plane).
    # Pushing along camera depth keeps objects at roughly the same screen-X,
    # avoiding lateral shifts that cascade into new collisions.
    _cam_pos = np.array(cam["position_m"], dtype=np.float64)
    _cam_look = np.array(cam["look_at_m"], dtype=np.float64)
    _cam_depth = _cam_look - _cam_pos
    _cam_depth[1] = 0.0  # project to XZ plane
    _cam_depth_n = np.linalg.norm(_cam_depth)
    if _cam_depth_n > 1e-6:
        _cam_depth /= _cam_depth_n
    else:
        _cam_depth = np.array([0.0, 0.0, -1.0])  # fallback: toward back wall

    def _push_vector(pm, pa, overlap_x, overlap_z):
        """Compute push direction and distance for moving pm away from pa.
        For wall-affinity objects, pushes along the wall.  For centre objects,
        pushes along camera depth direction to minimise lateral screen shift.
        Returns (direction, distance) or (None, inf) if object can't move."""
        dx = pm["position_m"][0] - pa["position_m"][0]
        dz = pm["position_m"][2] - pa["position_m"][2]
        wa = pm.get("wall_affinity", "centre")
        if pm.get("_facing_applied"):
            wa = "centre"
        space = _available_space(pm)  # (+x, -x, +z, -z)

        # For wall-affinity objects, push ALONG the wall (not camera depth)
        # to avoid pulling them away from the wall.
        if wa == "back":
            # Along-wall = X axis.  Push away from anchor in X.
            push_dir = np.array([1.0 if dx >= 0 else -1.0, 0.0, 0.0])
        elif wa == "left" or wa == "right":
            # Along-wall = Z axis.  Push away from anchor in Z.
            push_dir = np.array([0.0, 0.0, 1.0 if dz >= 0 else -1.0])
        else:
            # Centre: push along camera depth axis.
            sep = np.array([dx, 0.0, dz])
            along_cam = float(np.dot(sep, _cam_depth))
            if along_cam >= 0:
                push_dir = _cam_depth.copy()
            else:
                push_dir = -_cam_depth.copy()

        # Compute how far we need to push along this direction to clear the
        # AABB overlap.  The push must resolve overlap on BOTH axes.
        # dist_needed = max(overlap_x / |push_dir.x|, overlap_z / |push_dir.z|)
        abs_dx = abs(float(push_dir[0]))
        abs_dz = abs(float(push_dir[2]))
        needs = []
        if abs_dx > 1e-6 and overlap_x > 0:
            needs.append(overlap_x / abs_dx)
        if abs_dz > 1e-6 and overlap_z > 0:
            needs.append(overlap_z / abs_dz)
        if not needs:
            return None, float("inf")
        dist = max(needs) + _COLLISION_MARGIN

        # Check available space along push direction
        # Project push_dir onto +X/-X and +Z/-Z to sum available room
        avail_x = space[0] if push_dir[0] > 0 else space[1]
        avail_z = space[2] if push_dir[2] > 0 else space[3]
        # Conservative: check both axes have enough room
        if abs_dx > 1e-6 and avail_x < dist * abs_dx * 0.5:
            # Not enough X room — try opposite direction
            push_dir = -push_dir
            avail_x = space[0] if push_dir[0] > 0 else space[1]
            avail_z = space[2] if push_dir[2] > 0 else space[3]
            if abs_dx > 1e-6 and avail_x < dist * abs_dx * 0.5:
                return None, float("inf")
        if abs_dz > 1e-6 and avail_z < dist * abs_dz * 0.5:
            push_dir = -push_dir
            avail_x = space[0] if push_dir[0] > 0 else space[1]
            avail_z = space[2] if push_dir[2] > 0 else space[3]
            if abs_dz > 1e-6 and avail_z < dist * abs_dz * 0.5:
                return None, float("inf")

        return push_dir, dist

    def _move_cost(p, push_dir, push_dist):
        """Cost of moving p: deviation from mask origin + penalties.
        Lower is better (cheaper to move).  Returns inf for immovable."""
        if push_dir is None:
            return float("inf")
        wa = p.get("wall_affinity", "centre")
        # Wall-affinity objects can slide ALONG their wall but not away from it.
        # Block pushes where >30% of the direction is perpendicular to the wall.
        if wa != "centre" and not p.get("_facing_applied"):
            _perp_threshold = 0.3
            if wa == "back" and abs(float(push_dir[2])) > _perp_threshold:
                return float("inf")
            if wa in ("left", "right") and abs(float(push_dir[0])) > _perp_threshold:
                return float("inf")
        origin = p.get("_mask_origin_xz")
        if origin is None:
            return float("inf")
        new_x = p["position_m"][0] + float(push_dir[0]) * push_dist
        new_z = p["position_m"][2] + float(push_dir[2]) * push_dist
        deviation = np.sqrt((new_x - origin[0])**2 + (new_z - origin[1])**2)
        # Wall-affinity objects are strongly anchored — 10x penalty makes
        # centre objects preferred movers in almost all cases.
        if wa != "centre" and not p.get("_facing_applied"):
            deviation *= 10.0
        return deviation

    # Iteratively resolve collisions one at a time.  After each move,
    # re-detect collisions with updated positions so we don't double-push.
    # For each collision pair, the cheaper-to-move object is selected.
    _move_count: dict[int, int] = {}  # index → number of times moved
    _MAX_MOVES_PER_OBJ = 2    # cap total slides — small local fixes only
    _MAX_PUSH = 0.08          # 8 cm max per push (was 20 cm)
    _floor = [p for p in placements
              if not p.get("is_carpet") and not p.get("wall_mounted")
              and p.get("on_top_of") is None]
    _exhausted: set[int] = set()  # indices that can't be moved further
    # Track last position to detect oscillation (object bouncing back and forth)
    _prev_pos: dict[int, tuple[float, float]] = {}
    for _col_pass in range(len(_floor) * _MAX_MOVES_PER_OBJ):
        # Find the collision pair with the cheapest resolution
        best = None  # (mover, colliders_list, combined_push, push_len, cost)
        for pi in _floor:
            pi_idx = pi.get("index", -1)
            if pi_idx in _exhausted:
                continue
            if _move_count.get(pi_idx, 0) >= _MAX_MOVES_PER_OBJ:
                _exhausted.add(pi_idx)
                continue
            hx_i, hz_i = _col_xz_half_extents(pi)
            collisions = []
            # Skip collision between a _facing_applied object and its target
            _facing_tgt = pi.get("_facing_target_idx")
            for pj in _floor:
                if pj is pi:
                    continue
                # Chair placed at desk — they're supposed to be adjacent
                if _facing_tgt is not None and pj.get("index") == _facing_tgt:
                    continue
                if pj.get("_facing_target_idx") == pi_idx:
                    continue
                hx_j, hz_j = _col_xz_half_extents(pj)
                xi, zi = pi["position_m"][0], pi["position_m"][2]
                xj, zj = pj["position_m"][0], pj["position_m"][2]
                overlap_x = (hx_i + hx_j) - abs(xi - xj)
                overlap_z = (hz_i + hz_j) - abs(zi - zj)
                # Dining set exemption: a dining table is intentionally surrounded
                # by its tucked-in chairs — don't resolve table↔chair as a
                # collision, or the table gets shoved off-centre from its chairs.
                _ti = pi.get("type", "").lower().replace("-", " ").replace("_", " ")
                _tj = pj.get("type", "").lower().replace("-", " ").replace("_", " ")
                _dt_i = ("table" in _ti and any(k in _ti for k in ("dining", "meeting", "round"))
                         and not any(k in _ti for k in ("coffee", "side", "end", "cocktail", "console", "nightstand")))
                _dt_j = ("table" in _tj and any(k in _tj for k in ("dining", "meeting", "round"))
                         and not any(k in _tj for k in ("coffee", "side", "end", "cocktail", "console", "nightstand")))
                if ("chair" in _ti and _dt_j) or (_dt_i and "chair" in _tj):
                    continue
                # Use SAT hull collision if available, AABB otherwise
                if _poly_collides(pi, pj):
                    collisions.append((pj, overlap_x, overlap_z))
            if not collisions:
                continue
            # Compute combined push for pi from all its current collisions
            combined = np.zeros(3, dtype=np.float64)
            for pj, ov_x, ov_z in collisions:
                d, dist = _push_vector(pi, pj, ov_x, ov_z)
                if d is not None:
                    combined += d * dist
            push_len = float(np.linalg.norm(combined))
            if push_len < 0.001:
                continue
            push_dir = combined / push_len
            cost = _move_cost(pi, push_dir, push_len)
            if best is None or cost < best[4]:
                best = (pi, collisions, combined, push_len, cost)
        if best is None:
            break  # no more collisions
        pi, collisions, combined, push_len, cost = best
        if cost == float("inf"):
            _col_names = ", ".join(f"idx={pj.get('index')}" for pj, _, _ in collisions)
            print(f"  [collision] idx={pi.get('index')} ({pi.get('type')}) "
                  f"collides with [{_col_names}] — immovable, skipping")
            _exhausted.add(pi.get("index", -1))
            continue
        pi_idx = pi.get("index", -1)
        # Constrain push: wall_affinity is initial-only — forward-push
        # (perpendicular, away from the wall) is allowed and PREFERRED here
        # because it preserves the silhouette's along-wall position (which is
        # what the eye notices in monocular view), trading depth (harder to
        # judge) for accurate horizontal placement.  Only block movement
        # INTO the wall.
        _pi_wa = pi.get("wall_affinity", "centre")
        if _pi_wa == "back" and combined[2] < 0.0:
            combined[2] = 0.0
        elif _pi_wa == "left" and combined[0] < 0.0:
            combined[0] = 0.0
        elif _pi_wa == "right" and combined[0] > 0.0:
            combined[0] = 0.0
        # Facing objects: only push perpendicular to facing axis so the
        # object stays at the same distance/angle from its target.
        _ftgt_idx = pi.get("_facing_target_idx")
        if _ftgt_idx is not None:
            _ftgt = next((p for p in _floor if p.get("index") == _ftgt_idx), None)
            if _ftgt is not None:
                _fax = np.array([_ftgt["position_m"][0] - pi["position_m"][0],
                                 0.0,
                                 _ftgt["position_m"][2] - pi["position_m"][2]])
                _fax_len = float(np.linalg.norm(_fax))
                if _fax_len > 0.01:
                    _fax /= _fax_len
                    # Remove component along facing axis
                    _proj = float(np.dot(combined, _fax))
                    combined -= _proj * _fax
        push_len = float(np.linalg.norm(combined))
        if push_len < 0.001:
            _exhausted.add(pi_idx)
            continue
        # Cap push to prevent large single-step cascades
        if push_len > _MAX_PUSH:
            combined = combined * (_MAX_PUSH / push_len)
            push_len = _MAX_PUSH

        # Check if full push creates NEW collisions with non-involved objects.
        # If so, decompose into X-only and Z-only and use whichever is safe.
        _already_col_idxs = {pj.get("index") for pj, _, _ in collisions}
        _hx_pi, _hz_pi = _col_xz_half_extents(pi)

        def _push_creates_new(nx: float, nz: float) -> bool:
            # Temporarily move pi to test position, then check SAT collision
            _old_x, _old_z = pi["position_m"][0], pi["position_m"][2]
            pi["position_m"][0] = nx
            pi["position_m"][2] = nz
            _collides = False
            for _pc in _floor:
                if _pc is pi or _pc.get("is_carpet"):
                    continue
                if _pc.get("index") in _already_col_idxs:
                    continue
                if _poly_collides(pi, _pc):
                    _collides = True
                    break
            pi["position_m"][0] = _old_x
            pi["position_m"][2] = _old_z
            return _collides

        _new_x = pi["position_m"][0] + float(combined[0])
        _new_z = pi["position_m"][2] + float(combined[2])
        _cur_x, _cur_z = pi["position_m"][0], pi["position_m"][2]

        if _push_creates_new(_new_x, _new_z):
            # Full push blocked — try X-only then Z-only
            _applied_x = not _push_creates_new(_new_x, _cur_z) and abs(combined[0]) > 0.001
            _applied_z = not _push_creates_new(_cur_x if not _applied_x else _new_x, _new_z) and abs(combined[2]) > 0.001
            if _applied_x:
                _new_z = _cur_z
            if _applied_z:
                _new_x = _cur_x if not _applied_x else _new_x
            if not _applied_x and not _applied_z:
                _exhausted.add(pi_idx)
                print(f"  [collision] idx={pi_idx} ({pi.get('type')}) "
                      f"push blocked — would create new collision, marking exhausted")
                continue
            push_len = float(np.sqrt((_new_x - _cur_x)**2 + (_new_z - _cur_z)**2))

        _safe_dist = push_len
        # Oscillation detection: if we're returning to a recently visited position, stop
        _prev = _prev_pos.get(pi_idx)
        if _prev is not None and abs(_new_x - _prev[0]) < 0.02 and abs(_new_z - _prev[1]) < 0.02:
            # Last-chance escape for tables wedged between same-wall seating:
            # the wall_affinity was zeroing out the perpendicular axis above,
            # so the table can only move sideways into the seating. Nudge it
            # off the wall (a real coffee table sits in front of the sofas)
            # and re-enter the loop with wall_affinity downgraded to "centre".
            _TABLE_TYPES   = {"coffee_table", "side_table", "end_table", "ottoman"}
            _SEATING_TYPES = {"sofa", "loveseat", "chair", "armchair"}
            _pi_wa_orig    = pi.get("wall_affinity", "centre")
            _same_wall_seating = any(
                (pj.get("type") in _SEATING_TYPES and pj.get("wall_affinity") == _pi_wa_orig)
                for pj, _, _ in collisions)
            if (pi.get("type") in _TABLE_TYPES
                and _same_wall_seating
                and _pi_wa_orig in ("back", "left", "right")):
                _nudge = 0.15   # cap "escape" nudge for tables wedged between same-wall seating
                _nx, _nz = pi["position_m"][0], pi["position_m"][2]
                if   _pi_wa_orig == "back":   _nz += _nudge   # +Z = into room
                elif _pi_wa_orig == "left":   _nx += _nudge
                elif _pi_wa_orig == "right":  _nx -= _nudge
                pi["position_m"][0] = _nx
                pi["position_m"][2] = _nz
                pi["wall_affinity"] = "centre"        # release the wall lock
                _prev_pos[pi_idx]    = (_nx, _nz)
                _move_count[pi_idx]  = _move_count.get(pi_idx, 0) + 1
                print(f"  [collision] idx={pi_idx} ({pi.get('type')}) "
                      f"escape: nudged {_nudge:.2f}m off {_pi_wa_orig} wall "
                      f"(was wedged between same-wall seating)")
                continue
            _exhausted.add(pi_idx)
            print(f"  [collision] idx={pi_idx} ({pi.get('type')}) oscillating — marking exhausted")
            continue
        # Hard cap on cumulative mask drift: if applying this push would take
        # the object more than _MAX_MASK_DEVIATION m from its mask anchor,
        # abandon further pushes (the silhouette location is more trustworthy
        # than nudging the object across the room to clear a collision).
        # For wall-affine objects we decompose the deviation into:
        #   • along-wall (visible in monocular photo, tight cap)
        #   • perpendicular / forward off the wall (depth, harder to judge,
        #     loose cap so collisions can resolve by stepping objects forward
        #     instead of dragging them sideways across the silhouette).
        _MAX_MASK_DEVIATION       = 0.20   # default (Euclidean) for centre objects
        _MAX_MASK_DEV_ALONG_WALL  = 0.20   # wall-affine, along-wall (silhouette X)
        _MAX_MASK_DEV_OFF_WALL    = 0.55   # wall-affine, forward off-wall (depth)
        # Small free-standing tables (side/coffee/end tables) have a far less
        # reliable silhouette anchor than large anchored pieces, and an embedded
        # table reads as clearly wrong — so let them slide much further to clear a
        # collision instead of staying half-buried in the sofa.
        _SMALL_TABLE_T = {"side_table", "round_table", "end_table", "nightstand",
                          "coffee_table", "cocktail_table"}
        if pi.get("type", "").lower().replace("-", "_") in _SMALL_TABLE_T:
            _MAX_MASK_DEVIATION      = 1.2
            _MAX_MASK_DEV_ALONG_WALL = 1.2
            _MAX_MASK_DEV_OFF_WALL   = 1.2
        _org_xz = pi.get("_mask_origin_xz")
        if _org_xz is not None:
            _dx_dev = _new_x - _org_xz[0]
            _dz_dev = _new_z - _org_xz[1]
            if _pi_wa == "back":
                _along, _perp = abs(_dx_dev), abs(_dz_dev)
            elif _pi_wa in ("left", "right"):
                _along, _perp = abs(_dz_dev), abs(_dx_dev)
            else:
                _along = _perp = -1.0   # signals "use Euclidean cap below"
            if _along >= 0:
                # Wall-affine: per-axis caps
                if _along > _MAX_MASK_DEV_ALONG_WALL or _perp > _MAX_MASK_DEV_OFF_WALL:
                    _exhausted.add(pi_idx)
                    _col_names = ", ".join(f"idx={pj.get('index')}" for pj, _, _ in collisions)
                    print(f"  [collision] idx={pi_idx} ({pi.get('type')}) "
                          f"collides with [{_col_names}] — push would drift "
                          f"along={_along:.3f}m (cap {_MAX_MASK_DEV_ALONG_WALL:.2f}) "
                          f"perp={_perp:.3f}m (cap {_MAX_MASK_DEV_OFF_WALL:.2f}); "
                          f"leaving as-is, silhouette anchor wins")
                    continue
            else:
                # Centre / non-wall-affine: original Euclidean cap
                _proj_dev = float(np.sqrt(_dx_dev**2 + _dz_dev**2))
                if _proj_dev > _MAX_MASK_DEVIATION:
                    _exhausted.add(pi_idx)
                    _col_names = ", ".join(f"idx={pj.get('index')}" for pj, _, _ in collisions)
                    print(f"  [collision] idx={pi_idx} ({pi.get('type')}) "
                          f"collides with [{_col_names}] — "
                          f"push would drift {_proj_dev:.3f}m from mask "
                          f"(cap {_MAX_MASK_DEVIATION:.2f}m); leaving as-is, "
                          f"silhouette anchor wins")
                    continue
        _prev_pos[pi_idx] = (_cur_x, _cur_z)
        pi["position_m"][0] = _new_x
        pi["position_m"][2] = _new_z
        _move_count[pi_idx] = _move_count.get(pi_idx, 0) + 1
        _org = pi.get("_mask_origin_xz")
        _dev = (np.sqrt((pi["position_m"][0] - _org[0])**2
                      + (pi["position_m"][2] - _org[1])**2) if _org else 0)
        _col_names = ", ".join(f"idx={pj.get('index')}" for pj, _, _ in collisions)
        print(f"  [collision] idx={pi_idx} ({pi.get('type')}) "
              f"collides with [{_col_names}] "
              f"— pushed {_safe_dist:.3f}m (mask deviation: {_dev:.3f}m)")
    if _move_count:
        # Post-collision recovery: pull objects back toward their pre-collision
        # mask-aligned position in world space.  We avoid re-running mask_align
        # here because perspective projection couples X/Z shifts — a leftward
        # pixel correction drags Z toward the camera, making things worse.
        # Instead, directly interpolate back toward _mask_origin_xz.
        for p in placements:
            pi_idx = p.get("index", -1)
            if pi_idx not in _move_count:
                continue
            origin = p.get("_mask_origin_xz")
            if origin is None:
                continue
            cur_x, cur_z = p["position_m"][0], p["position_m"][2]
            org_x, org_z = origin[0], origin[1]
            # Pull back 80% toward the mask-aligned origin, but only along
            # the wall for wall-affinity objects (don't pull perpendicular).
            _alpha = 0.8
            _wa = p.get("wall_affinity", "centre")
            new_x = cur_x + _alpha * (org_x - cur_x)
            new_z = cur_z + _alpha * (org_z - cur_z)
            # Wall-affinity: only pull along the wall axis
            if _wa == "back":
                new_z = cur_z  # don't change Z (perpendicular to back wall)
            elif _wa in ("left", "right"):
                new_x = cur_x  # don't change X (perpendicular to side wall)
            dx = new_x - cur_x
            dz = new_z - cur_z
            if abs(dx) > 0.01 or abs(dz) > 0.01:
                # Check that pulling back doesn't re-introduce collisions (SAT)
                _old_x2, _old_z2 = p["position_m"][0], p["position_m"][2]
                p["position_m"][0] = new_x
                p["position_m"][2] = new_z
                _would_collide = any(
                    _poly_collides(p, pj)
                    for pj in _floor if pj is not p and not pj.get("is_carpet")
                )
                p["position_m"][0] = _old_x2
                p["position_m"][2] = _old_z2
                if _would_collide:
                    print(f"  [post_collision_align] idx={pi_idx} ({p.get('type')}) "
                          f"pullback blocked — would re-collide")
                else:
                    p["position_m"][0] = new_x
                    p["position_m"][2] = new_z
                    print(f"  [post_collision_align] idx={pi_idx} ({p.get('type')}) "
                          f"pulled back toward mask origin: "
                          f"({cur_x:.3f},{cur_z:.3f}) → ({new_x:.3f},{new_z:.3f}) "
                          f"(shift {dx:.3f},{dz:.3f})")

        # Re-clamp to room bounds after collision resolution
        for p in placements:
            if p.get("is_carpet") or p.get("wall_mounted"):
                continue
            if "size_m" not in p:
                continue
            try:
                _fin_room_w, _fin_room_d = _get_room_dims(Path(p.get("glb_path", ".")), cam)
                _local_hx = p["size_m"]["width_m"] / 2.0
                _local_hz = p["size_m"]["depth_m"] / 2.0
                _R = np.array(p["rotation_3x3"], dtype=np.float64)
                _hx = abs(float(_R[0][0])) * _local_hx + abs(float(_R[0][2])) * _local_hz
                _hz = abs(float(_R[2][0])) * _local_hx + abs(float(_R[2][2])) * _local_hz
                p["position_m"][0] = float(np.clip(
                    p["position_m"][0], _hx, max(_fin_room_w - _hx, _hx)))
                p["position_m"][2] = float(np.clip(
                    p["position_m"][2], _hz, max(_fin_room_d - _hz, _hz)))
            except Exception:
                pass
        # NOTE: `_mask_origin_xz` was previously stripped here, but that
        # silently disabled every post-loop cap (sr_collision, corner_escape,
        # aisle_clear, final_anchor) because those guards check `if origin is
        # not None`. Stripping is now done at the very end, right before the
        # JSON dump.

    # ── Same-wall pair resolver: forward push (perpendicular off the wall) ────
    # User preference: when two objects share a wall and still overlap after the
    # main collision loop, prefer staggering them FORWARD (off the wall, in the
    # perpendicular axis) over sliding them along the wall.  Along-wall sliding
    # deviates from the silhouette's horizontal anchor (which is what the eye
    # judges in monocular view), while forward push only changes depth (harder
    # to perceive).  Wall-affinity is treated as initial-only here.
    try:
        _ws_room_w, _ws_room_d = _get_room_dims(
            Path(_floor[0].get("glb_path", ".")) if _floor else Path("."), cam)
    except Exception:
        _ws_room_w, _ws_room_d = 5.0, 5.0

    # Per-wall: (perpendicular axis, forward sign, room-bound on that axis)
    _WALL_FORWARD = {
        "back":  (2, +1.0, _ws_room_d),   # +Z = forward into room
        "left":  (0, +1.0, _ws_room_w),   # +X = forward into room
        "right": (0, -1.0, _ws_room_w),   # -X = forward into room (room ends at 0)
    }

    # Cumulative forward-push budget per object across all passes.  Without this
    # cap, the loop iterates len(group)*4 times and each pass can apply
    # _avail (room-extent worth) of push — a small nightstand against the back
    # wall colliding with a wider bed has been pushed 1.8m forward into the
    # middle of the room, destroying its wall anchor.  Cap total displacement
    # of any single object so wall-affinity stays meaningful.
    _WALL_FWD_TOTAL_CAP = 0.25   # metres
    _wall_fwd_pushed: dict[int, float] = {}

    for _ws_wa in ("back", "left", "right"):
        _ws_group = [p for p in _floor if p.get("wall_affinity") == _ws_wa]
        if len(_ws_group) < 2:
            continue
        _perp_axis, _fwd_sign, _room_extent = _WALL_FORWARD[_ws_wa]

        for _ws_pass in range(len(_ws_group) * 4):
            _ws_moved = False
            for _i in range(len(_ws_group)):
                for _j in range(_i + 1, len(_ws_group)):
                    _pa, _pb = _ws_group[_i], _ws_group[_j]
                    if not _poly_collides(_pa, _pb):
                        continue

                    _hx_a, _hz_a = _col_xz_half_extents(_pa)
                    _hx_b, _hz_b = _col_xz_half_extents(_pb)
                    _hperp_a = _hx_a if _perp_axis == 0 else _hz_a
                    _hperp_b = _hx_b if _perp_axis == 0 else _hz_b
                    _ca = float(_pa["position_m"][_perp_axis])
                    _cb = float(_pb["position_m"][_perp_axis])

                    # Perpendicular overlap (signed, forward direction):
                    # how much further apart in the forward axis they need to be
                    # to clear, given current along-wall overlap.
                    _need = _hperp_a + _hperp_b + _COLLISION_MARGIN
                    _gap = abs(_ca - _cb)
                    _overlap_perp = max(0.0, _need - _gap) + 0.05  # 5cm safety pad

                    # Choose which to push forward: the one whose along-wall mask
                    # deviation is SMALLEST (it's still on its silhouette anchor,
                    # so nudging its depth costs the least).  Tie-break: the one
                    # with more forward space available.
                    def _along_dev(_p: dict) -> float:
                        _o = _p.get("_mask_origin_xz")
                        if _o is None:
                            return 0.0
                        if _ws_wa == "back":
                            return abs(float(_p["position_m"][0]) - float(_o[0]))
                        return abs(float(_p["position_m"][2]) - float(_o[1]))

                    def _fwd_space(_p: dict, _h: float) -> float:
                        _c = float(_p["position_m"][_perp_axis])
                        if _fwd_sign > 0:
                            return max(0.0, _room_extent - _c - _h - 0.02)
                        return max(0.0, _c - _h - 0.02)

                    _dev_a = _along_dev(_pa)
                    _dev_b = _along_dev(_pb)
                    _space_a = _fwd_space(_pa, _hperp_a)
                    _space_b = _fwd_space(_pb, _hperp_b)

                    # Pick pushee: smaller along-dev wins; tie-break by forward space
                    if abs(_dev_a - _dev_b) < 0.01:
                        _push_b = _space_b >= _space_a
                    else:
                        _push_b = _dev_b < _dev_a

                    _target = _pb if _push_b else _pa
                    _other  = _pa if _push_b else _pb
                    _avail  = _space_b if _push_b else _space_a

                    if _avail < 0.05:
                        # No forward space — try the other one
                        _target = _pa if _push_b else _pb
                        _other  = _pb if _push_b else _pa
                        _avail  = _space_a if _push_b else _space_b
                        if _avail < 0.05:
                            # Both blocked forward — leave overlap (rare; user prefers
                            # this over corrupting silhouette by sliding sideways)
                            continue

                    _already = _wall_fwd_pushed.get(id(_target), 0.0)
                    # Beds get a more generous cap because we compensate by
                    # shrinking their depth-along-front (so the foot silhouette
                    # stays roughly anchored even after a forward push).
                    _is_bed = _target.get("type") == "bed"
                    _cap_for_target = 0.55 if _is_bed else _WALL_FWD_TOTAL_CAP
                    _budget  = max(0.0, _cap_for_target - _already)
                    _push = min(_overlap_perp, _avail, _budget)
                    if _push < 0.005:
                        # Either small overlap, no forward space, or the per-
                        # object cumulative cap is reached — leaving overlap is
                        # better than displacing a wall-anchored object so far
                        # forward that it loses its wall anchor.
                        continue
                    _old_p = float(_target["position_m"][_perp_axis])
                    _new_p = _old_p + _fwd_sign * _push

                    # Don't push into a non-target neighbour
                    _target["position_m"][_perp_axis] = _new_p
                    _bad = any(
                        _q is not _target and not _q.get("is_carpet")
                        and not _q.get("wall_mounted")
                        and _q is not _other
                        and _poly_collides(_target, _q)
                        for _q in _floor)
                    if _bad:
                        _target["position_m"][_perp_axis] = _old_p
                        continue
                    _ws_moved = True
                    _wall_fwd_pushed[id(_target)] = _already + _push
                    print(f"  [wall_fwd] wa={_ws_wa} idx={_target.get('index')} "
                          f"({_target.get('type')}): pushed {_push:.3f}m forward "
                          f"({_old_p:.3f} → {_new_p:.3f}) to clear "
                          f"idx={_other.get('index')}  "
                          f"[cumul {_wall_fwd_pushed[id(_target)]:.3f}/"
                          f"{_cap_for_target:.2f}m]")

                    # Bed-only: shrink depth so the FOOT (camera-facing edge)
                    # stays anchored to its photo silhouette after the push.
                    # Math: pushing centre forward by _push moves the foot from
                    # z=depth to z=depth+push.  To pull the foot back to z=depth
                    # while keeping the new centre, shrink depth by 2*_push
                    # (centred shrink → foot moves back by depth_delta/2).
                    # Net result: foot at original photo position, back lifts
                    # off the wall by 2*_push (≤ ~30cm given the 0.55m cap).
                    if _is_bed and _ws_wa == "back":
                        _sz_loc = _target.setdefault("size_m", {})
                        _cur_dm = float(_sz_loc.get("depth_m", 0.0))
                        if _cur_dm > 0.5:
                            _new_dm = max(0.5, _cur_dm - 2.0 * _push)
                            _ratio  = _new_dm / _cur_dm
                            _cur_scale = list(_target.get("scale", [1, 1, 1]))
                            # Bed's local_front is +Z (deterministic for the
                            # detilted bed mesh); rotation for wall_aff="back"
                            # keeps local +Z → world +Z, so scaling local Z
                            # shrinks the dimension perpendicular to the wall.
                            _cur_scale[2] = float(_cur_scale[2]) * _ratio
                            _target["scale"] = _cur_scale
                            _sz_loc["depth_m"] = _new_dm
                            print(f"  [bed_depth_shrink] sz×{_ratio:.3f} "
                                  f"(depth_m {_cur_dm:.2f}→{_new_dm:.2f}m) — "
                                  f"foot silhouette anchored after fwd push")
            if not _ws_moved:
                break

    # ── Animation frame: after collision resolution + wall-spread ─────────────
    if _anim_frame_render is not None:
        _anim_frame_render(placements, label="collision_resolved")

    # ── Post-placement orientation verification ──────────────────────────────
    # Render the full scene and ask VLM to verify each seating object's
    # orientation against the reference photo.  This catches errors that
    # the pre-placement VLM orient missed.
    if vlm_post_verify:
        placements = _vlm_post_placement_verify(
            placements, seg_by_idx, cam, out_root,
        )

    # ── Scene-level VLM review ────────────────────────────────────────────────
    # After all individual placements and orientation checks, render the full
    # scene and ask VLM to compare with the reference photo.  Apply position/
    # scale corrections for any objects that are clearly misaligned.
    if vlm_scene_review:
        placements = _vlm_scene_review(placements, seg_by_idx, cam, out_root)

    # ── Post-scene-review collision resolution ────────────────────────────────
    # scene_review may have shifted objects into each other.  Run a quick
    # push-apart pass to clear any newly introduced collisions.
    _sr_floor = [p for p in placements
                 if not p.get("is_carpet") and not p.get("wall_mounted")
                 and p.get("on_top_of") is None]
    _sr_moved: set[int] = set()
    _sr_push_count: dict[int, int] = {}   # per-object push count
    _MAX_SR_PUSHES = 2
    _sr_prev_pos: dict[int, tuple[float, float]] = {}
    for _ in range(len(_sr_floor) * _MAX_SR_PUSHES):
        _best_sr = None
        for _pi in _sr_floor:
            _pi_idx = _pi.get("index", -1)
            if _pi_idx in _sr_moved:
                continue
            if _sr_push_count.get(_pi_idx, 0) >= _MAX_SR_PUSHES:
                _sr_moved.add(_pi_idx)
                continue
            _hxi, _hzi = _col_xz_half_extents(_pi)
            _cols = []
            _sr_facing_tgt = _pi.get("_facing_target_idx")
            for _pj in _sr_floor:
                if _pj is _pi:
                    continue
                # Skip facing pairs — desk and its chair are supposed to be adjacent
                if _sr_facing_tgt is not None and _pj.get("index") == _sr_facing_tgt:
                    continue
                if _pj.get("_facing_target_idx") == _pi_idx:
                    continue
                _hxj, _hzj = _col_xz_half_extents(_pj)
                _ox = (_hxi + _hxj) - abs(_pi["position_m"][0] - _pj["position_m"][0])
                _oz = (_hzi + _hzj) - abs(_pi["position_m"][2] - _pj["position_m"][2])
                if _poly_collides(_pi, _pj):
                    _cols.append((_pj, _ox, _oz))
            if not _cols:
                continue
            # Never move wall-affinity objects in sr_collision — they are
            # already positioned by mask_align relative to their wall.
            if _pi.get("wall_affinity", "centre") in ("back", "left", "right"):
                continue
            _cost = _move_cost(_pi, np.array([1.0, 0.0, 0.0]), 0.1)
            if _best_sr is None or _cost < _best_sr[1]:
                _best_sr = (_pi, _cost, _cols)
        if _best_sr is None:
            break
        _pi, _, _cols = _best_sr
        _pi_idx = _pi.get("index", -1)
        _comb = np.zeros(3, dtype=np.float64)
        for _pj, _ox, _oz in _cols:
            _d, _dist = _push_vector(_pi, _pj, _ox, _oz)
            if _d is not None:
                _comb += _d * _dist
        _pl = float(np.linalg.norm(_comb))
        if _pl < 0.001:
            _sr_moved.add(_pi_idx)
            continue
        if _pl > 0.08:
            _comb = _comb * (0.08 / _pl)
        _pi_wa = _pi.get("wall_affinity", "centre")
        if _pi_wa == "back":
            _comb[2] = 0.0
        elif _pi_wa in ("left", "right"):
            _comb[0] = 0.0

        # Anchor-aware push: a push that points away from the silhouette
        # anchor only drags the object further from where the photo says it
        # should be. Project out the anchor-receding component so the push
        # stays within the silhouette band, OR if the proposed push points
        # mostly away (>50% of magnitude), invert it to push *toward* the
        # anchor (perpendicular-to-camera direction, equivalent to "move
        # toward camera in the depth direction the silhouette suggested").
        _sr_anchor = _pi.get("_mask_origin_xz")
        if _sr_anchor is not None:
            _to_anc = np.array([_sr_anchor[0] - _pi["position_m"][0],
                                0.0,
                                _sr_anchor[1] - _pi["position_m"][2]],
                               dtype=np.float64)
            _to_anc_n = float(np.linalg.norm(_to_anc[[0, 2]]))
            if _to_anc_n > 0.05:  # only correct if currently > 5cm off anchor
                _to_anc_dir = _to_anc / max(_to_anc_n, 1e-6)
                _away_amt = float(_comb @ -_to_anc_dir)   # +ve = pointing away from anchor
                if _away_amt > 0.0:
                    # Project out the away-from-anchor component
                    _comb = _comb + _away_amt * _to_anc_dir
                    # If the resulting magnitude is too small to clear the
                    # collision, replace with a push *toward* the anchor of
                    # the same magnitude as before.
                    _comb_n = float(np.linalg.norm(_comb))
                    if _comb_n < 0.5 * _pl:
                        _comb = _to_anc_dir * min(_pl, _to_anc_n)
                        print(f"  [sr_collision] idx={_pi_idx} ({_pi.get('type')}) "
                              f"redirected push toward silhouette anchor "
                              f"({_away_amt:.3f}m of away-component removed)")

        _sr_new_x = _pi["position_m"][0] + float(_comb[0])
        _sr_new_z = _pi["position_m"][2] + float(_comb[2])
        # Oscillation check
        _sr_prev = _sr_prev_pos.get(_pi_idx)
        if (_sr_prev is not None
                and abs(_sr_new_x - _sr_prev[0]) < 0.02
                and abs(_sr_new_z - _sr_prev[1]) < 0.02):
            _sr_moved.add(_pi_idx)
            continue
        # Don't push more than 0.40m total from the mask-aligned origin
        _sr_origin = _pi.get("_mask_origin_xz")
        if _sr_origin is not None:
            _sr_dev = float(np.sqrt((_sr_new_x - _sr_origin[0])**2
                                   + (_sr_new_z - _sr_origin[1])**2))
            if _sr_dev > 0.15:
                _sr_moved.add(_pi_idx)
                continue
        # Don't apply push if it would introduce a new collision with an object
        # not already colliding with _pi (e.g. pushing a coffee table into a sofa).
        _sr_old_x, _sr_old_z = float(_pi["position_m"][0]), float(_pi["position_m"][2])
        _pi["position_m"][0] = _sr_new_x
        _pi["position_m"][2] = _sr_new_z
        _cols_set = {id(_pj) for _pj, _, _ in _cols}
        _sr_push_blocked = any(
            not _op.get("is_carpet") and not _op.get("wall_mounted")
            and _op is not _pi and id(_op) not in _cols_set
            and _poly_collides(_pi, _op)
            for _op in _sr_floor
        )
        if _sr_push_blocked:
            _pi["position_m"][0] = _sr_old_x
            _pi["position_m"][2] = _sr_old_z
            _sr_moved.add(_pi_idx)
            print(f"  [sr_collision] idx={_pi_idx} ({_pi.get('type')}) push BLOCKED — would hit another object")
            continue
        _sr_prev_pos[_pi_idx] = (_sr_old_x, _sr_old_z)
        _sr_push_count[_pi_idx] = _sr_push_count.get(_pi_idx, 0) + 1
        print(f"  [sr_collision] idx={_pi_idx} ({_pi.get('type')}) pushed "
              f"{float(np.linalg.norm(_comb)):.3f}m to clear scene_review overlap")

    # ── Snap-back toward mask origin ──────────────────────────────────────────
    # sr_collision may have pushed objects (especially chairs) away from their
    # silhouette-aligned positions.  Try to pull each deviated object back
    # toward _mask_origin_xz if no collision would result.
    for _snap_p in _sr_floor:
        _snap_origin = _snap_p.get("_mask_origin_xz")
        if _snap_origin is None:
            continue
        _snap_ox, _snap_oz = float(_snap_origin[0]), float(_snap_origin[1])
        _snap_cx = float(_snap_p["position_m"][0])
        _snap_cz = float(_snap_p["position_m"][2])
        _snap_dev = float(np.sqrt((_snap_cx - _snap_ox)**2 + (_snap_cz - _snap_oz)**2))
        if _snap_dev < 0.05:
            continue
        # Direction from current pos toward mask origin
        _snap_dir = np.array([_snap_ox - _snap_cx, _snap_oz - _snap_cz], dtype=np.float64)
        # Respect wall-affinity constraints
        _snap_wa = _snap_p.get("wall_affinity", "centre")
        if _snap_wa == "back":
            _snap_dir[1] = 0.0
        elif _snap_wa in ("left", "right"):
            _snap_dir[0] = 0.0
        _snap_len = float(np.linalg.norm(_snap_dir))
        if _snap_len < 1e-6:
            continue
        _snap_dir /= _snap_len
        # Binary search: find the furthest distance we can snap toward origin
        # without introducing a new collision.
        _snap_best = 0.0
        _snap_lo, _snap_hi = 0.0, _snap_len
        for _ in range(6):
            _snap_t = (_snap_lo + _snap_hi) / 2.0
            _snap_p["position_m"][0] = _snap_cx + _snap_dir[0] * _snap_t
            _snap_p["position_m"][2] = _snap_cz + _snap_dir[1] * _snap_t
            _snap_col = any(
                not _op.get("is_carpet") and not _op.get("wall_mounted")
                and _op is not _snap_p and _poly_collides(_snap_p, _op)
                for _op in _sr_floor
            )
            if _snap_col:
                _snap_hi = _snap_t
            else:
                _snap_best = _snap_t
                _snap_lo = _snap_t
        # Restore and apply best snap
        _snap_p["position_m"][0] = _snap_cx + _snap_dir[0] * _snap_best
        _snap_p["position_m"][2] = _snap_cz + _snap_dir[1] * _snap_best
        if _snap_best > 0.05:
            print(f"  [snap_back] idx={_snap_p.get('index')} ({_snap_p.get('type')}) "
                  f"snapped {_snap_best:.3f}m toward mask origin "
                  f"(was {_snap_dev:.3f}m away, now {_snap_dev - _snap_best:.3f}m away)")

    # ── Corner-escape nudge for back-wall objects ─────────────────────────────
    # If a back-wall object is within 25% of the room width from a side wall
    # (i.e. in a corner), nudge it toward the room center along X — up to 30cm
    # or until it would collide.  This handles cases where mask_align and VLM
    # place an object too close to the corner (e.g. plant in back-right corner).
    try:
        _ce_room_w, _ = _get_room_dims(
            Path(_sr_floor[0].get("glb_path", ".")) if _sr_floor else Path("."), cam)
    except Exception:
        _ce_room_w = 5.0
    _CORNER_ZONE = 0.25   # within 25% of room width from a side wall = "corner"
    _CE_MAX_NUDGE = 0.10  # max corner-escape nudge (small local fix only)
    for _ce_p in _sr_floor:
        if _ce_p.get("wall_affinity") != "back":
            continue
        _ce_x = float(_ce_p["position_m"][0])
        _ce_hx, _ = _col_xz_half_extents(_ce_p)
        # Distance from each side wall
        _dist_left  = _ce_x - _ce_hx          # distance from left wall (X=0)
        _dist_right = _ce_room_w - _ce_x - _ce_hx  # distance from right wall
        _corner_thresh = _ce_room_w * _CORNER_ZONE
        if _dist_left > _corner_thresh and _dist_right > _corner_thresh:
            continue  # not in a corner
        # Determine nudge direction: toward center
        _ce_dir = 1.0 if _dist_left < _dist_right else -1.0
        # Binary search: find max nudge without new collision
        _ce_best = 0.0
        _ce_lo, _ce_hi = 0.0, _CE_MAX_NUDGE
        _ce_orig_x = _ce_x
        for _ in range(6):
            _t = (_ce_lo + _ce_hi) / 2.0
            _ce_p["position_m"][0] = _ce_orig_x + _ce_dir * _t
            _ce_col = any(
                not _op.get("is_carpet") and not _op.get("wall_mounted")
                and _op is not _ce_p and _poly_collides(_ce_p, _op)
                for _op in _sr_floor
            )
            if _ce_col:
                _ce_hi = _t
            else:
                _ce_best = _t
                _ce_lo = _t
        _ce_p["position_m"][0] = _ce_orig_x + _ce_dir * _ce_best
        if _ce_best > 0.02:
            print(f"  [corner_escape] idx={_ce_p.get('index')} ({_ce_p.get('type')}) "
                  f"nudged {_ce_dir * _ce_best:+.3f}m away from corner "
                  f"(x: {_ce_orig_x:.3f} → {_ce_p['position_m'][0]:.3f})")

    # ── Back-direction push for seating (aisle access) ────────────────────────
    # Chairs placed facing a sofa can block the aisle in front of the sofa.
    # For each chair/seating object that has a sofa within 1.5m in its front
    # direction, try to push it toward its back direction by up to 0.30m so
    # the aisle between them stays accessible.
    _SEATING_TYPES = {"chair", "armchair", "stool"}
    _SOFA_TYPES    = {"sofa", "couch", "loveseat"}
    for _bk_p in _sr_floor:
        if _bk_p.get("type", "").lower() not in _SEATING_TYPES:
            continue
        _bk_R = np.array(_bk_p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
        _bk_lf = np.array(_bk_p.get("_local_front", [0.0, 0.0, 1.0]), dtype=np.float64)
        # World-space front and back directions (XZ only)
        _bk_front_w = (_bk_R @ _bk_lf)[[0, 2]]
        _bk_fn = float(np.linalg.norm(_bk_front_w))
        if _bk_fn < 1e-6:
            continue
        _bk_front_w /= _bk_fn
        _bk_back_w = -_bk_front_w   # direction to push chair toward
        _bk_cx, _bk_cz = float(_bk_p["position_m"][0]), float(_bk_p["position_m"][2])
        # Check for a sofa within 1.5m in the front direction
        _bk_sofa_nearby = any(
            _op.get("type", "").lower() in _SOFA_TYPES
            and abs(float(np.dot(
                np.array([float(_op["position_m"][0]) - _bk_cx,
                          float(_op["position_m"][2]) - _bk_cz]),
                _bk_front_w))) < 1.5
            and float(np.dot(
                np.array([float(_op["position_m"][0]) - _bk_cx,
                          float(_op["position_m"][2]) - _bk_cz]),
                _bk_front_w)) > 0   # sofa is in the front direction
            for _op in _sr_floor
        )
        if not _bk_sofa_nearby:
            continue
        # Binary search: push up to 0.30m toward back direction
        _bk_best = 0.0
        _bk_lo, _bk_hi = 0.0, 0.10   # aisle_clear: max 10 cm push (small local fix)
        for _ in range(6):
            _bk_t = (_bk_lo + _bk_hi) / 2.0
            _bk_p["position_m"][0] = _bk_cx + _bk_back_w[0] * _bk_t
            _bk_p["position_m"][2] = _bk_cz + _bk_back_w[1] * _bk_t
            _bk_col = any(
                not _op.get("is_carpet") and not _op.get("wall_mounted")
                and _op is not _bk_p and _poly_collides(_bk_p, _op)
                for _op in _sr_floor
            )
            if _bk_col:
                _bk_hi = _bk_t
            else:
                _bk_best = _bk_t
                _bk_lo = _bk_t
        _bk_p["position_m"][0] = _bk_cx + _bk_back_w[0] * _bk_best
        _bk_p["position_m"][2] = _bk_cz + _bk_back_w[1] * _bk_best
        if _bk_best > 0.02:
            print(f"  [aisle_clear] idx={_bk_p.get('index')} ({_bk_p.get('type')}) "
                  f"pushed {_bk_best:.3f}m toward back to preserve sofa aisle access")

    # ── Room-bounds clamp after scene_review + sr_collision ───────────────────
    # Both steps may push objects outside the room walls.  Re-clamp all
    # floor objects to stay within room bounds.
    for _p_clamp in placements:
        if _p_clamp.get("is_carpet") or _p_clamp.get("wall_mounted"):
            continue
        if "size_m" not in _p_clamp or "glb_path" not in _p_clamp:
            continue
        try:
            _cr_w, _cr_d = _get_room_dims(Path(_p_clamp["glb_path"]), cam)
            _cr_R = np.array(_p_clamp.get("rotation_3x3", np.eye(3).tolist()),
                             dtype=np.float64)
            _cr_lhx = _p_clamp["size_m"]["width_m"] / 2.0
            _cr_lhz = _p_clamp["size_m"]["depth_m"] / 2.0
            _cr_hx = abs(float(_cr_R[0][0])) * _cr_lhx + abs(float(_cr_R[0][2])) * _cr_lhz
            _cr_hz = abs(float(_cr_R[2][0])) * _cr_lhx + abs(float(_cr_R[2][2])) * _cr_lhz
            _p_clamp["position_m"][0] = float(np.clip(
                _p_clamp["position_m"][0], _cr_hx + 0.02, _cr_w - _cr_hx - 0.02))
            _p_clamp["position_m"][2] = float(np.clip(
                _p_clamp["position_m"][2], _cr_hz + 0.02, _cr_d - _cr_hz - 0.02))
        except Exception:
            pass

    # ── Final silhouette-anchor enforcement ────────────────────────────────────
    # Last-line-of-defense: after EVERY post-processing step (collision,
    # post_mask_clearance, scene_review, sr_collision, corner_escape,
    # aisle_clear, room clamp, facing logic, ...), make sure no floor
    # placement has drifted more than _FINAL_MAX_DEV m from its silhouette
    # anchor (_mask_origin_xz). Any piece beyond the cap is pulled back along
    # the line toward its anchor to land on the cap circle. Silhouette match
    # is the primary truth — a small overlap is better than an object on the
    # wrong side of the room.
    #
    # Set to a very large number (>= room diagonal) to effectively disable
    # the final anchor clamp — useful when running with aggressive VGGT
    # alignment via `--iter-max-dev <large>`.  Set to a smaller value to
    # tighten how close the final result must stay to mask_align.
    _FINAL_MAX_DEV = float(iter_max_dev_m)

    def _final_collides(_pp: dict) -> bool:
        """True if `_pp` overlaps any non-carpet/non-wall-mounted neighbour
        (uses SAT on hulls when available; AABB fallback)."""
        _R = np.array(_pp.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
        _sm = _pp.get("size_m", {})
        _lhx = float(_sm.get("width_m", 0.5)) / 2.0
        _lhz = float(_sm.get("depth_m", 0.5)) / 2.0
        _hx = abs(float(_R[0][0])) * _lhx + abs(float(_R[0][2])) * _lhz
        _hz = abs(float(_R[2][0])) * _lhx + abs(float(_R[2][2])) * _lhz
        for _q in placements:
            if _q is _pp or _q.get("is_carpet") or _q.get("wall_mounted"):
                continue
            _qR = np.array(_q.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
            _qsm = _q.get("size_m", {})
            _qlhx = float(_qsm.get("width_m", 0.5)) / 2.0
            _qlhz = float(_qsm.get("depth_m", 0.5)) / 2.0
            _qhx = abs(float(_qR[0][0])) * _qlhx + abs(float(_qR[0][2])) * _qlhz
            _qhz = abs(float(_qR[2][0])) * _qlhx + abs(float(_qR[2][2])) * _qlhz
            _ovx = (_hx + _qhx) - abs(_pp["position_m"][0] - _q["position_m"][0])
            _ovz = (_hz + _qhz) - abs(_pp["position_m"][2] - _q["position_m"][2])
            if _ovx > 0.02 and _ovz > 0.02:
                return True
        return False

    for _fp in placements:
        if _fp.get("is_carpet") or _fp.get("wall_mounted"):
            continue
        _origin = _fp.get("_mask_origin_xz")
        if _origin is None:
            continue
        _ox, _oz = float(_origin[0]), float(_origin[1])
        _cx = float(_fp["position_m"][0])
        _cz = float(_fp["position_m"][2])
        _dev = float(np.sqrt((_cx - _ox) ** 2 + (_cz - _oz) ** 2))
        if _dev <= _FINAL_MAX_DEV:
            continue
        # If collision-resolution moved this piece deliberately to clear a
        # neighbour, pulling it back to the anchor would recreate the overlap.
        # Try the full clamp first; if that creates a new collision, walk the
        # ratio outward (less clamp) until clear or exhausted, accepting a
        # larger drift in exchange for a non-colliding scene.
        _ratio_full = _FINAL_MAX_DEV / _dev
        _applied_ratio = None
        for _try_ratio in (_ratio_full, 0.65, 0.85, 1.0):
            # Skip ratios that would tighten the clamp further than _ratio_full
            if _try_ratio < _ratio_full:
                continue
            _try_x = _ox + (_cx - _ox) * _try_ratio
            _try_z = _oz + (_cz - _oz) * _try_ratio
            _orig_x, _orig_z = _fp["position_m"][0], _fp["position_m"][2]
            _fp["position_m"][0] = _try_x
            _fp["position_m"][2] = _try_z
            if not _final_collides(_fp):
                _applied_ratio = _try_ratio
                break
            # Restore for next iteration
            _fp["position_m"][0] = _orig_x
            _fp["position_m"][2] = _orig_z
        if _applied_ratio is None:
            # No non-colliding ratio in [_ratio_full, 1.0].  Behaviour split
            # by drift magnitude:
            #   - drift ≤ HARD_DRIFT_LIMIT: tolerate the over-drift to avoid
            #     recreating an overlap (current original behaviour).
            #   - drift > HARD_DRIFT_LIMIT: scene_review's shift went too
            #     far (e.g. chair shifted 40+cm in Z because the VLM
            #     misinterpreted a photo-edge crop as a 3D position).
            #     Force-clamp to the mask-origin + MAX_DEV vector even
            #     though it re-collides — a 0.15 m clamped drift with
            #     downstream collision-resolve is preferable to leaving
            #     the object 40 cm from where the silhouette says it
            #     belongs.  The animation also avoids the "giant leap"
            #     between the per-step frames and the final state.
            _HARD_DRIFT_LIMIT = 0.25
            # Only force-clamp into a re-collision when scene_review actually
            # moved this object (the original failure mode the hard-cap was
            # built for). If scene_review skipped it, the drift came from
            # init_slide / wall_clearance / sr_collision / aisle_clear — those
            # moves already respect collisions, so re-creating one is wrong.
            _sr_moved_this = bool(_fp.get("_scene_review_shifted"))
            if _dev > _HARD_DRIFT_LIMIT and _sr_moved_this:
                _force_x = _ox + (_cx - _ox) * _ratio_full
                _force_z = _oz + (_cz - _oz) * _ratio_full
                _fp["position_m"][0] = _force_x
                _fp["position_m"][2] = _force_z
                print(f"  [final_anchor] idx={_fp.get('index')} ({_fp.get('type')}) "
                      f"drift {_dev:.3f}m > {_HARD_DRIFT_LIMIT}m hard-cap — "
                      f"forced clamp to {_FINAL_MAX_DEV:.2f}m (accepting "
                      f"resulting collision, scene_review-moved): "
                      f"({_cx:.3f},{_cz:.3f}) → ({_force_x:.3f},{_force_z:.3f})")
            else:
                _why = ("clamp would re-collide"
                        if _dev <= _HARD_DRIFT_LIMIT
                        else "drift large but scene_review did not move this object")
                print(f"  [final_anchor] idx={_fp.get('index')} ({_fp.get('type')}) "
                      f"drift {_dev:.3f}m — {_why}; leaving at "
                      f"({_cx:.3f},{_cz:.3f})")
        else:
            _new_x = _fp["position_m"][0]
            _new_z = _fp["position_m"][2]
            _ratio_label = (f"{_FINAL_MAX_DEV:.2f}m" if _applied_ratio == _ratio_full
                            else f"{_dev * _applied_ratio:.2f}m (relaxed: full clamp would re-collide)")
            print(f"  [final_anchor] idx={_fp.get('index')} ({_fp.get('type')}) "
                  f"drift {_dev:.3f}m → clamped to {_ratio_label}: "
                  f"({_cx:.3f},{_cz:.3f}) → ({_new_x:.3f},{_new_z:.3f})")

    # ── Flat carpet/rug objects ─────────────────────────────────────���─────────
    # If place_carpet.py has already generated a GLB, use it as a world-space
    # pre-placed mesh (no further transform).  Otherwise fall back to flat-quad
    # rendering directly from the texture image.
    # ── Post-placement fine rotation ─────────────────────────────────────────
    # After all individual placements + scene_review + collision passes are
    # done, do one more small-rotation pass per object using the FULL
    # composited scene as context.  This catches yaw misalignments that
    # only become apparent once everything is in place (e.g. a chair that
    # looked correct in isolation now visibly faces the wrong way relative
    # to its sofa pair).  Caps from --vlm-fine-rotation-max-deg apply.
    try:
        # Same reference-photo discovery as the per-object fine_rot:
        # vggt/camera.json["image"] holds the path to the original photo.
        _ref_path = None
        try:
            _vc = json.loads((out_root / "vggt" / "camera.json").read_text())
            _img_field = (_vc[0] if isinstance(_vc, list) else _vc).get("image", "")
            if _img_field:
                _candidate = Path(_img_field)
                if not _candidate.is_absolute():
                    _candidate = Path(__file__).resolve().parents[2] / _candidate
                if _candidate.exists():
                    _ref_path = _candidate
        except Exception:
            _ref_path = None
        if _ref_path is not None:
            _post_preview = out_root / "furniture" / "_post_fine_rot_scene.png"
            _render_full_scene_preview(placements, cam, out_root, _post_preview)
            if _post_preview.exists():
                print(f"\n[post_fine_rot] starting full-scene yaw refinement pass ...")
                _img_w = int(cam["width_px"])
                _img_h = int(cam["height_px"])
                _post_max_deg = int(vlm_fine_rotation_max_deg)
                # Wall-anchored furniture must stay at its cardinal base yaw —
                # a continuous VLM tilt here produces off-axis yaws (-127°, -142°)
                # and breaks wall-flush.  Only tilt centre-of-room objects.
                _post_targets = [p for p in placements
                                 if not p.get("is_carpet")
                                 and not p.get("wall_mounted")
                                 and not p.get("on_top_of")
                                 and p.get("wall_affinity", "centre") in (None, "centre")
                                 and p.get("box_px")]
                for _pp in _post_targets:
                    _pidx = _pp.get("index", -1)
                    _pty  = _pp.get("type", "object")
                    _pbb  = tuple(_pp.get("box_px", ()))
                    if len(_pbb) != 4:
                        continue
                    _pwa  = _pp.get("wall_affinity", "centre")
                    _post_thresh = 5.0 if _pwa in ("back", "left", "right") else 1.0
                    _psize = _pp.get("size_m") or {}
                    try:
                        _pp_pos_xz = (float(_pp["position_m"][0]),
                                       float(_pp["position_m"][2]))
                        try:
                            _pp_room_w, _pp_room_d = _get_room_dims(
                                Path(_pp.get("glb_path", ".")), cam)
                        except Exception:
                            _pp_room_w, _pp_room_d = None, None
                        _pres = _vlm_fine_rotation_correction(
                            preview_path=_post_preview, reference_path=_ref_path,
                            obj_type=_pty, bbox=_pbb,
                            img_w=_img_w, img_h=_img_h, max_deg=_post_max_deg,
                            wall_affinity=_pwa,
                            obj_width_m=float(_psize.get("width_m", 0)) or None,
                            obj_depth_m=float(_psize.get("depth_m", 0)) or None,
                            obj_world_pos=_pp_pos_xz,
                            room_w=_pp_room_w, room_d=_pp_room_d,
                        )
                    except Exception as _e_pp:
                        print(f"  [post_fine_rot] idx={_pidx} VLM call failed: {_e_pp}")
                        continue
                    _pdelta = float(_pres.get("delta_deg", 0))
                    _pnotes = _pres.get("notes", "")
                    if abs(_pdelta) < _post_thresh:
                        print(f"  [post_fine_rot] idx={_pidx} ({_pty}) "
                              f"Δyaw={_pdelta:+.1f}° < thresh {_post_thresh:.0f}° "
                              f"(wa={_pwa}, no change)")
                        continue
                    _apply_y_rotation_deg(_pp, _pdelta)
                    print(f"  [post_fine_rot] idx={_pidx} ({_pty}) "
                          f"applied Δyaw={_pdelta:+.1f}° (wa={_pwa})  "
                          f"notes={_pnotes[:80]}")
                    _pp["_post_fine_rotation_deg"] = (
                        _pp.get("_post_fine_rotation_deg", 0.0) + _pdelta
                    )
                _post_preview.unlink(missing_ok=True)
        else:
            print("[post_fine_rot] no reference image found — skipping")
    except Exception as _e_postfr:
        print(f"[post_fine_rot] failed: {_e_postfr}")

    # ── Final wall-affinity cardinal snap (hard guarantee) ──────────────────
    # The VLM fine_rot / post_fine_rot passes apply continuous ±deg tilts on top
    # of the cardinal base with no re-snap, which produced off-axis yaws (e.g.
    # -127°, -142°, -155°) on wall-anchored furniture and broke wall-flush (the
    # flush loop below derives distance from the rotation).  Snap every
    # wall-affinity object back to a clean 0/90/180/270° before flushing/saving.
    for _p_snap in placements:
        if _p_snap.get("is_carpet") or _p_snap.get("wall_mounted"):
            continue
        if _p_snap.get("wall_affinity") not in ("back", "front", "left", "right"):
            continue
        try:
            _R_snap = np.asarray(_p_snap.get("rotation_3x3"), dtype=float)
            if _R_snap.shape != (3, 3):
                continue
            _yaw_snap = np.arctan2(float(_R_snap[0, 2]), float(_R_snap[0, 0]))
            _yaw_card = round(_yaw_snap / (np.pi / 2)) * (np.pi / 2)
            if abs(_yaw_snap - _yaw_card) > 1e-4:
                _p_snap["rotation_3x3"] = _rotation_y(_yaw_card).tolist()
                print(f"  [wall_cardinal_snap] idx={_p_snap.get('index')} "
                      f"({_p_snap.get('type')}) yaw {np.degrees(_yaw_snap):.1f}° → "
                      f"{np.degrees(_yaw_card):.1f}° (wa={_p_snap.get('wall_affinity')})")
        except Exception:
            continue

    furn_dir = out_root / "furniture"
    _inpainted_subdirs = ["inpainted", "inpainted5", "inpainted2", "inpainted0"]
    for seg in seg_by_idx.values():
        if seg.get("type") not in _FLAT_TYPES:
            continue
        idx = seg["index"]
        if seg.get("glb_file"):
            # World-space carpet GLB — create a passthrough placement (no transforms)
            glb_rel  = seg["glb_file"]
            glb_path = Path(glb_rel) if Path(glb_rel).is_absolute() else furn_dir / glb_rel
            if not glb_path.exists():
                print(f"\n[place_furn] idx={idx}  carpet GLB not found: {glb_path} — falling through")
            else:
                print(f"\n[place_furn] idx={idx}  carpet (world-space GLB): {glb_path.name}")
                import trimesh as _trimesh
                mesh = _trimesh.load(str(glb_path), force="mesh")
                if isinstance(mesh, _trimesh.Scene):
                    mesh = _trimesh.util.concatenate(mesh.dump())
                verts_c = mesh.vertices
                cx_w = float(verts_c[:, 0].mean())
                cz_w = float(verts_c[:, 2].mean())
                cp = {
                    "index":               idx,
                    "type":                seg["type"],
                    "glb_path":            str(glb_path),
                    "position_m":          [cx_w, 0.0, cz_w],
                    "rotation_3x3":        np.eye(3).tolist(),
                    "scale":               [1.0, 1.0, 1.0],
                    "skip_ground_removal": True,
                    "skip_floor_transform": True,
                }
                placements.insert(0, cp)
                print(f"  [carpet GLB] centre=({cx_w:.2f}, {cz_w:.2f})")
                if _anim_frame_render is not None:
                    _anim_frame_render(placements, label=f"idx{idx}_carpet",
                                       highlight_idx=idx)
                continue  # skip flat-quad fallback

        print(f"\n[place_furn] idx={idx}  carpet (flat quad)")
        # Find texture image: prefer inpainted, fall back to crop
        carpet_img: Path | None = None
        if seg.get("inpaint_file"):
            for subdir in _inpainted_subdirs:
                candidate = furn_dir / subdir / seg["inpaint_file"]
                if candidate.exists():
                    carpet_img = candidate
                    break
        if carpet_img is None and seg.get("crop_file"):
            candidate = furn_dir / "segmented" / seg["crop_file"]
            if candidate.exists():
                carpet_img = candidate
        if carpet_img is None:
            print(f"  [carpet] no image found — skipping")
            continue
        print(f"  [carpet] texture: {carpet_img.name}")
        cp = compute_carpet_placement(seg, cam, carpet_img, output_dir=out_root)
        if cp is not None:
            # Mask-iterative alignment: project quad to 2D, compare with
            # segmentation mask, adjust position + per-axis scale to match.
            cp = _carpet_mask_align(cp, cam)
            # VLM texture orientation check: ensure stripes/pattern match reference
            _orient_preview = _render_carpet_preview(cp, cam, out_root)
            if use_vlm and _orient_preview is not None:
                # The reference must be the SOURCE PHOTO — it is the only image
                # that actually contains the rug.  The renders below are of the
                # reconstruction (empty room + wall objects), so the VLM was
                # being asked to compare the carpet against a picture with no
                # carpet in it and answered, correctly, "No rug is visible in
                # Image 1 ... defaulting both to 0".  This check could never
                # fire until the photo was offered first.
                _orient_ref = next(
                    (rp for rp in [
                        (Path(base_image_path) if base_image_path else None),
                        out_root / f"{out_root.name}.png",
                        out_root / "wall_mounted" / "placements" / "render_objects_placed.png",
                        out_root / "render_final.png",
                        out_root / "render.png",
                    ] if rp is not None and rp.exists()), None)
                if _orient_ref is not None:
                    rot_corr, foot_corr = _vlm_carpet_orientation_check(
                        _orient_preview, _orient_ref)
                    if rot_corr != 0:
                        # Store texture rotation on the placement dict — applied
                        # when the texture is loaded, NOT by rotating the geometry.
                        # Rotating corners would swap width/depth for non-square carpets.
                        cp["_tex_rotation_deg"] = rot_corr
                        print(f"  [carpet] texture orientation corrected by {rot_corr}°")
                    if foot_corr == 90:
                        # The RECTANGLE itself is a quarter-turn out: rebuild the
                        # quad with its in-plane axes swapped about the centre.
                        # The texture check above cannot fix this — it only turns
                        # the image on the quad, leaving a long rug sitting on a
                        # short footprint.  Skipped when the quad is near-square,
                        # where a swap is meaningless and only adds jitter.
                        try:
                            _wc = np.asarray(cp["world_corners"], dtype=np.float64)
                            _c  = _wc.mean(axis=0)
                            _e1 = (_wc[1] - _wc[0])      # BL→BR  (width edge)
                            _e2 = (_wc[3] - _wc[0])      # BL→TL  (depth edge)
                            _l1, _l2 = float(np.linalg.norm(_e1)), float(np.linalg.norm(_e2))
                            _ratio = max(_l1, _l2) / max(min(_l1, _l2), 1e-6)
                            if _ratio < 1.10:
                                print(f"  [carpet] footprint swap skipped — quad is "
                                      f"near-square ({_l1:.2f}×{_l2:.2f} m, "
                                      f"ratio {_ratio:.2f}); a 90° swap is a no-op")
                            else:
                                _u1 = _e1 / max(_l1, 1e-9)
                                _u2 = _e2 / max(_l2, 1e-9)
                                # swap the EXTENTS, keep the axes
                                _h1, _h2 = _u1 * (_l2 / 2.0), _u2 * (_l1 / 2.0)
                                cp["world_corners"] = [
                                    (_c - _h1 - _h2).tolist(),   # BL
                                    (_c + _h1 - _h2).tolist(),   # BR
                                    (_c + _h1 + _h2).tolist(),   # TR
                                    (_c - _h1 + _h2).tolist(),   # TL
                                ]
                                print(f"  [carpet] footprint rotated 90° — "
                                      f"{_l1:.2f}×{_l2:.2f} → {_l2:.2f}×{_l1:.2f} m")
                        except Exception as _fe:
                            print(f"  [carpet] footprint swap failed: {_fe}")
                if _orient_preview.exists():
                    _orient_preview.unlink()
            placements.insert(0, cp)   # draw carpet first (floor level, painter's sort handles depth)
            if _anim_frame_render is not None:
                _anim_frame_render(placements, label=f"idx{idx}_carpet")

    if not placements:
        print("[place_furn] No objects placed successfully.")
        return out_root / "furniture"

    # ── Per-segment manual position nudge ─────────────────────────────────────
    # `position_offset_xz: [dx, dz]` (metres) on the segment_results entry is
    # applied AFTER all algorithmic adjustments (mask_align, VGGT, collisions,
    # scene_review).  Use this for one-off corrections like "this plant looks
    # 0.4m too right; nudge it left" without re-tuning the whole pipeline.
    # Mirrors the existing `yaw_offset_deg` mechanism.
    for p in placements:
        idx_p = p.get("index", -1)
        seg_p = seg_by_idx.get(idx_p, {})
        offs = (p.get("position_offset_xz")
                or seg_p.get("position_offset_xz"))
        if not offs or len(offs) < 2:
            continue
        try:
            dx, dz = float(offs[0]), float(offs[1])
        except (TypeError, ValueError):
            continue
        if abs(dx) < 1e-4 and abs(dz) < 1e-4:
            continue
        old_x = float(p["position_m"][0])
        old_z = float(p["position_m"][2])
        p["position_m"][0] = old_x + dx
        p["position_m"][2] = old_z + dz
        # Carpets store world_corners; shift them to keep the texture in sync.
        if p.get("is_carpet") and p.get("world_corners"):
            p["world_corners"] = [[c[0] + dx, c[1], c[2] + dz]
                                  for c in p["world_corners"]]
        print(f"  [position_offset] idx={idx_p} ({p.get('type')}): "
              f"({old_x:.3f},{old_z:.3f}) → ({old_x+dx:.3f},{old_z+dz:.3f}) "
              f"[dx={dx:+.3f} dz={dz:+.3f}]")

    # ── Post-placement layout resolve ────────────────────────────────────────
    # Two passes to make the saved positions physically consistent with the
    # room mesh:
    #   1. CLIP — every non-carpet footprint must lie inside the walls.obj
    #      AABB.  A back-projected position past the room boundary (very
    #      common for small bboxes near walls — see idx-8 cabinet at x=4.92
    #      with room_w=4.20) is shifted (NOT rescaled) so the AABB just fits.
    #      Without this the JSON `position_m` and the rendered mesh disagree
    #      (renderer's wall_snap pulls verts inside, JSON keeps the OOB value).
    #   2. PUSH — pairwise AABB collisions are resolved greedily in
    #      descending area order: the larger object stays put, the smaller
    #      object is pushed along the shorter overlap axis until the contact
    #      is broken (then re-clipped into the room).
    try:
        import trimesh as _trm
        _walls_obj = out_root / "walls.obj"
        print(f"\n[layout] starting clip/push resolve  walls={_walls_obj}  "
              f"exists={_walls_obj.exists()}  n_placements={len(placements)}")
        if _walls_obj.exists():
            _wm = _trm.load(str(_walls_obj), force="mesh", process=False)
            _b = _wm.bounds
            _RX0, _RX1 = float(_b[0][0]), float(_b[1][0])
            _RZ0, _RZ1 = float(_b[0][2]), float(_b[1][2])
            print(f"[layout] room bounds X=[{_RX0:.3f},{_RX1:.3f}]  Z=[{_RZ0:.3f},{_RZ1:.3f}]")

            def _foot(p):
                if "position_m" not in p:
                    return None
                sz = p.get("size_m") or {}
                w = float(sz.get("width_m", 0)) if isinstance(sz, dict) else 0.0
                d = float(sz.get("depth_m",  0)) if isinstance(sz, dict) else 0.0
                if w <= 0 or d <= 0:
                    return None
                cx = float(p["position_m"][0])
                cz = float(p["position_m"][2])
                # width_m/depth_m are semantic (along-front vs perpendicular),
                # not raw local-X/local-Z — for an object whose front was
                # relabelled onto X ([axis_swap]), undo that before combining
                # with R, which rotates the object's RAW local axes.
                _front = p.get("_local_front", [0, 0, -1])
                if abs(float(_front[0])) > abs(float(_front[2])):
                    _lx, _lz = d, w
                else:
                    _lx, _lz = w, d
                # Rotation-aware AABB: (_lx,_lz) is in the object's raw LOCAL
                # frame, so a rotated piece (e.g. a yaw=-90 sofa whose local-X
                # runs along world Z) has swapped/blended world extents.
                # Without this the clip/push used the wrong axis and let
                # rotated objects clip through walls.
                _R = p.get("rotation_3x3")
                if _R is not None:
                    c = abs(float(_R[0][0])); s = abs(float(_R[0][2]))
                    w_eff = c * _lx + s * _lz      # world X extent
                    d_eff = s * _lx + c * _lz      # world Z extent
                    return cx, cz, w_eff, d_eff
                return cx, cz, _lx, _lz

            def _clip(p):
                f = _foot(p)
                if f is None:
                    return False
                cx, cz, w, d = f
                hw, hd = w / 2, d / 2
                ncx = min(max(cx, _RX0 + hw), _RX1 - hw)
                ncz = min(max(cz, _RZ0 + hd), _RZ1 - hd)
                if abs(ncx - cx) > 1e-3 or abs(ncz - cz) > 1e-3:
                    p["position_m"][0] = ncx
                    p["position_m"][2] = ncz
                    print(f"  [layout/clip] idx={p.get('index')} "
                          f"({p.get('type')}): ({cx:.2f},{cz:.2f}) → "
                          f"({ncx:.2f},{ncz:.2f})  room=[{_RX0:.2f},{_RX1:.2f}]×"
                          f"[{_RZ0:.2f},{_RZ1:.2f}]")
                    return True
                return False

            def _aabb(p):
                f = _foot(p)
                if f is None:
                    return None
                cx, cz, w, d = f
                return cx - w/2, cz - d/2, cx + w/2, cz + d/2

            # ── Concave footprint (actual mesh geometry, not bbox) ──────────────
            # The AABB above spans an L-sofa's empty chaise corner, causing FALSE
            # collisions with objects that legitimately sit inside the L.  Build a
            # per-object world-XZ occupancy set from the mesh vertices (the L's
            # concavity has no vertices → empty), and use it to VETO AABB overlaps
            # that aren't real surface overlaps.  Vertices are dense on furniture
            # meshes, so their XZ projection fills the true footprint.
            _GEOM_CELL = 0.06   # metres per occupancy cell
            def _world_xz_cells(p):
                v = p.get("_verts")
                if v is None:
                    return None
                v = np.asarray(v, dtype=np.float64)
                if v.ndim != 2 or v.shape[0] < 8:
                    return None
                cx0 = (v[:, 0].min() + v[:, 0].max()) / 2.0
                cz0 = (v[:, 2].min() + v[:, 2].max()) / 2.0
                vl = v.copy(); vl[:, 0] -= cx0; vl[:, 2] -= cz0
                sc = np.asarray(p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
                if sc.ndim == 0:
                    sc = np.array([float(sc)] * 3)
                R = np.asarray(p.get("rotation_3x3", np.eye(3).tolist()), dtype=np.float64)
                pos = np.asarray(p["position_m"], dtype=np.float64)
                w = (vl * sc) @ R.T
                xz = w[:, [0, 2]] + np.array([pos[0], pos[2]])
                ij = np.floor(xz / _GEOM_CELL).astype(np.int64)
                return set(map(tuple, np.unique(ij, axis=0).tolist()))

            def _geom_collide(p, q):
                """True if the actual mesh footprints overlap (concave).  Falls
                back to True (trust the AABB) if vertices are unavailable."""
                ca = _world_xz_cells(p)
                cb = _world_xz_cells(q)
                if ca is None or cb is None:
                    return True
                cbd = set()
                for (i, j) in cb:                      # dilate q by 1 cell (~6 cm)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            cbd.add((i + di, j + dj))
                return not ca.isdisjoint(cbd)

            # Pass 1: clip every non-carpet placement.
            for _p in placements:
                if _p.get("is_carpet"):
                    continue
                _clip(_p)

            # Pass 2: collision push.  Anchor pieces CLOSEST to the deepest
            # corner (the room corner farthest from the camera) first; a colliding
            # piece FARTHER from that corner is pushed OUTWARD, away from the
            # corner — so a cornered object (e.g. a plant in the back-right corner)
            # is never shoved through a wall, and the neighbouring sofa/chair moves
            # into open space instead.
            _candidates = [_p for _p in placements
                           if not _p.get("is_carpet") and _foot(_p) is not None]
            _cam_xz = np.array([float(cam["position_m"][0]), float(cam["position_m"][2])])
            _corners = [(_RX0, _RZ0), (_RX1, _RZ0), (_RX0, _RZ1), (_RX1, _RZ1)]
            _deep = max(_corners, key=lambda c: (c[0] - _cam_xz[0])**2 + (c[1] - _cam_xz[1])**2)
            _dcx, _dcz = float(_deep[0]), float(_deep[1])
            print(f"[layout] deepest corner (farthest from camera) = ({_dcx:.2f},{_dcz:.2f})")

            # Anchor by DEPTH toward the deep corner's back wall: the piece
            # nearest the back wall (most cornered) is fixed first; a colliding
            # piece farther forward is pushed outward.  Using depth (distance to
            # the deep wall) rather than full corner-distance avoids the camera
            # offset flipping which object anchors (a plant cornered back-right
            # must stay put while the neighbouring chair moves forward).
            def _deep_depth(_q):
                f = _foot(_q)
                if not f:
                    return 1e9
                # distance from the deep corner's back-wall plane (z if deep
                # corner is at z≈RZ0, else from RZ1; analogous for x when the
                # collision is along x — here z dominates for back-wall scenes)
                _z_depth = abs(f[1] - _dcz)
                return _z_depth
            # Anchor LARGE wall-flush pieces (sofa/bed/cabinet/bookcase) FIRST —
            # their silhouette-aligned (mask_align) position is authoritative, so a
            # small neighbour (lamp, side table) must move AROUND them instead of
            # the big piece being shoved into a corner.  Within each tier, the
            # piece nearest the deep wall is anchored first (so a truly cornered
            # small object still isn't pushed through a wall).
            _BIG_ANCHOR = {"sofa", "couch", "sectional", "sectional_sofa", "loveseat",
                           "corner_sofa", "l_sofa", "bed", "cabinet", "bookcase",
                           "dresser", "wardrobe", "sideboard", "tv_stand"}

            def _anchor_tier(_q):
                t = _q.get("type", "").lower().replace("-", "_")
                big = (t in _BIG_ANCHOR
                       and _q.get("wall_affinity") in ("back", "left", "right"))
                return (0 if big else 1, _deep_depth(_q))
            _candidates.sort(key=_anchor_tier)   # big wall-flush anchored first, then by depth
            _fixed: list[dict] = []
            for _p in _candidates:
                a = _aabb(_p)
                if a is None:
                    _fixed.append(_p)
                    continue
                ax0, az0, ax1, az1 = a
                for _q in _fixed:
                    b = _aabb(_q)
                    if b is None:
                        continue
                    bx0, bz0, bx1, bz1 = b
                    ox = min(ax1, bx1) - max(ax0, bx0)
                    oz = min(az1, bz1) - max(az0, bz0)
                    if ox <= 1e-3 or oz <= 1e-3:
                        continue
                    # Concave gate: the AABBs overlap, but if the ACTUAL mesh
                    # footprints don't (e.g. this object sits inside an L-sofa's
                    # open corner), it's a false collision — don't push.
                    if not _geom_collide(_p, _q):
                        print(f"  [layout/push] idx={_p.get('index')} ({_p.get('type')}) "
                              f"↔ idx={_q.get('index')} ({_q.get('type')}): AABB overlap "
                              f"but mesh footprints clear — false collision, not pushed")
                        continue
                    cx, cz = float(_p["position_m"][0]), float(_p["position_m"][2])
                    w  = float(_p["size_m"]["width_m"])
                    d  = float(_p["size_m"]["depth_m"])
                    bcx, bcz = (bx0 + bx1) / 2, (bz0 + bz1) / 2
                    # A central coffee table belongs AT the sofa's front edge — its
                    # silhouette (mask) position legitimately abuts/overlaps the sofa
                    # footprint, and for an L-sectional the rectangular AABB spans the
                    # empty chaise corner, so this overlap is usually a FALSE collision.
                    # Don't shove the table off its silhouette to "clear" the sofa.
                    _SOFA_LIKE = {"sofa", "couch", "loveseat", "sectional",
                                  "sectional_sofa", "corner_sofa", "l_sofa"}
                    if (_p.get("type", "").lower().replace("-", "_")
                            in {"coffee_table", "cocktail_table", "round_table"}
                            and _q.get("type", "").lower().replace("-", "_") in _SOFA_LIKE):
                        print(f"  [layout/push] idx={_p.get('index')} (coffee_table) kept "
                              f"at silhouette/sofa-front — sofa-overlap exempt")
                        continue
                    # A dining/meeting table is intentionally SURROUNDED by its
                    # chairs (they overlap its rectangular AABB at the rim — a round
                    # table's AABB especially pokes into the corners where chairs
                    # sit).  The chair-group rule arranges them, so never shove the
                    # table (or a chair) to "clear" each other.
                    _tp = _p.get("type", "").lower().replace("-", " ").replace("_", " ")
                    _tq = _q.get("type", "").lower().replace("-", " ").replace("_", " ")

                    def _is_dtable(t):
                        return ("table" in t
                                and any(k in t for k in ("dining", "meeting", "round"))
                                and not any(k in t for k in ("coffee", "side", "end",
                                                             "cocktail", "console", "nightstand")))
                    if (("chair" in _tp and _is_dtable(_tq))
                            or (_is_dtable(_tp) and "chair" in _tq)):
                        print(f"  [layout/push] idx={_p.get('index')} ({_p.get('type')}) "
                              f"↔ idx={_q.get('index')} ({_q.get('type')}): dining "
                              f"table+chair group — exempt (chair-group arranges)")
                        continue
                    # Silhouette is the authority: if the pushed piece has a mask
                    # anchor, clear the overlap on whichever side keeps it CLOSEST
                    # to where its segmentation mask placed it — never shove it to
                    # the blocker's far edge across the room. Without this, a deep
                    # corner on one side flings a left-side chair past a wide coffee
                    # table to the far right (004160). Falls back to the deep-corner
                    # heuristic when no mask anchor is available.
                    _origin = _p.get("_mask_origin_xz")
                    if ox <= oz:
                        _to_left  = bx0 - w / 2 - 0.01   # just left of the blocker
                        _to_right = bx1 + w / 2 + 0.01   # just right of the blocker
                        if _origin is not None:
                            ncx = (_to_left
                                   if abs(_to_left - _origin[0]) <= abs(_to_right - _origin[0])
                                   else _to_right)
                        elif _dcx >= (_RX0 + _RX1) / 2:
                            ncx = _to_left    # corner on the right → push left
                        else:
                            ncx = _to_right   # corner on the left → push right
                        ncz = cz
                    else:
                        ncx = cx
                        _to_back  = bz1 + d / 2 + 0.01   # +z side of the blocker
                        _to_front = bz0 - d / 2 - 0.01   # -z side of the blocker
                        if _origin is not None:
                            ncz = (_to_back
                                   if abs(_to_back - _origin[1]) <= abs(_to_front - _origin[1])
                                   else _to_front)
                        elif _dcz <= (_RZ0 + _RZ1) / 2:
                            ncz = _to_back    # corner at back → push forward
                        else:
                            ncz = _to_front   # corner at front → push back
                    _p["position_m"][0] = ncx
                    _p["position_m"][2] = ncz
                    print(f"  [layout/push] idx={_p.get('index')} "
                          f"({_p.get('type')}) ↔ idx={_q.get('index')} "
                          f"({_q.get('type')}): "
                          f"({cx:.2f},{cz:.2f}) → ({ncx:.2f},{ncz:.2f})  "
                          f"axis={'x' if ox<=oz else 'z'}  overlap=({ox*100:.0f},{oz*100:.0f}) cm")
                    # A large overlap is almost never a genuine layout clash — it
                    # means one of the two footprints is mis-sized, and translating
                    # it away just hides the real error somewhere else. office8:
                    # the desk came out 1.46 m deep (should be ~0.70), overlapped
                    # the office chair by 68 cm, and got shoved 0.76 m off-centre
                    # instead of being re-measured. Flag it loudly and record it.
                    #
                    # Use min(), not max(): the PENETRATION depth is the smaller
                    # of the two axis overlaps (the push separates along that
                    # axis).  The larger one is routinely legitimate — a chair
                    # tucked in front of a desk shares the desk's whole X span,
                    # so max() reported a bogus "64 cm overlap" for a pair whose
                    # real interpenetration was 18 cm.
                    _big = min(ox, oz)
                    if _big > 0.50:
                        print(f"  [layout/push] ⚠ SIZE-SUSPECT: {_big * 100:.0f} cm overlap "
                              f"between idx={_p.get('index')} ({_p.get('type')} "
                              f"{_p['size_m']['width_m']:.2f}×{_p['size_m']['depth_m']:.2f}m) and "
                              f"idx={_q.get('index')} ({_q.get('type')} "
                              f"{_q['size_m']['width_m']:.2f}×{_q['size_m']['depth_m']:.2f}m) — "
                              f"a push cannot fix a bad size estimate; re-check these dimensions")
                        _p.setdefault("_size_suspect", []).append({
                            "against_index": _q.get("index"),
                            "against_type": _q.get("type"),
                            "overlap_x_m": round(float(ox), 3),
                            "overlap_z_m": round(float(oz), 3),
                            "pushed_from": [round(cx, 3), round(cz, 3)],
                            "pushed_to": [round(ncx, 3), round(ncz, 3)],
                        })
                    _clip(_p)  # re-clip in case push shoved it past a wall
                    a = _aabb(_p)
                    if a is None:
                        break
                    ax0, az0, ax1, az1 = a
                _fixed.append(_p)
            print(f"[layout] resolve done  ({len(_candidates)} objects processed)")

            # ── Depth-ordering pass ──────────────────────────────────────
            # The placement_analysis `depth` tag ("close"/"mid"/"far") was
            # written by the VLM and then read by NOTHING — like `group`, it
            # was write-only.  office8's reading nook shows the cost: the
            # side_table is tagged "close" and the lounge chair "mid", yet
            # they came out at z=1.92 and z=2.10 — the chair IN FRONT of the
            # table, inverted against the VLM's own reading of the photo.
            # Each object is positioned independently from its own mask, so
            # nothing ever compares two objects' relative depth.
            #
            # Camera sits at large +Z, so CLOSER == LARGER z.  Only ordering
            # within one analysis `group` is meaningful (a reading nook vs a
            # workspace across the room are not depth-comparable), and only a
            # strict rank difference is acted on.
            try:
                _DRANK = {"far": 0, "mid": 1, "close": 2}
                _DMARGIN = float(os.environ.get("SCENEWEAVE_DEPTH_MARGIN", "0.12"))
                _by_idx_e = {e.get("index"): e for e in order}
                _groups: dict[str, list] = {}
                for _p in placements:
                    if _p.get("is_carpet") or _p.get("wall_mounted"):
                        continue
                    _e = _by_idx_e.get(_p.get("index")) or {}
                    _g = _e.get("group")
                    _r = _DRANK.get(str(_e.get("depth") or "").lower())
                    if _g and _r is not None:
                        _groups.setdefault(_g, []).append((_r, _p))
                print(f"  [depth_order] scanning {len(placements)} placement(s) → "
                      f"{ {g: [(p.get('index'), round(float(p['position_m'][2]), 2)) for _, p in m] for g, m in _groups.items()} }")
                for _g, _members in _groups.items():
                    if len(_members) < 2:
                        continue
                    # ascending rank == ascending z (far behind, close in front)
                    _members.sort(key=lambda t: t[0])
                    for _a in range(len(_members)):
                        for _b in range(_a + 1, len(_members)):
                            _ra, _pa = _members[_a]
                            _rb, _pb = _members[_b]
                            if _ra == _rb:
                                continue
                            _za = float(_pa["position_m"][2])
                            _zb = float(_pb["position_m"][2])
                            _need = (_zb - _za) - _DMARGIN     # want zb > za
                            if _need >= 0:
                                continue
                            _shift = -_need / 2.0
                            _pa["position_m"][2] = _za - _shift
                            _pb["position_m"][2] = _zb + _shift
                            print(f"  [depth_order] {_g!r}: "
                                  f"idx={_pa.get('index')} "
                                  f"({_pa.get('type')},{['far','mid','close'][_ra]}) "
                                  f"z {_za:.2f}→{_pa['position_m'][2]:.2f}  |  "
                                  f"idx={_pb.get('index')} "
                                  f"({_pb.get('type')},{['far','mid','close'][_rb]}) "
                                  f"z {_zb:.2f}→{_pb['position_m'][2]:.2f}")
                            _clip(_pa); _clip(_pb)
            except Exception as _de:
                print(f"  [depth_order] skipped: {_de}")
        else:
            print("[layout] walls.obj missing — skipping clip/collision resolve")
    except Exception as _le:
        import traceback as _tb
        print(f"[layout] resolve skipped: {_le}")
        _tb.print_exc()

    # Deterministic wall-snap: the analysis sometimes labels a sofa/cabinet that
    # actually backs onto a wall as "centre", leaving it un-aligned (tilted at a
    # diagonal front-detect yaw instead of flush with its back to the wall).  If
    # such a piece's FOOTPRINT sits within a small gap of a wall, set its back
    # against that wall so the flush + orientation logic below aligns it.  Pieces
    # genuinely mid-room stay "centre" (their gap exceeds the threshold).  Chairs/
    # armchairs (facing-group handling) and beds (silhouette yaw-pick) are excluded
    # so this doesn't fight those; central-by-design types are never in the set.
    _WALL_BACK_TYPES = {"sofa", "couch", "sectional", "loveseat", "cabinet",
                        "dresser", "console", "console_table", "media_console",
                        "bookcase", "tv_stand", "sideboard", "desk"}
    _SNAP_GAP_M = 0.45
    try:
        _snap_rw = _snap_rd = None
        for _p in placements:
            _t = _p.get("type", "").lower().replace("-", "_")
            if _t not in _WALL_BACK_TYPES:
                continue
            if _p.get("wall_affinity") in ("back", "left", "right"):
                continue
            if _p.get("_facing_applied") or not _p.get("position_m") or not _p.get("size_m"):
                continue
            if _snap_rw is None:
                try:
                    _snap_rw, _snap_rd = _get_room_dims(Path(_p.get("glb_path", "")), cam)
                except Exception:
                    break
            _Rs = np.array(_p.get("rotation_3x3", np.eye(3).tolist()), float)
            _sm = _p["size_m"]; _w = float(_sm.get("width_m", 0.5)); _d = float(_sm.get("depth_m", 0.5))
            _hx = abs(_Rs[0, 0]) * _w / 2 + abs(_Rs[0, 2]) * _d / 2
            _hz = abs(_Rs[2, 0]) * _w / 2 + abs(_Rs[2, 2]) * _d / 2
            _xc, _zc = float(_p["position_m"][0]), float(_p["position_m"][2])
            # distance from the footprint edge to each wall (skip the front/camera wall)
            _dists = {"left": _xc - _hx, "right": _snap_rw - (_xc + _hx), "back": _zc - _hz}
            _wall, _gap = min(_dists.items(), key=lambda kv: kv[1])
            if _gap < _SNAP_GAP_M:
                _p["wall_affinity"] = _wall
                print(f"  [wall_snap] idx={_p.get('index')} {_t}: 'centre' → {_wall} "
                      f"(footprint {_gap:.2f}m from wall)")
    except Exception as _we:
        print(f"  [wall_snap] skipped: {_we}")

    # Snap position_m perpendicular-to-wall for flush objects so the stored value
    # matches what wall_snap produces during rendering, preventing VLM-iteration
    # drift where position_m stays at the pre-snap centroid.
    _flush_gap_save = 0.01
    for _p in placements:
        _wa = _p.get("wall_affinity", "centre")
        if _wa not in ("back", "left", "right"):
            continue
        if _p.get("_facing_applied") or _p.get("_pos_override"):
            continue
        _sm = _p.get("size_m", {})
        _R_arr = _p.get("rotation_3x3")
        if not _R_arr or not _sm:
            continue
        _R_p = np.array(_R_arr, dtype=np.float64)
        _w_m = float(_sm.get("width_m", 0))
        _d_m = float(_sm.get("depth_m", 0))
        if _wa == "left":
            _eff_d = abs(float(_R_p[0, 0])) * _w_m + abs(float(_R_p[0, 2])) * _d_m
            _p["position_m"][0] = _eff_d / 2.0 + _flush_gap_save
        elif _wa == "right":
            try:
                _rw_p, _ = _get_room_dims(Path(_p.get("glb_path", "")), cam)
            except Exception:
                continue
            _eff_d = abs(float(_R_p[0, 0])) * _w_m + abs(float(_R_p[0, 2])) * _d_m
            _p["position_m"][0] = _rw_p - _eff_d / 2.0 - _flush_gap_save
        elif _wa == "back":
            _eff_d = abs(float(_R_p[2, 0])) * _w_m + abs(float(_R_p[2, 2])) * _d_m
            _p["position_m"][2] = _eff_d / 2.0 + _flush_gap_save

    # Save placement metadata (strip non-serialisable fields like numpy arrays)
    # Persist front-face cache for reuse when VLM is unavailable
    _save_front_cache(out_root / "furniture")

    # Deterministic orientation guard: ensure wall-adjacent seating has its back
    # on the wall (front into the room) before rendering/saving.
    _enforce_seating_front_into_room(placements)
    # Bed orientation from the silhouette: pick the cardinal yaw whose projected
    # bed silhouette best matches the segmentation mask (front-detect leaves the
    # bed yaw under-constrained; back-wall vs side-wall headboard can't be told
    # from position alone).
    _pick_bed_yaw_by_silhouette(placements, cam, out_root)
    # Re-fit each object's scale to its segmentation-mask silhouette now that the
    # wall/orientation are final (the earlier mask-fit is stale after reconciliation).
    _fit_scale_to_silhouette(placements, cam)
    # Cap oversized scales (bbox-fit can inflate objects past life-size in
    # diagonal corner views) to realistic per-type footprints.
    _clamp_furniture_scale(placements)
    # Raise under-tall surface furniture (side/coffee tables, desks) to a
    # plausible height relative to the larger pieces — Y-only, so footprint and
    # the clamp above are unaffected.
    _enforce_min_surface_height(placements)
    # Keep free-standing objects inside the room (diagonal views can back-project
    # a centre object's position past a wall).
    _clamp_positions_to_room(placements, cam)

    # ── Seating-group correction + collision resolution (geometry-driven) ────────
    # MUST run last — after orientation/scale/position adjustments — so it separates
    # the FINAL footprints (e.g. a side table the position-clamp pulled in-room must
    # still not overlap the sofa).  Re-anchors a central coffee table in front of the
    # sofa, faces accent chairs at it, separates footprint collisions, caps heights.
    try:
        from object_placement.furniture import seating_group as _seat
        _walls = out_root / "walls.obj"
        if _walls.exists():
            _v = np.array([[float(x) for x in _l.split()[1:4]]
                           for _l in open(_walls) if _l.startswith("v ")])
            _room_w, _room_d = float(_v[:, 0].max()), float(_v[:, 2].max())
            _seat.apply(placements, cam, out_root, _room_w, _room_d)
    except Exception as _se:
        print(f"[seating] correction skipped: {_se}")

    # ── FINAL rotation-aware bounds clamp (last word) ───────────────────────
    # Any earlier pass (depth-optim, vlm_layout, local collision push) can leave
    # an object's footprint clipping through a wall.  size_m is in the object's
    # LOCAL frame, so use the yaw to get world half-extents and clamp the centre
    # so the whole footprint sits inside [0,room_w]×[0,room_d].
    try:
        _wb = out_root / "walls.obj"
        if _wb.exists():
            _vv = np.array([[float(x) for x in _l.split()[1:4]]
                            for _l in open(_wb) if _l.startswith("v ")])
            _RXm, _RZm = float(_vv[:, 0].max()), float(_vv[:, 2].max())
            for _p in placements:
                if _p.get("is_carpet") or "position_m" not in _p:
                    continue
                _sz = _p.get("size_m") or {}
                _w = float(_sz.get("width_m", 0)); _d = float(_sz.get("depth_m", 0))
                if _w <= 0 or _d <= 0:
                    continue
                _R = _p.get("rotation_3x3")
                if _R is not None:
                    _c = abs(float(_R[0][0])); _s = abs(float(_R[0][2]))
                    _hw = (_c * _w + _s * _d) / 2.0; _hd = (_s * _w + _c * _d) / 2.0
                else:
                    _hw, _hd = _w / 2.0, _d / 2.0
                _px, _pz = float(_p["position_m"][0]), float(_p["position_m"][2])
                _nx = _px if 2 * _hw >= _RXm else min(max(_px, _hw), _RXm - _hw)
                _nz = _pz if 2 * _hd >= _RZm else min(max(_pz, _hd), _RZm - _hd)
                if abs(_nx - _px) > 1e-3 or abs(_nz - _pz) > 1e-3:
                    _p["position_m"][0] = _nx; _p["position_m"][2] = _nz
                    print(f"  [bounds_clamp] idx={_p.get('index')} {_p.get('type')}: "
                          f"({_px:.2f},{_pz:.2f}) → ({_nx:.2f},{_nz:.2f})")
    except Exception as _be:
        print(f"[bounds_clamp] skipped: {_be}")

    # ── Shrink occluded small side tables until they clear sofas + walls ──────
    # A side/end table wedged between or behind the sofas has almost no visible
    # silhouette (the sofas occlude it), so the size-fit has too little signal
    # and keeps an oversized default that pokes through the neighbouring sofa
    # and/or the wall.  Shrink such a table uniformly about its base-centre
    # (footprint XZ centre fixed, base stays on the floor) until its footprint
    # no longer significantly overlaps any other furniture and sits inside the
    # room box.  Only fires when there IS a real overlap, so well-placed tables
    # (e.g. an end table just touching a chair) are left alone.
    try:
        _SHRINK_TYPES = {"side_table", "end_table", "nightstand", "accent_table"}
        _rw, _rd = 0.0, 0.0
        _wb = out_root / "walls.obj"
        if _wb.exists():
            _vv = np.array([[float(x) for x in _l.split()[1:4]]
                            for _l in open(_wb) if _l.startswith("v ")])
            _rw, _rd = float(_vv[:, 0].max()), float(_vv[:, 2].max())

        def _foot_aabb(_q):
            _sm = _q.get("size_m") or {}
            _w = float(_sm.get("width_m", 0.0)); _d = float(_sm.get("depth_m", 0.0))
            _R = _q.get("rotation_3x3")
            if _R is not None:
                _c = abs(float(_R[0][0])); _s = abs(float(_R[0][2]))
                _hw = (_c * _w + _s * _d) / 2.0; _hd = (_s * _w + _c * _d) / 2.0
            else:
                _hw, _hd = _w / 2.0, _d / 2.0
            _x = float(_q["position_m"][0]); _z = float(_q["position_m"][2])
            return _x - _hw, _x + _hw, _z - _hd, _z + _hd

        def _ovf(_a, _b):           # fraction of _a's footprint covered by _b
            _ix = max(0.0, min(_a[1], _b[1]) - max(_a[0], _b[0]))
            _iz = max(0.0, min(_a[3], _b[3]) - max(_a[2], _b[2]))
            _area = (_a[1] - _a[0]) * (_a[3] - _a[2])
            return (_ix * _iz) / _area if _area > 1e-9 else 0.0

        for _p in placements:
            if (_p.get("type") not in _SHRINK_TYPES or _p.get("is_carpet")
                    or _p.get("wall_mounted") or "position_m" not in _p
                    or not _p.get("size_m")):
                continue
            _others = [_q for _q in placements
                       if _q is not _p and not _q.get("is_carpet")
                       and not _q.get("wall_mounted") and "position_m" in _q
                       and _q.get("size_m")]
            _it = 0
            while _it < 16:
                _a = _foot_aabb(_p)
                _worst = max((_ovf(_a, _foot_aabb(_q)) for _q in _others), default=0.0)
                _wall_over = 0.0
                if _rw > 0 and _rd > 0:
                    _wall_over = max(0.0 - _a[0], _a[1] - _rw, 0.0 - _a[2], _a[3] - _rd)
                # A side table wedged behind sofas is mostly occluded, so its
                # silhouette gives no reliable size — cap its footprint to a sane
                # side-table max (0.60 m) on top of clearing collisions/walls.
                _oversized = (float(_p["size_m"].get("width_m", 0)) > 0.60
                              or float(_p["size_m"].get("depth_m", 0)) > 0.60)
                # Acceptable: touching (<=12% footprint overlap), inside walls,
                # and not oversized.
                if _worst <= 0.12 and _wall_over <= 0.02 and not _oversized:
                    break
                # Don't shrink below a plausible minimum side-table size.
                if float(_p["size_m"].get("width_m", 0)) <= 0.30:
                    break
                _f = 0.90
                _sc = np.array(_p.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
                if _sc.ndim == 0:
                    _sc = np.array([float(_sc)] * 3)
                _p["scale"] = (_sc * _f).tolist()
                for _k in _p["size_m"]:
                    _p["size_m"][_k] = float(_p["size_m"][_k]) * _f
                if "eff_h" in _p:
                    _p["eff_h"] = float(_p["eff_h"]) * _f
                _it += 1
            if _it > 0:
                _sm = _p["size_m"]
                print(f"  [shrink_fit] idx={_p.get('index')} {_p.get('type')}: "
                      f"shrunk ×{0.90**_it:.2f} → "
                      f"{_sm.get('width_m',0):.2f}×{_sm.get('height_m',0):.2f}×"
                      f"{_sm.get('depth_m',0):.2f} m to clear neighbours/walls")
    except Exception as _se:
        print(f"[shrink_fit] skipped: {_se}")

    # ── Floor grounding: force every floor piece's support Y to 0 ─────────────
    # position_m[1] is the support-surface height (0 = floor).  compute_floor_
    # placement sets it to 0 and mask-align preserves it, but a late scale/mask
    # re-fit can leave it holding the object CENTRE (~half height).  _apply_floor_
    # transform then grounds the BASE to that lifted Y, so the object hovers
    # half its height off the floor.  Reset the support Y to 0 for every grounded
    # piece (skip carpets, wall-mounted, and on_top_of objects that legitimately
    # rest on another surface) so all furniture sits on the ground.
    for _gp in placements:
        if (_gp.get("is_carpet") or _gp.get("wall_mounted")
                or _gp.get("on_top_of") is not None):
            continue
        _pm = _gp.get("position_m")
        if isinstance(_pm, list) and len(_pm) == 3 and abs(float(_pm[1])) > 1e-4:
            print(f"  [ground_reset] idx={_gp.get('index')} {_gp.get('type')}: "
                  f"support Y {float(_pm[1]):.3f} → 0.000")
            _pm[1] = 0.0

    # FINAL containment: scale-down + shift any object that still exceeds the
    # room box (the clamps above only shift and skip wall pieces) so nothing
    # pokes through a wall/floor in the render or saved geometry.
    # ── FINAL silhouette LATERAL pin (env SCENEWEAVE_SILH_LATERAL) ────────────
    # Authoritative over EVERY wall/corner-snap + layout pass above: the
    # silhouette's left/right is trusted over (unreliable, occluded) VGGT depth,
    # so this is the last word on lateral position. _enforce_room_containment
    # below clamps any overshoot back inside the walls.
    if os.environ.get("SCENEWEAVE_SILH_LATERAL"):
        for _lp in placements:
            try:
                _silhouette_lateral_pin(_lp, cam, out_root)
            except Exception:
                pass
    # Fine-yaw refine AFTER the lateral pin: with lateral position now correct the
    # projected mesh overlaps its mask, so the silhouette IoU is meaningful and the
    # coarse 4-cardinal pick can actually decide "keep vs rotate 90° L/R" per object.
    if os.environ.get("SCENEWEAVE_FINE_YAW"):
        for _lp in placements:
            if _lp.get("wall_mounted") or _lp.get("is_carpet"):
                continue
            try:
                _fine_yaw_refine(_lp, cam, out_root)
            except Exception:
                pass
    try:
        _enforce_room_containment(placements, cam)
    except Exception as _ce:
        print(f"[contain] skipped: {_ce}")

    render_furniture(out_root, placements, cam, base_image_path=base_image_path)
    # Final footprint clamp as the LAST word: an earlier clamp can be silently
    # undone by a later re-fit (e.g. a sectional sofa re-inflating to its mask
    # silhouette), so re-apply it here so the SAVED geometry and the pyrender
    # composite below both use the clamped scale.
    _clamp_furniture_scale(placements)

    # ── Geometry audit (corrective) + top-down plan ─────────────────────────
    # Runs on the FINAL footprints, immediately before the geometry is baked, so
    # what the audit signs off on is what gets saved.  Unlike the `SIZE-SUSPECT`
    # print upstream, this one acts: wall penetrations and interpenetrations are
    # resolved (a wall-flush piece may only slide along its wall, never off it)
    # and every finding is recorded with the metres it was off by.
    # The plan render exists because wall adjacency and depth ordering are
    # ambiguous in the single reference-camera view and unambiguous from above.
    try:
        from object_placement import geometry_audit as _ga
        _wa = out_root / "walls.obj"
        if _wa.exists():
            _vv = np.array([[float(x) for x in _l.split()[1:4]]
                            for _l in open(_wa) if _l.startswith("v ")])
            _rw, _rd = float(_vv[:, 0].max()), float(_vv[:, 2].max())
            _rep = _ga.audit(placements, _rw, _rd, correct=True,
                             out_path=out_root / "furniture" / "geometry_audit.json")
            _ga.render_plan(placements, _rw, _rd,
                            out_root / "furniture" / "render_topdown.png",
                            findings=_rep["findings"], cam=cam,
                            title=f"{out_root.name} — furniture plan")
    except Exception as _ae:
        print(f"[geom_audit] skipped: {_ae}")

    save_placed_geometry(out_root, placements)

    # Save placements JSON *after* all post-passes (seating-front, silhouette re-fit,
    # scale clamp, min-height, position clamp, seating-group correction) and the final
    # clamp above, so the JSON matches the SAVED geometry and rendered composite.
    # Writing it earlier (pre-clamp) left a stale file that over-stated oversized
    # scales — downstream consumers (ceiling stage, assemble_scene) read this file.
    _SKIP_KEYS = {"_verts", "_faces", "_vc", "_mask_fit_data", "_pre_slide_colliders",
                  "_xz_hull_pts", "_mask_align_final_err_px", "_silhouette_w_m"}
    serialisable = [
        {k: v for k, v in p.items() if k not in _SKIP_KEYS}
        for p in placements
    ]
    out_json = out_root / "furniture" / "furniture_placements.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[place_furn] Placements saved → {out_json}")

    # Replace the flat per-vertex composite with a proper textured + lit pyrender
    # perspective render (per-pixel texture sampling).  Falls back silently to the
    # rasteriser output if pyrender / EGL isn't usable on this machine.
    try:
        render_furniture_pyrender(out_root, cam, placements=placements,
                                  base_image_path=base_image_path)
    except Exception as _pye:
        print(f"[render_furn_pyr] failed ({_pye}) — keeping rasteriser output")

    # Per-step build-up animation: render the placement one object at a time onto
    # the empty room, saved as render_placement_buildup.gif.
    try:
        render_placement_buildup(out_root, cam, placements,
                                 base_image_path=base_image_path)
    except Exception as _be:
        print(f"[buildup] failed ({_be}) — skipping animation")

    # ── Assemble animation GIF ─────────────────────────────────────────────────
    if animate and _anim_frame_paths:
        import shutil
        from PIL import Image as _PIL
        gif_h_ratio = None
        frames_pil: list[_PIL.Image] = []
        for fp in _anim_frame_paths:
            try:
                img = _PIL.open(fp).convert("RGB")
                if gif_h_ratio is None:
                    gif_h_ratio = img.height / img.width
                new_h = int(animate_gif_width * gif_h_ratio)
                frames_pil.append(img.resize((animate_gif_width, new_h), _PIL.LANCZOS))
            except Exception as _fe:
                print(f"  [animate] skipping frame {fp}: {_fe}")
        if len(frames_pil) >= 2:
            gif_path = out_root / "furniture" / "placement_animation.gif"
            frames_pil[0].save(
                str(gif_path), save_all=True, append_images=frames_pil[1:],
                duration=animate_frame_ms, loop=0,
            )
            print(f"\n[animate] Saved furniture animation → {gif_path} "
                  f"({len(frames_pil)} frames)")
        shutil.rmtree(_anim_tmp_dir, ignore_errors=True)

    # ── VGGT refine summary plot ───────────────────────────────────────────────
    if _vggt_calib is not None:
        try:
            _vggt_summary_path = _vggt_save_summary(out_root, placements)
            if _vggt_summary_path is not None:
                print(f"[vggt-refine] summary plot → {_vggt_summary_path}")
        except Exception as _vse:
            print(f"[vggt-refine] summary plot failed: {_vse}")

    return out_root / "furniture"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Place furniture GLBs on floor and render."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output directory")
    ap.add_argument("--indices", type=int, default=0,
                    help="Number of objects to place from placement_analysis.json "
                         "(0 = all, default)")
    ap.add_argument("--only-indices", type=int, nargs="+", default=None,
                    metavar="IDX",
                    help="Restrict placement to these specific object indices "
                         "(e.g. --only-indices 0 5 8). Overrides --indices.")
    ap.add_argument("--carpet-only", action="store_true",
                    help="Only place carpet/rug objects, skip all other furniture")
    ap.add_argument("--vlm-refine-iters", type=int, default=10,
                    help="Max VLM iterative refinement iterations per furniture object "
                         "(default 10; 0 to disable)")
    ap.add_argument("--animate", action="store_true",
                    help="Save furniture/placement_animation.gif showing each object "
                         "being added and VLM-refined step by step")
    ap.add_argument("--animate-width", type=int, default=960,
                    help="GIF frame width in pixels (default 960)")
    ap.add_argument("--animate-duration", type=int, default=1000,
                    help="Milliseconds per frame in the GIF (default 1000)")
    ap.add_argument("--no-vggt-refine", dest="vggt_refine", action="store_false",
                    default=True,
                    help="Disable VGGT-based ground-truth refinement after silhouette mask_align")
    ap.add_argument("--vggt-blend", type=float, default=0.5,
                    help="Blend weight toward VGGT prediction (0=ignore, 1=snap, default 0.5)")
    ap.add_argument("--vggt-max-shift", type=float, default=0.6,
                    help="Cap on per-object XZ shift (m) introduced by VGGT refine (default 0.6)")
    ap.add_argument("--vggt-face-offset", dest="vggt_face_offset",
                    action="store_true", default=False,
                    help="Apply a visible-face → bbox-center offset (only for visualization parity; "
                         "math is in lateral/depth components)")
    ap.add_argument("--vggt-lateral-weight", type=float, default=1.0,
                    help="Weight on the camera-right (lateral) component of VGGT's suggested shift "
                         "(default 1.0 — full lateral correction)")
    ap.add_argument("--vggt-depth-weight", type=float, default=0.0,
                    help="Weight on the camera-forward (depth) component of VGGT's suggested shift "
                         "(default 0.0 — depth is unreliable, ignore it)")
    ap.add_argument("--vggt-method", choices=("render_diff", "global_calib"),
                    default="render_diff",
                    help="render_diff: per-iter VGGT on rendered scene + diff vs photo (slow, "
                         "bias-cancelling, default). global_calib: one Sim(3) via walls.obj reused "
                         "for every object (fast, depth-biased).")
    ap.add_argument("--iter-max-dev", type=float, default=0.15,
                    help="Per-iteration silhouette anchor clamp (m). After every step "
                         "(VGGT, post_mask_clearance, etc.) the object is pulled back to "
                         "within this distance of its mask_align silhouette anchor. "
                         "Smaller = mask_align dominates; larger = VGGT can move objects "
                         "more aggressively (default 0.15).")
    ap.add_argument("--no-vlm-fine-rotation", dest="vlm_fine_rotation",
                    action="store_false", default=True,
                    help="Disable iterative fine-yaw VLM refinement of seating items "
                         "after silhouette mask_align (default on)")
    ap.add_argument("--no-vlm-post-verify", dest="vlm_post_verify",
                    action="store_false", default=True,
                    help="Disable post-placement VLM orientation verification pass "
                         "(default on)")
    ap.add_argument("--no-scene-review", dest="vlm_scene_review",
                    action="store_false", default=True,
                    help="Disable scene-level VLM review and position corrections "
                         "after all objects are placed (default on)")
    ap.add_argument("--allow-rotation-types", nargs="+", default=[], metavar="TYPE",
                    help="Furniture types to remove from the deterministic-rotation "
                         "override list, allowing the VLM cardinal rotation to apply "
                         "to them (e.g. --allow-rotation-types sofa chair)")
    ap.add_argument("--vlm-fine-rotation-max-deg", type=int, default=25,
                    help="Per-iteration cap on yaw correction in degrees (default 25)")
    ap.add_argument("--vlm-fine-rotation-iters", type=int, default=3,
                    help="Max VLM rotation refinement iterations per item (default 3)")
    ap.add_argument("--base-render", default=None,
                    help="Override base background image for furniture compositing.")
    args = ap.parse_args()
    run(output_dir=args.output_dir, n_objects=args.indices,
        carpet_only=args.carpet_only, only_indices=args.only_indices,
        vlm_refine_iters=args.vlm_refine_iters,
        animate=args.animate,
        animate_gif_width=args.animate_width,
        animate_frame_ms=args.animate_duration,
        vggt_refine=args.vggt_refine,
        vggt_blend=args.vggt_blend,
        vggt_max_shift_m=args.vggt_max_shift,
        vggt_visible_face_offset=args.vggt_face_offset,
        vggt_lateral_weight=args.vggt_lateral_weight,
        vggt_depth_weight=args.vggt_depth_weight,
        vggt_method=args.vggt_method,
        iter_max_dev_m=args.iter_max_dev,
        vlm_fine_rotation=args.vlm_fine_rotation,
        vlm_fine_rotation_max_deg=args.vlm_fine_rotation_max_deg,
        vlm_fine_rotation_iters=args.vlm_fine_rotation_iters,
        vlm_post_verify=args.vlm_post_verify,
        vlm_scene_review=args.vlm_scene_review,
        allow_rotation_types=frozenset(args.allow_rotation_types),
        base_image_path=args.base_render)


if __name__ == "__main__":
    main()
