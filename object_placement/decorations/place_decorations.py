"""
place_decorations.py — place decoration objects onto furniture surfaces.

Pipeline:
  1. Load segment_results.json (decorations with GLB paths)
  2. Load furniture_placements.json (furniture geometry)
  3. For each decoration:
     a. VLM estimates real-world size and placement relationship
     b. Compute 3D position/orientation on furniture surface
     c. Collision check with already-placed decorations on same surface
  4. Save decoration_placements.json

Placement types:
  - on_surface   : sits flat on top of furniture (books, monitor, lamp, keyboard)
  - against_back : placed on seat AND leaning against the back rest (pillows)
  - on_floor     : stands on the floor (floor lamp, large plant)

Usage:
    python -m object_placement.decorations.place_decorations \
        --output-dir outputs/office8
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
import requests

VLM_API_URL = "http://localhost:8080/v1/chat/completions"
from object_placement.vlm_backend import vlm_post as _vlm_post

# Common-sense real-world size fallbacks (width, height, depth in metres).
# Used when VLM call fails. Keys are substring-matched against phrase.
# (keyword, min_h, max_h, max_footprint_w, max_footprint_d)
# max_footprint caps width/depth AFTER uniform scale to prevent absurd proportions
_SIZE_CLAMPS: list[tuple[str, float, float, float, float]] = [
    # Monitor cap widened: silhouettes commonly want ~15-30% more than the
    # VLM's first guess (proj is mesh-only, mask includes anti-aliased halo).
    ("computer monitor",  0.25, 0.85,  0.95, 0.45),
    ("monitor",           0.25, 0.85,  0.95, 0.45),
    ("laptop",            0.02, 0.05,  0.40, 0.30),
    ("keyboard",          0.02, 0.06,  0.50, 0.25),
    # Lamp cap moderated: generated desk-lamp GLBs often have native_w ≈ 10×
    # native_h, so even tight uniform scaling produces a wide footprint.
    # 0.90m wide left lamps sprawling across coffee tables and overlapping
    # monitors / keyboards.  0.55m wide × 0.45m deep is a more realistic cap
    # while still letting silhouette refinement grow the lamp from the VLM's
    # initial guess (~0.30-0.40m base footprint typical).
    # Min heights bumped: a 14cm "lamp" looks like a paperweight; real desk
    # lamps are 35-60cm tall.  Silhouette refinement was shrinking lamps below
    # recognisable size when the photo masks were small / partially occluded.
    ("desk lamp",         0.40, 1.15,  0.85, 0.65),
    ("floor lamp",        1.20, 2.20,  0.90, 0.90),
    ("lamp",              0.35, 1.15,  0.85, 0.65),
    # Book stack: cap bumped to 0.50m so silhouette can grow a stack to
    # match the photo's "books span the table diameter" appearance.  Min
    # height bumped 0.18→0.22 so the default render starts taller.
    ("book",              0.22, 0.45,  0.50, 0.50),
    ("books",             0.22, 0.45,  0.50, 0.50),
    ("pillow",            0.30, 0.60,  0.60, 0.20),
    ("cushion",           0.25, 0.55,  0.55, 0.20),
    # Vase / plant min heights bumped: a 13cm "vase" reads as a knickknack.
    # Real decorative vases on desks/coffee_tables are 22-35cm.  Small potted
    # plants sit around 25-40cm including the pot.
    ("vase",              0.22, 0.50,  0.20, 0.20),
    ("plant",             0.30, 0.80,  0.40, 0.40),
    ("mug",               0.08, 0.15,  0.12, 0.12),
    ("cup",               0.08, 0.15,  0.12, 0.12),
    ("bowl",              0.05, 0.15,  0.25, 0.25),
    ("remote",            0.02, 0.05,  0.08, 0.22),
    ("tray",              0.03, 0.10,  0.50, 0.40),
]

def _clamp_vlm_size(phrase: str, size_dict: dict) -> dict:
    """Clamp VLM height to plausible range and enforce max footprint per object type.

    Special case for books: detect lying-flat vs standing-upright from the
    VLM's own aspect ratio.  When height << width/depth, the books are
    flat-laid on a surface (height = stack thickness, ~3-15 cm) and we
    skip the tall min-height clamp — uniform scaling on a clamped-up
    height would scale width/depth proportionally and produce a giant
    slab.  When height ≳ width/depth, books are an upright stack and the
    standard clamp applies.
    """
    pl = phrase.lower()
    for kw, lo, hi, max_w, max_d in _SIZE_CLAMPS:
        if kw in pl:
            h = float(size_dict.get("height", size_dict.get("height_m", 0.3)))
            w = float(size_dict.get("width",  size_dict.get("width_m",  h)))
            d = float(size_dict.get("depth",  size_dict.get("depth_m",  h)))
            _is_book = kw in ("book", "books")
            _avg_horiz = (w + d) / 2.0
            _flat_book = (_is_book
                          and _avg_horiz > 1e-3
                          and h < _avg_horiz * 0.5)
            if _flat_book:
                # Lying flat: keep the VLM's thickness (with a generous
                # absolute floor so the 3D generator doesn't render a 1 mm sliver).
                clamped = max(0.02, min(hi, h))
                if abs(clamped - h) > 0.001:
                    print(f"    [size_clamp] '{kw}' (lying flat — h={h:.2f}m << "
                          f"avg(w,d)={_avg_horiz:.2f}m): height {h:.3f}→"
                          f"{clamped:.3f} m (skipped tall-stack min)")
            else:
                clamped = max(lo, min(hi, h))
                if abs(clamped - h) > 0.001:
                    print(f"    [size_clamp] '{kw}': height {h:.3f}→{clamped:.3f} m")
            return {**size_dict, "height": clamped,
                    "_max_footprint_w": max_w, "_max_footprint_d": max_d}
    return size_dict


_SIZE_FALLBACKS: list[tuple[str, dict, str, str, str]] = [
    # (phrase_keyword, size_m, placement_type, surface_position, facing)
    ("computer monitor",  {"width": 0.55, "height": 0.45, "depth": 0.22}, "on_surface", "center", "toward_room"),
    ("monitor",           {"width": 0.55, "height": 0.45, "depth": 0.22}, "on_surface", "center", "toward_room"),
    ("laptop",            {"width": 0.35, "height": 0.02, "depth": 0.24}, "on_surface", "center", "toward_room"),
    ("keyboard",          {"width": 0.45, "height": 0.03, "depth": 0.15}, "on_surface", "front",  "toward_room"),
    ("computer keyboard", {"width": 0.45, "height": 0.03, "depth": 0.15}, "on_surface", "front",  "toward_room"),
    ("desk lamp",         {"width": 0.20, "height": 0.55, "depth": 0.20}, "on_surface", "right",  "any"),
    ("lamp",              {"width": 0.25, "height": 0.60, "depth": 0.25}, "on_surface", "right",  "any"),
    ("floor lamp",        {"width": 0.30, "height": 1.70, "depth": 0.30}, "on_floor",   "right",  "any"),
    ("book",              {"width": 0.24, "height": 0.06, "depth": 0.30}, "on_surface", "left",   "any"),
    ("books",             {"width": 0.24, "height": 0.12, "depth": 0.30}, "on_surface", "left",   "any"),
    ("pillow",            {"width": 0.50, "height": 0.50, "depth": 0.15}, "against_back", "center", "any"),
    ("cushion",           {"width": 0.45, "height": 0.45, "depth": 0.15}, "against_back", "center", "any"),
    ("vase",              {"width": 0.15, "height": 0.30, "depth": 0.15}, "on_surface", "center", "any"),
    ("plant",             {"width": 0.25, "height": 0.35, "depth": 0.25}, "on_surface", "right",  "any"),
    ("mug",               {"width": 0.10, "height": 0.12, "depth": 0.10}, "on_surface", "right",  "any"),
    ("cup",               {"width": 0.10, "height": 0.12, "depth": 0.10}, "on_surface", "right",  "any"),
    ("remote",            {"width": 0.05, "height": 0.02, "depth": 0.20}, "on_surface", "center", "any"),
    ("tray",              {"width": 0.40, "height": 0.05, "depth": 0.30}, "on_surface", "center", "any"),
]


def _count_mask_components(mask_path: "Path | None",
                           min_area_frac: float = 0.10,
                           max_components: int = 4) -> int:
    """Count distinct connected-component blobs in a SAM mask.

    Used to detect when a single segment captured multiple physical instances
    side-by-side (e.g. two book piles) — in which case the silhouette area
    over-reports a single instance's size by ~N×.  Components smaller than
    `min_area_frac` of the largest are treated as noise and ignored.

    Returns 1 if the mask is missing, unreadable, or only one significant blob.
    """
    if mask_path is None or not Path(mask_path).exists():
        return 1
    try:
        import cv2
        import numpy as _np
        arr = _np.array(Image.open(str(mask_path)).convert("RGBA"))
        m = (arr[..., 3] > 0).astype(_np.uint8)
        if m.sum() == 0:
            return 1
        n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        # stats[0] is the background row → drop it
        if n_labels <= 2:
            return 1
        areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
        if areas.size == 0:
            return 1
        thresh = float(areas.max()) * float(min_area_frac)
        n = int((areas >= thresh).sum())
        return max(1, min(max_components, n))
    except Exception:
        return 1


def _size_fallback(phrase: str) -> dict:
    """Return a fallback placement spec based on keyword matching."""
    pl = phrase.lower()
    for keyword, size_m, ptype, side, facing in _SIZE_FALLBACKS:
        if keyword in pl:
            return {
                "size_m": size_m,
                "placement_type": ptype,
                "surface_position": side,
                "depth_position": "middle",
                "facing": facing,
            }
    return {
        "size_m": {"width": 0.25, "height": 0.25, "depth": 0.25},
        "placement_type": "on_surface",
        "surface_position": "center",
        "depth_position": "middle",
        "facing": "any",
    }


# Seat height as fraction of sofa/chair eff_h (seat cushion sits ~45% up)
_SEAT_FRAC     = 0.46   # fraction of eff_h where seat cushion surface sits (~46% of back height)
# Margin to keep decorations away from furniture edges (metres)
_EDGE_MARGIN   = 0.03
# Sofa back-offset: how close to the sofa back the pillow centre lands
_BACK_MARGIN   = 0.05
# Fraction of sofa depth occupied by the back rest (from the back face inward)
_BACK_REST_FRAC = 0.12  # low-poly sofas GLBs typically have thin back cushions
                        # (~10% of sofa depth).  Was 0.28 which assumed deeper
                        # padded backs and left pillows floating in front of
                        # the actual back cushion.  Probe overrides this when
                        # it works; this is the fallback for rotated sofas
                        # where the horizontal raycast misses the mesh.
# Width of sofa arm rest (excluded from pillow placement zone)
_ARM_REST_WIDTH = 0.13

_SEATING_TYPES = {"sofa", "couch", "loveseat", "chair", "armchair", "stool"}

# Keywords in phrase that imply against_back → must land on seating furniture
# Phrases for soft items that belong ON SEATING (sofa/chair), not on tables.
# Drives both `prefer_seating` (geo_check won't force them onto a table) and
# `_vfy_is_ab` (the verify pass won't block a correction back onto seating).
# Blankets / throws are draped over a sofa, so they must be treated like pillows
# here — otherwise an "on_surface" blanket gets pushed onto the nearest coffee
# table and the correction back to the sofa is blocked.
_AGAINST_BACK_PHRASES = {"pillow", "cushion", "blanket", "throw"}


# ── De-tilt helpers (adapted from place_furniture) ────────────────────────────

def _dec_find_base_y(verts: np.ndarray) -> float:
    """Find Y level where the object's actual base starts (skips spike vertices)."""
    y_min = float(verts[:, 1].min())
    y_range = float(verts[:, 1].max() - y_min)
    if y_range < 1e-6:
        return y_min
    obj_spread = max(float(np.ptp(verts[:, 0])), float(np.ptp(verts[:, 2])))
    if obj_spread < 1e-6:
        return y_min
    for pct in range(0, 30, 2):
        y_level = y_min + (pct / 100.0) * y_range
        below = verts[verts[:, 1] <= y_level]
        if len(below) < 3:
            continue
        xz_spread = max(float(np.ptp(below[:, 0])), float(np.ptp(below[:, 2])))
        if xz_spread >= 0.15 * obj_spread:
            if pct > 0:
                return y_min + (pct / 100.0) * y_range
            return y_min
    return y_min


def _dec_level_base(verts: np.ndarray, max_angle_deg: float = 45.0) -> np.ndarray:
    """Level decoration mesh so its bottom rests flat (correct Hunyuan3D tilt).

    Fits a plane to bottom contact vertices and applies a pitch/roll correction
    rotation.  Caps at max_angle_deg to avoid flipping well-oriented models.
    Returns new vertex array with base anchored at Y=0.
    """
    y_min = float(verts[:, 1].min())
    y_range = float(verts[:, 1].max()) - y_min
    if y_range < 1e-6:
        return verts

    obj_spread = max(float(np.ptp(verts[:, 0])), float(np.ptp(verts[:, 2])))

    # Strategy 1: 4-quadrant corner contacts (bottom 25% of Y range)
    y_thresh = y_min + 0.25 * y_range
    bot = verts[verts[:, 1] <= y_thresh]
    pts = None
    if len(bot) >= 4:
        x_med = (float(verts[:, 0].min()) + float(verts[:, 0].max())) / 2.0
        z_med = (float(verts[:, 2].min()) + float(verts[:, 2].max())) / 2.0
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

    # Strategy 2: progressively wider bottom slices
    if pts is None:
        for frac in (0.05, 0.10, 0.15, 0.25):
            y_lo = y_min + frac * y_range
            cand = verts[verts[:, 1] <= y_lo]
            if len(cand) < 3:
                continue
            xz_spread = max(float(np.ptp(cand[:, 0])), float(np.ptp(cand[:, 2])))
            if xz_spread >= 0.15 * obj_spread:
                pts = cand.copy()
                break
        if pts is None:
            # No stable base found — just anchor Y_min to 0
            out = verts.copy()
            out[:, 1] -= y_min
            return out

    pts = pts.astype(np.float64)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[1] < 0:
        normal = -normal

    angle = float(np.degrees(np.arccos(np.clip(float(normal[1]), -1.0, 1.0))))
    out = verts.copy()
    if angle >= 0.5 and angle <= max_angle_deg:
        axis = np.cross(normal, np.array([0., 1., 0.]))
        axis_len = float(np.linalg.norm(axis))
        if axis_len > 1e-9:
            axis /= axis_len
            theta = float(np.arccos(np.clip(float(normal[1]), -1., 1.)))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=np.float64)
            R_level = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            out = out @ R_level.T
            print(f"    [detilt] corrected tilt {angle:.1f}°")

    base_y = _dec_find_base_y(out)
    out[:, 1] -= base_y
    return out


# ── Furniture surface raycasting ──────────────────────────────────────────────

_FURN_SCENE_MESH_CACHE: dict[str, "object"] = {}


def _get_furn_scene_mesh(out_dir: Path):
    """Load and cache scene_with_furniture.glb as a single trimesh for raycasting."""
    key = str(out_dir)
    if key in _FURN_SCENE_MESH_CACHE:
        return _FURN_SCENE_MESH_CACHE[key]
    try:
        import trimesh
        glb = out_dir / "furniture" / "scene_with_furniture.glb"
        if not glb.exists():
            _FURN_SCENE_MESH_CACHE[key] = None
            return None
        scene = trimesh.load(str(glb), force="scene")
        if isinstance(scene, trimesh.Scene):
            meshes = scene.dump()
        else:
            meshes = [scene]
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        _FURN_SCENE_MESH_CACHE[key] = mesh
        print(f"  [surface] furniture scene mesh cached for {out_dir.name}")
        return mesh
    except Exception as e:
        print(f"  [surface] failed to load furniture scene mesh: {e}")
        _FURN_SCENE_MESH_CACHE[key] = None
        return None


def _actual_surface_height(out_dir: Path, x: float, z: float,
                            nom_y: float, search_radius: float = 0.15) -> float:
    """Raycast downward against furniture GLB to find actual surface height at (x,z).

    Casts from nom_y+0.5 downward and returns the topmost real surface at or
    below nom_y+search_radius (above the floor).  Using "topmost at/below the
    expected top" rather than a symmetric ±search_radius window makes this
    robust to an over-estimated `eff_h`: Hunyuan-reconstructed tables/desks
    often report a bbox height well above the true tabletop (the same defect
    `_bed_mattress_height` works around for beds), which would push nom_y above
    the real surface so a symmetric window rejects the genuine hit and the
    object hovers.  Falls back to nom_y if no surface is found.
    """
    # Ablation wo_placement_order: place decorations WITHOUT any connection to the
    # supporting furniture — drop each one to the floor at its back-projected XZ
    # instead of snapping onto the furniture surface (shows the stage hierarchy's value).
    if __import__("os").environ.get("SCENEWEAVE_ABLATE_NO_DECO_ANCHOR") == "1":
        print(f"    [surface] ABLATION no-deco-anchor → floor (was nom={nom_y:.3f})")
        return 0.0
    mesh = _get_furn_scene_mesh(out_dir)
    if mesh is None:
        return nom_y
    try:
        ray_origin = np.array([[x, nom_y + 0.5, z]])
        ray_dir    = np.array([[0., -1., 0.]])
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origin, ray_directions=ray_dir, multiple_hits=True)
        if len(locs) > 0:
            # Keep hits above the floor and not above the expected top (plus a
            # small slack).  This excludes floor hits and any taller neighbour
            # poking above nom_y, while still accepting the true surface when it
            # sits well below an inflated nom_y.
            ys = locs[:, 1]
            valid = locs[(ys > 0.10) & (ys <= nom_y + search_radius)]
            if len(valid) > 0:
                hit_y = float(valid[:, 1].max())
                print(f"    [surface] raycast hit y={hit_y:.3f}  (nom={nom_y:.3f}  Δ={hit_y-nom_y:+.3f}m)")
                return hit_y
    except Exception as e:
        print(f"    [surface] raycast failed: {e}")
    return nom_y


def _bed_mattress_height(out_dir: Path, furn: dict) -> float | None:
    """For beds: find the mattress plateau (the large flat area), NOT the
    headboard top.  Beds reconstructed by Hunyuan have eff_h equal to the
    headboard height; placing decorations at eff_h puts pillows on top of
    the headboard instead of on the mattress.

    Strategy: cast a 5×5 grid of rays across the bed's XZ footprint, take
    each hit-Y, and return the MODE (the plateau covering the most area).
    The headboard is a small XZ footprint relative to the mattress, so its
    Y won't be the mode.  Returns None if the grid sampling fails.
    """
    mesh = _get_furn_scene_mesh(out_dir)
    if mesh is None:
        return None
    try:
        cx = float(furn["position_m"][0])
        cz = float(furn["position_m"][2])
        sm = furn.get("size_m", {})
        w  = float(sm.get("width_m", 1.5))
        d  = float(sm.get("depth_m", 2.0))
        eff_h = float(furn.get("eff_h", 0.6))
        # 5×5 grid, sampling 70% of the footprint (avoids edge artifacts)
        N = 5
        xs = np.linspace(cx - 0.35 * w, cx + 0.35 * w, N)
        zs = np.linspace(cz - 0.35 * d, cz + 0.35 * d, N)
        origins = []
        for xi in xs:
            for zi in zs:
                origins.append([xi, eff_h + 1.0, zi])
        origins = np.array(origins, dtype=np.float64)
        dirs    = np.tile([0.0, -1.0, 0.0], (len(origins), 1))
        locs, ray_idx, _ = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=dirs, multiple_hits=True)
        if len(locs) == 0:
            return None
        # Per-ray HIGHEST hit (the topmost surface visible from above).
        per_ray: dict[int, float] = {}
        for i, ri in enumerate(ray_idx):
            y = float(locs[i, 1])
            if y < 0.05:
                continue   # skip floor hits
            if int(ri) not in per_ray or y > per_ray[int(ri)]:
                per_ray[int(ri)] = y
        if not per_ray:
            return None
        ys = np.array(sorted(per_ray.values()))
        # Bin and find the mode (mattress plateau spans the widest XZ area).
        # Use 5cm bins.
        BIN = 0.05
        bins: dict[int, int] = {}
        for y in ys:
            b = int(round(y / BIN))
            bins[b] = bins.get(b, 0) + 1
        best_b, best_n = max(bins.items(), key=lambda kv: kv[1])
        mode_y = best_b * BIN
        # Refine: average all hits within ±BIN of the mode bin.
        in_bin = ys[(ys >= (best_b - 0.5) * BIN) & (ys < (best_b + 1.5) * BIN)]
        mattress_y = float(np.mean(in_bin)) if len(in_bin) else mode_y
        print(f"    [bed_surface] mattress plateau y={mattress_y:.3f} "
              f"(eff_h={eff_h:.3f}; {best_n}/{len(ys)} rays in mode bin)")
        return mattress_y
    except Exception as e:
        print(f"    [bed_surface] failed: {e}")
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_pil(img: Image.Image, max_side: int = 512) -> str:
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _glb_native_extents(glb_path: Path) -> np.ndarray:
    """Return [width, height, depth] of a GLB in its native units."""
    import trimesh
    mesh = trimesh.load(str(glb_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    extents = mesh.bounding_box.extents           # [dx, dy, dz]
    return extents.astype(np.float64)


# ── Furniture geometry helpers ────────────────────────────────────────────────

def _rot_matrix(furn: dict) -> np.ndarray:
    """3×3 rotation matrix as numpy array (row-major from JSON). Defaults to identity."""
    r = furn.get("rotation_3x3")
    if r is None:
        return np.eye(3, dtype=np.float64)
    return np.array(r, dtype=np.float64)


def _world_axes(furn: dict):
    """Return (right_world, up_world, front_world) unit vectors.

    right_world  = R @ [1,0,0]   (local X in world space)
    front_world  = R @ local_front  (actual front face direction)
    """
    R = _rot_matrix(furn)
    # Columns of R are local axes in world space
    right_world = R[:, 0]                          # local +X in world
    up_world    = np.array([0.0, 1.0, 0.0])
    lf          = np.array(furn.get("_local_front", [0.0, 0.0, 1.0]))
    front_world = R @ lf
    return right_world, up_world, front_world


def _top_surface_center(furn: dict) -> np.ndarray:
    """World-space centre of the furniture's top surface."""
    pos = np.array(furn["position_m"], dtype=np.float64)
    pos[1] += furn["eff_h"]
    return pos


def _seat_surface_center(furn: dict) -> np.ndarray:
    """World-space centre of the seat surface for seating furniture."""
    pos = np.array(furn["position_m"], dtype=np.float64)
    pos[1] += furn["eff_h"] * _SEAT_FRAC
    return pos


# ── VLM: furniture selection via mask location ────────────────────────────────

_SELECT_FURN_PROMPT = """\
You are an interior design expert.

Image 1 (reference photo): the original room photo with "{phrase}" highlighted \
in red (the red tinted region IS the object).  Look carefully at WHAT SURFACE or \
FURNITURE the red-highlighted object is physically resting on or placed on.

Image 2 (render): the 3D-rendered version of the same room with every \
furniture piece labelled "[N] type".

The labelled furniture pieces are:
{furn_list}

Task:
1. In Image 1, identify the TYPE of furniture "{phrase}" is resting on \
(e.g. "coffee table", "bookshelf", "sofa", "armchair"). LOOK AT THE WHOLE \
furniture piece, not just the region under the red highlight — a highlight \
covering a single cushion of a multi-cushion sofa is still on a SOFA, not a \
chair.  If it is on the floor, note that.
2. Also describe what the surface looks like in Image 1: is it LARGE or SMALL, \
CLOSE to the camera or FAR from it, is it round or rectangular?  These visual \
cues help disambiguate between same-type instances (e.g. two coffee tables).
3. In Image 2, pick the labelled furniture piece whose TYPE matches AND whose \
SIZE / CAMERA-DISTANCE / shape matches the cues from Image 1.  The metadata \
after each label tells you its real size in metres and how far it is from the \
camera — use that to pick between e.g. "the LARGER coffee table closer to the \
camera" vs "the smaller one farther away".
4. Return that furniture piece's index.

Return ONLY valid JSON, no markdown.  IMPORTANT: the integer in "furniture_index"
MUST be the same index you name in "reasoning".  Write the index into the
reasoning text FIRST, then copy it EXACTLY into the furniture_index field.  If
your reasoning concludes "[5] coffee_table", then furniture_index MUST be 5 —
not 2, not any other number.

{{
  "furniture_index": <integer label from Image 2, or -1 if on the floor>,
  "reasoning": "one sentence: what TYPE (chair? sofa? coffee_table? desk?) of surface {phrase} is on in Image 1, and which labelled index in Image 2 matches BOTH the type AND its room position"
}}
"""

_VERIFY_PLACEMENT_PROMPT = """\
You are an interior design expert reviewing a room render.

Image 1: the 3D-rendered room with furniture index labels "[N] type" and a red
         highlight showing where "{phrase}" should be (from the reference photo).
Image 2: the render after placing "{phrase}" — check where it actually landed.
Image 3: the original reference photo — use this to confirm which furniture type
         "{phrase}" actually sits on in the real room.

The labelled furniture pieces are:
{furn_list}

Step-by-step reasoning:
1. Look at Image 3 (reference photo): identify which furniture type "{phrase}" is
   on in the real room (e.g. coffee table, sofa, shelf). Note its surface type.
   Look at the WHOLE furniture piece — a highlight covering one cushion of a
   multi-cushion sofa is still on a SOFA, not a chair.  Also note visual cues:
   is it LARGE vs small, CLOSE to camera vs far, round vs rectangular?
2. COUNT: how many "{phrase}" instances does the reference photo show, and on
   which furniture piece does each sit?  If the reference shows ONE pillow per
   sofa across a sofa pair, but the render has TWO pillows on the same sofa and
   the other sofa is bare, the second pillow belongs on the bare sofa.  If the
   reference has a pillow on a spare CHAIR that matches the pillow's colour,
   and the render piled that pillow onto a sofa, the pillow should move to the
   chair.
3. Look at Image 2 (render): identify which labelled furniture "{phrase}" landed on.
4. Compare — does the furniture type in Image 2 match what you saw in Image 3?
   Also compare with the red-highlighted region in Image 1.  When multiple same-
   type candidates exist, use the size/camera-distance metadata after each label
   plus visual cues (larger vs smaller, closer vs farther) to pick the right
   instance — do NOT assume any labelled piece is correct without checking these
   cues.

If the placement is on the correct furniture (matches reference photo), return:
  {{"correct": true, "furniture_index": <current index>, "reasoning": "..."}}

If it landed on the wrong furniture, return:
  {{"correct": false, "furniture_index": <correct index from Image 1's labels — this MUST be a different index than the current placement index {current_furn_idx}>, "reasoning": "..."}}

IMPORTANT:
- Only mark as wrong if you are confident the furniture TYPE is wrong
  (e.g. landed on a sofa instead of a coffee table).
- Do not correct minor position differences within the same furniture piece.
- "furniture_index" when correct=false MUST be a label that exists in the furniture list above
  AND must be different from {current_furn_idx}.

Return ONLY valid JSON, no markdown.
"""

_ARM_REST_PROMPT = """\
You are an interior design expert reviewing a pillow placement on a sofa.

Image 1: the 3D render showing "{phrase}" placed on a sofa.
Image 2: the reference photo showing the correct placement.

Check whether "{phrase}" in Image 1 is visually overlapping or colliding with
the sofa's arm rest (the raised padded side panels).

If there is an arm-rest collision, which direction should the pillow move
(toward the sofa CENTER) to resolve it?

Return ONLY valid JSON, no markdown:
  {{"overlap": false, "reasoning": "..."}}
  {{"overlap": true, "direction": "left" | "right", "shift_m": 0.10, "reasoning": "..."}}

direction "left"/"right" is from the camera's perspective looking at the sofa.
shift_m is the suggested move in metres (0.05–0.25).
"""

_REORDER_PROMPT = """\
You are an interior design expert reviewing {n} pillows/cushions placed on a sofa.

Image 1: the 3D render — {n} pillows left-to-right, indexed 0 (leftmost) to {n1} (rightmost).
Image 2: the reference photo showing the desired arrangement.

STEP 1 — Describe each pillow in Image 1 from LEFT to RIGHT:
  For each index 0..{n1}: note its COLOR and SHAPE (rectangular/wider vs square/rounder).

STEP 2 — Describe each pillow in Image 2 from LEFT to RIGHT:
  For each position: note its COLOR and SHAPE.

STEP 3 — Match and check ordering:
  - DIFFERENT-colored pillows: position is the priority — compare left-to-right color sequence.
  - SAME-colored pillows: also compare shape (rectangular vs square) to check if they are swapped.
  Only suggest a swap if you are confident it improves the match to Image 2.

STEP 4 — Suggest swaps (0-based indices into Image 1's left-to-right order):
  List ALL swaps needed. You may suggest up to {n1} swaps.

Return ONLY valid JSON, no markdown:
  If no swaps needed: {{"correct": true, "swaps": [], "reasoning": "..."}}
  If swaps needed:    {{"correct": false, "swaps": [[i, j], ...], "reasoning": "..."}}

Each [i, j] means: swap the positions of pillow i and pillow j in Image 1.
Apply swaps in order (each swap uses the indices AFTER all previous swaps).
"""


def _vlm_arm_rest_check(
    render_img: "Image.Image",
    ref_photo: "Image.Image",
    phrase: str,
) -> "dict | None":
    """Check if a pillow overlaps the sofa arm rest.  Returns parsed JSON or None."""
    content: list[dict] = [
        {"type": "text", "text": _ARM_REST_PROMPT.format(phrase=phrase)},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_img)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_photo)}"}},
    ]
    return _vlm_call_json(content, max_tokens=150)


_PILLOW_POS_REFINE_PROMPT = """\
You are checking whether a pillow is properly placed ON a sofa.

Image 1: a 3D render of the room.  Find the pillow on the sofa — focus on
this specific pillow.

Examine the pillow's position relative to its sofa:
  • Is the pillow's full body sitting on top of the sofa's seat / cushions?
  • Or does any part of the pillow stick out past the sofa's edge —
    floating off the side, hanging past the front, or behind the sofa?

If the pillow IS fully on the sofa, reply correct.

If part of the pillow is OFF the sofa, suggest a horizontal nudge IN THE
SOFA'S OWN LOCAL FRAME:
  • "back"   = move the pillow toward the sofa's BACK-REST (deeper)
  • "front"  = move the pillow toward the sofa's FRONT (closer to where a
               person sits, away from back-rest)
  • "left"   = move the pillow toward the sofa's LEFT armrest (when facing
               the sofa from the front, where someone would sit)
  • "right"  = move the pillow toward the sofa's RIGHT armrest

Pick the SINGLE direction that would best move the pillow back onto the
sofa.  Magnitude is how far to move, in centimetres (small = 5, big = 20).

Return ONLY valid JSON, no markdown:
{{"on_sofa": true,  "reasoning": "..."}}
or
{{"on_sofa": false, "direction": "back"|"front"|"left"|"right",
  "magnitude_cm": <int 3..30>, "reasoning": "..."}}
"""


def _vlm_pillow_pos_refine(
    render_img: "Image.Image",
    phrase: str,
) -> "dict | None":
    """Ask the VLM whether a pillow is fully on its sofa, and which sofa-
    local direction to nudge if not.  Returns parsed JSON or None."""
    content: list[dict] = [
        {"type": "text", "text": _PILLOW_POS_REFINE_PROMPT},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_img)}"}},
    ]
    return _vlm_call_json(content, max_tokens=200)


def _vlm_reorder_group(
    render_img: "Image.Image",
    ref_photo: "Image.Image",
    phrase: str,
    n: int,
) -> "dict | None":
    """Compare a group of same-type placed objects vs reference. Returns multi-swap suggestion or None."""
    prompt = _REORDER_PROMPT.format(phrase=phrase, n=n, n1=n - 1)
    content: list[dict] = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_img)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_photo)}"}},
    ]
    return _vlm_call_json(content, max_tokens=400)


# ── Semantic position refinement: discrete slot picker ──────────────────────
#
# After geometric placement, the object can land in the wrong "slot" of its
# supporting furniture: e.g. a lamp ends up on the right of the desk
# instead of the left, or a pillow lands on the right armrest when the
# reference shows it centred.  Rather than asking the VLM for pixel
# offsets (which oscillates — see the removed `_vlm_pillow_pos_refine`),
# we let the VLM pick a DISCRETE slot in the furniture's own local frame
# and snap the object there.  Three sides × three depths = 9 slots.
_POS_REFINE_SEMANTIC_PROMPT = """\
You are checking the LOCATION of "{phrase}" on its supporting {furn_type}.

Image 1: a reference photo crop showing how "{phrase}" was originally \
placed on the {furn_type}.
Image 2: the current 3D render of the room.  Find the {phrase} on the \
{furn_type} in Image 2 — focus on this specific instance.

Compare the {phrase}'s position on the {furn_type} to Image 1, using \
WHAT YOU SEE IN THE IMAGES (camera / viewer perspective):
  - side  ∈ left | center | right
      "left"  = appears on the LEFT half of the {furn_type} in the image
      "right" = appears on the RIGHT half of the {furn_type} in the image
      "center" = horizontally centred on the {furn_type}
  - depth ∈ back | middle | front
      "front"  = closer to the viewer (front edge of the {furn_type} in the image)
      "back"   = farther from the viewer (back edge of the {furn_type} —
                 typically nearer the wall or back-rest)
      "middle" = between the two depth-wise

These are CAMERA / IMAGE directions, not the furniture's own internal \
frame.  Use Image 1 (reference) and Image 2 (render) directly: pick the \
slot the {phrase} OCCUPIES IN THE IMAGE in the reference.

If the {phrase} in Image 2 already sits in roughly the same image-space \
slot as in Image 1, return correct.

Otherwise return the slot the {phrase} SHOULD occupy in the rendered \
image to match the reference.

Return ONLY valid JSON, no markdown:
{{"correct": true,  "reasoning": "..."}}
or
{{"correct": false, "side": "left|center|right", "depth": "back|middle|front", "reasoning": "..."}}
"""


def _vlm_pos_refine_semantic(
    ref_crop: "Image.Image",
    render_img: "Image.Image",
    phrase: str,
    furn_type: str,
) -> "dict | None":
    """Ask the VLM which discrete slot ``phrase`` should occupy on its
    supporting furniture.  Returns parsed JSON or None."""
    content: list[dict] = [
        {"type": "text",
         "text": _POS_REFINE_SEMANTIC_PROMPT.format(
             phrase=phrase, furn_type=furn_type)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_crop)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_img)}"}},
    ]
    return _vlm_call_json(content, max_tokens=300)


_VERIFY_ORIENTATION_PROMPT = """\
You are an interior design expert reviewing a room render.

Image 1 (reference crop): the original photo crop showing "{phrase}" as it \
appears in the real room — use this as the ground truth for orientation.
Image 2 (render): the 3D render showing "{phrase}" as it was placed.

Focus ONLY on the horizontal yaw (around the vertical axis) of "{phrase}":
- Does the object sit with the SAME long-axis direction in both images?
- For elongated rectangular items (books, trays, boxes, laptops): check whether
  the LONGER side faces the camera in the reference.  If the render shows the
  shorter side toward the camera, that's a 90° error.
- For directional items with a clear front (monitor, keyboard, lamp): a
  180-degree flip is the most common error — the screen/head points the
  opposite way from the reference.
- For symmetric items (lamps with round shades, vases, bowls): if the silhouette
  looks identical rotated 180°, mark correct=true.

Return ONLY valid JSON, no markdown.  Pick ONE action only.
  {{"correct": true,  "reasoning": "..."}}               ← orientation matches
  {{"correct": false, "action": "flip_180",  "reasoning": "..."}}   ← 180° about Y
  {{"correct": false, "action": "rotate_90_cw",  "reasoning": "..."}}  ← 90° clockwise about Y (seen from above)
  {{"correct": false, "action": "rotate_90_ccw", "reasoning": "..."}}  ← 90° counter-clockwise about Y
"""


def _vlm_verify_orientation(
    ref_crop: "Image.Image",
    render_img: "Image.Image",
    phrase: str,
) -> str | None:
    """Return 'correct', 'flip_180', 'rotate_90_cw', 'rotate_90_ccw', or None.

    The VLM visually compares the rendered decoration to the reference crop
    and picks ONE yaw correction.  Book/tray/laptop-style items usually need
    a 90° rotation when their long axis in the render doesn't match the ref.
    """
    content: list[dict] = [
        {"type": "text",
         "text": _VERIFY_ORIENTATION_PROMPT.format(phrase=phrase)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_crop)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_img)}"}},
    ]
    # Bumped from 150 — the VLM's reasoning often exceeds the short budget,
    # truncating the JSON and forcing us to discard the entire response ("no
    # JSON in: {..."), so valid corrections were being silently dropped.
    result = _vlm_call_json(content, max_tokens=350)
    if result is None:
        return None
    if result.get("correct"):
        print(f"    [verify_orient] orientation correct. {result.get('reasoning', '')}")
        return "correct"
    action = str(result.get("action", "")).strip().lower()
    reason = result.get("reasoning", "")
    # Normalise a few obvious synonyms the VLM tends to produce.
    _syn = {
        "flip": "flip_180", "flip_vertical": "flip_180", "rotate_180": "flip_180",
        "rotate_180_cw": "flip_180", "rotate_180_ccw": "flip_180",
        "cw90": "rotate_90_cw", "rotate_cw_90": "rotate_90_cw", "clockwise_90": "rotate_90_cw",
        "ccw90": "rotate_90_ccw", "rotate_ccw_90": "rotate_90_ccw", "counter_clockwise_90": "rotate_90_ccw",
    }
    action = _syn.get(action, action)
    if action not in {"flip_180", "rotate_90_cw", "rotate_90_ccw"}:
        # Unknown action — default to flip_180 (backward-compatible) so the
        # caller still gets a usable value.
        print(f"    [verify_orient] orientation wrong → {action or '(unknown)'} "
              f"(normalising to flip_180). {reason}")
        return "flip_180"
    print(f"    [verify_orient] orientation wrong → {action}. {reason}")
    return action


_VERIFY_SIZE_PROMPT = """\
You are checking whether the object "{phrase}" in a 3D render is correctly \
SIZED compared to a reference photo of the same room.

Image 1: the reference photo of the actual room.  Find "{phrase}" in this \
photo and note how big it appears relative to the furniture / surface it \
sits on and the nearby objects.
Image 2: the current 3D render of the same room.  Find "{phrase}" here and \
note how big it appears relative to its surroundings.

Compare the relative size of "{phrase}" in BOTH images, using the \
furniture (sofas, coffee tables) and other objects as size anchors — they \
should be roughly equivalent across the two images, and "{phrase}" should \
take up about the same proportion of its supporting surface in both.

  - Looks about the same proportion → correct.
  - In Image 2 it occupies a much LARGER portion of its surface / dwarfs \
its neighbours compared to Image 1 → action="shrink".
  - In Image 2 it occupies a much SMALLER portion of its surface / is \
dwarfed compared to Image 1 → action="enlarge".

ONE common failure to watch for: a segmentation mask that captured \
TWO separate instances stacked side-by-side (e.g. two book piles) — \
the system then scales a SINGLE mesh to match the full mask area, so \
the one rendered copy ends up about 2× too big and visually fills the \
whole region where the photo had two distinct piles.  If you see this, \
return shrink.

Return ONLY valid JSON, no markdown:
  {{"correct": true,  "reasoning": "..."}}
  {{"correct": false, "action": "shrink",  "reasoning": "..."}}
  {{"correct": false, "action": "enlarge", "reasoning": "..."}}
"""


def _vlm_verify_size(
    ref_crop: "Image.Image",
    render_img: "Image.Image",
    phrase: str,
) -> str | None:
    """Return 'correct', 'shrink', 'enlarge', or None.

    Visually compares the rendered decoration to the reference crop and
    decides whether the rendered size looks correct.
    """
    content: list[dict] = [
        {"type": "text",
         "text": _VERIFY_SIZE_PROMPT.format(phrase=phrase)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_crop)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_img)}"}},
    ]
    result = _vlm_call_json(content, max_tokens=300)
    if result is None:
        return None
    if result.get("correct"):
        print(f"    [verify_size] size correct. "
              f"{str(result.get('reasoning', ''))[:120]}")
        return "correct"
    action = str(result.get("action", "")).strip().lower()
    reason = result.get("reasoning", "")
    _syn = {
        "smaller": "shrink", "reduce": "shrink", "downsize": "shrink",
        "scale_down": "shrink",
        "bigger": "enlarge", "increase": "enlarge", "upsize": "enlarge",
        "scale_up": "enlarge",
    }
    action = _syn.get(action, action)
    if action not in {"shrink", "enlarge"}:
        print(f"    [verify_size] size wrong → unknown action "
              f"'{action}' — skipping. {reason}")
        return None
    print(f"    [verify_size] size wrong → {action}. {str(reason)[:120]}")
    return action


# ── 4-yaw-candidate picker for orientation correction ────────────────────────
#
# The iterative verify_orient loop (single VLM call → apply correction →
# re-render → repeat) compounds errors: if iter 0 picks the wrong direction,
# iter 1's correction is interpreted relative to that wrong state, and after
# 3 iters we can swing back to the original orientation having achieved
# nothing.  Lamps and other rotationally-ambiguous objects suffer especially.
#
# Single-shot alternative: render the object at all 4 yaw rotations
# (0°, 90° CCW, 180°, 90° CW) from its current orientation, stitch the four
# scene renders into a 2×2 grid, and ask the VLM "which panel's orientation
# matches the reference?".  One VLM call, one chosen rotation, no oscillation.
#
# After applying the picked yaw, the caller may run ONE additional
# iterative-correction pass for fine adjustment (in case the true correction
# wasn't a clean multiple of 90°).

_PICK_YAW_PROMPT = """\
You are choosing the correct orientation for a "{phrase}" placed in a 3D scene.

Image 1 (reference): a crop of the "{phrase}" from the original room photo.
  Note which way the object faces (e.g. lamp head pointing left vs. right;
  monitor screen facing forward vs. backward).

Image 2 (candidates): a 2×2 grid of full-scene renders, each showing the SAME
3D model placed at the same position but at FOUR different yaw rotations
about the vertical axis:
  A (top-left)     — current orientation (0° rotation)
  B (top-right)    — rotated 90° counter-clockwise about Y (seen from above)
  C (bottom-left)  — rotated 180° about Y (a left-right flip in scene)
  D (bottom-right) — rotated 90° clockwise about Y

Pick the SINGLE panel where the "{phrase}"'s orientation BEST matches the
reference (Image 1).  Look at distinctive directional features — the lamp's
arm/head, a screen's face, a chair's backrest, a book's spine.

If the object is rotationally symmetric (a vase, a centred floor lamp shade,
plain pillows with no asymmetric features), all four panels will look the
same — in that case pick "A" (no rotation).

Return ONLY valid JSON, no markdown:
{{
  "best": "A" or "B" or "C" or "D",
  "reasoning": "one sentence: which directional feature matches between the reference and the picked panel"
}}
"""


def _vlm_pick_yaw_candidate(
    ref_crop: "Image.Image",
    candidates_grid: "Image.Image",
    phrase: str,
) -> "int | None":
    """Ask VLM to pick the best of 4 yaw candidates.  Returns 0/1/2/3
    (= A/B/C/D = 0°/90°CCW/180°/90°CW), or None on failure."""
    content = [
        {"type": "text", "text": _PICK_YAW_PROMPT.format(phrase=phrase)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_crop)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(candidates_grid)}"}},
    ]
    result = _vlm_call_json(content, max_tokens=200)
    if result is None:
        return None
    pick = str(result.get("best", "")).upper().strip()
    reason = result.get("reasoning", "")
    if pick not in ("A", "B", "C", "D"):
        print(f"    [yaw_pick] unparseable best='{pick}' ({reason})")
        return None
    print(f"    [yaw_pick] best={pick}  ({reason})")
    return {"A": 0, "B": 1, "C": 2, "D": 3}[pick]


_COMPARE_TWO_PROMPT = """\
You are picking which of TWO rendered orientations of a "{phrase}" matches
the reference image more closely.

Image 1 (reference): a crop of the "{phrase}" from the original room photo.
Image 2 (candidates): a horizontal panel with TWO renders side-by-side.
  A (left)  — first candidate
  B (right) — second candidate

Look at distinctive directional features (lamp head/arm direction, book
spine orientation, monitor screen direction, chair backrest).  Pick the
panel where those features point in the SAME direction as the reference.

Return ONLY valid JSON, no markdown:
{{
  "best": "A" or "B",
  "reasoning": "one sentence: which directional feature matches"
}}
"""


def _vlm_compare_two_renders(
    ref_crop: "Image.Image",
    render_a: "Image.Image",
    render_b: "Image.Image",
    phrase: str,
) -> "str | None":
    """Pairwise VLM comparison: which render is closer to the reference?

    Returns 'A' or 'B' (winner), or None on parse failure.  Side-by-side
    pairwise judgments are more reliable for VLMs than 4-way tournaments
    or single-view "is this correct?" checks — fewer simultaneous panels
    to disambiguate.
    """
    # Build A|B horizontal panel.
    _PANEL_PX = 512
    from PIL import Image as _PI, ImageDraw as _PD, ImageFont as _PF
    try:
        _font = _PF.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        _font = _PF.load_default()
    _panel = _PI.new("RGB", (_PANEL_PX * 2, _PANEL_PX), (200, 200, 200))
    for _i, (_img, _tag) in enumerate([(render_a, "A"), (render_b, "B")]):
        _r = _img.copy().resize((_PANEL_PX, _PANEL_PX))
        _d = _PD.Draw(_r)
        _d.rectangle([(0, 0), (_PANEL_PX, 44)], fill=(0, 0, 0))
        _d.text((8, 6), _tag, fill=(255, 200, 0), font=_font)
        _panel.paste(_r, (_i * _PANEL_PX, 0))
    content = [
        {"type": "text",
         "text": _COMPARE_TWO_PROMPT.format(phrase=phrase)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_crop)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(_panel)}"}},
    ]
    result = _vlm_call_json(content, max_tokens=200)
    if result is None:
        return None
    pick = str(result.get("best", "")).upper().strip()
    reason = result.get("reasoning", "")
    if pick not in ("A", "B"):
        return None
    print(f"    [compare] {pick} wins  ({reason})")
    return pick


_COUNT_INSTANCES_PROMPT = """\
You are an interior design expert.

The reference image below shows "{phrase}" — possibly more than one.  Count
the number of DISTINCT PILES / STACKS / COPIES visible.

Examples:
  - two separate stacks of books side-by-side → count = 2
  - a single stack of books, even if it contains multiple individual books
    piled on top of each other → count = 1 (it's one stack)
  - three pillows lined up on a sofa — each one separately visible → count = 3
  - one big continuous row of pillows with no gap → count = 1

Return ONLY valid JSON:
  {{"count": <1 | 2 | 3 | 4>, "reasoning": "one short sentence"}}

If the image doesn't clearly show more than one, return count=1.  Only return
count > 1 when the separate piles/stacks/copies are DISTINCTLY visible.
"""


def _vlm_count_instances(
    ref_crop: "Image.Image",
    phrase: str,
) -> int:
    """Ask the VLM how many distinct piles / stacks / instances of {phrase}
    are visible in the reference crop.  Returns 1 on failure or if the VLM
    isn't confident about a multi-pile interpretation.
    """
    content: list[dict] = [
        {"type": "text",
         "text": _COUNT_INSTANCES_PROMPT.format(phrase=phrase)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_crop)}"}},
    ]
    result = _vlm_call_json(content, max_tokens=200)
    if result is None:
        return 1
    try:
        n = int(result.get("count", 1))
    except Exception:
        return 1
    n = max(1, min(4, n))
    reason = result.get("reasoning", "")
    print(f"    [count_instances] {phrase}: count={n}. {reason}")
    return n


def _overlay_mask_on_ref(
    ref_img: "Image.Image",
    mask_path: "Path | None",
    box_px: "list[int] | None" = None,
    orig_ref_w: float = 3000.0,
) -> "Image.Image":
    """Composite the segmentation mask (red tint) onto the reference photo.

    The mask lives in original-reference-photo coordinate space (orig_ref_w pixels wide).
    ref_img may be a different resolution (e.g. 625px manhattan_reference.png).
    If mask_path is None or fails to load, falls back to a scaled bbox highlight.
    """
    from PIL import ImageDraw
    import numpy as _np_omr

    ref_w, ref_h = ref_img.size
    scale = ref_w / orig_ref_w  # scale from orig photo space to ref_img space

    base = ref_img.copy().convert("RGBA")

    loaded_mask = False
    if mask_path is not None and Path(mask_path).exists():
        try:
            mask_img = Image.open(str(mask_path)).convert("L")
            # Resize mask from original photo resolution to ref_img resolution
            mask_img = mask_img.resize((ref_w, ref_h), Image.NEAREST)
            mask_arr = _np_omr.array(mask_img, dtype=_np_omr.float32) / 255.0
            overlay = _np_omr.zeros((ref_h, ref_w, 4), dtype=_np_omr.uint8)
            overlay[..., 0] = 255           # R
            overlay[..., 3] = (mask_arr * 200).astype(_np_omr.uint8)
            base.alpha_composite(Image.fromarray(overlay, mode="RGBA"))
            loaded_mask = True
        except Exception:
            pass

    if not loaded_mask and box_px is not None:
        # Fallback: draw scaled bbox
        x1, y1, x2, y2 = [int(v * scale) for v in box_px]
        x1, x2 = max(0, x1), min(ref_w - 1, x2)
        y1, y2 = max(0, y1), min(ref_h - 1, y2)
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)
        draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 80), outline=(255, 0, 0, 255))
        base.alpha_composite(ov)

    return base.convert("RGB")


def _highlight_bbox(
    scene_img: Image.Image,
    box_px: list[int],
    scale: float = 1.0,
) -> Image.Image:
    """Draw a semi-transparent red overlay + red border over box_px on scene_img.

    scale: if scene_img was downsampled, multiply box_px coords by this factor.
    """
    from PIL import ImageDraw
    x1, y1, x2, y2 = [int(v * scale) for v in box_px]
    # Clamp to image bounds
    W, H = scene_img.size
    x1, x2 = max(0, x1), min(W - 1, x2)
    y1, y2 = max(0, y1), min(H - 1, y2)

    annotated = scene_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 80))   # semi-transparent fill
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=4)  # solid border
    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
    return annotated


def _annotate_furniture_labels(
    scene_img: Image.Image,
    furn_list: list[dict],
    camera: dict,
) -> Image.Image:
    """Draw furniture index + type labels onto scene_img at each piece's projected 2D position."""
    from PIL import ImageDraw, ImageFont
    from object_placement.wall_mounted.wall_mounted_object_placement import (
        _camera_axes, _project_vertex,
    )

    W, H = scene_img.size
    cam_pos  = np.array(camera["position_m"], dtype=np.float64)
    look_at  = np.array(camera["look_at_m"],  dtype=np.float64)
    up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    hfov = float(camera["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    annotated = scene_img.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for f in furn_list:
        if f.get("type") in ("carpet",) or f.get("index", -1) < 0:
            continue
        if "position_m" not in f:
            continue
        pos3d = np.array(f["position_m"], dtype=np.float64)
        eff_h = f.get("eff_h", f.get("size_m", {}).get("height_m", 0.5))
        label_pos = pos3d.copy()
        label_pos[1] += float(eff_h) * 0.8
        px, py, zc = _project_vertex(label_pos, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
        if zc <= 0.01 or not (0 <= px < W) or not (0 <= py < H):
            continue
        label = f"[{f['index']}] {f['type']}"
        for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            draw.text((px+dx, py+dy), label, fill=(0,0,0), font=font)
        draw.text((px, py), label, fill=(255,255,255), font=font)

    return annotated


def _vlm_call_json(content: list[dict], max_tokens: int = 200) -> dict | None:
    """Shared VLM JSON call with thinking-tag stripping."""
    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group())
        print(f"    [vlm] no JSON in: {raw[:150]}")
    except Exception as e:
        print(f"    [vlm] FAILED: {e}")
    return None


def _describe_furniture_for_vlm(
    placeable: list[dict],
    camera: "dict | None" = None,
) -> str:
    """Build a rich per-item descriptor for VLM prompts.

    For each piece: include size (W×D in m) and, when a camera is available,
    distance to the camera.  When MULTIPLE pieces of the same type exist, add
    relative hints — "LARGEST of N", "closest to camera", "farthest from camera"
    — so the VLM can disambiguate same-type instances by visual cue instead of
    just "wall=left" (which is often wrong or redundant).
    """
    lines: list[str] = []
    # Pre-compute per-furniture metrics.
    cam_pos = None
    if camera is not None and "position_m" in camera:
        try:
            cam_pos = np.array(camera["position_m"], dtype=np.float64)
        except Exception:
            cam_pos = None

    # Group by type so we can compute relative flags.
    by_type: "dict[str, list[tuple[dict, float, float]]]" = {}
    for f in placeable:
        t = f.get("type", "?")
        sz = f.get("size_m", {}) or {}
        w = float(sz.get("width_m", 0.0))
        d = float(sz.get("depth_m", 0.0))
        area = max(w * d, 0.0)
        pos = f.get("position_m")
        if cam_pos is not None and pos is not None:
            try:
                p = np.array(pos, dtype=np.float64)
                dist = float(np.linalg.norm(p[[0, 2]] - cam_pos[[0, 2]]))
            except Exception:
                dist = float("nan")
        else:
            dist = float("nan")
        by_type.setdefault(t, []).append((f, area, dist))

    for t, items in by_type.items():
        if len(items) == 1:
            f, area, dist = items[0]
            sz = f.get("size_m", {}) or {}
            w = float(sz.get("width_m", 0.0))
            d = float(sz.get("depth_m", 0.0))
            parts = [f"  [{f['index']}] {t}", f"size={w:.2f}×{d:.2f}m"]
            if f.get("wall_affinity") and f.get("wall_affinity") != "?":
                parts.append(f"wall={f['wall_affinity']}")
            if not np.isnan(dist):
                parts.append(f"camera_dist={dist:.1f}m")
            lines.append(" ".join(parts))
            continue

        # Multiple same-type: add relative descriptors
        areas = [a for _, a, _ in items]
        dists = [d for _, _, d in items]
        max_area = max(areas) if areas else 0.0
        min_area = min(areas) if areas else 0.0
        if cam_pos is not None and not any(np.isnan(d) for d in dists):
            min_dist = min(dists)
            max_dist = max(dists)
        else:
            min_dist = max_dist = None
        for f, area, dist in items:
            sz = f.get("size_m", {}) or {}
            w = float(sz.get("width_m", 0.0))
            d = float(sz.get("depth_m", 0.0))
            parts = [f"  [{f['index']}] {t}", f"size={w:.2f}×{d:.2f}m"]
            if f.get("wall_affinity") and f.get("wall_affinity") != "?":
                parts.append(f"wall={f['wall_affinity']}")
            if not np.isnan(dist):
                parts.append(f"camera_dist={dist:.1f}m")
            # Relative size hint (only when sizes differ)
            if max_area - min_area > 0.05:
                if area == max_area:
                    parts.append(f"(LARGEST {t})")
                elif area == min_area:
                    parts.append(f"(smallest {t})")
            # Relative distance hint
            if (min_dist is not None and max_dist is not None
                    and abs(max_dist - min_dist) > 0.30):
                if dist == min_dist:
                    parts.append("(CLOSEST to camera)")
                elif dist == max_dist:
                    parts.append("(farthest from camera)")
            lines.append(" ".join(parts))

    return "\n".join(lines)


def _vlm_select_furniture(
    combined_img: Image.Image,
    phrase: str,
    furn_list: list[dict],
    ref_photo: Image.Image | None = None,
    box_px: list[int] | None = None,
    bbox_scale: float = 1.0,
    camera: "dict | None" = None,
    already_used: "set[int] | None" = None,
) -> int | None:
    """Ask VLM which furniture the decoration is on.

    When ref_photo is provided (two-image mode):
      Image 1 = reference photo with decoration highlighted in red
      Image 2 = labeled furniture render
    Otherwise falls back to single-image mode (labeled render + highlight).

    `already_used` is a set of furn_idxs that already received a same-phrase
    decoration earlier in this run.  When the reference photo shows multiple
    instances of the same item (e.g. one lamp on each cabinet), the VLM is
    nudged to prefer an UNOCCUPIED same-type surface for subsequent instances
    instead of piling them on the most prominent piece.

    Returns furniture index, -1 for floor, or None on failure.
    """
    placeable = [f for f in furn_list
                 if f.get("type") not in ("carpet",) and f.get("index", -1) >= 0]
    furn_desc = _describe_furniture_for_vlm(placeable, camera=camera)
    furn_desc += "\n  [-1] floor (object stands directly on the floor)"

    # Annotate already-used surfaces so the VLM can spread duplicates.
    if already_used:
        _used_str = ", ".join(f"[{i}]" for i in sorted(already_used))
        furn_desc += (
            f"\n\nNOTE: a previous '{phrase}' has already been placed on "
            f"{_used_str}.  If the reference photo shows MULTIPLE '{phrase}' "
            f"instances spread across DIFFERENT surfaces of the same type "
            f"(e.g. one lamp on each of two cabinets), pick an UNUSED same-"
            f"type surface for THIS instance.  If the reference clearly shows "
            f"all instances on the SAME surface, you may pick a used surface."
        )

    if ref_photo is not None and box_px is not None:
        # Two-image mode: highlight on reference photo + labeled render
        ref_highlighted = _highlight_bbox(ref_photo, box_px, scale=bbox_scale)
        content: list[dict] = [
            {"type": "text",
             "text": _SELECT_FURN_PROMPT.format(phrase=phrase, furn_list=furn_desc)},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_highlighted)}"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_encode_pil(combined_img)}"}},
        ]
    else:
        content = [
            {"type": "text",
             "text": _SELECT_FURN_PROMPT.format(phrase=phrase, furn_list=furn_desc)},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_encode_pil(combined_img)}"}},
        ]

    valid_indices = {f["index"] for f in placeable} | {-1}
    result = _vlm_call_json(content, max_tokens=250)
    if result is not None:
        raw = result.get("furniture_index", -999)
        if isinstance(raw, list):
            raw = raw[0] if raw else -999
        idx = int(raw)
        if idx not in valid_indices:
            print(f"    [select_furn] invalid index {idx} — ignoring")
            return None
        print(f"    [select_furn] → [{idx}]  {result.get('reasoning', '')}")
        return idx
    return None


_PILLOW_SEQUENCE_PROMPT = """\
Look at this room photo.

Find the sofa or chair that has the most pillows/cushions on it. \
List every visible pillow from LEFT to RIGHT as they appear on that sofa.

For each pillow return:
- "color": dominant color — "grey", "dark_grey", "white", "orange", "brown", \
"beige", "blue", or "other"
- "shape": "square" (width ≈ height) or "rectangular" (clearly wider than tall)

Return ONLY valid JSON, no markdown:
{{
  "total": <integer>,
  "sequence": [
    {{"color": "grey", "shape": "square"}},
    {{"color": "orange", "shape": "rectangular"}},
    ...
  ]
}}
"""


def _classify_seg_color(seg: dict, inpaint_dir: "Path") -> str:
    """Return coarse color label for a segment from its inpaint PNG mean color."""
    inpaint_file = seg.get("inpaint_file")
    if not inpaint_file:
        return "unknown"
    try:
        img = Image.open(str(inpaint_dir / inpaint_file)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        mask = ~((arr[:, :, 0] > 230) & (arr[:, :, 1] > 230) & (arr[:, :, 2] > 230))
        sub = arr[mask] if mask.sum() > 100 else arr.reshape(-1, 3)
        r, g, b = sub[:, 0].mean(), sub[:, 1].mean(), sub[:, 2].mean()
        if r - b > 25:
            return "orange"
        elif max(r, g, b) > 210:
            return "white"
        elif max(r, g, b) < 70:
            return "dark_grey"
        else:
            return "grey"
    except Exception:
        return "unknown"


def _classify_seg_shape(seg: dict) -> str:
    """Return shape label from bbox aspect ratio."""
    box = seg.get("box_px", [0, 0, 1, 1])
    w = max(box[2] - box[0], 1)
    h = max(box[3] - box[1], 1)
    return "rectangular" if w / h > 1.4 else "square"


def _strip_color_adj(phrase: str) -> str:
    """Strip leading colour / size adjectives so the noun phrase remains.

    'yellow throw pillow' → 'throw pillow'
    'small succulent plant' → 'succulent plant'
    'black and white patterned throw pillow' → 'throw pillow'
    """
    _ADJ = {
        "yellow", "blue", "red", "green", "black", "white", "gray", "grey",
        "brown", "beige", "pink", "purple", "orange", "navy", "tan", "cream",
        "ivory", "gold", "silver", "dark", "light", "pale", "deep",
        "small", "large", "big", "tiny", "tall", "short", "round", "square",
        "rectangular", "circular", "oval", "patterned", "striped", "plain",
        "solid", "and",
    }
    words = phrase.lower().strip().split()
    while words and words[0] in _ADJ:
        words = words[1:]
    # Also strip any "X and Y" leading colour combos (e.g. "black and white")
    while len(words) >= 3 and words[0] in _ADJ and words[1] == "and" and words[2] in _ADJ:
        words = words[3:]
    return " ".join(words).strip() or phrase.lower().strip()


def _seg_mean_color(seg: dict, inpaint_dir: "Path") -> "tuple[float, float, float] | None":
    """Mean (R,G,B) of the segment's inpaint image excluding near-white padding.
    Returns None on any read/parse failure."""
    inpaint_file = seg.get("inpaint_file")
    if not inpaint_file:
        return None
    p = Path(inpaint_dir) / inpaint_file
    if not p.exists():
        return None
    try:
        img = Image.open(str(p))
        if img.mode in ("RGBA", "LA"):
            arr = np.array(img)
            alpha = arr[..., -1]
            if int(alpha.max()) > 0:
                rgb = arr[..., :3].astype(np.float32)
                m = alpha > 127
                if m.sum() > 50:
                    return tuple(float(v) for v in rgb[m].mean(axis=0))
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
        # Skip near-white background (corners) when present.
        m = ~((arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235))
        sub = arr[m] if m.sum() > 100 else arr.reshape(-1, 3)
        return tuple(float(v) for v in sub.mean(axis=0))
    except Exception:
        return None


def _pick_substitute_glb(missing_seg: dict, all_segs: list,
                         inpaint_dir: "Path", out_dir: "Path",
                         color_thresh: float = 80.0,
                         ) -> "dict | None":
    """When `missing_seg`'s GLB doesn't exist on disk, search for an already-
    generated decoration GLB whose noun phrase matches and whose mean colour
    is close, and return a dict {glb_file, donor_seg_index, delta_color}.

    Strategy:
      1. Strip leading colour/size adjectives from both phrases — the result
         is the "noun phrase" (e.g. 'yellow throw pillow' → 'throw pillow').
      2. Candidates are segments whose noun phrase matches AND whose
         glb_file exists on disk AND has a non-empty inpaint image.
      3. Compute mean RGB of `missing_seg` and each candidate.  Pick the
         candidate with smallest Euclidean distance in RGB space.
      4. If the closest distance > `color_thresh`, return None (no good
         match — better to skip than substitute with a wildly different
         pillow).  Same-noun pillows are usually within ~50; > 80 means
         the candidate is the wrong colour family.

    Returns None when no usable substitute is found.
    """
    target_noun  = _strip_color_adj(missing_seg.get("phrase", ""))
    target_color = _seg_mean_color(missing_seg, inpaint_dir)
    if not target_noun:
        return None

    best = None
    best_d = float("inf")
    for cand in all_segs:
        if cand is missing_seg:
            continue
        glb_rel = cand.get("glb_file")
        if not glb_rel:
            continue
        glb_abs = Path(out_dir) / glb_rel
        if not glb_abs.exists():
            continue
        cand_noun = _strip_color_adj(cand.get("phrase", ""))
        if cand_noun != target_noun:
            continue
        cand_color = _seg_mean_color(cand, inpaint_dir)
        if target_color is None or cand_color is None:
            # Phrase-noun match without colour info — accept with a moderate
            # distance penalty so a colour-comparable candidate wins.
            d = 50.0
        else:
            d = float(np.linalg.norm(np.array(target_color) - np.array(cand_color)))
        if d < best_d:
            best_d = d
            best = {
                "glb_file":         glb_rel,
                "donor_seg_index":  cand.get("seg_index", -1),
                "donor_phrase":     cand.get("phrase", ""),
                "delta_color":      d,
            }
    if best is None or best_d > color_thresh:
        return None
    return best


def _vlm_verify_placement(
    combined_img: Image.Image,
    render_img: Image.Image,
    phrase: str,
    furn_list: list[dict],
    current_furn_idx: int,
    ref_photo: "Image.Image | None" = None,
    camera: "dict | None" = None,
) -> int | None:
    """Post-placement check: compare render against the mask-highlighted furniture render.

    combined_img: labeled furniture render with bbox highlight — shows where object should be.
    render_img: the step render showing where the object was actually placed.
    ref_photo: original reference photo for silhouette/surface-type grounding.
    Returns the correct furniture_index, or None on failure.
    """
    placeable = [f for f in furn_list
                 if f.get("type") not in ("carpet",) and f.get("index", -1) >= 0]
    furn_desc = _describe_furniture_for_vlm(placeable, camera=camera)
    furn_desc += "\n  [-1] floor"

    content: list[dict] = [
        {"type": "text",
         "text": _VERIFY_PLACEMENT_PROMPT.format(phrase=phrase, furn_list=furn_desc,
                                                  current_furn_idx=current_furn_idx)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(combined_img)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(render_img)}"}},
    ]
    if ref_photo is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_encode_pil(ref_photo)}"},
        })

    valid_indices = {f["index"] for f in placeable} | {-1}
    result = _vlm_call_json(content, max_tokens=250)
    if result is not None:
        correct = result.get("correct", True)
        reason  = result.get("reasoning", "")
        if correct:
            print(f"    [verify] placement correct on [{current_furn_idx}]. {reason}")
            return current_furn_idx          # unchanged
        # VLM says wrong furniture
        raw_idx = result.get("furniture_index", current_furn_idx)
        if isinstance(raw_idx, list):
            raw_idx = raw_idx[0] if raw_idx else current_furn_idx
        idx = int(raw_idx)
        if idx not in valid_indices:
            print(f"    [verify] suggested invalid index {idx} — keeping {current_furn_idx}")
            return current_furn_idx
        if idx == current_furn_idx:
            # VLM returned the same index despite saying correct=false.  This
            # is a clear "this placement is wrong" signal but the VLM didn't
            # provide a useful alternative — return None so the caller can
            # run its geo-fallback (project every same-type surface and pick
            # the one nearest the bbox).  Common case: lamp on the WRONG
            # coffee table; VLM detects but can't pin the right index.
            print(f"    [verify] VLM said wrong but returned same index [{idx}] — "
                  f"triggering geo-fallback")
            return None
        print(f"    [verify] CORRECTION → [{idx}]  {reason}")
        return idx
    return None


# ── VLM: size + placement ────────────────────────────────────────────────────

_PLACEMENT_PROMPT = """\
You are a furniture and interior design expert.

Image 1: a decoration object — "{phrase}".
Image 2: the furniture it will be placed on — a {furn_type}.

Answer the following:

1. **Real-world size** — what is the typical real-world size of a {phrase}?
   Give approximate width × height × depth in METRES.
   Use common knowledge (e.g. a standard monitor is ~0.55m wide × 0.45m tall × 0.22m deep).

2. **Placement type** — choose ONE:
   - "on_surface"   : object rests flat on TOP of the furniture surface
                      (books, laptop, monitor, keyboard, lamp, vase, tray, plant pot)
   - "against_back" : object sits on the SEAT and leans against the BACK REST
                      (pillows, cushions on a sofa or chair)
   - "on_floor"     : object stands on the FLOOR next to the furniture
                      (floor lamp, large plant, umbrella stand)

3. **Surface position** (left/right on the furniture): "left", "center", or "right"

4. **Depth position** (front/back on the furniture, only for on_surface):
   "front", "middle", or "back"

5. **Facing direction** — does this object have a meaningful front face that must
   point AWAY from the furniture's back wall (toward the room / toward the user)?
   - "toward_room" : object has a clear front that faces the user
                     (monitor screen, laptop screen, keyboard typing side,
                      picture frame front, TV screen — anything the user looks at or uses)
   - "any"         : object is symmetric or has no meaningful front direction
                     (lamp, vase, book stack, plant, pillow, tray)

Return ONLY valid JSON, no markdown:
{{
  "size_m": {{"width": 0.0, "height": 0.0, "depth": 0.0}},
  "placement_type": "on_surface",
  "surface_position": "center",
  "depth_position": "middle",
  "facing": "toward_room",
  "reasoning": "one sentence"
}}
"""


def _vlm_placement(
    decor_img: Image.Image,
    furn_img: Image.Image | None,
    phrase: str,
    furn_type: str,
) -> dict:
    """Ask VLM for size estimate and placement details."""
    content: list[dict] = [
        {
            "type": "text",
            "text": _PLACEMENT_PROMPT.format(phrase=phrase, furn_type=furn_type),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_encode_pil(decor_img)}"},
        },
    ]
    if furn_img is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_encode_pil(furn_img)}"},
        })

    payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _vlm_post(payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip thinking tags
        raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
        # Strip markdown fences
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        # Extract first JSON object (greedy to capture nested braces)
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            result = json.loads(m.group())
            print(f"    [vlm] size={result.get('size_m')}  "
                  f"type={result.get('placement_type')}  "
                  f"side={result.get('surface_position')}  "
                  f"depth={result.get('depth_position')}  "
                  f"facing={result.get('facing', '(not returned)')}")
            print(f"    [vlm] reasoning: {result.get('reasoning', '')}")
            return result
        print(f"    [vlm] no JSON found in: {raw[:200]}")
    except Exception as e:
        print(f"    [vlm] FAILED: {e}")
    return None


# ── Scale computation ─────────────────────────────────────────────────────────

def _compute_scale(glb_path: Path, real_size: dict) -> tuple[float, dict]:
    """Compute uniform scale and real-world extents.

    Returns (scale_factor, size_m_dict).
    The scale factor maps GLB native units → metres.
    We use the height dimension as the anchor since VLM estimates it most
    reliably for common objects.
    """
    try:
        native = _glb_native_extents(glb_path)          # [nw, nh, nd]
    except Exception as e:
        print(f"    [scale] GLB load failed ({e}) — using default scale 0.3")
        rh = float(real_size.get("height", 0.3))
        return 1.0, {"width_m": rh, "height_m": rh, "depth_m": rh}

    rh = float(real_size.get("height", 0.3))
    nw, nh, nd = native[0], native[1], native[2]

    # Uniform scale anchored on height
    scale = (rh / nh) if nh > 1e-6 else 1.0

    # If max_footprint constraints were set by _clamp_vlm_size, enforce them.
    # Use the MINIMUM scale so no dimension exceeds its physical maximum.
    max_w = real_size.get("_max_footprint_w")
    max_d = real_size.get("_max_footprint_d")
    if max_w and nw > 1e-6:
        scale = min(scale, max_w / nw)
    if max_d and nd > 1e-6:
        scale = min(scale, max_d / nd)

    size_m = {
        "width_m":  float(nw * scale),
        "height_m": float(nh * scale),
        "depth_m":  float(nd * scale),
    }
    print(f"    [scale] native_h={nh:.3f}  target_h={rh:.3f}  scale={scale:.4f}  "
          f"→ {size_m['width_m']:.3f}×{size_m['height_m']:.3f}×{size_m['depth_m']:.3f} m")
    return float(scale), size_m


# ── Collision check ───────────────────────────────────────────────────────────

class _SurfaceTracker:
    """Tracks placed decoration footprints.

    Two layers:
    1. Per-surface 2D slots (local furniture space) — spreads objects within a piece.
    2. Global world-space 3D AABBs — prevents any two decorations from overlapping.
    """

    def __init__(self) -> None:
        # key: (furn_index, surface) → list of (cx, cz, hw, hd)
        self._slots: dict[tuple, list] = {}
        # global 3D AABBs: list of (cx, cy, cz, hx, hy, hz) in world metres
        self._world_aabbs: list[tuple] = []
        # seating pieces already used for against_back placements → set of furn_index
        self.seating_used: set[int] = set()

    def collides(self, furn_index: int, surface: str,
                 cx: float, cz: float, hw: float, hd: float) -> bool:
        slots = self._slots.get((furn_index, surface), [])
        for (ox, oz, ohw, ohd) in slots:
            if abs(cx - ox) < hw + ohw and abs(cz - oz) < hd + ohd:
                return True
        return False

    def world_collides(self, pos: "np.ndarray", half_extents: "np.ndarray",
                       margin: float = 0.02) -> bool:
        """Check if a world-space AABB overlaps any already-placed decoration."""
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
        hx, hy, hz = float(half_extents[0]) + margin, float(half_extents[1]) + margin, float(half_extents[2]) + margin
        for (ox, oy, oz, ohx, ohy, ohz) in self._world_aabbs:
            if (abs(cx - ox) < hx + ohx and
                    abs(cy - oy) < hy + ohy and
                    abs(cz - oz) < hz + ohz):
                return True
        return False

    def register(self, furn_index: int, surface: str,
                 cx: float, cz: float, hw: float, hd: float) -> None:
        key = (furn_index, surface)
        self._slots.setdefault(key, []).append((cx, cz, hw, hd))

    def register_world(self, pos: "np.ndarray", size_m: dict) -> None:
        """Register a placed decoration's world-space AABB."""
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
        hx = float(size_m.get("width_m",  0.3)) / 2.0
        hy = float(size_m.get("height_m", 0.3)) / 2.0
        hz = float(size_m.get("depth_m",  0.3)) / 2.0
        self._world_aabbs.append((cx, cy + hy, cz, hx, hy, hz))


# ── Position computation ──────────────────────────────────────────────────────


def _candidate_cx(sw: float, dw: float, preferred_side: str) -> list[float]:
    """Return a prioritised list of local-X candidates to try."""
    half = sw / 2 - dw / 2 - _EDGE_MARGIN
    half = max(half, 0.0)
    center = 0.0
    left   = -half
    right  =  half
    order = {"left": [left, center, right],
             "center": [center, left, right],
             "right": [right, center, left]}
    return order.get(preferred_side, [center, left, right])


def _candidate_cz(sd: float, dd: float, preferred_depth: str) -> list[float]:
    """Return a prioritised list of local-Z candidates to try (front→back)."""
    half = sd / 2 - dd / 2 - _EDGE_MARGIN
    half = max(half, 0.0)
    front  =  half
    middle =  0.0
    back   = -half
    order = {"front":  [front, middle, back],
             "middle": [middle, front, back],
             "back":   [back, middle, front]}
    return order.get(preferred_depth, [middle, front, back])


def _place_on_surface(
    furn: dict,
    decor_size: dict,
    side: str,
    depth_pos: str,
    tracker: _SurfaceTracker,
    out_dir: Path | None = None,
    cx_hint: "float | None" = None,
    cz_hint: "float | None" = None,
) -> np.ndarray | None:
    """Return world-space position (base of decoration) or None if blocked.

    When cx_hint / cz_hint are provided (e.g. from 3D back-projection of the
    reference bbox), tries that exact position first — clamped to the table
    bounds — before falling back to the coarse side/depth candidate grid.
    """
    sw = furn["size_m"]["width_m"]
    sd = furn["size_m"]["depth_m"]
    dw = decor_size["width_m"]
    dd = decor_size["depth_m"]

    cx_local = cz_local = None

    # Try the exact hint position first — back-projected from the reference
    # bbox this usually matches the intended layout precisely.
    if cx_hint is not None and cz_hint is not None:
        _cx_max = max(sw / 2 - dw / 2 - _EDGE_MARGIN, 0.0)
        _cz_max = max(sd / 2 - dd / 2 - _EDGE_MARGIN, 0.0)
        cx_try = float(np.clip(cx_hint, -_cx_max, _cx_max))
        cz_try = float(np.clip(cz_hint, -_cz_max, _cz_max))
        if not tracker.collides(furn["index"], "top", cx_try, cz_try, dw / 2, dd / 2):
            cx_local, cz_local = cx_try, cz_try
            print(f"    [on_surface] using back-proj hint cx={cx_try:+.3f} cz={cz_try:+.3f}")

    if cx_local is None:
        for cx in _candidate_cx(sw, dw, side):
            for cz in _candidate_cz(sd, dd, depth_pos):
                if not tracker.collides(furn["index"], "top", cx, cz, dw / 2, dd / 2):
                    cx_local, cz_local = cx, cz
                    break
            if cx_local is not None:
                break

    if cx_local is None:
        # Last-resort fallback: when NO candidate is collision-free (e.g. a
        # small side table already holds an oversized decoration that overlaps
        # every slot), place at the cx_hint (or table centre) anyway so the
        # item is VISIBLE rather than silently dropped.  Without this the lamp
        # / second book stack / etc. would just vanish from the render.
        _cx_max = max(sw / 2 - dw / 2 - _EDGE_MARGIN, 0.0)
        _cz_max = max(sd / 2 - dd / 2 - _EDGE_MARGIN, 0.0)
        if cx_hint is not None:
            cx_local = float(np.clip(cx_hint, -_cx_max, _cx_max))
        else:
            cx_local = float(np.clip(_candidate_cx(sw, dw, side)[0], -_cx_max, _cx_max))
        if cz_hint is not None:
            cz_local = float(np.clip(cz_hint, -_cz_max, _cz_max))
        else:
            cz_local = float(np.clip(_candidate_cz(sd, dd, depth_pos)[0], -_cz_max, _cz_max))
        print(f"    [collision] on_surface: no free spot on furniture {furn['index']} "
              f"— force-placing at cx={cx_local:+.3f} cz={cz_local:+.3f} (may overlap)")

    right_world, _, _ = _world_axes(furn)
    R = _rot_matrix(furn)
    local_z_world = R[:, 2]
    world_pos = _top_surface_center(furn).copy()
    world_pos += cx_local * right_world
    world_pos += cz_local * local_z_world

    # Refine Y using actual furniture surface geometry (raycast).
    # If the collision tracker pushed (cx,cz) to a bbox corner outside the
    # actual table mesh (common for round tables), raycast returns nom_y with
    # no hit printed.  Detect that by re-probing the furniture center — if the
    # center hits but our position doesn't, shrink (cx,cz) toward 0 until a
    # valid surface hit appears so the decoration rests on the real surface.
    if out_dir is not None:
        nom_y = world_pos[1]
        # Bed-specific: when placing on a bed, the per-point raycast at the
        # back of the bed (where pillows go) hits the headboard top, not the
        # mattress.  Override nom_y to the mattress plateau so the down-cast
        # below filters the headboard out of its ±search_radius window.
        _ftype_lower = (furn or {}).get("type", "").lower()
        if _ftype_lower in {"bed", "single_bed", "double_bed", "bunk_bed"}:
            _mattress_y = _bed_mattress_height(out_dir, furn)
            if _mattress_y is not None:
                nom_y = _mattress_y
                world_pos[1] = _mattress_y
        hit_y = _actual_surface_height(out_dir, float(world_pos[0]), float(world_pos[2]),
                                        nom_y)
        if hit_y == nom_y:
            center_world = _top_surface_center(furn).copy()
            center_hit = _actual_surface_height(out_dir,
                                                 float(center_world[0]),
                                                 float(center_world[2]),
                                                 nom_y)
            if center_hit != nom_y:
                for _shrink in (0.75, 0.5, 0.25, 0.0):
                    _probe = _top_surface_center(furn).copy()
                    _probe += (cx_local * _shrink) * right_world
                    _probe += (cz_local * _shrink) * local_z_world
                    _try_hit = _actual_surface_height(out_dir, float(_probe[0]),
                                                       float(_probe[2]), nom_y)
                    if _try_hit != nom_y:
                        cx_local *= _shrink
                        cz_local *= _shrink
                        world_pos = _probe
                        world_pos[1] = _try_hit
                        print(f"    [surface] shifted toward center ×{_shrink:.2f} "
                              f"to find table mesh")
                        break
                else:
                    world_pos[1] = hit_y
            else:
                world_pos[1] = hit_y
        else:
            world_pos[1] = hit_y

    tracker.register(furn["index"], "top", cx_local, cz_local, dw / 2, dd / 2)
    return world_pos


def _detect_backrest_local_z_sign(furn: dict, mesh) -> float:
    """Return +1 if the sofa's back-rest sits in the **+Z** local half (i.e.
    the furniture pipeline assigned a rotation that puts the back-rest on
    the opposite side of the convention), else −1.

    The placement code assumes ``−Z = toward backrest``.  When the
    pipeline picks the opposite rotation for a given GLB, every pillow
    formula (`target_local_z = −sd/2 + …`) puts the pillow at the front
    of the sofa.  We detect this directly: count the above-seat,
    central-strip vertices in each half of the sofa's local Z range and
    flip the sign when the +Z half clearly dominates.

    Returns −1 (the default convention) when:
      * the scene mesh isn't loaded
      * the sofa region is too sparse to judge (<100 verts)
      * fewer than 30 above-seat verts in the central strip
      * the two halves are roughly balanced (no clear winner)
    """
    if mesh is None:
        return -1.0
    try:
        sofa_pos = np.array(furn["position_m"], dtype=np.float64)
        R = _rot_matrix(furn)
        sw = float(furn["size_m"]["width_m"])
        sd = float(furn["size_m"]["depth_m"])
        sh = float(furn.get("eff_h", furn["size_m"].get("height_m", 0.85)))
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        in_region = (
            (np.abs(verts[:, 0] - sofa_pos[0]) <= sw / 2.0 + 0.30)
            & (np.abs(verts[:, 2] - sofa_pos[2]) <= sd / 2.0 + 0.30)
            & (verts[:, 1] >= sofa_pos[1] - 0.10)
            & (verts[:, 1] <= sofa_pos[1] + sh + 0.10)
        )
        sofa_verts = verts[in_region]
        if len(sofa_verts) < 100:
            return -1.0
        sofa_local = (sofa_verts - sofa_pos) @ R
        inner_hw = sw / 2.0 * 0.60
        # No |Z| filter for SIGN detection — we want to know which side
        # of the sofa has MORE above-seat mass, even when the mesh extends
        # past the metadata depth (which is exactly the case that breaks
        # the −Z assumption).  The actual back-rest geometry's position
        # along Z doesn't matter for sign — only its sign does.
        above_seat_central = (
            (np.abs(sofa_local[:, 0]) <= inner_hw)
            & (sofa_local[:, 1] > sh * _SEAT_FRAC + 0.05)
            & (sofa_local[:, 1] < sh + 0.10)
        )
        cz_vals = sofa_local[above_seat_central, 2]
        if len(cz_vals) < 30:
            return -1.0
        n_neg = int((cz_vals < 0).sum())
        n_pos = int((cz_vals > 0).sum())
        if n_pos > n_neg * 1.5 and n_pos >= 30:
            print(f"    [backrest_sign] sofa[{furn.get('index','?')}] +Z half "
                  f"has {n_pos} above-seat verts vs {n_neg} in -Z — "
                  f"INVERTED convention (backrest at +Z local)")
            return +1.0
        return -1.0
    except Exception as _be:
        print(f"    [backrest_sign] detection failed ({_be}) — using default −1")
        return -1.0


def _push_to_visible_face_against_back(
    world_pos: np.ndarray,
    furn: dict,
    decor_size: "dict | float",
    out_dir: "Path | None" = None,
) -> np.ndarray:
    """Push world_pos so a pillow/cushion rests on the visible seat surface.

    Strategy (combines forward + vertical resolution):
    1. The back rest zone is the rear _BACK_REST_FRAC of sofa depth.
       We only need to push forward until the pillow center clears the back rest
       front face — NOT until it exits the full sofa AABB.
    2. At each candidate XZ we raycast down to find the actual seat surface Y
       and snap the pillow base to it.
    3. If the surface hit is above (furn_top - dh * 0.3), we are still inside
       the solid back rest structure — push forward another step.
    4. Left/right jitter is tried if the seat surface is consistently too high
       (the pillow may be sitting on the arm rest instead of the seat).
    """
    if isinstance(decor_size, dict):
        dd = float(decor_size.get("depth_m", decor_size.get("depth", 0.15)))
        dh = float(decor_size.get("height_m", decor_size.get("height", 0.40)))
        dw = float(decor_size.get("width_m",  decor_size.get("width",  0.50)))
    else:
        dd = float(decor_size)
        dh = 0.40
        dw = 0.50

    R       = _rot_matrix(furn)
    sd      = furn["size_m"]["depth_m"]
    forward = R[:, 2]          # sofa local +Z → toward room
    right   = R[:, 0]          # sofa local +X → right side

    # Detect off-axis yaw (>20°) — for rotated sofas the local -Z direction
    # points toward a corner of the room, so a 2cm "lean into back" backward
    # nudge translates into a visible (+X, -Z) world drift that appears as
    # the pillow pushed off-centre.  Use a much smaller (or zero) lean for
    # rotated sofas; the back-rest contact is already implicit from
    # cz_local being clamped close to the seat centre upstream.
    try:
        _yaw_deg_pp = abs(np.degrees(np.arctan2(R[0, 2], R[0, 0])))
        _yaw_off_axis_pp = min(_yaw_deg_pp % 90.0, 90.0 - (_yaw_deg_pp % 90.0))
    except Exception:
        _yaw_off_axis_pp = 0.0
    _LEAN_NUDGE = 0.0 if _yaw_off_axis_pp > 20.0 else 0.02

    furn_base  = np.array(furn["position_m"], dtype=np.float64)
    furn_h     = float(furn.get("eff_h", furn["size_m"].get("height_m", 0.8)))
    furn_top_y = furn_base[1] + furn_h

    # Detect the actual back-cushion front face by casting a HORIZONTAL ray
    # from the front of the sofa BACKWARD (toward the back wall) at back-
    # cushion height.  The first surface the ray hits is the cushion's actual
    # front face.  This is more robust than a downward probe — it works
    # regardless of the cushion's height profile or whether the back rest
    # mesh is solid (downward raycasts can miss vertical front faces or
    # hit unexpected geometry on rotated sofas).
    # Falls back to a downward probe + _BACK_REST_FRAC estimate if the
    # horizontal ray finds nothing or out_dir is unavailable.
    actual_back_front_local_z: "float | None" = None
    _probe_diag = ""
    if out_dir is not None:
        try:
            mesh = _get_furn_scene_mesh(out_dir)
            if mesh is None:
                _probe_diag = "scene mesh unavailable"
            else:
                # Sample many heights across the back-rest band so we don't
                # miss the back cushion just because one ray slips through a
                # gap or the sample plane sits between mesh tessellation
                # rows.  35% → 95% of sofa height covers low + tall backs.
                _ray_y_samples = [furn_base[1] + furn_h * f
                                  for f in (0.35, 0.45, 0.55, 0.65,
                                            0.75, 0.85, 0.95)]
                _best_hit_local_z: "float | None" = None
                _samples_with_any_hits = 0
                _samples_passing_filters = 0
                for _y_sample in _ray_y_samples:
                    # Start at front of sofa at this height, cast backward
                    _ray_origin = furn_base.copy()
                    _ray_origin += (sd / 2.0 - 0.01) * forward     # +Z local (front)
                    _ray_origin[1] = _y_sample
                    _ray_dir = -forward                            # toward -Z local (back)
                    _locs, _, _ = mesh.ray.intersects_location(
                        ray_origins=np.array([_ray_origin]),
                        ray_directions=np.array([_ray_dir]),
                        multiple_hits=True,
                    )
                    if len(_locs) == 0:
                        continue
                    _samples_with_any_hits += 1
                    # Filter to hits within the sofa's XZ footprint and at
                    # the probe height (±30cm tolerance — vertices on the
                    # back rest's slanted surface can be 20-25cm above/below
                    # the sample plane and still belong to the cushion).
                    # Reject hits OUTSIDE the sofa's local-Z extent (e.g.
                    # the back wall behind, which would give local_z far
                    # past -sd/2).
                    _R_for = R
                    _offsets = _locs - furn_base
                    _cx_vals = _offsets @ _R_for[:, 0]              # local +X
                    _cz_vals = _offsets @ _R_for[:, 2]              # local +Z
                    _y_vals  = _locs[:, 1]
                    _half_w  = furn["size_m"]["width_m"] / 2.0 + 0.08
                    _in_bbox = ((np.abs(_cx_vals) <= _half_w)
                                & (_cz_vals <= sd / 2.0 + 0.05)
                                & (_cz_vals >= -sd / 2.0 - 0.05))
                    _at_height = np.abs(_y_vals - _y_sample) <= 0.30
                    _valid = _in_bbox & _at_height
                    # Also reject hits at the very front face (the sofa's own
                    # front edge) — they're not the back-cushion front face.
                    _valid &= (_cz_vals < sd / 2.0 - 0.05)
                    if not _valid.any():
                        continue
                    _samples_passing_filters += 1
                    # Front-most valid hit (largest cz_vals = closest to the
                    # room) — that's the cushion's front face from this side.
                    _v_cz = _cz_vals[_valid]
                    _front_cz = float(_v_cz.max())
                    if _best_hit_local_z is None or _front_cz > _best_hit_local_z:
                        _best_hit_local_z = _front_cz
                # Sanity: probe must land in the rear half of the sofa
                # (cushion lives behind the sofa midline); reject hits at
                # the very front (those are the sofa's front edge).
                if (_best_hit_local_z is not None
                        and -sd / 2.0 <= _best_hit_local_z <= 0.05):
                    pass  # valid
                elif _best_hit_local_z is not None:
                    print(f"    [back_face_probe] hit at local_z={_best_hit_local_z:.3f} "
                          f"outside plausible cushion zone [{-sd/2:.3f}, 0.05] — "
                          f"rejecting, falling through")
                    _best_hit_local_z = None
                if _best_hit_local_z is not None:
                    actual_back_front_local_z = _best_hit_local_z
                    print(f"    [back_face_probe] cushion front detected at "
                          f"local_z={actual_back_front_local_z:.3f} (horizontal ray, "
                          f"estimate was {-sd/2 + _BACK_REST_FRAC*sd:.3f})")
                else:
                    _probe_diag = (f"no valid hits after probing "
                                   f"{len(_ray_y_samples)} heights "
                                   f"({_samples_with_any_hits} hit something, "
                                   f"{_samples_passing_filters} passed the "
                                   f"bbox+height filter)")
        except Exception as _bpe:
            _probe_diag = f"exception: {_bpe}"
    if actual_back_front_local_z is None and _probe_diag:
        print(f"    [back_face_probe] failed — {_probe_diag}; trying mesh density")

    # Second-line detection: tight-bbox style, like furniture placement.
    # Take all sofa vertices that lie ABOVE the seat surface and inside the
    # sofa's XZ footprint, histogram their local-Z positions, find the peak
    # (back cushion mass), and walk forward from the peak until vertex
    # density drops — that's the cushion's actual front face.
    # Robust to mesh gaps and rotated GLBs that defeat the horizontal probe.
    if actual_back_front_local_z is None and out_dir is not None:
        try:
            mesh = _get_furn_scene_mesh(out_dir)
            if mesh is not None:
                _seat_y_world = furn_base[1] + furn_h * _SEAT_FRAC
                # Convert all mesh vertices to sofa-local frame.
                _v_world = np.asarray(mesh.vertices, dtype=np.float64)
                _v_offset = _v_world - furn_base
                _v_local_x = _v_offset @ R[:, 0]
                _v_local_z = _v_offset @ R[:, 2]
                _v_world_y = _v_world[:, 1]
                # Restrict X to the CENTRAL strip of the sofa width — this
                # excludes the arm rests (which sit at the ±X extremes and
                # often extend along the full sofa depth at heights above the
                # seat).  Without this filter the histogram peaks in front of
                # the actual back cushion because arms contribute equally to
                # every Z bin.  60% of half-width keeps almost all back-cushion
                # geometry while dropping arm-rest columns.
                _inner_half_w = furn["size_m"]["width_m"] / 2.0 * 0.60
                # Vertices that are:
                #   - inside the central cushion strip (no arms)
                #   - above the seat surface (where the back cushion lives)
                #   - within a reasonable Y window (skip ceiling lights etc.)
                _back_zone = (
                    (np.abs(_v_local_x) <= _inner_half_w)
                    & (np.abs(_v_local_z) <= sd / 2.0 + 0.05)
                    & (_v_world_y > _seat_y_world + 0.02)
                    & (_v_world_y < furn_base[1] + furn_h + 0.05)
                )
                _zs = _v_local_z[_back_zone]
                # Lowered threshold from 200 to 50 — low-poly low-poly sofas
                # (especially the ones whose probe also fails) tend to have
                # sparse back-rest geometry, but 50 verts in the back zone
                # is still enough to find a stable peak via histogram.
                if len(_zs) >= 50:
                    # Histogram along local Z.  30 bins ≈ 1.8cm per bin for sd=0.54m.
                    _hist, _edges = np.histogram(_zs, bins=30,
                                                  range=(-sd / 2.0, sd / 2.0))
                    _peak_idx = int(np.argmax(_hist))
                    _peak_count = int(_hist[_peak_idx])
                    # Walk forward from peak; the cushion's FRONT face is the
                    # bin where vertex count drops below threshold.
                    _density_thresh = max(_peak_count * 0.30, 3)
                    _front_bin_idx = _peak_idx
                    for _bi in range(_peak_idx + 1, len(_hist)):
                        if _hist[_bi] < _density_thresh:
                            _front_bin_idx = _bi
                            break
                    else:
                        _front_bin_idx = min(_peak_idx + 2, len(_hist) - 1)
                    # Bin centre of the front-edge bin.
                    _candidate_z = float((_edges[_front_bin_idx]
                                          + _edges[_front_bin_idx + 1]) / 2.0)
                    if -sd / 2.0 <= _candidate_z <= 0.05:
                        actual_back_front_local_z = _candidate_z
                        print(f"    [back_face_density] cushion front at "
                              f"local_z={_candidate_z:.3f} (from {len(_zs)} above-seat "
                              f"vertices, peak bin has {_peak_count} verts)")
                    else:
                        print(f"    [back_face_density] candidate "
                              f"local_z={_candidate_z:.3f} out of plausible "
                              f"range — using estimate "
                              f"({-sd/2 + _BACK_REST_FRAC*sd:.3f})")
                else:
                    print(f"    [back_face_density] only {len(_zs)} above-seat "
                          f"vertices in back zone (need ≥50) — using estimate "
                          f"({-sd/2 + _BACK_REST_FRAC*sd:.3f})")
        except Exception as _de:
            print(f"    [back_face_density] failed ({_de}) — using estimate")
    # Final fallback diagnostic
    if actual_back_front_local_z is None:
        print(f"    [back_face] no probe/density signal — falling back to "
              f"_BACK_REST_FRAC estimate {-sd/2 + _BACK_REST_FRAC*sd:.3f} for "
              f"sofa depth {sd:.3f}m")

    # Pillow target Z = back-rest mesh CENTROID (mass-aware, per sofa) —
    # but X stays at the caller-provided cx and Z is CLAMPED so the pillow's
    # back face never extends past the sofa's back wall.
    #
    # Why:
    #   - Using the centroid's X coordinate caused lateral drift on
    #     asymmetric meshes (the centroid sits off-axis, which pulled the
    #     pillow off-centre).  Lateral placement is better handled by the
    #     bbox/VLM cx hint already passed in.
    #   - Using the centroid's Z directly can place the pillow PAST the
    #     sofa's back wall when the back-rest mesh extends behind the
    #     metadata bbox.  Clamping the pillow's back face to at most
    #     `−sd/2 + 2 cm` keeps it inside the sofa.
    #
    # Detect which local-Z half holds the back-rest mass.  When the
    # furniture pipeline assigns an inverted rotation for a given sofa
    # GLB, the back-rest sits at +Z local instead of −Z; without this
    # check the placement formula puts the pillow at the front of the
    # sofa.  `backrest_sign` is +1 (inverted) or −1 (default), and is
    # stashed on `furn` so `_against_back_rotation` reads the same value.
    try:
        _mesh_for_centroid = _get_furn_scene_mesh(out_dir) if out_dir is not None else None
    except Exception:
        _mesh_for_centroid = None
    backrest_sign = _detect_backrest_local_z_sign(furn, _mesh_for_centroid)
    furn["_backrest_sign"] = backrest_sign

    # Bbox-fit fallback target — sign-aware.  Pillow's back face sits at
    # `backrest_sign * (sd/2 − 0.02)` (2 cm safety margin from the wall),
    # and the pillow centre is dd/2 forward of the back face (opposite the
    # backrest direction in local Z).
    #   sign=−1 → target = −(sd/2 − 0.02) + dd/2 = −0.15 (e.g. for sd=0.54, dd=0.20)
    #   sign=+1 → target = +(sd/2 − 0.02) − dd/2 = +0.15
    target_local_z = backrest_sign * (sd / 2.0 - 0.02 - dd / 2.0)

    # First compute centroid_back (if mesh available) — its result is
    # used both as a fallback when cushion-front detection fails AND as
    # a sanity check on cushion-front (see _CONSISTENCY_TOL below).
    _cz_back_centroid: "float | None" = None
    _back_face_z_centroid: "float | None" = None
    _back_rest_front_z_centroid: "float | None" = None
    _back_rest_back_z_centroid: "float | None" = None  # mesh back-most Z
    _centroid_used = False
    if _mesh_for_centroid is not None:
        try:
            _v_w = np.asarray(_mesh_for_centroid.vertices, dtype=np.float64)
            _region_pad = 0.30
            _region_mask = (
                (np.abs(_v_w[:, 0] - furn_base[0])
                    <= furn["size_m"]["width_m"] / 2.0 + _region_pad)
                & (np.abs(_v_w[:, 2] - furn_base[2])
                    <= furn["size_m"]["depth_m"] / 2.0 + _region_pad)
                & (_v_w[:, 1] >= furn_base[1] - 0.10)
                & (_v_w[:, 1] <= furn_base[1] + furn_h + 0.10)
            )
            _v_sofa = _v_w[_region_mask]
            if len(_v_sofa) >= 100:
                _v_loc = (_v_sofa - furn_base) @ R
                _inner_hw = furn["size_m"]["width_m"] / 2.0 * 0.60
                _seat_y_loc = (furn_h * _SEAT_FRAC) + 0.05
                # Restrict to the sofa's BACK-REST half (sign-aware), but
                # do NOT restrict by |z|: the mesh may extend past the
                # metadata depth, and that's exactly where we want to
                # place the pillow's back face.  Filtering with `|z| ≤ sd/2`
                # would have rejected sofa[1]'s 3850 back-rest verts that
                # sit at z ≈ −0.5, leaving the bbox-fit fallback to place
                # the pillow at metadata-z (way forward of the visible
                # back-rest).
                _backrest_mask = (
                    (np.abs(_v_loc[:, 0]) <= _inner_hw)
                    & (_v_loc[:, 1] > _seat_y_loc)
                    & (_v_loc[:, 1] < furn_h + 0.10)
                    & (_v_loc[:, 2] * backrest_sign > 0.0)         # backrest side
                )
                _br_local = _v_loc[_backrest_mask]
                if len(_br_local) >= 30:
                    _cz_back = float(_br_local[:, 2].mean())
                    # Mesh's actual back-most extent (sign-aware).  Pillow
                    # back face touches the centroid (where the cushion
                    # mass is), clamped so it doesn't extend past the
                    # mesh's own back-most vertex + 2 cm safety margin.
                    # NOTE: we deliberately do NOT clamp to the metadata
                    # depth here — when the generated meshes extends past
                    # metadata, the visible back-rest IS past metadata,
                    # and clamping pillows to the metadata back wall
                    # would put them at the front of the visible mesh
                    # (the original "pillow at front" failure).  X-axis
                    # extent IS clamped to the mesh below.
                    _br_z = _br_local[:, 2]
                    # Front-most back-rest vertex (sign-aware): this is
                    # where the back-rest cushion's FRONT FACE sits, i.e.
                    # the surface a pillow leans against.  We use the
                    # 90th percentile (sign-aware) to be robust to a
                    # handful of stray verts on the seat side that
                    # would otherwise pull the estimate forward.
                    if backrest_sign < 0:
                        _mesh_back_z  = float(_br_z.min())
                        _mesh_front_z = float(np.percentile(_br_z, 90))
                        _safe_back    = _mesh_back_z + 0.02
                        _back_face_z  = max(_cz_back, _safe_back)
                    else:
                        _mesh_back_z  = float(_br_z.max())
                        _mesh_front_z = float(np.percentile(_br_z, 10))
                        _safe_back    = _mesh_back_z - 0.02
                        _back_face_z  = min(_cz_back, _safe_back)
                    _cz_back_centroid = _cz_back
                    _back_face_z_centroid = _back_face_z
                    # Also store the front-most-vert estimate so the
                    # cushion-front-vs-centroid choice can fall back to
                    # it when cushion-front detection (density) is
                    # bogus — e.g. sofa[1]'s mesh has a density peak
                    # at the SEAT cushion's back edge (z≈-0.116) far
                    # forward of the back-rest cushion's actual front
                    # face (z≈-0.46).  In that case we want the
                    # pillow's back face at the back-rest cushion's
                    # front face, not at the spurious density peak.
                    _back_rest_front_z_centroid = _mesh_front_z
                    _back_rest_back_z_centroid  = _mesh_back_z
                    _clamped = abs(_back_face_z - _cz_back) > 1e-3
                    print(f"    [centroid_back] back-rest centroid_z="
                          f"{_cz_back:+.3f}, mesh back-most z="
                          f"{_mesh_back_z:+.3f} (sign={int(backrest_sign):+d}, "
                          f"from {len(_br_local)} verts), candidate pillow "
                          f"back face → {_back_face_z:+.3f}"
                          f"{' (CLAMPED to mesh extent)' if _clamped else ''}")
                else:
                    print(f"    [centroid_back] only {len(_br_local)} back-rest "
                          f"verts on sign={int(backrest_sign):+d} side (need ≥30) "
                          f"— using bbox-fit fallback (target={target_local_z:+.3f})")
            else:
                print(f"    [centroid_back] only {len(_v_sofa)} verts in sofa "
                      f"region — using bbox-fit fallback")
        except Exception as _ce:
            print(f"    [centroid_back] failed ({_ce}) — using bbox-fit fallback")

    # ── Choose target_local_z: cushion_front vs centroid_back ─────────
    # The cushion-front density detection is reliable when the
    # back-rest is a clean, well-defined cushion: peak vertex density
    # at the cushion's front face.  But on sofas whose back-rest mesh
    # is unusual (e.g. sofa[1] here, where back-rest verts extend
    # 50 cm past metadata), the density peak hits the SEAT cushion's
    # back edge instead — much shallower than the actual back-rest
    # face — and pillows end up centred on the seat with a big gap to
    # the visible back-rest.
    # Place the pillow's back face at the VERTICAL back-rest's front
    # face — the actual surface a real pillow leans against.  We use
    # the 90th percentile of back-rest verts (sign-aware) as a robust
    # estimate of that front face, with a 2 cm safety margin so the
    # pillow doesn't clip into the back-rest cushion mesh.
    # Cushion-front detection (vertex-density peak above the seat) is
    # used only when it's CONSISTENT with the back-rest centroid — on
    # sofas where it agrees with centroid_back the density peak IS
    # the back-rest cushion's front face; on others (e.g. sofa[1])
    # it's the seat cushion's back edge instead, which would leave
    # the pillow stranded on the seat in front of the visible
    # back-rest cushion.
    # The seat-vs-back-rest gap on broken meshes (sofa[1] has ~35 cm
    # of empty space between seat end and back-rest cushion) is left
    # for downstream Y-adjustment to resolve — the user prefers
    # "pillow visually against the back, bottom slightly hovering"
    # over "pillow on the seat, far from the back-rest".
    _CONSISTENCY_TOL = 0.25
    _cushion_front_used = False
    if (actual_back_front_local_z is not None
            and (_back_face_z_centroid is None
                 or abs(_back_face_z_centroid - actual_back_front_local_z)
                    <= _CONSISTENCY_TOL)):
        target_local_z = (actual_back_front_local_z
                           - backrest_sign * (dd / 2.0))
        _cushion_front_used = True
        print(f"    [back_face] cushion-front and centroid_back "
              f"consistent (centroid {_back_face_z_centroid} vs "
              f"cushion-front {actual_back_front_local_z:+.3f}) — "
              f"pillow back face at cushion-front, "
              f"cz_local={target_local_z:+.3f}")
    elif _back_rest_front_z_centroid is not None:
        # Use the back-rest's front-most vertex (90th-percentile, sign-aware) as
        # the cushion front face; pillow back face 2 cm in front to avoid clipping.
        _front_est = _back_rest_front_z_centroid
        # On ornate sofas the back-rest flares FORWARD (curved frame / top crest),
        # so the front-most vertex lands near the seat centre and strands the
        # pillow mid-seat. When there is NO cushion-front signal to corroborate it,
        # fall back to the metadata back-rest-front estimate and take whichever
        # sits further BACK, so the pillow actually leans on the back rest.
        if actual_back_front_local_z is None:
            _meta_front = backrest_sign * (sd / 2.0 - _BACK_REST_FRAC * sd)
            if _meta_front * backrest_sign > _front_est * backrest_sign:
                print(f"    [back_face] front-most vertex {_front_est:+.3f} flares "
                      f"forward (no cushion-front signal) — using metadata "
                      f"back-rest front {_meta_front:+.3f} instead")
                _front_est = _meta_front
        _safe_front_z = (_front_est - backrest_sign * 0.02)
        target_local_z = (_safe_front_z - backrest_sign * (dd / 2.0))
        _centroid_used = True
        if actual_back_front_local_z is not None:
            print(f"    [back_face] cushion-front {actual_back_front_local_z:+.3f} "
                  f"disagrees with centroid_back {_back_face_z_centroid:+.3f} "
                  f"by >{_CONSISTENCY_TOL*100:.0f}cm — using back-rest "
                  f"front-most vertex {_back_rest_front_z_centroid:+.3f} "
                  f"(with 2 cm safety) → pillow back face "
                  f"{_safe_front_z:+.3f}, cz_local={target_local_z:+.3f}")
        else:
            print(f"    [back_face] no cushion-front signal; back-rest front "
                  f"{_front_est:+.3f} → cz_local={target_local_z:+.3f}")
    elif _back_face_z_centroid is not None:
        target_local_z = (_back_face_z_centroid
                           - backrest_sign * (dd / 2.0))
        _centroid_used = True
        print(f"    [back_face] using centroid_back face "
              f"{_back_face_z_centroid:+.3f} → "
              f"cz_local={target_local_z:+.3f}")

    # ── Footprint-on-sofa sanity check (strict bbox containment) ──────
    # The pillow's top-down bounding box must lie fully inside the
    # sofa's metadata footprint along the depth axis.  No back-face
    # overhang into the back-rest cushion zone — when the generated meshes
    # extends past metadata (sofa[1]) the pillow may sit forward of
    # the visible cushion, but a top-down render must always show the
    # pillow's bbox contained inside the sofa's bbox (per user req).
    #
    # Account for the against_back 20° back-lean: pillow's projected
    # extent along sofa-Z is `dd*cos(lean)/2 + dh*sin(lean)/2`, not
    # just `dd/2`.  The leaning TOP of the pillow is what extends
    # furthest along sofa-Z, and `dd/2` would under-clamp by ~6cm for
    # a typical pillow (dh=0.43, dd=0.20, lean=20°: actual half-extent
    # 0.167 vs naive 0.10).
    _MARGIN_FP   = 0.01
    _LEAN_RAD    = np.radians(20.0)
    _ext_z_proj  = (dd / 2.0) * np.cos(_LEAN_RAD) + (dh / 2.0) * np.sin(_LEAN_RAD)
    _back_face_z  = target_local_z + backrest_sign * _ext_z_proj
    _front_face_z = target_local_z - backrest_sign * _ext_z_proj
    # Back wall: prefer the mesh-derived back-rest back-most Z when
    # available (matches the cx code path which already trusts mesh
    # over metadata).  When the generated meshes extends past metadata
    # (sofa[1]: mesh back at -0.620 vs metadata back at -0.259), the
    # metadata clamp would yank the pillow ~28cm forward of the visible
    # backrest.  Mesh-derived back wall lets the leaning pillow's top
    # extend into the back-rest cushion's volume — physically correct
    # for "pillow against the back".
    _back_wall_src = "metadata"
    if _back_rest_back_z_centroid is not None:
        _back_wall_z = _back_rest_back_z_centroid + backrest_sign * _MARGIN_FP
        _back_wall_src = "mesh back-rest back"
    else:
        _back_wall_z  = backrest_sign * (sd / 2.0 - _MARGIN_FP)
    _front_wall_z = -backrest_sign * (sd / 2.0 - _MARGIN_FP)
    _was_clamped = False
    if backrest_sign < 0:
        if _back_face_z < _back_wall_z:
            target_local_z = _back_wall_z + _ext_z_proj
            _was_clamped = True
        if _front_face_z > _front_wall_z:
            target_local_z = _front_wall_z - _ext_z_proj
            _was_clamped = True
    else:
        if _back_face_z > _back_wall_z:
            target_local_z = _back_wall_z - _ext_z_proj
            _was_clamped = True
        if _front_face_z < _front_wall_z:
            target_local_z = _front_wall_z + _ext_z_proj
            _was_clamped = True
    if _was_clamped:
        print(f"    [footprint_check] pillow back face {_back_face_z:+.3f} "
              f"or front face {_front_face_z:+.3f} out of bounds "
              f"(back wall {_back_wall_z:+.3f} [{_back_wall_src}], front wall "
              f"{_front_wall_z:+.3f}, sign={int(backrest_sign):+d}, "
              f"projected ext_z={_ext_z_proj:.3f} from dd={dd:.2f}, "
              f"dh={dh:.2f}, lean=20°) "
              f"— clamped target_local_z to {target_local_z:+.3f} "
              f"(pillow bbox fully inside seat band; top-down "
              f"containment enforced)")
    else:
        print(f"    [footprint_check] pillow footprint OK: back face "
              f"{_back_face_z:+.3f}, front face {_front_face_z:+.3f}, "
              f"back wall {_back_wall_z:+.3f} [{_back_wall_src}], front wall "
              f"{_front_wall_z:+.3f} (projected ext_z={_ext_z_proj:.3f})")

    # `back_rest_front` is still computed above (probe/density) but ONLY
    # used downstream as the seat-Y density filter's "forward of cushion"
    # bound — not as the pillow's Z.
    if actual_back_front_local_z is not None:
        back_rest_front = actual_back_front_local_z
    else:
        back_rest_front = -sd / 2.0 + _BACK_REST_FRAC * sd

    _STEP_M    = 0.04   # forward step size
    _MAX_STEPS = 20     # ≈ 80 cm maximum push
    _JITTER    = [0.0, 0.06, -0.06]   # lateral offsets tried at each forward step

    # nom_y centred on seat-Y; search_radius widened below to include cushion
    # tops (y up to ~0.85) without grabbing floor hits (y=0).
    nom_y = float(world_pos[1])
    # Save initial position so we can fall back to it if iteration over-shoots.
    _initial_world_pos = world_pos.copy()
    # Seat surface height: back rest top would be > this
    expected_seat_y = furn_base[1] + furn_h * _SEAT_FRAC

    # ── "Snap to back, drop down" path ──────────────────────────────────────
    # Per user request: don't iterate forward step-by-step (which over-shoots
    # the cushion when raycast verification has gaps).  Instead: snap pillow
    # directly to the target_local_z (back face just touching the cushion's
    # actual or estimated front face), then raycast DOWN from above to find
    # the seat surface and drop the pillow's Y to it.
    if out_dir is not None:
        # Preserve the input cx_local from world_pos (encodes the bbox/VLM
        # cx hint passed by the caller).  We deliberately do NOT use the
        # centroid for X — that caused lateral drift on asymmetric meshes
        # (the centroid was off-axis on noisy generated meshes, pulling the
        # pillow toward the sofa's edge).  X is handled by the bbox/VLM
        # hint in `_place_against_back`, with the additional clamp below.
        _init_offset = _initial_world_pos - furn_base
        _kept_cx_local = float(np.dot(_init_offset, right))
        _sw_meta = furn["size_m"]["width_m"]
        # Hard clamp: pillow's X-extent must stay inside the sofa's
        # metadata width.  Skipped when the placement function used
        # `_force_centre` and snapped to the visible mesh centre —
        # in that case the upstream chose visual fit over metadata
        # containment, and the metadata bbox itself doesn't match
        # the rendered geometry (broken generated meshes).
        _force_centre_used = bool(furn.get("_force_centre_used", False))
        if not _force_centre_used:
            _max_cx_meta = max(_sw_meta / 2.0 - dw / 2.0 - 0.02, 0.0)
            if abs(_kept_cx_local) > _max_cx_meta:
                print(f"    [snap_drop] cx_local={_kept_cx_local:+.3f} clamped to "
                      f"±{_max_cx_meta:.3f} (pillow edge would poke past sofa "
                      f"width {_sw_meta:.2f}m)")
                _kept_cx_local = float(np.clip(_kept_cx_local,
                                                -_max_cx_meta, _max_cx_meta))
        else:
            print(f"    [snap_drop] cx_local={_kept_cx_local:+.3f} kept "
                  f"(force_centre used; visible-mesh placement, metadata "
                  f"clamp skipped)")
        _snap_world = furn_base.copy()
        _snap_world += _kept_cx_local * right
        _snap_world += target_local_z * forward
        # Apply lean-nudge for axis-aligned sofas (skipped for rotated via
        # _LEAN_NUDGE = 0.0 set above).
        _snap_world = _snap_world - forward * _LEAN_NUDGE
        # Raycast DOWN from above the sofa to find the actual seat surface.
        # search_radius covers most of the sofa height so we catch cushion
        # tops AND the underlying seat surface; we want the LOWEST hit (the
        # seat) — not the cushion top — so we pull all hits and pick the
        # lowest one in the seat-y range.
        try:
            _mesh_for_drop = _get_furn_scene_mesh(out_dir)
        except Exception:
            _mesh_for_drop = None
        _seat_y_picked = expected_seat_y
        if _mesh_for_drop is not None:
            try:
                _ray_origin = np.array([[float(_snap_world[0]),
                                          furn_top_y + 0.20,
                                          float(_snap_world[2])]])
                _ray_dir    = np.array([[0.0, -1.0, 0.0]])
                _locs, _, _ = _mesh_for_drop.ray.intersects_location(
                    ray_origins=_ray_origin, ray_directions=_ray_dir,
                    multiple_hits=True)
                # Disabled per-mesh raycast — gave inconsistent Y between
                # paired sofas (one GLB had a detectable seat-top hit, the
                # other didn't), making one pillow look hovering above its
                # seat while the other looked correct.  Compute seat-Y the
                # same way for every sofa via the mesh-density approach
                # below so paired pillows always match vertically.
                _ = _locs   # available but unused
            except Exception as _de:
                print(f"    [snap_drop] raycast failed ({_de}) — using expected_seat_y")
        # Mesh-density seat-Y detection (consistent across paired sofas).
        # Same algorithm as the back-cushion density detector but on the Y
        # axis: take all sofa vertices in the seat zone (XZ footprint, Z
        # forward of the back cushion), histogram their world Y, find the
        # peak — that's the seat surface where pillows should sit.
        try:
            _mesh_for_drop2 = _get_furn_scene_mesh(out_dir) if out_dir is not None else None
        except Exception:
            _mesh_for_drop2 = None
        if _mesh_for_drop2 is not None:
            try:
                _vw = np.asarray(_mesh_for_drop2.vertices, dtype=np.float64)
                _voff = _vw - furn_base
                _vlx = _voff @ R[:, 0]
                _vlz = _voff @ R[:, 2]
                _vyw = _vw[:, 1]
                _half_w_seat = furn["size_m"]["width_m"] / 2.0 + 0.05
                # Seat zone: XZ inside footprint, Z forward of back cushion,
                # Y between floor and a margin below sofa top.
                _seat_zone = (
                    (np.abs(_vlx) <= _half_w_seat)
                    & (_vlz > back_rest_front + 0.02)              # forward of cushion
                    & (_vlz <= sd / 2.0 + 0.02)                    # not past sofa front
                    & (_vyw > furn_base[1] + 0.05)                 # above floor
                    & (_vyw < furn_base[1] + furn_h * 0.65)        # below back-cushion top
                )
                _seat_ys = _vyw[_seat_zone]
                if len(_seat_ys) >= 100:
                    _hist_y, _edges_y = np.histogram(_seat_ys, bins=20,
                                                     range=(furn_base[1],
                                                            furn_base[1] + furn_h))
                    _peak_y_idx = int(np.argmax(_hist_y))
                    # Centre of the peak bin.
                    _peak_y = float((_edges_y[_peak_y_idx]
                                     + _edges_y[_peak_y_idx + 1]) / 2.0)
                    # Plausibility: must be within ±0.08m of expected_seat_y.
                    if abs(_peak_y - expected_seat_y) <= 0.08:
                        _seat_y_picked = _peak_y
                    else:
                        print(f"    [seat_y_density] peak y={_peak_y:.3f} too far "
                              f"from expected {expected_seat_y:.3f} — using expected")
            except Exception as _se:
                print(f"    [seat_y_density] failed ({_se}) — using expected_seat_y")
        _snap_world[1] = _seat_y_picked
        _local_z_now = float(np.dot(forward, _snap_world - furn_base))
        _y_src = ("seat density peak" if abs(_seat_y_picked - expected_seat_y) > 1e-3
                  else "expected_seat_y")
        _back_face_local_z = _local_z_now - dd / 2.0
        print(f"    [snap_drop] cz_local={_local_z_now:.3f} (bbox-fit, "
              f"pillow back face @ {_back_face_local_z:+.3f} vs sofa back "
              f"@ {-sd/2:+.3f}), y={_seat_y_picked:.3f} via {_y_src}  "
              f"detected_cushion_front_local_z={back_rest_front:.3f}"
              f"{' (probe)' if actual_back_front_local_z is not None else ' (estimate)'} "
              f"[unused for placement]")
        return _snap_world

    # Asymmetric tolerance: the seat cushion is usually a flat, slightly-
    # compressed layer; hits ABOVE expected_seat_y by up to 20 cm are still
    # "seat-like" (lightly-packed cushion tops + back-rest bases), but hits
    # BELOW are almost always the frame/floor peeking through a mesh gap —
    # those make the pillow sink.  A tight -5 cm lower bound keeps the
    # pillow at real cushion height.
    # Upper tolerance was +20 cm which was too loose — it accepted hits on
    # ARM REST tops (typically 8-12 cm above the seat cushion) as "seat-like",
    # so pillows at the extreme-left cx_local would land on the armrest and
    # hover 10 cm above the real seat.  Tighten to +8 cm — still permissive
    # enough for slightly-compressed cushion tops but rejects armrests.
    _SEAT_Y_UP  = 0.08
    _SEAT_Y_DN  = 0.05   # accept hits only up to 5 cm below expected seat

    def _is_seat_surface(y: float) -> bool:
        """True if hit_y is within seat range (not back rest, not floor).
        Used for the pillow CENTER — which should rest on the flat seat."""
        if y > furn_top_y - dh * 0.20:
            return False
        return (y >= expected_seat_y - _SEAT_Y_DN
                and y <= expected_seat_y + _SEAT_Y_UP)

    def _is_back_or_seat(y: float) -> bool:
        """Accept seat hits OR back-cushion hits — used for the pillow BACK
        face, which is SUPPOSED to rest against the elevated back cushion.
        Previously we required the back face to also be "seat-level," which
        kept pushing the pillow forward out of the back cushion.  Only reject
        if the hit is below the seat (mesh gap / floor showing through)."""
        # Reject anything below the seat cushion — that means the raycast
        # fell through the mesh and hit frame/floor.
        return y >= expected_seat_y - _SEAT_Y_DN

    for step in range(_MAX_STEPS):
        local_z = float(np.dot(forward, world_pos - furn_base))

        for dx in _JITTER:
            candidate = world_pos + dx * right

            # Check BOTH pillow center AND back face to ensure full clearance
            if out_dir is not None:
                # nom_y at seat-y; radius covers cushion top (y ≈ 0.7-0.85
                # → seat_y + 0.50 = 0.88 covers it) but excludes floor (y=0
                # is below seat_y - 0.50 = -0.12).  The +0.50 upper bound
                # picks up back-cushion-top hits which the old 0.30 radius
                # missed.
                _sr_full = 0.50
                center_hit = _actual_surface_height(
                    out_dir, float(candidate[0]), float(candidate[2]),
                    nom_y, search_radius=_sr_full)
                # Back face is dd/2 behind the center in local -Z direction
                back_face_world = candidate - forward * (dd / 2.0)
                back_hit = _actual_surface_height(
                    out_dir, float(back_face_world[0]), float(back_face_world[2]),
                    nom_y, search_radius=_sr_full)

                center_ok = _is_seat_surface(center_hit)
                # Back face is EXPECTED to hit the back cushion (higher than
                # seat).  Only reject if it falls into a mesh gap / below seat.
                back_ok   = _is_back_or_seat(back_hit)

                # Relaxed criterion: accept the position when back_ok is True
                # AND we're past target_local_z, EVEN if center_ok is False.
                # Sofas with mesh gaps (the right-sofa GLB has holes where
                # the pillow centre XZ raycasts fall through to the floor)
                # otherwise trap the iteration in max_steps.  When center_hit
                # isn't a valid seat surface, fall back to the deterministic
                # expected_seat_y for the Y coordinate so the pillow doesn't
                # sink into the gap or float above the cushion top.
                if back_ok and local_z >= target_local_z:
                    # Small backward nudge so the pillow visually LEANS INTO
                    # the back cushion (soft pillows compress against the back
                    # when someone leans on them).  A forward push would leave
                    # the pillow sitting out in the middle of the seat.
                    # Skipped for rotated sofas (see _LEAN_NUDGE above): a
                    # 2cm backward nudge on a 45°-yaw sofa shifts the pillow
                    # ~1.4cm in BOTH +X and -Z world, visibly off-centre.
                    candidate = candidate - forward * _LEAN_NUDGE
                    if center_ok:
                        candidate[1] = center_hit
                    else:
                        candidate[1] = expected_seat_y
                    if step > 0 or dx != 0.0 or not center_ok:
                        total = step * _STEP_M
                        _y_src = "seat raycast" if center_ok else f"expected_seat_y (center_hit={center_hit:.3f} invalid)"
                        print(f"    [push_forward] {total:.3f}m fwd (leans {_LEAN_NUDGE*100:.0f}cm into back), lateral {dx:+.2f}m "
                              f"→ y={candidate[1]:.3f} via {_y_src}, back face y={back_hit:.3f}")
                    return candidate

            elif local_z >= target_local_z:
                # No raycast available — just clear the back rest zone
                return candidate

        # Center or back face still on back rest — advance forward
        world_pos = world_pos + forward * _STEP_M

    # Fallback: when iteration runs out of steps without finding a valid
    # (center+back) seat surface — typically because the sofa GLB has gaps
    # the raycast falls through, returning floor hits — DO NOT keep the
    # pillow at the over-shot world_pos (which is now far forward of the
    # sofa).  Instead, place pillow back-flush against the sofa BBox back
    # wall at seat height.  Keeps it inside the sofa footprint and at the
    # right vertical level even if raycast verification fails.
    # Compute cx_local from the INITIAL position (which has cx_local applied
    # but cz_local at -(sd/2 - back_margin - dd/2)).  Pillow's cx in local
    # frame is preserved across the forward-walk (forward is perpendicular
    # to right).
    _init_offset = _initial_world_pos - furn_base
    _kept_cx_local = float(np.dot(_init_offset, right))
    _fallback_cz_local = -sd / 2.0 + dd / 2.0 + 0.02   # back face ~2cm in from back wall
    _fallback_world = furn_base.copy()
    _fallback_world += _kept_cx_local * right
    _fallback_world += _fallback_cz_local * forward
    _fallback_world[1] = expected_seat_y
    print(f"    [push_forward] max steps reached (probably mesh gaps in sofa "
          f"GLB) — fallback to back-flush at seat: pos={_fallback_world.round(3).tolist()}")
    return _fallback_world


def _bbox_to_table_local(
    furn: dict,
    box_px: "list[int] | None",
    camera: "dict | None",
    img_w: int = 1500,
    img_h: "int | None" = None,
    out_dir: "Path | None" = None,
) -> "tuple[float, float] | None":
    """Back-project the bbox's bottom-center pixel onto the table's actual top
    surface and decompose the hit into furniture-local (cx, cz) metres.

    When the caller provides out_dir, we raycast the camera ray directly
    against the furniture GLB mesh — this gives the true world point where the
    ray first hits the table's visible top, independent of the furniture's
    nominal eff_h (which can be off by several cm and cause large horizontal
    back-projection errors for shallow-angle rays).  Falls back to the nominal
    plane intersection if no mesh is available.

    Returns (cx_local, cz_local) or None if projection fails.
    """
    if box_px is None or camera is None:
        return None
    try:
        from object_placement.wall_mounted.wall_mounted_object_placement import (
            _camera_axes, _backproject_pixel,
        )
        cp  = np.array(camera["position_m"], dtype=np.float64)
        la  = np.array(camera["look_at_m"],  dtype=np.float64)
        up  = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
        rv, uv, fv = _camera_axes(cp, la, up)
        fx = img_w / (2.0 * np.tan(np.radians(float(camera["hfov_deg"]) / 2.0)))
        _img_h_eff = img_h if img_h is not None else int(img_w * 3 // 4)
        cx2, cy2 = img_w / 2.0, _img_h_eff / 2.0

        furn_pos = np.array(furn["position_m"], dtype=np.float64)
        nominal_y = furn_pos[1] + float(
            furn.get("eff_h", furn["size_m"].get("height_m", 0.5))
        )

        bx = (box_px[0] + box_px[2]) / 2.0
        by = float(box_px[3])
        ray_dir = _backproject_pixel(bx, by, cp, rv, uv, fv, fx, fx, cx2, cy2)

        world_pt: "np.ndarray | None" = None

        # Prefer true mesh raycast — the ray hits wherever the actual table
        # top is, regardless of eff_h estimation error.
        if out_dir is not None:
            try:
                mesh = _get_furn_scene_mesh(out_dir)
                if mesh is not None:
                    locs, _, _ = mesh.ray.intersects_location(
                        ray_origins=np.array([cp]),
                        ray_directions=np.array([ray_dir]),
                        multiple_hits=True,
                    )
                    if len(locs) > 0:
                        # Restrict to hits near the furniture (within its bbox +slack)
                        _hw = furn["size_m"]["width_m"] / 2.0 + 0.10
                        _hd = furn["size_m"]["depth_m"] / 2.0 + 0.10
                        offsets = locs - furn_pos
                        right_world_tmp, _, _ = _world_axes(furn)
                        R_tmp = _rot_matrix(furn)
                        cx_vals = offsets @ right_world_tmp
                        cz_vals = offsets @ R_tmp[:, 2]
                        in_bbox = (np.abs(cx_vals) <= _hw) & (np.abs(cz_vals) <= _hd)
                        in_near_top = np.abs(locs[:, 1] - nominal_y) <= 0.20
                        valid = in_bbox & in_near_top
                        if valid.any():
                            valid_locs = locs[valid]
                            # Among hits on the furniture top, pick the one CLOSEST to camera
                            ds = np.linalg.norm(valid_locs - cp, axis=1)
                            world_pt = valid_locs[int(np.argmin(ds))]
            except Exception:
                world_pt = None

        if world_pt is None:
            # Fallback: intersect nominal surface plane.
            if abs(ray_dir[1]) < 1e-6:
                return None
            t = (nominal_y - cp[1]) / ray_dir[1]
            if t <= 0:
                return None
            world_pt = cp + t * ray_dir

        right_world, _, _ = _world_axes(furn)
        R = _rot_matrix(furn)
        local_z_world = R[:, 2]

        offset = world_pt - furn_pos
        cx_local = float(np.dot(offset, right_world))
        cz_local = float(np.dot(offset, local_z_world))
        return cx_local, cz_local
    except Exception:
        return None


def _bbox_preferred_cx(
    furn: dict,
    box_px: "list[int] | None",
    camera: "dict | None",
    img_w: int = 1500,
    img_h: "int | None" = None,
    rank_hint: "float | None" = None,
    furn_photo_box_px: "list[int] | None" = None,
) -> float:
    """Return the preferred cx_local (furniture-local right-axis offset) for a
    decoration whose reference bbox is `box_px`.  Positive = furniture right,
    negative = furniture left.  Returns 0.0 if projection fails.

    Uses the sofa center projected at seat height so the image-space reference
    matches where pillows actually appear (not the floor-level sofa base which
    can project outside the image entirely for sofas against a side wall).

    `furn_photo_box_px`, when provided, is the PHOTO bbox of the furniture (from
    `furniture/segment_results.json`).  Its centre is where the sofa actually
    appears in the photo — the pillow's `box_px` is anchored to that, not to
    the 3D-projected sofa centre.  When the 3D placement and the photo sofa
    disagree by >40 px (placement misalignment), we distrust the silhouette
    hint and snap toward centre.

    When the furniture's right_world axis is nearly aligned with the camera depth
    direction (px_per_m ≈ 0), falls back to y-projection: moving along right_world
    shifts the projected y-pixel instead of x-pixel, so cx_local is derived from
    the bbox's vertical position in the image.
    """
    if box_px is None or camera is None:
        return 0.0
    try:
        from object_placement.wall_mounted.wall_mounted_object_placement import (
            _camera_axes, _project_vertex,
        )
        cp  = np.array(camera["position_m"], dtype=np.float64)
        la  = np.array(camera["look_at_m"],  dtype=np.float64)
        up  = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
        rv, uv, fv = _camera_axes(cp, la, up)
        fx = img_w / (2.0 * np.tan(np.radians(float(camera["hfov_deg"]) / 2.0)))
        _img_h_eff = img_h if img_h is not None else int(img_w * 3 // 4)
        cx2, cy2 = img_w / 2.0, _img_h_eff / 2.0

        right_world, _, _ = _world_axes(furn)
        furn_pos = np.array(furn["position_m"], dtype=np.float64)
        half_w = furn["size_m"]["width_m"] / 2.0
        furn_h = float(furn.get("eff_h", furn["size_m"].get("height_m", 0.8)))

        # Project from seat height (not floor level) so the sofa center maps to
        # where pillows actually appear in the image.
        seat_y = furn_pos[1] + furn_h * _SEAT_FRAC
        sofa_center_3d = np.array([furn_pos[0], seat_y, furn_pos[2]], dtype=np.float64)

        # Project sofa center and a point 1m along right_world from the center
        sofa_px,  sofa_py,  sofa_z  = _project_vertex(sofa_center_3d,
                                                        cp, rv, uv, fv, fx, cx2, cy2)
        sofa_r_3d = sofa_center_3d + right_world  # 1 m along local right
        sofa_r_px, sofa_r_py, sofa_rz = _project_vertex(sofa_r_3d,
                                                          cp, rv, uv, fv, fx, cx2, cy2)

        if sofa_z <= 0.01 or sofa_rz <= 0.01:
            return 0.0

        # px_per_m: how many image pixels correspond to 1m of local right movement.
        # Sign encodes whether right_world points image-right (+) or image-left (-).
        px_per_m = sofa_r_px - sofa_px

        # Degenerate case: right_world is nearly along camera depth direction.
        # In this case, moving "right" on the furniture shifts y-pixel (depth cue)
        # rather than x-pixel. Use py_per_m to infer the left/right position.
        if abs(px_per_m) < 10.0:
            py_per_m = sofa_r_py - sofa_py
            if abs(py_per_m) > 5.0:
                bbox_cy = (box_px[1] + box_px[3]) / 2.0
                delta_py = bbox_cy - sofa_py
                cx_local_y = delta_py / py_per_m
                clamped_y = float(np.clip(cx_local_y, -half_w, half_w))
                print(f"    [bbox_cx] sofa_py={sofa_py:.0f}  bbox_cy={bbox_cy:.0f}  "
                      f"py/m={py_per_m:.1f}  cx_local(y)={cx_local_y:.3f}  → {clamped_y:.3f} "
                      f"(y-fallback: px/m={px_per_m:.2f} degenerate)")
                return clamped_y
            return 0.0

        # bbox center in image space (box_px is in original photo coords)
        bbox_cx = (box_px[0] + box_px[2]) / 2.0

        # Anchor selection: prefer the furniture's PHOTO bbox centre, which is
        # what the decoration's `box_px` was measured against.  Fall back to
        # the 3D-projected sofa centre when the photo bbox isn't available.
        # If the two anchors disagree by more than _ANCHOR_MISMATCH_PX, the
        # 3D sofa is misaligned with the photo sofa — soft-shrink the hint
        # (toward 0) by the misalignment ratio so silhouette doesn't drag the
        # pillow off-centre on a misaligned sofa.
        _ANCHOR_MISMATCH_PX = 40.0
        if furn_photo_box_px is not None:
            _photo_anchor_x = (float(furn_photo_box_px[0]) + float(furn_photo_box_px[2])) / 2.0
            _anchor_x = _photo_anchor_x
            _anchor_kind = "photo-bbox"
        else:
            _anchor_x = sofa_px
            _anchor_kind = "3d-projection"

        delta_px = bbox_cx - _anchor_x          # signed image offset from anchor
        cx_local = delta_px / px_per_m          # convert to metres in local frame

        # Misalignment handling:
        #   ≤ _ANCHOR_MISMATCH_PX (40)  → trust the bbox hint as-is
        #   between 40 and _HARD_MISMATCH_PX (110) → soft-shrink toward 0
        #   > _HARD_MISMATCH_PX → ABANDON the pixel hint entirely
        #
        # Beyond ~110 px the bbox center no longer corresponds to a real
        # spot on the world-positioned furniture (the placement and the
        # photo disagree about where the furniture is in the room), so any
        # X derived from pixel offsets is misleading.  Return NaN as a
        # sentinel — the caller falls back to the VLM's relative
        # `side`/`depth` placement (e.g. "centered on the sofa, against the
        # back rest"), which is more robust when projection alignment fails.
        _HARD_MISMATCH_PX = 110.0
        if furn_photo_box_px is not None:
            _mismatch_px = abs(_anchor_x - sofa_px)
            if _mismatch_px > _HARD_MISMATCH_PX:
                print(f"    [bbox_cx] anchor_mismatch_px={_mismatch_px:.0f} > "
                      f"{_HARD_MISMATCH_PX:.0f} → ABANDONING bbox hint, "
                      f"caller should use VLM relative side/depth instead")
                return float("nan")
            if _mismatch_px > _ANCHOR_MISMATCH_PX:
                _shrink = max(0.25, 1.0 - (_mismatch_px - _ANCHOR_MISMATCH_PX) / 80.0)
                _cx_orig = cx_local
                cx_local *= _shrink
                print(f"    [bbox_cx] anchor_mismatch_px={_mismatch_px:.0f} > "
                      f"{_ANCHOR_MISMATCH_PX:.0f} → shrunk hint ×{_shrink:.2f} "
                      f"({_cx_orig:.3f} → {cx_local:.3f})")

        # If the projection maps the pillow outside the sofa's physical extent,
        # the reference-photo camera doesn't match the 3D camera well enough to
        # use the absolute position. Fall back to center so packing order (from
        # the within-group sort by bbox_cx) still produces the correct L→R sequence.
        if abs(cx_local) > half_w:
            if rank_hint is not None:
                clamped = float(np.clip(rank_hint, -half_w, half_w))
                print(f"    [bbox_cx] sofa_px={sofa_px:.0f}  bbox_cx={bbox_cx:.0f}  "
                      f"px/m={px_per_m:.1f}  cx_local={cx_local:.3f} outside sofa "
                      f"[±{half_w:.3f}m] → rank hint {clamped:.3f}")
                return clamped
            print(f"    [bbox_cx] sofa_px={sofa_px:.0f}  bbox_cx={bbox_cx:.0f}  "
                  f"px/m={px_per_m:.1f}  cx_local={cx_local:.3f} outside sofa "
                  f"[±{half_w:.3f}m] → center fallback 0.0")
            return 0.0

        print(f"    [bbox_cx] sofa_px={sofa_px:.0f}  bbox_cx={bbox_cx:.0f}  "
              f"anchor={_anchor_kind}@{_anchor_x:.0f}  px/m={px_per_m:.1f}  "
              f"cx_local={cx_local:.3f}  → {cx_local:.3f}")
        return float(cx_local)
    except Exception as _e:
        print(f"    [bbox_cx] projection failed: {_e}")
        return float(rank_hint) if rank_hint is not None else 0.0


def _place_against_back(
    furn: dict,
    decor_size: dict,
    side: str,
    tracker: _SurfaceTracker,
    out_dir: "Path | None" = None,
    preferred_cx_hint: float = 0.0,
) -> np.ndarray:
    """Return world-space position for a pillow/cushion against the sofa/chair back.

    Never returns None — pillows are always placed (they sit flat on the seat
    against the back rest; their footprint is thin so they rarely truly block).
    If all preferred positions collide, pack them side-by-side clamped to the
    furniture width.
    """
    sw  = furn["size_m"]["width_m"]
    sd  = furn["size_m"]["depth_m"]
    dw  = decor_size["width_m"]
    dd  = decor_size["depth_m"]

    # Target: pillow leaning against the back rest.
    # Start at the ideal back-rest contact position; _push_to_visible_face_against_back
    # will shift forward along +Z if this ends up inside the back-rest mesh.
    # Note: the prior rotation-clamp (max back-lean = 5cm for rotated sofas)
    # was removed — the visual drift it was solving was actually caused by the
    # -2cm "lean into back" nudge in _push_to_visible_face_against_back, which
    # is now skipped for rotated sofas via _LEAN_NUDGE.  Clamping cz_local
    # here was over-correcting and leaving a visible gap between the pillow's
    # back face and the back cushion.  Use the full geometric lean.
    cz_local = -(sd / 2.0 - _BACK_MARGIN - dd / 2.0)

    # Usable cx range must leave room for ARM RESTS, not just the edge margin.
    # A typical sofa arm rest is 12-18 cm thick, so we reserve 15% of the sofa
    # half-width on each side (plus the normal edge/pillow-half-width margin)
    # as "off-limits" for pillow centres.  Previously we used the full sofa
    # half-width, so pillows at cx=-0.804 on a ~1m-half-width sofa extended
    # into the arm rest and visually collided with it.
    _ARM_FRAC = 0.15
    _arm_reserve = sw * _ARM_FRAC
    half_w = max(sw / 2 - dw / 2 - _EDGE_MARGIN - _arm_reserve, 0.0)

    # Fill-the-seat snap: when the pillow's footprint takes up most of the
    # sofa's usable width (≥70% of seat-minus-arms), there's no meaningful
    # left/right placement to do — any non-zero cx_local would push the
    # pillow's edge past the sofa.  Snap to centre regardless of the photo
    # bbox hint (which may be unreliable: pillow re-assigned to a different
    # sofa, occluded silhouette, photo-vs-3D misalignment).
    _seat_usable_w = max(sw - 2.0 * _arm_reserve, 0.01)
    _fill_ratio = dw / _seat_usable_w
    _force_centre = _fill_ratio >= 0.70

    # Place directly at the preferred cx hint — no per-surface collision avoidance.
    # Pillows are intentionally packed close together; any genuine 3D overlap is
    # handled later by the world-space AABB nudge system.
    # ``preferred_cx_hint`` may be NaN: that is the sentinel from
    # `_compute_preferred_cx` meaning "the photo-bbox X is too far from the
    # world-projected furniture to be trusted" — in that case we use the
    # VLM's relative ``side`` directly, treating the pillow as "centered on
    # the sofa back" rather than constrained to a pixel column.
    _hint_is_nan = (preferred_cx_hint != preferred_cx_hint)
    # Loosened safe range: drop the arm-rest reserve for the bbox-hint
    # "is it small enough to apply" check.  The hint's own magnitude
    # (≤ a few cm) is still tiny enough that the pillow's edge will
    # not actually clash with the arm; the arm-reserve was only
    # justified for VLM `side='left'` / `side='right'` placements
    # which can target the arm directly.
    _safe_half_no_arm = max(sw / 2.0 - dw / 2.0 - _EDGE_MARGIN, 0.0)
    if _force_centre:
        # When the pillow nearly fills the seat AND VLM said centre,
        # the pillow should sit at the centre of the VISIBLE sofa
        # (not metadata centre).  On well-formed sofas these agree
        # (mesh centre ≈ metadata 0); on broken generated meshes (e.g.
        # sofa[1] whose mesh is offset ~22cm from metadata) they
        # disagree, and metadata-centre leaves the pillow visually
        # off the visible sofa.
        # Resolve cx by probing the seat-band of the actual mesh for
        # its lateral centre.  We do NOT clamp to metadata here:
        # when mesh and metadata disagree, visual placement (mesh)
        # wins.  The placement loop sets `_pos_locked` so post-loop
        # passes (bbox_contain, pos_refine, final_clamp) don't pull
        # the pillow back toward metadata.
        _cx_target = 0.0
        _cx_src    = "metadata-centre fallback"
        try:
            _mesh_for_force = (_get_furn_scene_mesh(out_dir)
                               if out_dir is not None else None)
        except Exception:
            _mesh_for_force = None
        if _mesh_for_force is not None:
            try:
                _R_force      = _rot_matrix(furn)
                _f_pos_force  = np.array(furn["position_m"], dtype=np.float64)
                _f_h_force    = float(furn.get("eff_h",
                                               furn["size_m"].get("height_m", 0.85)))
                _v_w_force    = np.asarray(_mesh_for_force.vertices,
                                            dtype=np.float64)
                _wax = (sw + sd) / 2.0 / np.sqrt(2.0) + 0.05
                _region_force = (
                    (np.abs(_v_w_force[:, 0] - _f_pos_force[0]) <= _wax)
                    & (np.abs(_v_w_force[:, 2] - _f_pos_force[2]) <= _wax)
                    & (_v_w_force[:, 1] >= _f_pos_force[1] - 0.05)
                    & (_v_w_force[:, 1] <= _f_pos_force[1] + _f_h_force + 0.10)
                )
                _v_force = _v_w_force[_region_force]
                if len(_v_force) >= 100:
                    _v_loc_force = (_v_force - _f_pos_force) @ _R_force
                    _seat_mask_f = (
                        (_v_loc_force[:, 1] > _f_h_force * _SEAT_FRAC - 0.05)
                        & (_v_loc_force[:, 1] < _f_h_force + 0.10)
                        & (np.abs(_v_loc_force[:, 0]) <= sw / 2.0 + 0.05)
                        & (np.abs(_v_loc_force[:, 2]) <= sd / 2.0 + sd)
                    )
                    if _seat_mask_f.sum() >= 30:
                        _xs_f      = _v_loc_force[_seat_mask_f, 0]
                        _mesh_min  = float(_xs_f.min())
                        _mesh_max  = float(_xs_f.max())
                        _cx_target = (_mesh_min + _mesh_max) / 2.0
                        _cx_src    = (f"mesh-centre seat-band X "
                                      f"[{_mesh_min:+.3f},{_mesh_max:+.3f}]")
                        # Stash for downstream consumers (arm_rest clamp etc.)
                        # — sofa metadata bbox is symmetric around the mesh
                        # centre, but the actual seat band can be far off
                        # (sofa[1] in office8: seat X ∈ [-0.425, -0.011],
                        # centred at -0.218).  Without this the arm_rest
                        # shift clamps to ±metadata/2 and pushes the pillow
                        # past the visible seat edge.
                        furn["_seat_band_x"] = (_mesh_min, _mesh_max)
            except Exception:
                pass
        cx_local = float(_cx_target)
        # Tag the furniture so the placement loop knows to lock the
        # pillow's position (skip metadata-clamp post-passes).
        furn["_force_centre_used"] = True
        print(f"    [against_back] pillow fills {_fill_ratio*100:.0f}% of seat width "
              f"({dw:.2f}m on {_seat_usable_w:.2f}m usable) — centre "
              f"target {_cx_target:+.3f} from {_cx_src} (NO metadata clamp; "
              f"locked to follow visible mesh) → cx={cx_local:+.3f}")
    else:
        # Non-force-centre placement: use VLM `side` as the X cue —
        # NOT the photo-bbox pixel position.  The 3D sofa pose
        # generally doesn't match the reference photo's sofa pose,
        # so projecting bbox pixels into the 3D sofa frame produces
        # arbitrary offsets that drag the pillow off-axis.
        # `side` is a relative cue ('left'/'centre'/'right') — that
        # remains valid regardless of pose mismatch.
        _vlm_cx = _candidate_cx(sw, dw, side)[0]
        cx_local = float(np.clip(_vlm_cx, -half_w, half_w))
        print(f"    [against_back] VLM side='{side}' (relative; bbox-cx "
              f"hint {preferred_cx_hint:+.3f} ignored) → cx={cx_local:+.3f}")

    right_world, _, _ = _world_axes(furn)
    R = _rot_matrix(furn)
    local_z_world = R[:, 2]

    # Y: seat surface; pillow base rests on seat
    world_pos = _seat_surface_center(furn).copy()
    world_pos += cx_local * right_world
    world_pos += cz_local * local_z_world

    # Push forward minimally + snap Y to actual seat surface via raycast
    world_pos = _push_to_visible_face_against_back(world_pos, furn, decor_size, out_dir=out_dir)

    return world_pos


# ── Orientation ───────────────────────────────────────────────────────────────

_RY180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)


_SYMMETRIC_OBJECTS = frozenset({
    "lamp", "desk lamp", "floor lamp", "vase", "plant", "book", "books",
    "mug", "cup", "tray", "pillow", "cushion", "remote",
})

# Map view label → local front vector for the world-cardinal renderer.
# We place the camera at +Z (View A), -Z (View B), +X (View C), -X (View D),
# all looking toward the origin with +Y up.  "front_view=A" means the front
# face's outward normal points to +Z in the mesh's local frame.
_VIEW_TO_FRONT = {
    "A": np.array([0, 0,  1], dtype=np.float64),
    "B": np.array([0, 0, -1], dtype=np.float64),
    "C": np.array([1, 0,  0], dtype=np.float64),
    "D": np.array([-1, 0, 0], dtype=np.float64),
}

# Image-right direction (in mesh local frame) shown on the right side of each
# rendered view.  With +Y up and the camera looking at the origin:
#   A (cam +Z, fwd −Z): right = +Y × (−fwd) = +Y × +Z = +X
#   B (cam −Z, fwd +Z): right = +Y × −Z              = −X
#   C (cam +X, fwd −X): right = +Y × +X              = −Z
#   D (cam −X, fwd +X): right = +Y × −X              = +Z
# Used as the reflection axis when correcting a left↔right mirror flip in
# the picked view.
_VIEW_TO_RIGHT = {
    "A": np.array([1.0,  0.0,  0.0]),
    "B": np.array([-1.0, 0.0,  0.0]),
    "C": np.array([0.0,  0.0, -1.0]),
    "D": np.array([0.0,  0.0,  1.0]),
}


def _render_four_views(mesh, size: int = 256) -> "Image.Image":
    """Render GLB mesh from 4 cardinal directions; return a 2×2 stitched PIL image.

    Cameras at +Z, -Z, +X, -X all looking toward the origin from distance 2.
    """
    from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes
    from object_placement.wall_mounted.placements.fill_openings import (
        _get_vertex_colors, _rasterize_vc_tri,
    )

    # Normalise mesh to unit cube
    b = mesh.bounds
    centre = (b[0] + b[1]) / 2.0
    ext    = b[1] - b[0]
    sc     = 1.0 / max(ext) if max(ext) > 1e-9 else 1.0
    verts  = (mesh.vertices.astype(np.float64) - centre) * sc
    vc     = _get_vertex_colors(mesh)

    cam_dirs = [
        np.array([0, 0,  2.0]),   # A — camera at +Z, looks toward -Z (sees +Z face)
        np.array([0, 0, -2.0]),   # B — camera at -Z, looks toward +Z (sees -Z face)
        np.array([ 2.0, 0, 0]),   # C — camera at +X
        np.array([-2.0, 0, 0]),   # D — camera at -X
    ]
    up = np.array([0, 1, 0], dtype=np.float64)
    origin = np.array([0, 0, 0], dtype=np.float64)

    hfov = 45.0
    fx   = size / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx = cy = size / 2.0
    near = 0.01

    panels = []
    for cam_pos in cam_dirs:
        right_v, up_v, fwd_v = _camera_axes(cam_pos, origin, up)
        buf  = np.full((size, size, 3), 240, dtype=np.uint8)
        zbuf = np.full((size, size), np.inf, dtype=np.float32)

        from object_placement.wall_mounted.wall_mounted_object_placement import _project_vertex
        proj = [_project_vertex(v, cam_pos, right_v, up_v, fwd_v, fx, cx, cy)
                for v in verts]
        for fi in range(len(mesh.faces)):
            i0, i1, i2 = mesh.faces[fi]
            px0,py0,zc0 = proj[i0]; px1,py1,zc1 = proj[i1]; px2,py2,zc2 = proj[i2]
            if zc0 <= near or zc1 <= near or zc2 <= near:
                continue
            pts = np.array([[px0,py0,zc0],[px1,py1,zc1],[px2,py2,zc2]], dtype=np.float32)
            col = np.array([vc[i0], vc[i1], vc[i2]], dtype=np.uint8)
            _rasterize_vc_tri(buf, zbuf, pts, col)
        panels.append(buf)

    # Stitch 2×2 with visible quadrant labels
    labels = ["A", "B", "C", "D"]
    grid = np.full((size*2, size*2, 3), 200, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, 2)
        grid[r*size:(r+1)*size, c*size:(c+1)*size] = p
    grid_img = Image.fromarray(grid)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid_img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except Exception:
            font = ImageFont.load_default()
        for i, label in enumerate(labels):
            r, c = divmod(i, 2)
            ox, oy = c * size + 8, r * size + 8
            # Shadow
            draw.text((ox+2, oy+2), label, fill=(0, 0, 0), font=font)
            draw.text((ox, oy), label, fill=(255, 64, 0), font=font)
    except Exception:
        pass
    return grid_img


_FRONT_DETECT_PROMPT = """\
You are examining a 3D model of a "{phrase}" to determine its front face.

Image 1 (reference): a crop of this "{phrase}" from the original room photo.
  This shows the side that faces the camera — the user-visible functional front.

Image 2 (4-view grid): the same 3D model rendered from 4 directions:
  A (top-left)   — model seen from +Z direction
  B (top-right)  — model seen from -Z direction (opposite of A)
  C (bottom-left)  — model seen from +X direction (right side)
  D (bottom-right) — model seen from -X direction (left side)

Task:
Match Image 1 to one of the four views in Image 2.
Look for colour, shape, and surface detail that appear in both images:
  - Monitor: front = dark/black screen face; back = white/grey plastic casing
  - Keyboard: front = top surface with visible keys
  - Laptop: front = keyboard/screen side

Whichever view in Image 2 shows the same side as Image 1 is the FRONT view.

Return ONLY valid JSON:
{{
  "front_view": "A" or "B" or "C" or "D",
  "reasoning": "what colour/feature in Image 1 matches which view in Image 2"
}}
"""


def _detect_decoration_front(
    glb_path: Path,
    crop_path: Path | None,
    phrase: str,
) -> np.ndarray | None:
    """Render 4 cardinal views of the decoration GLB, ask VLM which is the front.

    Returns a local-space unit vector for the front direction, or None.
    """
    if any(kw in phrase.lower() for kw in _SYMMETRIC_OBJECTS):
        return None  # symmetric — no meaningful front

    try:
        import trimesh
        mesh = trimesh.load(str(glb_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        # World-cardinal 4-view render of the ORIGINAL mesh, untouched.
        # We do NOT pre-rotate or detilt the hunyuan output here: the only
        # downstream rotation is a Y-yaw inside `_surface_rotation`.
        grid_img = _render_four_views(mesh)

        # Save debug grid next to GLB
        debug_path = glb_path.with_suffix(".front_detect.png")
        grid_img.save(str(debug_path))
        print(f"    [front_detect] 4-view grid → {debug_path.name}")

        if crop_path is None or not crop_path.exists():
            print(f"    [front_detect] no crop — skipping VLM front check")
            return None

        crop_img = Image.open(crop_path).convert("RGB")

        content = [
            {"type": "text",
             "text": _FRONT_DETECT_PROMPT.format(phrase=phrase)},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_encode_pil(crop_img)}"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_encode_pil(grid_img)}"}},
        ]
        result = _vlm_call_json(content, max_tokens=150)
        if result is not None:
            view = str(result.get("front_view", "")).upper().strip()
            reason = result.get("reasoning", "")
            front = _VIEW_TO_FRONT.get(view)
            right = _VIEW_TO_RIGHT.get(view)
            if front is not None and right is not None:
                front = front.copy()
                right = right.copy()
                print(f"    [front_detect] front=View {view} → local={front.tolist()}  ({reason})")

                # ── Mirror-flip check ──────────────────────────────────
                # Single-image-to-3D models sometimes emit left-right
                # mirrored meshes.  Compare the chosen front view against
                # the reference crop; if asymmetric features are reversed,
                # reflect the mesh across the plane perpendicular to the
                # chosen view's RIGHT axis — so the mirror flips left↔right
                # in the picked view (e.g. for view C/D the right axis is
                # ±Z, not ±X, so a Z-mirror is correct).
                try:
                    chosen_panel = _crop_view_panel(grid_img, view)
                    if chosen_panel is not None:
                        mirror_result = _vlm_check_mirror(crop_img, chosen_panel, phrase)
                        if mirror_result is not None and mirror_result.get("mirrored"):
                            feature = mirror_result.get("feature", "")
                            mreason = mirror_result.get("reasoning", "")
                            print(f"    [mirror_check] MIRRORED — feature='{feature}' "
                                  f"({mreason})")
                            if _apply_mirror_to_glb(glb_path, right):
                                # Reflecting vertices across the plane normal
                                # to `right` swaps any feature on the +right
                                # side with the -right side.  Apply the same
                                # Householder reflection to `front` so it
                                # still points at the (now mirrored) front
                                # face — otherwise downstream
                                # `_surface_rotation` would aim the BACK at
                                # the camera.
                                front_dot_right = float(np.dot(front, right))
                                front = front - 2.0 * front_dot_right * right
                                print(f"    [mirror_check] front vector reflected "
                                      f"across right={right.tolist()} → "
                                      f"{front.tolist()}")
                                try:
                                    import trimesh as _tm
                                    _new_mesh = _tm.load(str(glb_path), force="mesh")
                                    if isinstance(_new_mesh, _tm.Scene):
                                        _new_mesh = _tm.util.concatenate(_new_mesh.dump())
                                    _render_four_views(_new_mesh).save(str(debug_path))
                                except Exception as _re:
                                    print(f"    [mirror_check] re-render skipped: {_re}")
                                print(f"    [mirror_check] applied mirror across "
                                      f"right axis to {glb_path.name}, re-saved")
                        elif mirror_result is not None:
                            print(f"    [mirror_check] OK — "
                                  f"{mirror_result.get('reasoning', '')}")
                except Exception as _me:
                    print(f"    [mirror_check] skipped: {_me}")

                return front
            print(f"    [front_detect] unrecognised view '{view}'")
        return None

    except Exception as e:
        import traceback
        print(f"    [front_detect] FAILED: {e}")
        print(f"      {traceback.format_exc().splitlines()[-1]}")
        return None


# ── Mirror-flip detection / correction for hunyuan3d outputs ──────────────────
#
# Hunyuan3D and similar single-image-to-3D models occasionally emit meshes
# whose visible features (logos, button placements, screen reflections) are
# LEFT-RIGHT MIRRORED relative to the source photo.  Geometry looks fine,
# texture is fine, but a viewer perceives "reversed reflections" because
# the asymmetric layout is reversed.
#
# Detection strategy: after _front_detect_one picks the chosen front view,
# crop that single panel out of the 2×2 grid and ask the VLM "are visible
# asymmetric features in the rendered front view mirrored vs the reference
# photo?".  If yes, apply a mirror-X transform to the mesh and re-export.

_MIRROR_CHECK_PROMPT = """\
You are inspecting whether a 3D model has been generated with LEFT-RIGHT
MIRRORED features compared to a reference photo.

Image 1: REFERENCE PHOTO of the real object — "{phrase}".  This is the
ground truth.

Image 2: 3D model rendered from the same front-facing direction.

Compare the two images for LEFT-RIGHT asymmetric features ONLY:
  - Logo placement (e.g. "Apple logo on the LEFT vs RIGHT of the back")
  - Button / port positions
  - Screen reflections that show a clear left-vs-right pattern
  - Any other asymmetric visual element

IMPORTANT:
  - Only flag mirroring if the asymmetric features are visibly reversed
    (left↔right) between the reference photo and the rendered view.
  - Do NOT flag mirroring for slight perspective differences, lighting
    variations, or overall colour shifts.
  - If the object is rotationally symmetric (lamp shade, vase, ball,
    plain pillow, plain books with no spine text visible), respond
    "mirrored": false.

Reply ONLY with JSON:
{{"mirrored": true/false, "feature": "<which asymmetric feature you used to decide>", "reasoning": "<one sentence>"}}
"""


def _crop_view_panel(grid_img: Image.Image, view: str) -> Image.Image | None:
    """Extract one of A/B/C/D panels from the 2×2 grid produced by
    `_render_four_views`.  Layout: A=top-left, B=top-right, C=bottom-left,
    D=bottom-right (matches the ravel(divmod(i, 2)) loop in the renderer)."""
    W, H = grid_img.size
    half_w, half_h = W // 2, H // 2
    boxes = {
        "A": (0,      0,      half_w, half_h),
        "B": (half_w, 0,      W,      half_h),
        "C": (0,      half_h, half_w, H),
        "D": (half_w, half_h, W,      H),
    }
    box = boxes.get(view.upper())
    return grid_img.crop(box) if box else None


def _vlm_check_mirror(crop_img: Image.Image, view_img: Image.Image,
                      phrase: str) -> dict | None:
    """Ask the VLM whether the rendered view is LEFT-RIGHT mirrored relative
    to the reference crop.  Returns parsed JSON or None on failure."""
    content = [
        {"type": "text",
         "text": _MIRROR_CHECK_PROMPT.format(phrase=phrase)},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(crop_img)}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_encode_pil(view_img)}"}},
    ]
    return _vlm_call_json(content, max_tokens=180)


def _apply_mirror_to_glb(glb_path: Path, mirror_axis_local: np.ndarray) -> bool:
    """Reflect the mesh in ``glb_path`` across the plane perpendicular to
    ``mirror_axis_local`` (a unit vector in the mesh's local frame) and
    re-save in place.  Inverts face winding after the reflection so normals
    stay outward.  Returns True on success.

    The chosen axis is typically the rendered view's "right" vector — that
    way, mirroring reflects left↔right in the picked front view regardless
    of how the mesh is yawed inside its own local frame.
    """
    try:
        import trimesh
        axis = np.asarray(mirror_axis_local, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            print(f"    [mirror_fix] degenerate mirror axis, skipping")
            return False
        axis = axis / norm
        scene = trimesh.load(str(glb_path), force="scene", process=False)
        if isinstance(scene, trimesh.Trimesh):
            scene = trimesh.Scene([scene])
        M = trimesh.transformations.scale_matrix(
            -1.0, origin=[0, 0, 0], direction=axis.tolist())
        for name, g in list(scene.geometry.items()):
            if isinstance(g, trimesh.Trimesh):
                g.apply_transform(M)
                g.invert()                # restore correct winding after reflection
        scene.export(str(glb_path))
        return True
    except Exception as e:
        print(f"    [mirror_fix] export failed: {e}")
        return False


def _front_to_yaw_rotation(front_local: np.ndarray) -> np.ndarray:
    """Given a local front vector, return a 3×3 Y-rotation matrix R such that
    R @ front_local ≈ +Z (the canonical outward direction after furniture R is applied).
    This lets _surface_rotation compose correctly.
    """
    # We want the local front to map to +Z after the per-decoration rotation.
    # Compute the angle between front_local (projected to XZ) and +Z.
    fx, _, fz = float(front_local[0]), float(front_local[1]), float(front_local[2])
    angle = np.arctan2(fx, fz)   # angle to rotate so (fx,fz) → (0,1)=(+Z)
    c, s = np.cos(angle), np.sin(angle)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return Ry


def _surface_rotation(furn: dict, facing: str = "any",
                      front_local: np.ndarray | None = None,
                      toward_dir: np.ndarray | None = None) -> list[list[float]]:
    """Rotation matrix for an on_surface decoration.

    facing="toward_room": rotate the GLB so its detected front (front_local)
        points along toward_dir — the furniture's own outward axis (R_furn @ +Z),
        i.e. the direction a user faces when using the furniture.
    facing="any": inherit furniture yaw R only.
    front_local: local-space front vector detected from 4-view VLM check.
                 If None, assumes GLB front is in -Z.
    toward_dir: world-space horizontal unit vector = furniture's outward front direction
                (only used when facing="toward_room").
    """
    if facing == "toward_room" and toward_dir is not None:
        # Direct yaw rotation: align front_local → toward_dir in XZ plane
        src = np.array(front_local if front_local is not None else [0, 0, -1],
                       dtype=np.float64)
        tgt = np.array(toward_dir, dtype=np.float64)
        a_src = np.arctan2(float(src[0]), float(src[2]))
        a_tgt = np.arctan2(float(tgt[0]), float(tgt[2]))
        angle = a_tgt - a_src
        c, s = np.cos(angle), np.sin(angle)
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]

    R = _rot_matrix(furn)

    # Align GLB front to +Z in furniture-local space, then apply furniture R
    if front_local is not None:
        R_align = _front_to_yaw_rotation(front_local)
    else:
        R_align = np.eye(3, dtype=np.float64)

    if facing == "toward_room":
        return (R @ _RY180 @ R_align).tolist()

    # facing == "any": decorative items (books, vases, trays, …) have no
    # functional facing.  Do NOT inherit the support furniture's full rotation
    # matrix — a coffee table whose reconstructed frame carries a yaw (e.g. 39°)
    # would tip the books diagonally even though the table looks square.  Orient
    # in the WORLD frame instead, so the item aligns to the room axes; the later
    # 4-way yaw-pick refines it to the world-cardinal orientation that best
    # matches the reference photo.
    return R_align.tolist()


def _against_back_rotation(furn: dict, lean_deg: float = 20.0) -> list[list[float]]:
    """Rotation for against_back: front toward room, top leaning toward sofa back.

    `R_lean` (R_X(+lean)) tilts the pillow's local +Y toward its local +Z.
    We need the pillow's local +Z to point at the sofa's back-rest in
    world.  The mapping depends on the sofa's `_backrest_sign` (set by
    `_push_to_visible_face_against_back` via `_detect_backrest_local_z_sign`):

      sign = −1 (default convention, backrest at sofa local −Z):
        R_face = R_furn @ _RY180  (flips +Z, +X) →
        GLB +Z ↦ −R_furn[:,2] = toward backrest ✓

      sign = +1 (inverted convention, backrest at sofa local +Z):
        R_face = R_furn  (no flip) →
        GLB +Z ↦ +R_furn[:,2] = toward backrest ✓
    """
    backrest_sign = float(furn.get("_backrest_sign", -1.0))
    R_furn = _rot_matrix(furn)
    if backrest_sign > 0:
        R_face = R_furn.copy()
    else:
        R_face = R_furn @ _RY180
    lr = np.radians(lean_deg)
    c, s = np.cos(lr), np.sin(lr)
    R_lean = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    return (R_face @ R_lean).tolist()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(output_dir: str | Path, first_only: bool = False, animate: bool = False,
        phrase_contains: "str | None" = None,
        pillow_cx_override: "float | None" = None,
        pillow_cz_override: "float | None" = None,
        pillow_y_offset: "float | None" = None,
        animate_gif_width: int = 1500, animate_frame_ms: int = 800,
        max_objects: int | None = None,
        pillow_slide_frames: int = 4,
        detect_missing: bool = False,
        use_vlm: bool = True) -> Path:
    out_dir      = Path(output_dir)
    decor_dir    = out_dir / "decorations"
    placements_dir = decor_dir / "placements"
    inpaint_dir  = decor_dir / "inpainted"
    results_path = decor_dir / "segment_results.json"
    furn_path    = out_dir / "furniture" / "furniture_placements.json"
    out_path     = placements_dir / "decoration_placements.json"
    placements_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale per-step renders from previous runs so the animation GIF
    # only contains frames from this run, in correct chronological order.
    # Without this, sorted-by-mtime still works for THIS run's frames but
    # any leftover frames from a different run with different filenames
    # (e.g. from a re-orient pass that previously produced different
    # _oriented{N}.png suffixes) sit in the directory and pollute the glob.
    for _stale in placements_dir.glob("render_step_*.png"):
        try:
            _stale.unlink()
        except Exception:
            pass

    if not results_path.exists():
        raise FileNotFoundError(f"segment_results.json not found: {results_path}")
    if not furn_path.exists():
        raise FileNotFoundError(f"furniture_placements.json not found: {furn_path}")

    with open(results_path) as f:
        seg_data = json.load(f)
    with open(furn_path) as f:
        furn_list: list[dict] = json.load(f)

    # ── Resolve duplicate furniture indices ─────────────────────────────────
    # When placement_analysis has multiple instances at the same `index`
    # (e.g. two sofas idx=1, one per wall), the existing dict-by-index
    # lookup loses every instance after the first.  Rename duplicates to
    # unique synthetic indices and redistribute decoration segments that
    # reference the original index round-robin across the new indices.
    from collections import defaultdict as _ddict
    _idx_groups: dict[int, list[int]] = _ddict(list)
    for _i, _f in enumerate(furn_list):
        _idx_groups[int(_f.get("index", -1))].append(_i)

    _existing_indices = {int(_f.get("index", -1)) for _f in furn_list}
    _synthetic_base = max(_existing_indices) + 1000 if _existing_indices else 1000
    for _orig_idx, _fl_indices in list(_idx_groups.items()):
        if len(_fl_indices) <= 1 or _orig_idx < 0:
            continue
        _new_indices = [_orig_idx]
        for _fl_i in _fl_indices[1:]:
            _new_idx = _synthetic_base
            while _new_idx in _existing_indices:
                _new_idx += 1
            furn_list[_fl_i]["index"] = _new_idx
            _existing_indices.add(_new_idx)
            _new_indices.append(_new_idx)
            _synthetic_base = _new_idx + 1
        print(f"  [duplicate-resolve] {len(_fl_indices)} furniture instances "
              f"share index={_orig_idx} → renamed to {_new_indices}")
        _matching = [_s for _s in seg_data.get("segments", [])
                     if int(_s.get("furniture_index", -999)) == _orig_idx]
        if _matching:
            # Duplicate each matching decoration once per furniture instance
            # so identical pillows/cushions appear on every sofa.  First
            # instance keeps the original (its furniture_index is already
            # _orig_idx == _new_indices[0]); each later instance gets a
            # deep copy with the new furniture_index assigned.  All
            # duplicate-resolved segments are flagged so the post-placement
            # VLM verify step doesn't reassign them based on the reference
            # photo (the reference shows decorations on only one of the
            # duplicated furniture pieces, so verify would funnel them all
            # back to that single piece).
            import copy as _copy
            _new_segments = []
            for _s in _matching:
                _s["_duplicate_resolved"] = True
                for _new_idx in _new_indices[1:]:
                    _clone = _copy.deepcopy(_s)
                    _clone["furniture_index"] = _new_idx
                    _clone["_duplicate_resolved"] = True
                    _new_segments.append(_clone)
            seg_data.setdefault("segments", []).extend(_new_segments)
            _dist = {_idx: (len(_matching) if _idx == _orig_idx
                            else sum(1 for _s in _new_segments
                                     if _s.get("furniture_index") == _idx))
                     for _idx in _new_indices}
            print(f"  [duplicate-resolve] duplicated {len(_matching)} "
                  f"decoration segments across {len(_new_indices)} furniture "
                  f"instances: {_dist}  (now {len(_matching) + len(_new_segments)} total)")

    # Photo-bbox lookup for each furniture piece, used by _bbox_preferred_cx
    # to anchor pillow offsets against the sofa's PHOTO position rather than
    # its 3D-projected position (the two diverge when the placement isn't
    # perfectly aligned with the photo, which drags the pillow off centre).
    furn_box_px_by_idx: dict[int, list[int]] = {}
    _furn_seg_path = out_dir / "furniture" / "segment_results.json"
    if _furn_seg_path.exists():
        try:
            _fseg = json.loads(_furn_seg_path.read_text())
            for _s in _fseg.get("segments", []) if isinstance(_fseg, dict) else []:
                _idx = _s.get("index")
                _bp  = _s.get("box_px")
                if _idx is not None and _bp is not None:
                    furn_box_px_by_idx[int(_idx)] = list(_bp)
        except Exception as _fe:
            print(f"[place_decorations] could not load furniture segments: {_fe}")

    segments  = seg_data.get("segments", [])
    if phrase_contains is not None:
        _kw = phrase_contains.strip().lower()
        _before = len(segments)
        segments = [s for s in segments
                    if _kw in (s.get("phrase", "").lower())]
        print(f"[place_decorations] --phrase-contains '{_kw}': "
              f"{_before} → {len(segments)} segments after filtering")
    if first_only:
        segments = segments[:1]
        print("[place_decorations] --first-only: processing 1 segment")
    elif max_objects is not None:
        segments = segments[:max_objects]
        print(f"[place_decorations] --max-objects {max_objects}: processing {len(segments)} segments")

    # Group by furniture_index so all objects for the same piece are placed consecutively.
    # Within each furniture group, against_back items (pillows/cushions) come first so
    # --max-objects N always processes all pillows before non-pillow items that VLM will
    # likely redirect to a different furniture anyway (e.g. a succulent mis-assigned to sofa).
    def _presort_key(s: dict) -> tuple:
        fi   = s.get("furniture_index", 999)
        ph   = s.get("phrase", "").strip().lower()
        # 0 = against_back pillow/cushion (place first within furniture group)
        # 1 = everything else (place after, VLM may reassign to a different piece)
        ab   = 0 if any(kw in ph for kw in ("pillow", "cushion")) else 1
        return (fi, ab, s.get("seg_index", 0))

    segments.sort(key=_presort_key)
    if segments:
        _furn_groups: dict[int, list[str]] = {}
        for _s in segments:
            _fi = _s.get("furniture_index", -1)
            _furn_groups.setdefault(_fi, []).append(_s.get("phrase", "?"))
        _ftype_map = {f["index"]: f.get("type", "?") for f in furn_list}
        print("[place_decorations] placement order by furniture:")
        for _fi, _phrases in _furn_groups.items():
            _ftype = _ftype_map.get(_fi, "floor") if _fi >= 0 else "floor"
            print(f"  [{_fi}] {_ftype} → {', '.join(_phrases)}")

    furn_map  = {f["index"]: f for f in furn_list}

    from PIL import Image as _PILImg
    _PILImg.MAX_IMAGE_PIXELS = None
    camera = _load_camera(out_dir)

    # Pre-warm furniture scene mesh cache for surface raycasting
    _get_furn_scene_mesh(out_dir)

    # Load reference photo for two-image VLM select_furn
    _ref_photo: Image.Image | None = None
    for _cand in [out_dir / "manhattan_reference.png", out_dir / "reference.jpg",
                  out_dir / "reference.png", out_dir / "input.jpg"]:
        if _cand.exists():
            _ref_photo = _PILImg.open(_cand).convert("RGB")
            print(f"  [ref_photo] loaded {_cand.name} ({_ref_photo.width}×{_ref_photo.height})")
            break

    # Load furniture render; build labeled + bbox-highlighted version for VLM
    _furn_render_path = out_dir / "furniture" / "render_furniture_placed.png"
    _labeled_render: Image.Image | None = None   # furniture render with index labels
    _furn_render_ds_w = 1500                      # downsampled render width

    # Determine original reference image width (box_px is in this coordinate space)
    # Use the first segmented mask file as a proxy for the original image size
    _orig_ref_w: float = 3000.0   # default
    _orig_ref_h: float = _orig_ref_w * 3 // 4  # default 4:3 aspect
    _seg_dir = out_dir / "decorations" / "segmented"
    _mask_probe = next(
        (f for f in _seg_dir.iterdir() if f.suffix == ".png" and "mask" in f.name),
        None,
    ) if _seg_dir.exists() else None
    if _mask_probe and _mask_probe.exists():
        try:
            _mask_img = _PILImg.open(_mask_probe)
            _orig_ref_w = float(_mask_img.width)
            _orig_ref_h = float(_mask_img.height)
        except Exception:
            pass

    _furn_render_ds: Image.Image | None = None
    if _furn_render_path.exists() and camera is not None:
        _fr_full = _PILImg.open(_furn_render_path).convert("RGB")
        _fr_native_w = float(_fr_full.width)
        if _fr_full.width > _furn_render_ds_w:
            _furn_render_ds = _fr_full.resize(
                (_furn_render_ds_w, int(_fr_full.height * _furn_render_ds_w / _fr_full.width)),
                _PILImg.LANCZOS,
            )
        else:
            _furn_render_ds = _fr_full
        # box_px is in original-photo space; map to downsampled render space
        # (both share the same camera, so spatial correspondence holds)
        _bbox_scale = float(_furn_render_ds.width) / _orig_ref_w
        print(f"  [bbox] box_px scale: orig_w={int(_orig_ref_w)} → render_ds_w={_furn_render_ds.width}"
              f"  factor={_bbox_scale:.3f}")
        try:
            _labeled_render = _annotate_furniture_labels(_furn_render_ds, furn_list, camera)
            _labeled_render.save(str(placements_dir / "furniture_labels.png"))
            print("  [label] saved furniture_labels.png for inspection")
        except Exception as e:
            print(f"  [label] annotation failed ({e}) — using unlabeled render")
            _labeled_render = _furn_render_ds
    else:
        _bbox_scale = 1.0

    # ── Within-group sort: against_back pillows on same furniture, L→R by bbox_cx ──
    # After the furniture-group pre-sort, reorder same-furniture pillow/cushion segments
    # by their projected image position (left→right) so the collision tracker packs
    # them in the correct visual order without relying on _bbox_preferred_cx accuracy.
    if camera is not None and furn_map:
        try:
            from object_placement.wall_mounted.wall_mounted_object_placement import (
                _camera_axes as _cam_axes_wg,
                _project_vertex as _proj_vx_wg,
            )
            _cp_wg = np.array(camera["position_m"], dtype=np.float64)
            _la_wg = np.array(camera["look_at_m"],  dtype=np.float64)
            _up_wg = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
            _rv_wg, _uv_wg, _fv_wg = _cam_axes_wg(_cp_wg, _la_wg, _up_wg)
            _iw_wg = int(_orig_ref_w)
            _ih_wg = int(_orig_ref_h)
            _fx_wg = _iw_wg / (2.0 * np.tan(np.radians(float(camera["hfov_deg"]) / 2.0)))
            _cxc_wg, _cyc_wg = _iw_wg / 2.0, _ih_wg / 2.0

            def _ppm_sign_wg(furn_f: dict) -> float:
                """Sign of px_per_m for furn's right axis: +1 if right_world→image-right."""
                try:
                    rw, _, _ = _world_axes(furn_f)
                    fp = np.array(furn_f["position_m"], dtype=np.float64)
                    fh = float(furn_f.get("eff_h", furn_f["size_m"].get("height_m", 0.8)))
                    sy = fp[1] + fh * _SEAT_FRAC
                    sc3d = np.array([fp[0], sy, fp[2]], dtype=np.float64)
                    spx, _, sz = _proj_vx_wg(sc3d, _cp_wg, _rv_wg, _uv_wg, _fv_wg,
                                             _fx_wg, _cxc_wg, _cyc_wg)
                    srpx, _, srz = _proj_vx_wg(sc3d + rw, _cp_wg, _rv_wg, _uv_wg, _fv_wg,
                                               _fx_wg, _cxc_wg, _cyc_wg)
                    if sz <= 0.01 or srz <= 0.01:
                        return 1.0
                    return 1.0 if (srpx - spx) >= 0 else -1.0
                except Exception:
                    return 1.0

            def _is_ab_seg(s: dict) -> bool:
                ph = s.get("phrase", "").strip().lower()
                fi = s.get("furniture_index", -1)
                ft = furn_map.get(fi, {}).get("type", "").lower() if fi >= 0 else ""
                return ft in _SEATING_TYPES and any(kw in ph for kw in ("pillow", "cushion"))

            # First pass: group pillows by (original) furniture_index so we can
            # detect duplicates per piece.
            _dedup_groups: "dict[int, list[dict]]" = {}
            for _wg_s in segments:
                if _is_ab_seg(_wg_s):
                    _dedup_groups.setdefault(_wg_s.get("furniture_index", -1), []).append(_wg_s)

            # Duplicate detection + redistribution: when two pillow bboxes on the
            # same seating piece have heavily-overlapping bboxes (IoU > 0.5), the
            # segmenter detected the SAME reference pillow twice (often under
            # different phrase labels like "pillow" + "cushion pillow", or the
            # same phrase with near-identical coords).  In that case the second
            # detection shouldn't pile onto the same seat — move it to the next
            # closest SAME-TYPE seating piece (e.g. the matching sofa of a
            # sofa-pair, or the nearest spare chair) so we end up with one
            # pillow per piece.  This MUTATES seg['furniture_index'] — then the
            # grouping below re-reads the updated indices.
            def _iou(b1, b2) -> float:
                x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
                if x2 <= x1 or y2 <= y1:
                    return 0.0
                inter = (x2 - x1) * (y2 - y1)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                u = a1 + a2 - inter
                return float(inter) / float(u) if u > 0 else 0.0

            for _wg_fi, _ab_segs_for_fi in list(_dedup_groups.items()):
                if len(_ab_segs_for_fi) < 2:
                    continue
                _orig_type = furn_map.get(_wg_fi, {}).get("type", "").lower() if _wg_fi >= 0 else ""
                if not _orig_type:
                    continue
                # Only consider TARGET seats that are ENTIRELY EMPTY (no other
                # pillow originally assigned).  This avoids piling onto a second
                # sofa that already has its own pillow (e.g. don't make sofa [6]
                # hold 2 pillows when it already has one), and protects cases
                # like living_room9 where every sofa already has 4 pillows
                # assigned — nothing would move there.
                _occupied = set(_dedup_groups.keys())
                _src_pos = np.array(furn_map.get(_wg_fi, {}).get("position_m", [0, 0, 0]),
                                    dtype=np.float64)
                def _cand_key(f):
                    # Prefer the leftmost chair-like spare (user explicitly said
                    # the extra pillow should go to "the leftmost spare chair
                    # with similar matching color"), else other same-type, then
                    # distance.
                    ft = f.get("type", "").lower()
                    _is_chair_bonus = 0 if "chair" in ft else 1
                    _same_type = 0 if ft == _orig_type else 1
                    d = float(np.linalg.norm(
                        np.array(f.get("position_m", [0, 0, 0]), dtype=np.float64)[[0, 2]]
                        - _src_pos[[0, 2]]))
                    return (_is_chair_bonus, _same_type, d)
                _other_seats = sorted(
                    [f for f in furn_list
                     if f.get("index", -2) != _wg_fi
                     and f.get("type", "").lower() in _SEATING_TYPES
                     and "position_m" in f
                     and f.get("index", -2) not in _occupied],
                    key=_cand_key,
                )
                _seen: list[dict] = []
                for _s in _ab_segs_for_fi:
                    _b = _s.get("box_px")
                    if _b is None:
                        _seen.append(_s)
                        continue
                    _is_dup = False
                    for _s_prev in _seen:
                        _bp = _s_prev.get("box_px")
                        if _bp is None:
                            continue
                        # Threshold 0.4 rather than 0.5: real-world duplicate
                        # detections (e.g. office8 seg 4 [857..1274] and seg 2
                        # [919..1236] — nearly the same pillow under slightly
                        # different phrases) give IoU ≈ 0.49, which is clearly
                        # the same pillow but just under the 0.5 line.  Distinct
                        # adjacent pillows typically share < 0.3 IoU.
                        if _iou(_b, _bp) >= 0.4:
                            _is_dup = True
                            break
                    if _is_dup and _other_seats:
                        _target = _other_seats.pop(0)
                        _s["furniture_index"] = int(_target["index"])
                        # Mark the REMAINING seg on the source piece as
                        # distribution-protected: once dedup has taken care of
                        # the duplicate, verify should NOT also move the
                        # legitimate seg off this sofa (or we'd leave the
                        # source sofa with zero pillows — which was the old
                        # bug: seg 4 moved to chair via dedup, then verify
                        # still moved seg 2 from sofa [1] to chair).
                        for _prev in _seen:
                            _prev["_dedup_protect_src"] = True
                        print(f"  [dedup] seg{_s.get('seg_index','?')} ({_s.get('phrase','?')}) "
                              f"duplicate bbox on [{_wg_fi}] ({_orig_type}) → reassigning to "
                              f"empty [{_target['index']}] ({_target.get('type','?')})")
                    else:
                        _seen.append(_s)

            # Now build the real per-furniture index/seg groups AFTER dedup.
            _ab_fi_positions: "dict[int, list[int]]" = {}
            _ab_fi_segs:      "dict[int, list[dict]]" = {}
            for _wg_i, _wg_s in enumerate(segments):
                if _is_ab_seg(_wg_s):
                    _wg_fi = _wg_s.get("furniture_index", -1)
                    _ab_fi_positions.setdefault(_wg_fi, []).append(_wg_i)
                    _ab_fi_segs.setdefault(_wg_fi, []).append(_wg_s)

            for _wg_fi, _wg_pos in _ab_fi_positions.items():
                if len(_wg_pos) < 2:
                    continue
                _wg_furn = furn_map.get(_wg_fi)
                _wg_sign = _ppm_sign_wg(_wg_furn) if _wg_furn else 1.0
                _wg_sorted = sorted(
                    _ab_fi_segs[_wg_fi],
                    key=lambda s: _wg_sign * (
                        (s.get("box_px", [0, 0, 0, 0])[0] + s.get("box_px", [0, 0, 0, 0])[2]) / 2.0
                    ),
                )
                _wg_sw = _wg_furn["size_m"]["width_m"] if _wg_furn else 2.0
                _wg_half_w = _wg_sw / 2.0 * 0.75
                _wg_step = 0.40  # physical meters per pillow slot

                # Color-guided ordering: warm-colored pillows (orange, brown, beige)
                # go to center positions; neutral pillows (grey, white) go to extremes.
                # Pattern: [grey, orange, orange, grey] — common sofa arrangement.
                _wg_final_order: list[dict] = []
                try:
                    _wg_colors = [_classify_seg_color(s, inpaint_dir) for s in _wg_sorted]
                    # Persist color on each segment so placement loop can use it for group-scale
                    for _s, _c in zip(_wg_sorted, _wg_colors):
                        _s["_pillow_color"] = _c
                    _WARM = {"orange", "brown", "beige", "red", "yellow"}
                    _warm_segs  = [s for s, c in zip(_wg_sorted, _wg_colors) if c in _WARM]
                    _neut_segs  = [s for s, c in zip(_wg_sorted, _wg_colors) if c not in _WARM]
                    print(f"    [pillow_color] {[(s['seg_index'], c) for s, c in zip(_wg_sorted, _wg_colors)]}")
                    # If we have a mix, interleave: neutral at extremes, warm in center.
                    if _warm_segs and _neut_segs:
                        _n_left = len(_neut_segs) // 2
                        _wg_final_order = (
                            _neut_segs[:_n_left] + _warm_segs + _neut_segs[_n_left:]
                        )
                    else:
                        _wg_final_order = list(_wg_sorted)
                except Exception as _coe:
                    print(f"    [pillow_color] classification failed: {_coe}")

                if not _wg_final_order:
                    _wg_final_order = list(_wg_sorted)

                # Assign pack-step rank hints for all positions (real + virtual)
                _n_total = len(_wg_final_order)
                for _ri, _seg_o in enumerate(_wg_final_order):
                    _offset = (_ri - (_n_total - 1) / 2.0) * _wg_step
                    _seg_o["_cx_rank_hint"] = float(np.clip(_offset, -_wg_half_w, _wg_half_w))

                # Write real segments back to their slots in segments list
                _real_segs = [s for s in _wg_final_order if not s.get("_virtual")]
                for _wg_k, _wg_idx in enumerate(_wg_pos):
                    if _wg_k < len(_real_segs):
                        segments[_wg_idx] = _real_segs[_wg_k]

                # Append virtual segments to be processed after real ones
                _virtual_segs = [s for s in _wg_final_order if s.get("_virtual")]
                for _vs in _virtual_segs:
                    segments.append(_vs)

                _ids = [
                    f"seg{s.get('seg_index','v')}({s.get('_cx_rank_hint',0):+.2f})"
                    + ("*" if s.get("_virtual") else "")
                    for s in _wg_final_order
                ]
                print(f"  [within-group sort] fi={_wg_fi} sign={_wg_sign:+.0f} L→R: {', '.join(_ids)}")
        except Exception as _wg_e:
            print(f"  [within-group sort] failed: {_wg_e}")

    tracker   = _SurfaceTracker()
    placements: list[dict] = []

    def _resolve_furniture(box_px: list[int] | None, phrase: str,
                           furn_idx_orig: int,
                           mask_file: "str | None" = None) -> tuple[int, dict | None, str]:
        """Overlay the decoration's mask on the reference photo → VLM picks furniture.

        Image 1: reference photo with decoration mask (or bbox) overlaid in red.
        Image 2: labeled 3D furniture render.
        The VLM sees the actual scene context in Image 1, making furniture
        identification reliable regardless of camera parameter mismatches.

        After VLM selects, we run a geometric sanity check.
        """
        # Build reference photo with mask (or scaled bbox) overlaid
        ref_with_mask: "Image.Image | None" = None
        if _ref_photo is not None:
            try:
                _mask_path = None
                if mask_file:
                    _mp = out_dir / "decorations" / "segmented" / mask_file
                    if _mp.exists():
                        _mask_path = _mp
                ref_with_mask = _overlay_mask_on_ref(
                    _ref_photo, _mask_path,
                    box_px=box_px,
                    orig_ref_w=_orig_ref_w,
                )
            except Exception as _rm_e:
                print(f"    [mask_overlay] failed: {_rm_e}")

        highlighted = None
        if _labeled_render is not None:
            try:
                # Highlight the object's location on the labeled render too (secondary)
                if box_px is not None:
                    highlighted = _highlight_bbox(_labeled_render, box_px, scale=_bbox_scale)
                else:
                    highlighted = _labeled_render
            except Exception as e:
                print(f"    [highlight] failed: {e}")
                highlighted = _labeled_render

        _phrase_key = _norm_phrase(phrase)
        _already_used = _phrase_occupied.get(_phrase_key, set())
        if not use_vlm:
            vlm_idx = None
        elif ref_with_mask is not None and highlighted is not None:
            # Primary: mask on reference photo → VLM identifies furniture from real scene
            vlm_idx = _vlm_select_furniture(
                highlighted, phrase, furn_list,
                ref_photo=ref_with_mask,
                box_px=None,  # already baked into ref_with_mask
                bbox_scale=1.0,
                camera=camera,
                already_used=_already_used,
            )
        elif highlighted is not None:
            vlm_idx = _vlm_select_furniture(
                highlighted, phrase, furn_list,
                camera=camera, already_used=_already_used,
            )
        else:
            vlm_idx = None

        furn_idx = vlm_idx if vlm_idx is not None else furn_idx_orig
        if vlm_idx is None:
            print(f"    [select_furn] using segmentation assignment: {furn_idx}")

        # Whether this item should be on seating (pillows/cushions) or a surface table.
        # Computed here so trust-original and geo_check blocks both share it.
        _pl_lower_trust = phrase.lower()
        prefer_seating = any(kw in _pl_lower_trust for kw in _AGAINST_BACK_PHRASES)

        # Trust-original rules: the original segmentation assignment is usually more
        # spatially accurate than VLM for distinguishing same-type furniture instances.
        _trust_orig_final = False  # may be set True below; gates geo_check
        if furn_idx_orig >= 0 and furn_idx_orig in furn_map:
            _orig_f_trust    = furn_map[furn_idx_orig]
            _orig_ftype_trust = _orig_f_trust.get("type", "").lower()
            _orig_is_valid_surface = (
                _orig_ftype_trust not in _SEATING_TYPES
                and _orig_ftype_trust not in {"plant", "floor_lamp", "floor lamp", "lamp",
                                               "tree", "sculpture", "statue"}
                and "position_m" in _orig_f_trust
                and "size_m"     in _orig_f_trust
                and "eff_h"      in _orig_f_trust
            )
            if _orig_is_valid_surface:
                _vlm_ftype_trust = furn_map.get(furn_idx, {}).get("type", "").lower() if furn_idx in furn_map else ""
                _vlm_invalid = (
                    furn_idx == -1  # VLM said floor
                    or _vlm_ftype_trust in {"plant", "floor_lamp", "floor lamp", "lamp",
                                            "tree", "sculpture", "statue"}  # non-surface type
                    or (not prefer_seating and _vlm_ftype_trust in _SEATING_TYPES)  # seating for on_surface item
                )
                # Only trust original over VLM for same-type seating (sofas/chairs are
                # visually similar and VLM often confuses them).  For tables/surfaces,
                # let geo-check + verify resolve conflicts — VLM distinguishes office
                # tables from coffee tables better than spatial segmentation does.
                _vlm_same_type = (
                    furn_idx != furn_idx_orig
                    and furn_idx in furn_map
                    and _vlm_ftype_trust == _orig_ftype_trust
                    and _orig_ftype_trust in _SEATING_TYPES  # only for seating confusion
                )
                _trust_orig_final = False
                if _vlm_invalid:
                    print(f"    [select_furn] VLM chose invalid/floor [{furn_idx}] ({_vlm_ftype_trust or 'floor'}); "
                          f"original [{furn_idx_orig}] ({_orig_ftype_trust}) → trusting original (skip geo_check)")
                    furn_idx = furn_idx_orig
                    _trust_orig_final = True
                elif _vlm_same_type:
                    print(f"    [select_furn] VLM picked same-type [{furn_idx}] ({_vlm_ftype_trust}); "
                          f"original [{furn_idx_orig}] ({_orig_ftype_trust}) → trusting original (skip geo_check)")
                    furn_idx = furn_idx_orig
                    _trust_orig_final = True

        # Geometric nearest-furniture check: find the closest valid furniture
        # to the mask centre across ALL furniture (not just VLM vs original).
        # This catches cases where both VLM and segmentation picked the wrong piece.
        # Skipped when trust-original already locked in the furniture choice.
        if not _trust_orig_final and box_px is not None and camera is not None and _labeled_render is not None:
            try:
                from object_placement.wall_mounted.wall_mounted_object_placement import (
                    _camera_axes, _project_vertex,
                )
                cam_pos  = np.array(camera["position_m"], dtype=np.float64)
                look_at  = np.array(camera["look_at_m"],  dtype=np.float64)
                up_world = np.array(camera.get("up", [0,1,0]), dtype=np.float64)
                right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
                W = _labeled_render.width
                hfov = float(camera["hfov_deg"])
                fx = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
                cx2, cy2 = W / 2.0, _labeled_render.height / 2.0

                x1, y1, x2, y2 = [v * _bbox_scale for v in box_px]
                mx = (x1 + x2) / 2.0
                my = (y1 + y2) / 2.0

                def _proj_dist(f: dict) -> float:
                    if "position_m" not in f or "size_m" not in f or "eff_h" not in f:
                        return float("inf")
                    p3 = np.array(f["position_m"], dtype=np.float64)
                    p3[1] += f.get("eff_h", 0.5)
                    px, py, zc = _project_vertex(p3, cam_pos, right_v, up_c_v, fwd_v, fx, cx2, cy2)
                    if zc <= 0.01:
                        return float("inf")
                    return float((px - mx)**2 + (py - my)**2)

                # For against_back items (pillows/cushions), only rank seating furniture
                pl_lower = phrase.lower()
                prefer_seating = any(kw in pl_lower for kw in _AGAINST_BACK_PHRASES)

                # For against_back items, the original segmentation (GDino+SAM) is
                # very reliable at identifying WHICH sofa/chair a pillow is on —
                # it ran directly on the reference photo.  VLM often confuses sofas
                # on different walls (left vs back), so if the original assigned this
                # pillow to seating, trust that assignment.
                # Only let VLM override when the original was NOT seating (e.g. the
                # segment_results.json mistakenly put a pillow on a coffee table).
                _orig_type_gc = (
                    furn_map.get(furn_idx_orig, {}).get("type", "").lower()
                    if furn_idx_orig >= 0 else ""
                )
                orig_is_seating = prefer_seating and (_orig_type_gc in _SEATING_TYPES)

                vlm_chose_seating = (
                    prefer_seating
                    and vlm_idx is not None
                    and furn_map.get(vlm_idx, {}).get("type", "").lower() in _SEATING_TYPES
                )
                if orig_is_seating:
                    # Original segmentation put this pillow on a specific seating piece.
                    # Usually the segmentation is reliable (it saw the real photo), but
                    # it sometimes groups multiple items onto the SAME seating piece when
                    # they should be distributed (e.g. two pillows — one for each sofa of
                    # a pair — both tagged as sofa [1]).  Resolve via geometric distance:
                    # if the VLM picks a DIFFERENT seating piece and that piece is closer
                    # (in projected-pixel d²) to the pillow's bbox centre than the
                    # original's is, trust the VLM — this lets the VLM redistribute
                    # same-type items the seg collapsed onto one piece.
                    _use_vlm_over_orig = False
                    if (vlm_idx is not None and vlm_idx != furn_idx_orig
                            and vlm_idx in furn_map
                            and furn_idx_orig in furn_map
                            and box_px is not None and camera is not None
                            and _labeled_render is not None):
                        try:
                            from object_placement.wall_mounted.wall_mounted_object_placement import (
                                _camera_axes as _ca, _project_vertex as _pv,
                            )
                            _cp  = np.array(camera["position_m"], dtype=np.float64)
                            _la  = np.array(camera["look_at_m"],  dtype=np.float64)
                            _up  = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
                            _rv, _uv, _fv = _ca(_cp, _la, _up)
                            _W = _labeled_render.width
                            _H = _labeled_render.height
                            _fx = _W / (2.0 * np.tan(np.radians(float(camera["hfov_deg"]) / 2.0)))
                            _cx2, _cy2 = _W / 2.0, _H / 2.0
                            _x1, _y1, _x2, _y2 = [v * _bbox_scale for v in box_px]
                            _mx, _my = (_x1 + _x2) / 2.0, (_y1 + _y2) / 2.0
                            def _d2_to_seat(fi: int) -> float:
                                f2 = furn_map.get(fi)
                                if f2 is None:
                                    return float("inf")
                                p3 = np.array(f2["position_m"], dtype=np.float64)
                                p3[1] += f2.get("eff_h", 0.5) * _SEAT_FRAC
                                px, py, zc = _pv(p3, _cp, _rv, _uv, _fv, _fx, _cx2, _cy2)
                                if zc <= 0.01:
                                    return float("inf")
                                return float((px - _mx) ** 2 + (py - _my) ** 2)
                            _d_orig = _d2_to_seat(furn_idx_orig)
                            _d_vlm  = _d2_to_seat(vlm_idx)
                            # Accept VLM only when it's MEANINGFULLY closer (≤0.77× d²,
                            # ~12% closer in pixel distance) — avoids flipping on tiny
                            # margins that reflect camera-model jitter rather than real
                            # visual cues.
                            if _d_vlm < _d_orig * 0.77:
                                _use_vlm_over_orig = True
                                print(f"    [geo_check] pillow: VLM [{vlm_idx}] "
                                      f"d²={_d_vlm:.0f} notably closer than orig [{furn_idx_orig}] "
                                      f"d²={_d_orig:.0f} → trusting VLM (probable redistribution)")
                            else:
                                print(f"    [geo_check] pillow: orig [{furn_idx_orig}] "
                                      f"d²={_d_orig:.0f} vs VLM [{vlm_idx}] d²={_d_vlm:.0f} "
                                      f"— keeping orig (VLM not meaningfully closer)")
                        except Exception as _pge:
                            print(f"    [geo_check] pillow geo-distance check failed: {_pge}")

                    if _use_vlm_over_orig:
                        furn_idx = vlm_idx  # type: ignore[assignment]
                    elif vlm_idx is not None and vlm_idx != furn_idx_orig:
                        print(f"    [geo_check] pillow: original seating [{furn_idx_orig}] "
                              f"overrides VLM [{vlm_idx}] (trust original seating assignment)")
                        furn_idx = furn_idx_orig
                    else:
                        print(f"    [geo_check] pillow: original seating [{furn_idx_orig}] "
                              f"— VLM agrees or no VLM result")
                elif vlm_chose_seating:
                    # Original was not seating but VLM redirected to seating → trust VLM.
                    print(f"    [geo_check] pillow: VLM redirected to seating [{vlm_idx}] "
                          f"(orig was non-seating) — skipping geo override")
                else:
                    # Furniture types that cannot host on_surface decorations
                    _NON_SURFACE_TYPES = {
                        "plant", "floor_lamp", "floor lamp", "lamp", "tree",
                        "sculpture", "statue",
                    }
                    # Desk-only items that should never land on seating furniture
                    # (geo_check must not override VLM's desk choice to nearest chair)
                    _DESK_ONLY_PHRASES = {
                        "lamp", "desk lamp", "monitor", "computer monitor",
                        "keyboard", "computer keyboard", "laptop", "screen",
                        "printer", "speaker", "headphone",
                    }
                    _pl_is_desk_item = any(kw in pl_lower for kw in _DESK_ONLY_PHRASES)

                    # Bed is not technically "seating" but a desk lamp / monitor /
                    # similar tabletop item should never be redirected onto a bed —
                    # the geometric nearest from the bbox of a nightstand-lamp will
                    # often be the bed itself (it's the largest floor footprint).
                    _BED_TYPES = {"bed", "bunk_bed", "single_bed", "double_bed"}

                    def _is_valid_candidate(f: dict) -> bool:
                        if f.get("index", -1) < 0:
                            return False
                        ftype = f.get("type", "").lower()
                        if prefer_seating and ftype not in _SEATING_TYPES:
                            return False
                        # Desk-only items must not be redirected onto seating furniture
                        # nor onto a bed (bed is not seating but isn't a desk surface either).
                        if _pl_is_desk_item and (ftype in _SEATING_TYPES or ftype in _BED_TYPES):
                            return False
                        # on_surface non-pillow items (books, bowl, plant…) must not
                        # land on seating furniture — they belong on tables/desks.
                        if not prefer_seating and ftype in _SEATING_TYPES:
                            return False
                        # Never place on_surface decorations on non-surface objects
                        if not prefer_seating and ftype in _NON_SURFACE_TYPES:
                            return False
                        return True

                    # Rank all valid furniture by distance to mask centre.
                    # NOTE: We do NOT auto-override the silhouette-nearest just
                    # because it already holds a same-phrase decoration.  The
                    # VLM was already told (in `_vlm_select_furniture`) which
                    # surfaces are occupied and is allowed to stack on a single
                    # surface when the reference photo supports that — it's OK
                    # to have two same-phrase items on one surface as long as
                    # the VLM thinks that's correct.  Geometric distance still
                    # ranks candidates; occupancy is just one input the VLM
                    # weighs against the reference layout.
                    ranked = sorted(
                        [(f["index"], _proj_dist(f)) for f in furn_list
                         if _is_valid_candidate(f)],
                        key=lambda x: x[1],
                    )
                    if not ranked:
                        # Relax type constraint but still exclude non-surface types
                        # and seating for non-pillow on_surface items.
                        ranked = sorted(
                            [(f["index"], _proj_dist(f)) for f in furn_list
                             if f.get("index", -1) >= 0
                             and f.get("type", "").lower() not in _NON_SURFACE_TYPES
                             and (prefer_seating or f.get("type", "").lower() not in _SEATING_TYPES)],
                            key=lambda x: x[1],
                        )
                    if not ranked:
                        ranked = sorted(
                            [(f["index"], _proj_dist(f)) for f in furn_list
                             if f.get("index", -1) >= 0],
                            key=lambda x: x[1],
                        )
                    nearest_idx, nearest_d2 = ranked[0] if ranked else (furn_idx, float("inf"))
                    # If the current choice is a physically-invalid furniture type for
                    # this placement (non-surface object or seating for on_surface items),
                    # treat cur_d2 as ∞ so we always prefer the nearest valid candidate.
                    _cur_ftype_gc = furn_map.get(furn_idx, {}).get("type", "").lower() if furn_idx in furn_map else ""
                    _cur_invalid_gc = (
                        furn_idx < 0  # floor — always override with nearest valid furniture
                        or _cur_ftype_gc in _NON_SURFACE_TYPES
                        or (not prefer_seating and _cur_ftype_gc in _SEATING_TYPES)
                    )
                    if _cur_invalid_gc:
                        cur_d2 = float("inf")
                        print(f"    [geo_check] current [{furn_idx}] type '{_cur_ftype_gc}' invalid for this item — forcing override")
                    else:
                        cur_d2 = _proj_dist(furn_map[furn_idx]) if furn_idx in furn_map else float("inf")

                    # Threshold for accepting the geometric override:
                    #   - Different-type override: 4× in d² (2× pixel distance) — VLM is
                    #     usually right about TYPE, so require strong geometric evidence.
                    #   - Same-type override: ~1.3× in d² (any meaningful geometric win) —
                    #     VLM commonly confuses two instances of the same type (e.g.
                    #     "central coffee table" vs a side coffee table), and the
                    #     projected mask centre is a far better signal than VLM language.
                    _nearest_ftype_gc = (
                        furn_map.get(nearest_idx, {}).get("type", "").lower()
                        if nearest_idx in furn_map else ""
                    )
                    _same_type_override = (
                        _nearest_ftype_gc == _cur_ftype_gc
                        and not _cur_invalid_gc
                        and _cur_ftype_gc != ""
                    )
                    _gc_divisor = 1.3 if _same_type_override else 4.0
                    if nearest_d2 < cur_d2 / _gc_divisor and nearest_d2 < float("inf"):
                        print(f"    [geo_check] nearest=[{nearest_idx}] d²={nearest_d2:.0f}  "
                              f"cur=[{furn_idx}] d²={cur_d2:.0f}  → would use nearest")
                        # When the current choice was invalid (forced cur_d2=inf), skip
                        # geo_vlm and trust pure geometry — VLM repeatedly picks wrong
                        # furniture for items like succulents on small corner tables.
                        if _cur_invalid_gc:
                            print(f"    [geo_check] current was invalid type — using geometric nearest [{nearest_idx}] directly")
                            furn_idx = nearest_idx
                        # Same-type override: VLM already picked the wrong instance of the
                        # correct type (e.g. confused two coffee tables).  Skip the VLM
                        # disambiguation re-ask — it's prone to repeating the same mistake —
                        # and trust the geometric nearest directly.
                        elif _same_type_override:
                            print(f"    [geo_check] same-type override → using geometric nearest [{nearest_idx}] directly")
                            furn_idx = nearest_idx
                        # VLM disambiguation: when geo_check would override, ask VLM to
                        # compare reference vs labeled render to resolve depth ambiguity.
                        _geo_vlm_confirmed = False
                        try:
                            if (not _cur_invalid_gc and not _same_type_override
                                    and _labeled_render is not None and _ref_photo is not None and box_px is not None):
                                _geo_result = _vlm_select_furniture(
                                    _labeled_render, phrase, furn_list,
                                    ref_photo=_ref_photo, box_px=box_px, bbox_scale=_bbox_scale,
                                    camera=camera,
                                )
                                if _geo_result is not None and _geo_result != -999:
                                    _geo_ftype = furn_map.get(_geo_result, {}).get("type", "").lower() if _geo_result >= 0 else ""
                                    _geo_invalid = (
                                        _geo_result == -1
                                        or _geo_ftype in _NON_SURFACE_TYPES
                                        or (not prefer_seating and _geo_ftype in _SEATING_TYPES)
                                    )
                                    if _geo_invalid:
                                        print(f"    [geo_vlm] VLM returned invalid [{_geo_result}] "
                                              f"({_geo_ftype or 'floor'}); using geometric nearest [{nearest_idx}]")
                                        furn_idx = nearest_idx
                                    else:
                                        print(f"    [geo_vlm] VLM chose [{_geo_result}] to resolve "
                                              f"ambiguity (geo nearest=[{nearest_idx}] cur=[{furn_idx}])")
                                        furn_idx = _geo_result
                                    _geo_vlm_confirmed = True
                        except Exception as _gve:
                            print(f"    [geo_vlm] VLM disambiguation failed: {_gve}")
                        if not _geo_vlm_confirmed:
                            furn_idx = nearest_idx
                    else:
                        print(f"    [geo_check] cur=[{furn_idx}] d²={cur_d2:.0f}  "
                              f"nearest=[{nearest_idx}] d²={nearest_d2:.0f}  → keeping cur")
            except Exception as e:
                print(f"    [geo_check] failed: {e}")

        furn = furn_map.get(furn_idx)
        furn_type = furn.get("type", "furniture") if furn is not None else "floor"
        # Return the ORIGINAL VLM pick (before geo_check) too, so the
        # verify pass can tell whether geo_check moved the index away
        # from the VLM's choice.  When verify wants to revert that move
        # (verify == seg == orig VLM, all disagreeing with geo_check),
        # we override.  When the original VLM agreed with geo_check,
        # verify alone shouldn't undo a same-type call.
        return furn_idx, furn, furn_type, vlm_idx

    # Render the base scene (no decorations) as the first animation frame
    try:
        _do_render(out_dir, [], placements_dir / "render_base.png", inpaint_dir=inpaint_dir)
        print("[place_decorations] Base scene render saved → render_base.png")
    except Exception as _be:
        print(f"[place_decorations] Base render failed: {_be}")

    # Track against_back groups for post-group VLM reorder
    _ab_groups: "dict[tuple, list[dict]]" = {}   # (furn_idx,) → [entry, ...]
    # Same-color group scale: first placed pillow of each color on a furniture sets the scale
    _ab_group_scales: "dict[tuple, float]" = {}  # (furn_idx, color) → scale
    # Track which surfaces already received a same-phrase decoration, so when
    # the reference shows multiple instances spread across same-type furniture
    # (e.g. one lamp on each of two cabinets) the VLM can prefer empty surfaces
    # for subsequent instances instead of piling them all on the most prominent
    # piece.  Key = normalized phrase, value = set of furn_idxs.
    _phrase_occupied: "dict[str, set[int]]" = {}
    def _norm_phrase(p: str) -> str:
        return " ".join(p.lower().split())

    for _seg_loop_idx, seg in enumerate(segments):
        seg_idx   = seg["seg_index"]
        phrase    = seg.get("phrase", "decoration")
        glb_rel   = seg.get("glb_file")

        # Substitute path: when this segment's GLB is missing, try to reuse
        # an already-generated decoration GLB of the same noun phrase whose
        # mean colour matches.  This is the "if a pillow's GLB failed to
        # generate, use a similar-looking pillow instead" fallback — better
        # than dropping the placement entirely.
        _substitute_used = None
        if not glb_rel or not (out_dir / glb_rel).exists():
            sub = _pick_substitute_glb(seg, segments, inpaint_dir, out_dir)
            if sub is None:
                _why = "no glb_file" if not glb_rel else "GLB not found"
                print(f"\n[place_decorations] {seg_idx:02d} '{phrase}' — "
                      f"{_why} and no similar-colour substitute available, skipping")
                continue
            print(f"\n[place_decorations] {seg_idx:02d} '{phrase}' — GLB missing, "
                  f"substituting GLB from seg {sub['donor_seg_index']:02d} "
                  f"'{sub['donor_phrase']}' (Δcolor={sub['delta_color']:.1f})")
            glb_rel = sub["glb_file"]
            _substitute_used = sub
            # Persist on the segment dict so downstream stages and the saved
            # placements JSON record the donor info.
            seg["glb_file"] = glb_rel
            seg["_glb_substituted_from"] = {
                "donor_seg_index": sub["donor_seg_index"],
                "donor_phrase":    sub["donor_phrase"],
                "delta_color":     sub["delta_color"],
            }
            # Also borrow the donor's inpaint/mask/canvas files so downstream
            # checks (which require an inpaint image) don't reject the
            # substituted segment.  Only fill fields the missing seg lacks —
            # don't overwrite a real segmentation that just happens to lack
            # a GLB.
            try:
                _donor = next(s for s in segments
                              if s.get("seg_index") == sub["donor_seg_index"])
                for _k in ("inpaint_file", "mask_file", "canvas_file"):
                    if not seg.get(_k) and _donor.get(_k):
                        seg[_k] = _donor[_k]
                        seg.setdefault("_borrowed_files", []).append(_k)
            except StopIteration:
                pass

        glb_path = out_dir / glb_rel
        if not glb_path.exists():
            print(f"\n[place_decorations] {seg_idx:02d} '{phrase}' — GLB not found, skipping")
            continue

        # Load decoration image first (needed for VLM furniture selection)
        inpaint_file = seg.get("inpaint_file")
        decor_img = None
        if inpaint_file and (inpaint_dir / inpaint_file).exists():
            decor_img = Image.open(inpaint_dir / inpaint_file).convert("RGB")
        if decor_img is None:
            print(f"\n[place_decorations] {seg_idx:02d} '{phrase}' — no inpaint image, skipping")
            continue

        # VLM: select which furniture this decoration belongs on.
        # Project the segmentation mask onto the reference photo so the VLM sees
        # the actual scene context — much more reliable than bbox-on-render.
        furn_idx_orig = seg.get("furniture_index", -1)
        box_px        = seg.get("box_px")       # [x1,y1,x2,y2] in original photo space
        mask_file_seg = seg.get("mask_file")    # e.g. "decor_02_pillows_mask.png"
        if seg.get("_duplicate_resolved"):
            # Pre-assigned by the duplicate-resolve pass (one decoration per
            # furniture instance for duplicated furniture).  Skip the VLM
            # reassignment — it would funnel everything back to the single
            # furniture instance shown in the reference photo.
            furn_idx = furn_idx_orig
            furn = furn_map.get(furn_idx)
            furn_type = furn.get("type", "furniture") if furn is not None else "floor"
            _orig_vlm_furn_idx = furn_idx
            print(f"    [duplicate-resolve] honoring pre-assigned "
                  f"furniture_index={furn_idx} (skip VLM select)")
        else:
            furn_idx, furn, furn_type, _orig_vlm_furn_idx = _resolve_furniture(
                box_px, phrase, furn_idx_orig, mask_file=mask_file_seg
            )

        print(f"\n[place_decorations] {seg_idx:02d} '{phrase}' → furniture [{furn_idx}]"
              f"{' (was '+str(furn_idx_orig)+')' if furn_idx != furn_idx_orig else ''}")

        if furn is None and furn_idx == -1:
            pass  # on_floor — handled in placement logic
        elif furn is None:
            print(f"  furniture index {furn_idx} not found — skipping")
            continue
        elif furn is not None and ("position_m" not in furn or "size_m" not in furn or "eff_h" not in furn):
            # Only fall back to segmentation assignment if:
            # - geo_check did NOT already keep the VLM choice over the original
            #   (i.e. furn_idx == furn_idx_orig means geo_check reverted, or VLM agreed)
            # - original furniture has valid placement data
            orig_furn_check = furn_map.get(furn_idx_orig)
            orig_ok = (orig_furn_check is not None and
                       "position_m" in orig_furn_check and
                       "size_m" in orig_furn_check and
                       "eff_h" in orig_furn_check)
            if orig_ok and furn_idx == furn_idx_orig:
                # geo_check already reverted to original (or VLM agreed), just skip — orig is also incomplete
                print(f"  furniture [{furn_idx}] ({furn_type}) has incomplete placement data — skipping")
                continue
            elif orig_ok:
                # geo_check kept VLM but VLM target is incomplete — try original
                print(f"  furniture [{furn_idx}] ({furn_type}) has incomplete placement data — "
                      f"falling back to segmentation assignment [{furn_idx_orig}]")
                furn_idx = furn_idx_orig
                furn = orig_furn_check
                furn_type = furn.get("type", "furniture")
            else:
                # Both VLM choice and original are incomplete → skip
                print(f"  furniture [{furn_idx}] ({furn_type}) has incomplete data and "
                      f"original [{furn_idx_orig}] also unusable — skipping")
                continue

        # Load furniture inpainted image for scale context
        furn_img = None
        if furn is not None:
            furn_glb = furn.get("glb_path", "")
            for candidate in [
                out_dir / "furniture" / "inpainted" / (Path(furn_glb).stem + ".png"),
                out_dir / furn_glb.replace(".glb", ".png") if furn_glb else None,
            ]:
                if candidate and Path(candidate).exists():
                    furn_img = Image.open(candidate).convert("RGB")
                    break

        # 4-view front detection: determine local front direction of the GLB
        crop_path = None
        crop_file = seg.get("crop_file")
        if crop_file and (out_dir / "decorations" / "segmented" / crop_file).exists():
            crop_path = out_dir / "decorations" / "segmented" / crop_file
        front_local = _detect_decoration_front(glb_path, crop_path, phrase) if use_vlm else None
        if front_local is not None:
            print(f"    [front_detect] local front={np.round(front_local, 2).tolist()}")

        # VLM: estimate size and placement
        if use_vlm:
            vlm = _vlm_placement(decor_img, furn_img, phrase, furn_type)
        else:
            vlm = None
        if vlm is None:
            vlm = _size_fallback(phrase)
            print(f"    [vlm] using keyword fallback: {vlm}")

        placement_type = vlm.get("placement_type", "on_surface")
        side           = vlm.get("surface_position", "center")
        depth_pos      = vlm.get("depth_position", "middle")
        facing         = vlm.get("facing")   # may be None if VLM didn't return it

        # Objects with a clear user-facing side → always toward_room
        _TOWARD_ROOM_TYPES = {
            "monitor", "computer monitor", "laptop", "keyboard",
            "computer keyboard", "tv", "television", "screen", "picture frame",
            "desk lamp",  # desk lamp head is directional — honour VLM facing
        }
        # Objects that are symmetric / omni-directional → always any
        _SYMMETRIC_FACING = {
            "floor lamp", "vase", "plant", "mug", "cup",
            "tray", "book", "books", "remote", "pillow", "cushion",
        }
        pl = phrase.lower()
        if any(kw in pl for kw in _TOWARD_ROOM_TYPES):
            facing = "toward_room"
            if facing != vlm.get("facing"):
                print(f"    [facing] type-default → toward_room for '{phrase}'")
        elif any(kw in pl for kw in _SYMMETRIC_FACING):
            facing = "any"
        elif facing is None:
            facing = vlm.get("facing", "any") or "any"

        # Override placement type for seating + pillow combination
        if (placement_type == "on_surface"
                and furn_type.lower() in _SEATING_TYPES
                and any(w in phrase.lower() for w in ("pillow", "cushion"))):
            placement_type = "against_back"
            print(f"    overriding to against_back (pillow on seating)")

        # Bed pillows: keep flat on surface (no tilting like a sofa cushion)
        # but force depth=back so they sit against the headboard end of the
        # bed instead of floating in the middle.  Beds are NOT in
        # _SEATING_TYPES so the against_back override above doesn't fire,
        # which is correct for bed pillows (they lie flat); but the default
        # depth=middle leaves them adrift on the mattress.
        _BED_TYPES_DEC = {"bed", "single_bed", "double_bed", "bunk_bed"}
        if (placement_type == "on_surface"
                and furn_type.lower() in _BED_TYPES_DEC
                and any(w in phrase.lower() for w in ("pillow", "cushion"))):
            depth_pos = "back"
            print(f"    overriding depth_position → 'back' (pillow on bed → against headboard)")

        # Clamp VLM size to plausible range, then compute scale
        raw_size = _clamp_vlm_size(phrase, vlm.get("size_m", {}))
        scale, size_m = _compute_scale(glb_path, raw_size)

        # Same-color group-scale normalization: all SQUARE pillows of the same color
        # on the same furniture share the scale of the first-placed one.
        # Rectangular pillows are excluded — their aspect ratio is intentional.
        if placement_type == "against_back" and _classify_seg_shape(seg) == "square":
            _pcolor = seg.get("_pillow_color", "")
            if _pcolor and _pcolor != "unknown":
                _gk = (furn_idx, _pcolor)
                if _gk in _ab_group_scales:
                    try:
                        _gscale = _ab_group_scales[_gk]
                        _nat = _glb_native_extents(glb_path)
                        scale = _gscale
                        size_m = {
                            "width_m":  float(_nat[0] * scale),
                            "height_m": float(_nat[1] * scale),
                            "depth_m":  float(_nat[2] * scale),
                        }
                        print(f"    [group_scale] color={_pcolor} → scale={scale:.4f} "
                              f"({size_m['width_m']:.3f}×{size_m['height_m']:.3f}×{size_m['depth_m']:.3f} m)")
                    except Exception as _gse:
                        print(f"    [group_scale] failed: {_gse}")
                else:
                    _ab_group_scales[_gk] = scale

        # Compute world position
        world_pos: np.ndarray | None = None

        if placement_type == "against_back" and furn_type.lower() in _SEATING_TYPES:
            if furn is None:
                print(f"  against_back requested but no furniture — skipping")
                continue

            # The tracker handles lateral collision — multiple pillows can go on
            # the same sofa, packed side-by-side.  No seating_used redirect needed.

            # Compute preferred lateral position from bbox projection onto sofa's local right axis.
            _ref_img_w = int(_orig_ref_w)
            _ref_img_h = int(_orig_ref_h)
            _pref_cx = _bbox_preferred_cx(furn, box_px, camera, img_w=_ref_img_w, img_h=_ref_img_h,
                                          rank_hint=seg.get("_cx_rank_hint"),
                                          furn_photo_box_px=furn_box_px_by_idx.get(furn.get("index")))

            # Always returns a position (never None) — packs pillows side-by-side
            world_pos = _place_against_back(furn, size_m, side, tracker,
                                            out_dir=out_dir, preferred_cx_hint=_pref_cx)
            # Manual override: --pillow-cx-local / --pillow-cz-local lets
            # the user dial in an exact (cx, cz) in the sofa's local frame
            # for debugging.  Keeps the y from the original placement
            # (raycast-derived seat height) so we don't move the pillow
            # off the seat in Y.  The entry is tagged with
            # `_pos_locked` so post-loop passes (bbox_contain,
            # final_clamp, pos_refine) skip it — otherwise the strict
            # footprint clamps would pull a deep manual override back
            # into the metadata bbox.
            _pillow_pos_locked = False
            if (pillow_cx_override is not None
                    or pillow_cz_override is not None):
                _R_ovr = _rot_matrix(furn)
                _f_pos_ovr = np.array(furn["position_m"],
                                       dtype=np.float64)
                _orig_off = world_pos - _f_pos_ovr
                _cx_orig = float(_orig_off @ _R_ovr[:, 0])
                _cz_orig = float(_orig_off @ _R_ovr[:, 2])
                _cx_use = (float(pillow_cx_override)
                           if pillow_cx_override is not None
                           else _cx_orig)
                _cz_use = (float(pillow_cz_override)
                           if pillow_cz_override is not None
                           else _cz_orig)
                _new_pos_ovr = _f_pos_ovr.copy()
                _new_pos_ovr += _cx_use * _R_ovr[:, 0]
                _new_pos_ovr += _cz_use * _R_ovr[:, 2]
                _new_pos_ovr[1] = float(world_pos[1])
                print(f"    [pillow_override] forced sofa-local "
                      f"cx={_cx_use:+.3f} cz={_cz_use:+.3f} "
                      f"(was cx={_cx_orig:+.3f} cz={_cz_orig:+.3f}) "
                      f"→ world {_new_pos_ovr.round(3).tolist()}")
                world_pos = _new_pos_ovr
                _pillow_pos_locked = True
            # Optional Y offset (raise/lower the pillow) — used to
            # resolve the seat-vs-back-rest gap on broken meshes
            # where pillow back face is at the back-rest cushion
            # but the seat surface ends earlier.  A small +Y bump
            # lifts the pillow off the missing seat into the back-
            # rest cushion's visible volume.
            if (placement_type == "against_back"
                    and pillow_y_offset is not None):
                _y_old = float(world_pos[1])
                world_pos[1] = _y_old + float(pillow_y_offset)
                print(f"    [pillow_y_offset] raised y by "
                      f"{pillow_y_offset:+.3f}m: {_y_old:.3f} → "
                      f"{world_pos[1]:.3f}")
                _pillow_pos_locked = True
            # Force-centre placements (visible-mesh-centred) must be
            # locked too: post-loop bbox_contain / final_clamp would
            # pull the pillow back into metadata's bbox, undoing the
            # mesh-centre snap that placed it on the visible sofa.
            if (placement_type == "against_back"
                    and furn is not None
                    and furn.get("_force_centre_used")):
                _pillow_pos_locked = True
                furn["_force_centre_used"] = False  # consume the flag
            rotation  = _against_back_rotation(furn)
        elif placement_type == "on_floor":
            if furn is not None:
                right_world, _, _ = _world_axes(furn)
                furn_pos  = np.array(furn["position_m"], dtype=np.float64)
                half_w    = furn["size_m"]["width_m"] / 2
                offset    = (half_w + size_m["width_m"] / 2 + 0.05)
                side_sign = 1.0 if side == "right" else -1.0
                world_pos = furn_pos + side_sign * offset * right_world
                print(f"    on_floor: next to furniture side={side}")
            else:
                # No reference furniture — place at floor origin with a small offset
                world_pos = np.array([0.5, 0.0, 0.5], dtype=np.float64)
                print(f"    on_floor: no reference furniture — placed at floor origin")
            rotation  = [[1,0,0],[0,1,0],[0,0,1]]
        else:
            placement_type = "on_surface"
            if furn is None:
                print(f"  on_surface requested but no furniture — skipping")
                continue
            # Derive an exact (cx, cz) hint by back-projecting the reference
            # bbox bottom-center onto the table's top-surface plane.  This
            # reliably distinguishes multiple objects on the same table where
            # the earlier px/py-only projection clamped everything to one side.
            _cx_hint = _cz_hint = None
            _local_xz = _bbox_to_table_local(
                furn, box_px, camera,
                img_w=int(_orig_ref_w),
                img_h=int(_orig_ref_h),
                out_dir=out_dir,
            )

            # Pixel-space lateral hint as a sanity check.  When the bbox bottom
            # is below the table top in image (mask captured pixels under the
            # surface), the mesh raycast can hit the wrong side and return a
            # cx with the WRONG SIGN.  The pure-pixel `_bbox_preferred_cx`
            # logic projects the table's local +X axis to image and divides
            # the bbox-center delta — robust to the bottom-pixel ambiguity.
            _cx_pixel = _bbox_preferred_cx(
                furn, box_px, camera,
                img_w=int(_orig_ref_w), img_h=int(_orig_ref_h),
                furn_photo_box_px=furn_box_px_by_idx.get(furn.get("index")),
            )

            if _local_xz is not None:
                _cx_hint, _cz_hint = _local_xz
                _half_w_surf = furn["size_m"]["width_m"] / 2.0
                _half_d_surf = furn["size_m"]["depth_m"] / 2.0
                _side_thresh_x = max(_half_w_surf * 0.25, 0.10)
                _side_thresh_z = max(_half_d_surf * 0.25, 0.10)
                # Only override the VLM's side/depth when the back-projection
                # lies within the furniture's footprint.  Out-of-bounds values
                # (common when eff_h or camera params are slightly off) are
                # UNRELIABLE — they would otherwise flip "side=left" from the
                # VLM to "side=right" via a spurious cx=+1.441 (as happened
                # with the desk lamp).  Keep the VLM's side/depth in that case.
                _bp_cx_in_bounds = abs(_cx_hint) <= _half_w_surf
                _bp_cz_in_bounds = abs(_cz_hint) <= _half_d_surf

                # Sign-disagreement with pixel hint → mesh raycast hit a wrong
                # surface.  Trust the pixel hint (which used the projected
                # local +X axis directly).  Treat near-zero |_cx_pixel| (<5cm)
                # as "centred" and skip the override.
                if (abs(_cx_pixel) > 0.05
                        and abs(_cx_hint) > 0.05
                        and (_cx_pixel * _cx_hint) < 0):
                    print(f"    [surface_backproj] cx sign disagreement: "
                          f"mesh_raycast={_cx_hint:+.3f} vs pixel={_cx_pixel:+.3f} "
                          f"→ trusting pixel-based")
                    _cx_hint = float(np.clip(_cx_pixel, -_half_w_surf, _half_w_surf))
                    _bp_cx_in_bounds = True

                if _bp_cx_in_bounds:
                    if _cx_hint > _side_thresh_x:
                        side = "right"
                    elif _cx_hint < -_side_thresh_x:
                        side = "left"
                else:
                    print(f"    [surface_backproj] cx={_cx_hint:+.3f}m out of bounds "
                          f"(±{_half_w_surf:.2f}m) — falling back to pixel hint "
                          f"cx={_cx_pixel:+.3f}")
                    # Pixel hint is bounded by half_w in _bbox_preferred_cx,
                    # so it's safe to use directly when mesh-raycast was bad.
                    if abs(_cx_pixel) > 0.05:
                        _cx_hint = float(np.clip(_cx_pixel, -_half_w_surf, _half_w_surf))
                        if _cx_hint > _side_thresh_x:
                            side = "right"
                        elif _cx_hint < -_side_thresh_x:
                            side = "left"
                    else:
                        _cx_hint = None
                if _bp_cz_in_bounds:
                    if _cz_hint > _side_thresh_z:
                        depth_pos = "front"
                    elif _cz_hint < -_side_thresh_z:
                        depth_pos = "back"
                else:
                    print(f"    [surface_backproj] cz={_cz_hint:+.3f}m out of bounds "
                          f"(±{_half_d_surf:.2f}m) — keeping VLM depth='{depth_pos}'")
                    _cz_hint = None
                _cx_str = f"{_cx_hint:+.3f}" if _cx_hint is not None else "None"
                _cz_str = f"{_cz_hint:+.3f}" if _cz_hint is not None else "None"
                print(f"    [surface_backproj] local=({_cx_str}, {_cz_str})m "
                      f"→ side={side}, depth={depth_pos}  (pixel cx hint={_cx_pixel:+.3f})")

            # toward_room rotation: use the desk's OWN front (R_furn @ +Z) so
            # monitors/keyboards/lamps end up parallel to the desk's front
            # edge.  The prior "aim at nearest seating" rule produced diagonal
            # rotations when the chair sat off-axis.  Chair-centre cx snap is
            # now ONLY applied when the bbox back-projection gave an
            # out-of-bounds value (i.e. unreliable due to eff_h error) — in
            # that case we fall back to placing the item directly in front of
            # the matched chair.  When back-proj is reliable, trust it so
            # side/back/front distinctions from the ref photo are preserved.
            toward_dir: np.ndarray | None = None
            _matched_seat: "dict | None" = None
            _backproj_out_of_bounds = (
                _local_xz is not None
                and (abs(_local_xz[0]) > furn["size_m"]["width_m"] / 2.0
                     or abs(_local_xz[1]) > furn["size_m"]["depth_m"] / 2.0)
            )
            if facing == "toward_room" and furn_type.lower() not in _SEATING_TYPES:
                R_furn = _rot_matrix(furn)
                fwd = R_furn @ np.array([0.0, 0.0, 1.0])
                fwd[1] = 0.0
                fwd_n = float(np.linalg.norm(fwd))
                if fwd_n > 0.01:
                    toward_dir = fwd / fwd_n
                    _furn_xz = np.array(furn["position_m"])[[0, 2]]
                    _td_xz = np.array([toward_dir[0], toward_dir[2]])
                    seating_in_front = [
                        f for f in furn_list
                        if f.get("type", "").lower() in _SEATING_TYPES
                        and "position_m" in f
                        and float(np.dot(
                            np.array(f["position_m"])[[0, 2]] - _furn_xz, _td_xz
                        )) > 0.1
                    ]
                    if seating_in_front:
                        _matched_seat = min(
                            seating_in_front,
                            key=lambda f: float(np.linalg.norm(
                                np.array(f["position_m"])[[0, 2]] - _furn_xz
                            )),
                        )
                        # Chair-centre policy:
                        #   * "User-interactive" screen items (monitor, keyboard,
                        #     laptop, tv) that sit DIRECTLY IN FRONT of the user
                        #     should always snap to the chair's cx.  These items
                        #     are ergonomically centred on the user, so the
                        #     reference photo's exact bbox cx is less meaningful
                        #     than chair-alignment.
                        #   * All other items (lamps, books, vases) should use
                        #     their bbox back-projection when it's in-bounds —
                        #     they sit AROUND the screen, not centred on the user.
                        #   * If back-proj is out-of-bounds OR absent AND VLM said
                        #     "center", still fall back to chair-centre.
                        # Bug fix: the VLM dict uses "surface_position", not
                        # "side" — the missing key always defaulted to "center",
                        # forcing chair_centre to fire even when the VLM said
                        # "left"/"right".  That stomped a perfectly good
                        # back-proj cx and dropped lamps onto the chair-axis
                        # column instead of the photo-bbox-implied location.
                        _vlm_side_is_center = (
                            str(vlm.get("surface_position", "center"))
                                .lower() == "center"
                        )
                        _SCREEN_ITEMS_CENTRE = (
                            "monitor", "computer monitor", "laptop",
                            "keyboard", "computer keyboard", "tv", "television", "screen",
                        )
                        _is_screen_centre = any(kw in phrase.lower() for kw in _SCREEN_ITEMS_CENTRE)
                        _use_chair_centre = (
                            _is_screen_centre
                            or (
                                (_backproj_out_of_bounds or _cx_hint is None)
                                and _vlm_side_is_center
                            )
                        )
                        if _use_chair_centre:
                            _right_w, _, _ = _world_axes(furn)
                            _seat_cx_world = float(np.dot(
                                (np.array(_matched_seat["position_m"]) - np.array(furn["position_m"]))[[0, 2]],
                                _right_w[[0, 2]],
                            ))
                            _fhw_clamp = max(
                                furn["size_m"]["width_m"] / 2.0 - size_m.get("width_m", 0.3) / 2.0 - _EDGE_MARGIN,
                                0.0,
                            )
                            _cx_hint = float(np.clip(_seat_cx_world, -_fhw_clamp, _fhw_clamp))
                            _reason = "screen item" if _is_screen_centre else "back-proj unreliable"
                            print(f"    [chair_centre] {_reason} → "
                                  f"using seat [{_matched_seat['index']}] cx={_cx_hint:+.3f}")
                            # For SCREEN items only: also snap cz to the BACK
                            # of the desk so the monitor sits at center-BACK
                            # (ergonomic position), not front-edge where the
                            # back-projection often points due to monitor
                            # height extending vertically above the surface.
                            if _is_screen_centre:
                                _fhd_clamp = max(
                                    furn["size_m"]["depth_m"] / 2.0 - size_m.get("depth_m", 0.25) / 2.0 - _EDGE_MARGIN,
                                    0.0,
                                )
                                # ~70% toward back: gives ergonomic distance
                                _cz_hint = -_fhd_clamp * 0.7
                                print(f"    [chair_centre] screen → cz={_cz_hint:+.3f} "
                                      f"(back of desk, ergonomic position)")
                        else:
                            _cx_str = (f"{_cx_hint:+.3f}"
                                       if _cx_hint is not None else "None")
                            print(f"    [toward] desk front with matched seat "
                                  f"[{_matched_seat['index']}]; keeping "
                                  f"back-proj cx={_cx_str}")
                    else:
                        print(f"    [toward] using furniture's own front (no seating in front)")
                else:
                    # Furniture has no yaw — last-resort aim at nearest seating.
                    seating_nearby = [
                        f for f in furn_list
                        if f.get("type", "").lower() in _SEATING_TYPES
                        and "position_m" in f
                    ]
                    if seating_nearby and world_pos is None:  # may need a position first
                        pass
                    if seating_nearby:
                        _furn_xz = np.array(furn["position_m"])[[0, 2]]
                        closest_seat = min(
                            seating_nearby,
                            key=lambda f: float(np.linalg.norm(
                                np.array(f["position_m"])[[0, 2]] - _furn_xz))
                        )
                        d = np.array(closest_seat["position_m"])[[0, 2]] - _furn_xz
                        d_n = float(np.linalg.norm(d))
                        if d_n > 0.05:
                            toward_dir = np.array([d[0] / d_n, 0.0, d[1] / d_n])
                            print(f"    [toward] no furniture yaw → aiming at nearest seating "
                                  f"[{closest_seat['index']}]")

            world_pos = _place_on_surface(furn, size_m, side, depth_pos, tracker,
                                          out_dir=out_dir,
                                          cx_hint=_cx_hint, cz_hint=_cz_hint)
            rotation  = _surface_rotation(furn, facing, front_local, toward_dir)
            # Legacy camera-flip: only as a last-resort fallback when we have
            # no toward_dir.  With toward_dir set, the rotation is correct by
            # construction — the camera is a render viewpoint, not the user's
            # position, so flipping based on it was often wrong.
            if (facing == "toward_room" and front_local is not None and camera is not None
                    and toward_dir is None):
                R_check = np.array(rotation, dtype=np.float64)
                screen_world = R_check @ np.array(front_local, dtype=np.float64)
                cam_p = np.array(camera["position_m"], dtype=np.float64)
                furn_p = np.array(furn["position_m"], dtype=np.float64)
                to_cam = cam_p - furn_p; to_cam[1] = 0.0
                to_cam_norm = float(np.linalg.norm(to_cam))
                if to_cam_norm > 0.01:
                    to_cam /= to_cam_norm
                    dot = float(np.dot(screen_world[[0,2]], to_cam[[0,2]]))
                    if dot < 0:  # screen faces away from camera → flip
                        rotation = (np.array(rotation) @ _RY180).tolist()
                        print(f"    [orient] flipped — screen was facing away from camera (dot={dot:.2f})")
                    else:
                        print(f"    [orient] screen faces camera correctly (dot={dot:.2f})")
            print(f"    facing={facing}  front_local={np.round(front_local,2).tolist() if front_local is not None else 'None'}"
                  f"  toward_dir={np.round(toward_dir,2).tolist() if toward_dir is not None else 'None'}")

        # Fallback: if placement failed and geo/VLM redirected away from the original
        # segmentation furniture, retry on the original furniture before giving up.
        if world_pos is None and placement_type == "on_surface":
            _orig_fi = seg.get("furniture_index", -1)
            _orig_furn_fb = furn_map.get(_orig_fi)
            if (_orig_fi >= 0 and _orig_fi != furn_idx
                    and _orig_furn_fb is not None
                    and "position_m" in _orig_furn_fb
                    and "size_m"     in _orig_furn_fb
                    and "eff_h"      in _orig_furn_fb):
                print(f"  [fallback] placement blocked on [{furn_idx}] — retrying on original [{_orig_fi}]")
                _fb_local = _bbox_to_table_local(
                    _orig_furn_fb, box_px, camera,
                    img_w=int(_orig_ref_w), img_h=int(_orig_ref_h),
                    out_dir=out_dir,
                )
                _fb_cx, _fb_cz = (_fb_local if _fb_local is not None else (None, None))
                world_pos = _place_on_surface(_orig_furn_fb, size_m, side, depth_pos, tracker,
                                              out_dir=out_dir,
                                              cx_hint=_fb_cx, cz_hint=_fb_cz)
                if world_pos is not None:
                    furn      = _orig_furn_fb
                    furn_idx  = _orig_fi
                    furn_type = _orig_furn_fb.get("type", "furniture")
                    rotation  = _surface_rotation(_orig_furn_fb, facing, front_local, None)

        if world_pos is None:
            print(f"  placement blocked — skipping")
            continue

        entry: dict = {
            "seg_index":       seg_idx,
            "phrase":          phrase,
            "furniture_index": furn_idx,
            "furniture_type":  furn_type,
            "placement_type":  placement_type,
            "surface_position": side,
            "depth_position":  depth_pos,
            "facing":          facing,
            "_front_local":    front_local.tolist() if front_local is not None else None,
            "glb_file":        glb_rel,
            "glb_path":        str(glb_path),   # absolute path for renderer
            "position_m":      world_pos.tolist(),
            "rotation_3x3":    rotation,
            "scale":           [scale, scale, scale],
            "size_m":          size_m,
            # Renderer helpers
            "index":           seg_idx,
            "type":            phrase.replace(" ", "_"),
            "wall_affinity":   "centre",
            "on_top_of":       furn_idx,
            "skip_ground_removal": True,
            "inpaint_file":    seg.get("inpaint_file"),
        }
        if seg.get("quantity", 1) > 1:
            entry["quantity"] = seg["quantity"]
        # Tag pillow-position overrides so post-loop passes skip
        # bbox_contain / final_clamp / pos_refine for this entry.
        if placement_type == "against_back" and _pillow_pos_locked:
            entry["_pos_locked"] = True

        placements.append(entry)
        # Track surface occupancy so the next same-phrase decoration can
        # prefer a different surface (one lamp per cabinet, etc.)
        if furn_idx is not None and furn_idx >= 0:
            _phrase_occupied.setdefault(_norm_phrase(phrase), set()).add(int(furn_idx))
        print(f"  → pos={world_pos.round(3).tolist()}  scale={scale:.4f}  type={placement_type}")

        # Silhouette-based scale refinement: compare projected 2D bbox vs mask
        # box_px.  Now also runs for against_back placements (pillows): the
        # default 0.50×0.50m size left them visibly oversized on rotated sofas,
        # but the mask in the photo is the ground truth for the visible
        # silhouette, so we honour it here too — type-specific size caps
        # downstream still prevent egregious blow-up.
        if (camera is not None and _furn_render_ds is not None and box_px is not None):
            # Count distinct mask blobs so the silhouette refinement can size
            # for ONE instance when SAM merged several piles into one segment.
            _mask_blob_count = 1
            _mf_seg = seg.get("mask_file")
            if _mf_seg:
                _mp_seg = out_dir / "decorations" / "segmented" / _mf_seg
                _mask_blob_count = _count_mask_components(_mp_seg)
            refined_scale = _silhouette_scale_refine(
                entry, box_px, camera,
                render_w=_furn_render_ds.width,
                render_h=_furn_render_ds.height,
                orig_ref_w=int(_orig_ref_w),
                orig_ref_h=int(_orig_ref_h),
                n_instances_in_mask=_mask_blob_count,
            )
            # Enforce type-specific real-world size caps (books have a known
            # size range — silhouette matching shouldn't push a book stack
            # past ~30×25 cm footprint even if the mask is huge).  Look up
            # the cap by phrase keyword and shrink refined_scale if needed so
            # no dimension exceeds its physical maximum.
            _cap_w = None
            _cap_d = None
            _cap_h = None
            _floor_h = None
            for _kw, _lo_h, _hi_h, _max_w, _max_d in _SIZE_CLAMPS:
                if _kw in phrase.lower():
                    _cap_w = _max_w
                    _cap_d = _max_d
                    _cap_h = _hi_h
                    _floor_h = _lo_h
                    break
            if _cap_w is not None and scale > 1e-6:
                _native_w = entry["size_m"].get("width_m",  0.3) / scale
                _native_h = entry["size_m"].get("height_m", 0.3) / scale
                _native_d = entry["size_m"].get("depth_m",  0.3) / scale
                _cap_scale = refined_scale
                if _cap_w and _native_w > 1e-6:
                    _cap_scale = min(_cap_scale, _cap_w / _native_w)
                if _cap_d and _native_d > 1e-6:
                    _cap_scale = min(_cap_scale, _cap_d / _native_d)
                if _cap_h and _native_h > 1e-6:
                    _cap_scale = min(_cap_scale, _cap_h / _native_h)
                if _cap_scale < refined_scale - 1e-4:
                    print(f"    [size_cap] '{phrase}': silhouette scale {refined_scale:.4f} "
                          f"exceeds footprint cap {_cap_w}×{_cap_d}m / height {_cap_h}m "
                          f"→ clamping to {_cap_scale:.4f}")
                    refined_scale = _cap_scale
                # Enforce per-type MIN height as a floor on the
                # silhouette-refined scale.  Without this, the silhouette
                # can shrink a book stack below 0.22 m so it looks like a
                # single thin book rather than a recognisable stack.
                # IMPORTANT: clamp the floor scale at the cap scale —
                # a flat-stack GLB (native_h=0.49, native_w=1.96, ratio
                # 1:4) needs a huge uniform scale to hit a min_h of
                # 0.22, which then violates the type's max width cap
                # and produces ~0.9 m wide books that fill the table.
                # Cap takes precedence over floor.
                if _floor_h is not None and _native_h > 1e-6:
                    _floor_scale = _floor_h / _native_h
                    # Don't let the floor exceed the cap.  _cap_scale
                    # was just computed above; refined_scale was
                    # clamped to it if needed, so refined_scale is now
                    # the cap-respected scale.  Use that as the upper
                    # bound for the floor.
                    _floor_scale_capped = min(_floor_scale, _cap_scale)
                    if refined_scale < _floor_scale_capped - 1e-4:
                        if _floor_scale > _cap_scale + 1e-4:
                            print(f"    [size_floor] '{phrase}': min height "
                                  f"{_floor_h}m demands scale "
                                  f"{_floor_scale:.4f}, but cap allows max "
                                  f"{_cap_scale:.4f} (flat-stack GLB) — "
                                  f"raising silhouette {refined_scale:.4f} "
                                  f"to cap {_cap_scale:.4f} only")
                        else:
                            print(f"    [size_floor] '{phrase}': silhouette "
                                  f"scale {refined_scale:.4f} below min "
                                  f"height {_floor_h}m floor → raising to "
                                  f"{_floor_scale_capped:.4f}")
                        refined_scale = _floor_scale_capped

            if abs(refined_scale - scale) > 1e-4:
                entry["scale"] = [refined_scale, refined_scale, refined_scale]
                # Recompute size_m to reflect the new scale
                for k in ("width_m", "height_m", "depth_m"):
                    if k in entry.get("size_m", {}):
                        entry["size_m"][k] *= refined_scale / scale
                scale = refined_scale

        # ── VLM multi-instance check ────────────────────────────────────────
        # When the size-capped mesh is much smaller than the mask (i.e. the
        # mask contains multiple identical piles / stacks / copies of the item),
        # ask the VLM how many distinct instances are visible in the reference
        # crop.  If count > 1, DUPLICATE the entry and arrange copies along
        # the mask's long axis instead of stretching one mesh to cover them.
        _MULTI_CHECK_TYPES = ("book", "books", "tray", "magazine", "journal",
                               "notebook", "folder", "candle", "mug", "cup",
                               "plant", "vase", "pillow", "cushion")
        if (placement_type == "on_surface"
                and box_px is not None and crop_path is not None
                and any(kw in phrase.lower() for kw in _MULTI_CHECK_TYPES)
                and seg.get("quantity", 1) <= 1  # segmentation didn't already say N
                and camera is not None and _furn_render_ds is not None):
            try:
                # Only ask the VLM when the mask is meaningfully wider or
                # taller than the capped mesh projects — otherwise one copy
                # already fills it and duplication would pile items on top of
                # each other.
                _mask_w_px = float(box_px[2] - box_px[0])
                _mask_h_px = float(box_px[3] - box_px[1])
                _ds = _furn_render_ds.width / float(_orig_ref_w)
                _mw_r = _mask_w_px * _ds
                _mh_r = _mask_h_px * _ds
                # Re-project the mesh bbox to estimate current footprint.
                # Use the silhouette function's internal logic (cheaper: just
                # compare size_m against expected mask footprint).
                _cur_w = float(entry["size_m"].get("width_m", 0.3))
                _cur_h = float(entry["size_m"].get("height_m", 0.3))
                # A rough pixel-per-metre conversion from the silhouette step
                # (mask_w_r / native_proj_w ≈ current scale ratio).  Trigger
                # the count check only when the mask's wider dimension is at
                # least 1.6× what a single copy would fill.
                _need_count = (_mw_r / max(_cur_w * 400, 1) >= 1.6
                               or _mh_r / max(_cur_h * 400, 1) >= 1.6)
                # Mask-blob hint: if connected-components already say there
                # are N piles, trust that as a hard floor on _n_instances —
                # the silhouette refine has already shrunk per-instance size,
                # so we DO want N copies regardless of the bbox heuristic.
                if _mask_blob_count > 1:
                    _need_count = True
                if _need_count:
                    _crop_img = _PILImg.open(str(crop_path)).convert("RGB")
                    _n_instances = _vlm_count_instances(_crop_img, phrase)
                    if _mask_blob_count > 1 and _n_instances < _mask_blob_count:
                        print(f"    [count_instances] VLM said {_n_instances}, "
                              f"raising to mask-blob count {_mask_blob_count}")
                        _n_instances = _mask_blob_count
                else:
                    _n_instances = 1
                if _n_instances > 1:
                    # Duplicate N-1 additional copies.  We want the copies to
                    # appear SIDE-BY-SIDE in the rendered image along the
                    # mask's long axis — NOT along the furniture's local
                    # right axis, which may be diagonal relative to the
                    # camera (e.g. living_room9's coffee table has its local
                    # +X pointing into the -x,-z quadrant, so spreading along
                    # it made the two book stacks appear "one behind the
                    # other" in the render).  Use the camera-aligned
                    # horizontal-in-image direction (or vertical, for tall
                    # masks), projected onto the table's horizontal plane.
                    _mask_is_wide = _mask_w_px >= _mask_h_px
                    _axis_world: np.ndarray | None = None
                    try:
                        from object_placement.wall_mounted.wall_mounted_object_placement import (
                            _camera_axes as _ca_dup,
                        )
                        _cp_d = np.array(camera["position_m"], dtype=np.float64)
                        _la_d = np.array(camera["look_at_m"],  dtype=np.float64)
                        _up_d = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
                        _rv_d, _uv_d, _fv_d = _ca_dup(_cp_d, _la_d, _up_d)
                        # If the mask is wide (spread in image-x) use the
                        # camera's right axis projected onto the horizontal
                        # plane; if tall, use the in-plane projection of the
                        # camera-forward (how depth maps into image-y).
                        _base = _rv_d if _mask_is_wide else _fv_d
                        _proj = np.array([_base[0], 0.0, _base[2]], dtype=np.float64)
                        _n_proj = float(np.linalg.norm(_proj))
                        if _n_proj > 1e-6:
                            _axis_world = _proj / _n_proj
                    except Exception:
                        _axis_world = None
                    if _axis_world is None:
                        # Fallback: furniture local axis (old behaviour)
                        _R_furn_dup = _rot_matrix(furn) if furn is not None else np.eye(3)
                        _axis_world = _R_furn_dup[:, 0 if _mask_is_wide else 2]
                    # Spacing: one item's width plus a small gap.  The "width"
                    # here is the mesh's side-along-long-axis extent.  Use the
                    # larger of width/depth so adjacent copies don't visually
                    # overlap even if the camera axis isn't perfectly aligned
                    # with the mesh's width axis.
                    _mesh_extent = max(
                        float(entry["size_m"].get("width_m", 0.25)),
                        float(entry["size_m"].get("depth_m", 0.25)),
                    )
                    _spacing = _mesh_extent * 1.05
                    # Total span for N items; centre at original pos
                    _total = (_n_instances - 1) * _spacing
                    _start_offset = -_total / 2.0
                    _base_pos = np.array(entry["position_m"], dtype=np.float64)
                    # Move the original entry to the first slot
                    entry["position_m"] = (_base_pos + _start_offset * _axis_world).tolist()
                    # Append N-1 duplicates
                    import copy as _copy_mod
                    for _k in range(1, _n_instances):
                        _dup = _copy_mod.deepcopy(entry)
                        _dup["seg_index"] = seg_idx * 100 + _k
                        _dup["position_m"] = (
                            _base_pos + (_start_offset + _spacing * _k) * _axis_world
                        ).tolist()
                        placements.append(_dup)
                    print(f"    [multi_instance] duplicated to {_n_instances} copies "
                          f"of '{phrase}' spaced {_spacing:.2f}m along "
                          f"{'width' if _mask_is_wide else 'depth'} axis")
                    # Re-render with the new copies
                    _render_incremental(out_dir, placements, seg_idx,
                                        phrase + "_multi", placements_dir,
                                        inpaint_dir=inpaint_dir)
            except Exception as _mie:
                print(f"    [multi_instance] check failed: {_mie}")

        # For against_back pillows/cushions: compute non-uniform scale from the
        # reference bbox aspect ratio so the GLB shape matches the reference silhouette.
        # The pillow's X axis (GLB local X) corresponds to the sofa's width direction.
        # Shrinking scale_x makes a wide rectangular pillow appear more square.
        #
        # IMPORTANT: skip this when the mask bbox has an extreme aspect ratio —
        # real throw pillows are approximately 0.7–2.0 width/height.  Anything
        # below 0.7 (very tall and narrow) strongly suggests the pillow is
        # OCCLUDED in the reference photo (e.g. another pillow in front hides
        # half of it), and shrinking scale_x to fit the occluded bbox turns a
        # square pillow into a thin vertical strip.  Keep the GLB's native
        # uniform scale in that case.
        # NOTE: the per-pillow anisotropic [shape] adjustment is disabled.
        # When two pillows in a pair (left+right sofa) had different bbox
        # aspects in the photo, one would get scale_x shrunk (paired with a
        # full-aspect pillow) while the other kept native uniform — they
        # ended up looking visibly different even though they're meant to
        # be matching.  Keeping uniform scale across the pair is more
        # important than matching each pillow's individual silhouette.
        if False and (placement_type == "against_back" and box_px is not None
                and size_m.get("width_m", 0) > 1e-4 and size_m.get("height_m", 0) > 1e-4):
            pass

        # Per-object incremental render (initial placement)
        safe       = phrase.replace(" ", "_")
        step_path  = placements_dir / f"render_step_{seg_idx:02d}_{safe}.png"
        _render_incremental(out_dir, placements, seg_idx, phrase, placements_dir, inpaint_dir=inpaint_dir)

        # For against_back: VLM arm-rest collision check.
        # If the pillow visually overlaps a sofa arm rest, slide it toward center.
        if placement_type == "against_back" and _ref_photo is not None and step_path.exists():
            try:
                _arm_render = _PILImg.open(step_path).convert("RGB")
                _arm_result = _vlm_arm_rest_check(_arm_render, _ref_photo, phrase)
                if _arm_result is not None and _arm_result.get("overlap"):
                    _arm_dir = _arm_result.get("direction", "")
                    _arm_shift = float(_arm_result.get("shift_m", 0.10))
                    # Skip arm-rest shift entirely when the pillow is already
                    # essentially centred on its sofa.  The VLM frequently
                    # reports "overlap on left arm rest" as a false positive
                    # for centred pillows on rotated sofas (perspective makes
                    # the pillow look pushed to one side), and applying the
                    # blind shift then drives the pillow OUT of the sofa BBox.
                    _arm_pre_pos = np.array(entry["position_m"], dtype=np.float64)
                    if furn is not None:
                        _arm_R = _rot_matrix(furn)
                        _arm_furn_pos = np.array(furn["position_m"], dtype=np.float64)
                        _arm_offset = _arm_pre_pos - _arm_furn_pos
                        _arm_cx_local = float(np.dot(_arm_offset, _arm_R[:, 0]))
                    else:
                        _arm_cx_local = 0.0
                    # Decide whether to apply the shift.  Block in two cases:
                    #   1. Pillow is already centred (likely VLM false positive
                    #      on a rotated sofa).
                    #   2. The visible seat band is so narrow that the pillow
                    #      already fills it (no room to shift without going
                    #      past the sofa edge).
                    _skip_arm_shift_reason = None
                    if abs(_arm_cx_local) < 0.05:
                        _skip_arm_shift_reason = (
                            f"pillow already centred (cx_local="
                            f"{_arm_cx_local:.3f}m)")
                    else:
                        _seat_band_pre = (furn.get("_seat_band_x")
                                          if furn is not None else None)
                        if _seat_band_pre is not None:
                            _sb_min, _sb_max = _seat_band_pre
                            _pillow_w = float(size_m.get("width_m", 0.4))
                            _seat_room_total = (_sb_max - _sb_min) - _pillow_w
                            if _seat_room_total <= 0.05:
                                _skip_arm_shift_reason = (
                                    f"pillow ({_pillow_w:.2f}m) already fills "
                                    f"visible seat band [{_sb_min:+.3f},"
                                    f"{_sb_max:+.3f}] (only "
                                    f"{_seat_room_total*100:+.0f}cm slack)")
                    if _skip_arm_shift_reason is not None:
                        print(f"    [arm_rest] VLM said overlap on {_arm_dir} but "
                              f"{_skip_arm_shift_reason} — ignoring")
                    else:
                        print(f"    [arm_rest] VLM: overlap on {_arm_dir} arm rest — shift {_arm_shift:.2f}m toward center")
                        # Shift in the OPPOSITE direction of the overlapping arm rest,
                        # using the camera's horizontal right axis (not sofa-local) so
                        # "left/right" matches camera perspective which VLM reasons about.
                        _arm_sign = -1.0 if _arm_dir == "right" else 1.0  # move AWAY from arm rest
                        if camera is not None:
                            from object_placement.wall_mounted.wall_mounted_object_placement import _camera_axes as _cam_ax
                            _c_cp = np.array(camera["position_m"], dtype=np.float64)
                            _c_la = np.array(camera["look_at_m"],  dtype=np.float64)
                            _c_up = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
                            _cam_right, _, _ = _cam_ax(_c_cp, _c_la, _c_up)
                            _cam_right = np.array(_cam_right, dtype=np.float64)
                            _cam_right[1] = 0.0  # keep horizontal
                            _cam_right /= (np.linalg.norm(_cam_right) + 1e-9)
                            _shift_dir = _cam_right
                        else:
                            _right_w, _, _ = _world_axes(furn) if furn is not None else (np.array([1, 0, 0]), None, None)
                            _shift_dir = _right_w
                        _new_pos = _arm_pre_pos + _arm_sign * _arm_shift * _shift_dir
                        # Clamp result to keep pillow inside sofa BBox (sofa-local
                        # cx must stay within sofa_half_w − pillow_half_w − arm_reserve).
                        if furn is not None:
                            _arm_sw   = float(furn.get("size_m", {}).get("width_m", 0.8))
                            _arm_dw   = float(size_m.get("width_m", 0.4))
                            _arm_resv = _arm_sw * 0.10                       # ~10% per side
                            _arm_max  = max(_arm_sw / 2.0 - _arm_dw / 2.0 - _arm_resv, 0.0)
                            _new_offset   = _new_pos - _arm_furn_pos
                            _new_cx_local = float(np.dot(_new_offset, _arm_R[:, 0]))
                            _clamped_cx_local = float(np.clip(_new_cx_local, -_arm_max, _arm_max))
                            if abs(_clamped_cx_local - _new_cx_local) > 1e-3:
                                # Reverse the cx_local shift in world coords.
                                _delta = _clamped_cx_local - _new_cx_local
                                _new_pos = _new_pos + _delta * _arm_R[:, 0]
                                print(f"    [arm_rest] post-shift cx_local={_new_cx_local:+.3f}m exceeds "
                                      f"sofa-half-w-minus-padding ±{_arm_max:.3f}m → clamped to "
                                      f"{_clamped_cx_local:+.3f}m to keep pillow inside sofa BBox")
                        # Animate the slide so the arm-rest fix is visible
                        # in the GIF — without intermediate frames the pillow
                        # appears to teleport from over-the-arm to centred,
                        # making the procedure hard to see.
                        _render_pillow_slide(
                            out_dir, placements, entry,
                            old_pos=_arm_pre_pos.tolist(),
                            new_pos=_new_pos.tolist(),
                            seg_idx=seg_idx, phrase=phrase + "_arm_fixed",
                            placements_dir=placements_dir,
                            n_steps=pillow_slide_frames,
                            inpaint_dir=inpaint_dir,
                            label_suffix="armslide",
                        )
                        entry["position_m"] = _new_pos.tolist()
                        world_pos = _new_pos
                        _render_incremental(out_dir, placements, seg_idx, phrase + "_arm_fixed", placements_dir, inpaint_dir=inpaint_dir)
                else:
                    print(f"    [arm_rest] VLM: no arm-rest overlap detected. {_arm_result.get('reasoning','') if _arm_result else ''}")
            except Exception as _ar_e:
                print(f"    [arm_rest] check failed: {_ar_e}")

        # ── Verify size (DISABLED) ────────────────────────────────────
        # Was: VLM-driven post-placement size check that asked
        # whether the rendered object looked too big or too small
        # vs the reference photo, then applied ±30% scale.
        # Disabled because the VLM was unreliable here: it
        # oscillated between "shrink" and "enlarge" between iters,
        # frequently hallucinated "the reference photo is missing"
        # even when supplied, and on small objects (books, plants)
        # judged size against the wrong neighbour and produced
        # ±30% drift each run.  Silhouette refinement already
        # handles the common case (mesh-projected bbox vs mask bbox).
        # Set the gate to `False` to keep the implementation around
        # in case we want to re-enable for a specific item type.
        if (False
                and placement_type == "on_surface"
                and _ref_photo is not None and step_path.exists()):
            try:
                _SIZE_MAX_ITERS = 2
                _SIZE_SHRINK = 0.7
                _SIZE_ENLARGE = 1.3
                _last_size_action: "str | None" = None
                for _size_iter in range(_SIZE_MAX_ITERS):
                    _size_render = _PILImg.open(step_path).convert("RGB")
                    _size_action = _vlm_verify_size(
                        _ref_photo, _size_render, phrase)
                    if _size_action is None or _size_action == "correct":
                        break
                    if _size_iter > 0 and _size_action == _last_size_action:
                        # Same action repeated — VLM still not happy
                        # but probably converging asymptotically; stop
                        # to avoid runaway shrinking.
                        print(f"    [verify_size] iter {_size_iter}: "
                              f"{_size_action} repeated — stopping")
                        break
                    _factor = (_SIZE_SHRINK if _size_action == "shrink"
                               else _SIZE_ENLARGE)
                    _old_scale = float(entry["scale"][0])
                    _new_scale = _old_scale * _factor
                    entry["scale"] = [_new_scale, _new_scale, _new_scale]
                    for _k in ("width_m", "height_m", "depth_m"):
                        if _k in entry.get("size_m", {}):
                            entry["size_m"][_k] *= _factor
                    print(f"    [verify_size] iter {_size_iter}: "
                          f"applied {_size_action} (×{_factor}) → "
                          f"scale {_old_scale:.4f} → {_new_scale:.4f} "
                          f"(size_m {entry['size_m'].get('width_m',0):.3f}×"
                          f"{entry['size_m'].get('height_m',0):.3f}×"
                          f"{entry['size_m'].get('depth_m',0):.3f}m)")
                    _last_size_action = _size_action
                    _render_incremental(out_dir, placements, seg_idx,
                                          phrase + f"_size{_size_iter}",
                                          placements_dir,
                                          inpaint_dir=inpaint_dir)
            except Exception as _se:
                print(f"    [verify_size] check failed: {_se}")

        # Track against_back group for post-group reorder check.
        # Key is (furn_idx,) so ALL pillows/cushions on the same sofa are grouped together,
        # regardless of their individual phrase ("pillows", "gray throw pillow", etc.).
        # This lets the VLM compare the full sofa arrangement in one shot.
        if placement_type == "against_back":
            _ab_key = (furn_idx,)
            # Carry the within-group rank hint into the entry so the later
            # sidebyside repack sorts by the intended L→R colour sequence
            # instead of by initial positions (which collapse for duplicate
            # bboxes like two "pillows" phrases detecting the same sofa pillow).
            _rank_carry = seg.get("_cx_rank_hint")
            if _rank_carry is not None:
                entry["_cx_rank_hint"] = float(_rank_carry)
            _ab_groups.setdefault(_ab_key, []).append(entry)

        # Global 3D AABB collision check: if the new decoration overlaps an
        # already-placed one, nudge it in 8 horizontal directions then give up.
        # Skip for against_back (pillows) — rank hints space them; sofa pillows
        # naturally touch and the nudge disrupts the intended arrangement.
        _half = np.array([
            size_m.get("width_m",  0.3) / 2.0,
            size_m.get("height_m", 0.3) / 2.0,
            size_m.get("depth_m",  0.3) / 2.0,
        ])
        _resolve_pos = world_pos.copy()
        _was_nudged = False
        if placement_type != "against_back" and tracker.world_collides(_resolve_pos, _half):
            _nudge_step = max(float(_half[0]), float(_half[2])) * 1.2 + 0.04
            _resolved = False
            for _angle in np.linspace(0, 2 * np.pi, 9)[:-1]:
                _candidate = world_pos.copy()
                _candidate[0] += _nudge_step * np.cos(_angle)
                _candidate[2] += _nudge_step * np.sin(_angle)
                if not tracker.world_collides(_candidate, _half):
                    _resolve_pos = _candidate
                    _resolved = True
                    _was_nudged = True
                    print(f"    [collide_resolve] nudged {np.degrees(_angle):.0f}° → "
                          f"{_resolve_pos.round(3).tolist()}")
                    break
            if not _resolved:
                print(f"    [collide_resolve] could not resolve — placing anyway")
            # After the horizontal nudge we may have moved off the original
            # surface (e.g. lamp pushed past the table edge).  Re-raycast Y to
            # either snap back to whatever surface exists here, or revert the
            # nudge if no surface is present (don't leave the object floating
            # at the original table's height in mid-air).
            if _was_nudged and out_dir is not None and placement_type == "on_surface":
                _old_y = float(world_pos[1])
                _new_y = _actual_surface_height(out_dir,
                                                 float(_resolve_pos[0]),
                                                 float(_resolve_pos[2]),
                                                 _old_y)
                if _new_y == _old_y:
                    print(f"    [collide_resolve] post-nudge raycast missed — reverting nudge")
                    _resolve_pos = world_pos.copy()
                    _was_nudged = False
                else:
                    _resolve_pos[1] = _new_y
            world_pos = _resolve_pos
            entry["position_m"] = world_pos.tolist()

        tracker.register_world(world_pos, size_m)

        # Render again if object was nudged to show the moved state
        if _was_nudged:
            _render_incremental(out_dir, placements, seg_idx, phrase + "_moved", placements_dir, inpaint_dir=inpaint_dir)

        # VLM post-placement verification — catch wrong furniture assignments
        # Rebuild the combined (labeled+highlighted) image for this decoration
        _verify_combined: "Image.Image | None" = None
        if _labeled_render is not None and box_px is not None:
            try:
                _verify_combined = _highlight_bbox(_labeled_render, box_px, scale=_bbox_scale)
            except Exception:
                _verify_combined = _labeled_render

        if (_verify_combined is not None and step_path.exists()
                and not seg.get("_duplicate_resolved")):
            try:
                step_img = _PILImg.open(step_path).convert("RGB")
                correct_idx = _vlm_verify_placement(
                    _verify_combined, step_img,
                    phrase, furn_list, furn_idx,
                    ref_photo=_ref_photo,
                    camera=camera,
                )

                # Verify-VLM JSON-parse fallback: when the VLM returns no
                # parseable JSON (correct_idx is None), the iteration was
                # silently skipping the correction even when the original
                # segmentation was wrong (e.g. lamp on small coffee_table [5]
                # in the photo, but actually shown on large coffee_table [3]).
                # Run a geo-search ourselves to find the same-type surface
                # geometrically closest to the decoration's bbox in the
                # current render — if a different surface is meaningfully
                # closer, switch to it.
                if (correct_idx is None and placement_type == "on_surface"
                        and box_px is not None and camera is not None
                        and furn_idx >= 0):
                    _cur_type_lower = furn_map.get(furn_idx, {}).get("type", "").lower()
                    if _cur_type_lower:
                        try:
                            from object_placement.wall_mounted.wall_mounted_object_placement import (
                                _camera_axes as _vfb_cax, _project_vertex as _vfb_pv,
                            )
                            _vfb_cp = np.array(camera["position_m"], dtype=np.float64)
                            _vfb_la = np.array(camera["look_at_m"],  dtype=np.float64)
                            _vfb_up = np.array(camera.get("up",[0,1,0]), dtype=np.float64)
                            _vfb_rv, _vfb_uv, _vfb_fv = _vfb_cax(_vfb_cp, _vfb_la, _vfb_up)
                            _vfb_W = _labeled_render.width if _labeled_render else 1500
                            _vfb_H = _labeled_render.height if _labeled_render else 1125
                            _vfb_fx = _vfb_W / (2.0*np.tan(np.radians(float(camera["hfov_deg"])/2.0)))
                            _vfb_cx2, _vfb_cy2 = _vfb_W/2.0, _vfb_H/2.0
                            _vfb_mx = (box_px[0]+box_px[2])/2.0*_bbox_scale
                            _vfb_my = (box_px[1]+box_px[3])/2.0*_bbox_scale
                            _vfb_best_idx = furn_idx
                            _vfb_best_d2  = float("inf")
                            _vfb_cur_d2   = float("inf")
                            for _vfb_f in furn_list:
                                _vfb_fi = _vfb_f.get("index", -1)
                                if _vfb_fi < 0 or _vfb_f.get("type", "").lower() != _cur_type_lower:
                                    continue
                                if "position_m" not in _vfb_f or "size_m" not in _vfb_f:
                                    continue
                                _vfb_p = np.array(_vfb_f["position_m"], dtype=np.float64)
                                _vfb_p[1] += _vfb_f.get("eff_h", 0.5)
                                _vfb_px2, _vfb_py2, _vfb_zc2 = _vfb_pv(
                                    _vfb_p, _vfb_cp, _vfb_rv, _vfb_uv, _vfb_fv,
                                    _vfb_fx, _vfb_cx2, _vfb_cy2)
                                if _vfb_zc2 <= 0.01:
                                    continue
                                _d2 = (_vfb_px2 - _vfb_mx)**2 + (_vfb_py2 - _vfb_my)**2
                                if _vfb_fi == furn_idx:
                                    _vfb_cur_d2 = _d2
                                if _d2 < _vfb_best_d2:
                                    _vfb_best_d2 = _d2
                                    _vfb_best_idx = _vfb_fi
                            # Only switch if the new index is meaningfully closer
                            # (saves us from same-distance ties / numerical noise).
                            if (_vfb_best_idx != furn_idx
                                    and _vfb_best_d2 < _vfb_cur_d2 * 0.5):
                                print(f"  [verify no-json fallback] VLM didn't parse — "
                                      f"geo nearest {_cur_type_lower} is [{_vfb_best_idx}] "
                                      f"(d²={_vfb_best_d2:.0f} vs cur [{furn_idx}] d²={_vfb_cur_d2:.0f}) "
                                      f"→ correcting")
                                correct_idx = _vfb_best_idx
                        except Exception as _vfbe:
                            print(f"  [verify no-json fallback] failed: {_vfbe}")

                if correct_idx is not None and correct_idx != furn_idx:
                    # For against_back items (pillows) on seating: block verify from
                    # moving to a DIFFERENT seating piece — VLM confuses sofas.
                    # Allow only corrections to non-seating furniture (genuine mistakes).
                    _vfy_ph = phrase.lower()
                    _vfy_is_ab = any(kw in _vfy_ph for kw in _AGAINST_BACK_PHRASES)
                    _vfy_cur_type = furn_map.get(furn_idx, {}).get("type", "").lower() if furn_idx >= 0 else ""
                    _vfy_new_type = furn_map.get(correct_idx, {}).get("type", "").lower() if correct_idx >= 0 else ""
                    if seg.get("_dedup_protect_src") and _vfy_is_ab:
                        # Dedup already redistributed the duplicate off this
                        # piece — the seg still here is the LEGITIMATE occupant.
                        # Don't let verify move it elsewhere (cross-type or
                        # otherwise), or the source seat ends up with zero
                        # pillows (which was the user's "sofa [1] is empty" bug).
                        print(f"  [verify] pillow on [{furn_idx}] is dedup-protected "
                              f"(sibling already redistributed) — blocking correction to [{correct_idx}]")
                        correct_idx = furn_idx
                    elif (_vfy_is_ab and _vfy_cur_type in _SEATING_TYPES
                            and _vfy_new_type in _SEATING_TYPES
                            and _vfy_cur_type == _vfy_new_type):
                        # Only block same-type seating corrections (sofa↔sofa, chair↔chair)
                        # — VLM reliably confuses same-type instances but is correct when
                        # switching types (e.g. sofa→chair = genuine fix).
                        print(f"  [verify] pillow on {_vfy_cur_type} [{furn_idx}] — blocking same-type "
                              f"correction to [{correct_idx}] (VLM confuses same-room {_vfy_cur_type}s)")
                        correct_idx = furn_idx
                    elif (not _vfy_is_ab
                          and _vfy_cur_type not in _SEATING_TYPES
                          and _vfy_new_type in _SEATING_TYPES):
                        # Block moving flat-surface items (books, lamps, monitors,
                        # keyboards) from a table onto seating.  These items can't
                        # physically rest on a sofa/chair seat — correcting them
                        # here lands them on the back-rest height and floats
                        # them in mid-air.  If the VLM believes the reference
                        # truly shows books on a chair, it's more often a VLM
                        # misread than a real design choice.
                        # Geo-fallback: instead of fully ignoring the VLM
                        # signal, find the geometrically-nearest SAME-TYPE
                        # surface (e.g. a different coffee_table) — the VLM
                        # is at least telling us that the current furniture
                        # is wrong, even if it picked the wrong replacement.
                        print(f"  [verify] on_surface item on {_vfy_cur_type} [{furn_idx}] "
                              f"— blocking correction to seating {_vfy_new_type} [{correct_idx}], "
                              f"falling back to nearest same-type surface")
                        try:
                            from object_placement.wall_mounted.wall_mounted_object_placement import (
                                _camera_axes as _gf_cax, _project_vertex as _gf_pv,
                            )
                            _gf_cp = np.array(camera["position_m"], dtype=np.float64)
                            _gf_la = np.array(camera["look_at_m"],  dtype=np.float64)
                            _gf_up = np.array(camera.get("up",[0,1,0]), dtype=np.float64)
                            _gf_rv, _gf_uv, _gf_fv = _gf_cax(_gf_cp, _gf_la, _gf_up)
                            _gf_W = _labeled_render.width if _labeled_render else 1500
                            _gf_H = _labeled_render.height if _labeled_render else 1125
                            _gf_fx = _gf_W / (2.0*np.tan(np.radians(float(camera["hfov_deg"])/2.0)))
                            _gf_cx2, _gf_cy2 = _gf_W/2.0, _gf_H/2.0
                            _gf_mx = (box_px[0]+box_px[2])/2.0*_bbox_scale
                            _gf_my = (box_px[1]+box_px[3])/2.0*_bbox_scale
                            _best_idx = furn_idx
                            _best_d2  = float("inf")
                            for _gf_f in furn_list:
                                _gf_fi = _gf_f.get("index", -1)
                                if _gf_fi < 0 or _gf_f.get("type", "").lower() != _vfy_cur_type:
                                    continue
                                if "position_m" not in _gf_f or "size_m" not in _gf_f:
                                    continue
                                _gf_p = np.array(_gf_f["position_m"], dtype=np.float64)
                                _gf_p[1] += _gf_f.get("eff_h", 0.5)
                                _gf_px2, _gf_py2, _gf_zc2 = _gf_pv(
                                    _gf_p, _gf_cp, _gf_rv, _gf_uv, _gf_fv,
                                    _gf_fx, _gf_cx2, _gf_cy2)
                                if _gf_zc2 <= 0.01:
                                    continue
                                _d2 = (_gf_px2 - _gf_mx)**2 + (_gf_py2 - _gf_my)**2
                                if _d2 < _best_d2:
                                    _best_d2  = _d2
                                    _best_idx = _gf_fi
                            if _best_idx != furn_idx and _best_idx >= 0:
                                print(f"  [verify geo-fallback] nearest {_vfy_cur_type} "
                                      f"is [{_best_idx}] (d²={_best_d2:.0f}) — re-placing there")
                                correct_idx = _best_idx
                            else:
                                print(f"  [verify geo-fallback] current [{furn_idx}] is "
                                      f"already nearest same-type — keeping original")
                                correct_idx = furn_idx
                        except Exception as _gfe:
                            print(f"  [verify geo-fallback] failed ({_gfe}) — keeping original")
                            correct_idx = furn_idx
                    else:
                        print(f"  [verify] re-placing on furniture [{correct_idx}] ...")
                    new_furn = furn_map.get(correct_idx)
                    # Skip correction if target furniture is incomplete
                    if new_furn is not None and (
                        "position_m" not in new_furn or "size_m" not in new_furn or "eff_h" not in new_furn
                    ):
                        print(f"  [verify] corrected index {correct_idx} has incomplete data — keeping original")
                        new_furn = None
                        correct_idx = furn_idx
                    # Geometric check: don't move to a furniture that's farther from mask centre
                    if new_furn is not None and box_px is not None and camera is not None:
                        try:
                            from object_placement.wall_mounted.wall_mounted_object_placement import (
                                _camera_axes, _project_vertex,
                            )
                            _cp  = np.array(camera["position_m"], dtype=np.float64)
                            _la  = np.array(camera["look_at_m"],  dtype=np.float64)
                            _up  = np.array(camera.get("up",[0,1,0]), dtype=np.float64)
                            _rv, _uv, _fv = _camera_axes(_cp, _la, _up)
                            _W = _labeled_render.width if _labeled_render else 1500
                            _H = _labeled_render.height if _labeled_render else 1125
                            _fx = _W / (2.0*np.tan(np.radians(float(camera["hfov_deg"])/2.0)))
                            _cx2, _cy2 = _W/2.0, _H/2.0
                            _mx = (box_px[0]+box_px[2])/2.0*_bbox_scale
                            _my = (box_px[1]+box_px[3])/2.0*_bbox_scale
                            def _pd(fidx):
                                f2 = furn_map.get(fidx)
                                if f2 is None or "position_m" not in f2: return float("inf")
                                p2 = np.array(f2["position_m"],dtype=np.float64)
                                p2[1] += f2.get("eff_h",0.5)
                                px2,py2,zc2 = _project_vertex(p2,_cp,_rv,_uv,_fv,_fx,_cx2,_cy2)
                                return float("inf") if zc2<=0.01 else (px2-_mx)**2+(py2-_my)**2
                            d_new = _pd(correct_idx)
                            d_cur = _pd(furn_idx)
                            print(f"    [verify geo] new=[{correct_idx}] d²={d_new:.0f}  cur=[{furn_idx}] d²={d_cur:.0f}")
                            # Skip geo check when furniture TYPES differ — the VLM
                            # correctly identified a type mismatch (e.g. sofa→coffee table),
                            # so trust the type correction even if geometrically farther.
                            _vfy_cur_ftype = furn_map.get(furn_idx, {}).get("type", "").lower()
                            _vfy_new_ftype = furn_map.get(correct_idx, {}).get("type", "").lower() if correct_idx >= 0 else "floor"
                            _type_mismatch = _vfy_cur_ftype != _vfy_new_ftype
                            # Three-signal agreement override: only fire when
                            # the verify VLM's pick matches BOTH the
                            # segmentation's `furn_idx_orig` AND the original
                            # select_furn VLM's pick (`_orig_vlm_furn_idx`).
                            # Geo_check is then the lone dissenter.
                            #
                            # Tightening this from "verify == segmentation"
                            # to "verify == segmentation == original VLM" was
                            # a deliberate fix: when the original VLM AGREED
                            # with geo_check (i.e. select_furn picked the
                            # same furniture geo_check kept), verify alone
                            # shouldn't undo a well-supported same-type call
                            # — that pattern is exactly the "VLM verify
                            # routinely swaps same-type instances" case the
                            # default geo_block protects against.
                            _orig_vlm_match = (
                                _orig_vlm_furn_idx is not None
                                and correct_idx == _orig_vlm_furn_idx
                            )
                            _two_vlm_agree = (correct_idx == furn_idx_orig
                                               and _orig_vlm_match
                                               and correct_idx != furn_idx
                                               and correct_idx >= 0)
                            if (not _type_mismatch and d_new >= d_cur * 1.0
                                    and not _two_vlm_agree):
                                # Same furniture type: skip the correction whenever the
                                # proposed furniture is NOT geometrically closer than the
                                # current one.  VLM verify routinely swaps same-type
                                # instances based on language-level reasoning ("should be
                                # the round one, not the rectangular one") even when the
                                # current placement is geometrically correct — e.g. bowl
                                # on coffee_table [4] at d²=9145 getting moved to [5] at
                                # d²=13893.  Trust geometry for same-type disambiguation.
                                print(f"    [verify geo] same-type, current [{furn_idx}] not farther "
                                      f"(d²={d_cur:.0f} vs new d²={d_new:.0f}) — skipping correction")
                                new_furn = None
                                correct_idx = furn_idx
                            elif _two_vlm_agree and not _type_mismatch:
                                # Cap the override on geo magnitude.  If the
                                # segmentation centroid is dramatically closer
                                # to `furn_idx` than to `correct_idx` (e.g.
                                # books d²=766 vs 417335, a 540× ratio), the
                                # books literally aren't on `correct_idx` —
                                # both VLMs are hallucinating.  Trust geometry.
                                _GEO_OVERRIDE_RATIO_MAX = 10.0
                                _geo_dissent_extreme = (
                                    d_new > _GEO_OVERRIDE_RATIO_MAX * max(d_cur, 1.0)
                                )
                                if _geo_dissent_extreme:
                                    print(f"    [verify geo] two-VLM agree on "
                                          f"[{correct_idx}] but geo dissent extreme "
                                          f"(d²={d_new:.0f} vs cur d²={d_cur:.0f}, "
                                          f"ratio {d_new/max(d_cur,1):.0f}×) — "
                                          f"VLMs likely hallucinating, keeping geo "
                                          f"pick [{furn_idx}]")
                                    new_furn = None
                                    correct_idx = furn_idx
                                else:
                                    print(f"    [verify geo] same-type, but segmentation "
                                          f"[{furn_idx_orig}] AND verify-VLM agree against "
                                          f"placement-time geo override [{furn_idx}] — "
                                          f"trusting two-VLM agreement, applying correction")
                            elif _type_mismatch:
                                # Don't allow against_back pillows to be moved
                                # OFF seating onto a non-seating surface — a
                                # pillow against a coffee_table doesn't make
                                # geometric sense.  This guards against a
                                # common VLM hallucination where it claims a
                                # photo pillow is on a table.
                                _SEATING_LOWER = {t.lower() for t in _SEATING_TYPES}
                                _both_seating = (_vfy_cur_ftype in _SEATING_LOWER
                                                  and _vfy_new_ftype in _SEATING_LOWER)
                                # Seating→seating swaps (sofa↔chair, sofa↔armchair…)
                                # for an against_back pillow are treated like
                                # same-type swaps for geometric purposes:
                                # don't trust the type correction if the new
                                # furniture is materially FARTHER from the
                                # bbox centre.  This catches the case where
                                # strict bbox containment placed the pillow
                                # at the metadata back wall (forward of the
                                # visible cushion on broken meshes), making
                                # verify VLM mistakenly identify it as on a
                                # neighbouring seating piece.
                                if (placement_type == "against_back"
                                        and _vfy_cur_ftype in _SEATING_LOWER
                                        and _vfy_new_ftype not in _SEATING_LOWER):
                                    print(f"    [verify geo] type mismatch "
                                          f"({_vfy_cur_ftype}→{_vfy_new_ftype}) but pillow needs "
                                          f"a seat back — blocking move off seating")
                                    new_furn = None
                                    correct_idx = furn_idx
                                elif (placement_type == "against_back"
                                        and _both_seating
                                        and d_new > 2.0 * d_cur):
                                    print(f"    [verify geo] seating→seating swap "
                                          f"({_vfy_cur_ftype}[{furn_idx}]→{_vfy_new_ftype}"
                                          f"[{correct_idx}]) but new is "
                                          f"{d_new/max(d_cur,1):.1f}× farther "
                                          f"(d²={d_new:.0f} vs {d_cur:.0f}) — "
                                          f"blocking spurious type correction")
                                    new_furn = None
                                    correct_idx = furn_idx
                                elif d_new > 5.0 * max(d_cur, 1.0):
                                    # General sanity check: even when the verify
                                    # VLM picks a different TYPE, reject if the
                                    # new target is dramatically farther in
                                    # pixel space.  Common failure modes:
                                    #   - VLM hallucinates "vase on coffee_table"
                                    #     when the photo bbox is clearly over
                                    #     the desk (5290 → 359149, 68× worse)
                                    #   - VLM says "bed pillows are on the
                                    #     coffee_table" when they're plainly on
                                    #     the bed (43471 → 808428, 19× worse).
                                    # If the mask centre's projected distance
                                    # to the proposed correction is > 5× worse
                                    # than the current placement, the VLM is
                                    # describing the wrong object — trust geometry.
                                    print(f"    [verify geo] type mismatch "
                                          f"({_vfy_cur_ftype}[{furn_idx}]→{_vfy_new_ftype}"
                                          f"[{correct_idx}]) but new is "
                                          f"{d_new/max(d_cur,1):.1f}× farther "
                                          f"(d²={d_new:.0f} vs {d_cur:.0f}) — "
                                          f"VLM likely hallucinating, blocking correction")
                                    new_furn = None
                                    correct_idx = furn_idx
                                else:
                                    print(f"    [verify geo] type mismatch ({_vfy_cur_ftype}→{_vfy_new_ftype}) — trusting type correction")
                        except Exception as _ge:
                            print(f"    [verify geo] check failed: {_ge}")
                    if new_furn is None and correct_idx != -1 and correct_idx != furn_idx:
                        print(f"  [verify] corrected index {correct_idx} not found — keeping original")
                    elif correct_idx == furn_idx:
                        pass  # correction was blocked/same-furniture — original placement stands
                    else:
                        # Re-compute position on corrected furniture
                        new_furn_type = new_furn.get("type", "furniture") if new_furn else "floor"
                        new_world_pos = None
                        new_rotation  = rotation

                        if placement_type == "against_back" and new_furn_type.lower() in _SEATING_TYPES:
                            _vfy_pref_cx = _bbox_preferred_cx(new_furn, box_px, camera,
                                                              img_w=int(_orig_ref_w),
                                                              img_h=int(_orig_ref_h),
                                                              furn_photo_box_px=furn_box_px_by_idx.get(new_furn.get("index")))
                            new_world_pos = _place_against_back(new_furn, size_m, side, tracker,
                                                                out_dir=out_dir, preferred_cx_hint=_vfy_pref_cx)
                            new_rotation  = _against_back_rotation(new_furn)
                        elif placement_type == "on_floor":
                            if new_furn is not None:
                                right_world, _, _ = _world_axes(new_furn)
                                new_fp  = np.array(new_furn["position_m"], dtype=np.float64)
                                half_w  = new_furn["size_m"]["width_m"] / 2
                                offset  = half_w + size_m["width_m"] / 2 + 0.05
                                side_sign = 1.0 if side == "right" else -1.0
                                new_world_pos = new_fp + side_sign * offset * right_world
                            else:
                                new_world_pos = np.array([0.5, 0.0, 0.5], dtype=np.float64)
                            new_rotation = [[1,0,0],[0,1,0],[0,0,1]]
                        else:
                            if new_furn is not None:
                                _vfy_local = _bbox_to_table_local(
                                    new_furn, box_px, camera,
                                    img_w=int(_orig_ref_w), img_h=int(_orig_ref_h),
                                    out_dir=out_dir,
                                )
                                _vfy_cx, _vfy_cz = (_vfy_local if _vfy_local is not None else (None, None))
                                # Pixel-space sanity check on cx (lateral).  When the
                                # bbox bottom is below the table top in image, mesh
                                # raycast can return cx with the wrong SIGN, putting
                                # a left-side decoration on the right of the new
                                # surface.  Override with pixel hint when they
                                # disagree in sign.
                                _vfy_cx_pixel = _bbox_preferred_cx(
                                    new_furn, box_px, camera,
                                    img_w=int(_orig_ref_w), img_h=int(_orig_ref_h),
                                    furn_photo_box_px=furn_box_px_by_idx.get(new_furn.get("index")),
                                )
                                _vfy_hw = new_furn["size_m"]["width_m"] / 2.0
                                if (_vfy_cx is not None
                                        and abs(_vfy_cx_pixel) > 0.05
                                        and abs(_vfy_cx) > 0.05
                                        and (_vfy_cx_pixel * _vfy_cx) < 0):
                                    print(f"    [verify backproj] cx sign disagreement: "
                                          f"mesh_raycast={_vfy_cx:+.3f} vs pixel={_vfy_cx_pixel:+.3f} "
                                          f"→ trusting pixel-based")
                                    _vfy_cx = float(np.clip(_vfy_cx_pixel, -_vfy_hw, _vfy_hw))
                                # Respect the VLM's explicit side/depth preference
                                # when it disagrees with the back-proj hint.  The
                                # back-proj can disagree when the new furniture's
                                # coordinate frame differs significantly from the
                                # originally-assigned one (e.g. lamp re-placed
                                # from small side table [5] to large desk [3] —
                                # the ref bbox maps to different local coords on
                                # the new surface).  If VLM said side=left, don't
                                # let a back-proj cx=+0.56 (right side) on the new
                                # table override that.
                                if _vfy_cx is not None and new_furn is not None:
                                    _nf_hw = new_furn["size_m"]["width_m"] / 2.0
                                    _side_lower = str(side).lower()
                                    if _side_lower == "left" and _vfy_cx > 0:
                                        _vfy_cx = -abs(_vfy_cx)
                                        print(f"    [verify] VLM side='left' overrides back-proj cx "
                                              f"sign → cx={_vfy_cx:+.3f}")
                                    elif _side_lower == "right" and _vfy_cx < 0:
                                        _vfy_cx = abs(_vfy_cx)
                                        print(f"    [verify] VLM side='right' overrides back-proj cx "
                                              f"sign → cx={_vfy_cx:+.3f}")
                                if _vfy_cz is not None and new_furn is not None:
                                    _depth_lower = str(depth_pos).lower()
                                    if _depth_lower == "back" and _vfy_cz > 0:
                                        _vfy_cz = -abs(_vfy_cz)
                                        print(f"    [verify] VLM depth='back' overrides back-proj cz "
                                              f"sign → cz={_vfy_cz:+.3f}")
                                    elif _depth_lower == "front" and _vfy_cz < 0:
                                        _vfy_cz = abs(_vfy_cz)
                                        print(f"    [verify] VLM depth='front' overrides back-proj cz "
                                              f"sign → cz={_vfy_cz:+.3f}")
                                new_world_pos = _place_on_surface(new_furn, size_m, side, depth_pos, tracker,
                                                                  out_dir=out_dir,
                                                                  cx_hint=_vfy_cx, cz_hint=_vfy_cz)
                                # recompute toward_dir for new furniture's own outward axis
                                new_toward_dir: np.ndarray | None = None
                                if facing == "toward_room":
                                    R_nf = _rot_matrix(new_furn)
                                    fwd2 = R_nf @ np.array([0.0, 0.0, 1.0])
                                    fwd2[1] = 0.0
                                    n2 = float(np.linalg.norm(fwd2))
                                    if n2 > 0.01:
                                        new_toward_dir = fwd2 / n2
                                new_rotation  = _surface_rotation(new_furn, facing, front_local, new_toward_dir)

                        if new_world_pos is not None:
                            # Capture pre-swap position for slide animation —
                            # makes the verify-swap obvious in the GIF instead
                            # of a one-frame teleport between sofas.
                            _vfy_pre_pos = list(entry.get("position_m", new_world_pos.tolist()))
                            entry.update({
                                "furniture_index": correct_idx,
                                "furniture_type":  new_furn_type,
                                "position_m":      new_world_pos.tolist(),
                                "rotation_3x3":    new_rotation,
                                "on_top_of":       correct_idx,
                            })
                            print(f"    corrected → pos={new_world_pos.round(3).tolist()}")
                            _render_pillow_slide(
                                out_dir, placements, entry,
                                old_pos=_vfy_pre_pos,
                                new_pos=new_world_pos.tolist(),
                                seg_idx=seg_idx, phrase=phrase + "_verify_swap",
                                placements_dir=placements_dir,
                                n_steps=pillow_slide_frames,
                                inpaint_dir=inpaint_dir,
                                label_suffix="vfyslide",
                            )
                            entry["position_m"] = new_world_pos.tolist()

                            # Re-run silhouette refinement against the new
                            # surface.  The original cap/scale was sized for
                            # the prior (possibly smaller) furniture; once the
                            # piece moves to a different surface (e.g. lamp
                            # going from small side table → large coffee
                            # table), the silhouette can grow it further.
                            if (placement_type == "on_surface"
                                    and new_furn is not None
                                    and box_px is not None):
                                try:
                                    _vfy_refined = _silhouette_scale_refine(
                                        entry, box_px, camera,
                                        render_w=_furn_render_ds.width,
                                        render_h=_furn_render_ds.height,
                                        orig_ref_w=int(_orig_ref_w),
                                        orig_ref_h=int(_orig_ref_h),
                                    )
                                    _vfy_old_scale = float(entry["scale"][0])
                                    if abs(_vfy_refined - _vfy_old_scale) > 1e-4:
                                        # Apply per-type cap on the refined scale
                                        _vfy_cap_w = _vfy_cap_d = _vfy_cap_h = None
                                        _vfy_floor_h = None
                                        for _vk, _vlh, _vhh, _vmw, _vmd in _SIZE_CLAMPS:
                                            if _vk in phrase.lower():
                                                _vfy_cap_w = _vmw
                                                _vfy_cap_d = _vmd
                                                _vfy_cap_h = _vhh
                                                _vfy_floor_h = _vlh
                                                break
                                        _vfy_cap_scale = _vfy_refined
                                        _vfy_native_h = None
                                        if _vfy_old_scale > 1e-6 and _vfy_cap_w:
                                            _nw = entry["size_m"].get("width_m",  0.3) / _vfy_old_scale
                                            _nh = entry["size_m"].get("height_m", 0.3) / _vfy_old_scale
                                            _nd = entry["size_m"].get("depth_m",  0.3) / _vfy_old_scale
                                            _vfy_native_h = _nh
                                            if _vfy_cap_w and _nw > 1e-6:
                                                _vfy_cap_scale = min(_vfy_cap_scale, _vfy_cap_w / _nw)
                                            if _vfy_cap_d and _nd > 1e-6:
                                                _vfy_cap_scale = min(_vfy_cap_scale, _vfy_cap_d / _nd)
                                            if _vfy_cap_h and _nh > 1e-6:
                                                _vfy_cap_scale = min(_vfy_cap_scale, _vfy_cap_h / _nh)
                                        if _vfy_cap_scale < _vfy_refined - 1e-4:
                                            print(f"    [verify size_cap] silhouette {_vfy_refined:.4f} "
                                                  f"clamped to {_vfy_cap_scale:.4f}")
                                            _vfy_refined = _vfy_cap_scale
                                        # Per-type MIN-height floor (mirrors the
                                        # placement-time size_floor): when verify
                                        # moves the item to a different surface
                                        # the silhouette re-pass can shrink it
                                        # further below recognisable size — e.g.
                                        # books on the smaller coffee table
                                        # ending up at h≈0.14m.  Raise scale to
                                        # at least the type's lo_h floor — but
                                        # CAP at _vfy_cap_scale (cap takes
                                        # precedence; flat-stack GLBs would
                                        # blow up width if floor were applied
                                        # uncapped).
                                        if (_vfy_floor_h is not None
                                                and _vfy_native_h is not None
                                                and _vfy_native_h > 1e-6):
                                            _vfy_floor_scale = (_vfy_floor_h
                                                                 / _vfy_native_h)
                                            _vfy_floor_capped = min(
                                                _vfy_floor_scale,
                                                _vfy_cap_scale)
                                            if (_vfy_refined
                                                    < _vfy_floor_capped - 1e-4):
                                                if (_vfy_floor_scale
                                                        > _vfy_cap_scale + 1e-4):
                                                    print(f"    [verify size_floor] "
                                                          f"min height {_vfy_floor_h}m "
                                                          f"demands {_vfy_floor_scale:.4f} "
                                                          f"but cap allows max "
                                                          f"{_vfy_cap_scale:.4f} — "
                                                          f"raising {_vfy_refined:.4f} "
                                                          f"to cap only")
                                                else:
                                                    print(f"    [verify size_floor] silhouette "
                                                          f"{_vfy_refined:.4f} below min "
                                                          f"height {_vfy_floor_h}m floor "
                                                          f"→ raising to "
                                                          f"{_vfy_floor_capped:.4f}")
                                                _vfy_refined = _vfy_floor_capped
                                        if abs(_vfy_refined - _vfy_old_scale) > 1e-4:
                                            entry["scale"] = [_vfy_refined, _vfy_refined, _vfy_refined]
                                            for _kk in ("width_m", "height_m", "depth_m"):
                                                if _kk in entry.get("size_m", {}):
                                                    entry["size_m"][_kk] *= _vfy_refined / _vfy_old_scale
                                            print(f"    [verify silhouette] re-scaled on new surface "
                                                  f"{_vfy_old_scale:.4f} → {_vfy_refined:.4f}")
                                except Exception as _vse:
                                    print(f"    [verify silhouette] refresh failed: {_vse}")

                            # Re-render with corrected position
                            _render_incremental(out_dir, placements, seg_idx, phrase, placements_dir, inpaint_dir=inpaint_dir)
                        else:
                            print(f"  [verify] corrected placement blocked — keeping original")
            except Exception as e:
                print(f"  [verify] post-placement check failed: {e}")

        # VLM orientation check for directional objects (desk lamp, monitor, etc.)
        # Compare the reference crop to the step render and flip 180° if wrong.
        _ORIENT_CHECK_TYPES = {
            "desk lamp", "monitor", "computer monitor", "laptop",
            "keyboard", "computer keyboard", "tv", "television", "screen",
        }
        # Skip the VLM orientation check for SCREEN-like items (monitor,
        # keyboard, laptop, tv) when we have a reliable toward_dir from the
        # desk front + matched chair.  For those items the screen MUST face
        # the user at the chair, and the render camera sees the back — the
        # VLM routinely calls that "wrong" and flips, which then makes the
        # screen face the camera (and away from the user).  LAMPS are
        # different: their orientation is visual-aesthetic only, so the VLM
        # comparison to the reference crop is still useful.
        _SCREEN_ORIENT_TYPES = {
            "monitor", "computer monitor", "laptop", "keyboard",
            "computer keyboard", "tv", "television", "screen",
        }
        # facing=any rectangular on_surface items (books, trays, boxes) also
        # benefit from a VLM 90°-rotation visual check — the GLB's default yaw
        # may put the long axis in the wrong direction.
        _RECT_ANY_ORIENT_TYPES = (
            "book", "books", "tray", "laptop", "remote",
            "magazine", "journal", "notebook", "folder",
        )
        _is_screen_item = any(kw in phrase.lower() for kw in _SCREEN_ORIENT_TYPES)
        _is_rect_any_item = (
            facing == "any"
            and placement_type == "on_surface"
            and any(kw in phrase.lower() for kw in _RECT_ANY_ORIENT_TYPES)
        )
        # Screens with a reliable toward_dir should NOT get VLM orient
        # corrections — we've seen the VLM flip back and forth between
        # flip_180 / rotate_90 each run because it can't tell whether the
        # render shows the screen's back (correct, screen facing the user at
        # the chair away from the render camera) or the wrong side.  Both
        # failure modes — leaving the monitor stuck slightly off, or
        # repeatedly rotating it into fresh wrong states — are bad.  Trust
        # the geometric rotation from toward_dir and skip the VLM check for
        # screens.  Lamps and non-screen directional items still get checked.
        _screen_with_toward = _is_screen_item and toward_dir is not None
        _should_check_orient = (
            crop_path is not None
            and not _screen_with_toward
            and (
                (facing == "toward_room"
                 and any(kw in phrase.lower() for kw in _ORIENT_CHECK_TYPES))
                or _is_rect_any_item
            )
        )
        if _should_check_orient:
            try:
                _crop_img = _PILImg.open(str(crop_path)).convert("RGB")
                # ── Step 1: 4-yaw candidate pick ─────────────────────────
                # Render the object at all 4 yaw rotations (0°, 90° CCW,
                # 180°, 90° CW) from its current orientation, stitch into a
                # 2×2 grid, and ask the VLM "which panel best matches the
                # reference?".  One VLM call, no compounding errors.
                _Y_ROT_CCW90 = np.array([
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ], dtype=np.float64)
                _Y_ROT_CW90 = np.array([
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ], dtype=np.float64)
                _yaw_options = [
                    ("A", "0°",      None,         False),  # original
                    ("B", "90° CCW", _Y_ROT_CCW90, True),   # swap w/d
                    ("C", "180°",    _RY180,       False),
                    ("D", "90° CW",  _Y_ROT_CW90,  True),   # swap w/d
                ]
                _orig_rot  = list(entry["rotation_3x3"])
                _orig_size = dict(entry.get("size_m", {}))
                _R_orig    = np.array(_orig_rot, dtype=np.float64)
                _orig_w    = _orig_size.get("width_m")
                _orig_d    = _orig_size.get("depth_m")
                _cand_paths: list[Path] = []
                for _tag, _label, _R_corr, _swap_wd in _yaw_options:
                    if _R_corr is None:
                        entry["rotation_3x3"] = list(_orig_rot)
                    else:
                        entry["rotation_3x3"] = (_R_orig @ _R_corr).tolist()
                    if _swap_wd and _orig_w is not None and _orig_d is not None:
                        entry["size_m"]["width_m"] = _orig_d
                        entry["size_m"]["depth_m"] = _orig_w
                    else:
                        if _orig_w is not None: entry["size_m"]["width_m"] = _orig_w
                        if _orig_d is not None: entry["size_m"]["depth_m"] = _orig_d
                    _cand_path = (placements_dir
                                   / f"render_yawcand_{seg_idx:02d}_{_tag}.png")
                    _do_render(out_dir, placements, _cand_path,
                               inpaint_dir=inpaint_dir)
                    _cand_paths.append(_cand_path)
                # Build labelled 2×2 grid (downsampled for VLM throughput).
                _PANEL_PX = 512
                _panels: list[Image.Image] = []
                from PIL import ImageDraw, ImageFont
                try:
                    _font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        36)
                except Exception:
                    _font = ImageFont.load_default()
                for (_tag, _label, _, _), _p in zip(_yaw_options, _cand_paths):
                    _img = Image.open(_p).convert("RGB")
                    _img = _img.resize((_PANEL_PX, _PANEL_PX))
                    _draw = ImageDraw.Draw(_img)
                    _draw.rectangle([(0, 0), (_PANEL_PX, 36)], fill=(0, 0, 0))
                    _draw.text((6, 4),
                               f"{_tag}: {_label}",
                               fill=(255, 200, 0), font=_font)
                    _panels.append(_img)
                _grid = Image.new("RGB", (_PANEL_PX * 2, _PANEL_PX * 2),
                                  (200, 200, 200))
                _grid.paste(_panels[0], (0,         0))
                _grid.paste(_panels[1], (_PANEL_PX, 0))
                _grid.paste(_panels[2], (0,         _PANEL_PX))
                _grid.paste(_panels[3], (_PANEL_PX, _PANEL_PX))
                _grid_path = (placements_dir
                               / f"render_yawcand_{seg_idx:02d}_grid.png")
                _grid.save(str(_grid_path))
                # Ask VLM to pick the best panel.
                _pick_idx = _vlm_pick_yaw_candidate(_crop_img, _grid, phrase)
                if _pick_idx is None:
                    _pick_idx = 0   # default: keep original
                    print(f"    [yaw_pick] no parseable answer — keeping 0°")
                _picked_tag, _picked_label, _picked_R, _picked_swap = (
                    _yaw_options[_pick_idx])
                # Apply the picked rotation permanently as the new "current"
                # orientation.  Reset size_m accordingly.
                if _picked_R is None:
                    entry["rotation_3x3"] = list(_orig_rot)
                else:
                    entry["rotation_3x3"] = (_R_orig @ _picked_R).tolist()
                if _picked_swap and _orig_w is not None and _orig_d is not None:
                    entry["size_m"]["width_m"] = _orig_d
                    entry["size_m"]["depth_m"] = _orig_w
                    size_m["width_m"], size_m["depth_m"] = _orig_d, _orig_w
                else:
                    if _orig_w is not None: entry["size_m"]["width_m"] = _orig_w
                    if _orig_d is not None: entry["size_m"]["depth_m"] = _orig_d
                    if _orig_w is not None: size_m["width_m"] = _orig_w
                    if _orig_d is not None: size_m["depth_m"] = _orig_d
                # Re-render at the picked orientation and update step_path so
                # the iterative fine-tune below sees this baseline.
                _render_incremental(out_dir, placements, seg_idx,
                                    phrase + "_yaw_picked", placements_dir,
                                    inpaint_dir=inpaint_dir)
                print(f"    [yaw_pick] applied {_picked_tag}={_picked_label} "
                      f"as the orientation baseline")
                # Track this as the starting cumulative for the iterative
                # fine-tune below — important so the same-action / oscillation
                # checks see the right history.
                _yaw_pick_initial_deg = (
                    0.0 if _pick_idx == 0
                    else (90.0 if _pick_idx == 1
                          else (180.0 if _pick_idx == 2 else -90.0))
                )

                # ── Step 2: iterative fine-tune ──────────────────────────
                # If the VLM still says the picked yaw is wrong, allow up to
                # _MAX_ORIENT_ITERS more incremental corrections to fix the
                # remaining error (typically 0 — the 4-way pick usually nails
                # it for axis-aligned objects).
                _MAX_ORIENT_ITERS = 3
                _cumulative_rot_deg = float(_yaw_pick_initial_deg)
                _last_orient_action: "str | None" = None
                _same_action_count_orient = 0
                _visited_rots: list[float] = [_cumulative_rot_deg]
                # Tracks whether the iter loop ended because of a guard
                # (oscillation / same-action repeat / visited-revisit /
                # max-iters), in which case the *current* state was just
                # flagged as wrong by the iterative VLM.  The double-check
                # tournament uses this to bias against picking A=current.
                _loop_exit_via_guard = False


                def _final_yaw_double_check(_reason: str) -> None:
                    """At the END of verify_orient, render the CURRENT state
                    plus the 3 other 90° yaw rotations from current and let
                    the VLM pick the best of 4.  Catches:
                      - "VLM correct" exits where the VLM falsely accepted
                        a 90°-off state (single mono-view check is unreliable).
                      - Guard exits where the current state may not be the
                        best of all 4 yaws.

                    Always runs once at loop end.  Cost: 4 renders + 1 VLM."""
                    nonlocal _cumulative_rot_deg
                    # Snapshot current state (this is "candidate A" — no rot).
                    _curr_R          = list(entry["rotation_3x3"])
                    _curr_w          = entry.get("size_m", {}).get("width_m")
                    _curr_d          = entry.get("size_m", {}).get("depth_m")
                    _curr_size_m_w   = size_m.get("width_m")
                    _curr_size_m_d   = size_m.get("depth_m")
                    _curr_cum        = _cumulative_rot_deg
                    _R_curr_arr      = np.array(_curr_R, dtype=np.float64)
                    # 4 candidates relative to CURRENT.
                    _final_options = [
                        ("A", "current",  None,         False, 0.0),
                        ("B", "+90° CCW", _Y_ROT_CCW90, True,   90.0),
                        ("C", "+180°",    _RY180,       False, 180.0),
                        ("D", "+90° CW",  _Y_ROT_CW90,  True,  -90.0),
                    ]
                    _final_states: list[dict] = []
                    for _tag, _label, _R_corr, _swap_wd, _delta in _final_options:
                        if _R_corr is None:
                            entry["rotation_3x3"] = list(_curr_R)
                            _w_t, _d_t = _curr_w, _curr_d
                        else:
                            entry["rotation_3x3"] = (_R_curr_arr @ _R_corr).tolist()
                            _w_t = _curr_d if _swap_wd else _curr_w
                            _d_t = _curr_w if _swap_wd else _curr_d
                        if _w_t is not None: entry["size_m"]["width_m"] = _w_t
                        if _d_t is not None: entry["size_m"]["depth_m"] = _d_t
                        if _w_t is not None: size_m["width_m"] = _w_t
                        if _d_t is not None: size_m["depth_m"] = _d_t
                        # Use a non-`render_step_` prefix so the GIF
                        # assembly's `render_step_*.png` glob doesn't pick
                        # up these candidates as if they were placement
                        # steps — they're just internal renders for the
                        # VLM to compare and would mislead the viewer into
                        # thinking the lamp rotated through 4 more states
                        # after the iter loop finished.
                        _path = (placements_dir
                                  / f"render_doublecand_{seg_idx:02d}"
                                    f"_{phrase.replace(' ', '_')}"
                                    f"_yaw{_tag}.png")
                        try:
                            _do_render(out_dir, placements, _path,
                                       inpaint_dir=inpaint_dir)
                        except Exception as _re:
                            print(f"    [verify_orient] final-yaw render {_tag} failed: {_re}")
                            continue
                        _final_states.append({
                            "tag":        _tag,
                            "label":      _label,
                            "R":          list(entry["rotation_3x3"]),
                            "w":          _w_t,
                            "d":          _d_t,
                            "size_m_w":   _w_t,
                            "size_m_d":   _d_t,
                            "cum_deg":    _curr_cum + _delta,
                            "render":     str(_path),
                        })
                    # Restore current pending tournament outcome.
                    entry["rotation_3x3"] = list(_curr_R)
                    if _curr_w is not None: entry["size_m"]["width_m"] = _curr_w
                    if _curr_d is not None: entry["size_m"]["depth_m"] = _curr_d
                    if _curr_size_m_w is not None: size_m["width_m"] = _curr_size_m_w
                    if _curr_size_m_d is not None: size_m["depth_m"] = _curr_size_m_d
                    if len(_final_states) < 2:
                        print(f"    [verify_orient] double-check skipped "
                              f"({_reason}; only {len(_final_states)} "
                              f"render(s) on disk)")
                        return
                    # Build 2×2 grid for VLM.
                    try:
                        _PANEL_PX = 512
                        from PIL import Image as _PI, ImageDraw as _PD, ImageFont as _PF
                        try:
                            _font = _PF.truetype(
                                "/usr/share/fonts/truetype/dejavu/"
                                "DejaVuSans-Bold.ttf", 22)
                        except Exception:
                            _font = _PF.load_default()
                        _panels = []
                        for _st in _final_states:
                            _img = _PI.open(_st["render"]).convert("RGB")
                            _img = _img.resize((_PANEL_PX, _PANEL_PX))
                            _draw = _PD.Draw(_img)
                            _draw.rectangle([(0, 0), (_PANEL_PX, 36)],
                                            fill=(0, 0, 0))
                            _draw.text((6, 4),
                                       f"{_st['tag']}: {_st['label']}",
                                       fill=(255, 200, 0), font=_font)
                            _panels.append(_img)
                        while len(_panels) < 4:
                            _panels.append(_PI.new(
                                "RGB", (_PANEL_PX, _PANEL_PX), (24, 24, 24)))
                        _grid = _PI.new("RGB",
                                        (_PANEL_PX * 2, _PANEL_PX * 2),
                                        (200, 200, 200))
                        _grid.paste(_panels[0], (0,         0))
                        _grid.paste(_panels[1], (_PANEL_PX, 0))
                        _grid.paste(_panels[2], (0,         _PANEL_PX))
                        _grid.paste(_panels[3], (_PANEL_PX, _PANEL_PX))
                        _grid_path = (placements_dir
                                       / f"render_orient_double_check_"
                                         f"{seg_idx:02d}.png")
                        _grid.save(str(_grid_path))
                    except Exception as _ge:
                        print(f"    [verify_orient] double-check grid build "
                              f"failed ({_ge}) — keeping current")
                        return
                    # Pick best panel.  When the iter loop ended via a guard
                    # (current state was just rejected by VLM as wrong), use
                    # a stricter prompt that warns against defaulting to A
                    # — A was already flagged as wrong, so picking it again
                    # without strong evidence just keeps the rejected state.
                    _bias_against_A = "current was just rejected" in _reason
                    if _bias_against_A:
                        _bias_prompt = (
                            f"You are picking the BEST orientation for a "
                            f"\"{phrase}\" in a 3D scene render.\n\n"
                            f"Image 1 (reference): a crop of the \"{phrase}\".\n"
                            f"Image 2 (candidates): a 2x2 grid of the SAME 3D "
                            f"model at four different yaw rotations:\n"
                            f"  A (top-left)     — CURRENT orientation\n"
                            f"  B (top-right)    — rotated 90° CCW from current\n"
                            f"  C (bottom-left)  — rotated 180° from current\n"
                            f"  D (bottom-right) — rotated 90° CW from current\n\n"
                            f"IMPORTANT: An iterative VLM check has just "
                            f"flagged panel A (the current orientation) as "
                            f"WRONG — needing further rotation.  STRONGLY "
                            f"prefer B, C, or D unless A is unambiguously the "
                            f"closest match to the reference (i.e. you would "
                            f"pick A even if you didn't know it was the "
                            f"current state).\n\n"
                            f"Look at distinctive directional features in the "
                            f"reference (lamp head/arm direction, book spine "
                            f"orientation, monitor screen direction).  Pick "
                            f"the panel where those features point in the "
                            f"SAME direction as the reference.\n\n"
                            f"Return ONLY valid JSON, no markdown:\n"
                            f"{{\n"
                            f"  \"best\": \"A\" or \"B\" or \"C\" or \"D\",\n"
                            f"  \"reasoning\": \"one sentence: which feature\"\n"
                            f"}}"
                        )
                        try:
                            _bias_content = [
                                {"type": "text", "text": _bias_prompt},
                                {"type": "image_url",
                                 "image_url": {"url":
                                    f"data:image/png;base64,"
                                    f"{_encode_pil(_crop_img)}"}},
                                {"type": "image_url",
                                 "image_url": {"url":
                                    f"data:image/png;base64,"
                                    f"{_encode_pil(_grid)}"}},
                            ]
                            _bias_result = _vlm_call_json(
                                _bias_content, max_tokens=200)
                        except Exception:
                            _bias_result = None
                        if _bias_result is not None:
                            _pick = str(_bias_result.get("best", "")).upper().strip()
                            if _pick in {"A", "B", "C", "D"}:
                                _pick_idx = {"A": 0, "B": 1, "C": 2, "D": 3}[_pick]
                                print(f"    [yaw_pick] (biased against A) "
                                      f"best={_pick}  "
                                      f"({_bias_result.get('reasoning','')})")
                            else:
                                _pick_idx = None
                        else:
                            _pick_idx = None
                    else:
                        _pick_idx = _vlm_pick_yaw_candidate(_crop_img, _grid, phrase)
                    if _pick_idx is None or _pick_idx >= len(_final_states):
                        print(f"    [verify_orient] double-check ({_reason}): "
                              f"unparseable pick — keeping current")
                        return
                    _winner = _final_states[_pick_idx]
                    print(f"    [verify_orient] double-check ({_reason}): "
                          f"best={_winner['tag']} → {_winner['label']}")
                    # Apply winner.
                    entry["rotation_3x3"] = list(_winner["R"])
                    if _winner["w"] is not None:
                        entry["size_m"]["width_m"] = _winner["w"]
                    if _winner["d"] is not None:
                        entry["size_m"]["depth_m"] = _winner["d"]
                    if _winner["size_m_w"] is not None:
                        size_m["width_m"] = _winner["size_m_w"]
                    if _winner["size_m_d"] is not None:
                        size_m["depth_m"] = _winner["size_m_d"]
                    _cumulative_rot_deg = float(_winner["cum_deg"])
                    if _winner["tag"] != "A":
                        print(f"    [verify_orient] applied {_winner['label']} "
                              f"→ cumulative {_cumulative_rot_deg:+.0f}°")
                        try:
                            _render_incremental(
                                out_dir, placements, seg_idx,
                                phrase + "_double_check_pick",
                                placements_dir, inpaint_dir=inpaint_dir)
                        except Exception:
                            pass
                # ── Hill-climb pairwise yaw refinement ─────────────────────
                # Replaces the old iter loop ("VLM, is this rotation
                # correct?") and the 4-way double-check tournament.
                #
                # At each step: render current state, apply +90° CW (then
                # CCW), render, ASK VLM which is closer to the reference.
                # If the new state wins, advance and try again.  If the
                # current state wins, revert and stop.
                #
                # If CW chain didn't move from baseline, try CCW chain
                # from baseline.  If both directions stay at baseline,
                # baseline is the local maximum — leave it.
                #
                # Pairwise comparisons are more reliable for VLMs than
                # single-view "is this correct?" or 4-way tournaments,
                # and "first state is fine" emerges naturally — the very
                # first compare picks current over the rotated candidate.
                _HC_MAX_STEPS = 3   # 3 × 90° = 270° from baseline

                # Snapshot baseline (post-yaw_pick).
                _baseline_R          = list(entry["rotation_3x3"])
                _baseline_w          = entry.get("size_m", {}).get("width_m")
                _baseline_d          = entry.get("size_m", {}).get("depth_m")
                _baseline_size_m_w   = size_m.get("width_m")
                _baseline_size_m_d   = size_m.get("depth_m")
                _baseline_cum        = _cumulative_rot_deg

                # Render baseline.  Use a non-`render_step_` prefix so the
                # GIF assembly's `render_step_*.png` glob doesn't pick up
                # the hill-climb intermediate renders as placement steps.
                # Only the final accepted state goes through
                # `_render_incremental` (which writes a `render_step_*`
                # frame for the GIF) at the end of this block.
                _baseline_render_path = (
                    placements_dir
                    / f"render_hccand_{seg_idx:02d}"
                      f"_{phrase.replace(' ', '_')}_baseline.png")
                _do_render(out_dir, placements, _baseline_render_path,
                           inpaint_dir=inpaint_dir)
                _curr_render = _PILImg.open(str(_baseline_render_path)).convert("RGB")

                def _apply_state_to_entry(_R, _w, _d, _smw, _smd):
                    entry["rotation_3x3"] = list(_R)
                    if _w is not None: entry["size_m"]["width_m"] = _w
                    if _d is not None: entry["size_m"]["depth_m"] = _d
                    if _smw is not None: size_m["width_m"] = _smw
                    if _smd is not None: size_m["depth_m"] = _smd

                def _hill_climb_chain(_dir_tag: str, _R_step: "np.ndarray",
                                       _delta_per_step: float):
                    """Run one direction.  Returns (final_R, final_w, final_d,
                    final_size_m_w, final_size_m_d, final_cum, final_render,
                    n_steps_advanced)."""
                    nonlocal _curr_render
                    # Start from baseline.
                    _curr_R   = list(_baseline_R)
                    _curr_w   = _baseline_w
                    _curr_d   = _baseline_d
                    _curr_smw = _baseline_size_m_w
                    _curr_smd = _baseline_size_m_d
                    _curr_cum = _baseline_cum
                    _chain_curr_render = (_PILImg.open(
                        str(_baseline_render_path)).convert("RGB"))
                    _advanced = 0
                    for _hc in range(_HC_MAX_STEPS):
                        # Compute next state.
                        _new_R_arr = (np.array(_curr_R, dtype=np.float64)
                                       @ _R_step)
                        _new_R   = _new_R_arr.tolist()
                        _new_w   = _curr_d   # 90° rotation swaps w,d
                        _new_d   = _curr_w
                        _new_smw = _curr_d if _curr_smw is not None else None
                        _new_smd = _curr_w if _curr_smd is not None else None
                        _new_cum = _curr_cum + _delta_per_step
                        # Apply temporarily and render.
                        _apply_state_to_entry(_new_R, _new_w, _new_d,
                                              _new_smw, _new_smd)
                        _new_render_path = (
                            placements_dir
                            / f"render_hccand_{seg_idx:02d}"
                              f"_{phrase.replace(' ', '_')}"
                              f"_{_dir_tag}{_hc + 1}.png")
                        _do_render(out_dir, placements, _new_render_path,
                                   inpaint_dir=inpaint_dir)
                        _new_render = _PILImg.open(
                            str(_new_render_path)).convert("RGB")
                        # Pairwise compare new vs current.
                        _w_pick = _vlm_compare_two_renders(
                            _crop_img, _new_render, _chain_curr_render, phrase)
                        print(f"    [hillclimb] {_dir_tag} step {_hc + 1}: "
                              f"compare new (cum {_new_cum:+.0f}°) vs "
                              f"current (cum {_curr_cum:+.0f}°) → {_w_pick}")
                        if _w_pick == "A":   # new wins
                            _curr_R   = _new_R
                            _curr_w   = _new_w
                            _curr_d   = _new_d
                            _curr_smw = _new_smw
                            _curr_smd = _new_smd
                            _curr_cum = _new_cum
                            _chain_curr_render = _new_render
                            _advanced += 1
                        else:                 # current wins or VLM failed
                            break
                    return (_curr_R, _curr_w, _curr_d, _curr_smw, _curr_smd,
                            _curr_cum, _chain_curr_render, _advanced)

                # Try CW chain.
                (_cw_R, _cw_w, _cw_d, _cw_smw, _cw_smd, _cw_cum,
                 _cw_render, _cw_n) = _hill_climb_chain(
                    "CW", _Y_ROT_CW90, -90.0)

                if _cw_n == 0:
                    # CW didn't advance — try CCW from baseline.
                    print(f"    [hillclimb] CW chain made no progress; "
                          f"trying CCW from baseline")
                    (_ccw_R, _ccw_w, _ccw_d, _ccw_smw, _ccw_smd, _ccw_cum,
                     _ccw_render, _ccw_n) = _hill_climb_chain(
                        "CCW", _Y_ROT_CCW90, 90.0)
                    if _ccw_n > 0:
                        _final_R, _final_w, _final_d = _ccw_R, _ccw_w, _ccw_d
                        _final_smw, _final_smd, _final_cum = (
                            _ccw_smw, _ccw_smd, _ccw_cum)
                    else:
                        # Both chains stayed at baseline — leave it.
                        _final_R, _final_w, _final_d = (
                            _baseline_R, _baseline_w, _baseline_d)
                        _final_smw, _final_smd, _final_cum = (
                            _baseline_size_m_w, _baseline_size_m_d,
                            _baseline_cum)
                        print(f"    [hillclimb] both directions worsened "
                              f"baseline → keeping baseline (cum "
                              f"{_baseline_cum:+.0f}°)")
                else:
                    _final_R, _final_w, _final_d = _cw_R, _cw_w, _cw_d
                    _final_smw, _final_smd, _final_cum = (
                        _cw_smw, _cw_smd, _cw_cum)

                # Apply the chosen final state.
                _apply_state_to_entry(_final_R, _final_w, _final_d,
                                      _final_smw, _final_smd)
                _cumulative_rot_deg = _final_cum
                print(f"    [hillclimb] final cumulative rotation: "
                      f"{_cumulative_rot_deg:+.0f}°")
                # Final per-step render so the GIF reflects the chosen state.
                _render_incremental(out_dir, placements, seg_idx,
                                    phrase + "_hc_final",
                                    placements_dir, inpaint_dir=inpaint_dir)
            except Exception as _oe:
                print(f"  [verify_orient] orientation check failed: {_oe}")

    # ── Post-loop: side-by-side repositioning for against_back pillow groups ──
    # After all pillows are placed, repack them along the sofa width so they
    # touch each other with no large gaps (center-to-center = sum of half-widths).
    for (_ab_fi,), _ab_entries in _ab_groups.items():
        if len(_ab_entries) < 2:
            continue
        try:
            _ab_furn = furn_map.get(_ab_fi)
            if _ab_furn is None or "position_m" not in _ab_furn:
                continue
            _ab_right, _, _ = _world_axes(_ab_furn)

            # Project each pillow center onto the sofa right axis, sort, then repack
            def _proj_onto_right(e: dict) -> float:
                return float(np.dot(np.array(e["position_m"], dtype=np.float64), _ab_right))

            # Prefer the within-group rank hint (encoded L→R colour order from
            # _classify_seg_color) if present — otherwise fall back to current
            # projection.  The hint preserves the intended grey/orange/orange/grey
            # pattern even when multiple pillow bboxes collide to the same spot.
            def _sort_key(e: dict) -> float:
                _rk = e.get("_cx_rank_hint")
                if _rk is not None:
                    return float(_rk)
                return _proj_onto_right(e)

            _have_hints = any(e.get("_cx_rank_hint") is not None for e in _ab_entries)
            _ab_entries.sort(key=_sort_key)
            if _have_hints:
                print(f"  [sidebyside] fi={_ab_fi}: using rank hints for order "
                      f"→ {[e.get('_cx_rank_hint') for e in _ab_entries]}")

            # Compute half-widths along the sofa right axis for each pillow
            # (use the width_m from size_m — the right axis is roughly the sofa width direction)
            _ab_widths = [e["size_m"].get("width_m", 0.4) for e in _ab_entries]

            # Usable half-width of the sofa SEAT (not the full sofa).  Reserve
            # 18% of the sofa width on each side for armrests + some slack so
            # repacked pillows don't overshoot past the seat cushion onto the
            # armrest.  The OUTERMOST pillow's CENTRE must sit at most this
            # far from the sofa's centre, so its edge stays well clear of
            # the armrest's visible top.
            _sofa_full_hw = _ab_furn["size_m"]["width_m"] / 2.0
            _arm_reserve_sbs = _ab_furn["size_m"]["width_m"] * 0.18
            _ab_furn_hw = max(_sofa_full_hw - _arm_reserve_sbs, 0.0)
            print(f"  [sidebyside] fi={_ab_fi}: sofa_hw={_sofa_full_hw:.2f}m, "
                  f"usable_hw={_ab_furn_hw:.2f}m (reserved {_arm_reserve_sbs:.2f}m for armrests)")

            # Compute center positions: start from leftmost, pack to the right
            _ab_start = -sum(_ab_widths) / 2.0  # centered around 0
            _ab_cx = _ab_start + _ab_widths[0] / 2.0
            _new_projs: list[float] = [_ab_cx]
            for _k in range(1, len(_ab_entries)):
                _ab_cx += (_ab_widths[_k - 1] + _ab_widths[_k]) / 2.0
                _new_projs.append(_ab_cx)

            # For each pillow: its OUTER edge = |scaled_pos| + pillow_half_width
            # must stay ≤ _ab_furn_hw.  So scaled_pos ≤ _ab_furn_hw - pillow_half.
            # The pillow's scale factor is therefore:
            #   scale_i = (usable_hw - pillow_half_i) / |orig_pos_i|
            # and the stack must use the MIN across all pillows.
            _shrink = 1.0
            _before_edges = []
            for _k in range(len(_new_projs)):
                _orig_abs = abs(_new_projs[_k])
                _own_half = _ab_widths[_k] / 2.0
                _before_edges.append(_orig_abs + _own_half)
                if _orig_abs > 1e-6:
                    _max_allowed = max(_ab_furn_hw - _own_half, 0.0)
                    _shrink = min(_shrink, _max_allowed / _orig_abs)
            _max_before_edge = max(_before_edges) if _before_edges else 0.0
            if _shrink < 1.0:
                _new_projs = [p * _shrink for p in _new_projs]
                print(f"  [sidebyside] fi={_ab_fi}: outer pillow edge "
                      f"{_max_before_edge:.2f}m exceeded usable seat {_ab_furn_hw:.2f}m "
                      f"— compressed centres by ×{_shrink:.3f} so all edges fit")

            # Clamp each new center so this pillow's OUTER EDGE stays within
            # the usable seat width.  Using a single clamp ±_ab_furn_hw on the
            # centre lets wide pillows extend past the limit — we need a per-
            # pillow clamp that factors in each pillow's own half-width.
            _ab_furn_pos = np.array(_ab_furn["position_m"], dtype=np.float64)
            _ab_furn_proj = float(np.dot(_ab_furn_pos, _ab_right))
            _repack_moved = 0
            for _k, _e in enumerate(_ab_entries):
                _old_proj = _proj_onto_right(_e)
                _own_hw = _ab_widths[_k] / 2.0
                _per_pillow_max = max(_ab_furn_hw - _own_hw, 0.0)
                _new_proj = float(np.clip(_ab_furn_proj + _new_projs[_k],
                                          _ab_furn_proj - _per_pillow_max,
                                          _ab_furn_proj + _per_pillow_max))
                _delta = _new_proj - _old_proj
                if abs(_delta) > 0.01:
                    _old_pos = np.array(_e["position_m"], dtype=np.float64)
                    _e["position_m"] = (_old_pos + _delta * _ab_right).tolist()
                    _repack_moved += 1
            if _repack_moved:
                print(f"  [sidebyside] fi={_ab_fi}: repacked {_repack_moved} pillows to touch")
                _do_render(out_dir, placements, placements_dir / "render_sidebyside.png",
                           inpaint_dir=inpaint_dir)
        except Exception as _sse:
            print(f"  [sidebyside] fi={_ab_fi} failed: {_sse}")

    # ── Post-loop: cross-sofa back-alignment + vertical consistency ──────────
    # When two sofas hold pillows, the placements should look the same:
    # equal distance from the back rest, equal seat height.  Per-mesh probes
    # can leave one sofa's pillows visibly more forward (and at a higher Y)
    # than the other when the back-cushion / seat-Y density detection finds
    # a different stratum on each mesh.  Compute each sofa's (cz_local, y)
    # for its first against_back pillow; the most back-tucked sofa wins and
    # the others are pushed back and shifted vertically to match.
    _per_sofa_ab: list[dict] = []
    for (_ab_fi,), _ab_entries_x in _ab_groups.items():
        if not _ab_entries_x:
            continue
        _ab_furn_x = furn_map.get(_ab_fi)
        if _ab_furn_x is None or "position_m" not in _ab_furn_x:
            continue
        try:
            _x_right, _x_up, _x_fwd = _world_axes(_ab_furn_x)
        except Exception:
            continue
        _f_pos = np.array(_ab_furn_x["position_m"], dtype=np.float64)
        _e0    = _ab_entries_x[0]
        _p0    = np.array(_e0["position_m"], dtype=np.float64)
        _cz    = float(np.dot(_p0 - _f_pos, _x_fwd))
        _y0    = float(_p0[1])
        _per_sofa_ab.append({
            "fi": _ab_fi, "forward": _x_fwd, "pos": _f_pos,
            "cz": _cz, "y": _y0, "entries": _ab_entries_x,
        })

    if len(_per_sofa_ab) >= 2:
        # Cross-sofa alignment is now DISABLED in practice.  Each pillow's
        # placement uses its own sofa's actual mesh extent (via the
        # centroid-back path), so two sofas with different mesh shapes
        # SHOULD place pillows at different cz_local — that's correct, not
        # a bug.  Forcing them to match (the old behaviour) was pulling
        # the well-placed pillow off its seat to track a worse one.
        # We still detect and log very-different placements as a sanity
        # check, but no longer move anything.
        _ref = None
        _plausible = []
        _per_sofa_summary = ", ".join(
            f"sofa[{d['fi']}]:cz={d['cz']:+.3f}" for d in _per_sofa_ab)
        print(f"  [cross_sofa_align] disabled — each pillow tracks its own "
              f"mesh.  Per-sofa cz_local: {_per_sofa_summary}")
        # Tight tolerances: even a 1-2cm cz_local mismatch reads visually
        # as "one pillow tucked into the back, the other floating forward",
        # so equalise aggressively.
        _CZ_TOL = 0.005  # 5mm — push back almost any forward delta
        _Y_TOL  = 0.005  # 5mm
        if _ref is not None:
            print(f"  [cross_sofa_align] reference: sofa[{_ref['fi']}] "
                  f"cz_local={_ref['cz']:+.3f}m, y={_ref['y']:.3f}m "
                  f"(most back-tucked of {len(_plausible)} plausible sofas)")
        _moved_any = False
        for _info in (_plausible if _ref is not None else []):
            if _info["fi"] == _ref["fi"]:
                continue
            _dz = _info["cz"] - _ref["cz"]   # positive ⇒ more forward
            _dy = _info["y"]  - _ref["y"]    # positive ⇒ higher
            _push_back = max(_dz, 0.0) if _dz > _CZ_TOL else 0.0
            _shift_y   = _dy             if abs(_dy) > _Y_TOL else 0.0
            if _push_back == 0.0 and _shift_y == 0.0:
                continue
            print(f"  [cross_sofa_align] sofa[{_info['fi']}] is "
                  f"{_dz*100:+.1f}cm forward, {_dy*100:+.1f}cm higher than "
                  f"reference — pushing back {_push_back*100:.1f}cm, "
                  f"shifting y by {-_shift_y*100:.1f}cm")
            for _e in _info["entries"]:
                _old = np.array(_e["position_m"], dtype=np.float64)
                _new = _old - _info["forward"] * _push_back
                _new[1] -= _shift_y
                _e["position_m"] = _new.tolist()
                _moved_any = True
        if _moved_any:
            _do_render(out_dir, placements,
                       placements_dir / "render_cross_sofa_align.png",
                       inpaint_dir=inpaint_dir)

    # ── Post-loop: bbox-containment for against_back pillows ─────────────────
    # Final geometric safety net: every against_back pillow must lie INSIDE
    # the supporting sofa/chair's actual mesh AABB.  Catches cases where:
    #   (a) the leaning rotation extends the pillow's top past the sofa's
    #       front edge (when the convention-vs-mesh-orientation disagree),
    #   (b) bbox-fit using metadata sd produced an in-range cz_local but the
    #       actual mesh extent is smaller, or
    #   (c) cross_sofa_align nudged a pillow to a position that's now out of
    #       the supporting furniture's footprint.
    # Per-axis pushback in the SOFA'S LOCAL frame (X = width, Z = depth) so
    # the correction is robust to sofa yaw.  Top constraint is intentionally
    # NOT enforced — pillow tops may extend above the back-rest top.
    try:
        _contain_mesh = _get_furn_scene_mesh(out_dir)
    except Exception:
        _contain_mesh = None
    if _contain_mesh is not None:
        _verts_world = np.asarray(_contain_mesh.vertices, dtype=np.float64)
        _moved_contain = 0
        for _entry in placements:
            if _entry.get("placement_type") != "against_back":
                continue
            if _entry.get("_pos_locked"):
                continue
            _seat_idx = _entry.get("furniture_index")
            if _seat_idx is None:
                continue
            _sofa = furn_map.get(_seat_idx)
            if _sofa is None or "position_m" not in _sofa:
                continue
            _sofa_pos = np.array(_sofa["position_m"], dtype=np.float64)
            _sw_metadata = _sofa["size_m"]["width_m"]
            _sd_metadata = _sofa["size_m"]["depth_m"]
            _sh_eff = float(_sofa.get("eff_h",
                                       _sofa["size_m"].get("height_m", 0.85)))
            _R_sofa = np.array(_sofa.get("rotation_3x3", np.eye(3).tolist()),
                                dtype=np.float64)
            # Filter scene-mesh vertices to this sofa's region (generous so
            # we capture the full mesh even if metadata understates extent).
            _region_pad = 0.30
            _in_region = (
                (np.abs(_verts_world[:, 0] - _sofa_pos[0])
                    <= _sw_metadata / 2.0 + _region_pad)
                & (np.abs(_verts_world[:, 2] - _sofa_pos[2])
                    <= _sd_metadata / 2.0 + _region_pad)
                & (_verts_world[:, 1] >= _sofa_pos[1] - 0.10)
                & (_verts_world[:, 1] <= _sofa_pos[1] + _sh_eff + 0.10)
            )
            _sofa_verts = _verts_world[_in_region]
            if len(_sofa_verts) < 100:
                continue
            _meta_xz_min = np.array([-_sw_metadata / 2.0, -_sd_metadata / 2.0])
            _meta_xz_max = np.array([+_sw_metadata / 2.0, +_sd_metadata / 2.0])
            # Strict bbox containment: clamp to METADATA only.  Per user
            # req, the pillow's top-down bbox must be fully inside the
            # furniture it's placed on — the metadata bbox IS the
            # furniture's defined footprint.  Using a union with the
            # mesh AABB (which can extend past metadata on broken
            # generated meshes, e.g. sofa[1]) was letting the pillow sit
            # outside the sofa's bbox in top-down renders.
            _sofa_xz_min = _meta_xz_min
            _sofa_xz_max = _meta_xz_max
            _xz_source = "metadata"
            # Pillow position in sofa local frame.
            _pillow_pos = np.array(_entry["position_m"], dtype=np.float64)
            _p_offset = _pillow_pos - _sofa_pos
            _p_lx = float(_p_offset @ _R_sofa[:, 0])
            _p_lz = float(_p_offset @ _R_sofa[:, 2])
            # Pillow size (world dimensions; against_back rotation aligns
            # the pillow's width with the sofa's width axis, so use width
            # for X-extent and depth for Z-extent).
            _p_size = _entry.get("size_m", {})
            _p_dw   = float(_p_size.get("width_m", 0.5))
            _p_dd   = float(_p_size.get("depth_m", 0.15))
            _p_dh   = float(_p_size.get("height_m", 0.40))
            _px_min, _px_max = _p_lx - _p_dw / 2.0, _p_lx + _p_dw / 2.0
            # Account for the 20° back-lean: along sofa-Z the pillow's
            # projected half-extent is dd*cos(lean)/2 + dh*sin(lean)/2,
            # not just dd/2.  The leaning TOP of the pillow is what
            # actually pokes past the sofa bbox edge; ignoring lean
            # under-clamps by ~6cm and lets the top-down bbox extend
            # past the sofa boundary (the user's strict-containment
            # requirement).
            _LEAN_RAD_BC = np.radians(20.0)
            _ext_z_bc    = (_p_dd / 2.0 * np.cos(_LEAN_RAD_BC)
                            + _p_dh / 2.0 * np.sin(_LEAN_RAD_BC))
            _pz_min = _p_lz - _ext_z_bc
            _pz_max = _p_lz + _ext_z_bc
            # Push pillow inward so its AABB fits inside the sofa AABB.
            # Tiny margin (1 cm) so the pillow's extreme face doesn't sit
            # exactly on the sofa boundary.
            # Per-axis inside margins:
            #   - 5 cm on the SIDES (left/right) and the FRONT — pillow
            #     edge sits 5 cm inside the sofa boundary, looks like
            #     "pillow on the seat" not "pillow balancing on the edge"
            #   - 1 cm on the BACK side — pillow snug against the back
            #     rest, no visible gap to the cushion
            # `backrest_sign_e` tells us which Z side is the back: −1 →
            # back is at sofa_xz_min[1]; +1 → back is at sofa_xz_max[1].
            # Side margin relaxed to 1 cm: with strict bbox containment
            # enforced upstream (snap_drop / footprint_check), the
            # pillow is already inside metadata.  A 5 cm side margin
            # was forcing the pillow back toward metadata centre on
            # broken generated meshes — fighting the mesh-centre placement
            # the user wants.  1 cm is enough to avoid edge-flush
            # rendering artefacts.
            _MARGIN_SIDE = 0.01
            _MARGIN_BACK = 0.01
            _backrest_sign_e = float(_sofa.get("_backrest_sign", -1.0))
            _push_x = 0.0
            if _px_max > _sofa_xz_max[0] - _MARGIN_SIDE:
                _push_x = (_sofa_xz_max[0] - _MARGIN_SIDE) - _px_max
            elif _px_min < _sofa_xz_min[0] + _MARGIN_SIDE:
                _push_x = (_sofa_xz_min[0] + _MARGIN_SIDE) - _px_min
            # For Z: front side gets the larger margin; back side gets the
            # tight margin so the pillow lands flush against the back rest.
            if _backrest_sign_e < 0:
                _z_min_margin = _MARGIN_BACK   # back is at -Z (sofa_xz_min)
                _z_max_margin = _MARGIN_SIDE   # front is at +Z (sofa_xz_max)
            else:
                _z_min_margin = _MARGIN_SIDE
                _z_max_margin = _MARGIN_BACK
            _push_z = 0.0
            if _pz_max > _sofa_xz_max[1] - _z_max_margin:
                _push_z = (_sofa_xz_max[1] - _z_max_margin) - _pz_max
            elif _pz_min < _sofa_xz_min[1] + _z_min_margin:
                _push_z = (_sofa_xz_min[1] + _z_min_margin) - _pz_min
            if abs(_push_x) > 1e-3 or abs(_push_z) > 1e-3:
                _new = (_pillow_pos
                        + _push_x * _R_sofa[:, 0]
                        + _push_z * _R_sofa[:, 2])
                _entry["position_m"] = _new.tolist()
                print(f"  [bbox_contain] sofa[{_seat_idx}] pillow pushed "
                      f"inward by local ({_push_x*100:+.1f}, {_push_z*100:+.1f})cm — "
                      f"sofa AABB ({_xz_source}) X∈[{_sofa_xz_min[0]:+.2f},"
                      f"{_sofa_xz_max[0]:+.2f}] Z∈[{_sofa_xz_min[1]:+.2f},"
                      f"{_sofa_xz_max[1]:+.2f}]")
                _moved_contain += 1
        if _moved_contain:
            _do_render(out_dir, placements,
                       placements_dir / "render_bbox_contain.png",
                       inpaint_dir=inpaint_dir)

    # ── Post-loop: on_surface overlap resolution ─────────────────────────────
    # The collision tracker often gives up ("no free spot — force-placing")
    # when multiple on_surface decorations land on the same furniture, so a
    # lamp can end up sitting partly on top of a monitor.  Sweep pairwise:
    # for each overlapping pair, push the second-placed decoration along the
    # axis (X or Z, in the surface's local frame) with the SMALLER overlap
    # — that's the cheaper separation — clamped so the moved decoration
    # stays on the surface AABB.  If clamping prevents the push, fall back
    # to pushing the other decoration in the opposite direction.  Iterate
    # up to 4 times to clear chained overlaps.
    if _contain_mesh is not None:
        _on_surf_groups: dict[int, list[dict]] = {}
        for _entry in placements:
            if _entry.get("placement_type") != "on_surface":
                continue
            _seat_idx = _entry.get("furniture_index")
            if _seat_idx is None:
                continue
            _on_surf_groups.setdefault(_seat_idx, []).append(_entry)

        _verts_world_os = np.asarray(_contain_mesh.vertices, dtype=np.float64)
        _overlap_moved = 0
        for _surf_idx, _surf_group in _on_surf_groups.items():
            if len(_surf_group) < 2:
                continue
            _surf = furn_map.get(_surf_idx)
            if _surf is None or "position_m" not in _surf:
                continue
            _surf_pos = np.array(_surf["position_m"], dtype=np.float64)
            _surf_w_meta = _surf["size_m"]["width_m"]
            _surf_d_meta = _surf["size_m"]["depth_m"]
            _surf_h_eff  = float(_surf.get("eff_h",
                                           _surf["size_m"].get("height_m", 0.85)))
            _R_surf = np.array(_surf.get("rotation_3x3", np.eye(3).tolist()),
                                dtype=np.float64)
            # Surface local AABB (X,Z) from actual mesh — fall back to
            # metadata if the mesh region is sparse.
            _in_region_os = (
                (np.abs(_verts_world_os[:, 0] - _surf_pos[0])
                    <= _surf_w_meta / 2.0 + 0.30)
                & (np.abs(_verts_world_os[:, 2] - _surf_pos[2])
                    <= _surf_d_meta / 2.0 + 0.30)
                & (_verts_world_os[:, 1] >= _surf_pos[1] - 0.10)
                & (_verts_world_os[:, 1] <= _surf_pos[1] + _surf_h_eff + 0.10)
            )
            _surf_verts_os = _verts_world_os[_in_region_os]
            if len(_surf_verts_os) >= 100:
                _surf_local_os = (_surf_verts_os - _surf_pos) @ _R_surf
                _surf_xz_min_os = _surf_local_os[:, [0, 2]].min(axis=0)
                _surf_xz_max_os = _surf_local_os[:, [0, 2]].max(axis=0)
            else:
                _surf_xz_min_os = np.array([-_surf_w_meta / 2.0,
                                            -_surf_d_meta / 2.0])
                _surf_xz_max_os = np.array([_surf_w_meta / 2.0,
                                            _surf_d_meta / 2.0])

            def _local_rect(_e: dict) -> tuple[float, float, float, float]:
                _pos = np.array(_e["position_m"], dtype=np.float64)
                _off = _pos - _surf_pos
                _cx_l = float(_off @ _R_surf[:, 0])
                _cz_l = float(_off @ _R_surf[:, 2])
                _sz   = _e.get("size_m", {})
                _hw   = float(_sz.get("width_m", 0.30)) / 2.0
                _hd   = float(_sz.get("depth_m", 0.30)) / 2.0
                return _cx_l, _cz_l, _hw, _hd

            _MAX_OL_ITERS = 4
            for _ol_iter in range(_MAX_OL_ITERS):
                _changed = False
                for _i in range(len(_surf_group)):
                    for _j in range(_i + 1, len(_surf_group)):
                        _a = _surf_group[_i]
                        _b = _surf_group[_j]
                        _ax, _az, _ahw, _ahd = _local_rect(_a)
                        _bx, _bz, _bhw, _bhd = _local_rect(_b)
                        _ovx = (_ahw + _bhw) - abs(_bx - _ax)
                        _ovz = (_ahd + _bhd) - abs(_bz - _az)
                        if _ovx <= 0.0 or _ovz <= 0.0:
                            continue   # no overlap
                        # Pick the cheaper push axis (smaller overlap).
                        _eps = 0.005   # 5 mm extra clearance
                        if _ovx <= _ovz:
                            _push_x = (_ovx + _eps) * (1.0 if _bx >= _ax else -1.0)
                            _push_z = 0.0
                        else:
                            _push_x = 0.0
                            _push_z = (_ovz + _eps) * (1.0 if _bz >= _az else -1.0)
                        # Try moving B; clamp to surface AABB so it stays on the table.
                        _new_bx = float(np.clip(_bx + _push_x,
                                                _surf_xz_min_os[0] + _bhw,
                                                _surf_xz_max_os[0] - _bhw))
                        _new_bz = float(np.clip(_bz + _push_z,
                                                _surf_xz_min_os[1] + _bhd,
                                                _surf_xz_max_os[1] - _bhd))
                        _actual_dx = _new_bx - _bx
                        _actual_dz = _new_bz - _bz
                        # If the chosen-axis push on B got fully clamped, try the
                        # OTHER axis on B before giving up.  The smaller-overlap
                        # axis is just a heuristic for the cheapest separation;
                        # when it's blocked, the larger-overlap axis may still
                        # have room and is far better than dumping the entire
                        # overlap onto A (which can shove A off the table edge
                        # and trigger a downstream pos_refine VLM slot move).
                        if (abs(_actual_dx) < 1e-3 and abs(_actual_dz) < 1e-3
                                and (abs(_push_x) > 1e-3 or abs(_push_z) > 1e-3)):
                            if abs(_push_x) > 1e-3:
                                _alt_push_z = (_ovz + _eps) * (1.0 if _bz >= _az else -1.0)
                                _alt_push_x = 0.0
                            else:
                                _alt_push_x = (_ovx + _eps) * (1.0 if _bx >= _ax else -1.0)
                                _alt_push_z = 0.0
                            _new_bx = float(np.clip(_bx + _alt_push_x,
                                                    _surf_xz_min_os[0] + _bhw,
                                                    _surf_xz_max_os[0] - _bhw))
                            _new_bz = float(np.clip(_bz + _alt_push_z,
                                                    _surf_xz_min_os[1] + _bhd,
                                                    _surf_xz_max_os[1] - _bhd))
                            _actual_dx = _new_bx - _bx
                            _actual_dz = _new_bz - _bz
                            if abs(_actual_dx) > 1e-3 or abs(_actual_dz) > 1e-3:
                                # Alt-axis push on B succeeded; record the chosen
                                # axes for the print so the log is honest.
                                _push_x = _alt_push_x
                                _push_z = _alt_push_z
                        # If both axes on B were clamped, push A in the OPPOSITE
                        # direction instead.
                        if (abs(_actual_dx) < 1e-3 and abs(_actual_dz) < 1e-3
                                and (abs(_push_x) > 1e-3 or abs(_push_z) > 1e-3)):
                            _new_ax = float(np.clip(_ax - _push_x,
                                                    _surf_xz_min_os[0] + _ahw,
                                                    _surf_xz_max_os[0] - _ahw))
                            _new_az = float(np.clip(_az - _push_z,
                                                    _surf_xz_min_os[1] + _ahd,
                                                    _surf_xz_max_os[1] - _ahd))
                            _adx = _new_ax - _ax
                            _adz = _new_az - _az
                            if abs(_adx) > 1e-3 or abs(_adz) > 1e-3:
                                _new_pos_a = (np.array(_a["position_m"],
                                                       dtype=np.float64)
                                              + _adx * _R_surf[:, 0]
                                              + _adz * _R_surf[:, 2])
                                _a["position_m"] = _new_pos_a.tolist()
                                print(f"  [overlap_resolve] surf[{_surf_idx}] "
                                      f"'{_a.get('phrase','?')}' pushed "
                                      f"({_adx*100:+.1f},{_adz*100:+.1f})cm "
                                      f"(B was clamped) — clears "
                                      f"'{_b.get('phrase','?')}'")
                                _changed = True
                                _overlap_moved += 1
                        elif abs(_actual_dx) > 1e-3 or abs(_actual_dz) > 1e-3:
                            _new_pos_b = (np.array(_b["position_m"],
                                                   dtype=np.float64)
                                          + _actual_dx * _R_surf[:, 0]
                                          + _actual_dz * _R_surf[:, 2])
                            _b["position_m"] = _new_pos_b.tolist()
                            print(f"  [overlap_resolve] surf[{_surf_idx}] "
                                  f"'{_b.get('phrase','?')}' pushed "
                                  f"({_actual_dx*100:+.1f},{_actual_dz*100:+.1f})cm "
                                  f"to clear '{_a.get('phrase','?')}' "
                                  f"(overlap was {_ovx*100:.0f}×{_ovz*100:.0f}cm)")
                            _changed = True
                            _overlap_moved += 1
                if not _changed:
                    break
        if _overlap_moved:
            _do_render(out_dir, placements,
                       placements_dir / "render_overlap_resolved.png",
                       inpaint_dir=inpaint_dir)

    # ── Post-loop: final orientation refinement on the FULL scene ────────────
    # The per-decoration `verify_orient` step that runs during placement only
    # sees the scene as built so far — later items aren't placed yet — and
    # the 4-yaw picker + 3-iter loop sometimes converges to a yaw that the
    # VLM, given the full scene, would still call wrong.  Run another
    # verify_orient pass on the COMPLETED scene: render it, walk every
    # directional decoration, and keep applying VLM-suggested rotations
    # until either the VLM stops complaining or we hit a per-item iteration
    # cap (with the same visited-state guard as the placement-time loop, so
    # an oscillating VLM doesn't spin forever).
    # Screens (monitor / TV / laptop) are deliberately EXCLUDED here.
    # Their orientation is already pinned during placement by the
    # `_detect_decoration_front` 4-view picker + mirror check + the
    # per-decoration `verify_orient` loop.  Re-running verify_orient on
    # the completed scene tends to oscillate (the VLM keeps suggesting
    # alternating flip_180 ↔ rotate_90_cw because most of a screen looks
    # symmetric from many angles), and successive passes can stack to a
    # net 180° rotation — exactly the failure mode that put the monitor
    # facing backwards.  Other directional items (lamps, books) still
    # benefit from the final scene-context check, since their distinctive
    # silhouette (lamp arm, book spine) makes the VLM's judgement more
    # reliable.
    _REFINE_KEYWORDS = ("lamp", "keyboard", "book", "vase", "plant",
                        "tray", "remote")
    if _ref_photo is not None:
        # Disabled: the placement-time `yaw_pick` + `verify_orient`
        # already settles each directional item's orientation.
        # Re-running orientation checks here was layering additional
        # 90° rotations on top of the already-correct placement state,
        # then the cross-pass guard would stop at whatever orientation
        # was visited last — not necessarily the right one.  User
        # observation: the second-to-last gif frame (post-placement,
        # pre-final_refine) is correct; the last frame (post-
        # final_refine) is wrong.  Trust placement-time orientation.
        _MAX_REFINE_PASSES = 0
        _MAX_PER_ITEM_ITERS = 2
        _Y_ROT_CCW90 = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float64)
        _Y_ROT_CW90 = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float64)
        _action_to_rot = {
            "flip_180":      (_RY180,       180.0),
            "rotate_90_ccw": (_Y_ROT_CCW90,  90.0),
            "rotate_90_cw":  (_Y_ROT_CW90,  -90.0),
        }
        # Seg-index → crop_file lookup (built once)
        _crop_by_seg = {
            int(s.get("seg_index", -1)): s.get("crop_file")
            for s in seg_data.get("segments", [])
            if s.get("crop_file")
        }
        # Cross-pass entry rotation tracking — captures every
        # cumulative angle the entry has been at since the start of
        # final_refine.  Without this, pass 1's verify_orient can
        # rotate the entry back to a previously-visited orientation
        # (undoing pass 0's correction).  Used to detect cross-pass
        # revisits and stop them.  Per-pass `_visited` is no
        # longer sufficient because each pass resets it.
        _entry_global_visited: dict[int, list[float]] = {}
        _entry_global_cum: dict[int, float] = {}
        for _refine_pass in range(_MAX_REFINE_PASSES):
            _refine_render_path = (placements_dir
                                    / f"render_refine_pass{_refine_pass}.png")
            _do_render(out_dir, placements, _refine_render_path,
                       inpaint_dir=inpaint_dir)
            _refine_render = _PILImg.open(str(_refine_render_path)).convert("RGB")
            _changed_in_pass = False
            for _e in placements:
                _phrase = (_e.get("phrase") or "").lower()
                if not any(kw in _phrase for kw in _REFINE_KEYWORDS):
                    continue
                _seg_idx = int(_e.get("seg_index", -1))
                _crop_file = _crop_by_seg.get(_seg_idx)
                if not _crop_file:
                    continue
                _crop_path_r = (out_dir / "decorations" / "segmented"
                                / _crop_file)
                if not _crop_path_r.exists():
                    continue
                _crop_img_r = _PILImg.open(str(_crop_path_r)).convert("RGB")
                # Per-item iteration loop with visited-state guard.
                _cum_deg = 0.0
                _last_action: "str | None" = None
                _same_action_count_final = 0
                _visited = [0.0]
                # Initialize cross-pass tracking on first pass.
                if _seg_idx not in _entry_global_visited:
                    _entry_global_visited[_seg_idx] = [0.0]
                    _entry_global_cum[_seg_idx] = 0.0
                for _it in range(_MAX_PER_ITEM_ITERS):
                    # Re-render after applying this item's previous correction
                    # (only re-render if the current item moved this iter; for
                    # the first iter we use the pass render).
                    if _it == 0:
                        _orient_render = _refine_render
                    else:
                        _per_item_render = (placements_dir
                                              / f"render_refine_pass{_refine_pass}"
                                                f"_seg{_seg_idx:02d}_iter{_it}.png")
                        _do_render(out_dir, placements, _per_item_render,
                                   inpaint_dir=inpaint_dir)
                        _orient_render = _PILImg.open(
                            str(_per_item_render)).convert("RGB")
                    _action = _vlm_verify_orientation(
                        _crop_img_r, _orient_render, _phrase)
                    if not _action or _action == "correct":
                        if _it > 0:
                            print(f"    [final_refine] '{_phrase}' converged "
                                  f"(cumulative {_cum_deg:+.0f}°)")
                        break
                    if _action not in _action_to_rot:
                        break
                    # Same-direction repeat: directional items (lamp,
                    # monitor, etc.) get up to 2 consecutive
                    # same-action calls so two 90° steps can chain into
                    # a 180° net correction.  Symmetric / rectangular
                    # items (books, trays) stop on the first repeat —
                    # the VLM is unreliable on their orientation.
                    _DIRECTIONAL_KW_FR = (
                        "lamp", "monitor", "screen", "tv", "television",
                        "laptop", "keyboard",
                    )
                    _is_directional_fr = any(
                        kw in _phrase for kw in _DIRECTIONAL_KW_FR)
                    _SAME_ACTION_LIMIT_FR = 2 if _is_directional_fr else 1
                    if _it > 0 and _last_action == _action:
                        _same_action_count_final += 1
                    else:
                        _same_action_count_final = 0
                    if _same_action_count_final >= _SAME_ACTION_LIMIT_FR:
                        print(f"    [final_refine] '{_phrase}' iter {_it}: "
                              f"{_action} repeated "
                              f"{_same_action_count_final + 1}× in a row "
                              f"(limit {_SAME_ACTION_LIMIT_FR} for "
                              f"{'directional' if _is_directional_fr else 'rectangular/symmetric'}) "
                              f"— stopping at {_cum_deg:+.0f}°")
                        break
                    _R_corr, _delta = _action_to_rot[_action]
                    _proposed = _cum_deg + _delta
                    _norm_p = ((_proposed + 180.0) % 360.0) - 180.0
                    if any(abs(_norm_p - v) < 1.0 or
                           abs(((_norm_p - v + 180.0) % 360.0) - 180.0) < 1.0
                           for v in _visited):
                        print(f"    [final_refine] '{_phrase}' iter {_it}: "
                              f"{_action} would revisit {_norm_p:+.0f}° "
                              f"(per-pass) — stopping at {_cum_deg:+.0f}°")
                        break
                    # Cross-pass revisit guard: would this rotation
                    # land the entry at a cumulative orientation it's
                    # already been at since final_refine started?  If
                    # so, this is pass-N undoing an earlier pass's
                    # correction (lamp going +180° in pass 0, then
                    # pass 1's two -90°s bringing it back to 0°).
                    _proposed_global = (_entry_global_cum[_seg_idx]
                                         + _delta)
                    _norm_global = ((_proposed_global + 180.0)
                                     % 360.0) - 180.0
                    if any(abs(_norm_global - v) < 1.0 or
                           abs(((_norm_global - v + 180.0) % 360.0)
                               - 180.0) < 1.0
                           for v in _entry_global_visited[_seg_idx]):
                        print(f"    [final_refine] '{_phrase}' iter "
                              f"{_it}: {_action} would revisit cross-"
                              f"pass orientation {_norm_global:+.0f}° "
                              f"(history "
                              f"{_entry_global_visited[_seg_idx]}) — "
                              f"this would undo a previous pass's "
                              f"correction; stopping at "
                              f"{_cum_deg:+.0f}°")
                        break
                    _R_now = np.array(_e["rotation_3x3"], dtype=np.float64)
                    _e["rotation_3x3"] = (_R_now @ _R_corr).tolist()
                    if _action.startswith("rotate_90"):
                        _sz = _e.get("size_m", {})
                        _w_t = _sz.get("width_m")
                        _d_t = _sz.get("depth_m")
                        if _w_t is not None and _d_t is not None:
                            _sz["width_m"], _sz["depth_m"] = _d_t, _w_t
                    _cum_deg = _proposed
                    _entry_global_cum[_seg_idx] = _proposed_global
                    _entry_global_visited[_seg_idx].append(_norm_global)
                    _visited.append(_norm_p)
                    _last_action = _action
                    _changed_in_pass = True
                    print(f"    [final_refine] pass {_refine_pass} "
                          f"'{_phrase}' iter {_it}: applied {_action} "
                          f"(cumulative {_cum_deg:+.0f}° pass, "
                          f"{_entry_global_cum[_seg_idx]:+.0f}° global)")
            if not _changed_in_pass:
                print(f"  [final_refine] pass {_refine_pass}: no corrections "
                      f"needed — done")
                break
        # Final render after refinement
        _do_render(out_dir, placements,
                   placements_dir / "render_final_refine.png",
                   inpaint_dir=inpaint_dir)

    # ── Post-loop: VLM-guided semantic position refinement ──────────────────
    # The geometric placement uses photo-bbox cx hints + back-face detection
    # + silhouette scaling but can still leave the object in the WRONG SLOT
    # of its supporting furniture: e.g. lamp on the right of the desk
    # instead of the left, pillow at the right armrest instead of centred.
    # The pixel-offset VLM call we used to run here oscillated (it kept
    # contradicting itself); using DISCRETE slots in the furniture's own
    # local frame avoids that — the VLM picks one of nine slots
    # (left/center/right × back/middle/front) and we snap the object
    # there.  Per-item visited-slot guard prevents back-and-forth.
    if _ref_photo is not None:
        _POS_REFINE_KEYWORDS = ("pillow", "cushion", "lamp", "book",
                                "monitor", "keyboard", "vase", "plant",
                                "tray", "remote", "laptop", "screen",
                                "bowl", "candle", "frame", "clock")
        _POS_REFINE_PASSES = 2
        _POS_REFINE_TOL_M = 0.05  # 5 cm — already in slot if within this
        _crop_by_seg_pr = {
            int(s.get("seg_index", -1)): s.get("crop_file")
            for s in seg_data.get("segments", [])
            if s.get("crop_file")
        }

        # Camera-aligned horizontal axes — used to convert camera-frame
        # slots ("left", "front") into world offsets.  +cam_right_w is the
        # direction the camera sees as image-right; +cam_into_w is the
        # direction from the camera into the scene (away from the viewer).
        # "front" of a surface (closer to viewer) is the −cam_into_w side.
        try:
            _cp_pr  = np.array(camera["position_m"], dtype=np.float64)
            _la_pr  = np.array(camera["look_at_m"],  dtype=np.float64)
            _into_h = (_la_pr - _cp_pr).astype(np.float64)
            _into_h[1] = 0.0
            _n_into = float(np.linalg.norm(_into_h))
            if _n_into > 1e-9:
                _cam_into_w = _into_h / _n_into
                _cam_right_w = np.cross(np.array([0., 1., 0.]),
                                         _cam_into_w)
                _n_r = float(np.linalg.norm(_cam_right_w))
                if _n_r > 1e-9:
                    _cam_right_w = _cam_right_w / _n_r
                else:
                    _cam_right_w = np.array([1., 0., 0.])
            else:
                _cam_right_w = np.array([1., 0., 0.])
                _cam_into_w  = np.array([0., 0., 1.])
        except Exception:
            _cam_right_w = np.array([1., 0., 0.])
            _cam_into_w  = np.array([0., 0., 1.])

        def _pr_slot_to_local_xz(side: str, depth: str,
                                  entry: dict, furn: dict
                                  ) -> tuple[float, float]:
            """Map a discrete CAMERA-FRAME slot to (cx_local, cz_local)
            inside furn's local frame.

            The slot directions ("left", "right", "front", "back") are
            interpreted in the CAMERA / IMAGE frame — the same frame the
            VLM sees when comparing reference photo and render.  We
            compute the target offset from the furniture centre as a
            world vector along (camera_right, camera_into) basis, then
            project that vector onto the furniture's local axes for
            storage (since position_m is built from local cx/cz).

            The available offset magnitude is the FURNITURE'S extent
            projected onto the camera axes — for a yawed furniture
            (e.g. sofa[1] at 45°), local +X is not camera-right, so the
            "left" extent in camera frame combines width AND depth of
            the metadata bbox.

            For ``against_back`` items the cz_local is later overridden
            by the caller to preserve the visible-back-rest contact;
            here we still compute a sensible fallback so that pure
            "side"-only refinements work too.
            """
            sz = entry.get("size_m", {})
            dw = float(sz.get("width_m", 0.30))
            dd = float(sz.get("depth_m", 0.30))
            sw = furn["size_m"]["width_m"]
            sd = furn["size_m"]["depth_m"]
            R_f = np.array(furn.get("rotation_3x3", np.eye(3).tolist()),
                           dtype=np.float64)
            R0 = R_f[:, 0]   # local +X in world
            R2 = R_f[:, 2]   # local +Z in world
            # Furniture half-extent along each camera axis.  A rotated
            # rectangle's extent along a world axis u is
            # sw/2 * |R0·u| + sd/2 * |R2·u|.
            half_x_cam = (abs(float(R0 @ _cam_right_w)) * sw / 2.0
                          + abs(float(R2 @ _cam_right_w)) * sd / 2.0)
            half_z_cam = (abs(float(R0 @ _cam_into_w))  * sw / 2.0
                          + abs(float(R2 @ _cam_into_w))  * sd / 2.0)
            # Decor half-extent along each camera axis — use the
            # decor's CURRENT rotation (post orientation refinement) so
            # that an elongated object whose long axis aligns with the
            # camera's depth (e.g. a lamp arm pointing into the scene)
            # contributes its NARROW side to camera-X.  Using the
            # conservative max(dw, dd)/2 instead made usable_x ≈ 0 for
            # tall/long lamps on small tables — the lamp could not
            # move camera-left at all even when there was visible room.
            R_d = np.array(entry.get("rotation_3x3", np.eye(3).tolist()),
                           dtype=np.float64)
            D0 = R_d[:, 0]
            D2 = R_d[:, 2]
            decor_half_x_cam = (abs(float(D0 @ _cam_right_w)) * dw / 2.0
                                 + abs(float(D2 @ _cam_right_w)) * dd / 2.0)
            decor_half_z_cam = (abs(float(D0 @ _cam_into_w))  * dw / 2.0
                                 + abs(float(D2 @ _cam_into_w))  * dd / 2.0)
            ptype = entry.get("placement_type", "on_surface")
            # Tighter side margin (2 cm) lets the slot land closer to
            # the visible edge in the image — the user wants "lefter
            # enough", and bbox_contain still catches actual overhang.
            margin_side  = 0.02
            margin_front = 0.05
            margin_back  = 0.01 if ptype == "against_back" else 0.05
            usable_x = max(half_x_cam - margin_side
                           - decor_half_x_cam, 0.0)
            usable_back  = max(half_z_cam - margin_back
                                - decor_half_z_cam, 0.0)
            usable_front = max(half_z_cam - margin_front
                                - decor_half_z_cam, 0.0)
            side_factor  = {"left": -1.0, "center": 0.0,
                            "right": +1.0}.get(side, 0.0)
            depth_factor = {"front": -1.0, "middle": 0.0,
                            "back": +1.0}.get(depth, 0.0)
            side_offset_w = side_factor * usable_x
            if depth_factor > 0:
                depth_offset_w = depth_factor * usable_back
            else:
                depth_offset_w = depth_factor * usable_front
            world_offset = (side_offset_w * _cam_right_w
                             + depth_offset_w * _cam_into_w)
            # Project the world offset onto the furniture's local axes.
            cx = float(world_offset @ R0)
            cz = float(world_offset @ R2)
            # Clamp to the metadata bbox so we never propose a slot that
            # would put the centre off the canonical surface.  (The
            # raycast-miss revert later still catches off-mesh cases.)
            cx = float(np.clip(cx, -sw / 2.0 + dw / 2.0,
                                +sw / 2.0 - dw / 2.0))
            cz = float(np.clip(cz, -sd / 2.0 + dd / 2.0,
                                +sd / 2.0 - dd / 2.0))
            return cx, cz

        # Per-item visited-slot tracker — stop oscillation across passes.
        _pr_visited: dict[int, set[tuple[str, str]]] = {}
        for _pr_pass in range(_POS_REFINE_PASSES):
            _pr_render_path = (placements_dir
                                / f"render_pos_refine_pass{_pr_pass}.png")
            _do_render(out_dir, placements, _pr_render_path,
                       inpaint_dir=inpaint_dir)
            _pr_render = _PILImg.open(str(_pr_render_path)).convert("RGB")
            _pr_moved = 0
            for _e in placements:
                if _e.get("_pos_locked"):
                    continue
                _phrase = (_e.get("phrase") or "").lower()
                if not any(kw in _phrase for kw in _POS_REFINE_KEYWORDS):
                    continue
                _ptype = _e.get("placement_type", "on_surface")
                if _ptype not in ("against_back", "on_surface"):
                    continue
                _fi = _e.get("furniture_index")
                _furn = furn_map.get(_fi)
                if _furn is None or "position_m" not in _furn:
                    continue
                # Skip pos_refine when silhouette / back-proj already
                # gave a usable position — meaning the decor's current
                # footprint sits inside the furniture's footprint
                # (top-down, no extra margin).  Per the user's rule:
                # only "force-place" / clamp when the bottom is OUT of
                # the furniture; if it's already on the furniture, the
                # silhouette position is the gold standard and slot
                # moves would only drag it away from the reference
                # photo's intent.
                # Per-axis treatment:
                #   - against_back items only refine SIDE (cx), since
                #     cz is preserved by the centroid_back placement.
                #     We therefore only require cx-axis containment for
                #     the skip; cz being past metadata (sofa[1]'s
                #     visible-mesh-back placement) is intentional and
                #     should not pull the pillow off-centre.
                #   - on_surface items must have BOTH axes inside the
                #     metadata footprint to skip.
                _R_furn_skip = np.array(_furn.get("rotation_3x3",
                                                    np.eye(3).tolist()),
                                          dtype=np.float64)
                _R_decor_skip = np.array(_e.get("rotation_3x3",
                                                  np.eye(3).tolist()),
                                          dtype=np.float64)
                _sw_skip = float(_furn["size_m"]["width_m"])
                _sd_skip = float(_furn["size_m"]["depth_m"])
                _dw_skip = float(_e["size_m"].get("width_m", 0.30))
                _dd_skip = float(_e["size_m"].get("depth_m", 0.30))
                _ext_x_skip = (
                    abs(float(_R_decor_skip[:, 0]
                              @ _R_furn_skip[:, 0])) * _dw_skip / 2.0
                    + abs(float(_R_decor_skip[:, 2]
                                 @ _R_furn_skip[:, 0])) * _dd_skip / 2.0)
                _ext_z_skip = (
                    abs(float(_R_decor_skip[:, 0]
                              @ _R_furn_skip[:, 2])) * _dw_skip / 2.0
                    + abs(float(_R_decor_skip[:, 2]
                                 @ _R_furn_skip[:, 2])) * _dd_skip / 2.0)
                _f_pos_skip = np.array(_furn["position_m"],
                                        dtype=np.float64)
                _p_pos_skip = np.array(_e["position_m"],
                                        dtype=np.float64)
                _off_skip = _p_pos_skip - _f_pos_skip
                _cx_skip = float(_off_skip @ _R_furn_skip[:, 0])
                _cz_skip = float(_off_skip @ _R_furn_skip[:, 2])
                # Footprint corners within metadata bbox, with a small
                # tolerance so a 4-5cm corner overhang doesn't kick a
                # well-placed item back into pos_refine.  Without this,
                # a monitor whose silhouette-derived position has its
                # corner protruding 4cm past the metadata front edge
                # triggers a slot move to (center, back), which
                # cascades into overlap_resolve pushing nearby items
                # (e.g. the lamp) by 30+cm to clear the path.  bbox_contain
                # / final_clamp still handles real overhang cases.
                _SKIP_OVERHANG_TOL = 0.05
                _cx_inside = (
                    abs(_cx_skip) + _ext_x_skip
                        <= _sw_skip / 2.0 + _SKIP_OVERHANG_TOL
                )
                _cz_inside = (
                    abs(_cz_skip) + _ext_z_skip
                        <= _sd_skip / 2.0 + _SKIP_OVERHANG_TOL
                )
                if _ptype == "against_back":
                    _all_inside = _cx_inside
                    _skip_axes_str = "cx"
                else:
                    _all_inside = _cx_inside and _cz_inside
                    _skip_axes_str = "cx+cz"
                if _all_inside:
                    print(f"    [pos_refine] '{_phrase}' on "
                          f"{_furn.get('type','furniture')}[{_fi}]: "
                          f"silhouette/back-proj position already on "
                          f"furniture ({_skip_axes_str} corners "
                          f"inside metadata) — skipping VLM slot move "
                          f"(cx={_cx_skip:+.3f}±{_ext_x_skip:.3f} ≤ "
                          f"{_sw_skip/2:.3f}; "
                          f"cz={_cz_skip:+.3f}±{_ext_z_skip:.3f} ≤ "
                          f"{_sd_skip/2:.3f})")
                    continue
                _seg_idx = int(_e.get("seg_index", -1))
                _crop_file = _crop_by_seg_pr.get(_seg_idx)
                if not _crop_file:
                    continue
                _crop_path_pr = (out_dir / "decorations" / "segmented"
                                 / _crop_file)
                if not _crop_path_pr.exists():
                    continue
                try:
                    _crop_img_pr = _PILImg.open(
                        str(_crop_path_pr)).convert("RGB")
                except Exception as _ce:
                    print(f"    [pos_refine] '{_phrase}': failed to open "
                          f"crop {_crop_file}: {_ce}")
                    continue
                _ftype_pr = (_furn.get("type") or "furniture").lower()
                _result_pr = _vlm_pos_refine_semantic(
                    _crop_img_pr, _pr_render, _phrase, _ftype_pr)
                if _result_pr is None:
                    continue
                if _result_pr.get("correct"):
                    print(f"    [pos_refine] '{_phrase}' on "
                          f"{_ftype_pr}[{_fi}]: position correct. "
                          f"{str(_result_pr.get('reasoning', ''))[:80]}")
                    continue
                _side  = str(_result_pr.get("side", "")).strip().lower()
                _depth = str(_result_pr.get("depth", "")).strip().lower()
                if (_side not in {"left", "center", "right"}
                        or _depth not in {"back", "middle", "front"}):
                    print(f"    [pos_refine] '{_phrase}': VLM returned "
                          f"invalid slot ({_side},{_depth}) — skipping")
                    continue
                _slot_pr = (_side, _depth)
                _seen = _pr_visited.setdefault(_seg_idx, set())
                if _slot_pr in _seen:
                    print(f"    [pos_refine] '{_phrase}': VLM wants "
                          f"{_slot_pr} but already visited that slot — "
                          f"stopping to avoid oscillation")
                    continue
                _cx_new, _cz_new = _pr_slot_to_local_xz(
                    _side, _depth, _e, _furn)
                # Skip if the current placement is already within tolerance
                # of the requested slot — VLM may have flagged a small
                # visual mismatch the geometric move can't improve.
                _R_f_pr = np.array(_furn.get("rotation_3x3",
                                              np.eye(3).tolist()),
                                    dtype=np.float64)
                _f_pos_pr = np.array(_furn["position_m"],
                                      dtype=np.float64)
                _cur_pos_pr = np.array(_e["position_m"],
                                        dtype=np.float64)
                _off_pr = _cur_pos_pr - _f_pos_pr
                _cur_cx_pr = float(_off_pr @ _R_f_pr[:, 0])
                _cur_cz_pr = float(_off_pr @ _R_f_pr[:, 2])
                # For against_back items (pillows), only refine SIDE.
                # The depth (cz_local) was already pinned during placement
                # to the visible back-rest by centroid_back / mesh-aware
                # logic, which can sit BEHIND the metadata bbox on sofas
                # whose mesh extends past metadata.  Re-applying a "back"
                # slot in metadata coords would pull the pillow forward
                # of the visible back-rest, defeating the original fix.
                if _ptype == "against_back":
                    _cz_target = _cur_cz_pr
                else:
                    _cz_target = _cz_new
                if (abs(_cur_cx_pr - _cx_new) < _POS_REFINE_TOL_M
                        and abs(_cur_cz_pr - _cz_target) < _POS_REFINE_TOL_M):
                    print(f"    [pos_refine] '{_phrase}' on "
                          f"{_ftype_pr}[{_fi}]: already at slot "
                          f"({_side},{_depth}) within "
                          f"{_POS_REFINE_TOL_M*100:.0f}cm — skipping")
                    _seen.add(_slot_pr)
                    continue
                # Cap how far a slot move can shift the item.  When two
                # items share a surface (monitor + lamp on a coffee table)
                # the VLM can swap their sides — e.g. send monitor from
                # cx=+0.25 to cx=-0.28 (Δ=-0.53m) and lamp from cx=-0.24
                # to cx=+0.39 (Δ=+0.63m) — based on a one-shot reading of
                # the reference photo.  Any move >~30cm is almost certainly
                # a swap rather than a refinement; skip it.
                _sw_pr_cap = float(_furn.get("size_m", {}).get("width_m", 1.0))
                _sd_pr_cap = float(_furn.get("size_m", {}).get("depth_m", 0.5))
                _MAX_SLOT_MOVE_M = max(0.30,
                                        0.5 * min(_sw_pr_cap, _sd_pr_cap) / 2.0)
                _move_dx = _cx_new - _cur_cx_pr
                _move_dz = _cz_target - _cur_cz_pr
                _move_mag = (_move_dx ** 2 + _move_dz ** 2) ** 0.5
                if _move_mag > _MAX_SLOT_MOVE_M:
                    print(f"    [pos_refine] '{_phrase}' on "
                          f"{_ftype_pr}[{_fi}]: VLM slot ({_side},{_depth}) "
                          f"would move ({_move_dx*100:+.1f},"
                          f"{_move_dz*100:+.1f})cm = {_move_mag*100:.1f}cm "
                          f"> cap {_MAX_SLOT_MOVE_M*100:.0f}cm — likely a "
                          f"side-swap with another item, skipping")
                    _seen.add(_slot_pr)
                    continue
                # Conflict-with-other-item check: would the proposed slot put
                # this item on top of another existing item on the same
                # surface?  When the VLM hallucinates a swap (lamp's slot is
                # the monitor's current spot), applying the move would just
                # shove the items into each other and force overlap_resolve
                # to clean up — usually with a worse result than just leaving
                # them where they were.  Skip rather than create the overlap.
                _OVERLAP_TOL_M2 = 0.005   # 50 cm² — small overlaps OK
                _e_w = float(_e.get("size_m", {}).get("width_m", 0.30))
                _e_d = float(_e.get("size_m", {}).get("depth_m", 0.30))
                _e_hw, _e_hd = _e_w / 2.0, _e_d / 2.0
                _conflict_other = None
                for _other in placements:
                    if _other is _e:
                        continue
                    if _other.get("furniture_index") != _fi:
                        continue
                    if _other.get("placement_type") != "on_surface":
                        continue
                    _o_pos = np.array(_other["position_m"], dtype=np.float64)
                    _o_off = _o_pos - _f_pos_pr
                    _o_cx  = float(_o_off @ _R_f_pr[:, 0])
                    _o_cz  = float(_o_off @ _R_f_pr[:, 2])
                    _o_w   = float(_other.get("size_m", {}).get("width_m", 0.30))
                    _o_d   = float(_other.get("size_m", {}).get("depth_m", 0.30))
                    _o_hw, _o_hd = _o_w / 2.0, _o_d / 2.0
                    _ovx = (_e_hw + _o_hw) - abs(_cx_new - _o_cx)
                    _ovz = (_e_hd + _o_hd) - abs(_cz_target - _o_cz)
                    if _ovx > 0 and _ovz > 0 and _ovx * _ovz > _OVERLAP_TOL_M2:
                        _conflict_other = _other
                        _ov_area = _ovx * _ovz
                        break
                if _conflict_other is not None:
                    print(f"    [pos_refine] '{_phrase}' on "
                          f"{_ftype_pr}[{_fi}]: VLM slot ({_side},{_depth}) "
                          f"target overlaps existing "
                          f"'{_conflict_other.get('phrase','?')}' "
                          f"by {_ov_area*1e4:.0f}cm² — would force them on top "
                          f"of each other, skipping (likely VLM-hallucinated "
                          f"swap)")
                    _seen.add(_slot_pr)
                    continue
                _new_pos_pr = _f_pos_pr.copy()
                _new_pos_pr += _cx_new * _R_f_pr[:, 0]
                _new_pos_pr += _cz_target * _R_f_pr[:, 2]
                # Y: keep current Y for against_back (seat height won't
                # change with cx); for on_surface, raycast at the new
                # (x,z) so we sit on the actual table top there.  If the
                # raycast misses (returns the nominal Y unchanged) or
                # deviates by >10cm from the current Y, the new (x,z)
                # is OFF the table — revert the move rather than leave
                # the object floating in mid-air.
                if _ptype == "on_surface":
                    _new_y_pr = _actual_surface_height(
                        out_dir, float(_new_pos_pr[0]),
                        float(_new_pos_pr[2]),
                        float(_cur_pos_pr[1]))
                    if abs(_new_y_pr - float(_cur_pos_pr[1])) > 0.10:
                        print(f"    [pos_refine] '{_phrase}' on "
                              f"{_ftype_pr}[{_fi}]: target slot "
                              f"({_side},{_depth}) is OFF the table "
                              f"(Y would change by "
                              f"{(_new_y_pr-_cur_pos_pr[1])*100:+.1f}cm) "
                              f"— reverting move")
                        _seen.add(_slot_pr)
                        continue
                    _new_pos_pr[1] = _new_y_pr
                else:
                    _new_pos_pr[1] = float(_cur_pos_pr[1])
                _e["position_m"] = _new_pos_pr.tolist()
                _e["surface_position"] = _side
                if _ptype != "against_back":
                    _e["depth_position"] = _depth
                _seen.add(_slot_pr)
                _depth_log = (f"cz={_cz_target:+.3f}"
                              + (" (preserved — against_back)"
                                 if _ptype == "against_back" else ""))
                print(f"    [pos_refine] '{_phrase}' on "
                      f"{_ftype_pr}[{_fi}]: moved to slot "
                      f"({_side},{_depth}) → local "
                      f"cx={_cx_new:+.3f} {_depth_log} "
                      f"(was cx={_cur_cx_pr:+.3f} cz={_cur_cz_pr:+.3f}). "
                      f"{str(_result_pr.get('reasoning', ''))[:80]}")
                _pr_moved += 1
            if _pr_moved == 0:
                print(f"  [pos_refine] pass {_pr_pass}: no corrections "
                      f"needed — done")
                break
            print(f"  [pos_refine] pass {_pr_pass}: moved {_pr_moved} "
                  f"item(s)")

        # ── After pos_refine: re-run on-surface overlap resolution ──
        # The VLM often picks the SAME slot ("center, back") for two
        # items on the same surface (e.g. monitor + lamp on a coffee
        # table), so they end up stacked.  Sweep pairwise and push
        # the cheaper-to-separate item along the surface's local axis,
        # clamped to the table's metadata bbox.
        try:
            _ol_mesh = _get_furn_scene_mesh(out_dir)
        except Exception:
            _ol_mesh = None
        if _ol_mesh is not None:
            _verts_ol = np.asarray(_ol_mesh.vertices, dtype=np.float64)
            _ol_groups: dict[int, list[dict]] = {}
            for _e in placements:
                if _e.get("placement_type") != "on_surface":
                    continue
                _fidx = _e.get("furniture_index")
                if _fidx is not None:
                    _ol_groups.setdefault(_fidx, []).append(_e)
            _ol_moved_count = 0
            for _surf_idx, _surf_group in _ol_groups.items():
                if len(_surf_group) < 2:
                    continue
                _surf = furn_map.get(_surf_idx)
                if _surf is None or "position_m" not in _surf:
                    continue
                _surf_pos = np.array(_surf["position_m"], dtype=np.float64)
                _sw_meta_ol = _surf["size_m"]["width_m"]
                _sd_meta_ol = _surf["size_m"]["depth_m"]
                _R_surf_ol = np.array(_surf.get("rotation_3x3",
                                                 np.eye(3).tolist()),
                                       dtype=np.float64)
                # Use metadata bbox for clamping — same reasoning as the
                # slot AABB: union-with-mesh pulls in neighbours.
                _surf_xz_min_ol = np.array([-_sw_meta_ol / 2.0,
                                             -_sd_meta_ol / 2.0])
                _surf_xz_max_ol = np.array([+_sw_meta_ol / 2.0,
                                             +_sd_meta_ol / 2.0])

                def _local_rect_ol(_e: dict) -> tuple[
                        float, float, float, float]:
                    _pos = np.array(_e["position_m"], dtype=np.float64)
                    _off = _pos - _surf_pos
                    _cx_l = float(_off @ _R_surf_ol[:, 0])
                    _cz_l = float(_off @ _R_surf_ol[:, 2])
                    _sz   = _e.get("size_m", {})
                    _hw   = float(_sz.get("width_m", 0.30)) / 2.0
                    _hd   = float(_sz.get("depth_m", 0.30)) / 2.0
                    return _cx_l, _cz_l, _hw, _hd

                for _ol_iter in range(4):
                    _ol_changed = False
                    for _i in range(len(_surf_group)):
                        for _j in range(_i + 1, len(_surf_group)):
                            _a = _surf_group[_i]
                            _b = _surf_group[_j]
                            _ax, _az, _ahw, _ahd = _local_rect_ol(_a)
                            _bx, _bz, _bhw, _bhd = _local_rect_ol(_b)
                            _ovx = (_ahw + _bhw) - abs(_bx - _ax)
                            _ovz = (_ahd + _bhd) - abs(_bz - _az)
                            if _ovx <= 0.0 or _ovz <= 0.0:
                                continue
                            _eps_ol = 0.005
                            if _ovx <= _ovz:
                                _push_x_ol = ((_ovx + _eps_ol)
                                              * (1.0 if _bx >= _ax
                                                  else -1.0))
                                _push_z_ol = 0.0
                            else:
                                _push_x_ol = 0.0
                                _push_z_ol = ((_ovz + _eps_ol)
                                              * (1.0 if _bz >= _az
                                                  else -1.0))
                            _new_bx_ol = float(np.clip(
                                _bx + _push_x_ol,
                                _surf_xz_min_ol[0] + _bhw,
                                _surf_xz_max_ol[0] - _bhw))
                            _new_bz_ol = float(np.clip(
                                _bz + _push_z_ol,
                                _surf_xz_min_ol[1] + _bhd,
                                _surf_xz_max_ol[1] - _bhd))
                            _adx_ol = _new_bx_ol - _bx
                            _adz_ol = _new_bz_ol - _bz
                            # If B's chosen-axis push got clamped, try the
                            # OTHER axis on B before falling back to pushing A.
                            # See the matching comment in the first overlap
                            # pass (~line 6397) for rationale.
                            if (abs(_adx_ol) < 1e-3
                                    and abs(_adz_ol) < 1e-3
                                    and (abs(_push_x_ol) > 1e-3
                                         or abs(_push_z_ol) > 1e-3)):
                                if abs(_push_x_ol) > 1e-3:
                                    _alt_pz = ((_ovz + _eps_ol)
                                               * (1.0 if _bz >= _az else -1.0))
                                    _alt_px = 0.0
                                else:
                                    _alt_px = ((_ovx + _eps_ol)
                                               * (1.0 if _bx >= _ax else -1.0))
                                    _alt_pz = 0.0
                                _new_bx_ol = float(np.clip(
                                    _bx + _alt_px,
                                    _surf_xz_min_ol[0] + _bhw,
                                    _surf_xz_max_ol[0] - _bhw))
                                _new_bz_ol = float(np.clip(
                                    _bz + _alt_pz,
                                    _surf_xz_min_ol[1] + _bhd,
                                    _surf_xz_max_ol[1] - _bhd))
                                _adx_ol = _new_bx_ol - _bx
                                _adz_ol = _new_bz_ol - _bz
                                if abs(_adx_ol) > 1e-3 or abs(_adz_ol) > 1e-3:
                                    _push_x_ol = _alt_px
                                    _push_z_ol = _alt_pz
                            if (abs(_adx_ol) < 1e-3
                                    and abs(_adz_ol) < 1e-3
                                    and (abs(_push_x_ol) > 1e-3
                                         or abs(_push_z_ol) > 1e-3)):
                                # Both axes on B clamped — push A in the
                                # opposite direction.
                                _new_ax_ol = float(np.clip(
                                    _ax - _push_x_ol,
                                    _surf_xz_min_ol[0] + _ahw,
                                    _surf_xz_max_ol[0] - _ahw))
                                _new_az_ol = float(np.clip(
                                    _az - _push_z_ol,
                                    _surf_xz_min_ol[1] + _ahd,
                                    _surf_xz_max_ol[1] - _ahd))
                                _ax_d = _new_ax_ol - _ax
                                _az_d = _new_az_ol - _az
                                if (abs(_ax_d) > 1e-3
                                        or abs(_az_d) > 1e-3):
                                    _new_pos_a = (
                                        np.array(_a["position_m"],
                                                  dtype=np.float64)
                                        + _ax_d * _R_surf_ol[:, 0]
                                        + _az_d * _R_surf_ol[:, 2])
                                    _a["position_m"] = _new_pos_a.tolist()
                                    print(f"  [pos_refine/overlap] "
                                          f"surf[{_surf_idx}] "
                                          f"'{_a.get('phrase','?')}' "
                                          f"pushed "
                                          f"({_ax_d*100:+.1f},"
                                          f"{_az_d*100:+.1f})cm — "
                                          f"clears "
                                          f"'{_b.get('phrase','?')}'")
                                    _ol_changed = True
                                    _ol_moved_count += 1
                            elif (abs(_adx_ol) > 1e-3
                                    or abs(_adz_ol) > 1e-3):
                                _new_pos_b = (
                                    np.array(_b["position_m"],
                                              dtype=np.float64)
                                    + _adx_ol * _R_surf_ol[:, 0]
                                    + _adz_ol * _R_surf_ol[:, 2])
                                _b["position_m"] = _new_pos_b.tolist()
                                print(f"  [pos_refine/overlap] "
                                      f"surf[{_surf_idx}] "
                                      f"'{_b.get('phrase','?')}' "
                                      f"pushed "
                                      f"({_adx_ol*100:+.1f},"
                                      f"{_adz_ol*100:+.1f})cm "
                                      f"to clear "
                                      f"'{_a.get('phrase','?')}' "
                                      f"(overlap was "
                                      f"{_ovx*100:.0f}×{_ovz*100:.0f}cm)")
                                _ol_changed = True
                                _ol_moved_count += 1
                    if not _ol_changed:
                        break
            if _ol_moved_count > 0:
                print(f"  [pos_refine/overlap] separated "
                      f"{_ol_moved_count} item(s) on shared surfaces")

        # ── Top-down footprint clamp ──────────────────────────────────
        # Rule: the decoration's BOTTOM-FACE FOOTPRINT (top-down view)
        # must lie inside the supporting furniture's footprint.  Apply
        # only when part of the decor footprint extends past the
        # furniture footprint — silhouette / back-proj positions stay
        # untouched when they're already inside.
        # We project the decor's rotated (dw × dd) footprint onto the
        # furniture's local axes:
        #   extent_x_in_furn = sw/2-component of the decor footprint =
        #     |R_d[:,0]·R_f[:,0]|·dw/2 + |R_d[:,2]·R_f[:,0]|·dd/2
        #   extent_z_in_furn = same with R_f[:,2]
        # Then clamp the decor's centre so |cx|+extent_x_in_furn ≤ sw/2-margin
        # and |cz|+extent_z_in_furn ≤ sd/2-margin (i.e. the four decor
        # corners stay inside the canonical metadata bbox).
        # After the clamp we re-raycast Y at the new (x, z) so a moved
        # pillow sits on the actual seat surface there rather than
        # hovering at its original Y.
        _CLAMP_MARGIN = 0.02
        for _entry_fc in placements:
            if _entry_fc.get("_pos_locked"):
                continue
            _ptype_fc = _entry_fc.get("placement_type", "")
            if _ptype_fc not in ("against_back", "on_surface"):
                continue
            _seat_fc = _entry_fc.get("furniture_index")
            _furn_fc = furn_map.get(_seat_fc)
            if _furn_fc is None or "position_m" not in _furn_fc:
                continue
            _sw_fc = float(_furn_fc["size_m"]["width_m"])
            _sd_fc = float(_furn_fc["size_m"]["depth_m"])
            _dw_fc = float(_entry_fc["size_m"].get("width_m", 0.30))
            _dd_fc = float(_entry_fc["size_m"].get("depth_m", 0.30))
            _dh_fc = float(_entry_fc["size_m"].get("height_m", 0.30))
            _R_furn_fc = np.array(_furn_fc.get("rotation_3x3",
                                                 np.eye(3).tolist()),
                                   dtype=np.float64)
            _R_decor_fc = np.array(_entry_fc.get("rotation_3x3",
                                                   np.eye(3).tolist()),
                                    dtype=np.float64)
            # Decor footprint extent projected onto furniture's local
            # axes — full 3-axis projection so back-leaning items
            # (against_back pillows tilt 20° around their X axis,
            # mixing pillow-Y into sofa-Z) are correctly bounded.
            _ext_x = (abs(float(_R_decor_fc[:, 0] @ _R_furn_fc[:, 0]))
                       * _dw_fc / 2.0
                      + abs(float(_R_decor_fc[:, 1] @ _R_furn_fc[:, 0]))
                       * _dh_fc / 2.0
                      + abs(float(_R_decor_fc[:, 2] @ _R_furn_fc[:, 0]))
                       * _dd_fc / 2.0)
            _ext_z = (abs(float(_R_decor_fc[:, 0] @ _R_furn_fc[:, 2]))
                       * _dw_fc / 2.0
                      + abs(float(_R_decor_fc[:, 1] @ _R_furn_fc[:, 2]))
                       * _dh_fc / 2.0
                      + abs(float(_R_decor_fc[:, 2] @ _R_furn_fc[:, 2]))
                       * _dd_fc / 2.0)
            # Maximum centre offset that keeps the corners inside.  If
            # decor is bigger than furniture along that axis, clamp to
            # 0 (best we can do — keeps it centred).
            _max_cx = max(_sw_fc / 2.0 - _CLAMP_MARGIN - _ext_x, 0.0)
            _f_pos_fc = np.array(_furn_fc["position_m"], dtype=np.float64)
            _p_pos_fc = np.array(_entry_fc["position_m"], dtype=np.float64)
            _off_fc = _p_pos_fc - _f_pos_fc
            _cx_fc = float(_off_fc @ _R_furn_fc[:, 0])
            _cz_fc = float(_off_fc @ _R_furn_fc[:, 2])
            _cx_clamp = float(np.clip(_cx_fc, -_max_cx, _max_cx))
            # Strict symmetric Z clamp: the decor's footprint must lie
            # fully inside the furniture's metadata bbox along the
            # depth axis too.  No back-face overhang for against_back
            # items — even though that means a pillow on a sofa with a
            # broken generated meshes (back-rest behind metadata) can't
            # touch the visible cushion, top-down bbox containment is
            # the user-required invariant.
            _max_cz = max(_sd_fc / 2.0 - _CLAMP_MARGIN - _ext_z, 0.0)
            _cz_clamp = float(np.clip(_cz_fc, -_max_cz, _max_cz))
            if (abs(_cx_clamp - _cx_fc) > 1e-3
                    or abs(_cz_clamp - _cz_fc) > 1e-3):
                _new_pos_fc = _f_pos_fc.copy()
                _new_pos_fc += _cx_clamp * _R_furn_fc[:, 0]
                _new_pos_fc += _cz_clamp * _R_furn_fc[:, 2]
                # Re-raycast Y at the new (x, z) so the decor sits on
                # the actual surface there.  Without this, a clamped
                # pillow keeps its old Y which was the seat height at
                # the OLD cz — at a new cz the seat may dip and the
                # pillow visibly hovers.
                try:
                    _new_y_fc = _actual_surface_height(
                        out_dir,
                        float(_new_pos_fc[0]),
                        float(_new_pos_fc[2]),
                        float(_p_pos_fc[1]))
                    _new_pos_fc[1] = _new_y_fc
                except Exception:
                    _new_pos_fc[1] = float(_p_pos_fc[1])
                _entry_fc["position_m"] = _new_pos_fc.tolist()
                _ftype_fc = (_furn_fc.get("type") or "furniture").lower()
                print(f"  [final_clamp] '{_entry_fc.get('phrase','?')}' "
                      f"on {_ftype_fc}[{_seat_fc}]: footprint clamped "
                      f"inside furniture (local "
                      f"cx {_cx_fc:+.3f}→{_cx_clamp:+.3f}, "
                      f"cz {_cz_fc:+.3f}→{_cz_clamp:+.3f}; "
                      f"y {_p_pos_fc[1]:.3f}→{_new_pos_fc[1]:.3f}; "
                      f"furn ½= {_sw_fc/2:.2f}×{_sd_fc/2:.2f}, "
                      f"decor footprint extent ={_ext_x:.2f}×{_ext_z:.2f})")

        _do_render(out_dir, placements,
                   placements_dir / "render_pos_refine_final.png",
                   inpaint_dir=inpaint_dir)

    # ── Post-loop: VLM-guided reorder (disabled — place in silhouette order without swapping) ──
    if False and _ref_photo is not None:
        def _entry_mean_color(entry: dict) -> "tuple | None":
            """Mean (R,G,B) of the entry's inpaint image, or None on failure."""
            try:
                _si = entry.get("seg_index", -1)
                _seg_info = next(
                    (s for s in seg_data.get("segments", []) if s.get("seg_index") == _si),
                    None,
                )
                if _seg_info is None:
                    return None
                _inf = _seg_info.get("inpaint_file")
                if not _inf:
                    return None
                _ip = inpaint_dir / _inf
                if not _ip.exists():
                    return None
                _im = _PILImg.open(str(_ip)).convert("RGB")
                _arr = np.array(_im, dtype=np.float32)
                return tuple(int(v) for v in _arr.mean(axis=(0, 1)))
            except Exception:
                return None

        def _colors_close(c1, c2, thresh: float = 35.0) -> bool:
            if c1 is None or c2 is None:
                return False
            return float(np.linalg.norm(np.array(c1, dtype=float) - np.array(c2, dtype=float))) < thresh

        for (_ab_fi,), _ab_entries in _ab_groups.items():
            if len(_ab_entries) < 2:
                continue
            try:
                # Use the most recent step render as the "current state"
                _last_steps = sorted(placements_dir.glob("render_step_*.png"))
                _reorder_render_path = _last_steps[-1] if _last_steps else None
                if _reorder_render_path is None:
                    continue
                _reorder_render = _PILImg.open(_reorder_render_path).convert("RGB")
                _ab_phrase = _ab_entries[0].get("phrase", "pillows")
                print(f"\n[reorder] checking {len(_ab_entries)} items on sofa [{_ab_fi}] vs reference ...")
                _ro = _vlm_reorder_group(_reorder_render, _ref_photo, _ab_phrase, len(_ab_entries))
                if _ro is None:
                    continue
                if _ro.get("correct"):
                    print(f"  [reorder] arrangement correct. {_ro.get('reasoning', '')}")
                    continue
                # Multi-swap: apply each [i, j] pair in sequence on the current ordering
                _swaps = _ro.get("swaps", [])
                # Back-compat: also accept old single-swap format
                if not _swaps and "swap_i" in _ro and "swap_j" in _ro:
                    _swaps = [[int(_ro["swap_i"]), int(_ro["swap_j"])]]
                if not _swaps:
                    print(f"  [reorder] no swaps specified — skipping")
                    continue
                # Build an index array representing the current order (0..n-1)
                _n = len(_ab_entries)
                _order = list(range(_n))   # _order[k] = which original entry is at position k
                _any_swapped = False
                for _sw in _swaps:
                    if not (isinstance(_sw, (list, tuple)) and len(_sw) == 2):
                        continue
                    _si, _sj = int(_sw[0]), int(_sw[1])
                    if _si < 0 or _sj < 0 or _si >= _n or _sj >= _n or _si == _sj:
                        print(f"  [reorder] invalid swap [{_si},{_sj}] — skipping")
                        continue
                    _order[_si], _order[_sj] = _order[_sj], _order[_si]
                    _any_swapped = True
                    print(f"  [reorder] swap [{_si}↔{_sj}] applied")
                if not _any_swapped:
                    continue

                # Determine if all pillows in this group share the same dominant color.
                # Same-color: also permute GLB/scale so silhouettes move with positions.
                # Different-color: position-only swap (color identifies which is which).
                _reo_colors = [_entry_mean_color(e) for e in _ab_entries]
                _all_same_color = len(_reo_colors) > 1 and all(
                    _colors_close(_reo_colors[0], _reo_colors[_ci])
                    for _ci in range(1, len(_reo_colors))
                )
                print(f"  [reorder] dominant colors: {_reo_colors}  "
                      f"all_same_color={_all_same_color} → "
                      f"{'GLB+pos swap' if _all_same_color else 'pos-only swap'}")

                # Snapshot originals before permuting
                _orig_positions  = [list(_ab_entries[k]["position_m"])   for k in range(_n)]
                _orig_rotations  = [list(_ab_entries[k]["rotation_3x3"]) for k in range(_n)]
                if _all_same_color:
                    _orig_glb_files  = [_ab_entries[k].get("glb_file")  for k in range(_n)]
                    _orig_glb_paths  = [_ab_entries[k].get("glb_path")  for k in range(_n)]
                    _orig_scales     = [_ab_entries[k].get("scale")     for k in range(_n)]
                    _orig_scale_xyzs = [_ab_entries[k].get("scale_xyz") for k in range(_n)]

                for _k in range(_n):
                    _ab_entries[_k]["position_m"]   = _orig_positions[_order[_k]]
                    _ab_entries[_k]["rotation_3x3"] = _orig_rotations[_order[_k]]
                    if _all_same_color:
                        _ab_entries[_k]["glb_file"] = _orig_glb_files[_order[_k]]
                        _ab_entries[_k]["glb_path"] = _orig_glb_paths[_order[_k]]
                        _ab_entries[_k]["scale"]    = _orig_scales[_order[_k]]
                        _sxyz = _orig_scale_xyzs[_order[_k]]
                        if _sxyz is not None:
                            _ab_entries[_k]["scale_xyz"] = _sxyz
                        elif "scale_xyz" in _ab_entries[_k]:
                            del _ab_entries[_k]["scale_xyz"]

                print(f"  [reorder] final order: {_order}. {_ro.get('reasoning', '')}")
                # Re-render all decorations with reordered positions
                _do_render(out_dir, placements, placements_dir / "render_reordered.png", inpaint_dir=inpaint_dir)
            except Exception as _roe:
                print(f"  [reorder] failed for [{_ab_fi}] {_ab_phrase}: {_roe}")

    with open(out_path, "w") as f:
        json.dump(placements, f, indent=2)

    # Top-down plan of the finished scene (furniture footprints + decoration
    # dots).  Host-assignment mistakes are obvious from above and invisible in
    # the reference-camera view: a lamp that landed on the wrong table reads as
    # a dot sitting on the wrong rectangle.
    try:
        from object_placement import geometry_audit as _ga
        _wa = out_dir / "walls.obj"
        if _wa.exists():
            import numpy as _np
            _vv = _np.array([[float(x) for x in _l.split()[1:4]]
                             for _l in open(_wa) if _l.startswith("v ")])
            _cam = None
            for _c in (out_dir / "camera_vggt.json", out_dir / "camera.json"):
                if _c.exists():
                    _cam = json.loads(_c.read_text())
                    break
            _ga.render_plan(furn_list, float(_vv[:, 0].max()), float(_vv[:, 2].max()),
                            placements_dir / "render_topdown.png",
                            decorations=placements, cam=_cam,
                            title=f"{out_dir.name} — scene plan "
                                  f"({len(placements)} decorations)")
    except Exception as _pe:
        print(f"[topdown] decoration plan skipped: {_pe}")

    # Final render with all decorations
    _render_final(out_dir, placements, placements_dir, inpaint_dir=inpaint_dir)

    # Export combined scene GLB (furniture + decorations)
    _export_scene_with_decorations(out_dir, placements, placements_dir)

    # ── Animation GIF ─────────────────────────────────────────────────────────
    if animate:
        try:
            from PIL import Image as _PIL
            base_render = placements_dir / "render_base.png"
            # Sort step renders by modification time so the GIF plays in the
            # actual placement order, not alphabetical filename order.
            # Filenames embed seg_index (e.g. render_step_02_…), but the
            # placement order is determined by the presort key (against_back
            # before on_surface, etc.), not by seg_index, so an alphabetical
            # sort would interleave frames out of sequence (00 first even
            # though it was placed 5th).
            step_files = sorted(
                placements_dir.glob("render_step_*.png"),
                key=lambda p: p.stat().st_mtime,
            )
            final_render = placements_dir / "render_decorations_placed.png"
            if final_render.exists():
                step_files = list(step_files) + [final_render]
            if base_render.exists():
                step_files = [base_render] + list(step_files)
            if len(step_files) >= 2:
                frames: list[_PIL.Image] = []
                for fp in step_files:
                    try:
                        fr = _PIL.open(str(fp)).convert("RGB")
                        if fr.width != animate_gif_width:
                            ratio = animate_gif_width / fr.width
                            fr = fr.resize(
                                (animate_gif_width, int(fr.height * ratio)),
                                _PIL.LANCZOS,
                            )
                        frames.append(fr)
                    except Exception:
                        pass
                if frames:
                    gif_path = placements_dir / "decoration_animation.gif"
                    frames[0].save(
                        str(gif_path),
                        save_all=True,
                        append_images=frames[1:],
                        duration=animate_frame_ms,
                        loop=0,
                        optimize=False,
                    )
                    print(f"\n[animate] Saved decoration animation → {gif_path} "
                          f"({len(frames)} frames)")
        except Exception as _ae:
            print(f"[animate] GIF assembly failed: {_ae}")

    print(f"\n[place_decorations] Done. {len(placements)}/{len(segments)} decorations placed "
          f"→ {out_path}")

    # ── Optional: detect + place items missing from the scene ────────────────
    # When --detect-missing is on, ask the VLM whether anything (e.g. lamp,
    # sculpture, vase) appears in the photo but not in the render.  If so,
    # delegate to the standalone place_missing_items stage.  Kept off by
    # default so the decoration loop stays a single-shot operation; users
    # who want the missing-item pass run it explicitly (see CLI).
    if detect_missing:
        try:
            from object_placement.missing_items.place_missing_items import (
                run as _run_missing,
            )
            print("\n[place_decorations] running missing-item detection …")
            _run_missing(output_dir=str(out_dir),
                         no_texture=False,
                         rot_steps=4,
                         rot_step_deg=30.0)
        except Exception as _me:
            print(f"[place_decorations] missing-item stage failed: {_me}")

    return out_path


# ── Rendering ─────────────────────────────────────────────────────────────────

def _load_camera(output_dir: Path) -> dict | None:
    """Load camera dict with position_m/look_at_m/hfov_deg."""
    for cand in [
        output_dir / "camera_vggt.json",   # match furniture stage priority
        output_dir / "camera.json",
    ]:
        if cand.exists():
            return json.loads(cand.read_text())
    return None


NEAR_CLIP = 0.01


def _render_scene_glb(
    decoration_placements: list[dict],
    camera: dict,
    base_img: "Image.Image",
    dst: Path,
    furn_zbuf: "np.ndarray | None" = None,
    inpaint_dir: "Path | None" = None,
) -> None:
    """Project decoration GLBs onto the furniture base render.

    The base image already has furniture at correct pixel positions.
    We project only decoration meshes (with their position/rotation/scale)
    on top using a fresh z-buffer so decorations composite correctly.
    Saves to dst — furniture folder is never touched.
    """
    import trimesh
    from object_placement.wall_mounted.wall_mounted_object_placement import (
        _camera_axes, _project_vertex,
    )
    from object_placement.wall_mounted.placements.fill_openings import (
        _get_vertex_colors, _rasterize_vc_tri,
    )

    W, H = base_img.size
    buf  = np.array(base_img, dtype=np.uint8).copy()
    # Use pre-computed furniture z-buffer if provided, else start empty
    zbuf = furn_zbuf.copy() if furn_zbuf is not None else np.full((H, W), np.inf, dtype=np.float32)

    cam_pos  = np.array(camera["position_m"], dtype=np.float64)
    look_at  = np.array(camera["look_at_m"],  dtype=np.float64)
    up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
    right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
    hfov = float(camera["hfov_deg"])
    fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
    cx, cy = W / 2.0, H / 2.0

    def _project_mesh(verts: np.ndarray, faces: np.ndarray,
                      vert_colors: np.ndarray,
                      zbuf_override: "np.ndarray | None" = None,
                      face_colors: "np.ndarray | None" = None) -> None:
        """Rasterize triangles. If face_colors (F,3) uint8 is provided, use flat
        shading (same color all 3 vertices) to avoid UV-seam interpolation noise."""
        _zbuf = zbuf_override if zbuf_override is not None else zbuf
        proj = [_project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
                for v in verts]
        for fi in range(len(faces)):
            i0, i1, i2 = int(faces[fi,0]), int(faces[fi,1]), int(faces[fi,2])
            px0,py0,zc0 = proj[i0]; px1,py1,zc1 = proj[i1]; px2,py2,zc2 = proj[i2]
            if zc0 <= NEAR_CLIP or zc1 <= NEAR_CLIP or zc2 <= NEAR_CLIP:
                continue
            pts = np.array([[px0,py0,zc0],[px1,py1,zc1],[px2,py2,zc2]], dtype=np.float32)
            if face_colors is not None:
                fc = face_colors[fi]
                col = np.array([fc, fc, fc], dtype=np.uint8)
            else:
                col = np.array([vert_colors[i0],vert_colors[i1],vert_colors[i2]], dtype=np.uint8)
            _rasterize_vc_tri(buf, _zbuf, pts, col)

    # ── Decoration objects (apply position/rotation/scale) ───────────────
    for p in decoration_placements:
        glb_path = Path(p.get("glb_path", ""))
        if not glb_path.exists():
            continue
        try:
            sc = trimesh.load(str(glb_path), force="scene")
            if isinstance(sc, trimesh.Scene):
                meshes = sc.dump()  # applies node transforms; UV still intact per sub-mesh
            else:
                meshes = [sc]
            # Extract vertex colors from each sub-mesh BEFORE concatenation loses UV/texture info
            _per_vc = [_get_vertex_colors(m, glb_path=glb_path) for m in meshes]
            _pre_vc = np.concatenate(_per_vc, axis=0) if len(_per_vc) > 1 else _per_vc[0]
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        except Exception as e:
            print(f"  [render] decoration GLB load failed {glb_path.name}: {e}")
            continue

        scale = float(p["scale"][0])
        # scale_xyz: per-axis non-uniform scale (X=width, Y=height, Z=depth in GLB space).
        # Falls back to uniform scale if not set.
        _sxyz = p.get("scale_xyz")
        scale_xyz = np.array(_sxyz, dtype=np.float64) if _sxyz else np.array([scale, scale, scale])
        R     = np.array(p["rotation_3x3"], dtype=np.float64)
        pos   = np.array(p["position_m"],   dtype=np.float64)

        # Centre, scale, rotate, translate into world space
        bounds = mesh.bounds
        centre = (bounds[0] + bounds[1]) / 2.0
        verts  = (mesh.vertices.astype(np.float64) - centre) * scale_xyz
        # De-tilt: level the mesh so it rests flat (for on_surface objects)
        placement_type = p.get("placement_type", "on_surface")
        if placement_type == "on_surface":
            verts = _dec_level_base(verts)
        # Rotate (yaw placement rotation), then anchor bottom at pos Y
        verts  = verts @ R.T
        base_y = float(verts[:, 1].min())
        verts[:, 1] -= base_y
        verts  += pos

        # Try inpaint PNG as primary color source (richer than generated UV textures).
        # Fall back to pre-concatenation vertex colors if inpaint unavailable/dark.
        #
        # Previous version iterated sub-meshes and compared a single sub-mesh's
        # UV count to the CONCATENATED total vertex count, so multi-sub-mesh
        # GLBs always silently failed the `len(_sampled) == len(vc)` check and
        # fell back to grey _pre_vc — which is why whole decorations rendered
        # untextured.  Now we sample UVs from every sub-mesh and concatenate.
        vc = _pre_vc
        # Prefer the mesh's OWN baseColorTexture over sampling the inpaint PNG
        # through UVs.  The inpaint-via-UV path assumes UVs map to the 2D image —
        # true for vertex-color / plane meshes, but Hunyuan meshes carry an
        # ATLAS UV baseColorTexture, so sampling the inpaint through atlas UVs
        # scrambles it into a flat average (a textured plant → muddy green).
        # _pre_vc (=_get_vertex_colors) already samples the real texture, so only
        # fall back to the inpaint when the mesh has no texture of its own.
        _sub_list0 = meshes if isinstance(meshes, list) else [mesh]
        _has_own_texture = False
        for _sm in _sub_list0:
            _mat0 = getattr(getattr(_sm, "visual", None), "material", None)
            if _mat0 is not None and (
                getattr(_mat0, "baseColorTexture", None) is not None
                or getattr(_mat0, "image", None) is not None
            ):
                _has_own_texture = True
                break
        _inpaint_fname = p.get("inpaint_file")
        if _inpaint_fname and not _has_own_texture:
            _inpaint_path = inpaint_dir / _inpaint_fname
            if _inpaint_path.exists():
                try:
                    _bright = np.array(Image.open(str(_inpaint_path)).convert("RGB"),
                                       dtype=np.uint8)
                    if _bright.mean() > 50:
                        _sub_list = meshes if isinstance(meshes, list) else [mesh]
                        _ih, _iw = _bright.shape[:2]
                        _per_sub_sampled: list = []
                        _all_ok = True
                        for _sm in _sub_list:
                            _vis = _sm.visual
                            _uv = getattr(_vis, "uv", None)
                            if _uv is None:
                                _all_ok = False
                                break
                            _uv_arr = np.asarray(_uv)
                            if _uv_arr.shape[0] != len(_sm.vertices):
                                _all_ok = False
                                break
                            _u_px = np.clip((_uv_arr[:, 0] * (_iw - 1)).astype(int), 0, _iw - 1)
                            _v_px = np.clip(((1.0 - _uv_arr[:, 1]) * (_ih - 1)).astype(int), 0, _ih - 1)
                            _per_sub_sampled.append(_bright[_v_px, _u_px, :3].astype(np.uint8))
                        if _all_ok and _per_sub_sampled:
                            _cat = np.concatenate(_per_sub_sampled, axis=0) \
                                if len(_per_sub_sampled) > 1 else _per_sub_sampled[0]
                            if len(_cat) == len(vc):
                                vc = _cat
                except Exception:
                    pass

        # ── Lambertian shading ──────────────────────────────────────────────
        # Furniture is rendered with FLAT texture colors (no shading modulation,
        # since the reference inpaint already has photographic lighting baked
        # in).  Previously decorations applied strong Lambertian with ambient=
        # 0.35 and two directional terms summing to 0.70 — so decorations were
        # 35-105% of texture brightness, i.e. 15-65% darker than furniture on
        # average.  That's the main reason decorations "looked grey" next to
        # furniture even when their inpaint textures sampled correctly.
        #
        # Match furniture's flat-lit look: ambient=0.85, tiny directional term
        # just to keep a whisper of 3D depth on normals facing away from the
        # key light.  Max brightness is now 1.0, min is 0.85 (was 0.35).
        try:
            raw_normals = mesh.vertex_normals.astype(np.float64)
            normals_world = (raw_normals @ R.T)
            nlen = np.linalg.norm(normals_world, axis=1, keepdims=True)
            nlen = np.where(nlen < 1e-8, 1.0, nlen)
            normals_world /= nlen
            _key  = np.array([0.6,  1.2,  0.8], dtype=np.float64)
            _fill = np.array([-0.3, 0.4, -0.4], dtype=np.float64)
            _key  /= np.linalg.norm(_key)
            _fill /= np.linalg.norm(_fill)
            L = np.clip(0.85
                        + 0.10 * np.clip(normals_world @ _key,  0, 1)
                        + 0.05 * np.clip(normals_world @ _fill, 0, 1),
                        0.0, 1.0)[:, None]
            vc = np.clip(vc.astype(np.float64) * L, 0, 255).astype(np.uint8)
        except Exception:
            pass

        # ── Z-buffer strategy ───────────────────────────────────────────────
        # Always depth-test against the FURNITURE z-buffer so a decoration is
        # occluded by furniture genuinely in front of it (e.g. a bowl on a side
        # table BEHIND the sofa must be hidden by the sofa — not painted on top
        # like a sticker).  Earlier this forced on_surface/against_back items to a
        # fresh empty buffer (drawn always-on-top) — useful only as a debug aid;
        # the real render must reflect actual depth.
        dec_zbuf = zbuf.copy()

        _project_mesh(verts, mesh.faces, vc, zbuf_override=dec_zbuf)
        np.minimum(zbuf, dec_zbuf, out=zbuf)
        mean_c = vc.mean(axis=0).astype(int) if vc.ndim == 2 else vc
        print(f"  [render] decoration '{p.get('phrase')}' projected  "
              f"pos={np.round(pos,2).tolist()}  mean_color={mean_c.tolist()}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(buf).save(str(dst))
    print(f"  [render] saved → {dst.name}")


# Cache: (out_dir_str, W, H) → furniture z-buffer ndarray
_FURN_ZBUF_CACHE: dict[tuple, "np.ndarray"] = {}


def _build_furn_zbuf(out_dir: Path, camera: dict, W: int, H: int) -> "np.ndarray | None":
    """Build furniture z-buffer once and cache it (keyed by out_dir+size)."""
    key = (str(out_dir), W, H)
    if key in _FURN_ZBUF_CACHE:
        return _FURN_ZBUF_CACHE[key]

    furn_glb = out_dir / "furniture" / "scene_with_furniture.glb"
    if not furn_glb.exists():
        return None

    try:
        import trimesh
        from object_placement.wall_mounted.wall_mounted_object_placement import (
            _camera_axes, _project_vertex,
        )
        from object_placement.wall_mounted.placements.fill_openings import (
            _get_vertex_colors, _rasterize_vc_tri,
        )

        cam_pos  = np.array(camera["position_m"], dtype=np.float64)
        look_at  = np.array(camera["look_at_m"],  dtype=np.float64)
        up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
        right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
        hfov = float(camera["hfov_deg"])
        fx   = W / (2.0 * np.tan(np.radians(hfov / 2.0)))
        cx, cy = W / 2.0, H / 2.0

        furn_scene = trimesh.load(str(furn_glb), force="scene")
        if isinstance(furn_scene, trimesh.Scene):
            furn_meshes = furn_scene.dump()  # applies node transforms
        else:
            furn_meshes = [furn_scene]
        furn_mesh = trimesh.util.concatenate(furn_meshes) if len(furn_meshes) > 1 else furn_meshes[0]

        fv = furn_mesh.vertices.astype(np.float64)
        ff = furn_mesh.faces
        fproj = [_project_vertex(v, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy) for v in fv]
        fvc = np.full((len(fv), 3), 128, dtype=np.uint8)
        zbuf = np.full((H, W), np.inf, dtype=np.float32)
        tmp_buf = np.zeros((H, W, 3), dtype=np.uint8)

        for fi in range(len(ff)):
            i0, i1, i2 = int(ff[fi, 0]), int(ff[fi, 1]), int(ff[fi, 2])
            px0, py0, zc0 = fproj[i0]; px1, py1, zc1 = fproj[i1]; px2, py2, zc2 = fproj[i2]
            if zc0 <= NEAR_CLIP or zc1 <= NEAR_CLIP or zc2 <= NEAR_CLIP:
                continue
            pts = np.array([[px0,py0,zc0],[px1,py1,zc1],[px2,py2,zc2]], dtype=np.float32)
            col = np.array([fvc[i0],fvc[i1],fvc[i2]], dtype=np.uint8)
            _rasterize_vc_tri(tmp_buf, zbuf, pts, col)

        # Bias so decorations sitting ON surfaces pass the depth test
        zbuf[zbuf < np.inf] += 0.30
        _FURN_ZBUF_CACHE[key] = zbuf
        print(f"  [render] furniture z-buffer built and cached ({W}×{H})")
        return zbuf

    except Exception as e:
        print(f"  [render] furniture z-buffer failed: {e}")
        return None


def _do_render(out_dir: Path, placements_so_far: list[dict], dst: Path,
               inpaint_dir: "Path | None" = None,
               base_img: "Image.Image | None" = None) -> None:
    """Render scene_with_furniture + decorations placed so far → dst."""
    from PIL import Image as _PIL_Image
    _PIL_Image.MAX_IMAGE_PIXELS = None

    camera = _load_camera(out_dir)
    if camera is None:
        print("  [render] no camera_vggt.json found — skipping")
        return

    if base_img is None:
        # Composite decorations on the FURNITURE render. The ceiling phase now runs
        # AFTER decorations (ceiling fixture composites on TOP of the decoration
        # render), so we must NOT base on any ceiling render here — one present
        # would be stale from a prior run and would drop the new decorations.
        base_render = out_dir / "furniture" / "render_furniture_placed.png"
        if base_render.exists():
            base_img = _PIL_Image.open(base_render).convert("RGB")
            print(f"  [render] base = {base_render.parent.name}/{base_render.name}")
        else:
            print("  [render] no furniture render found — cannot render")
            return

    orig_w, orig_h = base_img.size
    max_side = 3000
    if orig_w > max_side:
        scale_down = max_side / orig_w
        new_w = int(orig_w * scale_down)
        new_h = int(orig_h * scale_down)
        base_img = base_img.resize((new_w, new_h), _PIL_Image.LANCZOS)
        print(f"  [render] downsampled {orig_w}×{orig_h} → {new_w}×{new_h}")

    W, H = base_img.size
    furn_zbuf = _build_furn_zbuf(out_dir, camera, W, H)
    _inpaint_dir = inpaint_dir or (out_dir / "decorations" / "inpainted")

    # Resolve stale absolute glb_path values (e.g. copied from a different
    # scene dir) by falling back to glb_file relative to out_dir.
    resolved = []
    for p in placements_so_far:
        glb_abs = Path(p.get("glb_path", ""))
        if not glb_abs.exists():
            glb_rel = p.get("glb_file", "")
            candidate = out_dir / glb_rel if glb_rel else None
            if candidate and candidate.exists():
                p = dict(p, glb_path=str(candidate))
        resolved.append(p)

    _render_scene_glb(resolved, camera, base_img, dst,
                      furn_zbuf=furn_zbuf, inpaint_dir=_inpaint_dir)


def _silhouette_scale_refine(
    entry: dict,
    box_px: list[float] | None,
    camera: dict,
    render_w: int,
    render_h: int,
    orig_ref_w: int,
    orig_ref_h: "int | None" = None,
    n_instances_in_mask: int = 1,
) -> float:
    """
    Compare the decoration's projected 2D bbox in render space against the
    original segment mask box_px (from the reference photo) and return an
    adjusted scale.

    Clamp policy:
      * Partially-visible item (mask touches a ref-photo edge): ratio in [0.8, 1.5]
        — a clamp protects against one-sided masks that would blow up the scale.
      * Fully-visible item (mask margin ≥ 2% from every edge): ratio in [0.5, 4.0]
        — we CAN trust the silhouette area as an accurate size reference, so
        match it directly even if that means a 3–4× rescale (a lamp whose mesh
        was tiny will scale up to reach the mask's silhouette).

    render_w / render_h : dimensions of the downsampled render (e.g. 1500×1125)
    orig_ref_w / orig_ref_h : dimensions of the original reference photo
    """
    if box_px is None:
        return float(entry["scale"][0])

    try:
        from object_placement.wall_mounted.wall_mounted_object_placement import (
            _camera_axes, _project_vertex,
        )

        scale  = float(entry["scale"][0])
        R      = np.array(entry["rotation_3x3"], dtype=np.float64)
        pos    = np.array(entry["position_m"],   dtype=np.float64)
        size_m = entry.get("size_m", {})
        hw = float(size_m.get("width_m",  0.3)) / 2.0
        hh = float(size_m.get("height_m", 0.3)) / 2.0
        hd = float(size_m.get("depth_m",  0.3)) / 2.0

        # 8 corners of the axis-aligned bounding box in world space.
        # pos is the bottom contact point; shift up by hh to get the geometric center.
        centre_world = pos + np.array([0.0, hh, 0.0])
        signs = np.array([[sx, sy, sz]
                          for sx in (-1, 1)
                          for sy in (-1, 1)
                          for sz in (-1, 1)], dtype=np.float64)
        half  = np.array([hw, hh, hd], dtype=np.float64)
        corners_world = centre_world + (signs * half) @ R.T   # (8, 3)

        cam_pos  = np.array(camera["position_m"], dtype=np.float64)
        look_at  = np.array(camera["look_at_m"],  dtype=np.float64)
        up_world = np.array(camera.get("up", [0, 1, 0]), dtype=np.float64)
        right_v, up_c_v, fwd_v = _camera_axes(cam_pos, look_at, up_world)
        hfov = float(camera["hfov_deg"])
        fx   = render_w / (2.0 * np.tan(np.radians(hfov / 2.0)))
        cx, cy = render_w / 2.0, render_h / 2.0

        px_list, py_list = [], []
        for c in corners_world:
            px, py, zc = _project_vertex(c, cam_pos, right_v, up_c_v, fwd_v, fx, cx, cy)
            if zc > 0.01:
                px_list.append(px)
                py_list.append(py)

        if not px_list:
            return scale

        proj_w = max(px_list) - min(px_list)
        proj_h = max(py_list) - min(py_list)
        proj_area = max(proj_w * proj_h, 1.0)

        # box_px is in original photo coords — scale to render space
        ds = render_w / orig_ref_w
        x1, y1, x2, y2 = [v * ds for v in box_px]
        mask_w = max(x2 - x1, 1.0)
        mask_h = max(y2 - y1, 1.0)

        # Skip refinement when the mask is too small in original reference space.
        # A tiny mask (e.g. < 20px in either dim at 3000px) means the object was
        # far away or partially visible — the absolute pixel count is too noisy to
        # use as a reliable size reference.  Trust the VLM estimate instead.
        orig_mask_w = mask_w / ds   # back to original photo pixels
        orig_mask_h = mask_h / ds
        _MIN_MASK_PX = 20.0
        if orig_mask_w < _MIN_MASK_PX or orig_mask_h < _MIN_MASK_PX:
            print(f"    [silhouette] mask too small ({orig_mask_w:.1f}×{orig_mask_h:.1f}px orig) "
                  f"— skipping refinement, trusting VLM scale {scale:.4f}")
            return scale

        mask_area = mask_w * mask_h

        # Detect "complete view": the mask bbox has a healthy margin from every
        # edge of the reference photo.  If the bbox TOUCHES an edge, the object
        # is partially out-of-frame in the ref, so matching the mask area would
        # scale the full mesh to fit only the visible portion — wrong.
        _ref_h_eff = float(orig_ref_h) if orig_ref_h else float(orig_ref_w) * 3.0 / 4.0
        _orig_x1 = float(box_px[0]); _orig_y1 = float(box_px[1])
        _orig_x2 = float(box_px[2]); _orig_y2 = float(box_px[3])
        _edge_margin_ref = max(float(orig_ref_w), _ref_h_eff) * 0.02  # 2% margin
        _complete_in_view = (
            _orig_x1 > _edge_margin_ref
            and _orig_y1 > _edge_margin_ref
            and _orig_x2 < float(orig_ref_w) - _edge_margin_ref
            and _orig_y2 < _ref_h_eff - _edge_margin_ref
        )

        # Per-dimension ratios. With min(w_r, h_r) the mesh fits INSIDE the mask
        # in both dimensions but stays smaller than the photo silhouette when
        # the GLB's projected aspect differs from the mask's aspect — which is
        # how a tall lamp ends up rendered too short, or a monitor too small.
        # The user wants placed objects sized closer to the photo, so use the
        # geometric mean (= sqrt of area ratio): the uniform-scaled mesh
        # matches the mask AREA, splitting any aspect mismatch evenly between
        # over- and under-coverage.  Type-specific size caps downstream (in
        # the caller) keep this from blowing past physical plausibility.
        # Multi-instance correction: when the SAM mask captured N piles
        # side-by-side (e.g. two book stacks), the bbox spans the whole
        # group.  Treat a single instance as occupying ~1/√N of the bbox
        # along each axis so the per-mesh size matches one instance, then
        # let the duplication path replicate.  Without this the silhouette
        # ratio doubles each book to fill the combined area.
        _n_inst = max(1, int(n_instances_in_mask))
        if _n_inst > 1:
            _div = float(np.sqrt(_n_inst))
            mask_w = mask_w / _div
            mask_h = mask_h / _div
            print(f"    [silhouette] {_n_inst} mask blobs detected — "
                  f"shrinking effective mask to per-instance "
                  f"({mask_w:.1f}×{mask_h:.1f}px, ÷{_div:.2f})")
        w_ratio = mask_w / max(proj_w, 1.0)
        h_ratio = mask_h / max(proj_h, 1.0)
        ratio_min  = min(w_ratio, h_ratio)
        ratio_max  = max(w_ratio, h_ratio)
        ratio_geom = float(np.sqrt(w_ratio * h_ratio))

        # Aspect-mismatch guard: if the projected GLB and the mask have very
        # different aspect ratios (>2× apart), uniform scaling cannot honour
        # both axes — that mismatch usually means either the mask is partial /
        # occluded (a pillow with another in front of it: tall-thin mask) or
        # the GLB's pose / shape doesn't match the photo (lamp arm folded
        # vs extended).  Treat that as "partial" so we don't make the mesh
        # tiny just because the visible mask is a narrow strip.
        _proj_aspect = proj_w / max(proj_h, 1.0)
        _mask_aspect = mask_w / max(mask_h, 1.0)
        _aspect_mismatch = (max(_proj_aspect, _mask_aspect)
                            / max(min(_proj_aspect, _mask_aspect), 1e-6))
        _aspect_partial = _aspect_mismatch > 2.0

        if _complete_in_view and not _aspect_partial:
            # Full object visible AND aspect roughly matches → silhouette
            # trustworthy.  When EITHER dimension wants growth (ratio > 1),
            # bias toward MAX so the larger axis matches the photo (the
            # smaller axis may slightly under-cover the mask, but visually
            # the object is properly sized).  When both dims want shrink,
            # use geometric mean (equal area-match).  Allow 0.5×–4×.
            if ratio_max >= 1.0:
                _ratio_choice = ratio_max
            else:
                _ratio_choice = ratio_geom
            ratio_clamped = float(np.clip(_ratio_choice, 0.5, 4.0))
            _mode = "complete-view"
        else:
            # Partial / edge-touching silhouette OR aspect-mismatched (=
            # occlusion / pose mismatch) → don't trust area for scale.  Tight
            # clamp around 1.0 keeps the VLM scale mostly intact, avoiding
            # half-sized pillows behind another pillow.
            ratio_clamped = float(np.clip(ratio_geom, 0.85, 1.30))
            _mode = "partial-aspect" if _aspect_partial else "partial"
        new_scale = scale * ratio_clamped
        print(f"    [silhouette] proj={proj_w:.1f}×{proj_h:.1f}px  "
              f"mask={mask_w:.1f}×{mask_h:.1f}px  w_r={w_ratio:.2f} h_r={h_ratio:.2f} "
              f"→ geom={ratio_geom:.3f} (min={ratio_min:.3f} max={ratio_max:.3f}) "
              f"aspect_mm={_aspect_mismatch:.2f} "
              f"clamped={ratio_clamped:.3f} ({_mode})  "
              f"scale {scale:.4f}→{new_scale:.4f}")
        return new_scale

    except Exception as e:
        print(f"    [silhouette] refinement failed: {e}")
        return float(entry["scale"][0])


def _render_incremental(
    out_dir: Path,
    placements: list[dict],
    seg_idx: int,
    phrase: str,
    placements_dir: Path,
    inpaint_dir: "Path | None" = None,
) -> None:
    try:
        safe = phrase.replace(" ", "_")
        dst = placements_dir / f"render_step_{seg_idx:02d}_{safe}.png"
        _do_render(out_dir, placements, dst, inpaint_dir=inpaint_dir)
    except Exception as e:
        print(f"  [render] incremental render failed: {e}")


def _render_pillow_slide(
    out_dir: Path,
    placements: list[dict],
    entry: dict,
    old_pos: "list[float] | np.ndarray",
    new_pos: "list[float] | np.ndarray",
    seg_idx: int,
    phrase: str,
    placements_dir: Path,
    n_steps: int,
    inpaint_dir: "Path | None" = None,
    label_suffix: str = "slide",
) -> None:
    """Render N intermediate frames showing `entry` sliding from old_pos
    to new_pos.  Each step is a separate PNG so the GIF assembly picks
    them all up.  No-op when n_steps <= 0.

    Caller is responsible for committing `entry["position_m"]` to the
    final value AFTER this function returns (we end on new_pos).
    """
    if n_steps <= 0:
        return
    try:
        safe = phrase.replace(" ", "_")
        old_arr = np.asarray(old_pos, dtype=np.float64)
        new_arr = np.asarray(new_pos, dtype=np.float64)
        # Skip if the move is tiny (under 1 cm) — animating it adds
        # near-identical frames that just slow the GIF.
        if float(np.linalg.norm(new_arr - old_arr)) < 0.01:
            return
        for _i in range(1, int(n_steps) + 1):
            t = _i / float(n_steps)
            entry["position_m"] = (old_arr * (1.0 - t) + new_arr * t).tolist()
            dst = placements_dir / (
                f"render_step_{seg_idx:02d}_{safe}_{label_suffix}{_i:02d}.png"
            )
            _do_render(out_dir, placements, dst, inpaint_dir=inpaint_dir)
    except Exception as e:
        print(f"  [render] slide animation failed: {e}")


def _export_scene_with_decorations(
    out_dir: Path,
    placements: list[dict],
    placements_dir: Path,
) -> None:
    """Merge furniture scene GLB + all placed decoration GLBs → scene_with_decoration.glb."""
    try:
        import trimesh

        furn_glb = out_dir / "furniture" / "scene_with_furniture.glb"
        if not furn_glb.exists():
            print(f"  [glb_export] furniture GLB not found: {furn_glb} — skipping")
            return

        scene = trimesh.load(str(furn_glb), force="scene")
        if not isinstance(scene, trimesh.Scene):
            scene = trimesh.scene.scene.Scene(geometry={"furniture": scene})

        for p in placements:
            glb_path = Path(p.get("glb_path", ""))
            if not glb_path.exists():
                # Portable resolution: decoration_placements.json stores paths
                # RELATIVE to the scene dir (survives folder moves). Also self-heal
                # legacy absolute paths that point at a pre-move location by
                # relocating to this scene's decorations/objects/<basename>.
                cand = (out_dir / glb_path) if not glb_path.is_absolute() else None
                if cand is not None and cand.exists():
                    glb_path = cand
                else:
                    heal = out_dir / "decorations" / "objects" / glb_path.name
                    if heal.exists():
                        glb_path = heal
            if not glb_path.exists():
                continue

            scale   = float(p["scale"][0])
            R       = np.array(p["rotation_3x3"], dtype=np.float64)
            pos     = np.array(p["position_m"],   dtype=np.float64)
            phrase  = p.get("phrase", "decoration")
            seg_idx = p.get("seg_index", 0)

            try:
                dec_scene = trimesh.load(str(glb_path), force="scene")
                if isinstance(dec_scene, trimesh.Scene):
                    meshes = dec_scene.dump()  # applies node transforms
                else:
                    meshes = [dec_scene]
                mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            except Exception as e:
                print(f"  [glb_export] failed to load {glb_path.name}: {e}")
                continue

            # Place decoration: centre → scale → detilt → placement-rotate → translate
            bounds = mesh.bounds
            centre = (bounds[0] + bounds[1]) / 2.0
            verts_w = (mesh.vertices.astype(np.float64) - centre) * scale
            # De-tilt so bottom face rests flat (on_surface decorations)
            placement_type = p.get("placement_type", "on_surface")
            if placement_type == "on_surface":
                verts_w = _dec_level_base(verts_w)
            # Yaw rotation then anchor to surface
            verts_w = verts_w @ R.T
            base_y  = float(verts_w[:, 1].min())
            verts_w[:, 1] -= base_y
            verts_w += pos

            mesh_out = mesh.copy()
            mesh_out.vertices = verts_w
            name = f"decor_{seg_idx:02d}_{phrase.replace(' ', '_')}"
            scene.add_geometry(mesh_out, node_name=name)

        dst = placements_dir / "scene_with_decoration.glb"
        scene.export(str(dst))
        print(f"  [glb_export] saved → {dst}")

        # Re-texture the room shell (walls/floor/ceiling) + wall-mounted objects
        # so the decoration GLB is FULLY textured — furniture + decorations are
        # already textured, but the merged-in furniture scene carries the flat
        # baked walls.  Same retexture the assemble stage uses.
        try:
            from object_placement.assemble_scene_glb import retexture_scene as _retexture_scene
            _retexture_scene(out_dir, dst)
            print(f"  [glb_export] re-textured walls/floor/ceiling + wall objects → {dst}")
        except Exception as _rex:
            print(f"  [glb_export] wall/floor retexture skipped (flat walls): {_rex}")

    except Exception as e:
        print(f"  [glb_export] FAILED: {e}")


def _render_final(
    out_dir: Path,
    placements: list[dict],
    placements_dir: Path,
    inpaint_dir: "Path | None" = None,
) -> None:
    try:
        dst = placements_dir / "render_decorations_placed.png"
        _do_render(out_dir, placements, dst, inpaint_dir=inpaint_dir)
        print(f"[render] Final render → {dst}")
    except Exception as e:
        print(f"[render] final render failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Place decoration objects onto furniture surfaces."
    )
    ap.add_argument("--output-dir", required=True,
                    help="Pipeline output dir (contains furniture/ and decorations/)")
    ap.add_argument("--first-only", action="store_true",
                    help="Process only the first decoration (for testing)")
    ap.add_argument("--animate", action="store_true",
                    help="Assemble per-step renders into decoration_animation.gif")
    ap.add_argument("--animate-width", type=int, default=1500,
                    help="Width of animation GIF in pixels (default 1500). "
                         "Bumped from 960 so small items (succulents, mugs) "
                         "stay legible at GIF playback size.")
    ap.add_argument("--animate-duration", type=int, default=800,
                    help="Duration per frame in ms (default 800)")
    ap.add_argument("--animate-pillow-slide-frames", type=int, default=4,
                    help="Number of intermediate frames to render when a "
                         "pillow slides between two positions (arm-rest "
                         "shift, verify-swap). 0 = disable slide animation. "
                         "Default 4.")
    ap.add_argument("--detect-missing", action="store_true",
                    help="After the decoration loop finishes, ask the VLM "
                         "whether anything visible in the reference photo is "
                         "still missing from the render (e.g. a lamp the "
                         "VLM analysis didn't catch). When true, delegates "
                         "to object_placement.missing_items.place_missing_items "
                         "which segments + generates + places each missing "
                         "item.  Off by default — run that stage manually "
                         "if you want isolated control.")
    ap.add_argument("--max-objects", type=int, default=None,
                    help="Process only the first N decorations (for testing)")
    ap.add_argument("--phrase-contains", type=str, default=None,
                    help="Only process segments whose phrase contains this "
                         "substring (case-insensitive).  Useful with "
                         "--first-only to debug a specific decoration "
                         "type, e.g. --phrase-contains pillow --first-only.")
    ap.add_argument("--pillow-cx-local", type=float, default=None,
                    help="Manual override: place against_back pillows at "
                         "this cx (sofa-local frame, in metres).  Overrides "
                         "all heuristics.")
    ap.add_argument("--pillow-cz-local", type=float, default=None,
                    help="Manual override: place against_back pillows at "
                         "this cz (sofa-local frame, in metres).  Overrides "
                         "all heuristics.  Negative = toward back-rest, "
                         "positive = toward seat front.")
    ap.add_argument("--pillow-y-offset", type=float, default=None,
                    help="Raise (positive) or lower (negative) the pillow's "
                         "Y by this many metres.  Useful when the pillow's "
                         "back face is at the back-rest cushion but the "
                         "seat surface ends earlier — a small +Y bump "
                         "lifts the pillow's bottom out of the empty "
                         "gap into the back-rest cushion's visible volume.")
    args = ap.parse_args()
    run(output_dir=args.output_dir, first_only=args.first_only,
        animate=args.animate, animate_gif_width=args.animate_width,
        animate_frame_ms=args.animate_duration, max_objects=args.max_objects,
        phrase_contains=args.phrase_contains,
        pillow_cx_override=args.pillow_cx_local,
        pillow_cz_override=args.pillow_cz_local,
        pillow_y_offset=args.pillow_y_offset,
        pillow_slide_frames=args.animate_pillow_slide_frames,
        detect_missing=args.detect_missing)


if __name__ == "__main__":
    main()
