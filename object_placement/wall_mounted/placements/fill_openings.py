"""
fill_openings.py — place generated GLB objects at wall openings and render.

For each wall opening:
  1. Match to the corresponding segmented window/door GLB
  2. Load the GLB with trimesh
  3. Scale it to the opening dimensions, orient it to face the room interior,
     translate it to the opening centre on the wall
  4. Project all vertices to screen space with the same camera used by render_room
  5. Rasterize with per-vertex colour interpolation and a depth buffer
  6. Composite onto the base openings render

Outputs written to <output_dir>/wall_mounted/placements/:
    opening_placements.json
    render_openings_filled.png

Usage:
    python -m object_placement.wall_mounted.placements.fill_openings \\
        --output-dir outputs/20260331_031530
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


NEAR_CLIP = 0.05   # metres — skip vertices behind this distance from camera
EPS       = 0.005  # metres inward from wall so mesh sits inside the hole


# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers  (identical to window_filling.py / render_room.py)
# ─────────────────────────────────────────────────────────────────────────────

def _camera_axes(pos, look_at, up):
    fwd = look_at - pos
    n   = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0., 0., 1.])
    right = np.cross(fwd, up)
    rn    = np.linalg.norm(right)
    if rn < 1e-9:
        right = np.cross(fwd, np.array([0., 0., 1.]))
        rn    = np.linalg.norm(right)
    right /= rn
    up_c  = np.cross(right, fwd)
    up_c /= np.linalg.norm(up_c)
    return right, up_c, fwd


def _make_projector(pos, look_at, up, hfov_deg, W_px, H_px):
    right, _, fwd = _camera_axes(pos, look_at, up)
    fx = W_px / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    cx, cy = W_px / 2.0, H_px / 2.0

    fwd_h   = np.array([fwd[0], 0.0, fwd[2]], dtype=float)
    fwd_h_n = float(np.linalg.norm(fwd_h))
    if fwd_h_n > 1e-9:
        fwd_h /= fwd_h_n
        tilt   = float(fwd[1]) / fwd_h_n
    else:
        fwd_h  = fwd.copy()
        tilt   = 0.0

    def _project(v3d: np.ndarray):
        d  = v3d - pos
        xc = float(d @ right)
        zc = float(d @ fwd_h)
        yc = float(d[1]) - tilt * zc
        if zc <= NEAR_CLIP:
            return None
        # depth = camera-space Z (for z-buffer)
        return cx + fx * xc / zc, cy - fx * yc / zc, zc

    return _project


# ─────────────────────────────────────────────────────────────────────────────
# Opening geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _opening_centre(op: dict, room: dict) -> np.ndarray | None:
    room_w = float(room.get("floor_width_m", 5.0))
    room_d = float(room.get("floor_depth_m", 4.0))
    wall   = op.get("wall", "back")
    off    = float(op.get("offset_from_left_m", 0.0))
    w      = float(op.get("width_m",  1.0))
    sill   = float(op.get("sill_height_m", 0.8))
    h      = float(op.get("height_m", 1.2))
    cy     = sill + h / 2

    if wall == "back":    return np.array([off + w/2,            cy, EPS])
    if wall == "front":   return np.array([room_w - off - w/2,   cy, room_d - EPS])
    if wall == "left":    return np.array([EPS,                  cy, room_d - off - w/2])
    if wall == "right":   return np.array([room_w - EPS,         cy, off + w/2])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GLB loading and placement
# ─────────────────────────────────────────────────────────────────────────────

def _read_color0_from_glb(glb_path: Path, n_vertices: int) -> np.ndarray | None:
    """Read the COLOR_0 attribute directly from a GLB binary file.

    trimesh creates TextureVisuals when a material with a texture is present
    and does not expose the embedded COLOR_0 vertex-color attribute through
    its normal API.  This function parses the raw GLB binary to extract it.

    Returns (n_vertices, 3) uint8 array, or None if the attribute is absent
    or cannot be decoded.
    """
    import struct as _struct, json as _json
    try:
        raw = Path(glb_path).read_bytes()
        # Parse GLB header + chunks
        off, json_bytes, bin_data = 12, None, b""
        while off < len(raw):
            clen, ctype = _struct.unpack_from("<II", raw, off)
            off += 8
            chunk = raw[off: off + clen]
            if   ctype == 0x4E4F534A: json_bytes = chunk   # JSON chunk
            elif ctype == 0x004E4942: bin_data   = chunk   # BIN  chunk
            off += clen
        if json_bytes is None:
            return None
        gltf  = _json.loads(json_bytes)
        attrs = (gltf.get("meshes", [{}])[0]
                     .get("primitives", [{}])[0]
                     .get("attributes", {}))
        if "COLOR_0" not in attrs:
            return None
        acc  = gltf["accessors"][attrs["COLOR_0"]]
        bv   = gltf["bufferViews"][acc["bufferView"]]
        if bv.get("buffer", 0) != 0:   # only handle the embedded BIN chunk
            return None
        n_comp    = 4 if acc["type"] == "VEC4" else 3
        comp_type = acc["componentType"]
        offset    = (bv.get("byteOffset") or 0) + (acc.get("byteOffset") or 0)
        if comp_type == 5121:          # UNSIGNED_BYTE
            fmt, stride, scale = f"{n_comp}B", bv.get("byteStride") or n_comp, 1.0
        elif comp_type == 5123:        # UNSIGNED_SHORT (normalised)
            fmt, stride, scale = f"{n_comp}H", bv.get("byteStride") or (2*n_comp), 255.0/65535.0
        elif comp_type == 5126:        # FLOAT (normalised)
            fmt, stride, scale = f"{n_comp}f", bv.get("byteStride") or (4*n_comp), 255.0
        else:
            return None
        n = acc["count"]
        colors = np.array(
            [_struct.unpack_from(fmt, bin_data, offset + i * stride) for i in range(n)],
            dtype=np.float32,
        )
        rgb = (colors[:, :3] * scale).clip(0, 255).astype(np.uint8)
        return rgb if len(rgb) == n_vertices else None
    except Exception:
        return None


def _get_vertex_colors(mesh, glb_path: Path | None = None) -> np.ndarray:
    """Extract per-vertex RGB uint8 from a trimesh mesh, falling back to grey.

    Handles TextureVisuals (UV-mapped GLBs from the 3D generator) by sampling the texture
    image at each vertex's UV coordinate.  When trimesh's API fails to expose
    the embedded texture (common for window-plane GLBs with COLOR_0 + material),
    falls back to reading COLOR_0 directly from the GLB binary.
    """
    n = len(mesh.vertices)

    # 1. TextureVisuals with UV + image
    try:
        vis = mesh.visual
        if hasattr(vis, "uv") and vis.uv is not None and hasattr(vis, "material"):
            mat  = vis.material
            img  = None
            if hasattr(mat, "image") and mat.image is not None:
                img = mat.image
            elif hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                img = mat.baseColorTexture
            if img is not None:
                img_arr = np.array(img.convert("RGB"))
                ih, iw  = img_arr.shape[:2]
                uv      = np.array(vis.uv)
                u_px    = np.clip((uv[:, 0] * (iw - 1)).astype(int), 0, iw - 1)
                v_px    = np.clip(((1.0 - uv[:, 1]) * (ih - 1)).astype(int), 0, ih - 1)
                return img_arr[v_px, u_px, :3].astype(np.uint8)
    except Exception:
        pass

    # 2. Direct vertex_colors attribute
    try:
        vc = np.array(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape == (n, 4):
            return vc[:, :3].astype(np.uint8)
        if vc.ndim == 2 and vc.shape[1] >= 3:
            return vc[:, :3].astype(np.uint8)
    except Exception:
        pass

    # 3. to_color() conversion
    try:
        colored = mesh.visual.to_color()
        vc = np.array(colored.vertex_colors)
        if vc.ndim == 2 and vc.shape == (n, 4):
            return vc[:, :3].astype(np.uint8)
    except Exception:
        pass

    # 4. Read COLOR_0 directly from the GLB binary (window-plane GLBs: trimesh
    #    creates TextureVisuals because of the material, hiding the COLOR_0 data)
    if glb_path is not None:
        vc = _read_color0_from_glb(glb_path, n)
        if vc is not None:
            return vc

    return np.full((n, 3), 180, dtype=np.uint8)


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _transform_glb_to_opening(
    glb_path: Path, op: dict, room: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load GLB, scale to opening dimensions, orient to wall, translate to position.

    Returns (verts_world, faces, vertex_colors) or None on failure.
    Assumes GLB natural orientation: X=width, Y=height, Z=depth (toward camera).
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("fill_openings requires trimesh: pip install trimesh")

    if not glb_path.exists():
        print(f"  [place] GLB not found: {glb_path}")
        return None

    try:
        scene = trimesh.load(str(glb_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if not geoms:
                return None
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        else:
            mesh = scene
    except Exception as e:
        print(f"  [place] Failed to load {glb_path.name}: {e}")
        return None

    if len(mesh.vertices) == 0:
        return None

    bb_min, bb_max = mesh.bounds
    bb_size   = (bb_max - bb_min).astype(np.float64)
    bb_center = ((bb_min + bb_max) / 2).astype(np.float64)

    verts = mesh.vertices.astype(np.float64) - bb_center

    # De-tilt: generated reconstructions can have a global tilt so the frame's
    # "up" direction is not exactly the Y axis.  Estimate the actual up
    # direction from the centroid of the top-third vs bottom-third of
    # vertices and rotate to align it with [0, 1, 0] before scaling/placing.
    p67 = np.percentile(verts[:, 1], 67)
    p33 = np.percentile(verts[:, 1], 33)
    top_mask = verts[:, 1] >= p67
    bot_mask = verts[:, 1] <= p33
    if top_mask.any() and bot_mask.any():
        up_est  = verts[top_mask].mean(axis=0) - verts[bot_mask].mean(axis=0)
        up_norm = np.linalg.norm(up_est)
        if up_norm > 1e-6:
            up_est /= up_norm
            cross   = np.cross(up_est, np.array([0., 1., 0.]))
            sin_a   = np.linalg.norm(cross)
            if sin_a > np.sin(np.radians(1.0)):   # only correct if > 1°
                axis    = cross / sin_a
                cos_a   = float(np.clip(np.dot(up_est, np.array([0., 1., 0.])), -1, 1))
                K = np.array([[0., -axis[2], axis[1]],
                              [axis[2], 0., -axis[0]],
                              [-axis[1], axis[0], 0.]])
                R_detilt = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
                verts    = verts @ R_detilt.T
                print(f"  [place] de-tilt applied: {np.degrees(np.arcsin(sin_a)):.1f}°")

    # Scale to opening width × height.
    # Give the window realistic physical depth (back face flush with wall plane,
    # front face protruding into the room) so the generator's 3D frame is visible.
    FRAME_DEPTH_M = 0.15   # physical depth of a typical window/door frame
    half_d = FRAME_DEPTH_M / 2.0

    opening_w = float(op["width_m"])
    opening_h = float(op["height_m"])
    wall       = op.get("wall", "back")

    # generated reconstructions have sparse stray vertices at the extremes that
    # inflate the bounding box beyond the actual frame structure.  Use the
    # p5/p95 robust range for the scale factors so the main frame fills the
    # opening; outlier vertices are clamped to the opening bounds afterward.
    p05 = np.percentile(verts, 5, axis=0)
    p95 = np.percentile(verts, 95, axis=0)
    robust_size = p95 - p05

    sx = opening_w     / max(robust_size[0], 1e-6)
    sy = opening_h     / max(robust_size[1], 1e-6)
    sz = FRAME_DEPTH_M / max(bb_size[2],     1e-6)
    verts *= np.array([sx, sy, sz])

    # Rotate so +Z (front face) points into the room interior, then translate
    # so the back face is flush with the wall plane and front face is inside.
    room_w = float(room.get("floor_width_m", 5.0))
    room_d = float(room.get("floor_depth_m", 4.0))
    off    = float(op.get("offset_from_left_m", 0.0))
    w      = float(op["width_m"])
    sill   = float(op.get("sill_height_m", 0.8))
    h      = float(op["height_m"])
    cy     = sill + h / 2

    if wall == "back":
        # back face at z=0, front face at z=FRAME_DEPTH_M (into room)
        center = np.array([off + w/2,          cy, half_d])
    elif wall == "front":
        verts  = verts @ _rot_y(np.pi).T
        center = np.array([room_w - off - w/2, cy, room_d - half_d])
    elif wall == "left":
        # +pi/2: +Z → +X; back face at x=0, front at x=FRAME_DEPTH_M
        verts  = verts @ _rot_y(np.pi / 2).T
        center = np.array([half_d,             cy, room_d - off - w/2])
    elif wall == "right":
        # -pi/2: +Z → -X; back face at x=room_w, front at x=room_w-FRAME_DEPTH_M
        verts  = verts @ _rot_y(-np.pi / 2).T
        center = np.array([room_w - half_d,    cy, off + w/2])
    else:
        return None

    verts += center

    # ── Clamp to opening bounds ──────────────────────────────────────────────
    # Stray outlier vertices (from generator noise) may extend beyond the opening
    # rectangle.  Clamp all axes to the opening bounds so no geometry leaks
    # into the adjacent wall or outside the room.
    if wall == "back":
        verts[:, 0] = np.clip(verts[:, 0], off, off + w)      # wall-horizontal
        verts[:, 1] = np.clip(verts[:, 1], sill, sill + h)    # vertical
        verts[:, 2] = np.maximum(verts[:, 2], 0.0)            # interior only
    elif wall == "front":
        verts[:, 0] = np.clip(verts[:, 0], room_w - off - w, room_w - off)
        verts[:, 1] = np.clip(verts[:, 1], sill, sill + h)
        verts[:, 2] = np.minimum(verts[:, 2], room_d)
    elif wall == "left":
        verts[:, 2] = np.clip(verts[:, 2], room_d - off - w, room_d - off)
        verts[:, 1] = np.clip(verts[:, 1], sill, sill + h)
        verts[:, 0] = np.maximum(verts[:, 0], 0.0)
    elif wall == "right":
        verts[:, 2] = np.clip(verts[:, 2], off, off + w)
        verts[:, 1] = np.clip(verts[:, 1], sill, sill + h)
        verts[:, 0] = np.minimum(verts[:, 0], room_w)

    vc    = _get_vertex_colors(mesh)
    faces = mesh.faces

    print(f"  [place] GLB {glb_path.name}: {len(verts)} verts, "
          f"centre={np.round(center,3).tolist()}, "
          f"scale=({sx:.3f}, {sy:.3f})")
    return verts, faces, vc


# ─────────────────────────────────────────────────────────────────────────────
# Software rasterizer with vertex-colour interpolation + z-buffer
# ─────────────────────────────────────────────────────────────────────────────

def _rasterize_vc_tri(
    buf:    np.ndarray,   # (H, W, 3) uint8 — modified in place
    zbuf:   np.ndarray,   # (H, W) float32 depth — modified in place
    pts:    np.ndarray,   # (3, 3): [x_px, y_px, depth] per vertex
    colors: np.ndarray,   # (3, 3) uint8 RGB
) -> None:
    H, W = buf.shape[:2]
    # Skip triangles whose any vertex projected to NaN/Inf — happens when an
    # upstream perspective divide hits Z_cam ≈ 0 (vertex at/behind the cam
    # plane). Without this guard, np.floor(NaN) → int crashes the rasterizer.
    if not np.isfinite(pts).all():
        return
    xmin = max(0,   int(np.floor(pts[:, 0].min())))
    xmax = min(W-1, int(np.ceil (pts[:, 0].max())))
    ymin = max(0,   int(np.floor(pts[:, 1].min())))
    ymax = min(H-1, int(np.ceil (pts[:, 1].max())))
    if xmin > xmax or ymin > ymax:
        return

    p0, p1, p2 = pts[0, :2], pts[1, :2], pts[2, :2]
    v0 = p1 - p0;  v1 = p2 - p0
    denom = float(v0[0]*v1[1] - v0[1]*v1[0])
    if abs(denom) < 0.5:
        return

    ys, xs = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    qx = xs.ravel().astype(np.float32) - p0[0]
    qy = ys.ravel().astype(np.float32) - p0[1]
    s  = (qx * v1[1] - qy * v1[0]) / denom
    t  = (qy * v0[0] - qx * v0[1]) / denom
    inside = (s >= 0) & (t >= 0) & (s + t <= 1.0)
    if not inside.any():
        return

    pxi = xs.ravel()[inside].astype(np.int32)
    pyi = ys.ravel()[inside].astype(np.int32)
    si  = s[inside];  ti = t[inside]

    # Interpolated depth
    depth = pts[0,2] + si * (pts[1,2] - pts[0,2]) + ti * (pts[2,2] - pts[0,2])

    # Z-buffer test
    vis = depth < zbuf[pyi, pxi]
    if not vis.any():
        return
    pxi = pxi[vis];  pyi = pyi[vis];  si = si[vis];  ti = ti[vis];  depth = depth[vis]

    # Interpolated vertex colour
    c0 = colors[0].astype(np.float32)
    c1 = colors[1].astype(np.float32)
    c2 = colors[2].astype(np.float32)
    col = c0 + si[:, None] * (c1 - c0) + ti[:, None] * (c2 - c0)

    buf[pyi, pxi]  = np.clip(col, 0, 255).astype(np.uint8)
    zbuf[pyi, pxi] = depth


# ─────────────────────────────────────────────────────────────────────────────
# Render: project placed GLBs onto the base render
# ─────────────────────────────────────────────────────────────────────────────

def render_with_glbs(
    output_dir:      str | Path,
    render_in_path:  str | None = None,
    render_out_path: str | None = None,
) -> str:
    out_dir      = Path(output_dir)
    openings_dir = out_dir / "openings"
    wm_dir       = out_dir / "wall_mounted"
    place_dir    = wm_dir / "placements"
    obj_dir      = wm_dir / "objects"

    placements_path = place_dir / "opening_placements.json"
    anal_path       = out_dir / "floorplan_analysis.json"
    for _cp in [out_dir / "camera_vggt.json",
                out_dir / "camera.json",
                openings_dir / "camera.json"]:
        if _cp.exists():
            cam_path = _cp
            break
    else:
        raise FileNotFoundError(f"No camera.json found in {out_dir}")

    if not placements_path.exists():
        raise FileNotFoundError("opening_placements.json not found — run fill_openings first")
    if not cam_path.exists():
        raise FileNotFoundError(f"camera.json not found in {out_dir}")

    placements = json.loads(placements_path.read_text())
    openings   = json.loads((openings_dir / "openings.json").read_text())
    analysis   = json.loads(anal_path.read_text()) if anal_path.exists() else {}
    room       = dict(analysis.get("room", {}))
    cam        = json.loads(cam_path.read_text())

    # Override room dims with actual walls OBJ bounds (VGGT-calibrated depth may
    # differ from floorplan_analysis.json estimate).
    base_obj = openings_dir / "walls_with_openings.obj"
    obj_dims = _room_dims_from_obj(base_obj)
    if obj_dims:
        room.update(obj_dims)

    pos     = np.array(cam["position_m"],       dtype=float)
    look_at = np.array(cam["look_at_m"],         dtype=float)
    up      = np.array(cam.get("up", [0,1,0]),  dtype=float)
    hfov    = float(cam["hfov_deg"])
    W_px    = int(cam["width_px"])
    H_px    = int(cam["height_px"])

    _project = _make_projector(pos, look_at, up, hfov, W_px, H_px)

    # Base render
    base_candidates = [
        openings_dir / "render_openings_refine_3.png",
        openings_dir / "render_openings_refine_2.png",
        openings_dir / "render_openings_refine_1.png",
        openings_dir / "render_openings.png",
    ]
    base_png = render_in_path or next(
        (str(p) for p in base_candidates if p.exists()), None
    )
    if base_png is None:
        raise FileNotFoundError("No base render found in openings/")

    base_img = Image.open(base_png).convert("RGB")
    if base_img.width != W_px or base_img.height != H_px:
        base_img = base_img.resize((W_px, H_px), Image.LANCZOS)

    buf  = np.array(base_img, dtype=np.uint8)
    zbuf = np.full((H_px, W_px), np.inf, dtype=np.float32)

    matched = [p for p in placements if p.get("matched_segment_index") is not None]
    print(f"[fill_openings] Placing {len(matched)} GLB(s) onto base render")

    for p in matched:
        seg_idx  = p["matched_segment_index"]
        obj_type = p["matched_type"]
        glb_name = f"inpaint_{seg_idx:02d}_{obj_type}.glb"
        glb_path = obj_dir / glb_name

        op = openings[p["opening_index"]]
        print(f"\n[fill_openings] Opening {p['opening_index']} ({op['wall']} {obj_type})")

        result = _transform_glb_to_opening(glb_path, op, room)
        if result is None:
            print(f"  skipping — GLB unavailable or empty")
            continue

        verts, faces, vc = result

        # Project all vertices once
        proj = []
        for v in verts:
            pt = _project(v)
            proj.append(pt)   # (x, y, depth) or None

        n_drawn = 0
        for tri in faces:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            p0 = proj[i0];  p1 = proj[i1];  p2 = proj[i2]
            if p0 is None or p1 is None or p2 is None:
                continue

            pts_arr    = np.array([p0, p1, p2], dtype=np.float32)
            colors_arr = np.array([vc[i0], vc[i1], vc[i2]], dtype=np.uint8)
            _rasterize_vc_tri(buf, zbuf, pts_arr, colors_arr)
            n_drawn += 1

        print(f"  {n_drawn}/{len(faces)} faces drawn")

    out_path = render_out_path or str(place_dir / "render_openings_filled.png")
    Image.fromarray(buf).save(out_path)
    print(f"\n[fill_openings] Render → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Matching: project opening centres → greedy nearest-segment assignment
# ─────────────────────────────────────────────────────────────────────────────

def _match_and_save_placements(
    openings, room, _project, W_px, H_px, segments, place_dir, obj_dir: Path | None = None
) -> list[dict]:
    # Only match segments whose GLB actually exists on disk
    def _glb_exists(seg: dict) -> bool:
        if not seg.get("glb_file"):
            return False
        if obj_dir is None:
            return True
        return (obj_dir / Path(seg["glb_file"]).name).exists()

    candidates = [
        s for s in segments
        if s.get("type") in ("window", "door") and _glb_exists(s)
    ]
    print(f"[fill_openings] {len(candidates)} window/door segment(s) with GLB available")

    def _centre(box_px):
        x1,y1,x2,y2 = box_px
        return ((x1+x2)/2, (y1+y2)/2)

    proj_centres = []
    for op in openings:
        cw = _opening_centre(op, room)
        if cw is None:
            proj_centres.append(None)
            continue
        pt = _project(cw)
        proj_centres.append((pt[0], pt[1]) if pt else None)

    n_open = len(openings)
    n_seg  = len(candidates)
    costs  = [[float("inf")] * n_seg for _ in range(n_open)]
    for i, px in enumerate(proj_centres):
        if px is None:
            continue
        for j, seg in enumerate(candidates):
            sc = _centre(seg["box_px"])
            d  = math.hypot(px[0]-sc[0], px[1]-sc[1])
            if d <= 250:
                costs[i][j] = d

    matched_segs = [None] * n_open
    used = set()
    for _ in range(min(n_open, n_seg)):
        best_d, best_i, best_j = float("inf"), -1, -1
        for i in range(n_open):
            for j in range(n_seg):
                if j not in used and costs[i][j] < best_d:
                    best_d, best_i, best_j = costs[i][j], i, j
        if best_i == -1:
            break
        matched_segs[best_i] = candidates[best_j]
        used.add(best_j)
        for j in range(n_seg):
            costs[best_i][j] = float("inf")

    placements = []
    for i, (op, px, seg) in enumerate(zip(openings, proj_centres, matched_segs)):
        cw     = _opening_centre(op, room)
        centre = cw.tolist() if cw is not None else None
        rec = {
            "opening_index":         i,
            "wall":                  op["wall"],
            "type":                  op["type"],
            "opening_width_m":       op["width_m"],
            "opening_height_m":      op["height_m"],
            "sill_height_m":         op.get("sill_height_m", 0.0),
            "position":              centre,
            "projected_px":          list(px) if px else None,
            "matched_segment_index": seg["index"]        if seg else None,
            "matched_type":          seg.get("type")     if seg else None,
            "glb_file":              seg.get("glb_file") if seg else None,
            "bbox_px":               seg["box_px"]       if seg else None,
        }
        if seg:
            sc = _centre(seg["box_px"])
            print(f"[fill_openings] Opening {i} ({op['wall']} {op['type']}) "
                  f"→ segment {seg['index']} "
                  f"(proj={[round(v,1) for v in px] if px else 'off-screen'}, "
                  f"bbox_c={[round(v,1) for v in sc]})")
        else:
            print(f"[fill_openings] Opening {i} ({op['wall']} {op['type']}) → no match")
        placements.append(rec)

    out = place_dir / "opening_placements.json"
    with open(out, "w") as f:
        json.dump(placements, f, indent=2)
    print(f"[fill_openings] Placements → {out}")
    return placements


# ─────────────────────────────────────────────────────────────────────────────
# Opening backing quad helpers
# ─────────────────────────────────────────────────────────────────────────────

def _opening_corners(op: dict, room: dict) -> list[list[float]] | None:
    """Return the 4 world-space corners of an opening as a flat rectangle.

    The quad sits just inside the wall plane (EPS offset) so it fills the hole
    in the walls mesh without Z-fighting.  Winding is counter-clockwise when
    viewed from the room interior.

    Corner order: bottom-left, bottom-right, top-right, top-left.
    """
    room_w = float(room.get("floor_width_m", 5.0))
    room_d = float(room.get("floor_depth_m", 4.0))
    wall   = op.get("wall", "back")
    off    = float(op.get("offset_from_left_m", 0.0))
    w      = float(op["width_m"])
    sill   = float(op.get("sill_height_m", 0.8))
    h      = float(op["height_m"])

    if wall == "back":
        # Interior face looks in +Z direction; EPS inside room at z=EPS
        return [
            [off,     sill,   EPS],
            [off + w, sill,   EPS],
            [off + w, sill+h, EPS],
            [off,     sill+h, EPS],
        ]
    if wall == "front":
        # Interior face looks in -Z direction
        return [
            [room_w - off - w, sill,   room_d - EPS],
            [room_w - off,     sill,   room_d - EPS],
            [room_w - off,     sill+h, room_d - EPS],
            [room_w - off - w, sill+h, room_d - EPS],
        ]
    if wall == "left":
        # Interior face looks in +X direction; opening spans z=[room_d-off-w, room_d-off]
        return [
            [EPS, sill,   room_d - off - w],
            [EPS, sill,   room_d - off    ],
            [EPS, sill+h, room_d - off    ],
            [EPS, sill+h, room_d - off - w],
        ]
    if wall == "right":
        # Interior face looks in -X direction
        return [
            [room_w - EPS, sill,   off    ],
            [room_w - EPS, sill,   off + w],
            [room_w - EPS, sill+h, off + w],
            [room_w - EPS, sill+h, off    ],
        ]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OBJ export: append placed GLB geometry to the walls mesh
# ─────────────────────────────────────────────────────────────────────────────

def save_combined_obj(
    output_dir: str | Path,
    placements:  list[dict],
    openings:    list[dict],
    room:        dict,
    obj_in_path:  str | None = None,
    obj_out_path: str | None = None,
) -> str | None:
    """Append transformed GLB vertices/faces to walls_with_openings.obj.

    Each opening's window/door GLB is scaled and placed at the correct wall
    position (interior side only).  The result is saved as walls_with_windows.obj
    in the placements directory, ready for render_room.py or inspection in a
    3D viewer.

    No textures are added for the window faces — they will inherit the solid-
    grey fallback that render_room.py applies to any face without a recognised
    material.
    """
    out_dir      = Path(output_dir)
    openings_dir = out_dir / "openings"
    wm_dir       = out_dir / "wall_mounted"
    place_dir    = wm_dir / "placements"
    obj_dir      = wm_dir / "objects"

    obj_in  = Path(obj_in_path)  if obj_in_path  else openings_dir / "walls_with_openings.obj"
    obj_out = Path(obj_out_path) if obj_out_path else place_dir    / "walls_with_windows.obj"

    if not obj_in.exists():
        print(f"[fill_openings] Base OBJ not found: {obj_in} — skipping mesh export")
        return None

    # Read original lines; count existing vertices so new indices are correct
    orig_lines = obj_in.read_text().splitlines(keepends=True)
    n_verts    = sum(1 for ln in orig_lines if ln.strip().startswith("v "))

    new_verts: list[str] = []
    new_faces: list[str] = ["\n# ── Window / door placements ──────────────────\n"]

    matched = [p for p in placements if p.get("matched_segment_index") is not None]
    print(f"[fill_openings] Building combined OBJ for {len(matched)} opening(s)")

    for p in matched:
        seg_idx  = p["matched_segment_index"]
        obj_type = p["matched_type"]
        glb_path = obj_dir / f"inpaint_{seg_idx:02d}_{obj_type}.glb"
        op       = openings[p["opening_index"]]

        new_faces.append(
            f"# opening {p['opening_index']} ({op['wall']} {obj_type})\n"
        )

        # ── 1. Solid backing quad — guarantees the hole is completely filled ──
        corners = _opening_corners(op, room)
        if corners:
            base_q = n_verts + 1
            for cx, cy, cz in corners:
                new_verts.append(f"v {cx:.6f} {cy:.6f} {cz:.6f}\n")
            # Two triangles (CCW winding): 0-1-2 and 0-2-3
            new_faces.append(
                f"f {base_q} {base_q+1} {base_q+2}\n"
                f"f {base_q} {base_q+2} {base_q+3}\n"
            )
            n_verts += 4
            print(f"  Opening {p['opening_index']} ({op['wall']} {obj_type}): "
                  f"backing quad added")

        # ── 2. generated GLBs mesh — adds 3-D frame detail on top ─────────────────
        result = _transform_glb_to_opening(glb_path, op, room)
        if result is None:
            continue

        verts, faces, _ = result   # colours not used in OBJ export

        base = n_verts + 1   # OBJ vertex indices are 1-based
        for x, y, z in verts:
            new_verts.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        for tri in faces:
            i0 = int(tri[0]) + base
            i1 = int(tri[1]) + base
            i2 = int(tri[2]) + base
            new_faces.append(f"f {i0} {i1} {i2}\n")

        n_verts += len(verts)
        print(f"  Opening {p['opening_index']} ({op['wall']} {obj_type}): "
              f"{len(verts)} verts, {len(faces)} GLB faces appended")

    place_dir.mkdir(parents=True, exist_ok=True)
    with open(obj_out, "w") as f:
        f.writelines(orig_lines)
        f.writelines(new_verts)
        f.writelines(new_faces)

    print(f"[fill_openings] Scene mesh → {obj_out}")
    return str(obj_out)


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-accurate opening refinement
# ─────────────────────────────────────────────────────────────────────────────

def _refine_and_rebuild(
    placements:  list[dict],
    cam:         dict,
    room:        dict,
    analysis:    dict,
    openings_dir: Path,
) -> list[dict]:
    """Back-project matched segment bboxes to get pixel-accurate opening positions,
    then regenerate walls_with_openings.obj with corrected holes.

    For each matched opening we:
      1. Feed its bbox_px as a pixel_rect to wall_openings._pixel_rects_to_openings,
         which back-projects all 8 bbox samples (corners + edge midpoints) onto the
         wall plane to derive offset_from_left_m / width_m / sill_height_m / height_m.
      2. Regenerate walls_with_openings.obj via build_walls_with_openings_obj.
      3. Update each placement dict with the refined opening dimensions and centre.

    Returns the updated placements list, or the original list unchanged if anything fails.
    """
    try:
        from floorplan.openings.wall_openings import (
            _pixel_rects_to_openings,
            build_walls_with_openings_obj,
        )
    except ImportError as e:
        print(f"[fill_openings] wall_openings import failed — skipping refinement: {e}")
        return placements

    matched_idx = [(i, p) for i, p in enumerate(placements)
                   if p.get("bbox_px") and p.get("matched_segment_index") is not None]
    if not matched_idx:
        print("[fill_openings] No matched placements — skipping pixel refinement")
        return placements

    raw_openings = [
        {"type": p.get("matched_type", "window"), "pixel_rect": p["bbox_px"]}
        for _, p in matched_idx
    ]
    wall_hints = [p["wall"] for _, p in matched_idx]

    room_ctx = {**room, "_visible_walls": list({p["wall"] for _, p in matched_idx})}
    refined = _pixel_rects_to_openings(raw_openings, cam, room_ctx, wall_hints=wall_hints)
    if not refined:
        print("[fill_openings] Pixel refinement yielded no openings — keeping VLM positions")
        return placements

    # ── Log the delta ──────────────────────────────────────────────────────────
    for ref, (_, orig_p) in zip(refined, matched_idx):
        print(
            f"[fill_openings] Pixel-refined {ref['wall']} {ref['type']}: "
            f"offset {orig_p.get('offset_from_left_m','?')} → {ref['offset_from_left_m']:.3f}m  "
            f"w {orig_p.get('opening_width_m','?')} → {ref['width_m']:.3f}m  "
            f"sill {orig_p.get('sill_height_m','?')} → {ref['sill_height_m']:.3f}m  "
            f"h {orig_p.get('opening_height_m','?')} → {ref['height_m']:.3f}m"
        )

    # ── Rule: window top ≤ curtain top on same wall ────────────────────────────
    # Curtains are often taller than what their mask shows (furniture occlusion),
    # but their TOP edge is reliably visible.  A window hole that extends above
    # the curtain top would peek out above it, which is physically impossible.
    # We back-project each curtain's topmost bbox pixel to world-y on the nearest
    # matching wall, then clamp the window height accordingly.
    try:
        from floorplan.openings.wall_openings import _pixel_to_nearest_wall

        seg_results_path = openings_dir.parent / "wall_mounted" / "segment_results.json"
        if seg_results_path.exists():
            curtain_segs = [
                s for s in json.loads(seg_results_path.read_text()).get("segments", [])
                if s.get("type") == "curtain" and s.get("box_px")
            ]
            W_r   = float(room.get("floor_width_m",    5.0))
            D_r   = float(room.get("floor_depth_m",    4.0))
            ceil_ = float(room.get("ceiling_height_m", 2.7))

            # Pair (wall, screen-x-centre) for each matched window — used to
            # assign each curtain to a wall by horizontal proximity.
            window_wall_centers = [
                (p["wall"], (p["bbox_px"][0] + p["bbox_px"][2]) / 2.0)
                for _, p in matched_idx if p.get("bbox_px")
            ]

            curtain_wall_tops: dict[str, float] = {}   # wall → min curtain top world-y
            for seg in curtain_segs:
                x1, y1, x2, y2 = seg["box_px"]
                cx_s = (x1 + x2) / 2.0
                if not window_wall_centers:
                    continue
                # Assign this curtain to the wall whose window is nearest in screen x
                curtain_wall = min(window_wall_centers,
                                   key=lambda wc: abs(wc[1] - cx_s))[0]
                # Back-project curtain top pixel to world-y on that wall
                top_hit = _pixel_to_nearest_wall(
                    cx_s, float(y1), cam, W_r, D_r, ceil_, {curtain_wall}
                )
                if top_hit is None:
                    continue
                curtain_top_y = min(float(top_hit[2]), ceil_)
                if curtain_wall not in curtain_wall_tops or curtain_top_y < curtain_wall_tops[curtain_wall]:
                    curtain_wall_tops[curtain_wall] = curtain_top_y

            for i, ref in enumerate(refined):
                wall = ref["wall"]
                if wall not in curtain_wall_tops:
                    continue
                max_top  = curtain_wall_tops[wall]
                win_top  = float(ref["sill_height_m"]) + float(ref["height_m"])
                if win_top > max_top + 0.01:
                    new_h = max(0.1, max_top - float(ref["sill_height_m"]))
                    print(f"[fill_openings] {wall} window top {win_top:.3f}m → {max_top:.3f}m "
                          f"(curtain-top constraint)")
                    refined[i] = dict(ref)
                    refined[i]["height_m"] = round(new_h, 3)

    except Exception as e:
        print(f"[fill_openings] Window-curtain top constraint failed: {e}")

    # ── Save refined openings.json ─────────────────────────────────────────────
    (openings_dir / "openings.json").write_text(json.dumps(refined, indent=2))
    print("[fill_openings] Refined openings.json saved")

    # ── Regenerate wall mesh with pixel-aligned holes ──────────────────────────
    build_walls_with_openings_obj(
        {**analysis, "room": room},
        refined,
        openings_dir,   # input_dir: where walls.mtl would be (may not exist)
        openings_dir,   # out_dir:   writes walls_with_openings.obj here
    )
    print("[fill_openings] walls_with_openings.obj rebuilt with pixel-aligned holes")

    # ── Update placement dicts with refined opening dimensions ─────────────────
    W    = float(room.get("floor_width_m",    5.0))
    D    = float(room.get("floor_depth_m",    4.0))
    ceil = float(room.get("ceiling_height_m", 2.7))

    updated = list(placements)
    for ref, (list_idx, _) in zip(refined, matched_idx):
        wall = ref["wall"]
        off  = float(ref["offset_from_left_m"])
        w    = float(ref["width_m"])
        s    = float(ref["sill_height_m"])
        h    = float(ref["height_m"])
        ctr  = off + w / 2.0
        cy   = s + h / 2.0

        if wall == "back":
            pos3d = [ctr,       cy, EPS]
        elif wall == "front":
            pos3d = [W - ctr,   cy, D - EPS]
        elif wall == "left":
            pos3d = [EPS,       cy, D - ctr]
        elif wall == "right":
            pos3d = [W - EPS,   cy, ctr]
        else:
            pos3d = updated[list_idx].get("position", [0.0, cy, 0.0])

        p2 = dict(updated[list_idx])
        p2["opening_width_m"]      = w
        p2["opening_height_m"]     = h
        p2["sill_height_m"]        = s
        p2["offset_from_left_m"]   = off
        p2["position"]             = pos3d
        updated[list_idx] = p2

    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Room dimension helpers
# ─────────────────────────────────────────────────────────────────────────────

def _room_dims_from_obj(obj_path: Path) -> dict[str, float]:
    """Parse an OBJ file and return actual room bounding-box dimensions.

    The pipeline may rebuild the mesh with VGGT-calibrated dimensions that
    differ from those stored in floorplan_analysis.json.  Reading the OBJ
    vertex bounds is the ground truth for the mesh that openings were cut into.
    """
    xs, ys, zs = [], [], []
    try:
        with open(obj_path) as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split()
                    xs.append(float(p[1]))
                    ys.append(float(p[2]))
                    zs.append(float(p[3]))
    except Exception:
        return {}
    if not xs:
        return {}
    return {
        "floor_width_m":   max(xs) - min(xs),
        "floor_depth_m":   max(zs) - min(zs),
        "ceiling_height_m": max(ys) - min(ys),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(output_dir: str | Path, holes_only: bool = False) -> None:
    """Run the fill-openings pipeline.

    holes_only=True  — stop after creating wall holes and saving opening_placements.json.
                       GLB placement and rendering are skipped; use this when
                       place_objects.py handles all wall-mounted objects (including
                       windows) via the unified mask-guided corner-based pipeline.
    holes_only=False — full pipeline: create holes, place window GLBs, render.
    """
    out_dir      = Path(output_dir)
    openings_dir = out_dir / "openings"
    wm_dir       = out_dir / "wall_mounted"
    place_dir    = wm_dir / "placements"
    place_dir.mkdir(parents=True, exist_ok=True)

    anal_path = out_dir / "floorplan_analysis.json"
    # Prefer VGGT-calibrated camera; fall back to VLM estimate then openings copy
    for _cp in [out_dir / "camera_vggt.json",
                out_dir / "camera.json",
                openings_dir / "camera.json"]:
        if _cp.exists():
            cam_path = _cp
            break
    else:
        raise FileNotFoundError(f"No camera.json found in {out_dir}")

    openings = json.loads((openings_dir / "openings.json").read_text())
    analysis = json.loads(anal_path.read_text()) if anal_path.exists() else {}
    room     = dict(analysis.get("room", {}))   # mutable copy

    # Override room dimensions with values measured from the actual walls OBJ.
    # The mesh may have been built with VGGT-calibrated dimensions that differ
    # from the initial VLM estimate stored in floorplan_analysis.json.
    base_obj = openings_dir / "walls_with_openings.obj"
    obj_dims = _room_dims_from_obj(base_obj)
    if obj_dims:
        for key, val in obj_dims.items():
            if abs(val - float(room.get(key, 0))) > 0.01:
                print(f"[fill_openings] Room {key}: "
                      f"analysis={room.get(key,'?')} → OBJ={val:.3f}m (using OBJ)")
        room.update(obj_dims)
    cam      = json.loads(cam_path.read_text())
    segments = json.loads((wm_dir / "segment_results.json").read_text()).get("segments", [])

    pos     = np.array(cam["position_m"],       dtype=float)
    look_at = np.array(cam["look_at_m"],         dtype=float)
    up      = np.array(cam.get("up", [0,1,0]),  dtype=float)
    _project = _make_projector(
        pos, look_at, up,
        float(cam["hfov_deg"]),
        int(cam["width_px"]), int(cam["height_px"]),
    )

    placements = _match_and_save_placements(
        openings, room, _project,
        int(cam["width_px"]), int(cam["height_px"]),
        segments, place_dir,
        obj_dir=wm_dir / "objects",
    )

    # Refine opening hole positions using pixel back-projection from matched bboxes,
    # then regenerate walls_with_openings.obj with pixel-aligned holes.
    placements = _refine_and_rebuild(placements, cam, room, analysis, openings_dir)

    # Reload openings (may have been updated by refinement) and refresh room dims.
    openings = json.loads((openings_dir / "openings.json").read_text())
    obj_dims = _room_dims_from_obj(base_obj)
    if obj_dims:
        room.update(obj_dims)

    if holes_only:
        print("[fill_openings] holes_only=True — skipping GLB fill and render.")
        print(f"[fill_openings] Holes written to {base_obj}")
        print(f"[fill_openings] Placements metadata → {place_dir / 'opening_placements.json'}")
        return

    obj_out = save_combined_obj(output_dir, placements, openings, room)
    _render_combined(obj_out, cam_path, place_dir, out_dir)


def _render_combined(
    obj_path: str | None,
    cam_path: Path,
    place_dir: Path,
    out_dir: Path,
) -> str:
    """Two-pass render:
    Pass 1 — render_room.py renders wall textures; backing quads fill window holes
             with the matching wall texture so no raw void is visible.
    Pass 2 — software rasterizer overlays generated GLBs vertex colors (the actual
             window texture) on top of the pass-1 image.
    """
    out_path = str(place_dir / "render_openings_filled.png")

    if obj_path is None or not Path(obj_path).exists():
        print("[fill_openings] No combined OBJ — falling back to GLB overlay render")
        render_with_glbs(str(out_dir), render_out_path=out_path)
        return out_path

    from floorplan.wall_line.render_room import render_room

    tex_dir = str(out_dir / "openings")

    # Pass 1: wall textures via render_room
    walls_render_path = str(place_dir / "_walls_render_tmp.png")
    print(f"[fill_openings] Pass 1 — rendering {Path(obj_path).name} with render_room …")
    render_room(
        mesh_path        = obj_path,
        camera_json_path = str(cam_path),
        out_path         = walls_render_path,
        texture_dir      = tex_dir,
    )

    # Pass 2: overlay GLB vertex colors (the 3D generator texture) on top
    print(f"[fill_openings] Pass 2 — overlaying GLB vertex colors …")
    render_with_glbs(str(out_dir), render_in_path=walls_render_path, render_out_path=out_path)

    print(f"[fill_openings] Render → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Place generated GLBs objects at wall openings and render."
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--render-only", action="store_true",
                    help="Re-render existing walls_with_windows.obj without re-matching")
    ap.add_argument("--holes-only", action="store_true",
                    help="Create wall holes and save opening_placements.json only; "
                         "skip GLB fill and render (use with place_objects.py --types window,...)")
    args = ap.parse_args()

    out_dir   = Path(args.output_dir)
    place_dir = out_dir / "wall_mounted" / "placements"
    for _cp in [out_dir / "camera_vggt.json",
                out_dir / "camera.json",
                out_dir / "openings" / "camera.json"]:
        if _cp.exists():
            cam_path = _cp
            break
    else:
        raise FileNotFoundError(f"No camera.json found in {out_dir}")

    if args.render_only:
        obj_path = str(place_dir / "walls_with_windows.obj")
        _render_combined(obj_path, cam_path, place_dir, out_dir)
    else:
        run(args.output_dir, holes_only=args.holes_only)


if __name__ == "__main__":
    main()
