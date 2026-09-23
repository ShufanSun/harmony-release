"""
render_room.py — headless software renderer with texture mapping.

Pure numpy + PIL — no display, no OpenGL required.

Pipeline
--------
1. Load walls.obj (trimesh)
2. Load / auto-derive camera.json
3. Load walls_metadata.json (per-wall textures & features) if present
4. Project vertices → camera space → screen pixels
5. Sort faces back-to-front (painter's algorithm)
6. Scanline-rasterize each triangle with tiled UV texture sampling
   Per-wall textures from walls_metadata.json (orientation → texture file).
   Face normals are mapped to orientation strings:
     floor (N·Y > 0.7) → "floor"
     ceiling (N·Y < -0.7) → "ceiling"
     back wall (N·Z < -0.7) → "back"
     front wall (N·Z > 0.7) → "front"
     left wall (N·X < -0.7) → "left"
     right wall (N·X > 0.7) → "right"
   Falls back to global wall_texture.png / floor_texture.png if metadata absent.

Usage
-----
    python -m floorplan.wall_line.render_room \\
        --mesh   outputs/.../walls.obj \\
        --camera outputs/.../camera.json \\   # optional
        --out    outputs/.../render.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

NEAR_CLIP   = 0.05   # metres — discard vertices closer than this
TILE_SIZE_M = 1.5    # one texture tile covers this many metres

# Orientations textured by direct projective sampling of the reference image
# (proj_tex_image_path) instead of a rectified per-surface texture. Overridable
# via the SCENEWEAVE_PROJ_ORIENTS env var (comma-separated).
import os as _os
_PROJ_ORIENTS = set(
    (_os.environ.get("SCENEWEAVE_PROJ_ORIENTS") or "back").replace(" ", "").split(",")
) - {""}


# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────────────────────────────

def _camera_axes(pos: np.ndarray, look_at: np.ndarray,
                 up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (right, up_c, fwd) orthonormal camera basis.

    Convention: camera looks in +fwd direction (not -Z as in OpenGL).
      right = normalize(up × fwd)       — screen right
      up_c  = normalize(fwd × right)    — screen up
      fwd   = normalize(look_at - pos)  — into scene
    """
    fwd = look_at - pos
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0., 0., 1.])

    right = np.cross(fwd, up)           # fwd × up  → +X for -Z-looking camera
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        up = np.array([0., 0., 1.])
        right = np.cross(fwd, up)
        rn = np.linalg.norm(right)
    right /= rn

    up_c = np.cross(right, fwd)         # right × fwd → +Y screen-up
    up_c /= np.linalg.norm(up_c)
    return right, up_c, fwd


# ─────────────────────────────────────────────────────────────────────────────
# Texture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solid_texture(r: float, g: float, b: float, size: int = 128) -> np.ndarray:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:] = (int(r * 255), int(g * 255), int(b * 255))
    return arr


def _load_texture(path: Path, fallback_rgb: tuple) -> np.ndarray:
    if path.exists():
        img = Image.open(path).convert("RGB")
        print(f"[render] Texture loaded: {path}  ({img.width}×{img.height}px)")
        return np.array(img)
    print(f"[render] Texture not found: {path}  — using flat colour {fallback_rgb}")
    return _solid_texture(*fallback_rgb)


def _face_uv(verts3d: np.ndarray, normal: np.ndarray,
             tile_size_m: float = TILE_SIZE_M,
             uv_rotation: int = 0,
             tile_size_v_m: float | None = None) -> np.ndarray:
    """
    Compute (3,2) UV coordinates for a triangle by planar projection.
    Tiles every tile_size_m metres horizontally; uv_rotation (0/90/180/270)
    rotates the texture pattern on the surface.

    tile_size_v_m sets a SEPARATE vertical tile size for walls (the V/Y axis).
    A full-wall texture is wider than tall, so using tile_size_m (= wall width)
    for V would only reach V≈width/height < 1 at the ceiling and stretch the
    lower band; pass the wall HEIGHT here so V maps 0..1 over the wall. If None,
    V uses tile_size_m (legacy tiling behaviour). Ignored for floor/ceiling.
    """
    ax = int(np.argmax(np.abs(normal)))
    tv = tile_size_v_m if tile_size_v_m else tile_size_m
    if ax == 1:                           # floor / ceiling → XZ (both horizontal)
        uv = verts3d[:, [0, 2]] / tile_size_m
    elif ax == 0:                         # X-facing side wall → ZY
        uv = np.stack([verts3d[:, 2] / tile_size_m, verts3d[:, 1] / tv], axis=1)
    else:                                 # Z-facing wall → XY
        uv = np.stack([verts3d[:, 0] / tile_size_m, verts3d[:, 1] / tv], axis=1)
    if uv_rotation == 90:
        uv = np.stack([-uv[:, 1], uv[:, 0]], axis=1)
    elif uv_rotation == 180:
        uv = -uv
    elif uv_rotation == 270:
        uv = np.stack([uv[:, 1], -uv[:, 0]], axis=1)
    return uv


def _proj_uv(verts3d: np.ndarray,
             cam_pos: np.ndarray, right: np.ndarray, fwd_h: np.ndarray,
             tilt_tan: float,
             fx_ref: float, cx_ref: float, cy_ref: float,
             img_w: int, img_h: int) -> np.ndarray:
    """Project world-space vertices through the reference camera → UV in [0,1].

    fx_ref/cx_ref/cy_ref must be calibrated to the reference photo pixel
    dimensions (not the output render size) so UVs stay in [0,1].
    """
    uvs = np.zeros((len(verts3d), 2), dtype=np.float32)
    for i, v in enumerate(verts3d):
        d   = np.asarray(v, dtype=float) - cam_pos
        xc  = float(np.dot(d, right))
        zc  = max(float(np.dot(d, fwd_h)), 1e-4)
        yc  = float(d[1]) - tilt_tan * zc
        uvs[i, 0] = (fx_ref * xc / zc + cx_ref) / img_w
        uvs[i, 1] = (cy_ref - fx_ref * yc / zc) / img_h
    return uvs


# ─────────────────────────────────────────────────────────────────────────────
# Scanline rasterizer
# ─────────────────────────────────────────────────────────────────────────────

def _rasterize(buf: np.ndarray,
               pts2d: np.ndarray,    # (3, 2) screen pixel coords
               uvs:   np.ndarray,    # (3, 2) UV coords
               tex:   np.ndarray,    # (TH, TW, 3) uint8 texture
               light: float,         # Lambertian factor [0, 1]
               depths: np.ndarray | None = None,  # (3,) per-vertex camera depth for perspective-correct interp
               alpha: float | None = None,         # dissolve override: None = fully opaque
               clamp_uv: bool = False,             # clamp instead of tile (for projective texturing)
               ) -> None:
    """Fill one triangle into buf in-place using perspective-correct UV interpolation.

    If alpha is given (0 < alpha < 1), the triangle is alpha-blended over the
    existing buffer contents rather than overwriting them.
    """
    H, W = buf.shape[:2]
    TH, TW = tex.shape[:2]

    xmin = max(0,   int(np.floor(pts2d[:, 0].min())))
    xmax = min(W-1, int(np.ceil (pts2d[:, 0].max())))
    ymin = max(0,   int(np.floor(pts2d[:, 1].min())))
    ymax = min(H-1, int(np.ceil (pts2d[:, 1].max())))
    if xmin > xmax or ymin > ymax:
        return

    p0, p1, p2 = pts2d[0], pts2d[1], pts2d[2]
    v0 = p1 - p0   # edge 0→1
    v1 = p2 - p0   # edge 0→2
    denom = float(v0[0] * v1[1] - v0[1] * v1[0])
    if abs(denom) < 0.5:
        return      # degenerate / line triangle

    ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    qx = xs.ravel().astype(np.float32) - p0[0]
    qy = ys.ravel().astype(np.float32) - p0[1]

    s = (qx * v1[1] - qy * v1[0]) / denom   # barycentric coord for p1
    t = (qy * v0[0] - qx * v0[1]) / denom   # barycentric coord for p2
    mask = (s >= 0) & (t >= 0) & (s + t <= 1.0)
    if not mask.any():
        return

    si = s[mask]; ti = t[mask]

    # Perspective-correct UV interpolation:
    #   interpolate (uv/z) and (1/z) linearly in screen space, then divide.
    #   This eliminates the diagonal shear seam between the two triangles of a quad.
    if depths is not None and (depths > 0).all():
        inv_z  = (1.0 / depths).astype(np.float64)    # (3,)
        uv_h   = uvs.astype(np.float64) * inv_z[:, None]  # (3,2) = uv/z
        uv_h_i = uv_h[0] + si[:, None] * (uv_h[1] - uv_h[0]) + ti[:, None] * (uv_h[2] - uv_h[0])
        w_i    = inv_z[0] + si * (inv_z[1] - inv_z[0]) + ti * (inv_z[2] - inv_z[0])
        uv     = (uv_h_i / w_i[:, None]).astype(np.float32)
    else:
        # Fallback: linear (used when depths unavailable)
        uv = uvs[0] + si[:, None] * (uvs[1] - uvs[0]) + ti[:, None] * (uvs[2] - uvs[0])
    if clamp_uv:
        tx = np.clip((uv[:, 0] * TW).astype(np.int32), 0, TW - 1)
        ty = np.clip((uv[:, 1] * TH).astype(np.int32), 0, TH - 1)
    else:
        tx = (uv[:, 0] % 1.0 * TW).astype(np.int32) % TW
        ty = (uv[:, 1] % 1.0 * TH).astype(np.int32) % TH

    rgb = tex[ty, tx].astype(np.float32) * light

    pxi = xs.ravel()[mask].astype(np.int32)
    pyi = ys.ravel()[mask].astype(np.int32)

    if alpha is not None:
        # Alpha-blend: out = alpha * src_colour + (1 - alpha) * existing
        existing = buf[pyi, pxi].astype(np.float32)
        blended  = alpha * np.clip(rgb, 0.0, 255.0) + (1.0 - alpha) * existing
        buf[pyi, pxi] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        buf[pyi, pxi] = np.clip(rgb, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Near-plane clipper (Sutherland-Hodgman, single plane zc = NEAR_CLIP)
# ─────────────────────────────────────────────────────────────────────────────

def _clip_triangle_near(
    xc3: np.ndarray,   # (3,) camera-space x
    yc3: np.ndarray,   # (3,) camera-space y
    zc3: np.ndarray,   # (3,) camera-space z (depth along camera forward axis)
    uv3: np.ndarray,   # (3, 2) UV coordinates matching the three vertices
    fx: float, cx: float, cy: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Clip a triangle against the near plane zc = NEAR_CLIP.

    Returns a (possibly empty) list of ``(pts2d, uvs, depths)`` tuples ready
    for ``_rasterize``.  Each tuple is a clipped sub-triangle fully in front
    of the near plane; ``depths`` holds the per-vertex camera depth for
    perspective-correct UV interpolation.
    """
    in_front = zc3 > NEAR_CLIP

    if not in_front.any():
        return []

    def _proj(xc, yc, zc):
        return np.array([cx + fx * xc / zc, cy - fx * yc / zc], dtype=np.float32)

    if in_front.all():
        pts2d  = np.array([_proj(xc3[k], yc3[k], zc3[k]) for k in range(3)])
        depths = zc3.astype(np.float64)
        return [(pts2d, uv3.astype(np.float32), depths)]

    # Sutherland-Hodgman: clip polygon against half-space zc > NEAR_CLIP
    # Represent each vertex as (xc, yc, zc, u, v)
    poly = [
        (float(xc3[k]), float(yc3[k]), float(zc3[k]),
         float(uv3[k, 0]), float(uv3[k, 1]))
        for k in range(3)
    ]

    clipped: list[tuple] = []
    n = len(poly)
    for k in range(n):
        cur = poly[k]
        nxt = poly[(k + 1) % n]
        cur_inside = cur[2] > NEAR_CLIP
        nxt_inside = nxt[2] > NEAR_CLIP

        if cur_inside:
            clipped.append(cur)

        if cur_inside != nxt_inside:
            # Compute intersection point with plane zc = NEAR_CLIP
            dz = nxt[2] - cur[2]
            t  = (NEAR_CLIP - cur[2]) / dz
            ix = cur[0] + t * (nxt[0] - cur[0])
            iy = cur[1] + t * (nxt[1] - cur[1])
            iu = cur[3] + t * (nxt[3] - cur[3])
            iv = cur[4] + t * (nxt[4] - cur[4])
            clipped.append((ix, iy, NEAR_CLIP + 1e-6, iu, iv))

    if len(clipped) < 3:
        return []

    # Triangulate the clipped polygon as a fan from clipped[0]
    result: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for k in range(1, len(clipped) - 1):
        tri_verts = [clipped[0], clipped[k], clipped[k + 1]]
        pts2d  = np.array([_proj(v[0], v[1], v[2]) for v in tri_verts], dtype=np.float32)
        uvs    = np.array([[v[3], v[4]] for v in tri_verts], dtype=np.float32)
        depths = np.array([v[2] for v in tri_verts], dtype=np.float64)
        result.append((pts2d, uvs, depths))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MTL / OBJ material parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_mtl_materials(obj_path: Path) -> dict[str, dict]:
    """
    Parse the MTL file referenced by an OBJ and return a dict:
        { material_name: { "d": float, "Kd": (r,g,b), ... } }

    Only reads properties relevant to rendering: d (dissolve/alpha), Kd (diffuse colour).
    Returns an empty dict if no MTL is found or parsing fails.
    """
    try:
        mtllib_name: str | None = None
        for line in obj_path.read_text(errors="replace").splitlines():
            if line.strip().startswith("mtllib "):
                mtllib_name = line.strip().split(None, 1)[1].strip()
                break
        if not mtllib_name:
            return {}
        mtl_path = obj_path.parent / mtllib_name
        if not mtl_path.exists():
            return {}

        materials: dict[str, dict] = {}
        cur: dict | None = None
        for line in mtl_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("newmtl "):
                cur = {}
                materials[line.split(None, 1)[1].strip()] = cur
            elif cur is not None:
                if line.startswith("d "):
                    try:
                        cur["d"] = float(line.split()[1])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Tr "):
                    # Tr is inverse of d: Tr 1.0 = fully transparent
                    try:
                        cur["d"] = 1.0 - float(line.split()[1])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Kd "):
                    parts = line.split()
                    try:
                        cur["Kd"] = (float(parts[1]), float(parts[2]), float(parts[3]))
                    except (ValueError, IndexError):
                        pass
        return materials
    except Exception as e:
        print(f"[render] MTL parse warning: {e}")
        return {}


def _parse_obj_face_materials(obj_path: Path, n_faces: int) -> list[str | None]:
    """
    Walk the OBJ face list and return a per-face material name list.
    Faces with no preceding usemtl get None.
    """
    face_mtl: list[str | None] = []
    cur_mtl: str | None = None
    try:
        for line in obj_path.read_text(errors="replace").splitlines():
            s = line.strip()
            if s.startswith("usemtl "):
                cur_mtl = s.split(None, 1)[1].strip()
            elif s.startswith("f "):
                face_mtl.append(cur_mtl)
    except Exception as e:
        print(f"[render] OBJ face-material parse warning: {e}")
    # Pad / trim to match actual face count (trimesh may reorder)
    while len(face_mtl) < n_faces:
        face_mtl.append(None)
    return face_mtl[:n_faces]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def render_room(
    mesh_path: str,
    camera_json_path: str | None = None,
    out_path: str = "render.png",
    texture_dir: str | None = None,
    ref_image_path: str | None = None,
    proj_tex_image_path: str | None = None,
    ambient: float = 0.75,
    light_intensity: float = 1.0,
    hide_orients: set[str] | None = None,
    bg_color: tuple[int, int, int] = (26, 26, 26),
    no_legend: bool = True,
) -> str:
    """Render the room mesh and save to out_path.  Returns out_path.

    hide_orients
        Set of orientation strings whose faces are forcibly skipped.
        Defaults to empty — back-face culling (inward-normal dot camera) handles
        all exterior-facing faces automatically.
    """
    if hide_orients is None:
        hide_orients = set()

    try:
        import trimesh
    except ImportError as e:
        raise ImportError("render_room requires trimesh: pip install trimesh") from e

    # ── camera ────────────────────────────────────────────────────────────────
    if camera_json_path and Path(camera_json_path).exists():
        cam = json.loads(Path(camera_json_path).read_text())
        # VGGT inference writes a *list* of frames; HARMONY stages 1-3
        # write a *dict*. Handle both transparently.
        if isinstance(cam, list):
            if not cam:
                raise ValueError(f"camera.json is an empty list: {camera_json_path}")
            cam = cam[0]

        # VGGT inference writes (extrinsic_3x4, intrinsic_3x3, image_size_hw_orig)
        # but render_room expects (position_m, look_at_m, up, hfov_deg, …).
        # Convert if needed.
        if "position_m" not in cam and "extrinsic_3x4" in cam:
            E = np.asarray(cam["extrinsic_3x4"], dtype=np.float64)   # (3, 4) cam-from-world
            K = np.asarray(cam["intrinsic_3x3"], dtype=np.float64)   # (3, 3)
            R_cw = E[:3, :3]
            t_cw = E[:3, 3]
            R_wc = R_cw.T                          # world-from-camera
            cam_pos = -R_wc @ t_cw                 # camera centre in world
            cam_fwd = R_wc @ np.array([0.0, 0.0, 1.0])  # OpenCV +Z is forward
            cam_up  = -R_wc @ np.array([0.0, 1.0, 0.0])  # OpenCV +Y is down → world-up = -row
            look_at = cam_pos + cam_fwd
            ih, iw  = cam.get("image_size_hw_orig", [int(K[1, 2] * 2), int(K[0, 2] * 2)])
            fx      = float(K[0, 0])
            hfov    = float(np.degrees(2.0 * np.arctan(iw / (2.0 * fx))))
            cam = {
                "position_m":  cam_pos.tolist(),
                "look_at_m":   look_at.tolist(),
                "up":          cam_up.tolist(),
                "hfov_deg":    hfov,
                "width_px":    int(iw),
                "height_px":   int(ih),
                "wall_context": {},
                # Keep originals for any consumer that wants them.
                "_vggt_extrinsic_3x4": cam["extrinsic_3x4"],
                "_vggt_intrinsic_3x3": cam["intrinsic_3x3"],
            }
            print(f"[render] converted VGGT camera schema → "
                  f"pos={cam['position_m']}, hfov={cam['hfov_deg']:.1f}°, "
                  f"{cam['width_px']}×{cam['height_px']}")
    else:
        print("[render] camera.json not found — deriving from mesh bounding box")
        from floorplan.wall_line.camera_placement import camera_from_mesh
        cam = camera_from_mesh(mesh_path, ref_image_path=ref_image_path)

    pos      = np.array(cam["position_m"], dtype=float)
    look_at  = np.array(cam["look_at_m"],  dtype=float)
    up       = np.array(cam["up"],          dtype=float)
    hfov_deg = float(cam["hfov_deg"])
    W        = int(cam["width_px"])
    H        = int(cam["height_px"])
    wall_ctx = cam.get("wall_context", {})

    print(f"[render] Camera  : pos={np.round(pos,3).tolist()}  "
          f"look_at={np.round(look_at,3).tolist()}")
    print(f"[render] Canvas  : {W}×{H}px  hfov={hfov_deg}°")

    # ── mesh ──────────────────────────────────────────────────────────────────
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    verts   = mesh.vertices.astype(np.float64)
    faces   = mesh.faces
    normals = mesh.face_normals.astype(np.float64)

    # ── MTL material properties (dissolve / alpha per face) ───────────────────
    # Parse the OBJ for usemtl groups, then the MTL for d (dissolve) values.
    # face_dissolve[i] = 1.0 means fully opaque; 0.0 means fully transparent.
    # face_kd[i] = (r,g,b) float override when a solid colour replaces texture.
    face_dissolve = np.ones(len(faces), dtype=np.float32)   # default: opaque
    face_kd: dict[int, tuple[float, float, float]] = {}     # face → RGB override

    _mtl_props = _parse_mtl_materials(Path(mesh_path))
    if _mtl_props:
        _face_mtl = _parse_obj_face_materials(Path(mesh_path), len(faces))
        for fi, mname in enumerate(_face_mtl):
            if mname and mname in _mtl_props:
                props = _mtl_props[mname]
                face_dissolve[fi] = float(props.get("d", 1.0))
                if "Kd" in props:
                    face_kd[fi] = props["Kd"]

    # ── textures ──────────────────────────────────────────────────────────────
    tex_dir = Path(texture_dir) if texture_dir else Path(mesh_path).parent

    # Fallback global textures
    _global_wall_tex    = _load_texture(tex_dir / "wall_texture.png",  (0.85, 0.82, 0.78))
    _global_floor_tex   = _load_texture(tex_dir / "floor_texture.png", (0.58, 0.48, 0.38))
    _global_ceiling_tex = _load_texture(tex_dir / "ceiling_texture.png", (0.90, 0.93, 0.97))

    # Per-wall textures from walls_metadata.json
    _orient_tex: dict[str, np.ndarray] = {}
    _walls_meta: dict = {}
    meta_path = tex_dir / "walls_metadata.json"
    if meta_path.exists():
        _walls_meta = json.loads(meta_path.read_text())
        print(f"[render] walls_metadata.json loaded — {len(_walls_meta)} walls")
        for orient, info in _walls_meta.items():
            tp = info.get("texture_path")
            if tp and Path(tp).exists():
                _orient_tex[orient] = _load_texture(Path(tp), (0.85, 0.82, 0.78))
            else:
                # fall back per-orientation defaults
                if orient == "floor":
                    _orient_tex[orient] = _global_floor_tex
                elif orient == "ceiling":
                    _orient_tex[orient] = _global_ceiling_tex
                else:
                    _orient_tex[orient] = _global_wall_tex
    else:
        print("[render] walls_metadata.json not found — using global textures")

    def _normal_to_orient(n: np.ndarray) -> str:
        """Map a face normal to one of: floor/ceiling/back/front/left/right.

        Room-box faces point INWARD (toward camera).  The inward normal of a
        named surface is opposite to what you'd expect from the name:
          back wall  (Z=0)    → inward normal +Z  → "back"
          front wall (Z=D)    → inward normal -Z  → "front"
          left wall  (X=0)    → inward normal +X  → "left"
          right wall (X=W)    → inward normal -X  → "right"
          floor      (Y=0)    → inward normal +Y  → "floor"
          ceiling    (Y=H)    → inward normal -Y  → "ceiling"
        """
        ax = int(np.argmax(np.abs(n)))
        if ax == 1:
            return "floor" if n[1] > 0 else "ceiling"
        elif ax == 0:
            # +X inward → left wall (at X=0);  -X inward → right wall (at X=W)
            return "left" if n[0] > 0 else "right"
        else:
            # +Z inward → back wall (at Z=0);  -Z inward → front wall (at Z=D)
            return "back" if n[2] > 0 else "front"

    def _pick_tex(n: np.ndarray) -> np.ndarray:
        orient = _normal_to_orient(n)
        if orient in _orient_tex:
            return _orient_tex[orient]
        # global fallbacks
        if orient == "floor":
            return _global_floor_tex
        if orient == "ceiling":
            return _global_ceiling_tex
        return _global_wall_tex

    # Merge wall feature context from metadata (overrides camera.json if present)
    if _walls_meta and not wall_ctx:
        wall_ctx = {
            o: {"features": info.get("features", [])}
            for o, info in _walls_meta.items()
            if info.get("features")
        }

    # ── camera axes & projection ───────────────────────────────────────────────
    _, up_c, fwd = _camera_axes(pos, look_at, up)
    fx  = W / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))
    cx  = W / 2.0
    cy  = H / 2.0

    # ── Architectural (two-point) perspective ─────────────────────────────────
    # Use horizontal-only forward for depth so vertical world lines always
    # project as vertical screen lines regardless of camera tilt.
    #
    # IMPORTANT: right must be computed from world-up [0,1,0], NOT from the
    # camera's up field. The camera up may include a Manhattan roll correction
    # which gives right[1] ≠ 0, breaking the architectural property and
    # introducing a pixel offset mismatch with the alignment solver.
    _right_raw = np.cross(fwd, np.array([0., 1., 0.]))
    _right_n   = float(np.linalg.norm(_right_raw))
    right = _right_raw / _right_n if _right_n > 1e-9 else np.array([1., 0., 0.])
    #
    # Standard perspective (fwd includes tilt y-component):
    #   zc = d @ fwd  →  varies along world-vertical lines  →  converging verticals
    #
    # Architectural fix:
    #   fwd_h = normalize((fwd_x, 0, fwd_z))   — no y component
    #   zc    = d @ fwd_h                       — constant along world-vertical lines
    #   yc    = d_y - tan(tilt) * zc            — vertical deviation from look direction
    #
    # right_y is always 0 when world-up=(0,1,0), so xc is already decoupled from y.
    fwd_h   = np.array([fwd[0], 0.0, fwd[2]], dtype=float)
    fwd_h_n = float(np.linalg.norm(fwd_h))
    if fwd_h_n > 1e-9:
        fwd_h  /= fwd_h_n
        tilt_tan = float(fwd[1]) / fwd_h_n   # tan(tilt): fwd[1]=sin(t), fwd_h_n=cos(t)
    else:
        fwd_h    = fwd.copy()   # camera pointing straight up/down — fallback
        tilt_tan = 0.0

    # ── projective texture (optional) ────────────────────────────────────────
    _proj_tex_data: tuple | None = None
    if proj_tex_image_path and Path(proj_tex_image_path).exists():
        import cv2 as _cv2
        _ptex = _cv2.imread(str(proj_tex_image_path))
        if _ptex is not None:
            _ptex = _cv2.cvtColor(_ptex, _cv2.COLOR_BGR2RGB)
            _ptex_iw = _ptex.shape[1]
            _ptex_ih = _ptex.shape[0]
            # Recompute fx/cx/cy calibrated to the reference photo's pixel dimensions,
            # not the output render dimensions.  Using the output fx would give UVs
            # far outside [0,1] whenever the two resolutions differ, causing shattered
            # wrapping artifacts.
            _ptex_fx = _ptex_iw / (2.0 * np.tan(np.radians(hfov_deg / 2.0)))
            _ptex_cx = _ptex_iw / 2.0
            _ptex_cy = _ptex_ih / 2.0
            _proj_tex_data = (_ptex, _ptex_fx, _ptex_cx, _ptex_cy, _ptex_iw, _ptex_ih)
            print(f"[render_room] projective texture: {Path(proj_tex_image_path).name} "
                  f"({_ptex_iw}×{_ptex_ih}px  fx_ref={_ptex_fx:.1f})")

    # ── synthetic ceiling-extension faces (fill mesh height gap) ──────────────
    # When walls_metadata.json has height_m but the loaded mesh doesn't reach
    # the ceiling (common when VGGT slightly under-estimates room height),
    # add extension quads to fill the uncovered strip up to the true ceiling.
    if _walls_meta:
        _ceiling_h: float | None = None
        for _mi in _walls_meta.values():
            if isinstance(_mi, dict) and "height_m" in _mi and float(_mi["height_m"]) > 0.1:
                _ceiling_h = float(_mi["height_m"])
                break
        _mesh_top = float(verts[:, 1].max())
        if _ceiling_h is not None and _mesh_top < _ceiling_h - 0.05:
            _ext_y0 = _mesh_top
            _ext_y1 = _ceiling_h
            _W = float(verts[:, 0].max())
            _D = float(verts[:, 2].max())
            _syn_v: list = []
            _syn_f: list = []
            _syn_n: list = []

            def _ext_quad(nx, ny, nz, v0, v1, v2, v3):
                b = len(verts) + len(_syn_v)
                _syn_v.extend([list(v0), list(v1), list(v2), list(v3)])
                _syn_f.extend([[b, b+1, b+2], [b, b+2, b+3]])
                _syn_n.extend([[nx, ny, nz], [nx, ny, nz]])

            # back wall (nz>0.7): Z≈0, spans X=0..W
            _bw = [i for i, n in enumerate(normals) if n[2] > 0.7]
            if _bw:
                _bz = float(verts[faces[_bw].flatten(), 2].mean())
                _ext_quad(0, 0, 1,
                          [0,  _ext_y0, _bz], [_W, _ext_y0, _bz],
                          [_W, _ext_y1, _bz], [0,  _ext_y1, _bz])
            # left wall (nx>0.7): X≈0, spans Z=0..D
            _lw = [i for i, n in enumerate(normals) if n[0] > 0.7]
            if _lw:
                _lx = float(verts[faces[_lw].flatten(), 0].mean())
                _ext_quad(1, 0, 0,
                          [_lx, _ext_y0, 0], [_lx, _ext_y0, _D],
                          [_lx, _ext_y1, _D], [_lx, _ext_y1, 0])
            # right wall (nx<-0.7): X≈W, spans Z=0..D
            _rw = [i for i, n in enumerate(normals) if n[0] < -0.7]
            if _rw:
                _rx = float(verts[faces[_rw].flatten(), 0].mean())
                _ext_quad(-1, 0, 0,
                          [_rx, _ext_y0, 0], [_rx, _ext_y0, _D],
                          [_rx, _ext_y1, _D], [_rx, _ext_y1, 0])

            if _syn_v:
                verts   = np.vstack([verts,   np.array(_syn_v, dtype=np.float64)])
                faces   = np.vstack([faces,   np.array(_syn_f, dtype=np.int32)])
                normals = np.vstack([normals, np.array(_syn_n, dtype=np.float64)])
                _ext_cnt = len(_syn_f) // 2
                face_dissolve = np.concatenate([
                    face_dissolve,
                    np.ones(len(_syn_f), dtype=np.float32),
                ])
                print(f"[render] Extended {_ext_cnt} wall(s) {_ext_y0:.2f}m→{_ext_y1:.2f}m")

    # ── room footprint extension when camera is outside mesh XZ bounds ─────────
    # The camera can land outside the mesh if VGGT places it near a corner.
    # Any axis where pos is outside [mesh_min, mesh_max] leaves part of the frame
    # with no geometry (void).  We add floor + ceiling patches and a new outer
    # wall for each violated axis so the room wraps around the camera.
    _margin_ext = 0.10          # metres buffer past camera position
    _ry0  = 0.0
    _ry1  = float(verts[:, 1].max())   # ceiling (already updated by ceiling extension)
    _rxlo = float(verts[:, 0].min());  _rxhi = float(verts[:, 0].max())
    _rzlo = float(verts[:, 2].min());  _rzhi = float(verts[:, 2].max())

    _fv: list = []; _ff: list = []; _fn: list = []

    def _foot_quad(nx, ny, nz, v0, v1, v2, v3):
        b = len(verts) + len(_fv)
        _fv.extend([list(v0), list(v1), list(v2), list(v3)])
        _ff.extend([[b, b+1, b+2], [b, b+2, b+3]])
        _fn.extend([[nx, ny, nz], [nx, ny, nz]])

    # +X: camera is to the right of the room
    if pos[0] > _rxhi + _margin_ext:
        _nx = float(pos[0]) + _margin_ext
        _foot_quad(0,  1, 0,   [_rxhi, _ry0, _rzlo], [_nx,   _ry0, _rzlo],
                                [_nx,   _ry0, _rzhi], [_rxhi, _ry0, _rzhi])   # floor
        _foot_quad(0, -1, 0,   [_rxhi, _ry1, _rzhi], [_nx,   _ry1, _rzhi],
                                [_nx,   _ry1, _rzlo], [_rxhi, _ry1, _rzlo])   # ceiling
        _foot_quad(0, 0,  1,   [_rxhi, _ry0, _rzlo], [_nx,   _ry0, _rzlo],
                                [_nx,   _ry1, _rzlo], [_rxhi, _ry1, _rzlo])   # back strip
        _foot_quad(0, 0, -1,   [_nx,   _ry0, _rzhi], [_rxhi, _ry0, _rzhi],
                                [_rxhi, _ry1, _rzhi], [_nx,   _ry1, _rzhi])   # front strip
        _foot_quad(-1, 0, 0,   [_nx,   _ry0, _rzlo], [_nx,   _ry0, _rzhi],
                                [_nx,   _ry1, _rzhi], [_nx,   _ry1, _rzlo])   # outer right wall
        print(f"[render] Room extended right: X {_rxhi:.2f}→{_nx:.2f}m  (camera at {pos[0]:.2f})")
        _rxhi = _nx

    # -X: camera is to the left of the room
    if pos[0] < _rxlo - _margin_ext:
        _nx = float(pos[0]) - _margin_ext
        _foot_quad(0,  1, 0,   [_nx,   _ry0, _rzlo], [_rxlo, _ry0, _rzlo],
                                [_rxlo, _ry0, _rzhi], [_nx,   _ry0, _rzhi])   # floor
        _foot_quad(0, -1, 0,   [_nx,   _ry1, _rzhi], [_rxlo, _ry1, _rzhi],
                                [_rxlo, _ry1, _rzlo], [_nx,   _ry1, _rzlo])   # ceiling
        _foot_quad(0, 0,  1,   [_nx,   _ry0, _rzlo], [_rxlo, _ry0, _rzlo],
                                [_rxlo, _ry1, _rzlo], [_nx,   _ry1, _rzlo])   # back strip
        _foot_quad(0, 0, -1,   [_rxlo, _ry0, _rzhi], [_nx,   _ry0, _rzhi],
                                [_nx,   _ry1, _rzhi], [_rxlo, _ry1, _rzhi])   # front strip
        _foot_quad(1, 0, 0,    [_nx,   _ry0, _rzhi], [_nx,   _ry0, _rzlo],
                                [_nx,   _ry1, _rzlo], [_nx,   _ry1, _rzhi])   # outer left wall
        print(f"[render] Room extended left:  X {_rxlo:.2f}→{_nx:.2f}m  (camera at {pos[0]:.2f})")
        _rxlo = _nx

    # +Z: camera is in front of the front wall
    if pos[2] > _rzhi + _margin_ext:
        _nz = float(pos[2]) + _margin_ext
        _foot_quad(0,  1, 0,   [_rxlo, _ry0, _rzhi], [_rxhi, _ry0, _rzhi],
                                [_rxhi, _ry0, _nz  ], [_rxlo, _ry0, _nz  ])   # floor
        _foot_quad(0, -1, 0,   [_rxlo, _ry1, _nz  ], [_rxhi, _ry1, _nz  ],
                                [_rxhi, _ry1, _rzhi], [_rxlo, _ry1, _rzhi])   # ceiling
        _foot_quad(-1, 0, 0,   [_rxhi, _ry0, _rzhi], [_rxhi, _ry0, _nz  ],
                                [_rxhi, _ry1, _nz  ], [_rxhi, _ry1, _rzhi])   # right strip
        _foot_quad(1, 0, 0,    [_rxlo, _ry0, _nz  ], [_rxlo, _ry0, _rzhi],
                                [_rxlo, _ry1, _rzhi], [_rxlo, _ry1, _nz  ])   # left strip
        print(f"[render] Room extended front: Z {_rzhi:.2f}→{_nz:.2f}m  (camera at {pos[2]:.2f})")
        _rzhi = _nz

    # -Z: camera is behind the back wall
    if pos[2] < _rzlo - _margin_ext:
        _nz = float(pos[2]) - _margin_ext
        _foot_quad(0,  1, 0,   [_rxlo, _ry0, _nz  ], [_rxhi, _ry0, _nz  ],
                                [_rxhi, _ry0, _rzlo], [_rxlo, _ry0, _rzlo])   # floor
        _foot_quad(0, -1, 0,   [_rxlo, _ry1, _rzlo], [_rxhi, _ry1, _rzlo],
                                [_rxhi, _ry1, _nz  ], [_rxlo, _ry1, _nz  ])   # ceiling
        _foot_quad(-1, 0, 0,   [_rxhi, _ry0, _nz  ], [_rxhi, _ry0, _rzlo],
                                [_rxhi, _ry1, _rzlo], [_rxhi, _ry1, _nz  ])   # right strip
        _foot_quad(1, 0, 0,    [_rxlo, _ry0, _rzlo], [_rxlo, _ry0, _nz  ],
                                [_rxlo, _ry1, _nz  ], [_rxlo, _ry1, _rzlo])   # left strip
        print(f"[render] Room extended back:  Z {_rzlo:.2f}→{_nz:.2f}m  (camera at {pos[2]:.2f})")

    if _fv:
        verts   = np.vstack([verts,   np.array(_fv, dtype=np.float64)])
        faces   = np.vstack([faces,   np.array(_ff, dtype=np.int32)])
        normals = np.vstack([normals, np.array(_fn, dtype=np.float64)])
        face_dissolve = np.concatenate([
            face_dissolve,
            np.ones(len(_ff), dtype=np.float32),
        ])

    d   = verts - pos[None]
    xc  = d @ right                    # horizontal screen offset (right_y=0, no y-coupling)
    zc  = d @ fwd_h                    # horizontal depth  (no y-coupling → vertical lines vertical)
    yc  = d[:, 1] - tilt_tan * zc     # vertical screen offset (corrected for tilt)

    valid = zc > NEAR_CLIP
    safe_z = np.where(valid, zc, 1.0)
    px  = np.where(valid, cx + fx * xc / safe_z, np.nan)
    py  = np.where(valid, cy - fx * yc / safe_z, np.nan)

    # ── lighting (one-sided directional + sky fill) ───────────────────────────
    # Indoor lighting is dominated by a key light from above (ceiling fixtures
    # or sun coming through windows up-and-forward of the camera). Using a
    # one-sided dot product (max with 0 instead of abs) means surfaces facing
    # away from the key go to ambient only — that's what gives the floor/back
    # wall/ceiling the brightness ratio you see in real photos.
    def _n(v): return v / (np.linalg.norm(v) + 1e-9)
    key_dir  = _n(fwd * 0.5 + up_c * 1.0)    # above + slightly into scene
    fill_dir = _n(up_c)                        # straight up (sky fill)
    diffuse  = (np.maximum(normals @ key_dir, 0.0) * 0.55 +
                np.maximum(normals @ fill_dir, 0.0) * 0.30).clip(0.0, 1.0)
    light_factors = (light_intensity *
                     (ambient + (1.0 - ambient) * diffuse)
                    ).clip(0.0, 1.0).astype(np.float32)

    # ── painter's sort (back to front) ────────────────────────────────────────
    face_z_mean = zc[faces].mean(axis=1)
    order = np.argsort(face_z_mean)[::-1]

    # ── rasterize ─────────────────────────────────────────────────────────────
    buf = np.full((H, W, 3), bg_color, dtype=np.uint8)
    print(f"[render] Rasterizing {len(faces)} faces …")

    for i in order:
        tri = faces[i]

        # Skip faces completely behind near plane
        if not valid[tri].any():
            continue

        orient = _normal_to_orient(normals[i])

        # Skip faces whose orientation should be hidden (explicit override)
        if hide_orients and orient in hide_orients:
            continue

        # Back-face culling: face inward-normal must point toward the camera.
        # For interior-facing room geometry this correctly removes any wall whose
        # exterior is facing the camera (e.g. front wall when camera is outside).
        face_center_w = verts[faces[i]].mean(axis=0)
        if np.dot(normals[i], pos - face_center_w) < 0:
            continue

        # Respect MTL dissolve: skip fully transparent faces; note partial alpha
        dissolve = float(face_dissolve[i])
        if dissolve <= 0.0:
            continue   # fully transparent — show bg_color through

        lf     = float(light_factors[i])

        # Projective texturing: only apply to back-wall faces (camera-facing,
        # roughly perpendicular to the view ray).  Side walls, floor, and
        # ceiling fall back to per-wall extracted textures to avoid the
        # corner-seam bending artifact that projective mapping produces when
        # the same photo pixels are projected onto two adjacent faces at once.
        _use_proj = (_proj_tex_data is not None and orient in _PROJ_ORIENTS)
        if _use_proj:
            _ptex_img, _ptex_fx, _ptex_cx, _ptex_cy, _ptex_w, _ptex_h = _proj_tex_data
            uvs    = _proj_uv(verts[tri], pos, right, fwd_h, tilt_tan,
                              _ptex_fx, _ptex_cx, _ptex_cy, _ptex_w, _ptex_h)
            tex    = _ptex_img
            _clamp = True
        else:
            tex    = _pick_tex(normals[i])
            # MTL Kd colour override (e.g. glass pane solid colour)
            if i in face_kd:
                r, g, b = face_kd[i]
                tex = _solid_texture(r, g, b)
            _tmeta       = _walls_meta.get(orient, {})
            _tile        = float(_tmeta.get("tile_size_m") or TILE_SIZE_M)
            _tile_v      = _tmeta.get("tile_size_v_m")
            _default_rot = 90 if orient == "floor" else 0
            _uv_rot      = int(_tmeta.get("uv_rotation_deg") or _default_rot)
            uvs          = _face_uv(verts[tri], normals[i], tile_size_m=_tile,
                                    uv_rotation=_uv_rot,
                                    tile_size_v_m=float(_tile_v) if _tile_v else None)
            _clamp  = False

        if valid[tri].all():
            # Fast path: all vertices in front of near plane
            pts = np.stack([px[tri], py[tri]], axis=1)
            if (pts[:, 0].max() < 0 or pts[:, 0].min() > W or
                    pts[:, 1].max() < 0 or pts[:, 1].min() > H):
                continue
            _rasterize(buf, pts, uvs, tex, lf, depths=zc[tri].astype(np.float64),
                       alpha=dissolve if dissolve < 1.0 else None,
                       clamp_uv=_clamp)
        else:
            # Slow path: clip straddling triangle at near plane
            for pts2d, uvs_c, depths_c in _clip_triangle_near(
                    xc[tri], yc[tri], zc[tri], uvs, fx, cx, cy):
                if (pts2d[:, 0].max() < 0 or pts2d[:, 0].min() > W or
                        pts2d[:, 1].max() < 0 or pts2d[:, 1].min() > H):
                    continue
                _rasterize(buf, pts2d, uvs_c, tex, lf, depths=depths_c,
                           alpha=dissolve if dissolve < 1.0 else None,
                           clamp_uv=_clamp)

    # ── left-to-right wall feature summary (stdout) ───────────────────────────
    # For each non-hidden wall, project its geometric centre to screen X and
    # print features ordered from the leftmost to the rightmost wall in view.
    _room_w   = float(verts[:, 0].max() - verts[:, 0].min())
    _room_h   = float(verts[:, 1].max())
    _cam_z_r  = float(pos[2])
    _wall_centers_world: dict[str, np.ndarray] = {
        "back":  np.array([_room_w / 2.0,  _room_h / 2.0, 0.0]),
        "left":  np.array([0.0,             _room_h / 2.0, _cam_z_r / 2.0]),
        "right": np.array([_room_w,         _room_h / 2.0, _cam_z_r / 2.0]),
        "front": np.array([_room_w / 2.0,   _room_h / 2.0, _cam_z_r]),
    }

    _wall_screen_px: list[tuple[float, str]] = []
    for orient, wc in _wall_centers_world.items():
        if hide_orients and orient in hide_orients:
            continue
        d   = wc - pos
        xc  = float(d @ right)
        zc  = float(d @ fwd_h)   # use architectural horizontal depth
        if zc < NEAR_CLIP:
            continue
        px_val = cx + fx * xc / zc
        _wall_screen_px.append((px_val, orient))

    if _wall_screen_px:
        _wall_screen_px.sort(key=lambda t: t[0])
        print(f"\n[render] Visible walls left → right:")
        for _px_val, _orient in _wall_screen_px:
            _info  = _walls_meta.get(_orient) or wall_ctx.get(_orient, {})
            _feats = _info.get("features", [])
            _surf  = _info.get("surface", "") or _info.get("surface", "")
            _feat_str = (
                ", ".join(
                    f"{f['type']} {f.get('width_m','?')}m×{f.get('height_m','?')}m"
                    for f in _feats
                ) if _feats else "—"
            )
            _surf_str = f"  [{_surf[:40]}]" if _surf and _surf not in ("string", "?", "") else ""
            print(f"  px≈{int(_px_val):5d}  {_orient:6s}: {_feat_str}{_surf_str}")

    # ── feature legend ────────────────────────────────────────────────────────
    legend_lines: list[str] = []
    # From walls_metadata (richer info)
    for orient, info in _walls_meta.items():
        parts: list[str] = []
        feats = info.get("features", [])
        if feats:
            parts.append(", ".join(
                f"{f['type']} {f.get('width_m','?')}m" for f in feats))
        surface = info.get("surface", "")
        if surface and surface not in ("string", "?", ""):
            parts.append(f"[{surface[:30]}]")
        if parts:
            legend_lines.append(f"{orient}: {' '.join(parts)}")
    # Fall back to wall_ctx from camera.json if metadata had nothing
    if not legend_lines and wall_ctx:
        for orient, info in wall_ctx.items():
            feats = info.get("features", [])
            if feats:
                fs = ", ".join(f"{f['type']} {f.get('width_m','?')}m" for f in feats)
                legend_lines.append(f"{orient}: {fs}")

    if legend_lines and not no_legend:
        from PIL import ImageDraw
        img = Image.fromarray(buf)
        draw = ImageDraw.Draw(img)
        y = 8
        for line in legend_lines:
            draw.text((9, y + 1), line, fill=(0, 0, 0))
            draw.text((8, y),     line, fill=(255, 255, 180))
            y += 16
        buf = np.array(img)

    # ── save ──────────────────────────────────────────────────────────────────
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(buf).save(out_path)
    print(f"[render] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Headless software render of a room mesh with textures.")
    parser.add_argument("--mesh",     required=True,
                        help="Path to walls.obj")
    parser.add_argument("--camera",   default=None,
                        help="Path to camera.json (auto-derived if omitted)")
    parser.add_argument("--out",      default="render.png",
                        help="Output PNG path")
    parser.add_argument("--tex_dir",  default=None,
                        help="Directory containing wall_texture.png / floor_texture.png "
                             "(defaults to same directory as walls.obj)")
    parser.add_argument("--ref",      default=None,
                        help="Reference image path — used for render resolution "
                             "and deepest-corner pixel alignment when no camera.json exists")
    parser.add_argument("--ambient",  type=float, default=0.55,
                        help="Ambient light fraction 0–1")
    parser.add_argument("--show",     nargs="*", default=None,
                        help="Orientation(s) to show even if normally hidden "
                             "(e.g. --show ceiling front). Pass empty to show all.")
    args = parser.parse_args()

    hide = {"front"}
    if args.show is not None:
        hide -= set(args.show)   # remove explicitly shown orients

    render_room(
        mesh_path=args.mesh,
        camera_json_path=args.camera,
        out_path=args.out,
        texture_dir=args.tex_dir,
        ref_image_path=args.ref,
        ambient=args.ambient,
        hide_orients=hide,
    )